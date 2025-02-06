#include "TsetlinMachine.h"
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cmath>
#include <climits>
#include <iostream>
using namespace std;

// 생성자: 절의 수, 투표 임계값, s 파라미터를 받아 내부 벡터들을 초기화합니다.
TsetlinMachine::TsetlinMachine(int clauses, int threshold, double s)
        : clauses(clauses), threshold(threshold), s(s) {
    // INT_SIZE, FEATURES 등은 상수로 정의됨
    la_chunks = (NUM_LITERALS + INT_SIZE - 1) / INT_SIZE; // 예: (2*784)/32
    clause_chunks = (clauses + INT_SIZE - 1) / INT_SIZE;

    // 3차원 벡터 ta_state[clauses][la_chunks][STATE_BITS] 초기화, initialize
    ta_state.resize(clauses, vector<vector<unsigned int>>(la_chunks, vector<unsigned int>(STATE_BITS, 0)));
    for (int j = 0; j < clauses; j++) {
        for (int k = 0; k < la_chunks; k++) {
            // 초기: 하위 STATE_BITS-1 비트는 모두 1 (즉, ~0), 마지막 비트는 0 → Exclude 상태
            for (int b = 0; b < STATE_BITS - 1; b++) {
                ta_state[j][k][b] = ~0u; // 모든 비트를 1로
            }
            ta_state[j][k][STATE_BITS - 1] = 0;
        }
    }

    // 절 출력 및 피드백 벡터 초기화 (모두 0)
    clause_output.assign(clause_chunks, 0);
    feedback_to_la.assign(la_chunks, 0);
    feedback_to_clauses.assign(clause_chunks, 0);

    // 랜덤 seed 초기화
    srand((unsigned) time(0));
}

// 내부: 피드백용 random stream 초기화
//  – 모든 피드백 비트를 0으로 초기화한 후, 2*FEATURES 중 약 1/S 비트를 활성화합니다.
void TsetlinMachine::initialize_random_streams() {
    // feedback_to_la를 0으로 초기화
    for (int k = 0; k < la_chunks; k++) {
        feedback_to_la[k] = 0;
    }
    int n = NUM_LITERALS;
    double p = 1.0 / s;
    // 평균적으로 활성화될 개수
    int active = int(round(n * p));
    if (active > n) active = n;
    if (active < 0) active = 0;
    while (active--) {
        int f = rand() % n;
        int chunk = f / INT_SIZE;
        int pos = f % INT_SIZE;
        // 이미 활성화되어 있으면 재선택
        while (feedback_to_la[chunk] & (1u << pos)) {
            f = rand() % n;
            chunk = f / INT_SIZE;
            pos = f % INT_SIZE;
        }
        feedback_to_la[chunk] |= (1u << pos);
    }
}


// 내부: 선택된 automata의 상태를 증가시키는 함수 (비트 단위 캐리 연산)
void TsetlinMachine::inc(int clause, int chunk, unsigned int active) {
    unsigned int carry = active;
    for (int b = 0; b < STATE_BITS; b++) {
        if (carry == 0)
            break;
        unsigned int carry_next = ta_state[clause][chunk][b] & carry;  // overflow 비트 계산
        ta_state[clause][chunk][b] ^= carry;                           // XOR로 더함
        carry = carry_next;
    }
    if (carry > 0) {
        // overflow가 남으면 모든 비트에 해당 carry를 OR
        for (int b = 0; b < STATE_BITS; b++) {
            ta_state[clause][chunk][b] |= carry;
        }
    }
}

// 내부: 선택된 automata의 상태를 감소시키는 함수
void TsetlinMachine::dec(int clause, int chunk, unsigned int active) {
    unsigned int carry = active;
    for (int b = 0; b < STATE_BITS; b++) {
        if (carry == 0)
            break;
        unsigned int carry_next = (~ta_state[clause][chunk][b]) & carry;
        ta_state[clause][chunk][b] ^= carry;
        carry = carry_next;
    }
    if (carry > 0) {
        for (int b = 0; b < STATE_BITS; b++) {
            ta_state[clause][chunk][b] &= ~carry;
        }
    }
}

// 내부: 각 절의 출력 계산
// predict가 true이면 예측 모드(모든 절이 모두 Exclude인 경우 출력 0으로 강제),
// false이면 업데이트 모드로 계산합니다.
void TsetlinMachine::calculate_clause_output(const vector<unsigned int>& Xi, bool predict) {
    // 먼저 clause_output를 0으로 초기화
    for (int i = 0; i < clause_chunks; i++) {
        clause_output[i] = 0;
    }

    // 마지막 청크에 사용할 필터: 마지막 청크에 유효한 비트 수(32보다 작을 수 있음)
    unsigned int filter;
    int rem = NUM_LITERALS % INT_SIZE;
    if (rem == 0) filter = ~0u; else filter = ((1u << rem) - 1);

    // 각 절 j에 대해 출력 계산
    for (int j = 0; j < clauses; j++) {
        bool output = true;
        bool all_exclude = true;
        // k = 0 ~ la_chunks-2
        for (int k = 0; k < la_chunks - 1; k++) {
            // ta_state[j][k][STATE_BITS-1]는 automata의 결정 비트(Include 여부)
            //j번째 clause의 k번 째 literal의 state bit
            // 절이 활성화되려면, ta_state의 결정 비트가 설정된 모든 자리에서 입력 Xi의 해당 비트가 1이어야 함.
    /*        Clause Output이 1이 되는 조건은 다음과 같습니다:

            해당 Clause에 포함(Include)된 리터럴들만 고려.
            이 리터럴들이 입력 데이터(Xi)와 일치하면 Clause Output은 1.
            만약 하나라도 불일치하면 Clause Output은 0 */
            if ((ta_state[j][k][STATE_BITS - 1] & Xi[k]) != ta_state[j][k][STATE_BITS - 1]) {
                output = false;
                break;
            }
            if (ta_state[j][k][STATE_BITS - 1] != 0)
                all_exclude = false;
        }
        // 마지막 청크 처리
        if (output) {
            if (((ta_state[j][la_chunks - 1][STATE_BITS - 1] & Xi[la_chunks - 1] & filter) !=
                 (ta_state[j][la_chunks - 1][STATE_BITS - 1] & filter))) {
                output = false;
            }
            if ((ta_state[j][la_chunks - 1][STATE_BITS - 1] & filter) != 0)
                all_exclude = false;
        }
        // 예측 모드에서 모든 리터럴이 Exclude이면 절 출력은 0으로 강제
        if (predict && all_exclude)
            output = false;

        // 절 j의 출력이 true이면, clause_output의 해당 비트를 1로 설정
        if (output) {
            int clause_chunk = j / INT_SIZE; //몇 번째 청크
            int bit_pos = j % INT_SIZE; //청크 내에 몇 번째 리터럴
            clause_output[clause_chunk] |= (1u << bit_pos);
        }
    }
}

//절들의 투표를 합산하여 클래스 점수를 계산
// 짝수 절은 +1, 홀수 절은 -1로 투표하며, 결과를 [-threshold, threshold] 범위로 클립함.
int TsetlinMachine::sum_up_class_votes() {
    int class_sum = 0;
    for (int i = 0; i < clause_chunks; i++) {
        // 0x55555555: 0101... (짝수 비트 mask), 0xaaaaaaaa: 1010... (홀수 비트 mask)
        class_sum += __builtin_popcount(clause_output[i] & 0x55555555);
        class_sum -= __builtin_popcount(clause_output[i] & 0xaaaaaaaa);
    }
    if (class_sum > threshold) class_sum = threshold;
    if (class_sum < -threshold) class_sum = -threshold;
    return class_sum;
}


//  – 먼저 절의 출력을 계산한 후, 전체 투표(class_sum)를 구하고,
//    각 절에 대해 Type I / Type II 피드백을 확률적으로 적용.
void TsetlinMachine::update(const vector<unsigned int>& Xi, int target) {
    // UPDATE 모드로 절 출력 계산
    calculate_clause_output(Xi, false);
    int class_sum = sum_up_class_votes();

    // 피드백 확률 p = (1/(2*threshold))*(threshold + (1-2*target)*class_sum)
    float p = (1.0f / (threshold * 2)) * (threshold + (1 - 2 * target) * class_sum);

    // feedback_to_clauses를 0으로 초기화한 후, 각 절에 대해 확률 p로 피드백 적용 여부 결정
    for (int i = 0; i < clause_chunks; i++) {
        feedback_to_clauses[i] = 0;
    }
    for (int j = 0; j < clauses; j++) {
        int clause_chunk = j / INT_SIZE;
        int bit_pos = j % INT_SIZE;
        // 확률 p보다 작으면 해당 절에 피드백 적용 (rand()/(RAND_MAX+1.0) ∈ [0,1))
        if (((float)rand() / (RAND_MAX + 1.0f)) <= p) {
            feedback_to_clauses[clause_chunk] |= (1u << bit_pos);
        }
    }

    // 각 절에 대해 피드백 적용
    for (int j = 0; j < clauses; j++) {
        int clause_chunk = j / INT_SIZE; //몇 번째 clause
        int bit_pos = j % INT_SIZE; //리터럴 위치
        // 피드백이 없는 절은 건너뜀
        if (!(feedback_to_clauses[clause_chunk] & (1u << bit_pos)))
            continue;

        // clause polarity: 짝수 절은 positive, 홀수 절은 negative
        // (2*target-1) * (1-2*(j&1)) == -1 → Type II, == 1 → Type I
        int polarity = (1 - 2 * (j & 1)); // 짝수: 1, 홀수: -1, 최하위 비트를 보고 짝수, 홀수 판별
        if ((2 * target - 1) * polarity == -1) {
            // Type II 피드백: 절이 활성화되었을 때,
            // 각 청크에 대해, 입력 Xi의 0인 자리와 automata의 Include 비트(~ta_state[..][STATE_BITS-1])에 대해 inc
            int out_chunk = j / INT_SIZE;
            if (clause_output[out_chunk] & (1u << (j % INT_SIZE))) {//literal이 1이라면
                for (int k = 0; k < la_chunks; k++) {
                    //둘을 AND한 결과는 입력에서도 0이고, 현재 자동자도 Include 상태(결정 비트 1)가 아닌 리터럴들의 위치를 나타냄.
                    unsigned int active = (~Xi[k]) & ~(ta_state[j][k][STATE_BITS - 1]);
                    inc(j, k, active);
                }
            }
        }
        else if ((2 * target - 1) * polarity == 1) {
            // Type I 피드백
            // 먼저, 초기화된 피드백 스트림을 사용
            initialize_random_streams();
            int out_chunk = j / INT_SIZE;
            if (clause_output[out_chunk] & (1u << (j % INT_SIZE))) {
                for (int k = 0; k < la_chunks; k++) {
                    // BOOST_TRUE_POSITIVE_FEEDBACK 옵션은 생략하고,
                    // 입력이 1인 자리 중 피드백 스트림에 포함되지 않은 곳에 대해 inc,
                    // 입력이 0인 자리 중 피드백 스트림에 포함된 곳에 대해 dec.
                    unsigned int active_inc = Xi[k] & ~(feedback_to_la[k]);
                    inc(j, k, active_inc);
                    unsigned int active_dec = (~Xi[k]) & feedback_to_la[k];
                    dec(j, k, active_dec);
                }
            }
            else {
                for (int k = 0; k < la_chunks; k++) {
                    dec(j, k, feedback_to_la[k]);
                }
            }
        }
    }
}

// score 함수: 예측 모드로 절 출력 계산한 후, 투표 합을 반환합니다.
int TsetlinMachine::score(const vector<unsigned int>& Xi) {
    calculate_clause_output(Xi, true);
    return sum_up_class_votes();
}


// 디버깅용: 특정 절과 리터럴의 상태값을 반환
// 각 automaton의 상태는 STATE_BITS개의 비트를 모아 표현됨
int TsetlinMachine::getState(int clause, int la) {
    int chunk = la / INT_SIZE;
    int pos = la % INT_SIZE;
    int state = 0;
    for (int b = 0; b < STATE_BITS; b++) {
        if (ta_state[clause][chunk][b] & (1u << pos))
            state |= (1 << b);
    }
    return state;
}

// 디버깅용: 특정 절과 리터럴의 행동(Include 여부)을 반환
int TsetlinMachine::action(int clause, int la) {
    int chunk = la / INT_SIZE;
    int pos = la % INT_SIZE;
    return (ta_state[clause][chunk][STATE_BITS - 1] & (1u << pos)) ? 1 : 0;
}
