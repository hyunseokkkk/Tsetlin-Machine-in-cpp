//
// Created by loveh on 25. 2. 2.
//

#ifndef TSETLIN_MACHINE_TSETLINMACHINE_H
#define TSETLIN_MACHINE_TSETLINMACHINE_H


#include <vector>
using namespace std;

class TsetlinMachine {
public:
    // 생성자: clauses = 절의 수, threshold = 투표 임계값, s = 업데이트 확률 조절 파라미터
    TsetlinMachine(int clauses, int threshold, double s);

    // 온라인 학습: 입력 Xi (비트 청크 배열)와 target (0 또는 1)를 이용해 업데이트
    void update(const vector<unsigned int>& Xi, int target);

    // 예측 점수 계산: 입력 Xi에 대해 절들의 투표를 합산하여 점수를 반환
    int score(const vector<unsigned int>& Xi);

    // 디버깅용: clause번 절의 la번 automaton의 상태값을 반환
    int getState(int clause, int la);
    // 디버깅용: clause번 절의 la번 automaton이 현재 행동(Include:1 / Exclude:0)인지 반환
    int action(int clause, int la);

private:
    int clauses;      // 총 절의 수
    int threshold;    // 투표 임계값 (클립용)
    double s;         // 업데이트 확률 조절 파라미터

    // 내부 상수
    static const int FEATURES = 784;                   // MNIST 이미지 (28x28)
    static const int NUM_LITERALS = 2 * FEATURES;        // 각 특성과 그 부정 리터럴
    static const int INT_SIZE = sizeof(unsigned int) * 8;
    static const int STATE_BITS = 8;                   // 각 automaton이 가지는 상태 비트 수

    // 자동자 상태: 3차원 벡터 [절][LA_Chunk][STATE_BITS]
    vector<vector<vector<unsigned int>>> ta_state;
    // 각 절의 출력 (비트 단위로 저장, 절 하나당 한 비트)
    vector<unsigned int> clause_output;
    // 피드백을 줄 때 사용할 임시 벡터 (literal 단위)
    vector<unsigned int> feedback_to_la;
    // 각 절에 피드백 적용 여부를 저장 (비트 단위)
    vector<unsigned int> feedback_to_clauses;

    // 내부: 초기화 함수 (ta_state 등 초기화)
    void initialize();
    // 내부: 각 절의 출력(클래스 vote용)을 계산 (predict 모드와 update 모드 구분) 하나의 clause
    void calculate_clause_output(const vector<unsigned int>& Xi, bool predict);
    // 내부: 모든 절의 투표 합산 (짝수 절은 +, 홀수 절은 -)
    int sum_up_class_votes();

    // 내부: 선택된 automata에 대해 상태를 증가(inc) (비트 단위 캐리 연산)
    void inc(int clause, int chunk, unsigned int active);
    // 내부: 선택된 automata에 대해 상태를 감소(dec)
    void dec(int clause, int chunk, unsigned int active);
    // 내부: 피드백용 random stream을 초기화 (feedback_to_la를 무작위 활성화)
    void initialize_random_streams();

    // 편의를 위해 CLAUSE_CHUNKS (절들을 비트로 저장하기 위한 청크 수)를 계산
    int clause_chunks;
    // LA_CHUNKS: 2*FEATURES를 INT_SIZE 단위로 나눈 청크 수
    int la_chunks;
};

#endif //TSETLIN_MACHINE_TSETLINMACHINE_H
