
#ifndef TSETLIN_MACHINE_MULTICLASSTSETLIN_H
#define TSETLIN_MACHINE_MULTICLASSTSETLIN_H

#include "TsetlinMachine.h"
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace std;

class MultipleClassTsetlin {
public:

    MultipleClassTsetlin(int num_classes, int clauses, int threshold, double s)
            : num_classes(num_classes)
    {

        srand((unsigned)time(0));

        for (int i = 0; i < num_classes; i++) {
            machines.push_back(new TsetlinMachine(clauses, threshold, s));
        }
    }

    ~MultipleClassTsetlin() {
        for (int i = 0; i < num_classes; i++) {
            delete machines[i];
        }
    }



 //타깃 클래스에는 긍정 피드백(1), 임의의 다른 클래스에는 부정 피드백(0)을 적용.
    void train(const vector<unsigned int>& Xi, int target_class) {
        // 타깃 클래스에 대해 긍정 피드백 업데이트
        machines[target_class]->update(Xi, 1);

        // 타깃 클래스와 다른 임의의 클래스 선택 (클래스 수가 2 이상이라고 가정)
        int negative_class = rand() % (num_classes - 1);
        if (negative_class >= target_class) {
            negative_class++;  // target_class와 중복되지 않도록 조정
        }
        machines[negative_class]->update(Xi, 0);
    }



     // 가장 높은 점수를 가진 클래스의 인덱스를 반환합니다.
    int predict(const vector<unsigned int>& Xi) {
        int best_class = 0;
        int best_score = machines[0]->score(Xi);
        for (int i = 1; i < num_classes; i++) {
            int score = machines[i]->score(Xi);
            if (score > best_score) {
                best_score = score;
                best_class = i;
            }
        }
        return best_class;
    }


    //predict 여러번
    double evaluate(const vector<vector<unsigned int>>& X, const vector<int>& y) {
        int errors = 0;
        int num_examples = X.size();
        for (int i = 0; i < num_examples; i++) {
            int predicted = predict(X[i]);
            if (predicted != y[i]) {
                errors++;
            }
        }
        return 1.0 - static_cast<double>(errors) / num_examples;
    }

    //배치 사용 시
    void fit(const vector<vector<unsigned int>>& X, const vector<int>& y, int epochs) {
        int num_examples = X.size();
        for (int epoch = 0; epoch < epochs; epoch++) {
            //데이터 셔플링
            for (int i = 0; i < num_examples; i++) {
                train(X[i], y[i]);
            }
        }
    }

private:
    int num_classes;                        // 분류할 클래스 수
    vector<TsetlinMachine*> machines;       // 각 클래스별 TsetlinMachine 인스턴스
};

#endif //TSETLIN_MACHINE_MULTICLASSTSETLIN_H
