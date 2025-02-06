// main.cpp
#include "MultiClassTsetlin.h"  // MultipleClassTsetlin 클래스 정의 헤더
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace std::chrono;


const int NUMBER_OF_TRAINING_EXAMPLES = 60000;
const int NUMBER_OF_TEST_EXAMPLES = 10000;
const int FEATURES = 784;           // MNIST: 28x28 이미지
const int INT_SIZE = 32;            // 32비트 unsigned int 사용
// 각 예제는 원본 784개와 보수 784개, 즉 총 1568개 리터럴을 저장하므로:
const int LA_CHUNKS = (2 * FEATURES + INT_SIZE - 1) / INT_SIZE;

// 한 예제의 픽셀 데이터를 읽어, 2*FEATURES 길이의 비트벡터(vector<unsigned int>)로 패킹하는 함수
vector<unsigned int> packExample(const vector<int> &sample) {
    vector<unsigned int> result(LA_CHUNKS, 0);//0으로 초기화, 784 * 2 만큼
    for (int j = 0; j < FEATURES; j++) {
        if (sample[j] == 1) {
            // 원본: 픽셀이 1이면 해당 비트를 켬
            int chunk_nr = j / INT_SIZE; //몇 번째 블록
            int chunk_pos = j % INT_SIZE; //몇 번째 자리
            result[chunk_nr] |= (1u << chunk_pos);
        } else {
            // 보수: 픽셀이 0이면, 보수 부분에서 (j+FEATURES)번째 리터럴에 1을 설정
            int index = j + FEATURES; //784만큼 뒤에 본인의 보수가 위치해 있음
            int chunk_nr = index / INT_SIZE;
            int chunk_pos = index % INT_SIZE;
            result[chunk_nr] |= (1u << chunk_pos); //그 부분을 1로 변경
        }
    }
    return result;
}


// 각 줄은 "b0 b1 ... b783 label" 형태
void readData(const string &filename, vector<vector<unsigned int>> &X, vector<int> &y, int numExamples) {

    ifstream inFile(filename);
    if (!inFile) {
        cerr << "Error opening file: " << filename << endl;
        perror("Error details");
        exit(EXIT_FAILURE);
    }

    string line;
    int count = 0;
    while (getline(inFile, line) && count < numExamples) {
        istringstream iss(line);
        vector<int> sample;
        sample.reserve(FEATURES);
        // 픽셀 데이터 784개 읽기
        for (int i = 0; i < FEATURES; i++) {
            int pixel;
            if (!(iss >> pixel))
                break;
            sample.push_back(pixel);
        }
        // 다음 토큰은 라벨
        int label;
        if (!(iss >> label))
            break;
        y.push_back(label);
        X.push_back(packExample(sample));
        count++;
    }
    inFile.close();
}

void printDigit(const vector<unsigned int> &Xi) {
    for (int row = 0; row < 28; row++) {
        for (int col = 0; col < 28; col++) {
            int index = row * 28 + col;
            int chunk_nr = index / INT_SIZE;
            int chunk_pos = index % INT_SIZE;
            cout << ((Xi[chunk_nr] & (1u << chunk_pos)) ? "@" : ".");
        }
        cout << "\n";
    }
}

int main() {
    srand(static_cast<unsigned>(time(nullptr)));

    vector<vector<unsigned int>> X_train; //픽셀
    vector<int> y_train;  //라벨
    vector<vector<unsigned int>> X_test;
    vector<int> y_test;
    vector<vector<unsigned int>> X_train_sampled;
    vector<int> y_train_sampled;

    cout << "Reading training data...\n";
    readData("MNISTTraining.txt", X_train, y_train, NUMBER_OF_TRAINING_EXAMPLES);

    cout << "Reading test data...\n";
    readData("MNISTTest.txt", X_test, y_test, NUMBER_OF_TEST_EXAMPLES);

    cout << "Reading sampled training data...\n";
    readData("MNISTTrainingSampled.txt", X_train_sampled, y_train_sampled, NUMBER_OF_TEST_EXAMPLES);

    // 임의의 테스트 예제를 선택하여 출력 (데이터 확인용)
    int example = rand() % X_test.size();
    cout << "\nExample digit (label = " << y_test[example] << "):\n\n";
    printDigit(X_test[example]);

    // MultipleClassTsetlin 객체를 직접 생성 (CreateMultiClassTsetlinMachine() 없이)
    int numClasses = 10;  // MNIST의 클래스 수: 0~9
    int clauses = 100;    // 각 클래스당 절의 수 (예시)
    int threshold = 15;   // 투표 임계값 (예시)
    double s = 3.9;       // 업데이트 확률 조절 파라미터 (예시)
    MultipleClassTsetlin mc_tm(numClasses, clauses, threshold, s);

    constexpr int EPOCHS = 100;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        cout << "\nEpoch " << (epoch + 1) << "\n";

        auto startTrain = steady_clock::now();
        // 모든 학습 예제에 대해 One-vs-All 방식 학습 (각 예제마다 한 번씩 업데이트)
        for (size_t i = 0; i < X_train.size(); i++) {
            mc_tm.train(X_train[i], y_train[i]);
        }
        auto endTrain = steady_clock::now();
        double trainTime = duration<double>(endTrain - startTrain).count();
        cout << "Training Time: " << trainTime << " s\n";

        // 테스트 데이터 평가
        auto startEval = steady_clock::now();
        int correct = 0;
        for (size_t i = 0; i < X_test.size(); i++) {
            int prediction = mc_tm.predict(X_test[i]);
            if (prediction == y_test[i])
                correct++;
        }
        auto endEval = steady_clock::now();
        double evalTime = duration<double>(endEval - startEval).count();
        double testAccuracy = 100.0 * correct / X_test.size();
        cout << "Evaluation Time: " << evalTime << " s\n";
        cout << "Test Accuracy: " << testAccuracy << " %\n";

        // 샘플 학습 데이터 평가 (빠른 확인용)
        correct = 0;
        for (size_t i = 0; i < X_train_sampled.size(); i++) {
            int prediction = mc_tm.predict(X_train_sampled[i]);
            if (prediction == y_train_sampled[i])
                correct++;
        }
        double trainSampleAccuracy = 100.0 * correct / X_train_sampled.size();
        cout << "Training Sample Accuracy: " << trainSampleAccuracy << " %\n";
    }

    //예시 출력
    int tmp = rand() % X_test.size();
    cout << "\n=== Training Completed ===\n";
    cout << "Displaying a random MNIST image from the test set:\n";
    cout << "True Label: " << y_test[tmp] << "\n";
    int predictedLabel = mc_tm.predict(X_test[tmp]);
    cout << "Predicted Label: " << predictedLabel << "\n\n";
    printDigit(X_test[tmp]);

    return 0;
}
