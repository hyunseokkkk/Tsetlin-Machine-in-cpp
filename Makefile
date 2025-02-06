# 컴파일러 설정
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

# 실행 파일 이름
TARGET = Tsetlin_Machine

# 소스 파일 목록
SRC = main.cpp TsetlinMachine.cpp MultiClassTsetlin.cpp
OBJ = $(SRC:.cpp=.o)

# 빌드 과정
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 실행
run: $(TARGET)
	./$(TARGET)

# 정리
clean:
	rm -f $(OBJ) $(TARGET)
