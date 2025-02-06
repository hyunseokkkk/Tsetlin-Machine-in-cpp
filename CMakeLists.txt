cmake_minimum_required(VERSION 3.28)
project(Tsetlin_Machine)

set(CMAKE_CXX_STANDARD 17)

# 소스 파일 목록
set(SOURCE_FILES
        main.cpp
        TsetlinMachine.cpp
        TsetlinMachine.h
        MultiClassTsetlin.cpp
        MultiClassTsetlin.h
)

# 실행 파일 생성
add_executable(Tsetlin_Machine ${SOURCE_FILES})
