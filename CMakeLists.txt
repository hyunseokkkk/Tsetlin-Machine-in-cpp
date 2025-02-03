cmake_minimum_required(VERSION 3.28)
project(Tsetlin_Machine)

set(CMAKE_CXX_STANDARD 17)

add_executable(Tsetlin_Machine main.cpp
        TsetlinMachine.cpp
        TsetlinMachine.h
        MultiClassTsetlin.cpp
        MultiClassTsetlin.h
)
