cmake_minimum_required(VERSION 3.20)
project(task1)

set(CMAKE_CXX_STANDARD 14)

add_executable(float main.cpp)

add_executable(double main.cpp)
target_compile_definitions(double PRIVATE -DUSE_DOUBLE)

