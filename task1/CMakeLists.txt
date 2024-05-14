cmake_minimum_required(VERSION 3.22)

project(task1)

add_executable(main main.cpp)

if(DATA_TYPE STREQUAL "DOUBLE")
    target_compile_definitions(main PRIVATE DATA_TYPE_DOUBLE)
elseif(DATA_TYPE STREQUAL "FLOAT")
    target_compile_definitions(main PRIVATE DATA_TYPE_FLOAT)
else()
    message(FATAL_ERROR "Unsupported data type. Please choose DOUBLE or FLOAT.")
endif()
