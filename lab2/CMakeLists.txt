cmake_minimum_required(VERSION 3.0.0)
project(lab2)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(lab2 main.cpp img_proc.hpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
