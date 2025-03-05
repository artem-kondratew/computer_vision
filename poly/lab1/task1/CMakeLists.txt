cmake_minimum_required(VERSION 3.0.0)
project(lab1 VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(lab1 main.cpp robot.hpp robot.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
