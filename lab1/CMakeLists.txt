cmake_minimum_required(VERSION 3.0.0)
project(lab1)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV 4.10.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
