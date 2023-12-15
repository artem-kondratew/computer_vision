cmake_minimum_required(VERSION 3.0.0)
project(lab4)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# add_executable(${PROJECT_NAME} main.cpp fourier.hpp timer.hpp)
add_executable(${PROJECT_NAME} old_main.cpp old.hpp old.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
