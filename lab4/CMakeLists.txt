cmake_minimum_required(VERSION 3.0.0)
project(lab4)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# add_executable(${PROJECT_NAME} main.cpp fourier.hpp timer.hpp)
add_executable(${PROJECT_NAME} old_main.cpp old.hpp old.cpp)
# add_executable(${PROJECT_NAME} task.cpp)
# add_executable(${PROJECT_NAME} g.cpp)
# add_executable(${PROJECT_NAME} v.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
