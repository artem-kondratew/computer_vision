cmake_minimum_required(VERSION 3.0.0)
project(portrait_mode CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4.10.0 REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -Ofast")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}  src/main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
