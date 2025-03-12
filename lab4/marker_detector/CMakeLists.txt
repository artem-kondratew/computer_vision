cmake_minimum_required(VERSION 3.0.0)
project(marker_detector CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4.10.0 REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -Ofast")

include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(marker_detector  src/marker_detector.cpp)
target_link_libraries(marker_detector ${OpenCV_LIBS})