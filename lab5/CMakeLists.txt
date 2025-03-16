cmake_minimum_required(VERSION 3.0.0)
project(lab5 CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4.10.0 REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -Ofast")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(match_corners  src/match_corners.cpp)
target_link_libraries(match_corners ${OpenCV_LIBS})

add_executable(find_homography_matrix  src/find_homography_matrix.cpp)
target_link_libraries(find_homography_matrix ${OpenCV_LIBS})
