cmake_minimum_required(VERSION 3.0.0)
project(rgb_filter CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4.10.0 REQUIRED)
find_package(GTest REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -Ofast")

# enable_testing()

include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(construct_rgb_filter  src/construct_rgb_filter.cpp)
target_link_libraries(construct_rgb_filter ${OpenCV_LIBS})

# add_executable(runTests tests/tests_main.cpp src/max_min_filter.cpp)
# target_link_libraries(runTests ${OpenCV_LIBS} GTest::GTest)
