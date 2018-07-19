cmake_minimum_required(VERSION 2.8)
project( gfrst )

set(CMAKE_CXX_FLAGS "-Wall -Wextra -fopenmp ${Open_MP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
add_executable(gfrst_test main.cpp gfrst.cpp)
target_link_libraries(gfrst_test ${OpenCV_LIBS} ${Open_MP_CXX_FLAGS})
