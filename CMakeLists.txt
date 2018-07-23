cmake_minimum_required(VERSION 2.8)
project( gfrst )

set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wextra -fopenmp ${Open_MP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_library(gfrst_lib gfrst.cpp gfrst.h gfrstMultiscale.cpp gfrstMultiscale.h polygone.cpp polygone.h)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
add_executable(gfrst_test main.cpp)
target_link_libraries(gfrst_test gfrst_lib ${OpenCV_LIBS} ${Open_MP_CXX_FLAGS})

add_executable(gradientfield gradientfield.cpp)
target_link_libraries(gradientfield gfrst_lib ${OpenCV_LIBS})

add_executable(multiscale_test multiscale_test.cpp)
target_link_libraries(multiscale_test gfrst_lib ${OpenCV_LIBS})
