cmake_minimum_required(VERSION 3.10)

project(EKF_Localization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(Eigen3 REQUIRED)

add_executable(ekf_localization src/main.cpp)
add_executable (test test/test.cpp)
target_link_libraries(ekf_localization Eigen3::Eigen)
target_link_libraries(test Eigen3::Eigen)