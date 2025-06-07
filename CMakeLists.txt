cmake_minimum_required(VERSION 3.10)

project(EKF_Localization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(Eigen3 REQUIRED)

# Create a library for the EKF implementation
add_library(ekf_lib src/ekf_localization.cpp)
target_include_directories(ekf_lib PUBLIC src)
target_link_libraries(ekf_lib Eigen3::Eigen)

# Main executable
add_executable(ekf_localization src/main.cpp)
target_link_libraries(ekf_localization ekf_lib Eigen3::Eigen)

# Test executable
add_executable(test test/test.cpp)
target_link_libraries(test ekf_lib Eigen3::Eigen)