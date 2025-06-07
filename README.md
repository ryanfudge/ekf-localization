# EKF Localization

A C++ implementation of Extended Kalman Filter (EKF) for 2D robot localization using landmark observations.

## Overview

This project implements a 2D EKF localization system that estimates a robot's pose (x, y, θ) by fusing:
- **Odometry data**: Distance and angular displacement measurements
- **Landmark observations**: Range and bearing measurements to known landmarks

## Features

- 3-DOF state estimation (x, y, theta)
- Motion prediction using odometry
- Sensor update using range-bearing landmark observations
- Configurable process and measurement noise parameters
- Comprehensive test suite

## Dependencies

- **Eigen3**: Linear algebra library
- **CMake**: Build system (version 3.10+)
- **C++17** compatible compiler

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

### Demo
Run the main demo:
```bash
./ekf_localization
```

### Tests
Run the test suite:
```bash
./test
```

## API

### Core Functions

```cpp
// Prediction step using odometry
void predict(Vec3& x, Mat3& P, double d, double dtheta, 
             double alpha1, double alpha2);

// Update step using landmark observations
void update(Vec3& x, Mat3& P, 
            const std::vector<Observation>& obs,
            const std::vector<Landmark>& map);
```

### Data Structures

```cpp
struct Landmark { int id; double x; double y; };
struct Observation { int id; double range; double bearing; };
```

## Test Cases

The test suite includes:
1. **Predict Only**: Odometry-based motion prediction
2. **Single Tag Update**: Localization with one landmark
3. **Two Tag Update**: Improved accuracy with multiple landmarks

Test cases were derived and verified by hand