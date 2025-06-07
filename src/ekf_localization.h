#pragma once

#include <Eigen/Dense>
#include <vector>

// Data structures
struct Landmark { 
    int id; 
    double x; 
    double y; 
};

struct Observation { 
    int id; 
    double range; 
    double bearing; 
};

// Type aliases
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

// Constants
constexpr double kSigmaR = 0.05; // range noise (m)
constexpr double kSigmaB = 0.02; // bearing noise (rad)

// Function declarations
void predict(Vec3& x, Mat3& P,
             double d, double dtheta,
             double alpha1, double alpha2);

void update(Vec3& x, Mat3& P,
            const std::vector<Observation>& obs,
            const std::vector<Landmark>& map);