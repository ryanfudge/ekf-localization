#include "ekf_localization.h"
#include <iostream>
#include <iomanip>

void print_state(const std::string& description, const Vec3& x, const Mat3& P) {
    std::cout << description << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "x = " << x.transpose() << std::endl;
    std::cout << "P =" << std::endl;
    std::cout << P << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "EKF Localization Demo" << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << std::endl;
    
    // Initial state
    Vec3 x{0.0, 0.0, 0.0};  // [x, y, theta]
    Mat3 P = Mat3::Identity() * 0.01;  // Initial covariance
    
    print_state("Initial state:", x, P);
    
    // Simulate robot movement
    std::cout << "Simulating robot movement (forward 1m)..." << std::endl;
    predict(x, P, 1.0, 0.0, 0.002, 0.002);  // move forward 1m
    
    print_state("After prediction:", x, P);
    
    // Simulate landmark observation
    std::vector<Landmark> map = {{0, 2.0, 1.0}};  // Landmark at (2, 1)
    std::vector<Observation> obs = {{0, 1.414, 0.785}};  // Range ~sqrt(2), bearing ~45deg
    
    std::cout << "Processing landmark observation..." << std::endl;
    update(x, P, obs, map);
    
    print_state("After update:", x, P);
    
    return 0;
}