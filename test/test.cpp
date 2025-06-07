#include "../src/ekf_localization.h"
#include <iostream>
#include <vector>
#include <iomanip>

void print_state(const std::string& test_name, const Vec3& x, const Mat3& P) {
    std::cout << test_name << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "x = " << x.transpose() << std::endl;
    std::cout << "P =" << std::endl;
    std::cout << P << std::endl;
    std::cout << std::endl;
}

bool test_predict_only() {
    std::cout << "=== Test 1 - Predict Only (Odometry) ===" << std::endl;
    
    // Initial pose:
    Vec3 x{0.0, 0.0, 0.0};
    Mat3 P = Mat3::Identity() * 0.01;
    
    // Odometry
    double delta_d = 1.0;
    double delta_theta = 0.0;
    
    predict(x, P, delta_d, delta_theta, 0.002, 0.002);
    
    print_state("After prediction:", x, P);
    
    // Expected x ≈ [1.0, 0.0, 0.0]
    const double tolerance = 1e-6;
    bool success = (std::abs(x[0] - 1.0) < tolerance) &&
                   (std::abs(x[1] - 0.0) < tolerance) &&
                   (std::abs(x[2] - 0.0) < tolerance);
    
    std::cout << "Test 1 " << (success ? "PASSED" : "FAILED") << std::endl;
    std::cout << std::endl;
    
    return success;
}

bool test_single_tag_update() {
    std::cout << "=== Test 2 - Single Tag Update ===" << std::endl;
    
    // Initial pose
    Vec3 x{1.5, 2.0, 0.0};
    Mat3 P = Mat3::Identity() * 0.01;
    
    // Landmark map
    std::vector<Landmark> map = { {0, 3.0, 1.5} };
    
    // Observation
    std::vector<Observation> obs = { {0, 5, 0.2} };
    
    update(x, P, obs, map);
    
    print_state("After single tag update:", x, P);
    
    float tolerance = 0.1;
    // successful if metrics are within tolerance
    // Expected: x ≈ [-1.15, 2.65, -0.35]
    bool success = (std::abs(x[0] + 1.168) < tolerance) &&
                (std::abs(x[1] - 2.647) < tolerance) &&
                (std::abs(x[2] + 0.362) < tolerance);
    
    std::cout << "Test 2 " << (success ? "PASSED" : "FAILED") << std::endl;
    std::cout << std::endl;
    
    return success;
}

bool test_two_tag_update() {
    std::cout << "=== Test 3 - Two Tag Update ===" << std::endl;
    
    // Initial pose
    Vec3 x{1.5, 2.0, 0.0};
    Mat3 P = Mat3::Identity() * 0.01;
    
    // Landmark map
    std::vector<Landmark> map = {
        {0, 3.0, 1.5},
        {1, 1.5, 4.0}
    };
    
    // Observations:
    std::vector<Observation> obs = {
        {0, 5.0, 0.2},
        {1, 1.8, 1.5}
    };
    
    update(x, P, obs, map);
    
    print_state("After two tag update:", x, P);
    
    float tolerance = 0.1;
    // successful if metrics are within tolerance
    // Expected: x ≈ [-0.5098, 2.8677, -0.7721]
    bool success = (std::abs(x[0] + 0.5098) < tolerance) &&
                (std::abs(x[1] - 2.8677) < tolerance) &&
                (std::abs(x[2] + 0.7721) < tolerance);      
    
    std::cout << "Test 3 " << (success ? "PASSED" : "FAILED") << std::endl;
    std::cout << std::endl;
    
    return success;
}

int main() {
    std::cout << "Running EKF Localization Test Cases" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << std::endl;
    
    bool test1_passed = test_predict_only();
    bool test2_passed = test_single_tag_update();
    bool test3_passed = test_two_tag_update();
    
    std::cout << "=== Test Summary ===" << std::endl;
    std::cout << "Test 1 (Predict Only): " << (test1_passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Test 2 (Single Tag):   " << (test2_passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Test 3 (Two Tags):     " << (test3_passed ? "PASSED" : "FAILED") << std::endl;
    
    int passed_tests = test1_passed + test2_passed + test3_passed;
    std::cout << std::endl;
    std::cout << "Tests passed: " << passed_tests << "/3" << std::endl;
}