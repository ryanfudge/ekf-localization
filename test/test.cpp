#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <iomanip>

#include <unordered_map>

struct Landmark { int id; double x; double y; };
struct Observation{ int id; double range; double bearing; };

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

constexpr double kSigmaR = 0.05; // range noise (m)
constexpr double kSigmaB = 0.02; // bearing noise (rad)

void predict(Vec3& x, Mat3& P,
             double d, double dtheta,
             double alpha1, double alpha2) {
    
    const double cos_th  = std::cos(x[2]);
    const double sin_th  = std::sin(x[2]);
    
    // update state
    x[0] += d * cos_th;
    x[1] += d * sin_th;
    x[2] += dtheta;
    x[2] = std::atan2(std::sin(x[2]), std::cos(x[2])); // normalize angle

    // build the Jacobian Fx = df/dx
    Mat3 Fx = Mat3::Identity();
    Fx(0, 2) = -d * sin_th;
    Fx(1, 2) = d * cos_th;

    // build the noise covariance matrix Q
    double var_d = alpha1 * d * d;
    double var_th = alpha2 * dtheta * dtheta;

    Eigen::Matrix2d M = Eigen::Matrix2d::Zero();
    M(0, 0) = var_d; // variance in x direction
    M(1, 1) = var_th; // variance in theta direction

    // build G, the Jacobian of the noise model
    Eigen::Matrix<double, 3, 2> G;
    G(0, 0) = cos_th;          // ∂x/∂d
    G(0, 1) = -d * sin_th;     // ∂x/∂θ
    G(1, 0) = sin_th;          // ∂y/∂d
    G(1, 1) = d * cos_th;      // ∂y/∂θ
    G(2, 0) = 0.0;             // ∂θ/∂d
    G(2, 1) = 1.0;             // ∂θ/∂θ

    Mat3 Q = G * M * G.transpose(); // 3 × 3 noise covariance matrix 
 //   Mat3 Q = Mat3::Zero();
   // Q(0, 0) = var_d;  // Noise in x direction
   // Q(1, 1) = var_d;  // Noise in y direction  
    //Q(2, 2) = var_th; // Noise in theta direction */

    // Update the covariance matrix P
    P = Fx * P * Fx.transpose() + Q;
}

void update(Vec3& x, Mat3& P,
            const std::vector<Observation>& obs,
            const std::vector<Landmark>&    map)
{
    const std::size_t m = obs.size();        
    if (m == 0) return;                      

    // 1.  Fast landmark lookup:  id → pointer to Landmark
    std::unordered_map<int, const Landmark*> id2lm;
    id2lm.reserve(map.size());
    for (const auto& lm : map) id2lm[lm.id] = &lm;

    // 2.  Allocate stacks:  innovation y  and Jacobian H (2m × 3)
    Eigen::VectorXd y(2 * m);
    Eigen::MatrixXd H(2 * m, 3);

    // 3.  Fill y and H, one block (2 rows) per observation
    std::size_t row = 0;
    for (const auto& z : obs)
    {
        auto it = id2lm.find(z.id);
        if (it == id2lm.end()) continue;     // skip unknown tag

        const double dx = it->second->x - x[0];
        const double dy = it->second->y - x[1];
        const double q  = dx*dx + dy*dy;          // range²
        const double r_pred = std::sqrt(q);
        const double b_pred = std::atan2(dy, dx) - x[2];
        
        // Normalize bearing prediction to [-pi, pi]
        const double b_pred_norm = std::atan2(std::sin(b_pred), std::cos(b_pred));

        // Innovation  (z − h(x))
        y(row)     = z.range - r_pred;

        // Properly handle angle wrapping for bearing innovation
        double bearing_diff = z.bearing - b_pred_norm;
        y(row+1)   = std::atan2(std::sin(bearing_diff), std::cos(bearing_diff));

        // Jacobian  H_i  (2×3)
        // dr/dx, dr/dy, dr/dθ
        H(row,   0) = -dx / r_pred;
        H(row,   1) = -dy / r_pred;
        H(row,   2) =  0.0;

        // dB/dx, dB/dy, dB/dθ
        H(row+1, 0) =  dy / q;
        H(row+1, 1) = -dx / q;
        H(row+1, 2) = -1.0;

        row += 2; // advance to next block
    }
    
    // Resize matrices to actual used rows (in case some landmarks were skipped)
    y.conservativeResize(row);
    H.conservativeResize(row, 3);

    // 4.  Measurement-noise covariance  R  (block-diagonal, 2m × 2m)
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(row, row);
    const double var_r = kSigmaR * kSigmaR;
    const double var_b = kSigmaB * kSigmaB;
    for (std::size_t i = 0; i < row/2; ++i) {
        R(2*i,   2*i)   = var_r;            // range variance
        R(2*i+1, 2*i+1) = var_b;            // bearing variance
    }

    // 5.  EKF update
    Eigen::MatrixXd S = H * P * H.transpose() + R;    // row × row
    Eigen::MatrixXd K = P * H.transpose() * S.inverse(); // 3 × row

    x += K * y;                                       // mean
    x[2] = std::atan2(std::sin(x[2]), std::cos(x[2])); // wrap θ

    const Mat3 I = Mat3::Identity();
    P = (I - K * H) * P;                              // covariance
}

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