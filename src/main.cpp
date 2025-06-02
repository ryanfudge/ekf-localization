#include <Eigen/Dense>
#include <iostream>
#include <vector>
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
    x[2] = std::atan2(std::sin(x[2]), std::cos(x[2])); // Normalize angle

    // build the Jacobian Fx = df/dx
    Mat3 Fx = Mat3::Identity();
    Fx(0, 2) = -d * sin_th;
    Fx(1, 2) = d * cos_th;

    // build the noise covariance matrix Q from PR 5.4
    // Assuming alpha1 is the noise in distance and alpha2 is the noise in angle
    const double sig_d2 = alpha1 * std::pow(d, 2);
    const double sig_th2 = alpha2 * std::pow(dtheta, 2);


    Mat3 Q = Mat3::Zero();
    Q(0, 0) = sig_d2;  // Noise in x direction
    Q(1, 1) = sig_d2;  // Noise in y direction  
    Q(2, 2) = sig_th2; // Noise in theta direction

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
        // ∂r/∂x, ∂r/∂y, ∂r/∂θ
        H(row,   0) = -dx / r_pred;
        H(row,   1) = -dy / r_pred;
        H(row,   2) =  0.0;

        // ∂β/∂x, ∂β/∂y, ∂β/∂θ
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


int main() {

    Vec3 x{0.0, 0.0, 0.0};
    Mat3 P = Mat3::Identity() * 0.01;

    // One-step smoke test
    double delta_d = 1.0;
    double delta_theta = 0.0;
    std::vector<Landmark> map = {
        {0, 2.0, 1.0},
        {1, -2.0, 1.5},
        {2, 0.0, -2.0},
        {3, 3.0, -1.0},
        {4, -3.0, -1.0}
    };
    
    std::vector<Observation> obs = { }; // Empty for smoke test
    predict(x, P, delta_d, delta_theta, 0.002, 0.002);
    // No update
    
    std::cout << "x = " << x.transpose() << "\n";
    std::cout << "P =\n" << P << "\n";
    return 0;
}