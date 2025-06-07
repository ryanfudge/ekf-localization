#include "ekf_localization.h"
#include <unordered_map>
#include <cmath>

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

    // Jacobian Fx = df/dx
    Mat3 Fx = Mat3::Identity();
    Fx(0, 2) = -d * sin_th;
    Fx(1, 2) = d * cos_th;

    // noise covariance matrix Q
    double var_d = alpha1 * d * d;
    double var_th = alpha2 * dtheta * dtheta;

    Eigen::Matrix2d M = Eigen::Matrix2d::Zero();
    M(0, 0) = var_d; // variance in x 
    M(1, 1) = var_th; // variance in theta

    // G, the Jacobian of the noise model
    Eigen::Matrix<double, 3, 2> G;
    G(0, 0) = cos_th;          // ∂x/∂d
    G(0, 1) = -d * sin_th;     // ∂x/∂θ
    G(1, 0) = sin_th;          // ∂y/∂d
    G(1, 1) = d * cos_th;      // ∂y/∂θ
    G(2, 0) = 0.0;             // ∂θ/∂d
    G(2, 1) = 1.0;             // ∂θ/∂θ

    Mat3 Q = G * M * G.transpose(); // 3 × 3 noise covariance matrix

    // Update covariance matrix P
    P = Fx * P * Fx.transpose() + Q;
}

void update(Vec3& x, Mat3& P,
            const std::vector<Observation>& obs,
            const std::vector<Landmark>&    map)
{
    const std::size_t m = obs.size();        
    if (m == 0) return;                      

    // fast landmark lookup:  id → pointer to Landmark
    std::unordered_map<int, const Landmark*> id2lm;
    id2lm.reserve(map.size());
    for (const auto& lm : map) id2lm[lm.id] = &lm;

    // Allocate stacks:  innovation y  and Jacobian H (2m × 3)
    Eigen::VectorXd y(2 * m);
    Eigen::MatrixXd H(2 * m, 3);

    // Fill y and H, one block (2 rows) per observation
    std::size_t row = 0;
    for (const auto& z : obs)
    {
        auto it = id2lm.find(z.id);
        if (it == id2lm.end()) continue;     // skip unknown tag

        const double dx = it->second->x - x[0];
        const double dy = it->second->y - x[1];
        const double q  = dx*dx + dy*dy;          // range^2
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

    // Measurement-noise covariance  R  (block-diagonal, 2m × 2m)
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