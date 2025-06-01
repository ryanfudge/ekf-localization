#include <Eigen/Dense>
#include <iostream>
#include <vector>

struct Landmark { int id; double x; double y; };
struct Observation{ int id; double range; double bearing; };

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

constexpr double kSigmaR = 0.05; // range noise (m)
constexpr double kSigmaB = 0.02; // bearing noise (rad)

void predict(Vec3& x, Mat3& P,
             double d, double dtheta,
             double alpha1, double alpha2) {

}

void update(Vec3& x, Mat3& P,
             const std::vector<Observation>& obs,
             const std::vector<Landmark>& map) {

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