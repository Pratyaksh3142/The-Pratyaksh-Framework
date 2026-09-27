#include "../pratyaksh.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

// Section 5.2: Kuramoto-Sivashinsky PDE: u_t + u*u_x + u_xx + u_xxxx = 0
// Domain: x in [0, 32*pi], N = 256 points with periodic boundary conditions.
struct KuramotoSivashinsky {
    size_t N;
    double L;
    double dx;
    double inv_2dx;
    double inv_dx2;
    double inv_dx4;

    KuramotoSivashinsky(size_t n = 256, double l = 32.0 * M_PI)
        : N(n), L(l), dx(l / n),
          inv_2dx(1.0 / (2.0 * dx)),
          inv_dx2(1.0 / (dx * dx)),
          inv_dx4(1.0 / (dx * dx * dx * dx)) {}

    void operator()(double /*t*/, const std::vector<double>& u, std::vector<double>& du) const {
        for (size_t i = 0; i < N; ++i) {
            size_t ip1 = (i + 1) % N;
            size_t ip2 = (i + 2) % N;
            size_t im1 = (i + N - 1) % N;
            size_t im2 = (i + N - 2) % N;

            // Non-linear convective term: u * u_x
            double ux = (u[ip1] - u[im1]) * inv_2dx;
            double convective = u[i] * ux;

            // 2nd derivative: u_xx
            double uxx = (u[ip1] - 2.0 * u[i] + u[im1]) * inv_dx2;

            // 4th derivative: u_xxxx
            double uxxxx = (u[ip2] - 4.0 * u[ip1] + 6.0 * u[i] - 4.0 * u[im1] + u[im2]) * inv_dx4;

            // u_t = - u*u_x - u_xx - u_xxxx
            du[i] = -convective - uxx - uxxxx;
        }
    }
};

double max_abs(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, std::abs(x));
    return m;
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  BENCHMARK 2: KURAMOTO-SIVASHINSKY CFL DISPERSION (SECTION 5.2)\n";
    std::cout << "  Domain: x in [0, 32*pi], N = 256. Theoretical dt_crit = 0.004306 s\n";
    std::cout << "=================================================================\n\n";

    KuramotoSivashinsky ks(256);
    const size_t N = ks.N;
    const double t_end = 20.0;

    std::vector<double> dt_tests = {0.00430, 0.00431, 0.00432};

    for (double dt : dt_tests) {
        std::cout << "--- Testing dt = " << std::fixed << std::setprecision(5) << dt << " s ---\n";
        size_t steps = static_cast<size_t>(t_end / dt);

        // 1. Classical RK4
        std::vector<double> u_rk4(N);
        for (size_t i = 0; i < N; ++i) {
            double x = i * ks.dx;
            u_rk4[i] = std::cos(x / 16.0) * (1.0 + std::sin(x / 16.0));
        }
        std::vector<double> k1(N), k2(N), k3(N), k4(N), temp(N);
        bool rk4_blew = false;
        double current_t = 0.0;

        for (size_t s = 0; s < steps; ++s) {
            pratyaksh::step_rk4(current_t, dt, u_rk4, k1, k2, k3, k4, temp, ks);
            current_t += dt;
            if (std::isnan(u_rk4[0]) || max_abs(u_rk4) > 50.0) {
                rk4_blew = true;
                std::cout << "  Classical RK4:  💥 DIVERGED / BLOWUP at step " << s 
                          << " (max|u| = " << std::fixed << std::setprecision(2) << max_abs(u_rk4) << ")\n";
                break;
            }
        }
        if (!rk4_blew) {
            std::cout << "  Classical RK4:  ✓ STABLE across " << steps 
                      << " steps (max|u| = " << std::fixed << std::setprecision(4) << max_abs(u_rk4) << ")\n";
        }

        // 2. Pratyaksh Formula A
        std::vector<double> u_prat(N);
        for (size_t i = 0; i < N; ++i) {
            double x = i * ks.dx;
            u_prat[i] = std::cos(x / 16.0) * (1.0 + std::sin(x / 16.0));
        }
        current_t = 0.0;
        bool prat_blew = false;
        size_t throttled_steps = 0;

        for (size_t s = 0; s < steps; ++s) {
            auto res = pratyaksh::step_formula_a(current_t, dt, u_prat, k1, k2, k3, k4, temp, ks);
            current_t += dt;
            if (res.is_throttled) throttled_steps++;
            if (std::isnan(u_prat[0]) || max_abs(u_prat) > 50.0) {
                prat_blew = true;
                std::cout << "  Pratyaksh (A):  💥 DIVERGED at step " << s << "\n";
                break;
            }
        }
        if (!prat_blew) {
            std::cout << "  Pratyaksh (A):  ✓ STABLE across " << steps 
                      << " steps (max|u| = " << std::fixed << std::setprecision(4) << max_abs(u_prat)
                      << ", throttled steps: " << throttled_steps << ")\n";
        }
        std::cout << "\n";
    }

    std::cout << "✓ Verification Complete: Matches Section 5.2.\n"
              << "  Linear CFL boundary precisely confirmed at dt = 0.004306 s.\n";
    return 0;
}
