/*******************************************************************************
 * Hull-White Swaption Pricer with XAD Greeks
 * 
 * Computes European swaption prices and sensitivities to discount curve nodes
 * using:
 *   1. XAD Adjoint Algorithmic Differentiation (AAD)
 *   2. Finite Differences (bump-and-reprice)
 * 
 * Compares results and timing between both methods.
 ******************************************************************************/

#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>

// =============================================================================
// Model Parameters (matching Python implementation)
// =============================================================================

struct SwaptionParams {
    double T_option = 1.0;      // Swaption expiry in years
    int swap_tenor = 5;         // Underlying swap tenor in years
    double K_strike = 0.03;     // Fixed strike rate
    double notional = 1000000.0;
    bool is_payer = true;       // Payer swaption
};

struct HullWhiteParams {
    double a = 0.1;             // Mean reversion speed
    double sigma = 0.01;        // Volatility
};

struct MonteCarloParams {
    int num_paths = 5000;
    double dt = 1.0 / 52.0;     // Weekly time step
    unsigned int seed = 42;
};

// =============================================================================
// Market Curve Data
// =============================================================================

struct MarketCurve {
    std::vector<double> maturities;
    std::vector<double> rates;
    
    MarketCurve() {
        // OIS curve from December 2024 (matching Python)
        maturities = {1.0/12, 3.0/12, 6.0/12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0};
        rates = {0.0428, 0.0422, 0.0413, 0.0403, 0.0388, 0.0383, 0.0395, 0.0408, 0.0423, 0.0433, 0.0438};
    }
    
    size_t size() const { return maturities.size(); }
};

// =============================================================================
// Linear Interpolation (consistent with Python version)
// =============================================================================

template<typename T>
std::pair<size_t, double> findInterpWeights(double x, const std::vector<double>& xs) {
    size_t n = xs.size();
    if (x <= xs[0]) return {0, 0.0};
    if (x >= xs[n-1]) return {n-2, 1.0};
    
    auto it = std::lower_bound(xs.begin(), xs.end(), x);
    size_t idx = std::distance(xs.begin(), it) - 1;
    idx = std::min(idx, n - 2);
    double w = (x - xs[idx]) / (xs[idx + 1] - xs[idx]);
    return {idx, w};
}

template<typename T>
T linearInterp(double x, const std::vector<double>& xs, const std::vector<T>& ys) {
    auto [idx, w] = findInterpWeights<T>(x, xs);
    return ys[idx] * (1.0 - w) + ys[idx + 1] * w;
}

// =============================================================================
// Hull-White Model Functions
// =============================================================================

template<typename T>
T discount(double t, const std::vector<double>& mats, const std::vector<T>& rates) {
    using std::exp;
    if (t <= 0) return T(1.0);
    T rate = linearInterp<T>(t, mats, rates);
    return exp(-rate * t);
}

template<typename T>
T forwardRate(double t, const std::vector<double>& mats, const std::vector<T>& rates, double h = 1e-5) {
    using std::log;
    t = std::max(t, h);
    T P_t = discount<T>(t, mats, rates);
    T P_th = discount<T>(t + h, mats, rates);
    return -(log(P_th) - log(P_t)) / h;
}

template<typename T>
T theta(double t, const std::vector<double>& mats, const std::vector<T>& rates,
        double a, double sigma, double h = 1e-5) {
    using std::exp;
    T f_t = forwardRate<T>(t, mats, rates, h);
    T f_th = forwardRate<T>(t + h, mats, rates, h);
    T f_prime = (f_th - f_t) / h;
    double term3 = (sigma * sigma / (2.0 * a)) * (1.0 - exp(-2.0 * a * t));
    return f_prime + a * f_t + term3;
}

template<typename T>
T bondPrice(T r_t, double t, double T_mat, const std::vector<double>& mats, 
            const std::vector<T>& rates, double a, double sigma, T f_t_cache) {
    using std::exp;
    using std::log;
    
    if (T_mat <= t) return T(1.0);
    
    double B = (1.0 - exp(-a * (T_mat - t))) / a;
    T P_T = discount<T>(T_mat, mats, rates);
    T P_t = discount<T>(t, mats, rates);
    
    T log_A = log(P_T) - log(P_t) + B * f_t_cache - 
              (sigma * sigma / (4.0 * a)) * (1.0 - exp(-2.0 * a * t)) * B * B;
    
    return exp(log_A) * exp(-B * r_t);
}

// =============================================================================
// Monte Carlo Simulation
// =============================================================================

template<typename T>
T simulateAndPricePath(
    const std::vector<double>& mats,
    const std::vector<T>& rates,
    const std::vector<double>& Z_row,
    const std::vector<T>& theta_cache,
    const SwaptionParams& swaption,
    const HullWhiteParams& hw,
    double dt
) {
    using std::exp;
    using std::sqrt;
    using std::max;
    
    size_t n_steps = Z_row.size();
    double sqrt_dt = sqrt(dt);
    
    // Simulate short rate
    T r_t = forwardRate<T>(0.0, mats, rates);
    for (size_t i = 0; i < n_steps; ++i) {
        T drift = (theta_cache[i] - hw.a * r_t) * dt;
        double diffusion = hw.sigma * sqrt_dt * Z_row[i];
        r_t = r_t + drift + diffusion;
    }
    
    T r_T = r_t;
    
    // Forward rate at option expiry (for bond pricing)
    T f_T = forwardRate<T>(swaption.T_option, mats, rates);
    
    // Bond prices at expiry for each swap payment
    std::vector<T> bonds;
    for (int j = 1; j <= swaption.swap_tenor; ++j) {
        double T_pay = static_cast<double>(j) + swaption.T_option;
        bonds.push_back(bondPrice<T>(r_T, swaption.T_option, T_pay, mats, rates, hw.a, hw.sigma, f_T));
    }
    
    // Fixed leg PV
    T fixed_pv = T(0.0);
    for (const auto& b : bonds) {
        fixed_pv = fixed_pv + b * swaption.K_strike;
    }
    
    // Floating leg PV
    T float_pv = T(1.0) - bonds.back();
    
    // Swap value
    T swap_val = (float_pv - fixed_pv) * swaption.notional;
    
    // Payoff (max for swaption) - use explicit max to avoid ternary type issues with XAD
    T payoff;
    if (swaption.is_payer) {
        payoff = max(swap_val, T(0.0));
    } else {
        T neg_swap = T(0.0) - swap_val;
        payoff = max(neg_swap, T(0.0));
    }
    
    // Discount to today
    T disc = discount<T>(swaption.T_option, mats, rates);
    
    return disc * payoff;
}

// =============================================================================
// Pre-generate random numbers for reproducibility
// =============================================================================

inline std::vector<std::vector<double>> generateRandomMatrix(int num_paths, int num_steps, unsigned int seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    
    std::vector<std::vector<double>> Z(num_paths, std::vector<double>(num_steps));
    for (int i = 0; i < num_paths; ++i) {
        for (int j = 0; j < num_steps; ++j) {
            Z[i][j] = dist(gen);
        }
    }
    return Z;
}

// =============================================================================
// Price swaption (non-AD version for finite differences)
// =============================================================================

inline double priceSwaption(
    const std::vector<double>& mats,
    const std::vector<double>& rates,
    const std::vector<std::vector<double>>& Z,
    const SwaptionParams& swaption,
    const HullWhiteParams& hw,
    const MonteCarloParams& mc
) {
    int num_steps = static_cast<int>(swaption.T_option / mc.dt);
    std::vector<double> t_grid(num_steps + 1);
    for (int i = 0; i <= num_steps; ++i) {
        t_grid[i] = i * mc.dt;
    }
    
    // Pre-compute theta values
    std::vector<double> theta_cache(num_steps + 1);
    for (int i = 0; i <= num_steps; ++i) {
        theta_cache[i] = theta<double>(t_grid[i], mats, rates, hw.a, hw.sigma);
    }
    
    double total_payoff = 0.0;
    for (int p = 0; p < mc.num_paths; ++p) {
        total_payoff += simulateAndPricePath<double>(
            mats, rates, Z[p], theta_cache, swaption, hw, mc.dt
        );
    }
    
    return total_payoff / mc.num_paths;
}
