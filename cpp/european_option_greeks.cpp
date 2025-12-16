/**
 * European Option Pricing with Monte Carlo Simulation
 * Greeks Calculation: XAD (AAD) vs Finite Differences Comparison
 *
 * This demonstrates:
 * 1. European call/put option pricing via Monte Carlo
 * 2. Greeks calculation using XAD Adjoint Algorithmic Differentiation
 * 3. Greeks calculation using Finite Differences
 * 4. Timing comparison between XAD and FD approaches
 *
 * Key: Each MC path uses its own tape, with adjoints computed per-path
 * and accumulated across all paths.
 * 
 * Build: cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . --config Release
 * Requires XAD library: https://github.com/auto-differentiation/XAD
 */

#define _USE_MATH_DEFINES
#include <cmath>

#include <XAD/XAD.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <string>

// XAD type aliases using adjoint mode
using mode = xad::adj<double>;
using tape_type = mode::tape_type;
using AReal = mode::active_type;

// =============================================================================
// Option Parameters Structure
// =============================================================================

struct OptionParams {
    double S0;      // Spot price
    double K;       // Strike price
    double T;       // Time to maturity (years)
    double r;       // Risk-free rate
    double sigma;   // Volatility
    bool isCall;    // true = call, false = put
};

// =============================================================================
// Greeks Results Structure
// =============================================================================

struct GreeksResult {
    double price;
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
};

// =============================================================================
// Random Number Generation
// =============================================================================

std::vector<double> generateNormals(size_t n, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> Z(n);
    for (size_t i = 0; i < n; ++i) {
        Z[i] = dist(gen);
    }
    return Z;
}

// =============================================================================
// Black-Scholes Analytical (for validation)
// =============================================================================

double normalCDF(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double normalPDF(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

GreeksResult blackScholesAnalytical(const OptionParams& opt) {
    double d1 = (std::log(opt.S0 / opt.K) + (opt.r + 0.5 * opt.sigma * opt.sigma) * opt.T) 
                / (opt.sigma * std::sqrt(opt.T));
    double d2 = d1 - opt.sigma * std::sqrt(opt.T);
    
    GreeksResult result;
    
    if (opt.isCall) {
        result.price = opt.S0 * normalCDF(d1) - opt.K * std::exp(-opt.r * opt.T) * normalCDF(d2);
        result.delta = normalCDF(d1);
        result.rho = opt.K * opt.T * std::exp(-opt.r * opt.T) * normalCDF(d2);
    } else {
        result.price = opt.K * std::exp(-opt.r * opt.T) * normalCDF(-d2) - opt.S0 * normalCDF(-d1);
        result.delta = normalCDF(d1) - 1.0;
        result.rho = -opt.K * opt.T * std::exp(-opt.r * opt.T) * normalCDF(-d2);
    }
    
    result.gamma = normalPDF(d1) / (opt.S0 * opt.sigma * std::sqrt(opt.T));
    result.vega = opt.S0 * normalPDF(d1) * std::sqrt(opt.T);
    
    double theta_common = -opt.S0 * normalPDF(d1) * opt.sigma / (2.0 * std::sqrt(opt.T));
    if (opt.isCall) {
        result.theta = (theta_common - opt.r * opt.K * std::exp(-opt.r * opt.T) * normalCDF(d2)) / 365.0;
    } else {
        result.theta = (theta_common + opt.r * opt.K * std::exp(-opt.r * opt.T) * normalCDF(-d2)) / 365.0;
    }
    
    return result;
}

// =============================================================================
// Monte Carlo Pricing (Standard double)
// =============================================================================

double monteCarloPrice(const OptionParams& opt, const std::vector<double>& Z) {
    size_t nPaths = Z.size();
    double sumPayoff = 0.0;
    
    double drift = (opt.r - 0.5 * opt.sigma * opt.sigma) * opt.T;
    double vol = opt.sigma * std::sqrt(opt.T);
    
    for (size_t i = 0; i < nPaths; ++i) {
        double S_T = opt.S0 * std::exp(drift + vol * Z[i]);
        double payoff = opt.isCall ? std::max(S_T - opt.K, 0.0) : std::max(opt.K - S_T, 0.0);
        sumPayoff += payoff;
    }
    
    return std::exp(-opt.r * opt.T) * sumPayoff / static_cast<double>(nPaths);
}

// =============================================================================
// Finite Differences Greeks
// =============================================================================

GreeksResult computeGreeksFD(const OptionParams& opt, const std::vector<double>& Z) {
    GreeksResult result;
    
    double dS = opt.S0 * 0.01;
    double dsigma = 0.01;
    double dT = 1.0 / 365.0;
    double dr = 0.0001;
    
    // Base price
    result.price = monteCarloPrice(opt, Z);
    
    // Delta and Gamma
    OptionParams opt_up = opt, opt_down = opt;
    opt_up.S0 = opt.S0 + dS;
    opt_down.S0 = opt.S0 - dS;
    double V_up = monteCarloPrice(opt_up, Z);
    double V_down = monteCarloPrice(opt_down, Z);
    result.delta = (V_up - V_down) / (2.0 * dS);
    result.gamma = (V_up - 2.0 * result.price + V_down) / (dS * dS);
    
    // Vega
    opt_up = opt; opt_down = opt;
    opt_up.sigma = opt.sigma + dsigma;
    opt_down.sigma = opt.sigma - dsigma;
    double V_sigma_up = monteCarloPrice(opt_up, Z);
    double V_sigma_down = monteCarloPrice(opt_down, Z);
    result.vega = (V_sigma_up - V_sigma_down) / (2.0 * dsigma);
    
    // Theta
    if (opt.T > dT) {
        opt_down = opt;
        opt_down.T = opt.T - dT;
        double V_T_down = monteCarloPrice(opt_down, Z);
        result.theta = V_T_down - result.price;  // Daily P&L as time passes
    } else {
        result.theta = 0.0;
    }
    
    // Rho
    opt_up = opt; opt_down = opt;
    opt_up.r = opt.r + dr;
    opt_down.r = opt.r - dr;
    double V_r_up = monteCarloPrice(opt_up, Z);
    double V_r_down = monteCarloPrice(opt_down, Z);
    result.rho = (V_r_up - V_r_down) / (2.0 * dr);
    
    return result;
}

// =============================================================================
// XAD Single Path Pricing (with soft-max for differentiability)
// =============================================================================

template<typename T>
T priceSinglePathXAD(T S0, T T_mat, T r, T sigma, double K, double z, bool isCall, double smoothing = 0.001) {
    double eps = smoothing * K;
    
    // GBM simulation
    T drift = (r - sigma * sigma * 0.5) * T_mat;
    T sqrt_T = xad::sqrt(T_mat);
    T S_T = S0 * xad::exp(drift + sigma * sqrt_T * z);
    
    // Payoff with soft-max approximation: max(x,0) ≈ (x + sqrt(x² + ε²)) / 2
    // Use explicit if/else to avoid ternary operator type conversion issues with XAD
    T x;
    if (isCall) {
        x = S_T - K;
    } else {
        x = K - S_T;
    }
    T payoff = (x + xad::sqrt(x * x + eps * eps)) * 0.5;
    
    // Discount
    T discount = xad::exp(-r * T_mat);
    
    return discount * payoff;
}

// =============================================================================
// XAD Greeks (OPTIMIZED: Single Tape for ALL paths - Batched Recording)
// =============================================================================
// XAD Greeks - Per-Path Tape with Reuse (Correct MC Pattern)
// =============================================================================
// For Monte Carlo, we must compute adjoints per-path because each path has
// different random numbers. We reuse the tape to minimize allocation overhead.

GreeksResult computeGreeksXAD(const OptionParams& opt, const std::vector<double>& Z) {
    size_t nPaths = Z.size();
    double n = static_cast<double>(nPaths);
    
    // Accumulators
    double accPrice = 0.0;
    double accDelta = 0.0;
    double accVega = 0.0;
    double accTheta = 0.0;
    double accRho = 0.0;
    
    // Create ONE tape and reuse it across paths
    tape_type tape;
    
    // Process each path
    for (size_t i = 0; i < nPaths; ++i) {
        // Create active variables
        AReal S0_ad = opt.S0;
        AReal T_ad = opt.T;
        AReal r_ad = opt.r;
        AReal sigma_ad = opt.sigma;
        
        // Register inputs
        tape.registerInput(S0_ad);
        tape.registerInput(T_ad);
        tape.registerInput(r_ad);
        tape.registerInput(sigma_ad);
        
        // Start recording
        tape.newRecording();
        
        // Compute discounted payoff for this path
        AReal payoff = priceSinglePathXAD(S0_ad, T_ad, r_ad, sigma_ad, opt.K, Z[i], opt.isCall);
        
        // Register output and compute adjoints
        tape.registerOutput(payoff);
        derivative(payoff) = 1.0;
        tape.computeAdjoints();
        
        // Accumulate
        accPrice += value(payoff);
        accDelta += derivative(S0_ad);
        accTheta += derivative(T_ad);
        accRho += derivative(r_ad);
        accVega += derivative(sigma_ad);
        
        // Clear tape for next path (keeps memory allocated - key optimization!)
        tape.clearAll();
    }
    
    GreeksResult result;
    result.price = accPrice / n;
    result.delta = accDelta / n;
    result.vega = accVega / n;
    result.rho = accRho / n;
    result.theta = -(accTheta / n) / 365.0;
    
    // Gamma via FD on delta
    double dS = opt.S0 * 0.01;
    
    auto computeDelta = [&](double S0_bumped) -> double {
        double accD = 0.0;
        for (size_t i = 0; i < nPaths; ++i) {
            AReal S0_ad = S0_bumped;
            tape.registerInput(S0_ad);
            tape.newRecording();
            AReal T_ad = opt.T, r_ad = opt.r, sigma_ad = opt.sigma;
            AReal payoff = priceSinglePathXAD(S0_ad, T_ad, r_ad, sigma_ad, opt.K, Z[i], opt.isCall);
            tape.registerOutput(payoff);
            derivative(payoff) = 1.0;
            tape.computeAdjoints();
            accD += derivative(S0_ad);
            tape.clearAll();
        }
        return accD / n;
    };
    
    result.gamma = (computeDelta(opt.S0 + dS) - computeDelta(opt.S0 - dS)) / (2.0 * dS);
    
    return result;
}

// =============================================================================
// Timing Helper
// =============================================================================// =============================================================================
// Timing Helper
// =============================================================================

class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// =============================================================================
// Print Results
// =============================================================================

void printResults(const std::string& name, const GreeksResult& r, double time = -1.0) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Price:  $" << r.price << "\n";
    std::cout << "  Delta:   " << r.delta << "\n";
    std::cout << "  Gamma:   " << r.gamma << "\n";
    std::cout << "  Vega:    " << r.vega << "\n";
    std::cout << "  Theta:  $" << r.theta << "/day\n";
    std::cout << "  Rho:     " << r.rho << "\n";
    if (time >= 0) {
        std::cout << "\n  Time:    " << std::setprecision(4) << time << " seconds\n";
    }
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << std::string(80, '=') << "\n";
    std::cout << "European Option Greeks: XAD (AAD) vs Finite Differences Comparison (C++)\n";
    std::cout << std::string(80, '=') << "\n";
    
    // Option parameters
    OptionParams opt;
    opt.S0 = 100.0;
    opt.K = 100.0;
    opt.T = 1.0;
    opt.r = 0.05;
    opt.sigma = 0.20;
    opt.isCall = true;
    
    size_t nPaths = 10000;  // Reduced for fair AAD comparison
    unsigned int seed = 42;
    
    std::cout << "\nOption Parameters:\n";
    std::cout << "  Spot (S0):     $" << opt.S0 << "\n";
    std::cout << "  Strike (K):    $" << opt.K << "\n";
    std::cout << "  Maturity (T):  " << opt.T << " years\n";
    std::cout << "  Rate (r):      " << opt.r * 100 << "%\n";
    std::cout << "  Volatility:    " << opt.sigma * 100 << "%\n";
    std::cout << "  Option Type:   " << (opt.isCall ? "Call" : "Put") << "\n";
    std::cout << "  MC Paths:      " << nPaths << "\n";
    
    // Generate random numbers once
    auto Z = generateNormals(nPaths, seed);
    
    // Analytical Solution
    std::cout << "\n" << std::string(80, '-') << "\n";
    std::cout << "Analytical Black-Scholes (Benchmark)\n";
    std::cout << std::string(80, '-') << "\n";
    
    auto bs = blackScholesAnalytical(opt);
    printResults("Analytical", bs);
    
    // Finite Differences
    std::cout << "\n" << std::string(80, '-') << "\n";
    std::cout << "Finite Differences Method\n";
    std::cout << std::string(80, '-') << "\n";
    
    Timer timer;
    timer.start();
    auto fdResults = computeGreeksFD(opt, Z);
    double fdTime = timer.elapsed();
    
    printResults("FD", fdResults, fdTime);
    std::cout << "  MC runs: 9 simulations x " << nPaths << " paths each\n";
    
    // XAD AAD
    std::cout << "\n" << std::string(80, '-') << "\n";
    std::cout << "XAD Adjoint Algorithmic Differentiation (Per-Path Tape)\n";
    std::cout << std::string(80, '-') << "\n";
    
    timer.start();
    auto xadResults = computeGreeksXAD(opt, Z);
    double xadTime = timer.elapsed();
    
    printResults("XAD", xadResults, xadTime);
    std::cout << "  MC runs: 3 simulations x " << nPaths << " paths (1 base + 2 for gamma)\n";
    
    // Comparison Table
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "COMPARISON SUMMARY\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "+-------------+----------------+----------------+----------------+-------------+\n";
    std::cout << "|   Greek     |   Analytical   |      FD        |      XAD       |  XAD Error  |\n";
    std::cout << "+-------------+----------------+----------------+----------------+-------------+\n";
    
    auto printRow = [&](const char* name, double analytical, double fd, double xad) {
        double error = std::abs(analytical) > 1e-10 ? std::abs((xad - analytical) / analytical) * 100 : 0.0;
        std::cout << "| " << std::setw(11) << std::left << name << " | "
                  << std::setw(14) << std::right << analytical << " | "
                  << std::setw(14) << std::right << fd << " | "
                  << std::setw(14) << std::right << xad << " | "
                  << std::setw(9) << std::setprecision(2) << error << "% |\n";
        std::cout << std::setprecision(6);
    };
    
    printRow("Price", bs.price, fdResults.price, xadResults.price);
    printRow("Delta", bs.delta, fdResults.delta, xadResults.delta);
    printRow("Gamma", bs.gamma, fdResults.gamma, xadResults.gamma);
    printRow("Vega", bs.vega, fdResults.vega, xadResults.vega);
    printRow("Theta", bs.theta, fdResults.theta, xadResults.theta);
    printRow("Rho", bs.rho, fdResults.rho, xadResults.rho);
    
    std::cout << "+-------------+----------------+----------------+----------------+-------------+\n";
    
    // Timing Analysis
    std::cout << "\n" << std::string(80, '-') << "\n";
    std::cout << "TIMING ANALYSIS\n";
    std::cout << std::string(80, '-') << "\n\n";
    
    std::cout << std::setprecision(4);
    std::cout << "  Finite Differences:  " << fdTime << " seconds\n";
    std::cout << "  XAD AAD:             " << xadTime << " seconds\n\n";
    
    if (fdTime > xadTime) {
        double speedup = fdTime / xadTime;
        std::cout << "  XAD is " << std::setprecision(2) << speedup << "x FASTER than Finite Differences\n";
    } else {
        double ratio = xadTime / fdTime;
        std::cout << "  FD is " << std::setprecision(1) << ratio << "x faster in this simple case\n";
    }
    
    std::cout << "\n  Computation breakdown:\n";
    std::cout << "    FD:  9 simulations x " << nPaths << " paths = " << 9 * nPaths << " path evaluations\n";
    std::cout << "    XAD: 3 simulations x " << nPaths << " paths = " << 3 * nPaths << " path evaluations\n";
    std::cout << "         (1 base gives Delta,Vega,Theta,Rho + 2 for Gamma)\n";
    
    std::cout << "\n  WHY FD IS FASTER HERE:\n";
    std::cout << "    - Simple model with only 4 inputs (S0, T, r, sigma)\n";
    std::cout << "    - FD uses vectorized loops (no tape overhead)\n";
    std::cout << "    - AAD tape records every operation (memory + compute overhead)\n";
    
    std::cout << "\n  WHEN AAD EXCELS:\n";
    std::cout << "    - 100+ inputs (e.g., yield curve nodes) -> O(1) vs O(n) for FD\n";
    std::cout << "    - See Hull-White swaption example: 10+ curve sensitivities in one pass\n";
    std::cout << "    - Complex path-dependent payoffs where FD bumps are expensive\n";
    std::cout << "    - Exact derivatives (no bump size tuning)\n";
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Analysis Complete\n";
    std::cout << std::string(80, '=') << "\n";
    
    return 0;
}
