/*******************************************************************************
 * Hull-White Swaption Pricer: XAD Greeks vs Finite Differences
 * 
 * This program computes sensitivities (Greeks) of a European swaption price
 * with respect to discount curve nodes using:
 *   1. XAD Adjoint Algorithmic Differentiation (AAD)
 *   2. Finite Differences (bump-and-reprice)
 * 
 * It compares the results and timing of both methods.
 ******************************************************************************/

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>

#include <XAD/XAD.hpp>
#include "HullWhite.hpp"

// =============================================================================
// XAD Greeks Computation
// =============================================================================

std::pair<double, std::vector<double>> computeGreeksXAD(
    const MarketCurve& curve,
    const std::vector<std::vector<double>>& Z,
    const SwaptionParams& swaption,
    const HullWhiteParams& hw,
    const MonteCarloParams& mc
) {
    using mode = xad::adj<double>;
    using tape_type = mode::tape_type;
    using AD = mode::active_type;
    
    size_t n_rates = curve.size();
    int num_steps = static_cast<int>(swaption.T_option / mc.dt);
    
    // Time grid
    std::vector<double> t_grid(num_steps + 1);
    for (int i = 0; i <= num_steps; ++i) {
        t_grid[i] = i * mc.dt;
    }
    
    // Accumulated Greeks and price
    std::vector<double> acc_greeks(n_rates, 0.0);
    double acc_price = 0.0;
    
    // Process each path
    for (int path = 0; path < mc.num_paths; ++path) {
        // Create tape
        tape_type tape;
        
        // Create AD variables for rates
        std::vector<AD> rates_ad(curve.rates.begin(), curve.rates.end());
        
        // Register inputs
        tape.registerInputs(rates_ad);
        
        // Start recording
        tape.newRecording();
        
        // Pre-compute theta values on tape
        std::vector<AD> theta_cache(num_steps + 1);
        for (int i = 0; i <= num_steps; ++i) {
            theta_cache[i] = theta<AD>(t_grid[i], curve.maturities, rates_ad, hw.a, hw.sigma);
        }
        
        // Simulate path and compute payoff
        AD payoff = simulateAndPricePath<AD>(
            curve.maturities, rates_ad, Z[path], theta_cache, swaption, hw, mc.dt
        );
        
        // Register output and seed adjoint
        tape.registerOutput(payoff);
        derivative(payoff) = 1.0;
        
        // Compute adjoints
        tape.computeAdjoints();
        
        // Accumulate
        acc_price += value(payoff);
        for (size_t i = 0; i < n_rates; ++i) {
            acc_greeks[i] += derivative(rates_ad[i]);
        }
    }
    
    // Average
    double avg_price = acc_price / mc.num_paths;
    for (auto& g : acc_greeks) {
        g /= mc.num_paths;
    }
    
    return {avg_price, acc_greeks};
}

// =============================================================================
// Finite Difference Greeks Computation
// =============================================================================

std::pair<double, std::vector<double>> computeGreeksFD(
    const MarketCurve& curve,
    const std::vector<std::vector<double>>& Z,
    const SwaptionParams& swaption,
    const HullWhiteParams& hw,
    const MonteCarloParams& mc,
    double bump = 1e-4
) {
    size_t n_rates = curve.size();
    
    // Base price
    double base_price = priceSwaption(curve.maturities, curve.rates, Z, swaption, hw, mc);
    
    // Greeks via central differences
    std::vector<double> greeks(n_rates);
    for (size_t i = 0; i < n_rates; ++i) {
        // Bump up
        std::vector<double> rates_up = curve.rates;
        rates_up[i] += bump;
        double price_up = priceSwaption(curve.maturities, rates_up, Z, swaption, hw, mc);
        
        // Bump down
        std::vector<double> rates_down = curve.rates;
        rates_down[i] -= bump;
        double price_down = priceSwaption(curve.maturities, rates_down, Z, swaption, hw, mc);
        
        // Central difference
        greeks[i] = (price_up - price_down) / (2.0 * bump);
    }
    
    return {base_price, greeks};
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Hull-White Swaption Greeks: XAD (AAD) vs Finite Differences\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // Parameters
    SwaptionParams swaption;
    HullWhiteParams hw;
    MonteCarloParams mc;
    MarketCurve curve;
    
    // Print market curve
    std::cout << "Market Curve (" << curve.size() << " nodes):\n";
    for (size_t i = 0; i < curve.size(); ++i) {
        std::cout << "  Node " << std::setw(2) << i 
                  << ": T=" << std::fixed << std::setprecision(3) << std::setw(6) << curve.maturities[i]
                  << "Y, Rate=" << std::setprecision(3) << curve.rates[i] * 100 << "%\n";
    }
    
    std::cout << "\nSwaption: " << swaption.T_option << "Y expiry x " << swaption.swap_tenor 
              << "Y swap, Strike=" << swaption.K_strike * 100 << "%\n";
    std::cout << "Notional: $" << std::fixed << std::setprecision(0) << swaption.notional
              << ", Type: " << (swaption.is_payer ? "Payer" : "Receiver") << "\n";
    std::cout << std::setprecision(2) << "Hull-White: a=" << hw.a << ", sigma=" << hw.sigma << "\n";
    
    int num_steps = static_cast<int>(swaption.T_option / mc.dt);
    std::cout << "Monte Carlo: " << mc.num_paths << " paths, " << num_steps << " steps (dt=" 
              << std::setprecision(4) << mc.dt << ")\n";
    
    // Generate random numbers
    auto Z = generateRandomMatrix(mc.num_paths, num_steps, mc.seed);
    
    // XAD Greeks
    std::cout << "\n" << std::string(70, '-') << "\n";
    std::cout << "Computing Greeks with XAD (AAD)...\n";
    
    auto start_xad = std::chrono::high_resolution_clock::now();
    auto [price_xad, greeks_xad] = computeGreeksXAD(curve, Z, swaption, hw, mc);
    auto end_xad = std::chrono::high_resolution_clock::now();
    double time_xad = std::chrono::duration<double>(end_xad - start_xad).count();
    
    std::cout << "XAD completed in " << std::fixed << std::setprecision(3) << time_xad << "s\n";
    
    // FD Greeks
    std::cout << "\n" << std::string(70, '-') << "\n";
    std::cout << "Computing Greeks with Finite Differences...\n";
    
    auto start_fd = std::chrono::high_resolution_clock::now();
    auto [price_fd, greeks_fd] = computeGreeksFD(curve, Z, swaption, hw, mc);
    auto end_fd = std::chrono::high_resolution_clock::now();
    double time_fd = std::chrono::duration<double>(end_fd - start_fd).count();
    
    std::cout << "FD completed in " << std::fixed << std::setprecision(3) << time_fd << "s\n";
    
    // Price comparison
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "PRICE COMPARISON\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  XAD Price:    $" << price_xad << "\n";
    std::cout << "  FD Price:     $" << price_fd << "\n";
    std::cout << "  Difference:   $" << std::abs(price_xad - price_fd) 
              << " (" << std::setprecision(6) << std::abs(price_xad - price_fd) / price_fd * 100 << "%)\n";
    
    // Greeks comparison
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "GREEKS COMPARISON (dPrice/dRate)\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << std::left << std::setw(5) << "Node" 
              << std::right << std::setw(8) << "Mat" 
              << std::setw(16) << "XAD" 
              << std::setw(16) << "FD" 
              << std::setw(14) << "Diff" 
              << std::setw(12) << "Rel%\n";
    std::cout << std::string(70, '-') << "\n";
    
    double max_rel = 0.0;
    for (size_t i = 0; i < curve.size(); ++i) {
        double g_xad = greeks_xad[i];
        double g_fd = greeks_fd[i];
        double diff = std::abs(g_xad - g_fd);
        double rel = diff / std::max(std::abs(g_fd), 1e-10) * 100;
        if (std::abs(g_fd) > 1) max_rel = std::max(max_rel, rel);
        
        std::cout << std::left << std::setw(5) << i 
                  << std::right << std::fixed << std::setprecision(3) << std::setw(8) << curve.maturities[i]
                  << std::setprecision(2) << std::setw(16) << g_xad
                  << std::setw(16) << g_fd
                  << std::setw(14) << diff
                  << std::setprecision(4) << std::setw(11) << rel << "%\n";
    }
    
    std::cout << std::string(70, '-') << "\n";
    std::cout << "Max Relative Difference (significant Greeks): " << std::setprecision(4) << max_rel << "%\n";
    
    // Timing analysis
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "TIMING ANALYSIS\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  XAD Time:     " << time_xad << "s (" << mc.num_paths << " tape recordings)\n";
    std::cout << "  FD Time:      " << time_fd << "s (" << 2 * curve.size() + 1 << " pricings)\n";
    
    double speedup = time_fd / time_xad;
    if (speedup > 1.0) {
        std::cout << "  Speedup:      XAD is " << std::setprecision(2) << speedup << "x FASTER than FD\n";
    } else {
        std::cout << "  Ratio:        XAD is " << std::setprecision(2) << 1.0/speedup << "x slower than FD\n";
    }
    
    std::cout << "\n  Note: AAD computes all " << curve.size() << " Greeks in one backward pass per path.\n";
    std::cout << "        FD requires " << 2 * curve.size() << " additional pricings (2 per Greek).\n";
    
    return 0;
}
