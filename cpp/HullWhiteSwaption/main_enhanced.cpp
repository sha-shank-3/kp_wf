/*******************************************************************************
 * Hull-White Swaption Pricer - Enhanced Version
 * 
 * Features:
 *   1. Real OIS curve data (SOFR-based, December 2024)
 *   2. Proper theta(t) calibration to match initial term structure
 *   3. ATM swaption volatility surface for HW calibration
 *   4. Greeks w.r.t. both curve nodes AND vol surface nodes
 *   5. XAD AAD vs Finite Differences comparison
 ******************************************************************************/

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <tuple>

#include <XAD/XAD.hpp>
#include "HullWhiteEnhanced.hpp"

// Forward declarations
struct HWParamGreeks {
    double price;
    double d_price_d_a;
    double d_price_d_sigma;
};

HWParamGreeks computeHWParamGreeksXAD(
    const RealOISCurve& curve,
    const std::vector<std::vector<double>>& Z,
    const SwaptionParams& swaption,
    const HullWhiteParams& hw,
    const MonteCarloParams& mc
);

// =============================================================================
// Greeks Computation - Rate Curve Sensitivities (Delta)
// =============================================================================

std::pair<double, std::vector<double>> computeRateGreeksXAD(
    const RealOISCurve& curve,
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
    
    std::vector<double> t_grid(num_steps + 1);
    for (int i = 0; i <= num_steps; ++i) {
        t_grid[i] = i * mc.dt;
    }
    
    std::vector<double> acc_greeks(n_rates, 0.0);
    double acc_price = 0.0;
    
    for (int path = 0; path < mc.num_paths; ++path) {
        tape_type tape;
        tape.registerInputs(std::vector<AD>());
        
        std::vector<AD> rates_ad(n_rates);
        for (size_t i = 0; i < n_rates; ++i) {
            rates_ad[i] = curve.rates[i];
        }
        
        tape.registerInputs(rates_ad);
        tape.newRecording();
        
        // Pre-compute theta on tape
        std::vector<AD> theta_cache(num_steps + 1);
        for (int i = 0; i <= num_steps; ++i) {
            theta_cache[i] = theta<AD>(t_grid[i], curve.maturities, rates_ad, hw.a, hw.sigma);
        }
        
        // Price the path
        AD payoff = simulateAndPricePath<AD>(
            t_grid, theta_cache, curve.maturities, rates_ad,
            Z[path], swaption, hw.a, hw.sigma, mc.dt
        );
        
        tape.registerOutput(payoff);
        derivative(payoff) = 1.0;
        tape.computeAdjoints();
        
        acc_price += value(payoff);
        for (size_t i = 0; i < n_rates; ++i) {
            acc_greeks[i] += derivative(rates_ad[i]);
        }
    }
    
    double price = acc_price / mc.num_paths;
    std::vector<double> greeks(n_rates);
    for (size_t i = 0; i < n_rates; ++i) {
        greeks[i] = acc_greeks[i] / mc.num_paths;
    }
    
    return {price, greeks};
}

// =============================================================================
// Vol Surface Greeks using Implicit Function Theorem (XAD + Chain Rule)
// 
// The swaption price V depends on HW params (a, σ) which are calibrated to 
// the market vol surface. Using chain rule:
//
//   dV/d(vol_ij) = (∂V/∂a)(∂a/∂vol_ij) + (∂V/∂σ)(∂σ/∂vol_ij)
//
// We compute:
//   1. ∂V/∂a and ∂V/∂σ using XAD AAD (ONE pricing run)
//   2. ∂a/∂vol_ij and ∂σ/∂vol_ij using FD on calibration (N_vol calibrations)
//
// Total cost: 1 AAD pricing + N_vol calibrations
// =============================================================================

struct VolSurfaceGreeks {
    double price;
    std::vector<std::vector<double>> vegas;           // dV/d(vol_ij) per 1bp
    std::vector<std::vector<double>> da_dvol;         // ∂a/∂vol_ij
    std::vector<std::vector<double>> dsigma_dvol;     // ∂σ/∂vol_ij
    double dV_da;                                      // ∂V/∂a from XAD
    double dV_dsigma;                                  // ∂V/∂σ from XAD
};

VolSurfaceGreeks computeVolSurfaceGreeksXAD(
    const RealOISCurve& curve,
    const ATMVolSurface& volSurface,
    const std::vector<std::vector<double>>& Z,
    const SwaptionParams& swaption,
    const HullWhiteParams& baseHW,
    const MonteCarloParams& mc,
    const std::vector<std::pair<size_t, size_t>>& calibInstruments,
    double bump_bps = 1.0
) {
    size_t n_exp = volSurface.expiries.size();
    size_t n_ten = volSurface.tenors.size();
    
    // Step 1: Compute ∂V/∂a and ∂V/∂σ using XAD (ONE pricing)
    auto hwGreeks = computeHWParamGreeksXAD(curve, Z, swaption, baseHW, mc);
    double dV_da = hwGreeks.d_price_d_a;
    double dV_dsigma = hwGreeks.d_price_d_sigma;
    
    // Step 2: Compute calibration Jacobian ∂(a,σ)/∂vol_ij using finite differences
    std::vector<std::vector<double>> da_dvol(n_exp, std::vector<double>(n_ten, 0.0));
    std::vector<std::vector<double>> dsigma_dvol(n_exp, std::vector<double>(n_ten, 0.0));
    std::vector<std::vector<double>> vegas(n_exp, std::vector<double>(n_ten, 0.0));
    
    double base_a = baseHW.a;
    double base_sigma = baseHW.sigma;
    
    for (size_t ei = 0; ei < n_exp; ++ei) {
        for (size_t ti = 0; ti < n_ten; ++ti) {
            // Bump this vol node
            ATMVolSurface bumpedSurface = volSurface;
            bumpedSurface.vols[ei][ti] += bump_bps;  // vols are in bps
            
            // Recalibrate with bumped vol
            CalibrationResult bumpedCalib = calibrateHullWhite(bumpedSurface, curve, calibInstruments);
            
            // Compute sensitivities of calibrated params to this vol node
            da_dvol[ei][ti] = (bumpedCalib.a - base_a) / bump_bps;
            dsigma_dvol[ei][ti] = (bumpedCalib.sigma - base_sigma) / bump_bps;
            
            // Apply chain rule
            vegas[ei][ti] = dV_da * da_dvol[ei][ti] + dV_dsigma * dsigma_dvol[ei][ti];
        }
    }
    
    return {hwGreeks.price, vegas, da_dvol, dsigma_dvol, dV_da, dV_dsigma};
}

// =============================================================================
// Vol Surface Greeks using Pure Finite Differences
// 
// For each vol node, bump vol -> recalibrate -> reprice
// 
// Total cost: N_vol * (1 calibration + 1 pricing)
// =============================================================================

VolSurfaceGreeks computeVolSurfaceGreeksFD(
    const RealOISCurve& curve,
    const ATMVolSurface& volSurface,
    const std::vector<std::vector<double>>& Z,
    const SwaptionParams& swaption,
    const HullWhiteParams& baseHW,
    const MonteCarloParams& mc,
    const std::vector<std::pair<size_t, size_t>>& calibInstruments,
    double base_price,
    double bump_bps = 1.0
) {
    size_t n_exp = volSurface.expiries.size();
    size_t n_ten = volSurface.tenors.size();
    
    std::vector<std::vector<double>> vegas(n_exp, std::vector<double>(n_ten, 0.0));
    std::vector<std::vector<double>> da_dvol(n_exp, std::vector<double>(n_ten, 0.0));
    std::vector<std::vector<double>> dsigma_dvol(n_exp, std::vector<double>(n_ten, 0.0));
    
    double base_a = baseHW.a;
    double base_sigma = baseHW.sigma;
    
    for (size_t ei = 0; ei < n_exp; ++ei) {
        for (size_t ti = 0; ti < n_ten; ++ti) {
            // Bump this vol node
            ATMVolSurface bumpedSurface = volSurface;
            bumpedSurface.vols[ei][ti] += bump_bps;
            
            // Recalibrate with bumped vol
            CalibrationResult bumpedCalib = calibrateHullWhite(bumpedSurface, curve, calibInstruments);
            HullWhiteParams bumpedHW = {bumpedCalib.a, bumpedCalib.sigma};
            
            // Reprice with new HW params
            auto [bumped_price, _] = computeRateGreeksXAD(curve, Z, swaption, bumpedHW, mc);
            
            // Store results
            da_dvol[ei][ti] = (bumpedCalib.a - base_a) / bump_bps;
            dsigma_dvol[ei][ti] = (bumpedCalib.sigma - base_sigma) / bump_bps;
            vegas[ei][ti] = (bumped_price - base_price) / bump_bps;
        }
    }
    
    return {base_price, vegas, da_dvol, dsigma_dvol, 0.0, 0.0};
}

// =============================================================================
// Greeks Computation - HW Parameter Sensitivities (Implementation)
// Direct sensitivities to a and sigma using AAD
// =============================================================================

HWParamGreeks computeHWParamGreeksXAD(
    const RealOISCurve& curve,
    const std::vector<std::vector<double>>& Z,
    const SwaptionParams& swaption,
    const HullWhiteParams& hw,
    const MonteCarloParams& mc
) {
    using mode = xad::adj<double>;
    using tape_type = mode::tape_type;
    using AD = mode::active_type;
    
    int num_steps = static_cast<int>(swaption.T_option / mc.dt);
    
    std::vector<double> t_grid(num_steps + 1);
    for (int i = 0; i <= num_steps; ++i) {
        t_grid[i] = i * mc.dt;
    }
    
    double acc_price = 0.0;
    double acc_d_a = 0.0;
    double acc_d_sigma = 0.0;
    
    for (int path = 0; path < mc.num_paths; ++path) {
        tape_type tape;
        
        // Make a and sigma AD types
        AD a_ad = hw.a;
        AD sigma_ad = hw.sigma;
        
        // Also rates (we need them for the computation but won't track their derivatives)
        std::vector<AD> rates_ad(curve.size());
        for (size_t i = 0; i < curve.size(); ++i) {
            rates_ad[i] = curve.rates[i];
        }
        
        tape.registerInput(a_ad);
        tape.registerInput(sigma_ad);
        tape.newRecording();
        
        // Simulate short rate with AD a and sigma
        double sqrt_dt = std::sqrt(mc.dt);
        
        // Initial short rate (doesn't depend on a, sigma)
        AD r_t = forwardRate<AD>(0.0, curve.maturities, rates_ad);
        
        // Pre-compute theta values with AD a, sigma
        for (int i = 0; i < num_steps; ++i) {
            double t = t_grid[i];
            
            // theta(t) = f'(t) + a*f(t) + sigma^2/(2a) * (1 - exp(-2at))
            AD f_t = forwardRate<AD>(t, curve.maturities, rates_ad);
            AD f_th = forwardRate<AD>(t + 1e-5, curve.maturities, rates_ad);
            AD f_prime = (f_th - f_t) / 1e-5;
            
            AD term3 = (sigma_ad * sigma_ad / (AD(2.0) * a_ad)) * 
                       (AD(1.0) - exp(AD(-2.0) * a_ad * t));
            AD theta_t = f_prime + a_ad * f_t + term3;
            
            r_t = r_t + (theta_t - a_ad * r_t) * mc.dt + sigma_ad * sqrt_dt * Z[path][i];
        }
        
        AD r_T = r_t;
        
        // Bond prices at option expiry
        AD f_T = forwardRate<AD>(swaption.T_option, curve.maturities, rates_ad);
        
        std::vector<AD> bonds;
        for (int j = 1; j <= swaption.swap_tenor; ++j) {
            double T_pay = swaption.T_option + j;
            double tau = T_pay - swaption.T_option;
            
            AD B = (AD(1.0) - exp(-a_ad * tau)) / a_ad;
            AD P_T_pay = discount<AD>(T_pay, curve.maturities, rates_ad);
            AD P_t = discount<AD>(swaption.T_option, curve.maturities, rates_ad);
            
            AD sigma2_4a = sigma_ad * sigma_ad / (AD(4.0) * a_ad);
            AD exp_neg2at = exp(AD(-2.0) * a_ad * swaption.T_option);
            
            AD log_A = log(P_T_pay) - log(P_t) + B * f_T -
                       sigma2_4a * (AD(1.0) - exp_neg2at) * B * B;
            
            bonds.push_back(exp(log_A) * exp(-B * r_T));
        }
        
        // Fixed leg
        AD fixed_pv = AD(0.0);
        for (const auto& b : bonds) {
            fixed_pv = fixed_pv + b * swaption.K_strike;
        }
        
        // Floating leg
        AD float_pv = AD(1.0) - bonds.back();
        
        // Swap value
        AD swap_val = (float_pv - fixed_pv) * swaption.notional;
        
        // Payoff
        AD payoff;
        if (swaption.is_payer) {
            payoff = max(swap_val, AD(0.0));
        } else {
            AD neg_swap = AD(0.0) - swap_val;
            payoff = max(neg_swap, AD(0.0));
        }
        
        // Discount
        AD disc = discount<AD>(swaption.T_option, curve.maturities, rates_ad);
        AD pv = disc * payoff;
        
        tape.registerOutput(pv);
        derivative(pv) = 1.0;
        tape.computeAdjoints();
        
        acc_price += value(pv);
        acc_d_a += derivative(a_ad);
        acc_d_sigma += derivative(sigma_ad);
    }
    
    return {
        acc_price / mc.num_paths,
        acc_d_a / mc.num_paths,
        acc_d_sigma / mc.num_paths
    };
}

// =============================================================================
// Finite Difference for Rate Greeks (for comparison)
// =============================================================================

std::pair<double, std::vector<double>> computeRateGreeksFD(
    const RealOISCurve& curve,
    const std::vector<std::vector<double>>& Z,
    const SwaptionParams& swaption,
    const HullWhiteParams& hw,
    const MonteCarloParams& mc,
    double bump = 0.0001
) {
    size_t n_rates = curve.size();
    int num_steps = static_cast<int>(swaption.T_option / mc.dt);
    double sqrt_dt = std::sqrt(mc.dt);
    
    std::vector<double> t_grid(num_steps + 1);
    for (int i = 0; i <= num_steps; ++i) {
        t_grid[i] = i * mc.dt;
    }
    
    // Pricing function
    auto priceWithRates = [&](const std::vector<double>& rates) -> double {
        double total = 0.0;
        
        // Pre-compute theta
        std::vector<double> theta_cache(num_steps + 1);
        for (int i = 0; i <= num_steps; ++i) {
            theta_cache[i] = theta<double>(t_grid[i], curve.maturities, rates, hw.a, hw.sigma);
        }
        
        for (int path = 0; path < mc.num_paths; ++path) {
            double r_t = forwardRate<double>(0.0, curve.maturities, rates);
            
            for (int i = 0; i < num_steps; ++i) {
                r_t = r_t + (theta_cache[i] - hw.a * r_t) * mc.dt + hw.sigma * sqrt_dt * Z[path][i];
            }
            
            double r_T = r_t;
            double f_T = forwardRate<double>(swaption.T_option, curve.maturities, rates);
            
            std::vector<double> bonds;
            for (int j = 1; j <= swaption.swap_tenor; ++j) {
                double T_pay = swaption.T_option + j;
                bonds.push_back(bondPrice<double>(r_T, swaption.T_option, T_pay, 
                                                  curve.maturities, rates, hw.a, hw.sigma, f_T));
            }
            
            double fixed_pv = 0.0;
            for (const auto& b : bonds) {
                fixed_pv += b * swaption.K_strike;
            }
            
            double float_pv = 1.0 - bonds.back();
            double swap_val = (float_pv - fixed_pv) * swaption.notional;
            double payoff = swaption.is_payer ? std::max(swap_val, 0.0) : std::max(-swap_val, 0.0);
            
            double disc = discount<double>(swaption.T_option, curve.maturities, rates);
            total += disc * payoff;
        }
        
        return total / mc.num_paths;
    };
    
    // Base price
    double base_price = priceWithRates(curve.rates);
    
    // Greeks via central differences
    std::vector<double> greeks(n_rates);
    for (size_t i = 0; i < n_rates; ++i) {
        std::vector<double> rates_up = curve.rates;
        std::vector<double> rates_down = curve.rates;
        rates_up[i] += bump;
        rates_down[i] -= bump;
        
        double price_up = priceWithRates(rates_up);
        double price_down = priceWithRates(rates_down);
        
        greeks[i] = (price_up - price_down) / (2.0 * bump);
    }
    
    return {base_price, greeks};
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << std::string(80, '=') << "\n";
    std::cout << "Hull-White Swaption Pricer - Enhanced Version\n";
    std::cout << "Real OIS Data | Theta Calibration | Vol Surface Greeks\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    // Load real OIS curve
    RealOISCurve curve;
    std::cout << "Real OIS Curve (SOFR-based, " << curve.curve_date << "):\n";
    std::cout << std::string(50, '-') << "\n";
    for (size_t i = 0; i < curve.size(); i += 2) {
        std::cout << "  " << std::fixed << std::setprecision(4) << std::setw(8) << curve.maturities[i]
                  << "Y: " << std::setprecision(3) << curve.rates[i] * 100 << "%\n";
    }
    
    // Load ATM vol surface
    ATMVolSurface volSurface;
    std::cout << "\nATM Swaption Vol Surface (Normal vols in bps):\n";
    std::cout << std::string(50, '-') << "\n";
    std::cout << "        ";
    for (double t : volSurface.tenors) {
        std::cout << std::setw(6) << t << "Y";
    }
    std::cout << "\n";
    for (size_t i = 0; i < volSurface.expiries.size(); i += 2) {
        std::cout << std::setw(6) << volSurface.expiries[i] << "Y ";
        for (size_t j = 0; j < volSurface.tenors.size(); ++j) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(0) << volSurface.vols[i][j];
        }
        std::cout << "\n";
    }
    
    // Calibrate Hull-White to vol surface
    std::cout << "\n" << std::string(80, '-') << "\n";
    std::cout << "Calibrating Hull-White to ATM Vol Surface...\n";
    
    // Use 1Y-5Y section for calibration (relevant to our swaption)
    std::vector<std::pair<size_t, size_t>> calibInstruments;
    for (size_t ei = 2; ei < 6; ++ei) {
        for (size_t ti = 0; ti < 5; ++ti) {
            calibInstruments.push_back({ei, ti});
        }
    }
    
    auto calibResult = calibrateHullWhite(volSurface, curve, calibInstruments);
    
    std::cout << "  Calibrated a     = " << std::setprecision(4) << calibResult.a << "\n";
    std::cout << "  Calibrated sigma = " << std::setprecision(6) << calibResult.sigma << "\n";
    std::cout << "  Calibration RMSE = " << std::setprecision(4) << calibResult.rmse * 10000 << " bps\n";
    
    // Setup
    SwaptionParams swaption;
    swaption.K_strike = forwardSwapRate(swaption.T_option, swaption.swap_tenor, curve);  // ATM strike
    std::cout << "\nSwaption: " << swaption.T_option << "Y x " << swaption.swap_tenor 
              << "Y, ATM Strike = " << std::setprecision(3) << swaption.K_strike * 100 << "%\n";
    std::cout << "Notional: $" << std::fixed << std::setprecision(0) << swaption.notional
              << ", Type: " << (swaption.is_payer ? "Payer" : "Receiver") << "\n";
    
    HullWhiteParams hw = {calibResult.a, calibResult.sigma};
    MonteCarloParams mc;
    mc.num_paths = 3000;  // Reduced for faster execution with vol surface calcs
    
    int num_steps = static_cast<int>(swaption.T_option / mc.dt);
    std::cout << "Hull-White: a=" << std::setprecision(4) << hw.a 
              << ", sigma=" << std::setprecision(6) << hw.sigma << "\n";
    std::cout << "Monte Carlo: " << mc.num_paths << " paths, " << num_steps << " steps\n";
    
    // Generate random numbers
    auto Z = generateRandomMatrix(mc.num_paths, num_steps, mc.seed);
    
    // ==========================================================================
    // PART 1: Rate Curve Greeks (Delta)
    // ==========================================================================
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "RATE CURVE GREEKS (Delta)\n";
    std::cout << std::string(80, '=') << "\n";
    
    // XAD
    std::cout << "\nComputing with XAD (AAD)...\n";
    auto start_xad = std::chrono::high_resolution_clock::now();
    auto [price_xad, rate_greeks_xad] = computeRateGreeksXAD(curve, Z, swaption, hw, mc);
    auto end_xad = std::chrono::high_resolution_clock::now();
    double time_xad = std::chrono::duration<double>(end_xad - start_xad).count();
    std::cout << "  XAD completed in " << std::setprecision(3) << time_xad << "s\n";
    
    // FD
    std::cout << "Computing with Finite Differences...\n";
    auto start_fd = std::chrono::high_resolution_clock::now();
    auto [price_fd, rate_greeks_fd] = computeRateGreeksFD(curve, Z, swaption, hw, mc);
    auto end_fd = std::chrono::high_resolution_clock::now();
    double time_fd = std::chrono::duration<double>(end_fd - start_fd).count();
    std::cout << "  FD completed in " << std::setprecision(3) << time_fd << "s\n";
    
    // Price comparison
    std::cout << "\nPrice Comparison:\n";
    std::cout << "  XAD Price: $" << std::setprecision(2) << price_xad << "\n";
    std::cout << "  FD Price:  $" << price_fd << "\n";
    
    // Rate Greeks comparison
    std::cout << "\nRate Greeks (dP/dRate, per 1% = 0.01):\n";
    std::cout << std::left << std::setw(10) << "Maturity" 
              << std::right << std::setw(16) << "XAD" 
              << std::setw(16) << "FD" 
              << std::setw(12) << "Rel Diff%\n";
    std::cout << std::string(55, '-') << "\n";
    
    double max_rel = 0.0;
    for (size_t i = 0; i < std::min(size_t(10), curve.size()); ++i) {
        double g_xad = rate_greeks_xad[i];
        double g_fd = rate_greeks_fd[i];
        double rel = std::abs(g_fd) > 1 ? std::abs(g_xad - g_fd) / std::abs(g_fd) * 100 : 0;
        if (std::abs(g_fd) > 100) max_rel = std::max(max_rel, rel);
        
        std::cout << std::left << std::fixed << std::setprecision(3) << std::setw(10) << curve.maturities[i]
                  << std::right << std::setprecision(2) << std::setw(16) << g_xad
                  << std::setw(16) << g_fd
                  << std::setprecision(4) << std::setw(11) << rel << "%\n";
    }
    std::cout << "Max Relative Difference (significant Greeks): " << std::setprecision(4) << max_rel << "%\n";
    
    // ==========================================================================
    // PART 2: Hull-White Parameter Greeks (Vega to a and sigma)
    // ==========================================================================
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "HULL-WHITE PARAMETER GREEKS\n";
    std::cout << std::string(80, '=') << "\n";
    
    std::cout << "\nComputing sensitivities to HW parameters (a, sigma) with XAD...\n";
    auto start_hw = std::chrono::high_resolution_clock::now();
    auto hwGreeks = computeHWParamGreeksXAD(curve, Z, swaption, hw, mc);
    auto end_hw = std::chrono::high_resolution_clock::now();
    double time_hw = std::chrono::duration<double>(end_hw - start_hw).count();
    
    std::cout << "  Completed in " << std::setprecision(3) << time_hw << "s\n";
    std::cout << "\n  Price:        $" << std::setprecision(2) << hwGreeks.price << "\n";
    std::cout << "  dP/da:        $" << std::setprecision(2) << hwGreeks.d_price_d_a << " per unit a\n";
    std::cout << "  dP/dsigma:    $" << std::setprecision(2) << hwGreeks.d_price_d_sigma << " per unit sigma\n";
    std::cout << "  dP/d(1% a):   $" << std::setprecision(2) << hwGreeks.d_price_d_a * 0.01 << "\n";
    std::cout << "  dP/d(1bp σ):  $" << std::setprecision(2) << hwGreeks.d_price_d_sigma * 0.0001 << "\n";
    
    // ==========================================================================
    // PART 3: Vol Surface Greeks using Implicit Function Theorem
    // ==========================================================================
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "VOL SURFACE GREEKS (using Implicit Function Theorem)\n";
    std::cout << std::string(80, '=') << "\n";
    
    std::cout << "\nComputing dV/d(vol_ij) via chain rule:\n";
    std::cout << "  dV/d(vol_ij) = (dV/da)(da/dvol_ij) + (dV/dsigma)(dsigma/dvol_ij)\n\n";
    
    // Show which vol nodes are in calibration set
    std::cout << "  Calibration uses vol nodes (marked with *):\n";
    std::cout << "  Expiries: ";
    for (size_t ei = 0; ei < volSurface.expiries.size(); ++ei) {
        bool in_calib = false;
        for (const auto& [ce, ct] : calibInstruments) {
            if (ce == ei) { in_calib = true; break; }
        }
        std::cout << volSurface.expiries[ei] << "Y" << (in_calib ? "*" : "") << " ";
    }
    std::cout << "\n  Tenors:   ";
    for (size_t ti = 0; ti < volSurface.tenors.size(); ++ti) {
        bool in_calib = false;
        for (const auto& [ce, ct] : calibInstruments) {
            if (ct == ti) { in_calib = true; break; }
        }
        std::cout << volSurface.tenors[ti] << "Y" << (in_calib ? "*" : "") << " ";
    }
    std::cout << "\n\n";
    
    // Method 1: XAD + Chain Rule (Implicit Function Theorem)
    std::cout << "  Method 1: XAD + Chain Rule...\n";
    auto start_vol_xad = std::chrono::high_resolution_clock::now();
    auto volGreeksXAD = computeVolSurfaceGreeksXAD(curve, volSurface, Z, swaption, hw, mc, calibInstruments, 1.0);
    auto end_vol_xad = std::chrono::high_resolution_clock::now();
    double time_vol_xad = std::chrono::duration<double>(end_vol_xad - start_vol_xad).count();
    std::cout << "    Completed in " << std::setprecision(3) << time_vol_xad << "s\n";
    
    // Method 2: Pure Finite Differences
    std::cout << "  Method 2: Pure Finite Differences...\n";
    auto start_vol_fd = std::chrono::high_resolution_clock::now();
    auto volGreeksFD = computeVolSurfaceGreeksFD(curve, volSurface, Z, swaption, hw, mc, calibInstruments, price_xad, 1.0);
    auto end_vol_fd = std::chrono::high_resolution_clock::now();
    double time_vol_fd = std::chrono::duration<double>(end_vol_fd - start_vol_fd).count();
    std::cout << "    Completed in " << std::setprecision(3) << time_vol_fd << "s\n";
    
    std::cout << "\n  XAD Components:\n";
    std::cout << "    dV/da     = $" << std::setprecision(2) << volGreeksXAD.dV_da << " per unit a\n";
    std::cout << "    dV/dsigma = $" << std::setprecision(2) << volGreeksXAD.dV_dsigma << " per unit sigma\n";
    
    // Comparison table
    std::cout << "\n  Vol Surface Vegas Comparison (XAD vs FD, per 1bp):\n";
    std::cout << "  " << std::string(80, '-') << "\n";
    std::cout << "  " << std::left << std::setw(12) << "Node" 
              << std::right << std::setw(12) << "XAD" 
              << std::setw(12) << "FD" 
              << std::setw(12) << "Diff" 
              << std::setw(10) << "In Calib?\n";
    std::cout << "  " << std::string(80, '-') << "\n";
    
    // Only show nodes with non-zero Greeks or in calibration
    int printed = 0;
    for (size_t ei = 0; ei < volSurface.expiries.size() && printed < 15; ++ei) {
        for (size_t ti = 0; ti < volSurface.tenors.size() && printed < 15; ++ti) {
            double vega_xad = volGreeksXAD.vegas[ei][ti];
            double vega_fd = volGreeksFD.vegas[ei][ti];
            
            bool in_calib = false;
            for (const auto& [ce, ct] : calibInstruments) {
                if (ce == ei && ct == ti) { in_calib = true; break; }
            }
            
            // Show if non-zero or in calibration
            if (std::abs(vega_xad) > 0.01 || std::abs(vega_fd) > 0.01 || in_calib) {
                std::stringstream node;
                node << std::fixed << std::setprecision(2) << volSurface.expiries[ei] << "Yx" << volSurface.tenors[ti] << "Y";
                std::cout << "  " << std::left << std::setw(12) << node.str()
                          << std::right << std::fixed << std::setprecision(2) << std::setw(12) << vega_xad
                          << std::setw(12) << vega_fd
                          << std::setw(12) << (vega_xad - vega_fd)
                          << std::setw(10) << (in_calib ? "Yes" : "No") << "\n";
                printed++;
            }
        }
    }
    std::cout << "  " << std::string(80, '-') << "\n";
    
    // Explain why zeros
    std::cout << "\n  NOTE: Vol nodes with zero Greeks are OUTSIDE the calibration set.\n";
    std::cout << "        Bumping those vols doesn't change calibrated (a, sigma), hence zero sensitivity.\n";
    
    // ==========================================================================
    // PART 4: Timing Summary
    // ==========================================================================
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TIMING SUMMARY\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "  Rate Greeks (XAD):       " << std::setprecision(3) << time_xad << "s (" 
              << curve.size() << " Greeks)\n";
    std::cout << "  Rate Greeks (FD):        " << time_fd << "s\n";
    std::cout << "  HW Param Greeks (XAD):   " << time_hw << "s (a and sigma)\n";
    std::cout << "\n  Vol Surface Greeks:\n";
    std::cout << "    XAD + Chain Rule:      " << time_vol_xad << "s (1 AAD + " 
              << volSurface.expiries.size() * volSurface.tenors.size() << " calibrations)\n";
    std::cout << "    Pure FD:               " << time_vol_fd << "s (" 
              << volSurface.expiries.size() * volSurface.tenors.size() << " calibrations + pricings)\n";
    
    double vol_speedup = time_vol_fd / time_vol_xad;
    std::cout << "\n  Vol Surface Greeks Speedup: XAD is " << std::setprecision(2) 
              << vol_speedup << "x " << (vol_speedup > 1 ? "FASTER" : "slower") << " than pure FD\n";
    
    std::cout << "\n  Why XAD + Chain Rule is faster:\n";
    std::cout << "    - XAD: 1 pricing (with AAD for dV/da, dV/dsigma) + N calibrations\n";
    std::cout << "    - FD:  N pricings + N calibrations\n";
    std::cout << "    - Savings: (N-1) expensive Monte Carlo pricings\n";
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Computation Complete\n";
    std::cout << std::string(80, '=') << "\n";
    
    return 0;
}
