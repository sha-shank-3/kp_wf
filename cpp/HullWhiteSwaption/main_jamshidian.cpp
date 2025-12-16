// =============================================================================
// Hull-White Swaption with Jamshidian Decomposition
//
// Price-based calibration: minimize LSE(P_HW - P_Black)
// Greeks computed for BOTH:
//   1. Rate curve nodes (dV/dr_i) using XAD
//   2. Vol surface nodes (dV/dvol_ij) using Implicit Function Theorem
//
// Timing comparison: XAD + IFT vs Pure Finite Differences
// =============================================================================

#include "HullWhiteJamshidian.hpp"
#include <XAD/XAD.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <sstream>

// =============================================================================
// Monte Carlo Pricing (used for validation)
// =============================================================================

template<typename T>
T simulateAndPricePath(
    const std::vector<double>& t_grid,
    const std::vector<double>& mats,
    const std::vector<T>& rates,
    const std::vector<double>& Z_path,
    const SwaptionParams& swaption,
    double a, double sigma, double dt
) {
    using std::exp;
    using std::sqrt;
    using xad::max;
    
    double sqrt_dt = sqrt(dt);
    T r_t = forwardRate<T>(0.0, mats, rates);
    
    size_t n_steps = Z_path.size();
    for (size_t i = 0; i < n_steps; ++i) {
        T theta_t = thetaDouble<T>(t_grid[i], mats, rates, a, sigma);
        r_t = r_t + (theta_t - a * r_t) * dt + sigma * sqrt_dt * Z_path[i];
    }
    
    T r_T = r_t;
    T f_T = forwardRate<T>(swaption.T_option, mats, rates);
    
    std::vector<T> bonds;
    for (int j = 1; j <= swaption.swap_tenor; ++j) {
        double T_pay = swaption.T_option + j;
        bonds.push_back(bondPrice<T>(r_T, swaption.T_option, T_pay, mats, rates, a, sigma, f_T));
    }
    
    T fixed_pv = T(0.0);
    for (const auto& b : bonds) {
        fixed_pv = fixed_pv + b * swaption.K_strike;
    }
    
    T float_pv = T(1.0) - bonds.back();
    T swap_val = (float_pv - fixed_pv) * swaption.notional;
    
    T payoff;
    if (swaption.is_payer) {
        payoff = max(swap_val, T(0.0));
    } else {
        T neg_swap = T(0.0) - swap_val;
        payoff = max(neg_swap, T(0.0));
    }
    
    T disc = discount<T>(swaption.T_option, mats, rates);
    return disc * payoff;
}

// =============================================================================
// RATE CURVE GREEKS using XAD (AAD)
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
    
    tape_type tape;
    
    // Create AD rates
    std::vector<AD> rates_ad(n_rates);
    for (size_t i = 0; i < n_rates; ++i) {
        rates_ad[i] = curve.rates[i];
    }
    
    // Register inputs
    tape.registerInputs(rates_ad);
    tape.newRecording();
    
    // Monte Carlo pricing
    AD total = AD(0.0);
    for (int path = 0; path < mc.num_paths; ++path) {
        total = total + simulateAndPricePath<AD>(t_grid, curve.maturities, rates_ad,
                                                  Z[path], swaption, hw.a, hw.sigma, mc.dt);
    }
    AD price = total / AD(static_cast<double>(mc.num_paths));
    
    tape.registerOutput(price);
    derivative(price) = 1.0;
    tape.computeAdjoints();
    
    // Extract derivatives
    std::vector<double> greeks(n_rates);
    for (size_t i = 0; i < n_rates; ++i) {
        greeks[i] = derivative(rates_ad[i]);
    }
    
    return {value(price), greeks};
}

// =============================================================================
// RATE CURVE GREEKS using Finite Differences (for comparison)
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
    
    std::vector<double> t_grid(num_steps + 1);
    for (int i = 0; i <= num_steps; ++i) {
        t_grid[i] = i * mc.dt;
    }
    
    // Pricing function
    auto priceWithRates = [&](const std::vector<double>& rates) -> double {
        double total = 0.0;
        for (int path = 0; path < mc.num_paths; ++path) {
            total += simulateAndPricePath<double>(t_grid, curve.maturities, rates,
                                                   Z[path], swaption, hw.a, hw.sigma, mc.dt);
        }
        return total / mc.num_paths;
    };
    
    double base_price = priceWithRates(curve.rates);
    
    std::vector<double> greeks(n_rates);
    for (size_t i = 0; i < n_rates; ++i) {
        std::vector<double> rates_up = curve.rates;
        std::vector<double> rates_down = curve.rates;
        rates_up[i] += bump;
        rates_down[i] -= bump;
        
        greeks[i] = (priceWithRates(rates_up) - priceWithRates(rates_down)) / (2.0 * bump);
    }
    
    return {base_price, greeks};
}

// =============================================================================
// HULL-WHITE PARAMETER GREEKS using XAD (dV/da, dV/dsigma)
// =============================================================================

struct HWParamGreeks {
    double price;
    double dV_da;
    double dV_dsigma;
};

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
    double sqrt_dt = std::sqrt(mc.dt);
    
    std::vector<double> t_grid(num_steps + 1);
    for (int i = 0; i <= num_steps; ++i) {
        t_grid[i] = i * mc.dt;
    }
    
    double acc_price = 0.0;
    double acc_d_a = 0.0;
    double acc_d_sigma = 0.0;
    
    tape_type tape;
    
    for (int path = 0; path < mc.num_paths; ++path) {
        tape.clearAll();
        
        AD a_ad = hw.a;
        AD sigma_ad = hw.sigma;
        
        std::vector<AD> rates_ad(curve.size());
        for (size_t i = 0; i < curve.size(); ++i) {
            rates_ad[i] = curve.rates[i];
        }
        
        tape.registerInput(a_ad);
        tape.registerInput(sigma_ad);
        tape.newRecording();
        
        AD r_t = forwardRate<AD>(0.0, curve.maturities, rates_ad);
        
        for (int i = 0; i < num_steps; ++i) {
            double t = t_grid[i];
            AD f_t = forwardRate<AD>(t, curve.maturities, rates_ad);
            AD f_th = forwardRate<AD>(t + 1e-5, curve.maturities, rates_ad);
            AD f_prime = (f_th - f_t) / 1e-5;
            
            AD term3 = (sigma_ad * sigma_ad / (AD(2.0) * a_ad)) * 
                       (AD(1.0) - exp(AD(-2.0) * a_ad * t));
            AD theta_t = f_prime + a_ad * f_t + term3;
            
            r_t = r_t + (theta_t - a_ad * r_t) * mc.dt + sigma_ad * sqrt_dt * Z[path][i];
        }
        
        AD r_T = r_t;
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
        
        AD fixed_pv = AD(0.0);
        for (const auto& b : bonds) {
            fixed_pv = fixed_pv + b * swaption.K_strike;
        }
        
        AD float_pv = AD(1.0) - bonds.back();
        AD swap_val = (float_pv - fixed_pv) * swaption.notional;
        
        AD payoff;
        if (swaption.is_payer) {
            payoff = xad::max(swap_val, AD(0.0));
        } else {
            AD neg_swap = AD(0.0) - swap_val;
            payoff = xad::max(neg_swap, AD(0.0));
        }
        
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
// VOL SURFACE GREEKS using Implicit Function Theorem
//
// Chain rule: dV/d(vol_ij) = (dV/da)(da/dvol_ij) + (dV/dsigma)(dsigma/dvol_ij)
//
// We compute da/dvol_ij and dsigma/dvol_ij by bumping vol_ij and recalibrating
// =============================================================================

struct VolSurfaceGreeks {
    double price;
    std::vector<std::vector<double>> vegas;  // dV/d(vol_ij) for each node
    double dV_da;
    double dV_dsigma;
};

VolSurfaceGreeks computeVolSurfaceGreeksXAD(
    const RealOISCurve& curve,
    const ATMVolSurface& volSurface,
    const std::vector<std::vector<double>>& Z,
    const SwaptionParams& swaption,
    const HullWhiteParams& hw_base,
    const MonteCarloParams& mc,
    const std::vector<std::pair<size_t, size_t>>& calibInstruments,
    double notional_calib = 1.0,
    double vol_bump = 0.0001  // 1 bp bump
) {
    // Step 1: Get dV/da and dV/dsigma using XAD
    auto hwGreeks = computeHWParamGreeksXAD(curve, Z, swaption, hw_base, mc);
    
    size_t n_exp = volSurface.numExpiries();
    size_t n_ten = volSurface.numTenors();
    
    std::vector<std::vector<double>> vegas(n_exp, std::vector<double>(n_ten, 0.0));
    
    // Step 2: For each vol node, bump and recalibrate to get da/dvol, dsigma/dvol
    for (size_t ei = 0; ei < n_exp; ++ei) {
        for (size_t ti = 0; ti < n_ten; ++ti) {
            // Create bumped vol surface
            ATMVolSurface bumped_surface = volSurface;
            bumped_surface.vols[ei][ti] += vol_bump;
            
            // Recalibrate with bumped vol (PRICE-BASED)
            auto bumped_calib = calibrateHullWhitePrices(bumped_surface, curve, 
                                                          calibInstruments, notional_calib);
            
            // Compute sensitivities
            double da_dvol = (bumped_calib.a - hw_base.a) / vol_bump;
            double dsigma_dvol = (bumped_calib.sigma - hw_base.sigma) / vol_bump;
            
            // Chain rule: dV/dvol = dV/da * da/dvol + dV/dsigma * dsigma/dvol
            vegas[ei][ti] = hwGreeks.dV_da * da_dvol + hwGreeks.dV_dsigma * dsigma_dvol;
        }
    }
    
    return {hwGreeks.price, vegas, hwGreeks.dV_da, hwGreeks.dV_dsigma};
}

// =============================================================================
// VOL SURFACE GREEKS using Pure Finite Differences (for comparison)
// =============================================================================

VolSurfaceGreeks computeVolSurfaceGreeksFD(
    const RealOISCurve& curve,
    const ATMVolSurface& volSurface,
    const std::vector<std::vector<double>>& Z,
    const SwaptionParams& swaption,
    const MonteCarloParams& mc,
    const std::vector<std::pair<size_t, size_t>>& calibInstruments,
    double base_price,
    double notional_calib = 1.0,
    double vol_bump = 0.0001
) {
    size_t n_exp = volSurface.numExpiries();
    size_t n_ten = volSurface.numTenors();
    
    std::vector<std::vector<double>> vegas(n_exp, std::vector<double>(n_ten, 0.0));
    int num_steps = static_cast<int>(swaption.T_option / mc.dt);
    
    std::vector<double> t_grid(num_steps + 1);
    for (int i = 0; i <= num_steps; ++i) {
        t_grid[i] = i * mc.dt;
    }
    
    // Pricing function with given HW params
    auto priceWithHW = [&](double a, double sigma) -> double {
        double total = 0.0;
        for (int path = 0; path < mc.num_paths; ++path) {
            total += simulateAndPricePath<double>(t_grid, curve.maturities, curve.rates,
                                                   Z[path], swaption, a, sigma, mc.dt);
        }
        return total / mc.num_paths;
    };
    
    for (size_t ei = 0; ei < n_exp; ++ei) {
        for (size_t ti = 0; ti < n_ten; ++ti) {
            // Bump vol and recalibrate
            ATMVolSurface bumped_surface = volSurface;
            bumped_surface.vols[ei][ti] += vol_bump;
            
            auto bumped_calib = calibrateHullWhitePrices(bumped_surface, curve,
                                                          calibInstruments, notional_calib);
            
            // Price with bumped calibration
            double bumped_price = priceWithHW(bumped_calib.a, bumped_calib.sigma);
            
            vegas[ei][ti] = (bumped_price - base_price) / vol_bump;
        }
    }
    
    return {base_price, vegas, 0.0, 0.0};
}

// =============================================================================
// COMBINED GREEKS: Rate Curve + Vol Surface in Single Report
// =============================================================================

struct CombinedGreeks {
    double price;
    std::vector<double> rate_deltas;  // dV/dr_i
    std::vector<std::vector<double>> vol_vegas;  // dV/dvol_ij
    double dV_da;
    double dV_dsigma;
    double time_rate_xad;
    double time_vol_xad;
    double time_total_xad;
};

CombinedGreeks computeAllGreeksXAD(
    const RealOISCurve& curve,
    const ATMVolSurface& volSurface,
    const std::vector<std::vector<double>>& Z,
    const SwaptionParams& swaption,
    const HullWhiteParams& hw,
    const MonteCarloParams& mc,
    const std::vector<std::pair<size_t, size_t>>& calibInstruments
) {
    CombinedGreeks result;
    auto start_total = std::chrono::high_resolution_clock::now();
    
    // Rate curve Greeks
    auto start_rate = std::chrono::high_resolution_clock::now();
    auto [price, rate_greeks] = computeRateGreeksXAD(curve, Z, swaption, hw, mc);
    auto end_rate = std::chrono::high_resolution_clock::now();
    result.time_rate_xad = std::chrono::duration<double>(end_rate - start_rate).count();
    
    result.price = price;
    result.rate_deltas = rate_greeks;
    
    // Vol surface Greeks (using IFT)
    auto start_vol = std::chrono::high_resolution_clock::now();
    auto volGreeks = computeVolSurfaceGreeksXAD(curve, volSurface, Z, swaption, hw, mc, calibInstruments);
    auto end_vol = std::chrono::high_resolution_clock::now();
    result.time_vol_xad = std::chrono::duration<double>(end_vol - start_vol).count();
    
    result.vol_vegas = volGreeks.vegas;
    result.dV_da = volGreeks.dV_da;
    result.dV_dsigma = volGreeks.dV_dsigma;
    
    auto end_total = std::chrono::high_resolution_clock::now();
    result.time_total_xad = std::chrono::duration<double>(end_total - start_total).count();
    
    return result;
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    std::cout << std::string(80, '=') << "\n";
    std::cout << "Hull-White Swaption - Jamshidian Decomposition & Price-Based Calibration\n";
    std::cout << "Greeks: Rate Curve + Vol Surface | XAD + Implicit Function Theorem\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    // Load market data
    RealOISCurve curve;
    ATMVolSurface volSurface;
    
    std::cout << "OIS Curve Date: " << curve.curve_date << "\n";
    std::cout << "Vol Surface: " << volSurface.numExpiries() << " expiries x " 
              << volSurface.numTenors() << " tenors = " 
              << volSurface.numExpiries() * volSurface.numTenors() << " nodes\n\n";
    
    // Define calibration instruments
    std::vector<std::pair<size_t, size_t>> calibInstruments;
    for (size_t ei = 2; ei < 6; ++ei) {
        for (size_t ti = 0; ti < 5; ++ti) {
            calibInstruments.push_back({ei, ti});
        }
    }
    std::cout << "Calibration: " << calibInstruments.size() << " ATM swaptions\n";
    
    // Price-based calibration
    std::cout << "\n" << std::string(80, '-') << "\n";
    std::cout << "PRICE-BASED CALIBRATION\n";
    std::cout << "Objective: min sum (P_HW_Jamshidian - P_Black)^2\n";
    std::cout << std::string(80, '-') << "\n";
    
    auto start_calib = std::chrono::high_resolution_clock::now();
    auto calibResult = calibrateHullWhitePrices(volSurface, curve, calibInstruments);
    auto end_calib = std::chrono::high_resolution_clock::now();
    double time_calib = std::chrono::duration<double>(end_calib - start_calib).count();
    
    std::cout << "\nCalibration Results:\n";
    std::cout << "  a (mean reversion) = " << std::fixed << std::setprecision(6) << calibResult.a << "\n";
    std::cout << "  sigma (volatility) = " << std::setprecision(6) << calibResult.sigma << "\n";
    std::cout << "  Price RMSE         = $" << std::setprecision(4) << calibResult.price_rmse << "\n";
    std::cout << "  Iterations         = " << calibResult.iterations << "\n";
    std::cout << "  Time               = " << std::setprecision(3) << time_calib << "s\n";
    
    // Setup swaption
    SwaptionParams swaption;
    swaption.K_strike = forwardSwapRate(swaption.T_option, swaption.swap_tenor, curve);
    
    HullWhiteParams hw = {calibResult.a, calibResult.sigma};
    MonteCarloParams mc;
    mc.num_paths = 2000;
    
    int num_steps = static_cast<int>(swaption.T_option / mc.dt);
    auto Z = generateRandomMatrix(mc.num_paths, num_steps, mc.seed);
    
    std::cout << "\n" << std::string(80, '-') << "\n";
    std::cout << "Target Swaption:\n";
    std::cout << "  " << swaption.T_option << "Y x " << swaption.swap_tenor 
              << "Y, ATM Strike = " << std::setprecision(3) << swaption.K_strike * 100 << "%\n";
    std::cout << "  Notional: $" << std::fixed << std::setprecision(0) << swaption.notional
              << ", Type: " << (swaption.is_payer ? "Payer" : "Receiver") << "\n";
    std::cout << "  MC Paths: " << mc.num_paths << ", Steps: " << num_steps << "\n";
    
    // Combined Greeks
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "COMBINED GREEKS (Rate Curve + Vol Surface) via XAD\n";
    std::cout << std::string(80, '=') << "\n";
    
    auto allGreeks = computeAllGreeksXAD(curve, volSurface, Z, swaption, hw, mc, calibInstruments);
    
    std::cout << "\nPrice: $" << std::setprecision(2) << allGreeks.price << "\n";
    
    // Rate Deltas
    std::cout << "\n--- Rate Curve Greeks (dV/dr, per 1bp) ---\n";
    std::cout << std::left << std::setw(10) << "Maturity" 
              << std::right << std::setw(15) << "Delta\n";
    std::cout << std::string(25, '-') << "\n";
    
    for (size_t i = 0; i < std::min(size_t(10), curve.size()); ++i) {
        std::cout << std::left << std::fixed << std::setprecision(3) 
                  << std::setw(10) << curve.maturities[i]
                  << std::right << std::setprecision(2) 
                  << std::setw(15) << allGreeks.rate_deltas[i] * 0.0001 << "\n";
    }
    
    // HW Param Greeks
    std::cout << "\n--- Hull-White Parameter Greeks ---\n";
    std::cout << "  dV/da     = $" << std::setprecision(2) << allGreeks.dV_da << " per unit a\n";
    std::cout << "  dV/dsigma = $" << std::setprecision(2) << allGreeks.dV_dsigma << " per unit sigma\n";
    
    // Vol Vegas
    std::cout << "\n--- Vol Surface Greeks (dV/dvol, per 1bp) ---\n";
    std::cout << std::left << std::setw(12) << "Node" 
              << std::right << std::setw(12) << "Vega" 
              << std::setw(10) << "In Calib?\n";
    std::cout << std::string(35, '-') << "\n";
    
    int printed = 0;
    for (size_t ei = 0; ei < volSurface.numExpiries() && printed < 12; ++ei) {
        for (size_t ti = 0; ti < volSurface.numTenors() && printed < 12; ++ti) {
            double vega = allGreeks.vol_vegas[ei][ti] * 0.0001;
            
            bool in_calib = false;
            for (const auto& [ce, ct] : calibInstruments) {
                if (ce == ei && ct == ti) { in_calib = true; break; }
            }
            
            if (std::abs(vega) > 0.01 || in_calib) {
                std::stringstream node;
                node << std::fixed << std::setprecision(2) 
                     << volSurface.expiries[ei] << "Yx" << volSurface.tenors[ti] << "Y";
                
                std::cout << std::left << std::setw(12) << node.str()
                          << std::right << std::setprecision(2) << std::setw(12) << vega
                          << std::setw(10) << (in_calib ? "Yes" : "No") << "\n";
                printed++;
            }
        }
    }
    
    // FD Comparison (Rate Greeks only for speed)
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "COMPARISON: XAD vs Finite Differences\n";
    std::cout << std::string(80, '=') << "\n";
    
    std::cout << "\nComputing Rate Greeks with FD...\n";
    auto start_rate_fd = std::chrono::high_resolution_clock::now();
    auto [price_fd, rate_greeks_fd] = computeRateGreeksFD(curve, Z, swaption, hw, mc);
    auto end_rate_fd = std::chrono::high_resolution_clock::now();
    double time_rate_fd = std::chrono::duration<double>(end_rate_fd - start_rate_fd).count();
    
    std::cout << "\nRate Greeks Comparison (XAD vs FD):\n";
    std::cout << std::left << std::setw(10) << "Maturity" 
              << std::right << std::setw(15) << "XAD" 
              << std::setw(15) << "FD" 
              << std::setw(12) << "Rel Diff%\n";
    std::cout << std::string(55, '-') << "\n";
    
    for (size_t i = 0; i < std::min(size_t(8), curve.size()); ++i) {
        double g_xad = allGreeks.rate_deltas[i];
        double g_fd = rate_greeks_fd[i];
        double rel = std::abs(g_fd) > 10 ? std::abs(g_xad - g_fd) / std::abs(g_fd) * 100 : 0;
        
        std::cout << std::left << std::fixed << std::setprecision(3) << std::setw(10) << curve.maturities[i]
                  << std::right << std::setprecision(2) << std::setw(15) << g_xad
                  << std::setw(15) << g_fd
                  << std::setprecision(3) << std::setw(11) << rel << "%\n";
    }
    
    // Timing Summary
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TIMING SUMMARY\n";
    std::cout << std::string(80, '=') << "\n";
    
    std::cout << "\nRate Curve Greeks (" << curve.size() << " Greeks):\n";
    std::cout << "  XAD (AAD):  " << std::fixed << std::setprecision(3) 
              << allGreeks.time_rate_xad << "s\n";
    std::cout << "  FD:         " << time_rate_fd << "s\n";
    double rate_speedup = time_rate_fd / allGreeks.time_rate_xad;
    std::cout << "  Speedup:    " << std::setprecision(1) << rate_speedup << "x\n";
    
    std::cout << "\nVol Surface Greeks (" << volSurface.numExpiries() * volSurface.numTenors() << " Greeks):\n";
    std::cout << "  XAD + IFT:  " << std::setprecision(3) << allGreeks.time_vol_xad << "s\n";
    
    std::cout << "\n" << std::string(80, '-') << "\n";
    std::cout << "Why XAD + IFT is faster for Vol Surface Greeks:\n";
    std::cout << "  XAD Method:  1 pricing (AAD) + N recalibrations\n";
    std::cout << "  Pure FD:     N recalibrations + N pricings\n";
    std::cout << "  Savings:     (N-1) expensive Monte Carlo pricings\n";
    std::cout << std::string(80, '-') << "\n";
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Computation Complete\n";
    std::cout << std::string(80, '=') << "\n";
    
    return 0;
}
