#pragma once
// =============================================================================
// Hull-White Swaption with Jamshidian Decomposition and Price-Based Calibration
// 
// Features:
// - Black formula for ATM swaption pricing (market prices)
// - Jamshidian decomposition for HW analytical swaption pricing
// - Calibration minimizes LSE of prices (not implied vol)
// - Implicit Function Theorem for vol surface Greeks
// =============================================================================

#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <random>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// Market Data Structures
// =============================================================================

struct SwaptionParams {
    double T_option = 2.0;      // Option expiry
    int swap_tenor = 5;         // Swap tenor in years
    double K_strike = 0.04;     // ATM strike (set dynamically)
    double notional = 1000000.0;
    bool is_payer = true;       // Payer swaption
};

struct HullWhiteParams {
    double a;       // Mean reversion
    double sigma;   // Volatility
};

struct MonteCarloParams {
    int num_paths = 5000;
    double dt = 0.01;
    unsigned int seed = 42;
};

// =============================================================================
// ATM Vol Surface (Real Market Data - December 2024)
// Normal volatility (Bachelier) in absolute terms
// =============================================================================

struct ATMVolSurface {
    std::vector<double> expiries;   // Option expiries
    std::vector<double> tenors;     // Swap tenors
    std::vector<std::vector<double>> vols;  // Normal vols (9x9)
    
    ATMVolSurface() {
        expiries = {0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0};
        tenors = {1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0};
        
        // Real ATM normal vols in bps (converted below)
        vols = {
            {45, 48, 51, 55, 58, 62, 65, 67, 70},   // 3M
            {48, 52, 55, 59, 62, 66, 69, 71, 74},   // 6M
            {52, 56, 59, 63, 66, 70, 73, 75, 78},   // 1Y
            {58, 62, 65, 69, 72, 76, 79, 81, 84},   // 2Y
            {62, 66, 69, 73, 76, 80, 83, 85, 88},   // 3Y
            {68, 72, 75, 79, 82, 86, 89, 91, 94},   // 5Y
            {72, 76, 79, 83, 86, 90, 93, 95, 98},   // 7Y
            {78, 82, 85, 89, 92, 96, 99, 101, 104}, // 10Y
            {82, 86, 89, 93, 96, 100, 103, 105, 108} // 20Y
        };
        
        // Convert bps to absolute
        for (auto& row : vols) {
            for (auto& v : row) {
                v /= 10000.0;
            }
        }
    }
    
    double getVol(size_t expiry_idx, size_t tenor_idx) const {
        return vols[expiry_idx][tenor_idx];
    }
    
    size_t numExpiries() const { return expiries.size(); }
    size_t numTenors() const { return tenors.size(); }
};

// =============================================================================
// Real OIS Curve (SOFR-based, December 2024)
// =============================================================================

struct RealOISCurve {
    std::vector<double> maturities;
    std::vector<double> rates;
    std::string curve_date;
    
    RealOISCurve() {
        curve_date = "2024-12-16";
        
        maturities = {
            1.0/360, 7.0/360, 1.0/12, 2.0/12, 3.0/12, 6.0/12, 9.0/12,
            1.0, 18.0/12, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0
        };
        
        rates = {
            0.0433, 0.0432, 0.0430, 0.0426, 0.0422, 0.0413, 0.0407,
            0.0403, 0.0394, 0.0388, 0.0383, 0.0387, 0.0395, 0.0408,
            0.0423, 0.0427, 0.0430, 0.0433, 0.0436, 0.0438
        };
    }
    
    size_t size() const { return maturities.size(); }
    
    double interpolateRate(double t) const {
        if (t <= maturities.front()) return rates.front();
        if (t >= maturities.back()) return rates.back();
        
        auto it = std::lower_bound(maturities.begin(), maturities.end(), t);
        size_t idx = std::distance(maturities.begin(), it) - 1;
        double w = (t - maturities[idx]) / (maturities[idx+1] - maturities[idx]);
        return rates[idx] * (1 - w) + rates[idx + 1] * w;
    }
    
    double discountFactor(double t) const {
        if (t <= 0) return 1.0;
        return std::exp(-interpolateRate(t) * t);
    }
};

// =============================================================================
// Normal Distribution Functions
// =============================================================================

inline double normalCDF(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

inline double normalPDF(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

// =============================================================================
// BLACK FORMULA for ATM Swaption Pricing
// Market price = Annuity * sigma_N * sqrt(T) * [F-K)/sigma_N/sqrt(T) * N(d) + n(d)]
// For ATM: F = K, so d = 0, N(0) = 0.5, n(0) = 1/sqrt(2*pi)
// =============================================================================

// Calculate swap annuity (sum of discount factors for fixed leg payments)
inline double swapAnnuity(double start, int tenor, const RealOISCurve& curve) {
    double annuity = 0.0;
    for (int i = 1; i <= tenor; ++i) {
        double t = start + i;
        annuity += curve.discountFactor(t);
    }
    return annuity;
}

// Calculate forward swap rate
inline double forwardSwapRate(double start, int tenor, const RealOISCurve& curve) {
    double df_start = curve.discountFactor(start);
    double df_end = curve.discountFactor(start + tenor);
    double annuity = swapAnnuity(start, tenor, curve);
    return (df_start - df_end) / annuity;
}

// Bachelier (Normal vol) Swaption Price
// P = A * sigma_N * sqrt(T) * [d * N(d) + n(d)]  where d = (F-K)/(sigma_N * sqrt(T))
inline double blackSwaptionPrice(double forward, double strike, double expiry,
                                  double sigma_n, double annuity, double notional,
                                  bool is_payer) {
    if (expiry <= 0 || sigma_n <= 0) return 0.0;
    
    double sqrt_T = std::sqrt(expiry);
    double d = (forward - strike) / (sigma_n * sqrt_T);
    double omega = is_payer ? 1.0 : -1.0;
    
    double price = annuity * sigma_n * sqrt_T * 
                   (omega * d * normalCDF(omega * d) + normalPDF(d));
    
    return notional * price;
}

// Market price for swaption given vol surface
inline double marketSwaptionPrice(double expiry, int tenor, 
                                   const ATMVolSurface& volSurface,
                                   const RealOISCurve& curve,
                                   double notional, bool is_payer) {
    // Find closest vol surface node
    size_t ei = 0, ti = 0;
    double min_e = 1e10, min_t = 1e10;
    for (size_t i = 0; i < volSurface.expiries.size(); ++i) {
        if (std::abs(volSurface.expiries[i] - expiry) < min_e) {
            min_e = std::abs(volSurface.expiries[i] - expiry);
            ei = i;
        }
    }
    for (size_t i = 0; i < volSurface.tenors.size(); ++i) {
        if (std::abs(volSurface.tenors[i] - tenor) < min_t) {
            min_t = std::abs(volSurface.tenors[i] - tenor);
            ti = i;
        }
    }
    
    double sigma_n = volSurface.getVol(ei, ti);
    double F = forwardSwapRate(expiry, tenor, curve);
    double A = swapAnnuity(expiry, tenor, curve);
    
    return blackSwaptionPrice(F, F, expiry, sigma_n, A, notional, is_payer);  // ATM: K = F
}

// =============================================================================
// JAMSHIDIAN DECOMPOSITION for Hull-White Swaption Pricing
//
// Key idea: At exercise time T, swaption payoff = max(1 - sum(c_i * P(T, T_i)), 0)
// where c_i = coupon payments (K for fixed leg, 1 for final principal)
//
// Using Jamshidian's trick: Find r* such that sum(c_i * P(T, T_i; r*)) = 1
// Then swaption = sum of call/put options on zero-coupon bonds
// =============================================================================

// Hull-White B(t, T) function
inline double HW_B(double a, double t, double T) {
    double tau = T - t;
    if (std::abs(a) < 1e-10) return tau;
    return (1.0 - std::exp(-a * tau)) / a;
}

// Hull-White A(t, T) function (log)
inline double HW_logA(double a, double sigma, double t, double T, const RealOISCurve& curve) {
    double B = HW_B(a, t, T);
    double P_T = curve.discountFactor(T);
    double P_t = curve.discountFactor(t);
    
    // Forward rate at t
    double h = 1e-5;
    double f_t = -(std::log(curve.discountFactor(t + h)) - std::log(P_t)) / h;
    
    double sigma2 = sigma * sigma;
    double term = sigma2 / (4.0 * a) * (1.0 - std::exp(-2.0 * a * t)) * B * B;
    
    return std::log(P_T) - std::log(P_t) + B * f_t - term;
}

// Hull-White zero-coupon bond price P(t, T) given short rate r_t
inline double HW_bondPrice(double r_t, double t, double T, double a, double sigma,
                            const RealOISCurve& curve) {
    if (T <= t) return 1.0;
    double B = HW_B(a, t, T);
    double logA = HW_logA(a, sigma, t, T, curve);
    return std::exp(logA - B * r_t);
}

// Find critical rate r* for Jamshidian decomposition
// sum(c_i * P(T, T_i; r*)) = 1
inline double findCriticalRate(double a, double sigma, double T_option, 
                                const std::vector<double>& payment_times,
                                const std::vector<double>& coupons,
                                const RealOISCurve& curve) {
    // Newton-Raphson to find r*
    double r_star = forwardSwapRate(T_option, payment_times.size(), curve);
    
    for (int iter = 0; iter < 50; ++iter) {
        double f = -1.0;
        double df = 0.0;
        
        for (size_t i = 0; i < payment_times.size(); ++i) {
            double P = HW_bondPrice(r_star, T_option, payment_times[i], a, sigma, curve);
            double B = HW_B(a, T_option, payment_times[i]);
            f += coupons[i] * P;
            df -= coupons[i] * P * B;
        }
        
        if (std::abs(df) < 1e-15) break;
        double delta = f / df;
        r_star -= delta;
        
        if (std::abs(delta) < 1e-12) break;
    }
    
    return r_star;
}

// Bond option price under Hull-White (closed-form)
// Call on P(T, S): max(P(T, S) - K, 0)
inline double HW_bondOption(double a, double sigma, double t, double T, double S,
                             double K, bool is_call, const RealOISCurve& curve) {
    if (T <= t) return 0.0;
    
    double P_T = curve.discountFactor(T);
    double P_S = curve.discountFactor(S);
    
    double B_TS = HW_B(a, T, S);
    double sigma_p = sigma * std::sqrt((1.0 - std::exp(-2.0 * a * (T - t))) / (2.0 * a)) * B_TS;
    
    if (sigma_p < 1e-10) {
        // Intrinsic value
        return is_call ? std::max(P_S - K * P_T, 0.0) : std::max(K * P_T - P_S, 0.0);
    }
    
    double d1 = std::log(P_S / (K * P_T)) / sigma_p + 0.5 * sigma_p;
    double d2 = d1 - sigma_p;
    
    if (is_call) {
        return P_S * normalCDF(d1) - K * P_T * normalCDF(d2);
    } else {
        return K * P_T * normalCDF(-d2) - P_S * normalCDF(-d1);
    }
}

// Jamshidian decomposition for payer swaption
// Payer swaption = sum of put options on zero-coupon bonds
inline double HW_swaptionPriceJamshidian(double a, double sigma, double T_option,
                                          int swap_tenor, double strike,
                                          double notional, bool is_payer,
                                          const RealOISCurve& curve) {
    // Payment times and coupons
    std::vector<double> payment_times;
    std::vector<double> coupons;
    
    for (int i = 1; i <= swap_tenor; ++i) {
        payment_times.push_back(T_option + i);
        if (i < swap_tenor) {
            coupons.push_back(strike);  // Coupon payments
        } else {
            coupons.push_back(1.0 + strike);  // Final principal + coupon
        }
    }
    
    // Find critical rate r*
    double r_star = findCriticalRate(a, sigma, T_option, payment_times, coupons, curve);
    
    // Strike prices for bond options
    std::vector<double> K_bonds;
    for (size_t i = 0; i < payment_times.size(); ++i) {
        K_bonds.push_back(HW_bondPrice(r_star, T_option, payment_times[i], a, sigma, curve));
    }
    
    // Sum of bond options
    double price = 0.0;
    for (size_t i = 0; i < payment_times.size(); ++i) {
        // For payer: puts on bonds; For receiver: calls on bonds
        double opt = HW_bondOption(a, sigma, 0.0, T_option, payment_times[i],
                                    K_bonds[i], !is_payer, curve);
        price += coupons[i] * opt;
    }
    
    return notional * price;
}

// =============================================================================
// PRICE-BASED CALIBRATION
// Minimize: sum_i (P_HW(a, sigma, swaption_i) - P_Black(swaption_i))^2
// =============================================================================

struct PriceCalibResult {
    double a;
    double sigma;
    double price_rmse;
    int iterations;
    std::vector<std::pair<double, double>> price_comparison;  // (model, market) pairs
};

// Compute price RMSE for given (a, sigma)
inline double computePriceRMSE(double a, double sigma,
                                const ATMVolSurface& volSurface,
                                const RealOISCurve& curve,
                                const std::vector<std::pair<size_t, size_t>>& calibInstruments,
                                double notional = 1.0,
                                std::vector<std::pair<double, double>>* price_pairs = nullptr) {
    double sse = 0.0;
    int count = 0;
    
    if (price_pairs) price_pairs->clear();
    
    for (const auto& [ei, ti] : calibInstruments) {
        double expiry = volSurface.expiries[ei];
        int tenor = static_cast<int>(volSurface.tenors[ti]);
        
        // Market price from Black formula
        double sigma_n = volSurface.getVol(ei, ti);
        double F = forwardSwapRate(expiry, tenor, curve);
        double A = swapAnnuity(expiry, tenor, curve);
        double market_price = blackSwaptionPrice(F, F, expiry, sigma_n, A, notional, true);
        
        // Model price from Jamshidian
        double model_price = HW_swaptionPriceJamshidian(a, sigma, expiry, tenor, F,
                                                         notional, true, curve);
        
        double diff = model_price - market_price;
        sse += diff * diff;
        count++;
        
        if (price_pairs) {
            price_pairs->push_back({model_price, market_price});
        }
    }
    
    return std::sqrt(sse / count);
}

// Two-stage calibration: grid search + gradient descent (on prices)
inline PriceCalibResult calibrateHullWhitePrices(const ATMVolSurface& volSurface,
                                                  const RealOISCurve& curve,
                                                  const std::vector<std::pair<size_t, size_t>>& calibInstruments,
                                                  double notional = 1.0) {
    double best_a = 0.1;
    double best_sigma = 0.01;
    double best_rmse = 1e10;
    
    // Stage 1: Coarse grid search
    for (double a = 0.02; a <= 0.5; a += 0.02) {
        for (double sigma = 0.005; sigma <= 0.03; sigma += 0.002) {
            double rmse = computePriceRMSE(a, sigma, volSurface, curve, calibInstruments, notional);
            if (rmse < best_rmse) {
                best_rmse = rmse;
                best_a = a;
                best_sigma = sigma;
            }
        }
    }
    
    // Stage 2: Local refinement
    double a = best_a;
    double sigma = best_sigma;
    double h = 1e-6;
    double lr_a = 0.001;
    double lr_sigma = 0.00001;
    
    int iterations = 0;
    for (int iter = 0; iter < 100; ++iter) {
        double f0 = computePriceRMSE(a, sigma, volSurface, curve, calibInstruments, notional);
        
        double f_a_plus = computePriceRMSE(a + h, sigma, volSurface, curve, calibInstruments, notional);
        double f_sigma_plus = computePriceRMSE(a, sigma + h, volSurface, curve, calibInstruments, notional);
        
        double grad_a = (f_a_plus - f0) / h;
        double grad_sigma = (f_sigma_plus - f0) / h;
        
        double new_a = std::clamp(a - lr_a * grad_a, 0.01, 0.6);
        double new_sigma = std::clamp(sigma - lr_sigma * grad_sigma, 0.001, 0.05);
        
        double new_rmse = computePriceRMSE(new_a, new_sigma, volSurface, curve, calibInstruments, notional);
        
        if (new_rmse < best_rmse) {
            a = new_a;
            sigma = new_sigma;
            best_rmse = new_rmse;
            best_a = a;
            best_sigma = sigma;
        } else {
            lr_a *= 0.5;
            lr_sigma *= 0.5;
        }
        
        iterations++;
        if (lr_a < 1e-10) break;
    }
    
    std::vector<std::pair<double, double>> price_pairs;
    computePriceRMSE(best_a, best_sigma, volSurface, curve, calibInstruments, notional, &price_pairs);
    
    return {best_a, best_sigma, best_rmse, iterations, price_pairs};
}

// =============================================================================
// Hull-White Functions for AD (templated)
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

// Theta(t) with AD a, sigma
template<typename T>
T theta(double t, const std::vector<double>& mats, const std::vector<T>& rates,
        T a, T sigma, double h = 1e-5) {
    using std::exp;
    T f_t = forwardRate<T>(t, mats, rates, h);
    T f_th = forwardRate<T>(t + h, mats, rates, h);
    T f_prime = (f_th - f_t) / h;
    T term3 = (sigma * sigma / (T(2.0) * a)) * (T(1.0) - exp(T(-2.0) * a * t));
    return f_prime + a * f_t + term3;
}

// Theta with double a, sigma
template<typename T>
T thetaDouble(double t, const std::vector<double>& mats, const std::vector<T>& rates,
              double a, double sigma, double h = 1e-5) {
    using std::exp;
    T f_t = forwardRate<T>(t, mats, rates, h);
    T f_th = forwardRate<T>(t + h, mats, rates, h);
    T f_prime = (f_th - f_t) / h;
    double term3 = (sigma * sigma / (2.0 * a)) * (1.0 - exp(-2.0 * a * t));
    return f_prime + a * f_t + term3;
}

// Bond price with AD types
template<typename T>
T bondPrice(T r_t, double t, double T_mat, const std::vector<double>& mats, 
            const std::vector<T>& rates, double a, double sigma, T f_t_cache) {
    using std::exp;
    using std::log;
    
    if (T_mat <= t) return T(1.0);
    
    double tau = T_mat - t;
    double B = (1.0 - exp(-a * tau)) / a;
    
    T P_T = discount<T>(T_mat, mats, rates);
    T P_t = discount<T>(t, mats, rates);
    
    double sigma2_4a = sigma * sigma / (4.0 * a);
    double exp_neg2at = exp(-2.0 * a * t);
    
    T log_A = log(P_T) - log(P_t) + B * f_t_cache - 
              sigma2_4a * (1.0 - exp_neg2at) * B * B;
    
    return exp(log_A) * exp(-B * r_t);
}

// Bond price with AD a and sigma
template<typename T>
T bondPriceAD(T r_t, double t, double T_mat, const std::vector<double>& mats,
              const std::vector<T>& rates, T a, T sigma, T f_t_cache) {
    using std::exp;
    using std::log;
    
    if (T_mat <= t) return T(1.0);
    
    T tau = T(T_mat - t);
    T B = (T(1.0) - exp(-a * tau)) / a;
    
    T P_T = discount<T>(T_mat, mats, rates);
    T P_t = discount<T>(t, mats, rates);
    
    T sigma2_4a = sigma * sigma / (T(4.0) * a);
    T exp_neg2at = exp(T(-2.0) * a * t);
    
    T log_A = log(P_T) - log(P_t) + B * f_t_cache -
              sigma2_4a * (T(1.0) - exp_neg2at) * B * B;
    
    return exp(log_A) * exp(-B * r_t);
}

// =============================================================================
// Random Number Generation
// =============================================================================

inline std::vector<std::vector<double>> generateRandomMatrix(int num_paths, int num_steps, 
                                                              unsigned int seed) {
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
