#pragma once
// =============================================================================
// Monte Carlo Swaption Pricing under HW1F
// Simulates short rate path and computes swaption payoff
//
// References:
// - OpenGamma: "Algorithmic Differentiation in Finance" (Henrard)
// - Mizuho: "Hull-White One Factor Model" (Gurrieri et al.)
// - KTH Thesis: "Computation of the Greeks using the AAD" (Carmelid)
//
// Key equations:
// - State: r(t) = x(t) + ψ(t), where dx = -a*x*dt + σ(t)*dW, x(0)=0
// - ψ(t) = f^mkt(0,t) + 0.5*G'(t) ensures P^model(0,T) = P^mkt(0,T)
// - Bond price: P(t,T|x) = A(t,T) * exp(-B(t,T)*x)
// - B(t,T) = (1 - exp(-a(T-t))) / a
// - A(t,T) = P(0,T)/P(0,t) * exp(-0.5*σ_P²(t,T))
// - σ_P²(t,T) = ∫₀ᵗ σ(u)² × B(u,T)² × exp(-2a(t-u)) du
// =============================================================================

#include "hw1f/hw1f_model.hpp"
#include "instruments/swaption.hpp"
#include "utils/common.hpp"
#include <vector>
#include <cmath>

namespace hw1f {

// =============================================================================
// Monte Carlo Configuration
// =============================================================================

struct MCConfig {
    int numPaths = 10000;
    int numSteps = 100;        // Steps per year
    bool antithetic = true;     // Use antithetic variates
    bool controlVariate = false; // Use Jamshidian as control variate
    unsigned seed = 12345;
    
    MCConfig() = default;
    MCConfig(int paths, int steps, bool anti = true, unsigned s = 12345)
        : numPaths(paths), numSteps(steps), antithetic(anti), seed(s) {}
};

// =============================================================================
// Monte Carlo Pricer
// =============================================================================

template<typename Real = double>
class MonteCarloPricer {
public:
    // Monte Carlo result structure
    struct MCResult {
        Real price;
        Real stdError;
        double elapsedTime;
    };
    
    MonteCarloPricer(
        const HW1FModel<Real>& model,
        const MCConfig& config = MCConfig()
    ) : model_(model), config_(config) {}
    
    // Price European swaption using Monte Carlo
    template<typename CurveReal>
    MCResult price(
        const EuropeanSwaption& swaption,
        const DiscountCurve<CurveReal>& curve,
        const std::vector<std::vector<double>>& randomNumbers = {}
    ) const {
        using std::exp;
        using std::sqrt;
        Timer timer;
        timer.start();
        
        double expiry = swaption.expiry;
        const VanillaSwap& swap = swaption.underlying;
        double dt = expiry / config_.numSteps;
        double a = model_.meanReversion();
        
        // Generate or use provided random numbers
        std::vector<std::vector<double>> Z;
        if (randomNumbers.empty()) {
            RNG rng(config_.seed);
            Z = rng.normalMatrix(config_.numPaths, config_.numSteps);
        } else {
            Z = randomNumbers;
        }
        
        int effectivePaths = config_.antithetic ? config_.numPaths / 2 : config_.numPaths;
        
        Real sumPayoff = Real(0.0);
        Real sumPayoffSq = Real(0.0);
        
        for (int path = 0; path < effectivePaths; ++path) {
            // Simulate path(s)
            std::vector<Real> payoffs;
            
            for (int anti = 0; anti < (config_.antithetic ? 2 : 1); ++anti) {
                Real x = Real(0.0);  // Start with x(0) = 0
                double t = 0.0;
                
                for (int step = 0; step < config_.numSteps; ++step) {
                    double t_next = t + dt;
                    double z = Z[path][step];
                    if (anti == 1) z = -z;  // Antithetic path
                    
                    // Exact OU transition for x process:
                    // x(t+dt) | x(t) ~ N(x(t)*exp(-a*dt), V_r(t, t+dt))
                    // where V_r(s,t) = ∫_s^t exp(-2a(t-u)) σ(u)² du
                    // 
                    // For piecewise-constant σ(u) = σ_i in [t_i, t_{i+1}):
                    // The model's V_r function handles this properly
                    Real decay = exp(Real(-a * dt));
                    
                    // Compute conditional variance using the model's V_r function
                    // This ensures consistency with the bond pricing formulas
                    Real V = model_.V_r(t, t_next);
                    Real stdDev = sqrt(V);
                    
                    x = x * decay + stdDev * Real(z);
                    t = t_next;
                }
                
                // x at expiry
                Real x_T = x;
                
                // Compute swap PV at expiry using proper bond pricing
                // P(E, Ti | x_E) = A(E, Ti) * exp(-B(E, Ti) * x_E)
                Real swapPV = computeSwapPVatExpiry(swap, expiry, x_T, curve);
                
                // Swaption payoff
                Real payoff;
                if (swap.isPayer) {
                    payoff = (swapPV > Real(0.0)) ? swapPV : Real(0.0);
                } else {
                    payoff = (-swapPV > Real(0.0)) ? -swapPV : Real(0.0);
                }
                
                // Discount to time 0 using market discount factor
                Real df0T = model_.P_0_T(expiry, curve);
                payoff = payoff * df0T;
                
                payoffs.push_back(payoff);
            }
            
            // Average antithetic paths for variance reduction
            Real avgPayoff = payoffs[0];
            if (config_.antithetic && payoffs.size() > 1) {
                avgPayoff = (payoffs[0] + payoffs[1]) * Real(0.5);
            }
            
            sumPayoff = sumPayoff + avgPayoff;
            sumPayoffSq = sumPayoffSq + avgPayoff * avgPayoff;
        }
        
        // Compute mean and standard error
        Real mean = sumPayoff / Real(effectivePaths);
        Real variance = sumPayoffSq / Real(effectivePaths) - mean * mean;
        Real stdError = (effectivePaths > 1) ? 
            sqrt(variance / Real(effectivePaths - 1)) : Real(0.0);
        
        MCResult result;
        result.price = mean;
        result.stdError = stdError;
        result.elapsedTime = timer.elapsed();
        
        return result;
    }
    
    // =========================================================================
    // Compute swap PV at expiry given x(expiry)
    // 
    // For a payer swap starting at expiry E with payment dates T_1, ..., T_N:
    // - Float leg value = N (receive notional worth par at swap start)
    // - Fixed leg value = Σ(α_i × K × N × P(E, T_i)) + N × P(E, T_N)
    // 
    // Payer swap PV = N - Σ(c_i × P(E, T_i))
    // where c_i = α_i × K × N for i < N, and c_N = α_N × K × N + N
    //
    // Bond pricing uses proper HW1F affine formula:
    // P(E, T_i | x_E) = A(E, T_i) × exp(-B(E, T_i) × x_E)
    // =========================================================================
    template<typename CurveReal>
    Real computeSwapPVatExpiry(
        const VanillaSwap& swap,
        double expiry,
        Real x_T,
        const DiscountCurve<CurveReal>& curve
    ) const {
        Schedule fixedSched = swap.fixedSchedule();
        size_t n = fixedSched.paymentDates.size();
        
        Real fixedPV = Real(0.0);
        for (size_t i = 0; i < n; ++i) {
            // Use the proper affine bond pricing formula
            // P(E, Ti | x) = A(E, Ti) * exp(-B(E, Ti) * x)
            Real P_E_Ti = model_.bondPrice(expiry, fixedSched.paymentDates[i], x_T, curve);
            
            double cashflow = fixedSched.accrualFactors[i] * swap.fixedRate * swap.notional;
            if (i == n - 1) {
                cashflow += swap.notional;  // Add principal at maturity
            }
            fixedPV = fixedPV + Real(cashflow) * P_E_Ti;
        }
        
        // Payer swap PV = N - Σ(c_i × P(E, T_i))
        return Real(swap.notional) - fixedPV;
    }
    
    // =========================================================================
    // Alternative: Compute forward swap rate at expiry
    // 
    // R^N_n(t) = (P(t, T_n) - P(t, T_N)) / S^N_n(t)
    // where S^N_n(t) = Σ_{i=n+1}^{N} (T_i - T_{i-1}) × P(t, T_i)  [accrual factor]
    //
    // This is per the OpenGamma paper formula (3.7)
    // =========================================================================
    template<typename CurveReal>
    Real computeForwardSwapRateAtExpiry(
        const VanillaSwap& swap,
        double expiry,
        Real x_T,
        const DiscountCurve<CurveReal>& curve
    ) const {
        Schedule fixedSched = swap.fixedSchedule();
        size_t n = fixedSched.paymentDates.size();
        if (n == 0) return Real(0.0);
        
        // P(E, T_start)
        Real P_E_start = model_.bondPrice(expiry, swap.startDate, x_T, curve);
        
        // P(E, T_end) - last payment date
        Real P_E_end = model_.bondPrice(expiry, fixedSched.paymentDates[n-1], x_T, curve);
        
        // Accrual factor S^N_n = Σ α_i × P(E, T_i)
        Real annuity = Real(0.0);
        for (size_t i = 0; i < n; ++i) {
            Real P_E_Ti = model_.bondPrice(expiry, fixedSched.paymentDates[i], x_T, curve);
            annuity = annuity + Real(fixedSched.accrualFactors[i]) * P_E_Ti;
        }
        
        if (value(annuity) < 1e-12) return Real(0.0);
        
        // Forward swap rate = (P_start - P_end) / annuity
        return (P_E_start - P_E_end) / annuity;
    }
    
    // Access model and config
    const HW1FModel<Real>& model() const { return model_; }
    const MCConfig& config() const { return config_; }

private:
    HW1FModel<Real> model_;
    MCConfig config_;
};

// Convenience function
template<typename Real, typename CurveReal>
Real priceMonteCarlo(
    const EuropeanSwaption& swaption,
    const DiscountCurve<CurveReal>& curve,
    const HW1FParams& params,
    const MCConfig& config = MCConfig()
) {
    HW1FModel<Real> model(params);
    MonteCarloPricer<Real> pricer(model, config);
    return pricer.price(swaption, curve).price;
}

} // namespace hw1f
