#pragma once
// =============================================================================
// Hull-White 1-Factor Model Implementation
// Piecewise-constant volatility, analytic bond pricing
// 
// References:
// - OpenGamma: "Algorithmic Differentiation in Finance" (Henrard, 2017)
// - Mizuho: "Hull-White One Factor Model" (Gurrieri et al., 2009)  
// - KTH Thesis: "Computation of the Greeks using AAD" (Carmelid, 2017)
// - Brigo & Mercurio: "Interest Rate Models - Theory and Practice" Ch. 3
//
// =============================================================================
// MODEL SPECIFICATION
// =============================================================================
//
// Short rate dynamics (original HW form):
//   dr(t) = (θ(t) - a·r(t))dt + σ(t)dW(t)
//
// State decomposition (used here): r(t) = x(t) + ψ(t)
//   dx = -a·x·dt + σ(t)·dW,  x(0) = 0
//   ψ(t) = f^mkt(0,t) + 0.5·G'(t)
//
// This decomposition ensures P^model(0,T) = P^mkt(0,T) exactly.
//
// =============================================================================
// KEY FUNCTIONS
// =============================================================================
//
// B(t,T) = (1 - e^{-a(T-t)}) / a
//   - Sensitivity of bond price to short rate
//   - For a → 0: B(t,T) → (T-t)
//
// V_r(s,t) = ∫_s^t e^{-2a(t-u)} σ(u)² du
//   - Variance of x(t) conditional on x(s)
//   - For constant σ: V_r(s,t) = σ²(1 - e^{-2a(t-s)}) / (2a)
//
// G(T) = ∫_0^T σ(u)² (1 - e^{-a(T-u)})² / a² du
//   - Core variance functional
//
// G'(T) = (2/a) ∫_0^T σ(u)² (e^{-a(T-u)} - e^{-2a(T-u)}) du
//   - Derivative of G, used in ψ(t)
//
// =============================================================================
// BOND PRICING (Affine Form)
// =============================================================================
//
// P(t,T | x_t) = A(t,T) × exp(-B(t,T) × x_t)
//
// where:
//   A(t,T) = P^mkt(0,T) / P^mkt(0,t) × exp(-0.5 × σ_P²(t,T))
//   σ_P²(t,T) = B(t,T)² × V_r(0,t)
//
// This gives exact market fit at t=0 and proper dynamics.
//
// =============================================================================
// ZCB OPTION PRICING (Black's Formula)
// =============================================================================
//
// Call on ZCB (maturity T, option expiry t, strike K):
//   C = P(0,T)·N(h) - K·P(0,t)·N(h - σ_P)
//
// where:
//   h = (1/σ_P)·ln(P(0,T)/(K·P(0,t))) + 0.5·σ_P
//   σ_P = σ_P(t,T)  (bond volatility)
//
// =============================================================================

#include "curve/discount_curve.hpp"
#include "utils/common.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace hw1f {

// =============================================================================
// HW1F Model Parameters
// =============================================================================

struct HW1FParams {
    double a;                          // Mean reversion speed
    std::vector<double> sigmaTimes;    // Sigma bucket boundaries [0, T1, T2, ..., TK]
    std::vector<double> sigmaValues;   // Sigma values for each bucket [sigma_1, ..., sigma_K]
    
    // Constructor for constant sigma
    HW1FParams(double meanReversion = 0.01, double constantSigma = 0.01)
        : a(meanReversion), sigmaTimes({0.0}), sigmaValues({constantSigma}) {}
    
    // Constructor for piecewise-constant sigma
    HW1FParams(double meanReversion, 
               const std::vector<double>& times,
               const std::vector<double>& sigmas)
        : a(meanReversion), sigmaTimes(times), sigmaValues(sigmas) {
        if (sigmaValues.size() != sigmaTimes.size()) {
            throw std::runtime_error("Sigma times and values must have same size");
        }
    }
    
    // Get sigma at time t (piecewise constant: σ(t) = σ_i for t ∈ [t_i, t_{i+1}))
    double sigma(double t) const {
        if (sigmaTimes.empty()) return 0.01;
        
        // Find the bucket containing t
        for (size_t i = sigmaTimes.size() - 1; i > 0; --i) {
            if (t >= sigmaTimes[i]) {
                return sigmaValues[i];
            }
        }
        return sigmaValues[0];
    }
    
    // Get bucket end time (for integration bounds)
    double bucketEnd(size_t i) const {
        if (i + 1 < sigmaTimes.size()) {
            return sigmaTimes[i + 1];
        }
        return 1e10;  // Last bucket extends to infinity
    }
    
    // Number of calibration parameters (a + sigma buckets)
    size_t numParams() const { return 1 + sigmaValues.size(); }
    
    // Get all parameters as a vector
    std::vector<double> toVector() const {
        std::vector<double> params;
        params.push_back(a);
        for (double s : sigmaValues) {
            params.push_back(s);
        }
        return params;
    }
    
    // Set parameters from vector
    void fromVector(const std::vector<double>& params) {
        if (params.size() < 1) return;
        a = std::max(1e-4, params[0]);  // Ensure a > 0
        for (size_t i = 0; i < sigmaValues.size() && i + 1 < params.size(); ++i) {
            sigmaValues[i] = std::max(1e-6, params[i + 1]);  // Ensure sigma > 0
        }
    }
};

// =============================================================================
// HW1F Model Core Functions
// Following the specification exactly:
//   B(t,T) = (1 - e^{-a(T-t)}) / a
//   G(T) = ∫_0^T σ(u)² (1 - e^{-a(T-u)})² / a² du
//   G'(T) = (2/a) ∫_0^T σ(u)² (e^{-a(T-u)} - e^{-2a(T-u)}) du
//   ψ(T) = f^mkt(0,T) + 0.5 * G'(T)
//   V_r(s,t) = ∫_s^t e^{-2a(t-u)} σ(u)² du
// =============================================================================

template<typename Real = double>
class HW1FModel {
public:
    explicit HW1FModel(const HW1FParams& params) : params_(params) {}
    
    // =========================================================================
    // B(t, T) = (1 - exp(-a(T-t))) / a
    // With series expansion for a → 0
    // =========================================================================
    Real B(double t, double T) const {
        using std::exp;
        double a = params_.a;
        double tau = T - t;
        
        if (std::abs(a) < 1e-8) {
            // Taylor expansion: B(t,T) ≈ τ - a*τ²/2 + a²*τ³/6 - ...
            return Real(tau);
        }
        return (Real(1.0) - exp(Real(-a * tau))) / Real(a);
    }
    
    // =========================================================================
    // I(T; s, e) = ∫_s^e (1 - e^{-a(T-u)})² du
    //            = (e-s) - (2/a)(e^{-a(T-e)} - e^{-a(T-s)}) 
    //                    + (1/2a)(e^{-2a(T-e)} - e^{-2a(T-s)})
    // =========================================================================
    double I_bucket(double T, double s, double e, double a) const {
        if (e <= s) return 0.0;
        
        if (std::abs(a) < 1e-8) {
            // Limit as a → 0: ∫_s^e (T-u)² du = (T-s)³/3 - (T-e)³/3
            double Ts = T - s;
            double Te = T - e;
            return (Ts * Ts * Ts - Te * Te * Te) / 3.0;
        }
        
        double exp_a_Te = std::exp(-a * (T - e));
        double exp_a_Ts = std::exp(-a * (T - s));
        double exp_2a_Te = std::exp(-2 * a * (T - e));
        double exp_2a_Ts = std::exp(-2 * a * (T - s));
        
        double term1 = e - s;
        double term2 = (2.0 / a) * (exp_a_Te - exp_a_Ts);
        double term3 = (1.0 / (2.0 * a)) * (exp_2a_Te - exp_2a_Ts);
        
        return term1 - term2 + term3;
    }
    
    // =========================================================================
    // G(T) = ∫_0^T σ(u)² (1 - e^{-a(T-u)})² / a² du = Σ_i σ_i²/a² * I(T; s_i, e_i)
    // This is the core variance functional from the spec
    // =========================================================================
    Real G(double T) const {
        double a = params_.a;
        if (T <= 0.0) return Real(0.0);
        
        double result = 0.0;
        
        for (size_t i = 0; i < params_.sigmaValues.size(); ++i) {
            double s = params_.sigmaTimes[i];
            double e = params_.bucketEnd(i);
            
            // Clip to [0, T]
            s = std::max(0.0, s);
            e = std::min(e, T);
            
            if (e <= s) continue;
            
            double sigma_i = params_.sigmaValues[i];
            double I_val = I_bucket(T, s, e, a);
            
            if (std::abs(a) < 1e-8) {
                result += sigma_i * sigma_i * I_val;
            } else {
                result += (sigma_i * sigma_i / (a * a)) * I_val;
            }
        }
        
        return Real(result);
    }
    
    // =========================================================================
    // G'(T) = (2/a) ∫_0^T σ(u)² (e^{-a(T-u)} - e^{-2a(T-u)}) du
    //       = Σ_i (2/a) σ_i² (J_1(T;s,e) - J_2(T;s,e))
    // where:
    //   J_1(T;s,e) = (1/a)(e^{-a(T-e)} - e^{-a(T-s)})
    //   J_2(T;s,e) = (1/2a)(e^{-2a(T-e)} - e^{-2a(T-s)})
    // =========================================================================
    Real Gprime(double T) const {
        double a = params_.a;
        if (T <= 0.0) return Real(0.0);
        
        double result = 0.0;
        
        for (size_t i = 0; i < params_.sigmaValues.size(); ++i) {
            double s = params_.sigmaTimes[i];
            double e = params_.bucketEnd(i);
            
            s = std::max(0.0, s);
            e = std::min(e, T);
            
            if (e <= s) continue;
            
            double sigma_i = params_.sigmaValues[i];
            double contribution = Gprime_bucket(T, s, e, a, sigma_i);
            result += contribution;
        }
        
        return Real(result);
    }
    
    // =========================================================================
    // G'_i(T) = (2/a) σ² (J_1 - J_2) for bucket [s, e]
    // =========================================================================
    double Gprime_bucket(double T, double s, double e, double a, double sigma) const {
        if (e <= s) return 0.0;
        
        if (std::abs(a) < 1e-8) {
            // Limit as a → 0: G' → 2 * σ² * ∫_0^T (T-u - (T-u)²/2) du (approx)
            // Actually for a→0: G'(T) → σ² * T² (from series expansion)
            // More precisely: lim_{a→0} G'(T) = σ² * T² for constant σ
            double Ts = T - s;
            double Te = T - e;
            return sigma * sigma * (Ts * Ts - Te * Te);
        }
        
        double exp_a_Te = std::exp(-a * (T - e));
        double exp_a_Ts = std::exp(-a * (T - s));
        double exp_2a_Te = std::exp(-2 * a * (T - e));
        double exp_2a_Ts = std::exp(-2 * a * (T - s));
        
        // J_1 = (1/a)(e^{-a(T-e)} - e^{-a(T-s)})
        double J1 = (1.0 / a) * (exp_a_Te - exp_a_Ts);
        
        // J_2 = (1/2a)(e^{-2a(T-e)} - e^{-2a(T-s)})
        double J2 = (1.0 / (2.0 * a)) * (exp_2a_Te - exp_2a_Ts);
        
        return (2.0 / a) * sigma * sigma * (J1 - J2);
    }
    
    // =========================================================================
    // ψ(T) = f^mkt(0,T) + 0.5 * G'(T)
    // The deterministic shift that ensures exact curve fit
    // =========================================================================
    template<typename CurveReal>
    Real psi(double T, const DiscountCurve<CurveReal>& curve) const {
        Real f_0_T = Real(value(curve.instFwd(T)));
        Real Gp = Gprime(T);
        return f_0_T + Real(0.5) * Gp;
    }
    
    // =========================================================================
    // V_r(s, t) = ∫_s^t e^{-2a(t-u)} σ(u)² du
    // This is the variance of x(t) conditional on x(s) (or unconditional if s=0)
    // Used in OU transitions and ZCB option pricing
    // =========================================================================
    Real V_r(double s, double t) const {
        double a = params_.a;
        if (t <= s) return Real(0.0);
        
        double result = 0.0;
        
        for (size_t i = 0; i < params_.sigmaValues.size(); ++i) {
            double t_start = params_.sigmaTimes[i];
            double t_end = params_.bucketEnd(i);
            
            // Clip to [s, t]
            t_start = std::max(t_start, s);
            t_end = std::min(t_end, t);
            
            if (t_end <= t_start) continue;
            
            double sigma_i = params_.sigmaValues[i];
            
            if (std::abs(a) < 1e-8) {
                // Limit: ∫ σ² du = σ² * (t_end - t_start)
                result += sigma_i * sigma_i * (t_end - t_start);
            } else {
                // ∫_{t_start}^{t_end} e^{-2a(t-u)} du 
                //   = (1/2a)(e^{-2a(t-t_end)} - e^{-2a(t-t_start)})
                double exp_2a_end = std::exp(-2 * a * (t - t_end));
                double exp_2a_start = std::exp(-2 * a * (t - t_start));
                double integral = (exp_2a_end - exp_2a_start) / (2.0 * a);
                result += sigma_i * sigma_i * integral;
            }
        }
        
        return Real(result);
    }
    
    // =========================================================================
    // σ_P²(t, T) = Variance of ln(P(t,T)) viewed from time 0
    // 
    // For constant σ (standard formula):
    // σ_P²(t,T) = (σ²/2a³) × (1 - e^{-a(T-t)})² × (1 - e^{-2at})
    //           = (σ/a)² × B(t,T)² × (1 - e^{-2at}) / (2a)
    //           = B(t,T)² × V_r(0,t)
    //
    // For piecewise-constant σ(u), we need:
    // σ_P²(t,T) = B(t,T)² × V_r(0,t)
    //
    // This is the covariance formula from Brigo & Mercurio Ch. 3
    // The variance of x(t) drives the variance of the bond price
    // =========================================================================
    Real sigmaP(double t, double T) const {
        using std::sqrt;
        
        if (t <= 0.0) return Real(0.0);
        if (T <= t) return Real(0.0);
        
        Real B_t_T = B(t, T);
        Real V = V_r(0.0, t);
        
        return B_t_T * sqrt(V);
    }
    
    // =========================================================================
    // A(t, T) for bond pricing: P(t, T | x_t) = A(t,T) * exp(-B(t,T) * x_t)
    // From the affine form, A ensures model matches market at t=0
    // 
    // Per PDFs (OpenGamma, Mizuho): The correct formula is:
    // A(t,T) = P^mkt(0,T) / P^mkt(0,t) * exp(B(t,T)*φ(t) - 0.5*σ_P²(t,T))
    // 
    // where φ(t) = f^mkt(0,t) + 0.5*V_r(0,t)/a² [for state r, not x]
    // 
    // But in the x-formulation (r = x + ψ, x(0)=0):
    // A(t,T) = P^mkt(0,T) / P^mkt(0,t) * exp(-0.5 * σ_P²(t,T))
    // This is the correct form for the state variable decomposition
    // =========================================================================
    template<typename CurveReal>
    Real A(double t, double T, const DiscountCurve<CurveReal>& curve) const {
        using std::exp;
        
        CurveReal P_0_T = curve.df(T);
        CurveReal P_0_t = curve.df(t);
        Real ratio = Real(value(P_0_T) / value(P_0_t));
        
        Real sigP = sigmaP(t, T);
        
        return ratio * exp(Real(-0.5) * sigP * sigP);
    }
    
    // =========================================================================
    // Bond price P(t, T | x_t) in HW1F model
    // P(t, T | x_t) = A(t,T) * exp(-B(t,T) * x_t)
    // =========================================================================
    template<typename CurveReal>
    Real bondPrice(double t, double T, Real x_t, const DiscountCurve<CurveReal>& curve) const {
        using std::exp;
        Real A_t_T = A(t, T, curve);
        Real B_t_T = B(t, T);
        return A_t_T * exp(-B_t_T * x_t);
    }
    
    // =========================================================================
    // Discount factor from 0 to T (matches market exactly by construction)
    // =========================================================================
    template<typename CurveReal>
    Real P_0_T(double T, const DiscountCurve<CurveReal>& curve) const {
        return Real(value(curve.df(T)));
    }
    
    // =========================================================================
    // Optional: θ(t) = ψ'(t) + a*ψ(t) for the original HW drift form
    // dr = (θ(t) - a*r) dt + σ(t) dW
    // Usually not needed if using affine bond formulas directly
    // =========================================================================
    template<typename CurveReal>
    Real theta(double t, const DiscountCurve<CurveReal>& curve, double dt = 1e-6) const {
        // Numerical derivative of psi
        Real psi_t = psi(t, curve);
        Real psi_tdt = psi(t + dt, curve);
        Real psi_prime = (psi_tdt - psi_t) / Real(dt);
        
        return psi_prime + Real(params_.a) * psi_t;
    }
    
    // Mean reversion
    double meanReversion() const { return params_.a; }
    
    // Get sigma at time t
    double sigma(double t) const { return params_.sigma(t); }
    
    // Access parameters
    const HW1FParams& params() const { return params_; }
    HW1FParams& params() { return params_; }

private:
    HW1FParams params_;
};

// =============================================================================
// Zero-Coupon Bond Option Pricing (HW1F)
// =============================================================================

// Price of a call option on a ZCB maturing at T, option expiry at t
// ZCB pays 1 at T, strike is K
template<typename Real, typename CurveReal>
Real zcbCallPrice(
    double t,              // Option expiry
    double T,              // Bond maturity
    double K,              // Strike
    const HW1FModel<Real>& model,
    const DiscountCurve<CurveReal>& curve
) {
    using std::sqrt;
    using std::log;
    
    if (t <= 0.0) {
        Real P_T = model.P_0_T(T, curve);
        Real intrinsic = P_T - Real(K);
        return intrinsic > Real(0.0) ? intrinsic : Real(0.0);
    }
    
    Real P_t = model.P_0_T(t, curve);
    Real P_T = model.P_0_T(T, curve);
    Real sigP = model.sigmaP(t, T);
    
    if (value(sigP) < 1e-12) {
        // Degenerate case
        Real intrinsic = P_T - Real(K) * P_t;
        return intrinsic > Real(0.0) ? intrinsic : Real(0.0);
    }
    
    Real h = (Real(1.0) / sigP) * log(P_T / (Real(K) * P_t)) + Real(0.5) * sigP;
    
    Real Nh = Real(normalCDF(value(h)));
    Real Nhm = Real(normalCDF(value(h) - value(sigP)));
    
    return P_T * Nh - Real(K) * P_t * Nhm;
}

// Price of a put option on a ZCB
template<typename Real, typename CurveReal>
Real zcbPutPrice(
    double t,
    double T,
    double K,
    const HW1FModel<Real>& model,
    const DiscountCurve<CurveReal>& curve
) {
    using std::sqrt;
    using std::log;
    
    if (t <= 0.0) {
        Real P_T = model.P_0_T(T, curve);
        Real intrinsic = Real(K) - P_T;
        return intrinsic > Real(0.0) ? intrinsic : Real(0.0);
    }
    
    Real P_t = model.P_0_T(t, curve);
    Real P_T = model.P_0_T(T, curve);
    Real sigP = model.sigmaP(t, T);
    
    if (value(sigP) < 1e-12) {
        Real intrinsic = Real(K) * P_t - P_T;
        return intrinsic > Real(0.0) ? intrinsic : Real(0.0);
    }
    
    Real h = (Real(1.0) / sigP) * log(P_T / (Real(K) * P_t)) + Real(0.5) * sigP;
    
    Real Nmh = Real(normalCDF(-value(h)));
    Real Nmhm = Real(normalCDF(-(value(h) - value(sigP))));
    
    return Real(K) * P_t * Nmhm - P_T * Nmh;
}

} // namespace hw1f
