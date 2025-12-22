#pragma once
// =============================================================================
// Black Swaption Vega and Greeks
//
// Provides analytical Black-76 Greeks for calibration instruments.
// Key formulas:
//
// Black-76 Call: C = A * N * (F * N(d1) - K * N(d2))
// where:
//   F = forward swap rate
//   K = strike
//   A = annuity (PV01)
//   N = notional
//   σ = Black volatility
//   T = option expiry
//   d1 = (ln(F/K) + 0.5σ²T) / (σ√T)
//   d2 = d1 - σ√T
//
// Vega: ∂C/∂σ = A * N * F * √T * n(d1)
//   where n(x) = standard normal PDF
//
// For ATM (K = F): d1 = d2 + σ√T = 0.5σ√T
//   Vega_ATM = A * N * F * √T * n(0.5σ√T)
//
// =============================================================================

#include "curve/discount_curve.hpp"
#include "instruments/swaption.hpp"
#include "utils/common.hpp"
#include <cmath>

namespace hw1f {

// =============================================================================
// Black Vega (∂Price/∂σ)
// =============================================================================

template<typename Real>
Real black76Vega(
    Real forward,
    Real strike,
    Real vol,
    double T,
    Real annuity,
    double notional = 1.0
) {
    using std::sqrt;
    using std::log;
    
    if (T <= 0.0) return Real(0.0);
    if (value(vol) < 1e-10) return Real(0.0);
    
    double sqrtT = sqrt(T);
    Real d1 = (log(forward / strike) + Real(0.5) * vol * vol * Real(T)) / (vol * Real(sqrtT));
    
    // n(d1) = standard normal PDF at d1
    double d1_val = value(d1);
    double nd1 = normalPDF(d1_val);
    
    // Vega = A * N * F * √T * n(d1)
    return annuity * Real(notional) * forward * Real(sqrtT) * Real(nd1);
}

// ATM Black Vega (simplified formula when K = F)
template<typename Real>
Real black76VegaATM(
    Real forward,
    Real vol,
    double T,
    Real annuity,
    double notional = 1.0
) {
    using std::sqrt;
    
    if (T <= 0.0) return Real(0.0);
    if (value(vol) < 1e-10) return Real(0.0);
    
    double sqrtT = sqrt(T);
    
    // For ATM: d1 = 0.5 * σ * √T
    double d1 = 0.5 * value(vol) * sqrtT;
    double nd1 = normalPDF(d1);
    
    // Vega_ATM = A * N * F * √T * n(d1)
    return annuity * Real(notional) * forward * Real(sqrtT) * Real(nd1);
}

// =============================================================================
// Black Delta (∂Price/∂F)
// =============================================================================

template<typename Real>
Real black76Delta(
    Real forward,
    Real strike,
    Real vol,
    double T,
    Real annuity,
    double notional = 1.0,
    bool isCall = true
) {
    using std::sqrt;
    using std::log;
    
    if (T <= 0.0) {
        if (isCall) {
            return value(forward) > value(strike) ? annuity * Real(notional) : Real(0.0);
        } else {
            return value(strike) > value(forward) ? -annuity * Real(notional) : Real(0.0);
        }
    }
    
    double sqrtT = sqrt(T);
    Real d1 = (log(forward / strike) + Real(0.5) * vol * vol * Real(T)) / (vol * Real(sqrtT));
    
    double d1_val = value(d1);
    double Nd1 = normalCDF(d1_val);
    
    if (isCall) {
        return annuity * Real(notional) * Real(Nd1);
    } else {
        return -annuity * Real(notional) * Real(1.0 - Nd1);
    }
}

// =============================================================================
// Full Black Greeks for Swaption
// =============================================================================

struct BlackGreeks {
    double price;
    double delta;    // ∂V/∂F (forward delta)
    double vega;     // ∂V/∂σ (vol sensitivity)
    double gamma;    // ∂²V/∂F² 
    double theta;    // ∂V/∂T (time decay, approximated)
};

template<typename Real>
BlackGreeks blackSwaptionGreeks(
    const EuropeanSwaption& swaption,
    const DiscountCurve<Real>& curve,
    double vol
) {
    BlackGreeks greeks;
    
    Real forward = forwardSwapRate(swaption.underlying, curve);
    Real annuity = swapAnnuity(swaption.underlying, curve);
    Real strike = Real(swaption.underlying.fixedRate);
    double T = swaption.expiry;
    double notional = swaption.underlying.notional;
    bool isCall = swaption.underlying.isPayer;
    
    // Price
    if (isCall) {
        greeks.price = value(black76Call(forward, strike, Real(vol), T, annuity) * Real(notional));
    } else {
        greeks.price = value(black76Put(forward, strike, Real(vol), T, annuity) * Real(notional));
    }
    
    // Vega
    greeks.vega = value(black76Vega(forward, strike, Real(vol), T, annuity, notional));
    
    // Delta
    greeks.delta = value(black76Delta(forward, strike, Real(vol), T, annuity, notional, isCall));
    
    // Gamma (second derivative)
    double sqrtT = std::sqrt(T);
    if (T > 0.0 && vol > 1e-10) {
        double d1 = (std::log(value(forward) / value(strike)) + 0.5 * vol * vol * T) / (vol * sqrtT);
        double nd1 = normalPDF(d1);
        greeks.gamma = value(annuity) * notional * nd1 / (value(forward) * vol * sqrtT);
    } else {
        greeks.gamma = 0.0;
    }
    
    // Theta (simple approximation: -0.5 * σ² * S² * gamma for at-the-money)
    greeks.theta = -0.5 * vol * vol * value(forward) * value(forward) * greeks.gamma;
    
    return greeks;
}

// =============================================================================
// Compute Black vega for a calibration instrument
// =============================================================================

template<typename Real>
double computeCalibrationVega(
    double expiry,
    double tenor,
    double vol,
    const DiscountCurve<Real>& curve,
    double notional
) {
    EuropeanSwaption swaption(expiry, tenor, 0.0, notional, true);
    Real fwd = forwardSwapRate(swaption.underlying, curve);
    swaption.underlying.fixedRate = value(fwd);  // ATM
    
    Real annuity = swapAnnuity(swaption.underlying, curve);
    
    return value(black76VegaATM(fwd, Real(vol), expiry, annuity, notional));
}

// =============================================================================
// V_Θ_direct: Direct price sensitivity to vol surface nodes
//
// For standard swaptions calibrated to Black prices, the exotic price
// typically does NOT depend directly on the Black vols (only indirectly
// through the calibrated HW parameters).
//
// However, if the exotic itself uses Black vols for any pricing component
// (e.g., volatility adjustment), this term would be non-zero.
//
// For most HW1F setups: V_Θ_direct = 0
// This function returns 0 but documents the mathematical completeness.
// =============================================================================

inline double computeVThetaDirect_HW1F(
    const EuropeanSwaption& /* exotic */,
    size_t /* volNodeExpiryIdx */,
    size_t /* volNodeTenorIdx */
) {
    // For HW1F: Exotic price depends on HW params, not directly on Black vols
    // The Black vols are market data used only in calibration
    return 0.0;
}

} // namespace hw1f
