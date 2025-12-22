#pragma once
// =============================================================================
// Financial Instruments: Vanilla Swap and European Swaption
// =============================================================================

#include "curve/discount_curve.hpp"
#include <vector>
#include <cmath>

namespace hw1f {

// =============================================================================
// Vanilla Interest Rate Swap
// =============================================================================

struct VanillaSwap {
    double startDate;      // Swap start date
    double endDate;        // Swap end date
    double fixedRate;      // Fixed rate K
    double notional;       // Notional amount
    bool isPayer;          // true = payer (pay fixed, receive float)
    int fixedFreq;         // Fixed leg frequency (1 = annual)
    int floatFreq;         // Float leg frequency (4 = quarterly)
    
    VanillaSwap(
        double start, double end, double K, double N = 1e6,
        bool payer = true, int fixFreq = 1, int floatFreq = 4
    ) : startDate(start), endDate(end), fixedRate(K), notional(N),
        isPayer(payer), fixedFreq(fixFreq), floatFreq(floatFreq) {}
    
    // Get fixed leg payment dates
    Schedule fixedSchedule() const {
        return generateFixedSchedule(startDate, endDate, fixedFreq);
    }
    
    // Get float leg schedule
    Schedule floatSchedule() const {
        return generateFloatSchedule(startDate, endDate, floatFreq);
    }
    
    // Swap tenor in years
    double tenor() const { return endDate - startDate; }
};

// =============================================================================
// Swap Valuation Functions
// =============================================================================

// Compute swap annuity (PV01): A(0) = sum(alpha_i * P(0, T_i))
template<typename Real>
Real swapAnnuity(const VanillaSwap& swap, const DiscountCurve<Real>& curve) {
    Schedule sched = swap.fixedSchedule();
    Real annuity = Real(0.0);
    for (size_t i = 0; i < sched.paymentDates.size(); ++i) {
        annuity = annuity + sched.accrualFactors[i] * curve.df(sched.paymentDates[i]);
    }
    return annuity;
}

// Compute forward swap rate: S(0) = (P(0, T_start) - P(0, T_end)) / A(0)
template<typename Real>
Real forwardSwapRate(const VanillaSwap& swap, const DiscountCurve<Real>& curve) {
    Real dfStart = curve.df(swap.startDate);
    Real dfEnd = curve.df(swap.endDate);
    Real annuity = swapAnnuity(swap, curve);
    return (dfStart - dfEnd) / annuity;
}

// Compute swap PV at time 0
template<typename Real>
Real swapPV(const VanillaSwap& swap, const DiscountCurve<Real>& curve) {
    Real fwdRate = forwardSwapRate(swap, curve);
    Real annuity = swapAnnuity(swap, curve);
    Real pv = (fwdRate - swap.fixedRate) * annuity * swap.notional;
    return swap.isPayer ? pv : -pv;
}

// =============================================================================
// European Swaption
// =============================================================================

struct EuropeanSwaption {
    double expiry;         // Option expiry
    VanillaSwap underlying; // Underlying swap (starts at expiry)
    
    EuropeanSwaption(double exp, const VanillaSwap& swap)
        : expiry(exp), underlying(swap) {
        // Swap should start at expiry
        if (std::abs(underlying.startDate - expiry) > 1e-8) {
            // Adjust swap start date to match expiry
            double tenor = underlying.tenor();
            underlying.startDate = expiry;
            underlying.endDate = expiry + tenor;
        }
    }
    
    // Convenience constructor
    EuropeanSwaption(double exp, double tenor, double strike, double notional = 1e6, bool isPayer = true)
        : expiry(exp), underlying(exp, exp + tenor, strike, notional, isPayer) {}
    
    // Get swap tenor
    double swapTenor() const { return underlying.tenor(); }
};

// =============================================================================
// Black-76 Swaption Pricing (for reference and calibration targets)
// =============================================================================

template<typename Real>
Real black76Call(Real forward, Real strike, Real vol, double T, Real annuity) {
    using std::sqrt;
    using std::log;
    
    if (T <= 0.0) {
        Real intrinsic = forward - strike;
        return annuity * (intrinsic > Real(0.0) ? intrinsic : Real(0.0));
    }
    
    Real sqrtT = sqrt(T);
    Real d1 = (log(forward / strike) + Real(0.5) * vol * vol * T) / (vol * sqrtT);
    Real d2 = d1 - vol * sqrtT;
    
    Real Nd1 = Real(normalCDF(value(d1)));
    Real Nd2 = Real(normalCDF(value(d2)));
    
    return annuity * (forward * Nd1 - strike * Nd2);
}

template<typename Real>
Real black76Put(Real forward, Real strike, Real vol, double T, Real annuity) {
    using std::sqrt;
    using std::log;
    
    if (T <= 0.0) {
        Real intrinsic = strike - forward;
        return annuity * (intrinsic > Real(0.0) ? intrinsic : Real(0.0));
    }
    
    Real sqrtT = sqrt(T);
    Real d1 = (log(forward / strike) + Real(0.5) * vol * vol * T) / (vol * sqrtT);
    Real d2 = d1 - vol * sqrtT;
    
    Real Nmd1 = Real(normalCDF(-value(d1)));
    Real Nmd2 = Real(normalCDF(-value(d2)));
    
    return annuity * (strike * Nmd2 - forward * Nmd1);
}

// Price swaption using Black-76
template<typename Real>
Real blackSwaptionPrice(
    const EuropeanSwaption& swaption,
    const DiscountCurve<Real>& curve,
    Real vol
) {
    Real forward = forwardSwapRate(swaption.underlying, curve);
    Real annuity = swapAnnuity(swaption.underlying, curve);
    Real strike = Real(swaption.underlying.fixedRate);
    
    if (swaption.underlying.isPayer) {
        return black76Call(forward, strike, vol, swaption.expiry, annuity) * swaption.underlying.notional;
    } else {
        return black76Put(forward, strike, vol, swaption.expiry, annuity) * swaption.underlying.notional;
    }
}

// Price ATM swaption (strike = forward rate)
template<typename Real>
Real blackATMSwaptionPrice(
    double expiry,
    double tenor,
    const DiscountCurve<Real>& curve,
    Real vol,
    double notional = 1.0,
    bool isPayer = true
) {
    VanillaSwap swap(expiry, expiry + tenor, 0.0, notional, isPayer);
    Real fwdRate = forwardSwapRate(swap, curve);
    swap.fixedRate = value(fwdRate);  // ATM
    
    EuropeanSwaption swaption(expiry, swap);
    return blackSwaptionPrice(swaption, curve, vol);
}

} // namespace hw1f
