#pragma once
// =============================================================================
// Jamshidian Swaption Pricing (Analytic HW1F)
// Decomposes swaption into sum of ZCB options
//
// References:
// - OpenGamma: "Algorithmic Differentiation in Finance" (Henrard)
// - Jamshidian (1989): "An Exact Bond Option Formula"
// - Brigo & Mercurio: Ch. 3.3 Swaption Pricing
//
// Key insight (Jamshidian decomposition):
// A payer swaption with payoff max(N - Σc_i·P(E,T_i), 0) can be written as
// a portfolio of put options on ZCBs when there exists x* such that
// Σc_i·P(E,T_i|x*) = N.
//
// Then: Payer Swaption = Σc_i·Put(P(E,T_i), K_i=P(E,T_i|x*))
//       Receiver Swaption = Σc_i·Call(P(E,T_i), K_i)
//
// This decomposition is exact for one-factor models like HW1F.
// =============================================================================

#include "hw1f/hw1f_model.hpp"
#include "instruments/swaption.hpp"
#include <vector>
#include <cmath>
#include <functional>
#include <limits>

namespace hw1f {

// =============================================================================
// Brent's Root Finding Algorithm
// =============================================================================

template<typename Func>
double brentSolve(
    Func f,
    double a,
    double b,
    double tol = 1e-10,
    int maxIter = 100
) {
    double fa = f(a);
    double fb = f(b);
    
    if (fa * fb > 0) {
        // Try to bracket the root
        double mid = 0.5 * (a + b);
        double fmid = f(mid);
        if (fa * fmid < 0) {
            b = mid; fb = fmid;
        } else if (fmid * fb < 0) {
            a = mid; fa = fmid;
        } else {
            // No sign change, return midpoint
            return mid;
        }
    }
    
    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }
    
    double c = a, fc = fa;
    bool mflag = true;
    double s = 0, d = 0;
    
    for (int i = 0; i < maxIter; ++i) {
        if (std::abs(b - a) < tol) break;
        if (std::abs(fb) < tol) return b;
        
        if (std::abs(fa - fc) > tol && std::abs(fb - fc) > tol) {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc))
              + b * fa * fc / ((fb - fa) * (fb - fc))
              + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }
        
        // Conditions for bisection
        double cond1 = (s < (3 * a + b) / 4 || s > b);
        double cond2 = mflag && std::abs(s - b) >= std::abs(b - c) / 2;
        double cond3 = !mflag && std::abs(s - b) >= std::abs(c - d) / 2;
        double cond4 = mflag && std::abs(b - c) < tol;
        double cond5 = !mflag && std::abs(c - d) < tol;
        
        if (cond1 || cond2 || cond3 || cond4 || cond5) {
            s = (a + b) / 2;
            mflag = true;
        } else {
            mflag = false;
        }
        
        double fs = f(s);
        d = c;
        c = b;
        fc = fb;
        
        if (fa * fs < 0) {
            b = s; fb = fs;
        } else {
            a = s; fa = fs;
        }
        
        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
    }
    
    return b;
}

// =============================================================================
// Jamshidian Decomposition
// =============================================================================

template<typename Real, typename CurveReal>
class JamshidianPricer {
public:
    JamshidianPricer(
        const HW1FModel<Real>& model,
        const DiscountCurve<CurveReal>& curve
    ) : model_(model), curve_(curve) {}
    
    // Price European swaption using Jamshidian decomposition
    Real price(const EuropeanSwaption& swaption) const {
        double expiry = swaption.expiry;
        const VanillaSwap& swap = swaption.underlying;
        Schedule fixedSched = swap.fixedSchedule();
        
        size_t n = fixedSched.paymentDates.size();
        if (n == 0) return Real(0.0);
        
        // Fixed leg cashflows: alpha_i * K * N at each T_i, plus N at T_n
        std::vector<double> cashflows(n);
        std::vector<double> payDates(n);
        for (size_t i = 0; i < n; ++i) {
            cashflows[i] = fixedSched.accrualFactors[i] * swap.fixedRate * swap.notional;
            payDates[i] = fixedSched.paymentDates[i];
        }
        // Add principal at last payment
        cashflows[n - 1] += swap.notional;
        
        // Find x* such that sum of cashflows * P(E, Ti | x*) = N
        // Jamshidian decomposition: at x*, the payer swap has zero value
        // Payer swap value = N - sum(c_i * P(E, Ti))
        // where c_i = alpha_i * K * N for i < n, and c_{n-1} += N
        
        auto swapPVatExpiry = [&](double x) -> double {
            double pv_fixed = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double P_E_Ti = value(model_.bondPrice(expiry, payDates[i], Real(x), curve_));
                pv_fixed += cashflows[i] * P_E_Ti;
            }
            // Payer swap PV = N - sum(c_i * P(E, Ti))
            return swap.notional - pv_fixed;
        };
        
        // Bracket x*
        double x_star = brentSolve(swapPVatExpiry, -0.5, 0.5, 1e-10, 100);
        
        // Compute strikes K_i = P(E, T_i | x*) for each payment date
        std::vector<Real> strikes(n);
        Real x_star_r = Real(x_star);
        for (size_t i = 0; i < n; ++i) {
            strikes[i] = model_.bondPrice(expiry, payDates[i], x_star_r, curve_);
        }
        
        // Swaption price = sum of ZCB option prices
        Real price = Real(0.0);
        
        if (swap.isPayer) {
            // Payer swaption = sum of puts on ZCBs with strikes K_i
            // Payoff = max(N - sum(c_i * P(E, Ti)), 0)
            // = sum of c_i * max(K_i - P(E, Ti), 0) = sum of c_i * put(K_i)
            for (size_t i = 0; i < n; ++i) {
                Real putPrice = zcbPutPrice(expiry, payDates[i], value(strikes[i]), model_, curve_);
                price = price + Real(cashflows[i]) * putPrice;
            }
        } else {
            // Receiver swaption = sum of calls on ZCBs
            for (size_t i = 0; i < n; ++i) {
                Real callPrice = zcbCallPrice(expiry, payDates[i], value(strikes[i]), model_, curve_);
                price = price + Real(cashflows[i]) * callPrice;
            }
        }
        
        return price;
    }
    
    // Price using simpler method (for validation)
    Real priceSimple(const EuropeanSwaption& swaption) const {
        double expiry = swaption.expiry;
        const VanillaSwap& swap = swaption.underlying;
        Schedule fixedSched = swap.fixedSchedule();
        
        size_t n = fixedSched.paymentDates.size();
        if (n == 0) return Real(0.0);
        
        // Compute annuity and forward swap rate
        Real annuity = Real(0.0);
        Real lastDf = Real(0.0);
        for (size_t i = 0; i < n; ++i) {
            Real df = model_.P_0_T(fixedSched.paymentDates[i], curve_);
            annuity = annuity + Real(fixedSched.accrualFactors[i]) * df;
            if (i == n - 1) lastDf = df;
        }
        
        Real dfStart = model_.P_0_T(swap.startDate, curve_);
        Real fwdRate = (dfStart - lastDf) / annuity;
        
        // Use approximate swaption vol from HW parameters
        Real hwSigma = Real(model_.sigma(expiry));
        double B_sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            B_sum += value(model_.B(expiry, fixedSched.paymentDates[i])) * fixedSched.accrualFactors[i];
        }
        
        // Approximate vol of swap rate
        double swapVol = model_.sigma(expiry) * B_sum / value(annuity / dfStart) * std::sqrt(expiry);
        
        // Black price
        Real strike = Real(swap.fixedRate);
        if (swap.isPayer) {
            return black76Call(fwdRate, strike, Real(swapVol / std::sqrt(expiry)), expiry, annuity) * swap.notional;
        } else {
            return black76Put(fwdRate, strike, Real(swapVol / std::sqrt(expiry)), expiry, annuity) * swap.notional;
        }
    }

private:
    const HW1FModel<Real>& model_;
    const DiscountCurve<CurveReal>& curve_;
};

// Convenience function
template<typename Real, typename CurveReal>
Real priceJamshidian(
    const EuropeanSwaption& swaption,
    const DiscountCurve<CurveReal>& curve,
    const HW1FParams& params
) {
    HW1FModel<Real> model(params);
    JamshidianPricer<Real, CurveReal> pricer(model, curve);
    return pricer.price(swaption);
}

} // namespace hw1f
