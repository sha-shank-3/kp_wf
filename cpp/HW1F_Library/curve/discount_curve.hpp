#pragma once
// =============================================================================
// Discount Curve Implementation
// Log-linear interpolation on discount factors
// Templated for AD support
// =============================================================================

#include "utils/common.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>

namespace hw1f {

// =============================================================================
// DiscountCurve Class
// =============================================================================

template<typename Real = double>
class DiscountCurve {
public:
    // Constructor from times and discount factors
    DiscountCurve(
        const std::vector<double>& times,
        const std::vector<Real>& discountFactors
    ) : times_(times), dfs_(discountFactors) {
        if (times_.size() != dfs_.size()) {
            throw std::runtime_error("Times and discount factors must have same size");
        }
        if (times_.empty()) {
            throw std::runtime_error("Curve must have at least one point");
        }
        // Ensure times are sorted
        for (size_t i = 1; i < times_.size(); ++i) {
            if (times_[i] <= times_[i-1]) {
                throw std::runtime_error("Times must be strictly increasing");
            }
        }
        // Compute log discount factors for log-linear interpolation
        logDfs_.resize(dfs_.size());
        for (size_t i = 0; i < dfs_.size(); ++i) {
            logDfs_[i] = log(dfs_[i]);
        }
    }
    
    // Constructor from times and zero rates
    static DiscountCurve fromZeroRates(
        const std::vector<double>& times,
        const std::vector<Real>& zeroRates
    ) {
        std::vector<Real> dfs(times.size());
        for (size_t i = 0; i < times.size(); ++i) {
            dfs[i] = exp(-zeroRates[i] * times[i]);
        }
        return DiscountCurve(times, dfs);
    }
    
    // Get discount factor at time T (log-linear interpolation)
    Real df(double T) const {
        using std::exp;
        using std::log;
        
        if (T <= 0.0) return Real(1.0);
        if (T <= times_.front()) {
            // Flat extrapolation before first point
            Real r = -logDfs_.front() / times_.front();
            return exp(-r * T);
        }
        if (T >= times_.back()) {
            // Flat rate extrapolation after last point
            Real r = -logDfs_.back() / times_.back();
            return exp(-r * T);
        }
        
        // Log-linear interpolation
        size_t i = findIndex(T, times_);
        double t1 = times_[i];
        double t2 = times_[i + 1];
        double w = (T - t1) / (t2 - t1);
        
        Real logDf = logDfs_[i] * (1.0 - w) + logDfs_[i + 1] * w;
        return exp(logDf);
    }
    
    // Get zero rate at time T
    Real zeroRate(double T) const {
        using std::log;
        if (T <= 0.0) return Real(0.0);
        return -log(df(T)) / T;
    }
    
    // Forward rate from t1 to t2 (simple compounding)
    Real fwdRate(double t1, double t2) const {
        if (t2 <= t1) return Real(0.0);
        Real df1 = df(t1);
        Real df2 = df(t2);
        return (df1 / df2 - Real(1.0)) / (t2 - t1);
    }
    
    // Instantaneous forward rate at time t
    // Uses finite difference or can be made AD-compatible
    Real instFwd(double t) const {
        const double h = 1e-5;
        if (t < h) {
            return -log(df(h)) / h;
        }
        Real df_t = df(t);
        Real df_th = df(t + h);
        return -log(df_th / df_t) / h;
    }
    
    // Get node times
    const std::vector<double>& times() const { return times_; }
    
    // Get discount factors at nodes
    const std::vector<Real>& discountFactors() const { return dfs_; }
    
    // Number of nodes
    size_t size() const { return times_.size(); }
    
    // Get time at node index
    double timeAt(size_t i) const { return times_[i]; }
    
    // Get DF at node index (for AD input registration)
    Real dfAt(size_t i) const { return dfs_[i]; }
    
    // Create a bumped curve (for FD sensitivities)
    DiscountCurve bump(size_t nodeIndex, double bumpSize) const {
        if (nodeIndex >= dfs_.size()) {
            throw std::runtime_error("Node index out of range");
        }
        std::vector<Real> bumpedDfs = dfs_;
        // Bump the zero rate at this node
        double t = times_[nodeIndex];
        Real r = -logDfs_[nodeIndex] / t;
        Real newR = r + bumpSize;
        bumpedDfs[nodeIndex] = exp(-newR * t);
        return DiscountCurve(times_, bumpedDfs);
    }
    
    // Get mutable reference to DFs (for AD)
    std::vector<Real>& mutableDfs() { return dfs_; }
    std::vector<Real>& mutableLogDfs() { return logDfs_; }
    
    // Update log DFs after modifying DFs
    void updateLogDfs() {
        for (size_t i = 0; i < dfs_.size(); ++i) {
            logDfs_[i] = log(dfs_[i]);
        }
    }

private:
    std::vector<double> times_;
    std::vector<Real> dfs_;
    std::vector<Real> logDfs_;
};

// =============================================================================
// ATM Vol Surface
// Bilinear interpolation in expiry/tenor
// =============================================================================

template<typename Real = double>
class ATMVolSurface {
public:
    ATMVolSurface(
        const std::vector<double>& expiries,
        const std::vector<double>& tenors,
        const std::vector<std::vector<Real>>& vols
    ) : expiries_(expiries), tenors_(tenors), vols_(vols) {
        if (expiries_.empty() || tenors_.empty()) {
            throw std::runtime_error("Vol surface must have at least one expiry and tenor");
        }
        if (vols_.size() != expiries_.size()) {
            throw std::runtime_error("Vol matrix rows must match number of expiries");
        }
        for (const auto& row : vols_) {
            if (row.size() != tenors_.size()) {
                throw std::runtime_error("Vol matrix columns must match number of tenors");
            }
        }
    }
    
    // Get ATM vol at (expiry, tenor) using bilinear interpolation
    Real atmVol(double expiry, double tenor) const {
        // Find expiry indices
        size_t ei = findIndex(expiry, expiries_);
        size_t ti = findIndex(tenor, tenors_);
        
        // Clamp to valid range
        ei = std::min(ei, expiries_.size() - 2);
        ti = std::min(ti, tenors_.size() - 2);
        
        double e1 = expiries_[ei], e2 = expiries_[ei + 1];
        double t1 = tenors_[ti], t2 = tenors_[ti + 1];
        
        double we = (expiry - e1) / (e2 - e1);
        double wt = (tenor - t1) / (t2 - t1);
        
        we = std::clamp(we, 0.0, 1.0);
        wt = std::clamp(wt, 0.0, 1.0);
        
        // Bilinear interpolation
        Real v00 = vols_[ei][ti];
        Real v01 = vols_[ei][ti + 1];
        Real v10 = vols_[ei + 1][ti];
        Real v11 = vols_[ei + 1][ti + 1];
        
        Real v0 = v00 * (1.0 - wt) + v01 * wt;
        Real v1 = v10 * (1.0 - wt) + v11 * wt;
        
        return v0 * (1.0 - we) + v1 * we;
    }
    
    // Get exact node vol
    Real nodeVol(size_t expiryIdx, size_t tenorIdx) const {
        return vols_[expiryIdx][tenorIdx];
    }
    
    // Accessors
    const std::vector<double>& expiries() const { return expiries_; }
    const std::vector<double>& tenors() const { return tenors_; }
    const std::vector<std::vector<Real>>& vols() const { return vols_; }
    
    // Single element accessors (for XAD indexing)
    double expiry(size_t i) const { return expiries_[i]; }
    double tenor(size_t i) const { return tenors_[i]; }
    Real vol(size_t ei, size_t ti) const { return vols_[ei][ti]; }
    
    size_t numExpiries() const { return expiries_.size(); }
    size_t numTenors() const { return tenors_.size(); }
    size_t numNodes() const { return expiries_.size() * tenors_.size(); }
    
    // Create bumped surface
    ATMVolSurface bump(size_t expiryIdx, size_t tenorIdx, double bumpSize) const {
        auto bumpedVols = vols_;
        bumpedVols[expiryIdx][tenorIdx] = bumpedVols[expiryIdx][tenorIdx] + bumpSize;
        return ATMVolSurface(expiries_, tenors_, bumpedVols);
    }
    
    // Mutable access for AD
    std::vector<std::vector<Real>>& mutableVols() { return vols_; }

private:
    std::vector<double> expiries_;
    std::vector<double> tenors_;
    std::vector<std::vector<Real>> vols_;
};

// =============================================================================
// Schedule Generation
// =============================================================================

struct Schedule {
    std::vector<double> paymentDates;  // Payment times
    std::vector<double> accrualFactors;  // Day count fractions
    std::vector<double> resetDates;  // Fixing dates (for floating leg)
};

// Generate annual fixed leg schedule
inline Schedule generateFixedSchedule(
    double startDate,
    double endDate,
    int frequency = 1  // 1 = annual, 2 = semi-annual
) {
    Schedule sched;
    double dt = 1.0 / frequency;
    double t = startDate + dt;
    while (t <= endDate + 1e-8) {
        sched.paymentDates.push_back(t);
        sched.accrualFactors.push_back(dt);
        t += dt;
    }
    return sched;
}

// Generate floating leg schedule (quarterly, 3M)
inline Schedule generateFloatSchedule(
    double startDate,
    double endDate,
    int frequency = 4  // 4 = quarterly
) {
    Schedule sched;
    double dt = 1.0 / frequency;
    double t = startDate;
    while (t < endDate - 1e-8) {
        double nextT = std::min(t + dt, endDate);
        sched.resetDates.push_back(t);
        sched.paymentDates.push_back(nextT);
        sched.accrualFactors.push_back(nextT - t);
        t = nextT;
    }
    return sched;
}

} // namespace hw1f
