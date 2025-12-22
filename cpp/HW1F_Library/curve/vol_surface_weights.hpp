#pragma once
// =============================================================================
// Vol Surface Interpolation Weights
//
// Provides explicit interpolation weights for vol surface Greeks.
// When computing ∂r/∂Θ, we need to know which vol nodes affect which 
// calibration instruments via bilinear interpolation.
//
// Key insight: With bilinear interpolation, bumping a single vol node
// affects UP TO 4 calibration instruments (those whose (expiry, tenor)
// uses that node in interpolation). This is NOT the same as assuming
// "only one residual changes" - that assumption is ONLY correct when
// calibration instruments coincide exactly with vol surface nodes.
//
// =============================================================================

#include <vector>
#include <cmath>
#include <algorithm>

namespace hw1f {

// =============================================================================
// Interpolation Weight Structure
// =============================================================================

struct InterpWeight {
    size_t expiryIdx;   // Vol surface expiry index
    size_t tenorIdx;    // Vol surface tenor index
    double weight;      // Interpolation weight (0 to 1)
    
    InterpWeight(size_t ei, size_t ti, double w) 
        : expiryIdx(ei), tenorIdx(ti), weight(w) {}
};

// =============================================================================
// Get interpolation weights for a point (expiry, tenor) on the vol surface
// Returns the 4 nodes used in bilinear interpolation with their weights
// =============================================================================

inline std::vector<InterpWeight> getInterpWeights(
    double expiry,
    double tenor,
    const std::vector<double>& expiries,
    const std::vector<double>& tenors
) {
    std::vector<InterpWeight> weights;
    
    size_t numExp = expiries.size();
    size_t numTen = tenors.size();
    
    if (numExp == 0 || numTen == 0) return weights;
    
    // Find bracketing expiry indices
    size_t ei_lo = 0;
    for (size_t i = 0; i < numExp - 1; ++i) {
        if (expiry >= expiries[i]) ei_lo = i;
    }
    size_t ei_hi = std::min(ei_lo + 1, numExp - 1);
    
    // Find bracketing tenor indices
    size_t ti_lo = 0;
    for (size_t i = 0; i < numTen - 1; ++i) {
        if (tenor >= tenors[i]) ti_lo = i;
    }
    size_t ti_hi = std::min(ti_lo + 1, numTen - 1);
    
    // Compute interpolation weights
    double e_lo = expiries[ei_lo];
    double e_hi = expiries[ei_hi];
    double t_lo = tenors[ti_lo];
    double t_hi = tenors[ti_hi];
    
    double we = 0.0, wt = 0.0;
    
    if (ei_hi != ei_lo) {
        we = (expiry - e_lo) / (e_hi - e_lo);
        we = std::clamp(we, 0.0, 1.0);
    }
    
    if (ti_hi != ti_lo) {
        wt = (tenor - t_lo) / (t_hi - t_lo);
        wt = std::clamp(wt, 0.0, 1.0);
    }
    
    // Bilinear interpolation weights for 4 corners
    // vol = (1-we)(1-wt)*v00 + (1-we)*wt*v01 + we*(1-wt)*v10 + we*wt*v11
    
    double w00 = (1.0 - we) * (1.0 - wt);
    double w01 = (1.0 - we) * wt;
    double w10 = we * (1.0 - wt);
    double w11 = we * wt;
    
    // Only include nodes with non-zero weight
    const double eps = 1e-12;
    
    if (w00 > eps) {
        weights.emplace_back(ei_lo, ti_lo, w00);
    }
    if (w01 > eps && ti_hi != ti_lo) {
        weights.emplace_back(ei_lo, ti_hi, w01);
    }
    if (w10 > eps && ei_hi != ei_lo) {
        weights.emplace_back(ei_hi, ti_lo, w10);
    }
    if (w11 > eps && ei_hi != ei_lo && ti_hi != ti_lo) {
        weights.emplace_back(ei_hi, ti_hi, w11);
    }
    
    return weights;
}

// =============================================================================
// Check if calibration instrument coincides with vol surface node
// (i.e., is it an EXACT node match, not interpolated?)
// =============================================================================

inline bool isExactNode(
    double expiry,
    double tenor,
    const std::vector<double>& expiries,
    const std::vector<double>& tenors,
    double tol = 1e-8
) {
    bool expMatch = false, tenMatch = false;
    
    for (double e : expiries) {
        if (std::abs(e - expiry) < tol) {
            expMatch = true;
            break;
        }
    }
    
    for (double t : tenors) {
        if (std::abs(t - tenor) < tol) {
            tenMatch = true;
            break;
        }
    }
    
    return expMatch && tenMatch;
}

// =============================================================================
// Sparse Jacobian ∂r/∂Θ for vol Greeks
//
// When calibration instruments are NOT at vol surface nodes, the
// Jacobian ∂r_k/∂Θ_{ij} is sparse due to bilinear interpolation.
// Each calibration instrument k affects at most 4 vol nodes.
//
// dr_k/dΘ_{ij} = d(marketPrice_k)/dΘ_{ij} (negated because r = model - market)
//              = (∂Black/∂σ)_k * (∂σ_k/∂Θ_{ij})
//              = vega_k * w_{k,ij}
//
// where w_{k,ij} is the interpolation weight.
// =============================================================================

struct SparseVolJacobian {
    size_t n_inst;          // Number of calibration instruments
    size_t n_vol_nodes;     // Number of vol surface nodes
    
    // Sparse representation: for each instrument k, store the (idx, weight) pairs
    std::vector<std::vector<InterpWeight>> weights_per_inst;
    
    // Vegas (∂Black/∂σ) for each instrument
    std::vector<double> vegas;
    
    // Get dense row k of ∂r/∂Θ
    std::vector<double> getRow(size_t k, size_t n_expiries, size_t n_tenors) const {
        std::vector<double> row(n_expiries * n_tenors, 0.0);
        
        for (const auto& w : weights_per_inst[k]) {
            size_t flat_idx = w.expiryIdx * n_tenors + w.tenorIdx;
            // Note: r = model - market, so ∂r/∂Θ = -∂market/∂Θ
            row[flat_idx] = -vegas[k] * w.weight;
        }
        
        return row;
    }
    
    // Get full dense Jacobian (n_inst × n_vol_nodes)
    std::vector<std::vector<double>> toDense(size_t n_expiries, size_t n_tenors) const {
        std::vector<std::vector<double>> J(n_inst, std::vector<double>(n_vol_nodes, 0.0));
        
        for (size_t k = 0; k < n_inst; ++k) {
            for (const auto& w : weights_per_inst[k]) {
                size_t flat_idx = w.expiryIdx * n_tenors + w.tenorIdx;
                J[k][flat_idx] = -vegas[k] * w.weight;
            }
        }
        
        return J;
    }
};

// =============================================================================
// Build sparse vol Jacobian
// =============================================================================

template<typename VolSurface, typename CalibInst, typename CurveType>
SparseVolJacobian buildSparseVolJacobian(
    const std::vector<CalibInst>& instruments,
    const VolSurface& volSurface,
    const CurveType& curve,
    double notional
) {
    SparseVolJacobian result;
    result.n_inst = instruments.size();
    result.n_vol_nodes = volSurface.numNodes();
    result.weights_per_inst.resize(result.n_inst);
    result.vegas.resize(result.n_inst);
    
    const auto& expiries = volSurface.expiries();
    const auto& tenors = volSurface.tenors();
    
    for (size_t k = 0; k < result.n_inst; ++k) {
        double exp_k = instruments[k].expiry;
        double ten_k = instruments[k].tenor;
        
        // Get interpolation weights
        result.weights_per_inst[k] = getInterpWeights(exp_k, ten_k, expiries, tenors);
        
        // Compute Black vega for this instrument
        // (This should come from black_vega.hpp, placeholder here)
        double vol = volSurface.atmVol(exp_k, ten_k);
        double fwd = curve.fwdRate(exp_k, exp_k + ten_k);  // Approximation
        double sqrtT = std::sqrt(exp_k);
        double annuity = 1.0;  // Simplified
        
        // Simplified vega: ∂Black/∂σ ≈ fwd * sqrt(T) * N'(d1) * annuity * notional
        // This is a placeholder - use proper Black vega formula
        result.vegas[k] = fwd * sqrtT * notional * 0.4;  // Rough approximation
    }
    
    return result;
}

} // namespace hw1f
