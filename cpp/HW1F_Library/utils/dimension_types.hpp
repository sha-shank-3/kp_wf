#pragma once
// =============================================================================
// HW1F Library - Dimension Types and Constants
// Provides explicit dimension variables for mathematical correctness
//
// Key Dimensions:
//   n_params     = Number of calibration parameters (1 + n_sigma = a + sigmas)
//   n_inst       = Number of calibration instruments
//   n_vol_nodes  = Number of vol surface nodes (n_expiries × n_tenors)
//   n_curve_nodes = Number of curve discount factor nodes
//
// Following OpenGamma notation:
//   Φ ∈ ℝ^{n_params}     - Calibrated HW parameters [a, σ₁, ..., σₖ]
//   Θ ∈ ℝ^{n_vol_nodes}  - Vol surface node values
//   C ∈ ℝ^{n_curve_nodes} - Curve node values (zero rates or DFs)
//   m = {Θ, C}           - Market data
//
// Key Matrices:
//   J ∈ ℝ^{n_inst × n_params}  - Jacobian ∂r/∂Φ (residuals w.r.t. params)
//   H ∈ ℝ^{n_params × n_params} - Gauss-Newton Hessian = JᵀWJ
//   W ∈ ℝ^{n_inst × n_inst}     - Weight matrix (diagonal for weighted LSQ)
//
// =============================================================================

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace hw1f {

// =============================================================================
// Dimension Constants Structure
// =============================================================================

struct ProblemDimensions {
    size_t n_params;       // Number of calibrated parameters (a + sigmas)
    size_t n_sigma;        // Number of sigma buckets = n_params - 1
    size_t n_inst;         // Number of calibration instruments
    size_t n_expiries;     // Number of vol surface expiries
    size_t n_tenors;       // Number of vol surface tenors
    size_t n_vol_nodes;    // = n_expiries × n_tenors
    size_t n_curve_nodes;  // Number of curve nodes
    
    // Default constructor
    ProblemDimensions() 
        : n_params(0), n_sigma(0), n_inst(0), n_expiries(0), 
          n_tenors(0), n_vol_nodes(0), n_curve_nodes(0) {}
    
    // Explicit constructor with validation
    ProblemDimensions(
        size_t num_sigma,
        size_t num_instruments,
        size_t num_expiries,
        size_t num_tenors,
        size_t num_curve_nodes
    ) : n_params(1 + num_sigma),   // a + sigmas
        n_sigma(num_sigma),
        n_inst(num_instruments),
        n_expiries(num_expiries),
        n_tenors(num_tenors),
        n_vol_nodes(num_expiries * num_tenors),
        n_curve_nodes(num_curve_nodes)
    {
        validate();
    }
    
    // Validate dimensions
    void validate() const {
        if (n_params == 0) {
            throw std::runtime_error("n_params must be > 0");
        }
        if (n_inst == 0) {
            throw std::runtime_error("n_inst must be > 0");
        }
        if (n_vol_nodes != n_expiries * n_tenors) {
            throw std::runtime_error("n_vol_nodes must equal n_expiries * n_tenors");
        }
    }
    
    // Calibration type
    bool isExactFit() const { return n_inst == n_params; }
    bool isOverDetermined() const { return n_inst > n_params; }
    bool isUnderDetermined() const { return n_inst < n_params; }
    
    // String representation for debugging
    std::string toString() const {
        std::string result = "ProblemDimensions:\n";
        result += "  n_params      = " + std::to_string(n_params) + " (1 a + " + std::to_string(n_sigma) + " σ)\n";
        result += "  n_inst        = " + std::to_string(n_inst) + "\n";
        result += "  n_vol_nodes   = " + std::to_string(n_vol_nodes) + " (" + std::to_string(n_expiries) + " exp × " + std::to_string(n_tenors) + " ten)\n";
        result += "  n_curve_nodes = " + std::to_string(n_curve_nodes) + "\n";
        result += "  Type: " + (isExactFit() ? "Exact-fit" : (isOverDetermined() ? "Over-determined (LSQ)" : "Under-determined")) + "\n";
        return result;
    }
};

// =============================================================================
// Matrix Dimension Validation Helpers
// =============================================================================

// Validate Jacobian J has shape (n_inst, n_params)
inline void validateJacobianShape(
    const std::vector<std::vector<double>>& J,
    const ProblemDimensions& dims
) {
    if (J.size() != dims.n_inst) {
        throw std::runtime_error(
            "Jacobian J has " + std::to_string(J.size()) + " rows, expected n_inst = " + 
            std::to_string(dims.n_inst)
        );
    }
    if (!J.empty() && J[0].size() != dims.n_params) {
        throw std::runtime_error(
            "Jacobian J has " + std::to_string(J[0].size()) + " columns, expected n_params = " + 
            std::to_string(dims.n_params)
        );
    }
}

// Validate Hessian H has shape (n_params, n_params)
inline void validateHessianShape(
    const std::vector<std::vector<double>>& H,
    const ProblemDimensions& dims
) {
    if (H.size() != dims.n_params) {
        throw std::runtime_error(
            "Hessian H has " + std::to_string(H.size()) + " rows, expected n_params = " + 
            std::to_string(dims.n_params)
        );
    }
    if (!H.empty() && H[0].size() != dims.n_params) {
        throw std::runtime_error(
            "Hessian H has " + std::to_string(H[0].size()) + " columns, expected n_params = " + 
            std::to_string(dims.n_params)
        );
    }
}

// Validate residual vector r has size n_inst
inline void validateResidualSize(
    const std::vector<double>& r,
    const ProblemDimensions& dims
) {
    if (r.size() != dims.n_inst) {
        throw std::runtime_error(
            "Residual r has size " + std::to_string(r.size()) + ", expected n_inst = " + 
            std::to_string(dims.n_inst)
        );
    }
}

// Validate parameter vector has size n_params
inline void validateParamSize(
    const std::vector<double>& params,
    const ProblemDimensions& dims
) {
    if (params.size() != dims.n_params) {
        throw std::runtime_error(
            "Parameter vector has size " + std::to_string(params.size()) + ", expected n_params = " + 
            std::to_string(dims.n_params)
        );
    }
}

// Validate weight matrix W has shape (n_inst, n_inst) or is empty (identity)
inline void validateWeightMatrix(
    const std::vector<std::vector<double>>& W,
    const ProblemDimensions& dims
) {
    if (W.empty()) return;  // Identity weight matrix
    
    if (W.size() != dims.n_inst) {
        throw std::runtime_error(
            "Weight matrix W has " + std::to_string(W.size()) + " rows, expected n_inst = " + 
            std::to_string(dims.n_inst)
        );
    }
    if (W[0].size() != dims.n_inst) {
        throw std::runtime_error(
            "Weight matrix W has " + std::to_string(W[0].size()) + " columns, expected n_inst = " + 
            std::to_string(dims.n_inst)
        );
    }
}

// =============================================================================
// Coverage Statistics for Calibration
// =============================================================================

struct CalibrationCoverage {
    size_t n_inst_total;       // Total instruments in calibration set
    size_t n_params;           // Number of calibrated parameters
    double coverage_ratio;     // n_inst / n_params (>1 for over-determined)
    double rmse;               // Root mean square error of residuals
    double max_residual;       // Maximum absolute residual
    bool is_exact_fit;         // True if n_inst == n_params and RMSE ≈ 0
    
    std::string toString() const {
        std::string result = "CalibrationCoverage:\n";
        result += "  Instruments:    " + std::to_string(n_inst_total) + "\n";
        result += "  Parameters:     " + std::to_string(n_params) + "\n";
        result += "  Coverage Ratio: " + std::to_string(coverage_ratio) + "\n";
        result += "  RMSE:           " + std::to_string(rmse) + "\n";
        result += "  Max Residual:   " + std::to_string(max_residual) + "\n";
        result += "  Exact Fit:      " + std::string(is_exact_fit ? "Yes" : "No") + "\n";
        return result;
    }
};

} // namespace hw1f
