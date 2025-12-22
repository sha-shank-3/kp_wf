# HW1F Library Refactoring Summary

## Overview

This document summarizes the comprehensive refactoring of the HW1F Greeks computation library, implementing all corrections required for mathematical correctness per the OpenGamma/Henrard IFT framework.

## Files Created/Modified

### New Files

| File | Purpose |
|------|---------|
| [utils/dimension_types.hpp](utils/dimension_types.hpp) | Explicit dimension types and validation helpers |
| [utils/cholesky_factorization.hpp](utils/cholesky_factorization.hpp) | Reusable Cholesky factorization with regularization |
| [calibration/calibration_refactored.hpp](calibration/calibration_refactored.hpp) | Refactored calibration engine with stored J, W, r |
| [curve/vol_surface_weights.hpp](curve/vol_surface_weights.hpp) | Interpolation weights for vol Greeks |
| [pricing/black_vega.hpp](pricing/black_vega.hpp) | Black swaption Greeks (vega, delta, gamma) |
| [risk/ift/ift_greeks_refactored.hpp](risk/ift/ift_greeks_refactored.hpp) | Corrected IFT Greeks engine |
| [apps/greeks_validation.cpp](apps/greeks_validation.cpp) | Validation app comparing all methods |
| [tests/test_dimensions.cpp](tests/test_dimensions.cpp) | Unit tests for dimension sanity |
| [tests/test_vr_buckets.cpp](tests/test_vr_buckets.cpp) | Unit tests for V_r bucket splitting |
| [tests/test_ift_vs_fd.cpp](tests/test_ift_vs_fd.cpp) | Unit tests for IFT vs FD validation |

## Key Corrections

### A. Dimension Consistency

**Before:** Ambiguous dimension variables, hard-coded values like `6x6`.

**After:** Explicit `ProblemDimensions` structure:
```cpp
struct ProblemDimensions {
    size_t n_params;       // Number of calibrated parameters (a + sigmas)
    size_t n_sigma;        // Number of sigma buckets = n_params - 1
    size_t n_inst;         // Number of calibration instruments
    size_t n_expiries;     // Number of vol surface expiries
    size_t n_tenors;       // Number of vol surface tenors
    size_t n_vol_nodes;    // = n_expiries × n_tenors
    size_t n_curve_nodes;  // Number of curve nodes
};
```

Validation helpers ensure matrices have correct shapes:
```cpp
validateJacobianShape(J, dims);  // J: n_inst × n_params
validateHessianShape(H, dims);   // H: n_params × n_params
validateResidualSize(r, dims);   // r: n_inst
```

### B. LSQ FOC Clarification

**Before:** Unclear distinction between exact-fit (r = 0) and LSQ (f = JᵀWr = 0).

**After:** Explicit documentation and coverage statistics:
```cpp
struct CalibrationCoverage {
    size_t n_inst_total;       // Total instruments
    size_t n_params;           // Number of parameters
    double coverage_ratio;     // n_inst / n_params
    double rmse;               // Root mean square error
    bool is_exact_fit;         // True if n_inst == n_params and RMSE ≈ 0
};
```

### C. Separated LM Damping from IFT Regularization

**Before:** `lambda` used for both LM damping and IFT adjoint.

**After:** Clear separation:
- `mu` = LM damping parameter in calibration (adaptive)
- `eps_reg` = IFT regularization (tiny, fixed, ~1e-10)
- `lambda` = IFT adjoint vector (solution to Hᵀλ = V_Φ)

```cpp
// In calibration:
double mu = 1e-3;  // LM damping (NOT used in IFT)

// In IFT:
double eps_reg = 1e-10;  // Tiny regularization for numerical stability only
std::vector<double> lambda = chol.solve(V_phi);  // IFT adjoint
```

### D. Corrected IFT Formula with Direct Terms

**Before:** Missing V_C_direct term for curve Greeks.

**After:** Complete OpenGamma formula:
```
dV/dΘ = V_Θ_direct - λᵀ f_Θ    (vol Greeks)
dV/dC = V_C_direct - λᵀ f_C    (curve Greeks)
```

Where:
- `V_Θ_direct = 0` for HW1F (exotic doesn't depend on Black vols)
- `V_C_direct ≠ 0` (exotic DOES depend on curve directly)
- `λ` solves `Hλ = V_Φ` where `H = JᵀWJ` (NOT LM-damped)

### E. Vol Interpolation Weights

**Before:** Assumed "only one residual changes" when bumping vol node.

**After:** Explicit interpolation weights for bilinear interpolation:
```cpp
std::vector<InterpWeight> getInterpWeights(
    double expiry, double tenor,
    const std::vector<double>& expiries,
    const std::vector<double>& tenors
);
```

Each calibration instrument can depend on up to 4 vol nodes through bilinear interpolation.

### F. Cholesky Factorization with Reuse

**Before:** Cholesky factorization computed and discarded.

**After:** `CholeskyFactorization` class stores L factor for reuse:
```cpp
CholeskyFactorization chol;
chol.factor(H, eps_reg);  // Factor once
auto lambda = chol.solve(V_phi);  // Solve multiple RHS
auto inverseH = chol.inverse();  // Optional
double logDet = chol.logDeterminant();  // Optional
```

### G. Exact-Fit vs LSQ Unified Output

**Before:** Different code paths for exact-fit and LSQ.

**After:** Single `CalibrationResult` structure with coverage statistics:
```cpp
result.coverage.is_exact_fit;  // True if n_inst == n_params and RMSE ≈ 0
result.coverage.coverage_ratio;  // n_inst / n_params
result.foc_norm;  // ||JᵀWr|| (should be ~0 at convergence)
```

### H. V_r Bucket Splitting

**Before:** Potential errors when integration interval crosses sigma bucket boundaries.

**After:** Comprehensive unit tests in `test_vr_buckets.cpp`:
- Single bucket interval
- One boundary crossing
- Multiple boundary crossings
- Degenerate cases (a→0, t=s)
- Consistency with σ_P formula

## Mathematical Summary

### Calibration (Least-Squares)

Objective: $h(\Phi; m) = \frac{1}{2} r^T W r$

Residuals: $r_i(\Phi) = P_{\text{model},i}(\Phi) - P_{\text{market},i}(m)$

FOC: $f = \nabla_\Phi h = J^T W r = 0$

Jacobian: $J_{ij} = \frac{\partial r_i}{\partial \Phi_j}$ (shape: $n_{\text{inst}} \times n_{\text{params}}$)

Gauss-Newton Hessian: $H = J^T W J$ (shape: $n_{\text{params}} \times n_{\text{params}}$)

### IFT Greeks

Adjoint equation: $H^T \lambda = V_\Phi$ (or $H \lambda = V_\Phi$ since H is symmetric)

Vol Greeks: $\frac{dV}{d\Theta} = V_{\Theta,\text{direct}} - \lambda^T f_\Theta$

Curve Greeks: $\frac{dV}{dC} = V_{C,\text{direct}} - \lambda^T f_C$

Where:
- $V_{\Theta,\text{direct}} = 0$ for HW1F
- $V_{C,\text{direct}} = \frac{\partial V_{\text{exotic}}}{\partial C}\bigg|_{\Phi=\text{const}}$ (non-zero!)
- $f_\Theta = J^T W \frac{\partial r}{\partial \Theta}$
- $f_C = J^T W \frac{\partial r}{\partial C}$

## Dimension Reference

| Symbol | Name | Size |
|--------|------|------|
| $n_{\text{params}}$ | Calibrated parameters | $1 + n_\sigma$ (typically 10) |
| $n_{\text{inst}}$ | Calibration instruments | 10 (exact-fit) or 81 (LSQ) |
| $n_{\text{vol}}$ | Vol surface nodes | $n_{\text{exp}} \times n_{\text{ten}}$ (typically 81) |
| $n_{\text{curve}}$ | Curve nodes | typically 12 |
| $J$ | Residual Jacobian | $n_{\text{inst}} \times n_{\text{params}}$ |
| $H$ | Gauss-Newton Hessian | $n_{\text{params}} \times n_{\text{params}}$ |
| $W$ | Weight matrix | $n_{\text{inst}} \times n_{\text{inst}}$ (diagonal) |
| $r$ | Residuals | $n_{\text{inst}}$ |
| $\lambda$ | IFT adjoint | $n_{\text{params}}$ |
| $V_\Phi$ | Price sensitivities | $n_{\text{params}}$ |

## Usage Example

```cpp
// 1. Calibrate
CalibrationEngine<double> calibEngine(curve, volSurface, notional);
for (const auto& [e, t] : calibInstruments) {
    calibEngine.addInstrument(e, t);
}
auto calibResult = calibEngine.calibrate(initialParams, 100, 1e-8, true);

// 2. Check dimensions
std::cout << calibResult.dims.toString();
std::cout << calibResult.coverage.toString();

// 3. Compute IFT Greeks (uses stored J, H from calibration)
IFTGreeksEngineRefactored<double> iftEngine(
    curve, volSurface, calibInstruments, notional
);
auto greeks = iftEngine.computeIFTFromCalibResult(exotic, calibResult, mcConfig);

// 4. Access results
std::cout << "Price: " << greeks.price << "\n";
std::cout << "||V_C_direct||: " << greeks.V_C_direct_norm << "\n";
std::cout << "||λ||: " << greeks.lambda_norm << "\n";
```

## Test Coverage

| Test File | Coverage |
|-----------|----------|
| `test_dimensions.cpp` | ProblemDimensions, J/H/r shape validation, calibration results |
| `test_vr_buckets.cpp` | V_r integration with piecewise-constant sigma |
| `test_ift_vs_fd.cpp` | IFT vs FD Naive comparison, direct terms |
| `greeks_validation.cpp` | Full comparison of all methods |

## Migration Guide

To migrate from the original code:

1. Replace `CalibrationEngine` with `calibration/calibration_refactored.hpp`
2. Replace IFT engine with `risk/ift/ift_greeks_refactored.hpp`
3. Use `calibResult.jacobian` and `calibResult.hessian` instead of recomputing
4. Ensure `eps_reg` in IFT is tiny (1e-10), NOT the LM damping value
5. Always include V_C_direct for curve Greeks
6. Run unit tests to validate correctness

## References

1. Henrard, M. (2017). *Algorithmic Differentiation in Finance Explained*. OpenGamma.
2. Gurrieri et al. (2009). *Hull-White One Factor Model*. Mizuho.
3. Carmelid, J. (2017). *Computation of the Greeks using AAD*. KTH Thesis.
4. Brigo, D. & Mercurio, F. (2006). *Interest Rate Models - Theory and Practice*. Ch. 3.
