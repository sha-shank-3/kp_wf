#pragma once
// =============================================================================
// HW1F Calibration Engine (Refactored)
// 
// CHANGES from original:
// 1. Explicit dimension tracking via ProblemDimensions
// 2. Renamed LM damping: lambda -> mu (avoid confusion with IFT adjoint)
// 3. Store J, W, r at optimum for IFT Greeks computation
// 4. Separate LM damping (mu) from IFT regularization (eps_reg)
// 5. LSQ First-Order Condition: f = J^T W r = 0 (not r = 0)
//
// Key Formulas (Least-Squares Calibration):
//   Objective:  h(Φ; m) = 0.5 * rᵀ W r
//   Residuals:  r_i(Φ) = P_model_i(Φ) - P_market_i(m)
//   FOC:        f = ∇_Φ h = JᵀWr = 0
//   Jacobian:   J_{ij} = ∂r_i/∂Φ_j  (shape: n_inst × n_params)
//   Hessian:    H = JᵀWJ (Gauss-Newton, shape: n_params × n_params)
//
// LM Update:
//   (H + mu * diag(H)) δ = -JᵀWr
//   Note: mu is LM damping, NOT to be confused with IFT regularization
//
// =============================================================================

#include "hw1f/hw1f_model.hpp"
#include "curve/discount_curve.hpp"
#include "instruments/swaption.hpp"
#include "pricing/jamshidian/jamshidian.hpp"
#include "utils/common.hpp"
#include "utils/dimension_types.hpp"
#include "utils/cholesky_factorization.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <optional>

namespace hw1f {

// =============================================================================
// Calibration Instrument
// =============================================================================

struct CalibrationInstrument {
    double expiry;
    double tenor;
    double marketVol;      // ATM Black vol
    double marketPrice;    // Black price (computed from vol)
    double weight;         // Weight in objective function
    
    CalibrationInstrument(double e, double t, double vol, double w = 1.0)
        : expiry(e), tenor(t), marketVol(vol), marketPrice(0.0), weight(w) {}
};

// =============================================================================
// Calibration Result (Extended)
// 
// Now stores all artifacts needed for IFT Greeks:
//   - Jacobian J at optimum
//   - Residuals r at optimum  
//   - Weight matrix W (diagonal)
//   - Gauss-Newton Hessian H = JᵀWJ
// =============================================================================

struct CalibrationResult {
    // Basic results
    HW1FParams params;
    std::vector<double> residuals;      // r at optimum (size n_inst)
    double rmse;
    int iterations;
    bool converged;
    double elapsedTime;
    
    // Dimensions
    ProblemDimensions dims;
    
    // Stored artifacts for IFT
    std::vector<std::vector<double>> jacobian;  // J at optimum (n_inst × n_params)
    std::vector<double> weights;                // Diagonal of W (size n_inst)
    std::vector<std::vector<double>> hessian;   // H = JᵀWJ at optimum (n_params × n_params)
    
    // FOC norm (should be near zero at convergence)
    double foc_norm;  // ||JᵀWr||
    
    // Coverage statistics
    CalibrationCoverage coverage;
    
    // Check if this is a valid result for IFT
    bool isValidForIFT() const {
        return converged && 
               !jacobian.empty() && 
               !hessian.empty() &&
               jacobian.size() == dims.n_inst &&
               hessian.size() == dims.n_params;
    }
};

// =============================================================================
// Calibration Engine (Refactored)
// =============================================================================

template<typename CurveReal = double>
class CalibrationEngine {
public:
    CalibrationEngine(
        const DiscountCurve<CurveReal>& curve,
        const ATMVolSurface<double>& volSurface,
        double notional = 1e6
    ) : curve_(curve), volSurface_(volSurface), notional_(notional) {}
    
    // Add calibration instrument
    void addInstrument(double expiry, double tenor, double weight = 1.0) {
        double vol = volSurface_.atmVol(expiry, tenor);
        instruments_.emplace_back(expiry, tenor, vol, weight);
    }
    
    // Add instruments from vol surface nodes (full surface calibration)
    void addAllSurfaceNodes() {
        for (size_t ei = 0; ei < volSurface_.numExpiries(); ++ei) {
            for (size_t ti = 0; ti < volSurface_.numTenors(); ++ti) {
                double expiry = volSurface_.expiries()[ei];
                double tenor = volSurface_.tenors()[ti];
                double vol = volSurface_.vols()[ei][ti];
                instruments_.emplace_back(expiry, tenor, vol, 1.0);
            }
        }
    }
    
    // Add specific instruments (sparse calibration)
    void addSparseInstruments(const std::vector<std::pair<double, double>>& expiryTenorPairs) {
        for (const auto& [expiry, tenor] : expiryTenorPairs) {
            double vol = volSurface_.atmVol(expiry, tenor);
            instruments_.emplace_back(expiry, tenor, vol, 1.0);
        }
    }
    
    // Get current problem dimensions
    ProblemDimensions getDimensions(const HW1FParams& params) const {
        return ProblemDimensions(
            params.sigmaValues.size(),    // n_sigma
            instruments_.size(),           // n_inst
            volSurface_.numExpiries(),    // n_expiries
            volSurface_.numTenors(),      // n_tenors
            curve_.size()                 // n_curve_nodes
        );
    }
    
    // Compute market prices from Black vols
    void computeMarketPrices() {
        for (auto& inst : instruments_) {
            EuropeanSwaption swaption(inst.expiry, inst.tenor, 0.0, notional_, true);
            double fwd = value(forwardSwapRate(swaption.underlying, curve_));
            swaption.underlying.fixedRate = fwd;
            inst.marketPrice = value(blackSwaptionPrice(swaption, curve_, inst.marketVol));
        }
    }
    
    // Compute model price for single instrument
    double modelPrice(const CalibrationInstrument& inst, const HW1FParams& params) const {
        HW1FModel<double> model(params);
        EuropeanSwaption swaption(inst.expiry, inst.tenor, 0.0, notional_, true);
        double fwd = value(forwardSwapRate(swaption.underlying, curve_));
        swaption.underlying.fixedRate = fwd;
        JamshidianPricer<double, CurveReal> pricer(model, curve_);
        return pricer.price(swaption);
    }
    
    // Compute residuals: r_i = (model_i - market_i)
    // Note: Weights are applied when computing objective and gradient
    std::vector<double> computeResiduals(const HW1FParams& params) const {
        const size_t n_inst = instruments_.size();
        std::vector<double> residuals(n_inst);
        
        for (size_t i = 0; i < n_inst; ++i) {
            double modelP = modelPrice(instruments_[i], params);
            double marketP = instruments_[i].marketPrice;
            residuals[i] = modelP - marketP;
        }
        
        return residuals;
    }
    
    // Compute weighted objective: h = 0.5 * rᵀWr
    double computeObjective(const std::vector<double>& residuals) const {
        double obj = 0.0;
        for (size_t i = 0; i < residuals.size(); ++i) {
            double w = instruments_[i].weight;
            obj += w * residuals[i] * residuals[i];
        }
        return 0.5 * obj;
    }
    
    // Compute RMSE (unweighted for interpretability)
    double computeRMSE(const std::vector<double>& residuals) const {
        double sum = 0.0;
        for (double r : residuals) {
            sum += r * r;
        }
        return std::sqrt(sum / residuals.size());
    }
    
    // Compute Jacobian J using finite differences
    // J_{ij} = ∂r_i/∂Φ_j (shape: n_inst × n_params)
    std::vector<std::vector<double>> computeJacobianFD(
        const HW1FParams& params,
        double eps = 1e-6
    ) const {
        const size_t n_inst = instruments_.size();
        const size_t n_params = params.numParams();
        
        std::vector<std::vector<double>> J(n_inst, std::vector<double>(n_params, 0.0));
        std::vector<double> baseResiduals = computeResiduals(params);
        
        std::vector<double> paramVec = params.toVector();
        
        for (size_t j = 0; j < n_params; ++j) {
            double h = eps * std::max(1.0, std::abs(paramVec[j]));
            
            std::vector<double> paramUp = paramVec;
            paramUp[j] += h;
            
            HW1FParams paramsUp = params;
            paramsUp.fromVector(paramUp);
            
            std::vector<double> residualsUp = computeResiduals(paramsUp);
            
            for (size_t i = 0; i < n_inst; ++i) {
                J[i][j] = (residualsUp[i] - baseResiduals[i]) / h;
            }
        }
        
        return J;
    }
    
    // Compute H = JᵀWJ (Gauss-Newton Hessian approximation)
    std::vector<std::vector<double>> computeGaussNewtonHessian(
        const std::vector<std::vector<double>>& J
    ) const {
        const size_t n_inst = J.size();
        const size_t n_params = J.empty() ? 0 : J[0].size();
        
        std::vector<std::vector<double>> H(n_params, std::vector<double>(n_params, 0.0));
        
        // H = JᵀWJ where W is diagonal with weights
        for (size_t i = 0; i < n_params; ++i) {
            for (size_t j = 0; j < n_params; ++j) {
                for (size_t k = 0; k < n_inst; ++k) {
                    double w_k = instruments_[k].weight;
                    H[i][j] += J[k][i] * w_k * J[k][j];
                }
            }
        }
        
        return H;
    }
    
    // Compute gradient g = JᵀWr (FOC)
    std::vector<double> computeGradient(
        const std::vector<std::vector<double>>& J,
        const std::vector<double>& residuals
    ) const {
        const size_t n_inst = J.size();
        const size_t n_params = J.empty() ? 0 : J[0].size();
        
        std::vector<double> g(n_params, 0.0);
        
        for (size_t j = 0; j < n_params; ++j) {
            for (size_t i = 0; i < n_inst; ++i) {
                double w_i = instruments_[i].weight;
                g[j] += J[i][j] * w_i * residuals[i];
            }
        }
        
        return g;
    }
    
    // =========================================================================
    // Levenberg-Marquardt Calibration (Refactored)
    //
    // Key changes:
    // 1. mu = LM damping parameter (NOT lambda, to avoid IFT confusion)
    // 2. Store J, W, r, H at optimum for IFT reuse
    // 3. Explicit dimension tracking
    // =========================================================================
    CalibrationResult calibrate(
        HW1FParams initialParams,
        int maxIter = 100,
        double tol = 1e-8,
        bool verbose = false
    ) {
        Timer timer;
        timer.start();
        
        // Ensure market prices are computed
        computeMarketPrices();
        
        HW1FParams params = initialParams;
        std::vector<double> paramVec = params.toVector();
        const size_t n_params = paramVec.size();
        const size_t n_inst = instruments_.size();
        
        // Get dimensions
        ProblemDimensions dims = getDimensions(params);
        
        // LM damping parameter (NOT to be confused with IFT adjoint λ)
        double mu = 1e-3;   // Initial damping
        double nu = 2.0;    // Damping adjustment factor
        
        std::vector<double> residuals = computeResiduals(params);
        double cost = computeObjective(residuals);
        
        // Store best Jacobian and Hessian for IFT
        std::vector<std::vector<double>> J_opt;
        std::vector<std::vector<double>> H_opt;
        std::vector<double> r_opt = residuals;
        
        CalibrationResult result;
        result.converged = false;
        result.dims = dims;
        
        for (int iter = 0; iter < maxIter; ++iter) {
            // Compute Jacobian J
            auto J = computeJacobianFD(params);
            
            // Compute H = JᵀWJ
            auto H = computeGaussNewtonHessian(J);
            
            // Compute gradient g = JᵀWr
            auto g = computeGradient(J, residuals);
            
            // Check FOC convergence: ||JᵀWr|| < tol
            double foc_norm = norm(g);
            
            // LM: Add damping to diagonal: A = H + mu * diag(H)
            auto A = H;
            for (size_t i = 0; i < n_params; ++i) {
                A[i][i] += mu * (H[i][i] + 1e-8);  // Add small constant for numerical stability
            }
            
            // Solve for step: A * delta = -g
            std::vector<double> neg_g(n_params);
            for (size_t i = 0; i < n_params; ++i) {
                neg_g[i] = -g[i];
            }
            
            std::vector<double> delta;
            try {
                delta = solveCholesky(A, neg_g);
            } catch (...) {
                // Matrix not positive definite, increase damping
                mu *= nu;
                continue;
            }
            
            // Try update
            std::vector<double> newParamVec(n_params);
            for (size_t i = 0; i < n_params; ++i) {
                newParamVec[i] = paramVec[i] + delta[i];
            }
            
            // Enforce bounds
            newParamVec[0] = std::clamp(newParamVec[0], 1e-4, 1.0);  // a
            for (size_t i = 1; i < n_params; ++i) {
                newParamVec[i] = std::clamp(newParamVec[i], 1e-6, 0.1);  // sigma
            }
            
            HW1FParams newParams = params;
            newParams.fromVector(newParamVec);
            
            std::vector<double> newResiduals = computeResiduals(newParams);
            double newCost = computeObjective(newResiduals);
            
            // Accept or reject
            if (newCost < cost) {
                params = newParams;
                paramVec = newParamVec;
                residuals = newResiduals;
                cost = newCost;
                mu /= nu;
                
                // Store artifacts at best point
                J_opt = J;
                H_opt = H;
                r_opt = residuals;
                
                if (verbose) {
                    std::cout << "  Iter " << std::setw(3) << iter 
                              << ": RMSE = " << std::scientific << std::setprecision(6) << computeRMSE(residuals)
                              << ", ||FOC|| = " << foc_norm
                              << ", mu = " << mu << "\n";
                }
                
                // Check convergence
                if (norm(delta) < tol * (norm(paramVec) + tol) || foc_norm < tol) {
                    result.converged = true;
                    result.iterations = iter + 1;
                    break;
                }
            } else {
                mu *= nu;
            }
            
            if (iter == maxIter - 1) {
                result.iterations = maxIter;
            }
        }
        
        // Final Jacobian and Hessian if not stored yet
        if (J_opt.empty()) {
            J_opt = computeJacobianFD(params);
            H_opt = computeGaussNewtonHessian(J_opt);
        }
        
        // Store results
        result.params = params;
        result.residuals = residuals;
        result.rmse = computeRMSE(residuals);
        result.elapsedTime = timer.elapsed();
        
        // Store IFT artifacts
        result.jacobian = J_opt;
        result.hessian = H_opt;
        result.weights.resize(n_inst);
        for (size_t i = 0; i < n_inst; ++i) {
            result.weights[i] = instruments_[i].weight;
        }
        
        // Compute final FOC norm
        auto g_final = computeGradient(J_opt, residuals);
        result.foc_norm = norm(g_final);
        
        // Compute coverage statistics
        result.coverage.n_inst_total = n_inst;
        result.coverage.n_params = n_params;
        result.coverage.coverage_ratio = static_cast<double>(n_inst) / n_params;
        result.coverage.rmse = result.rmse;
        result.coverage.max_residual = 0.0;
        for (double r : residuals) {
            result.coverage.max_residual = std::max(result.coverage.max_residual, std::abs(r));
        }
        result.coverage.is_exact_fit = (n_inst == n_params) && (result.rmse < 1e-6);
        
        return result;
    }
    
    // Get instruments for IFT computation
    const std::vector<CalibrationInstrument>& instruments() const { return instruments_; }
    
    // Get curve
    const DiscountCurve<CurveReal>& curve() const { return curve_; }
    
    // Get vol surface
    const ATMVolSurface<double>& volSurface() const { return volSurface_; }
    
    // Get notional
    double notional() const { return notional_; }

private:
    DiscountCurve<CurveReal> curve_;
    ATMVolSurface<double> volSurface_;
    std::vector<CalibrationInstrument> instruments_;
    double notional_;
};

} // namespace hw1f
