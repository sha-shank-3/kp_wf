#pragma once
// =============================================================================
// IFT Greeks Engine (Refactored)
//
// CORRECTED IMPLEMENTATION of OpenGamma Adjoint-IFT
//
// Key Formulas (from OpenGamma/Henrard):
// ============================================
// Calibration condition (FOC for least-squares):
//   f(C, Θ, Φ) = JᵀWr = 0
//   where r = P_model - P_market (residuals)
//
// Total sensitivity via IFT:
//   dV/dm = V_m_direct - λᵀ f_m
//   where λ solves: Hᵀ λ = V_Φ
//         H = JᵀWJ (Gauss-Newton Hessian, NOT LM-damped!)
//
// Expanding for market data m = {Θ, C}:
//   dV/dΘ = V_Θ_direct - λᵀ f_Θ    (vol Greeks)
//   dV/dC = V_C_direct - λᵀ f_C    (curve Greeks)
//
// CRITICAL NOTES:
// 1. H = JᵀWJ only (tiny regularization eps_reg for numerical stability)
//    DO NOT use LM damping mu from calibration!
// 2. V_Θ_direct = 0 for HW1F (exotic doesn't depend on Black vols directly)
// 3. V_C_direct ≠ 0 for curve Greeks (exotic DOES depend on curve directly)
// 4. Use correct interpolation weights for vol Greeks
//
// =============================================================================

#include "hw1f/hw1f_model.hpp"
#include "calibration/calibration_refactored.hpp"
#include "curve/discount_curve.hpp"
#include "curve/vol_surface_weights.hpp"
#include "pricing/black_vega.hpp"
#include "pricing/montecarlo/montecarlo.hpp"
#include "utils/common.hpp"
#include "utils/dimension_types.hpp"
#include "utils/cholesky_factorization.hpp"
#include <XAD/XAD.hpp>
#include <vector>
#include <cmath>

namespace hw1f {

// =============================================================================
// Greeks Result (Extended)
// =============================================================================

struct GreeksResult {
    double price;
    double dVda;                                // dV/da
    std::vector<double> dVdsigma;               // dV/dσ_i for each sigma bucket
    std::vector<std::vector<double>> volGreeks; // dV/dΘ_{ij} for vol surface
    std::vector<double> curveGreeks;            // dV/dC_i for curve nodes
    double elapsedTime;
    
    // Breakdown for debugging/validation
    double V_C_direct_norm;   // ||V_C_direct||
    double V_Theta_direct_norm; // Should be 0 for HW1F
    double lambda_norm;        // ||λ||
    
    // Dimensions used
    ProblemDimensions dims;
};

// =============================================================================
// IFT Greeks Engine (Refactored with explicit dimensions and correct formulas)
// =============================================================================

template<typename CurveReal = double>
class IFTGreeksEngineRefactored {
public:
    using ADReal = xad::AReal<double>;
    using Tape = xad::Tape<double>;
    
    IFTGreeksEngineRefactored(
        const DiscountCurve<CurveReal>& curve,
        const ATMVolSurface<double>& volSurface,
        const std::vector<std::pair<double, double>>& calibInstruments,
        double notional = 1e6,
        double eps_reg = 1e-10  // IFT regularization (NOT LM damping!)
    ) : curve_(curve), volSurface_(volSurface), 
        calibInstruments_(calibInstruments), notional_(notional),
        eps_reg_(eps_reg) {}
    
    // =========================================================================
    // MAIN ENTRY: Compute IFT Greeks with stored calibration result
    //
    // This uses the Jacobian J and Hessian H stored from calibration,
    // avoiding redundant computation.
    // =========================================================================
    GreeksResult computeIFTFromCalibResult(
        const EuropeanSwaption& swaption,
        const CalibrationResult& calibResult,
        const MCConfig& mcConfig,
        double bump = 1e-6
    ) const {
        Timer timer;
        timer.start();
        
        GreeksResult result;
        result.dims = calibResult.dims;
        
        // Validate calibration result
        if (!calibResult.isValidForIFT()) {
            throw std::runtime_error("CalibrationResult is not valid for IFT computation");
        }
        
        // Extract dimensions
        const size_t n_params = calibResult.dims.n_params;
        const size_t n_inst = calibResult.dims.n_inst;
        const size_t n_vol_nodes = calibResult.dims.n_vol_nodes;
        const size_t n_curve_nodes = calibResult.dims.n_curve_nodes;
        const size_t n_expiries = calibResult.dims.n_expiries;
        const size_t n_tenors = calibResult.dims.n_tenors;
        
        // Fixed random numbers for consistent pricing
        RNG rng(mcConfig.seed);
        auto Z = rng.normalMatrix(mcConfig.numPaths, mcConfig.numSteps);
        
        const HW1FParams& calibratedParams = calibResult.params;
        
        // =====================================================================
        // Step 1: Compute V_Φ = dV/dΦ using FD (central differences)
        // =====================================================================
        HW1FModel<double> model(calibratedParams);
        MonteCarloPricer<double> pricer(model, mcConfig);
        double basePrice = pricer.price(swaption, curve_, Z).price;
        result.price = basePrice;
        
        std::vector<double> V_phi = computeVPhiFD(swaption, calibratedParams, mcConfig, Z, bump);
        result.dVda = V_phi[0];
        result.dVdsigma.assign(V_phi.begin() + 1, V_phi.end());
        
        // =====================================================================
        // Step 2: Build H = JᵀWJ (Gauss-Newton, NOT LM-damped)
        // Use stored Jacobian from calibration
        // =====================================================================
        const auto& J = calibResult.jacobian;       // n_inst × n_params
        const auto& weights = calibResult.weights;  // n_inst
        
        // Compute H = JᵀWJ
        std::vector<std::vector<double>> H(n_params, std::vector<double>(n_params, 0.0));
        for (size_t i = 0; i < n_params; ++i) {
            for (size_t j = 0; j < n_params; ++j) {
                for (size_t k = 0; k < n_inst; ++k) {
                    H[i][j] += J[k][i] * weights[k] * J[k][j];
                }
            }
        }
        
        // Add IFT regularization (tiny, for numerical stability only)
        for (size_t i = 0; i < n_params; ++i) {
            H[i][i] += eps_reg_;
        }
        
        // Validate H dimensions
        validateHessianShape(H, calibResult.dims);
        
        // =====================================================================
        // Step 3: Solve λ from Hᵀλ = V_Φ (IFT adjoint)
        // Since H is symmetric (JᵀWJ), Hᵀ = H
        // Use Cholesky factorization with optional regularization
        // =====================================================================
        CholeskyFactorization chol;
        chol.factor(H, eps_reg_);
        std::vector<double> lambda = chol.solve(V_phi);
        
        result.lambda_norm = norm(lambda);
        
        // =====================================================================
        // Step 4: Vol Greeks using IFT formula
        // dV/dΘ = V_Θ_direct - λᵀ f_Θ
        //
        // For HW1F: V_Θ_direct = 0 (exotic doesn't depend on Black vols)
        // f_Θ = JᵀW(∂r/∂Θ) where ∂r/∂Θ = -∂MarketPrice/∂Θ
        // =====================================================================
        result.volGreeks.resize(n_expiries, std::vector<double>(n_tenors, 0.0));
        result.V_Theta_direct_norm = 0.0;  // Always 0 for HW1F
        
        // Compute Jᵀ for later use
        auto JT = transpose(J);
        
        // Compute ∂r/∂Θ for each vol node using FD
        for (size_t ei = 0; ei < n_expiries; ++ei) {
            for (size_t ti = 0; ti < n_tenors; ++ti) {
                // Bump vol node
                auto bumpedVolSurface = volSurface_.bump(ei, ti, bump);
                
                // Compute ∂r_k/∂Θ_{ei,ti} for each calibration instrument
                std::vector<double> dr_dtheta(n_inst, 0.0);
                for (size_t k = 0; k < n_inst; ++k) {
                    double exp_k = calibInstruments_[k].first;
                    double ten_k = calibInstruments_[k].second;
                    
                    double volBase = volSurface_.atmVol(exp_k, ten_k);
                    double volBumped = bumpedVolSurface.atmVol(exp_k, ten_k);
                    
                    EuropeanSwaption sw(exp_k, ten_k, 0.0, notional_, true);
                    double fwd = value(forwardSwapRate(sw.underlying, curve_));
                    sw.underlying.fixedRate = fwd;
                    
                    double priceBase = value(blackSwaptionPrice(sw, curve_, volBase));
                    double priceBumped = value(blackSwaptionPrice(sw, curve_, volBumped));
                    
                    // r = model - market, so ∂r/∂Θ = -∂market/∂Θ
                    dr_dtheta[k] = -(priceBumped - priceBase) / bump;
                }
                
                // f_Θ = JᵀW(∂r/∂Θ) for this vol node
                std::vector<double> f_theta(n_params, 0.0);
                for (size_t i = 0; i < n_params; ++i) {
                    for (size_t k = 0; k < n_inst; ++k) {
                        f_theta[i] += JT[i][k] * weights[k] * dr_dtheta[k];
                    }
                }
                
                // dV/dΘ = V_Θ_direct - λᵀ f_Θ
                // V_Θ_direct = 0 for HW1F
                double V_Theta_direct = 0.0;
                result.volGreeks[ei][ti] = V_Theta_direct - dot(lambda, f_theta);
            }
        }
        
        // =====================================================================
        // Step 5: Curve Greeks using IFT formula
        // dV/dC = V_C_direct - λᵀ f_C
        //
        // CRITICAL: V_C_direct ≠ 0 (exotic DOES depend on curve directly!)
        // f_C = JᵀW(∂r/∂C)
        // =====================================================================
        result.curveGreeks.resize(n_curve_nodes, 0.0);
        
        // Compute V_C_direct using FD on exotic price
        std::vector<double> V_C_direct(n_curve_nodes, 0.0);
        for (size_t i = 0; i < n_curve_nodes; ++i) {
            auto bumpedCurve = curve_.bump(i, bump);
            double directPrice = pricer.price(swaption, bumpedCurve, Z).price;
            V_C_direct[i] = (directPrice - basePrice) / bump;
        }
        result.V_C_direct_norm = norm(V_C_direct);
        
        // Compute curve Greeks
        // Pre-compute base model and market prices for all instruments
        std::vector<double> modelBaseVec(n_inst), marketBaseVec(n_inst);
        for (size_t k = 0; k < n_inst; ++k) {
            double exp_k = calibInstruments_[k].first;
            double ten_k = calibInstruments_[k].second;
            
            // Model price (Jamshidian)
            EuropeanSwaption swModel(exp_k, ten_k, 0.0, notional_, true);
            double fwd = value(forwardSwapRate(swModel.underlying, curve_));
            swModel.underlying.fixedRate = fwd;
            HW1FModel<double> jamModel(calibratedParams);
            JamshidianPricer<double, CurveReal> jamPricer(jamModel, curve_);
            modelBaseVec[k] = jamPricer.price(swModel);
            
            // Market price (Black)
            double vol = volSurface_.atmVol(exp_k, ten_k);
            marketBaseVec[k] = value(blackSwaptionPrice(swModel, curve_, vol));
        }
        
        for (size_t i = 0; i < n_curve_nodes; ++i) {
            auto bumpedCurve = curve_.bump(i, bump);
            
            // Compute ∂r/∂C_i for each calibration instrument
            std::vector<double> dr_dc(n_inst, 0.0);
            for (size_t k = 0; k < n_inst; ++k) {
                double exp_k = calibInstruments_[k].first;
                double ten_k = calibInstruments_[k].second;
                
                // Model price with bumped curve
                EuropeanSwaption swModel(exp_k, ten_k, 0.0, notional_, true);
                double fwdBumped = value(forwardSwapRate(swModel.underlying, bumpedCurve));
                swModel.underlying.fixedRate = fwdBumped;
                HW1FModel<double> jamModel(calibratedParams);
                JamshidianPricer<double, double> jamPricer(jamModel, bumpedCurve);
                double modelBumped = jamPricer.price(swModel);
                
                // Market price with bumped curve
                EuropeanSwaption swMkt(exp_k, ten_k, 0.0, notional_, true);
                double fwdMkt = value(forwardSwapRate(swMkt.underlying, bumpedCurve));
                swMkt.underlying.fixedRate = fwdMkt;
                double vol = volSurface_.atmVol(exp_k, ten_k);
                double marketBumped = value(blackSwaptionPrice(swMkt, bumpedCurve, vol));
                
                // r = model - market, so ∂r/∂C = ∂model/∂C - ∂market/∂C
                dr_dc[k] = (modelBumped - modelBaseVec[k]) / bump 
                         - (marketBumped - marketBaseVec[k]) / bump;
            }
            
            // f_C = JᵀW(∂r/∂C)
            std::vector<double> f_C(n_params, 0.0);
            for (size_t j = 0; j < n_params; ++j) {
                for (size_t k = 0; k < n_inst; ++k) {
                    f_C[j] += JT[j][k] * weights[k] * dr_dc[k];
                }
            }
            
            // dV/dC = V_C_direct - λᵀ f_C
            result.curveGreeks[i] = V_C_direct[i] - dot(lambda, f_C);
        }
        
        result.elapsedTime = timer.elapsed();
        return result;
    }
    
    // =========================================================================
    // Compute V_Φ = dV/dΦ using finite differences
    // =========================================================================
    std::vector<double> computeVPhiFD(
        const EuropeanSwaption& swaption,
        const HW1FParams& params,
        const MCConfig& mcConfig,
        const std::vector<std::vector<double>>& Z,
        double bump
    ) const {
        size_t n_params = params.numParams();
        std::vector<double> V_phi(n_params, 0.0);
        
        HW1FModel<double> baseModel(params);
        MonteCarloPricer<double> basePricer(baseModel, mcConfig);
        double basePrice = basePricer.price(swaption, curve_, Z).price;
        
        std::vector<double> paramVec = params.toVector();
        
        for (size_t j = 0; j < n_params; ++j) {
            double h = bump * std::max(1.0, std::abs(paramVec[j]));
            
            // Central difference
            std::vector<double> paramUp = paramVec;
            std::vector<double> paramDown = paramVec;
            paramUp[j] += h;
            paramDown[j] -= h;
            
            HW1FParams paramsUp = params;
            paramsUp.fromVector(paramUp);
            HW1FParams paramsDown = params;
            paramsDown.fromVector(paramDown);
            
            HW1FModel<double> modelUp(paramsUp);
            HW1FModel<double> modelDown(paramsDown);
            MonteCarloPricer<double> pricerUp(modelUp, mcConfig);
            MonteCarloPricer<double> pricerDown(modelDown, mcConfig);
            
            double priceUp = pricerUp.price(swaption, curve_, Z).price;
            double priceDown = pricerDown.price(swaption, curve_, Z).price;
            
            V_phi[j] = (priceUp - priceDown) / (2 * h);
        }
        
        return V_phi;
    }

private:
    DiscountCurve<CurveReal> curve_;
    ATMVolSurface<double> volSurface_;
    std::vector<std::pair<double, double>> calibInstruments_;
    double notional_;
    double eps_reg_;  // IFT regularization (NOT LM damping)
};

} // namespace hw1f
