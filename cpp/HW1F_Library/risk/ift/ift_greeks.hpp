#pragma once
// =============================================================================
// IFT-Based Greeks Computation
// Implicit Function Theorem for calibration-aware sensitivities
// Combined with XAD for efficient adjoint differentiation
// =============================================================================

#include "calibration/calibration.hpp"
#include "pricing/jamshidian/jamshidian.hpp"
#include "pricing/montecarlo/montecarlo.hpp"
#include "utils/common.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace hw1f {

// =============================================================================
// Greeks Result Structures
// =============================================================================

struct GreeksResult {
    double price;
    std::vector<double> curveGreeks;      // dV/dr for each curve node
    std::vector<std::vector<double>> volGreeks;  // dV/dvol for each vol node
    double dVda;                           // dV/da
    std::vector<double> dVdsigma;          // dV/dsigma for each sigma bucket
    double elapsedTime;
};

// =============================================================================
// Finite Difference Greeks (Naive - for comparison)
// =============================================================================

template<typename CurveReal = double>
class FDGreeksEngine {
public:
    FDGreeksEngine(
        const DiscountCurve<CurveReal>& curve,
        const ATMVolSurface<double>& volSurface,
        const std::vector<std::pair<double, double>>& calibInstruments,
        double notional = 1e6
    ) : curve_(curve), volSurface_(volSurface), 
        calibInstruments_(calibInstruments), notional_(notional) {}
    
    // Compute Greeks using bump-and-recalibrate (naive FD)
    GreeksResult computeNaiveFD(
        const EuropeanSwaption& swaption,
        const HW1FParams& calibratedParams,
        const MCConfig& mcConfig,
        double bump = 1e-4
    ) const {
        Timer timer;
        timer.start();
        
        GreeksResult result;
        
        // Base price
        HW1FModel<double> model(calibratedParams);
        MonteCarloPricer<double> pricer(model, mcConfig);
        
        // Use fixed random numbers for all pricings
        RNG rng(mcConfig.seed);
        auto Z = rng.normalMatrix(mcConfig.numPaths, mcConfig.numSteps);
        
        double basePrice = pricer.price(swaption, curve_, Z).price;
        result.price = basePrice;
        
        // Curve node Greeks
        size_t numCurveNodes = curve_.size();
        result.curveGreeks.resize(numCurveNodes, 0.0);
        
        for (size_t i = 0; i < numCurveNodes; ++i) {
            // Bump curve node
            auto bumpedCurve = curve_.bump(i, bump);
            
            // Recalibrate
            CalibrationEngine<CurveReal> calibEngine(bumpedCurve, volSurface_, notional_);
            for (const auto& [e, t] : calibInstruments_) {
                calibEngine.addInstrument(e, t);
            }
            auto calibResult = calibEngine.calibrate(calibratedParams, 50, 1e-8, false);
            
            // Reprice
            HW1FModel<double> bumpedModel(calibResult.params);
            MonteCarloPricer<double> bumpedPricer(bumpedModel, mcConfig);
            double bumpedPrice = bumpedPricer.price(swaption, bumpedCurve, Z).price;
            
            result.curveGreeks[i] = (bumpedPrice - basePrice) / bump;
        }
        
        // Vol surface Greeks
        size_t numExpiries = volSurface_.numExpiries();
        size_t numTenors = volSurface_.numTenors();
        result.volGreeks.resize(numExpiries, std::vector<double>(numTenors, 0.0));
        
        for (size_t ei = 0; ei < numExpiries; ++ei) {
            for (size_t ti = 0; ti < numTenors; ++ti) {
                // Bump vol node
                auto bumpedVolSurface = volSurface_.bump(ei, ti, bump);
                
                // Recalibrate
                CalibrationEngine<CurveReal> calibEngine(curve_, bumpedVolSurface, notional_);
                for (const auto& [e, t] : calibInstruments_) {
                    calibEngine.addInstrument(e, t);
                }
                auto calibResult = calibEngine.calibrate(calibratedParams, 50, 1e-8, false);
                
                // Reprice
                HW1FModel<double> bumpedModel(calibResult.params);
                MonteCarloPricer<double> bumpedPricer(bumpedModel, mcConfig);
                double bumpedPrice = bumpedPricer.price(swaption, curve_, Z).price;
                
                result.volGreeks[ei][ti] = (bumpedPrice - basePrice) / bump;
            }
        }
        
        result.elapsedTime = timer.elapsed();
        return result;
    }
    
private:
    DiscountCurve<CurveReal> curve_;
    ATMVolSurface<double> volSurface_;
    std::vector<std::pair<double, double>> calibInstruments_;
    double notional_;
};

// =============================================================================
// Chain Rule Greeks (FD + Chain Rule)
// =============================================================================

template<typename CurveReal = double>
class ChainRuleGreeksEngine {
public:
    ChainRuleGreeksEngine(
        const DiscountCurve<CurveReal>& curve,
        const ATMVolSurface<double>& volSurface,
        const std::vector<std::pair<double, double>>& calibInstruments,
        double notional = 1e6
    ) : curve_(curve), volSurface_(volSurface), 
        calibInstruments_(calibInstruments), notional_(notional) {}
    
    // Compute Greeks using Chain Rule: dV/dm = dV/dphi * dphi/dm
    GreeksResult computeChainRule(
        const EuropeanSwaption& swaption,
        const HW1FParams& calibratedParams,
        const MCConfig& mcConfig,
        double bump = 1e-6
    ) const {
        Timer timer;
        timer.start();
        
        GreeksResult result;
        
        // Fixed random numbers
        RNG rng(mcConfig.seed);
        auto Z = rng.normalMatrix(mcConfig.numPaths, mcConfig.numSteps);
        
        // Compute dV/dphi using FD
        size_t numParams = calibratedParams.numParams();
        std::vector<double> dVdphi(numParams, 0.0);
        
        HW1FModel<double> model(calibratedParams);
        MonteCarloPricer<double> pricer(model, mcConfig);
        double basePrice = pricer.price(swaption, curve_, Z).price;
        result.price = basePrice;
        
        std::vector<double> paramVec = calibratedParams.toVector();
        
        // dV/da and dV/dsigma
        for (size_t j = 0; j < numParams; ++j) {
            double h = bump * std::max(1.0, std::abs(paramVec[j]));
            
            std::vector<double> paramUp = paramVec;
            std::vector<double> paramDown = paramVec;
            paramUp[j] += h;
            paramDown[j] -= h;
            
            HW1FParams paramsUp = calibratedParams;
            paramsUp.fromVector(paramUp);
            HW1FParams paramsDown = calibratedParams;
            paramsDown.fromVector(paramDown);
            
            HW1FModel<double> modelUp(paramsUp);
            HW1FModel<double> modelDown(paramsDown);
            MonteCarloPricer<double> pricerUp(modelUp, mcConfig);
            MonteCarloPricer<double> pricerDown(modelDown, mcConfig);
            
            double priceUp = pricerUp.price(swaption, curve_, Z).price;
            double priceDown = pricerDown.price(swaption, curve_, Z).price;
            
            dVdphi[j] = (priceUp - priceDown) / (2 * h);
        }
        
        result.dVda = dVdphi[0];
        result.dVdsigma.assign(dVdphi.begin() + 1, dVdphi.end());
        
        // Compute dphi/dvol for each vol node using FD on calibration
        size_t numExpiries = volSurface_.numExpiries();
        size_t numTenors = volSurface_.numTenors();
        result.volGreeks.resize(numExpiries, std::vector<double>(numTenors, 0.0));
        
        for (size_t ei = 0; ei < numExpiries; ++ei) {
            for (size_t ti = 0; ti < numTenors; ++ti) {
                // Bump vol node
                auto bumpedVolSurface = volSurface_.bump(ei, ti, bump);
                
                // Recalibrate
                CalibrationEngine<CurveReal> calibEngine(curve_, bumpedVolSurface, notional_);
                for (const auto& [e, t] : calibInstruments_) {
                    calibEngine.addInstrument(e, t);
                }
                auto calibResult = calibEngine.calibrate(calibratedParams, 50, 1e-8, false);
                
                // Get dphi/dvol
                std::vector<double> bumpedParamVec = calibResult.params.toVector();
                
                // dV/dvol = sum_j (dV/dphi_j * dphi_j/dvol)
                double dVdvol = 0.0;
                for (size_t j = 0; j < numParams; ++j) {
                    double dphidvol = (bumpedParamVec[j] - paramVec[j]) / bump;
                    dVdvol += dVdphi[j] * dphidvol;
                }
                
                result.volGreeks[ei][ti] = dVdvol;
            }
        }
        
        // Curve Greeks (similar approach)
        size_t numCurveNodes = curve_.size();
        result.curveGreeks.resize(numCurveNodes, 0.0);
        
        for (size_t i = 0; i < numCurveNodes; ++i) {
            // First compute direct sensitivity dV/dr (price change without recalibration)
            auto bumpedCurve = curve_.bump(i, bump);
            double directPrice = pricer.price(swaption, bumpedCurve, Z).price;
            double dVdr_direct = (directPrice - basePrice) / bump;
            
            // Then compute dphi/dr via recalibration
            CalibrationEngine<CurveReal> calibEngine(bumpedCurve, volSurface_, notional_);
            for (const auto& [e, t] : calibInstruments_) {
                calibEngine.addInstrument(e, t);
            }
            auto calibResult = calibEngine.calibrate(calibratedParams, 50, 1e-8, false);
            
            std::vector<double> bumpedParamVec = calibResult.params.toVector();
            
            // dV/dr = dV/dr_direct + sum_j (dV/dphi_j * dphi_j/dr)
            double dVdr_indirect = 0.0;
            for (size_t j = 0; j < numParams; ++j) {
                double dphidr = (bumpedParamVec[j] - paramVec[j]) / bump;
                dVdr_indirect += dVdphi[j] * dphidr;
            }
            
            result.curveGreeks[i] = dVdr_direct + dVdr_indirect;
        }
        
        result.elapsedTime = timer.elapsed();
        return result;
    }
    
private:
    DiscountCurve<CurveReal> curve_;
    ATMVolSurface<double> volSurface_;
    std::vector<std::pair<double, double>> calibInstruments_;
    double notional_;
};

// =============================================================================
// IFT Greeks Engine (FD for dV/dΦ + OpenGamma Adjoint-IFT)
// Uses Implicit Function Theorem per OpenGamma paper:
//   dV/dm = V_m_direct - λ^T * f_m
//   where λ = solve(f_Φ^T, V_Φ)
// =============================================================================

template<typename CurveReal = double>
class IFTGreeksEngine {
public:
    IFTGreeksEngine(
        const DiscountCurve<CurveReal>& curve,
        const ATMVolSurface<double>& volSurface,
        const std::vector<std::pair<double, double>>& calibInstruments,
        double notional = 1e6
    ) : curve_(curve), volSurface_(volSurface), 
        calibInstruments_(calibInstruments), notional_(notional) {}
    
    // OpenGamma Adjoint-IFT implementation:
    //   f(C, Θ, Φ) = 0  (calibration condition: model prices = market prices)
    //   dV/dΘ = V_Θ_direct - λ^T * f_Θ
    //   dV/dC = V_C_direct - λ^T * f_C
    //   where λ = solve(f_Φ^T, V_Φ)
    GreeksResult computeIFT(
        const EuropeanSwaption& swaption,
        const HW1FParams& calibratedParams,
        const MCConfig& mcConfig,
        double bump = 1e-6
    ) const {
        Timer timer;
        timer.start();
        
        GreeksResult result;
        
        // Fixed random numbers for consistent pricing
        RNG rng(mcConfig.seed);
        auto Z = rng.normalMatrix(mcConfig.numPaths, mcConfig.numSteps);
        
        // =========================================================================
        // Step 1: Compute V_Φ = dV/dΦ using FD (central differences)
        // This is the sensitivity of exotic price to calibrated HW params
        // =========================================================================
        HW1FModel<double> model(calibratedParams);
        MonteCarloPricer<double> pricer(model, mcConfig);
        double basePrice = pricer.price(swaption, curve_, Z).price;
        result.price = basePrice;
        
        size_t numParams = calibratedParams.numParams();
        std::vector<double> paramVec = calibratedParams.toVector();
        
        // V_Φ = dV/dΦ (vector of size K = #calib params)
        std::vector<double> V_phi(numParams, 0.0);
        for (size_t j = 0; j < numParams; ++j) {
            double h = bump * std::max(1.0, std::abs(paramVec[j]));
            
            std::vector<double> paramUp = paramVec;
            std::vector<double> paramDown = paramVec;
            paramUp[j] += h;
            paramDown[j] -= h;
            
            HW1FParams paramsUp = calibratedParams;
            paramsUp.fromVector(paramUp);
            HW1FParams paramsDown = calibratedParams;
            paramsDown.fromVector(paramDown);
            
            HW1FModel<double> modelUp(paramsUp);
            HW1FModel<double> modelDown(paramsDown);
            MonteCarloPricer<double> pricerUp(modelUp, mcConfig);
            MonteCarloPricer<double> pricerDown(modelDown, mcConfig);
            
            double priceUp = pricerUp.price(swaption, curve_, Z).price;
            double priceDown = pricerDown.price(swaption, curve_, Z).price;
            
            V_phi[j] = (priceUp - priceDown) / (2 * h);
        }
        
        result.dVda = V_phi[0];
        result.dVdsigma.assign(V_phi.begin() + 1, V_phi.end());
        
        // =========================================================================
        // Step 2: Build f_Φ = ∂f/∂Φ (calibration Jacobian w.r.t. model params)
        // For least-squares: f = ∇_Φ h = J^T W r, so f_Φ ≈ J^T W J (Gauss-Newton)
        // =========================================================================
        CalibrationEngine<CurveReal> calibEngine(curve_, volSurface_, notional_);
        for (const auto& [e, t] : calibInstruments_) {
            calibEngine.addInstrument(e, t);
        }
        calibEngine.computeMarketPrices();
        
        // J_r[i][j] = ∂r_i/∂Φ_j (residual Jacobian)
        auto Jr = calibEngine.computeJacobianFD(calibratedParams, bump);
        size_t numInst = Jr.size();
        
        // f_Φ ≈ J^T J (Gauss-Newton approximation, ignoring second-order terms)
        auto JrT = transpose(Jr);
        std::vector<std::vector<double>> f_phi(numParams, std::vector<double>(numParams, 0.0));
        for (size_t i = 0; i < numParams; ++i) {
            for (size_t j = 0; j < numParams; ++j) {
                for (size_t k = 0; k < numInst; ++k) {
                    f_phi[i][j] += JrT[i][k] * Jr[k][j];
                }
            }
        }
        
        // Add regularization for numerical stability
        for (size_t i = 0; i < numParams; ++i) {
            f_phi[i][i] += 1e-8;
        }
        
        // =========================================================================
        // Step 3: Solve λ from f_Φ^T λ = V_Φ (OpenGamma adjoint-IFT)
        // Since f_Φ is symmetric (J^T J), we have f_Φ^T = f_Φ
        // =========================================================================
        std::vector<double> lambda = solveCholesky(f_phi, V_phi);
        
        // =========================================================================
        // Step 4: Compute f_Θ = ∂f/∂Θ for all vol nodes
        // For least-squares with f = J^T r:
        //   f_Θ ≈ J^T * (∂r/∂Θ) = J^T * (-∂MarketPrice/∂Θ)
        // Vol Greeks: dV/dΘ = V_Θ_direct - λ^T * f_Θ
        // Since exotic doesn't depend directly on Black vols: V_Θ_direct = 0
        // =========================================================================
        size_t numExpiries = volSurface_.numExpiries();
        size_t numTenors = volSurface_.numTenors();
        result.volGreeks.resize(numExpiries, std::vector<double>(numTenors, 0.0));
        
        for (size_t ei = 0; ei < numExpiries; ++ei) {
            for (size_t ti = 0; ti < numTenors; ++ti) {
                // Compute ∂r/∂Θ for this vol node using FD
                auto bumpedVolSurface = volSurface_.bump(ei, ti, bump);
                
                std::vector<double> dr_dtheta(numInst, 0.0);
                for (size_t k = 0; k < calibInstruments_.size(); ++k) {
                    double exp_k = calibInstruments_[k].first;
                    double ten_k = calibInstruments_[k].second;
                    
                    double volBase = volSurface_.atmVol(exp_k, ten_k);
                    double volBumped = bumpedVolSurface.atmVol(exp_k, ten_k);
                    
                    EuropeanSwaption sw(exp_k, ten_k, 0.0, notional_, true);
                    double fwd = value(forwardSwapRate(sw.underlying, curve_));
                    sw.underlying.fixedRate = fwd;
                    
                    double priceBase = value(blackSwaptionPrice(sw, curve_, volBase));
                    double priceBumped = value(blackSwaptionPrice(sw, curve_, volBumped));
                    
                    // r = modelPrice - marketPrice, so ∂r/∂Θ = -∂marketPrice/∂Θ
                    dr_dtheta[k] = -(priceBumped - priceBase) / bump;
                }
                
                // f_Θ = J^T * (∂r/∂Θ) for this vol node
                std::vector<double> f_theta = matVecMult(JrT, dr_dtheta);
                
                // dV/dΘ = V_Θ_direct - λ^T * f_Θ
                // V_Θ_direct = 0 (exotic price doesn't depend directly on Black vols)
                double dVdtheta = -dot(lambda, f_theta);
                result.volGreeks[ei][ti] = dVdtheta;
            }
        }
        
        // =========================================================================
        // Step 5: Curve Greeks using OpenGamma formula
        // dV/dC = V_C_direct - λ^T * f_C
        // V_C_direct = direct effect of curve on exotic price
        // f_C = J^T * (∂r/∂C) where r = model - market
        // =========================================================================
        size_t numCurveNodes = curve_.size();
        result.curveGreeks.resize(numCurveNodes, 0.0);
        
        for (size_t i = 0; i < numCurveNodes; ++i) {
            // V_C_direct: bump curve, reprice exotic (no recalibration)
            auto bumpedCurve = curve_.bump(i, bump);
            double directPrice = pricer.price(swaption, bumpedCurve, Z).price;
            double V_C_direct = (directPrice - basePrice) / bump;
            
            // Compute ∂r/∂C: curve affects both model price and forward rates
            std::vector<double> dr_dc(numInst, 0.0);
            for (size_t k = 0; k < calibInstruments_.size(); ++k) {
                double exp_k = calibInstruments_[k].first;
                double ten_k = calibInstruments_[k].second;
                
                // Model price with base curve
                double modelBase = calibEngine.modelPrice(calibEngine.instruments()[k], calibratedParams);
                
                // Model price with bumped curve
                CalibrationEngine<CurveReal> bumpedCalibEngine(bumpedCurve, volSurface_, notional_);
                for (const auto& [e2, t2] : calibInstruments_) {
                    bumpedCalibEngine.addInstrument(e2, t2);
                }
                bumpedCalibEngine.computeMarketPrices();
                double modelBumped = bumpedCalibEngine.modelPrice(bumpedCalibEngine.instruments()[k], calibratedParams);
                
                // Market price with bumped curve (changes through forward rate)
                EuropeanSwaption swBase(exp_k, ten_k, 0.0, notional_, true);
                double fwdBase = value(forwardSwapRate(swBase.underlying, curve_));
                swBase.underlying.fixedRate = fwdBase;
                double vol = volSurface_.atmVol(exp_k, ten_k);
                double marketBase = value(blackSwaptionPrice(swBase, curve_, vol));
                
                EuropeanSwaption swBumped(exp_k, ten_k, 0.0, notional_, true);
                double fwdBumped = value(forwardSwapRate(swBumped.underlying, bumpedCurve));
                swBumped.underlying.fixedRate = fwdBumped;
                double marketBumped = value(blackSwaptionPrice(swBumped, bumpedCurve, vol));
                
                // r = model - market, so ∂r/∂C = ∂model/∂C - ∂market/∂C
                dr_dc[k] = (modelBumped - modelBase) / bump - (marketBumped - marketBase) / bump;
            }
            
            // f_C = J^T * (∂r/∂C)
            std::vector<double> f_C = matVecMult(JrT, dr_dc);
            
            // dV/dC = V_C_direct - λ^T * f_C
            result.curveGreeks[i] = V_C_direct - dot(lambda, f_C);
        }
        
        result.elapsedTime = timer.elapsed();
        return result;
    }
    
private:
    DiscountCurve<CurveReal> curve_;
    ATMVolSurface<double> volSurface_;
    std::vector<std::pair<double, double>> calibInstruments_;
    double notional_;
};

// =============================================================================
// XAD + IFT Greeks Engine (Optimized Full AD)
// Uses XAD adjoint mode for:
//   1. dV/dphi - price sensitivities to HW params (ONE backward pass vs 2K FD)
//   2. df/dm   - calibration Jacobian w.r.t. market data (K backward passes)
// 
// Key optimizations over naive approach:
//   - V_Φ computed in 1 pass (vs 2K MC pricings in FD+IFT)
//   - Curve Greeks use IFT formula (no per-node MC repricing)
//   - Pre-compute bond pricing components outside AD tape
// =============================================================================

template<typename CurveReal = double>
class XADIFTGreeksEngine {
public:
    using ADReal = xad::AReal<double>;
    using Tape = xad::Tape<double>;
    
    XADIFTGreeksEngine(
        const DiscountCurve<CurveReal>& curve,
        const ATMVolSurface<double>& volSurface,
        const std::vector<std::pair<double, double>>& calibInstruments,
        double notional = 1e6
    ) : curve_(curve), volSurface_(volSurface), 
        calibInstruments_(calibInstruments), notional_(notional) {}
    
    // ==========================================================================
    // OPTIMIZED: Compute dV/dphi using XAD adjoint mode
    // Key insight: Pre-compute everything that doesn't depend on HW params
    // outside the tape, minimizing tape size and backward pass cost
    // ==========================================================================
    std::pair<double, std::vector<double>> computeDVDphiXAD(
        const EuropeanSwaption& swaption,
        const HW1FParams& baseParams,
        const MCConfig& mcConfig,
        const std::vector<std::vector<double>>& Z
    ) const {
        // Pre-compute values that don't depend on HW params
        double expiry = swaption.expiry;
        double dt = expiry / mcConfig.numSteps;
        double df0E = value(curve_.df(expiry));
        
        const VanillaSwap& swap = swaption.underlying;
        Schedule fixedSched = swap.fixedSchedule();
        size_t n = fixedSched.paymentDates.size();
        
        // Pre-compute market discount factors (don't depend on HW params)
        std::vector<double> P_0_Ti(n), df_ratios(n), accr_factors(n);
        double P_0_E = value(curve_.df(expiry));
        for (size_t i = 0; i < n; ++i) {
            P_0_Ti[i] = value(curve_.df(fixedSched.paymentDates[i]));
            df_ratios[i] = P_0_Ti[i] / P_0_E;  // P(0,T_i)/P(0,E)
            accr_factors[i] = fixedSched.accrualFactors[i] * swap.fixedRate * swap.notional;
        }
        double df_ratio_n = P_0_Ti[n-1] / P_0_E;
        
        // Pre-compute time grid for buckets
        const auto& sigmaTimes = baseParams.sigmaTimes;
        size_t numSigma = baseParams.sigmaValues.size();
        
        // Pre-compute bucket indices for each time step (no AD needed)
        std::vector<size_t> stepBuckets(mcConfig.numSteps);
        for (int step = 0; step < mcConfig.numSteps; ++step) {
            double t = step * dt;
            stepBuckets[step] = 0;
            for (size_t i = sigmaTimes.size() - 1; i > 0; --i) {
                if (t >= sigmaTimes[i]) {
                    stepBuckets[step] = i;
                    break;
                }
            }
        }
        
        // Pre-compute bond maturities relative to expiry
        std::vector<double> T_minus_E(n);
        for (size_t i = 0; i < n; ++i) {
            T_minus_E[i] = fixedSched.paymentDates[i] - expiry;
        }
        
        // NOW start the AD tape with minimal operations
        Tape tape;
        
        size_t numParams = baseParams.numParams();
        std::vector<ADReal> params_ad(numParams);
        std::vector<double> paramVec = baseParams.toVector();
        
        for (size_t j = 0; j < numParams; ++j) {
            params_ad[j] = paramVec[j];
            tape.registerInput(params_ad[j]);
        }
        tape.newRecording();
        
        ADReal a_ad = params_ad[0];
        std::vector<ADReal> sigma_ad(numSigma);
        for (size_t j = 0; j < numSigma; ++j) {
            sigma_ad[j] = params_ad[j + 1];
        }
        
        // Pre-compute V_r(0, E) on tape (only once, used for all bond prices)
        ADReal V_0_E = computeV_rAD(0.0, expiry, a_ad, sigmaTimes, sigma_ad);
        
        // Pre-compute B(E, T_i) for all payment dates (depends on a)
        std::vector<ADReal> B_E_Ti(n);
        for (size_t i = 0; i < n; ++i) {
            if (xad::value(a_ad) < 1e-8) {
                B_E_Ti[i] = ADReal(T_minus_E[i]);
            } else {
                B_E_Ti[i] = (ADReal(1.0) - exp(-a_ad * ADReal(T_minus_E[i]))) / a_ad;
            }
        }
        
        // Pre-compute step decay factors (depend on a)
        ADReal decay = exp(-a_ad * dt);
        
        // Simulate
        ADReal sumPayoff = ADReal(0.0);
        int effectivePaths = mcConfig.antithetic ? mcConfig.numPaths / 2 : mcConfig.numPaths;
        
        for (int path = 0; path < effectivePaths; ++path) {
            for (int anti = 0; anti < (mcConfig.antithetic ? 2 : 1); ++anti) {
                ADReal x = ADReal(0.0);
                
                // Path simulation with optimized loop
                for (int step = 0; step < mcConfig.numSteps; ++step) {
                    double t = step * dt;
                    double t_next = t + dt;
                    
                    // Use pre-computed bucket index
                    ADReal sigma_t = sigma_ad[stepBuckets[step]];
                    
                    double z = Z[path][step];
                    if (anti == 1) z = -z;
                    
                    // Optimized V_r for single step (typically within one bucket)
                    ADReal V_step = computeV_rStepAD(t, t_next, a_ad, sigmaTimes, sigma_ad, stepBuckets[step]);
                    x = x * decay + sqrt(V_step) * z;
                }
                
                // Compute swap PV using pre-computed components
                // σ_P²(E, T_i) = B(E,T_i)² * V_r(0,E)
                // A(E, T_i) = (P_0_Ti / P_0_E) * exp(-0.5 * σ_P²)
                // P(E, T_i) = A * exp(-B * x)
                
                // Float leg: N * (1 - P(E, T_n))
                ADReal sigmaP_sq_n = B_E_Ti[n-1] * B_E_Ti[n-1] * V_0_E;
                ADReal A_n = ADReal(df_ratio_n) * exp(ADReal(-0.5) * sigmaP_sq_n);
                ADReal P_E_Tn = A_n * exp(-B_E_Ti[n-1] * x);
                ADReal floatPV = ADReal(swap.notional) * (ADReal(1.0) - P_E_Tn);
                
                // Fixed leg
                ADReal fixedPV = ADReal(0.0);
                for (size_t i = 0; i < n; ++i) {
                    ADReal sigmaP_sq = B_E_Ti[i] * B_E_Ti[i] * V_0_E;
                    ADReal A_i = ADReal(df_ratios[i]) * exp(ADReal(-0.5) * sigmaP_sq);
                    ADReal P_E_Ti = A_i * exp(-B_E_Ti[i] * x);
                    fixedPV = fixedPV + ADReal(accr_factors[i]) * P_E_Ti;
                }
                
                ADReal swapPV = floatPV - fixedPV;
                ADReal payoff = xad::max(swapPV, ADReal(0.0)) * df0E;
                sumPayoff = sumPayoff + payoff;
            }
        }
        
        int totalPaths = effectivePaths * (mcConfig.antithetic ? 2 : 1);
        ADReal price_ad = sumPayoff / ADReal(totalPaths);
        
        tape.registerOutput(price_ad);
        xad::derivative(price_ad) = 1.0;
        tape.computeAdjoints();
        
        double price = xad::value(price_ad);
        std::vector<double> dVdphi(numParams);
        for (size_t j = 0; j < numParams; ++j) {
            dVdphi[j] = xad::derivative(params_ad[j]);
        }
        
        return {price, dVdphi};
    }
    
    // Optimized V_r for a single step (often stays within one bucket)
    ADReal computeV_rStepAD(double s, double t, ADReal a,
                            const std::vector<double>& sigmaTimes,
                            const std::vector<ADReal>& sigma_ad,
                            size_t hintBucket) const {
        if (t <= s) return ADReal(0.0);
        
        // Fast path: check if entire interval is in hint bucket
        size_t numBuckets = sigma_ad.size();
        double bucket_end = (hintBucket + 1 < sigmaTimes.size()) ? sigmaTimes[hintBucket + 1] : 1e10;
        
        if (s >= sigmaTimes[hintBucket] && t <= bucket_end) {
            // Entire step within one bucket - fast path
            ADReal sigma_i = sigma_ad[hintBucket];
            if (xad::value(a) < 1e-8) {
                return sigma_i * sigma_i * ADReal(t - s);
            } else {
                ADReal exp_2a_end = exp(ADReal(-2.0) * a * ADReal(t - t));  // = 1
                ADReal exp_2a_start = exp(ADReal(-2.0) * a * ADReal(t - s));
                ADReal integral = (ADReal(1.0) - exp_2a_start) / (ADReal(2.0) * a);
                return sigma_i * sigma_i * integral;
            }
        }
        
        // Slow path: step crosses bucket boundaries
        return computeV_rAD(s, t, a, sigmaTimes, sigma_ad);
    }
    
    // Get sigma at time t from piecewise-constant buckets (AD version)
    ADReal getSigmaAtTimeAD(double t, const std::vector<double>& sigmaTimes, 
                            const std::vector<ADReal>& sigma_ad) const {
        for (size_t i = sigmaTimes.size() - 1; i > 0; --i) {
            if (t >= sigmaTimes[i]) {
                return sigma_ad[i];
            }
        }
        return sigma_ad[0];
    }
    
    // Compute V_r(s, t) = ∫_s^t e^{-2a(t-u)} σ(u)² du with AD types
    ADReal computeV_rAD(double s, double t, ADReal a,
                        const std::vector<double>& sigmaTimes,
                        const std::vector<ADReal>& sigma_ad) const {
        if (t <= s) return ADReal(0.0);
        
        ADReal result = ADReal(0.0);
        size_t numBuckets = sigma_ad.size();
        
        for (size_t i = 0; i < numBuckets; ++i) {
            double bucket_start = sigmaTimes[i];
            double bucket_end = (i + 1 < sigmaTimes.size()) ? sigmaTimes[i + 1] : 1e10;
            
            double t_start = std::max(bucket_start, s);
            double t_end = std::min(bucket_end, t);
            
            if (t_end <= t_start) continue;
            
            ADReal sigma_i = sigma_ad[i];
            
            if (xad::value(a) < 1e-8) {
                result = result + sigma_i * sigma_i * ADReal(t_end - t_start);
            } else {
                ADReal exp_2a_end = exp(ADReal(-2.0) * a * ADReal(t - t_end));
                ADReal exp_2a_start = exp(ADReal(-2.0) * a * ADReal(t - t_start));
                ADReal integral = (exp_2a_end - exp_2a_start) / (ADReal(2.0) * a);
                result = result + sigma_i * sigma_i * integral;
            }
        }
        
        return result;
    }
    
    // Compute FULL HW1F bond price with AD types (for backward compatibility)
    ADReal computeBondPriceFullAD(
        double t, double T, 
        ADReal x_t,
        ADReal a, 
        const std::vector<double>& sigmaTimes,
        const std::vector<ADReal>& sigma_ad
    ) const {
        double P_0_T = value(curve_.df(T));
        double P_0_t = value(curve_.df(t));
        
        ADReal B_t_T;
        if (xad::value(a) < 1e-8) {
            B_t_T = ADReal(T - t);
        } else {
            B_t_T = (ADReal(1.0) - exp(-a * ADReal(T - t))) / a;
        }
        
        // V_r(0, t) - variance from time 0 to t
        ADReal V_0_t = computeV_rAD(0.0, t, a, sigmaTimes, sigma_ad);
        
        // σ_P²(t, T) = B(t,T)² * V_r(0,t)
        ADReal sigmaP_sq = B_t_T * B_t_T * V_0_t;
        
        // A(t,T) = P(0,T)/P(0,t) * exp(-0.5 * σ_P²)
        ADReal A_t_T = ADReal(P_0_T / P_0_t) * exp(ADReal(-0.5) * sigmaP_sq);
        
        return A_t_T * exp(-B_t_T * x_t);
    }
    
    // =========================================================================
    // OpenGamma Adjoint-IFT with XAD
    // Uses XAD for:
    //   1. V_Φ = dV/dΦ (one backward pass through MC pricing)
    //   2. f_Θ = ∂f/∂Θ for ALL vol nodes (K backward passes, K = #instruments)
    // Then: dV/dΘ = V_Θ_direct - λ^T * f_Θ where λ = solve(f_Φ^T, V_Φ)
    // =========================================================================
    GreeksResult computeXADIFT(
        const EuropeanSwaption& swaption,
        const HW1FParams& calibratedParams,
        const MCConfig& mcConfig,
        double bump = 1e-6
    ) const {
        Timer timer;
        timer.start();
        
        GreeksResult result;
        
        // Fixed random numbers
        RNG rng(mcConfig.seed);
        auto Z = rng.normalMatrix(mcConfig.numPaths, mcConfig.numSteps);
        
        size_t numParams = calibratedParams.numParams();
        std::vector<double> paramVec = calibratedParams.toVector();
        
        size_t numExpiries = volSurface_.numExpiries();
        size_t numTenors = volSurface_.numTenors();
        size_t numVolNodes = numExpiries * numTenors;
        size_t numCurveNodes = curve_.size();
        
        // =========================================================================
        // Step 1: Compute V_Φ = dV/dΦ using XAD adjoint (one backward pass)
        // =========================================================================
        auto [price, V_phi] = computeDVDphiXAD(swaption, calibratedParams, mcConfig, Z);
        result.price = price;
        result.dVda = V_phi[0];
        result.dVdsigma.assign(V_phi.begin() + 1, V_phi.end());
        
        // =========================================================================
        // Step 2: Build f_Φ = J^T * J (Gauss-Newton Hessian approximation)
        // =========================================================================
        CalibrationEngine<CurveReal> calibEngine(curve_, volSurface_, notional_);
        for (const auto& [e, t] : calibInstruments_) {
            calibEngine.addInstrument(e, t);
        }
        calibEngine.computeMarketPrices();
        
        // Jr[i][j] = ∂r_i/∂Φ_j
        auto Jr = calibEngine.computeJacobianFD(calibratedParams, bump);
        size_t numInst = Jr.size();
        
        auto JrT = transpose(Jr);
        
        // f_Φ ≈ J^T * J
        std::vector<std::vector<double>> f_phi(numParams, std::vector<double>(numParams, 0.0));
        for (size_t i = 0; i < numParams; ++i) {
            for (size_t j = 0; j < numParams; ++j) {
                for (size_t k = 0; k < numInst; ++k) {
                    f_phi[i][j] += JrT[i][k] * Jr[k][j];
                }
            }
        }
        
        // Add regularization
        for (size_t i = 0; i < numParams; ++i) {
            f_phi[i][i] += 1e-8;
        }
        
        // =========================================================================
        // Step 3: Solve λ from f_Φ^T λ = V_Φ (OpenGamma adjoint-IFT, ONE solve!)
        // Since f_Φ is symmetric (J^T J), f_Φ^T = f_Φ
        // =========================================================================
        std::vector<double> lambda = solveCholesky(f_phi, V_phi);
        
        // =========================================================================
        // Step 4: Compute f_Θ for ALL vol nodes using XAD
        // For least-squares: f = J^T r, so f_Θ = J^T * (∂r/∂Θ)
        // where ∂r/∂Θ = -∂MarketPrice/∂Θ
        // Use XAD: K backward passes (one per calibration instrument)
        // Each pass gives ∂MarketPrice_k/∂Θ for ALL vol nodes
        // =========================================================================
        
        // dr_dm_vol[k][idx] = ∂r_k/∂Θ_idx = -∂MarketPrice_k/∂Θ_idx
        std::vector<std::vector<double>> dr_dm_vol(numInst, std::vector<double>(numVolNodes, 0.0));
        
        for (size_t k = 0; k < numInst; ++k) {
            Tape tape;
            
            // Register all vol nodes as inputs
            std::vector<ADReal> volNodes_ad(numVolNodes);
            for (size_t ei = 0; ei < numExpiries; ++ei) {
                for (size_t ti = 0; ti < numTenors; ++ti) {
                    size_t idx = ei * numTenors + ti;
                    volNodes_ad[idx] = volSurface_.vol(ei, ti);
                }
            }
            
            for (size_t idx = 0; idx < numVolNodes; ++idx) {
                tape.registerInput(volNodes_ad[idx]);
            }
            tape.newRecording();
            
            // Interpolate vol for this calibration instrument
            double expiry_k = calibInstruments_[k].first;
            double tenor_k = calibInstruments_[k].second;
            ADReal vol_ad = interpolateVolAD(expiry_k, tenor_k, volNodes_ad, numExpiries, numTenors);
            
            // Black price with AD vol
            EuropeanSwaption sw(expiry_k, tenor_k, 0.0, notional_, true);
            double fwd = value(forwardSwapRate(sw.underlying, curve_));
            sw.underlying.fixedRate = fwd;
            ADReal blackPrice = blackSwaptionPriceAD(sw, curve_, vol_ad);
            
            // Backward pass
            tape.registerOutput(blackPrice);
            xad::derivative(blackPrice) = 1.0;
            tape.computeAdjoints();
            
            // ∂r_k/∂Θ = -∂MarketPrice_k/∂Θ
            for (size_t idx = 0; idx < numVolNodes; ++idx) {
                dr_dm_vol[k][idx] = -xad::derivative(volNodes_ad[idx]);
            }
        }
        
        // =========================================================================
        // Step 5: Vol Greeks using OpenGamma formula
        // dV/dΘ = V_Θ_direct - λ^T * f_Θ
        // V_Θ_direct = 0 (exotic price doesn't depend directly on Black vols)
        // f_Θ = J^T * (∂r/∂Θ) for each vol node
        // =========================================================================
        result.volGreeks.resize(numExpiries, std::vector<double>(numTenors, 0.0));
        
        for (size_t ei = 0; ei < numExpiries; ++ei) {
            for (size_t ti = 0; ti < numTenors; ++ti) {
                size_t idx = ei * numTenors + ti;
                
                // Get ∂r/∂Θ for this vol node (column of the Jacobian w.r.t. this node)
                std::vector<double> dr_dtheta(numInst);
                for (size_t k = 0; k < numInst; ++k) {
                    dr_dtheta[k] = dr_dm_vol[k][idx];
                }
                
                // f_Θ = J^T * (∂r/∂Θ)
                std::vector<double> f_theta = matVecMult(JrT, dr_dtheta);
                
                // dV/dΘ = V_Θ_direct - λ^T * f_Θ = 0 - λ^T * f_Θ
                result.volGreeks[ei][ti] = -dot(lambda, f_theta);
            }
        }
        
        // =========================================================================
        // Step 6: Curve Greeks using OpenGamma formula (FULLY OPTIMIZED)
        // dV/dC = V_C_direct - λ^T * f_C
        // 
        // OPTIMIZATION: Use XAD to compute V_C_direct for ALL curve nodes
        // in a SINGLE backward pass, instead of N separate MC pricings!
        // This is the KEY optimization that makes XAD+IFT much faster.
        // =========================================================================
        result.curveGreeks.resize(numCurveNodes, 0.0);
        
        // Pre-compute base model and market prices for all instruments (cheap: Jamshidian)
        std::vector<double> modelBaseVec(numInst), marketBaseVec(numInst);
        for (size_t k = 0; k < numInst; ++k) {
            modelBaseVec[k] = calibEngine.modelPrice(calibEngine.instruments()[k], calibratedParams);
            
            double exp_k = calibInstruments_[k].first;
            double ten_k = calibInstruments_[k].second;
            EuropeanSwaption sw(exp_k, ten_k, 0.0, notional_, true);
            double fwd = value(forwardSwapRate(sw.underlying, curve_));
            sw.underlying.fixedRate = fwd;
            double vol = volSurface_.atmVol(exp_k, ten_k);
            marketBaseVec[k] = value(blackSwaptionPrice(sw, curve_, vol));
        }
        
        // =====================================================================
        // OPTIMIZED: Compute V_C_direct for ALL curve nodes using XAD
        // One backward pass gives dV/d(all curve nodes) simultaneously!
        // =====================================================================
        std::vector<double> V_C_direct_all = computeVCdirectXAD(swaption, calibratedParams, mcConfig, Z);
        
        // Curve Greeks loop - NOW only needs FD for f_C (cheap: Jamshidian only)
        for (size_t i = 0; i < numCurveNodes; ++i) {
            double V_C_direct = V_C_direct_all[i];
            
            // Compute ∂r/∂C using FD on Jamshidian prices (FAST - no MC!)
            auto bumpedCurve = curve_.bump(i, bump);
            std::vector<double> dr_dc(numInst, 0.0);
            
            for (size_t k = 0; k < numInst; ++k) {
                double exp_k = calibInstruments_[k].first;
                double ten_k = calibInstruments_[k].second;
                
                // Model price with bumped curve (Jamshidian - fast!)
                EuropeanSwaption swModel(exp_k, ten_k, 0.0, notional_, true);
                double fwdModel = value(forwardSwapRate(swModel.underlying, bumpedCurve));
                swModel.underlying.fixedRate = fwdModel;
                HW1FModel<double> jamModel(calibratedParams);
                JamshidianPricer<double, double> jamPricer(jamModel, bumpedCurve);
                double modelBumped = jamPricer.price(swModel);
                
                // Market price with bumped curve
                EuropeanSwaption swMkt(exp_k, ten_k, 0.0, notional_, true);
                double fwdMkt = value(forwardSwapRate(swMkt.underlying, bumpedCurve));
                swMkt.underlying.fixedRate = fwdMkt;
                double vol = volSurface_.atmVol(exp_k, ten_k);
                double marketBumped = value(blackSwaptionPrice(swMkt, bumpedCurve, vol));
                
                dr_dc[k] = (modelBumped - modelBaseVec[k]) / bump - (marketBumped - marketBaseVec[k]) / bump;
            }
            
            // f_C = J^T * (∂r/∂C)
            std::vector<double> f_C = matVecMult(JrT, dr_dc);
            
            // dV/dC = V_C_direct - λ^T * f_C
            result.curveGreeks[i] = V_C_direct - dot(lambda, f_C);
        }
        
        result.elapsedTime = timer.elapsed();
        return result;
    }
    
    // Helper: Interpolate vol from AD vol surface (flat interpolation for simplicity)
    ADReal interpolateVolAD(
        double expiry, double tenor,
        const std::vector<ADReal>& volNodes,
        size_t numExpiries, size_t numTenors
    ) const {
        // Find closest expiry and tenor indices
        size_t ei = 0, ti = 0;
        double minExpDist = 1e10, minTenDist = 1e10;
        
        for (size_t i = 0; i < numExpiries; ++i) {
            double dist = std::abs(volSurface_.expiry(i) - expiry);
            if (dist < minExpDist) {
                minExpDist = dist;
                ei = i;
            }
        }
        for (size_t i = 0; i < numTenors; ++i) {
            double dist = std::abs(volSurface_.tenor(i) - tenor);
            if (dist < minTenDist) {
                minTenDist = dist;
                ti = i;
            }
        }
        
        return volNodes[ei * numTenors + ti];
    }
    
    // ==========================================================================
    // OPTIMIZED: Compute V_C_direct for ALL curve nodes in ONE backward pass
    // This is the KEY optimization - instead of N MC pricings, we do ONE!
    // ==========================================================================
    std::vector<double> computeVCdirectXAD(
        const EuropeanSwaption& swaption,
        const HW1FParams& baseParams,
        const MCConfig& mcConfig,
        const std::vector<std::vector<double>>& Z
    ) const {
        double expiry = swaption.expiry;
        double dt = expiry / mcConfig.numSteps;
        
        const VanillaSwap& swap = swaption.underlying;
        Schedule fixedSched = swap.fixedSchedule();
        size_t n = fixedSched.paymentDates.size();
        size_t numCurveNodes = curve_.size();
        
        // Pre-compute time grid for sigma buckets
        const auto& sigmaTimes = baseParams.sigmaTimes;
        size_t numSigma = baseParams.sigmaValues.size();
        double a = baseParams.a;
        
        // Pre-compute step info (no AD needed)
        std::vector<size_t> stepBuckets(mcConfig.numSteps);
        for (int step = 0; step < mcConfig.numSteps; ++step) {
            double t = step * dt;
            stepBuckets[step] = 0;
            for (size_t i = sigmaTimes.size() - 1; i > 0; --i) {
                if (t >= sigmaTimes[i]) {
                    stepBuckets[step] = i;
                    break;
                }
            }
        }
        
        // Bond maturities relative to expiry
        std::vector<double> T_minus_E(n);
        for (size_t i = 0; i < n; ++i) {
            T_minus_E[i] = fixedSched.paymentDates[i] - expiry;
        }
        
        // START XAD TAPE - register curve discount factors as inputs
        Tape tape;
        
        std::vector<ADReal> df_ad(numCurveNodes);
        for (size_t i = 0; i < numCurveNodes; ++i) {
            df_ad[i] = curve_.dfAt(i);  // Get DF at node i
            tape.registerInput(df_ad[i]);
        }
        tape.newRecording();
        
        // Interpolate DFs at key times using curve nodes (log-linear)
        auto interpDF = [&](double t) -> ADReal {
            // Find bracketing nodes
            size_t lo = 0, hi = numCurveNodes - 1;
            for (size_t i = 0; i < numCurveNodes - 1; ++i) {
                if (curve_.timeAt(i) <= t && t <= curve_.timeAt(i+1)) {
                    lo = i;
                    hi = i + 1;
                    break;
                }
            }
            if (t <= curve_.timeAt(0)) return df_ad[0];
            if (t >= curve_.timeAt(numCurveNodes-1)) return df_ad[numCurveNodes-1];
            
            double t_lo = curve_.timeAt(lo);
            double t_hi = curve_.timeAt(hi);
            double w = (t - t_lo) / (t_hi - t_lo);
            
            // Log-linear interpolation
            return exp((ADReal(1.0) - ADReal(w)) * log(df_ad[lo]) + ADReal(w) * log(df_ad[hi]));
        };
        
        // Get interpolated DFs for expiry and payment dates
        ADReal P_0_E = interpDF(expiry);
        std::vector<ADReal> P_0_Ti(n);
        for (size_t i = 0; i < n; ++i) {
            P_0_Ti[i] = interpDF(fixedSched.paymentDates[i]);
        }
        
        // Pre-compute DF ratios
        std::vector<ADReal> df_ratios(n);
        for (size_t i = 0; i < n; ++i) {
            df_ratios[i] = P_0_Ti[i] / P_0_E;
        }
        ADReal df_ratio_n = df_ratios[n-1];
        
        // Pre-compute V_r(0, E) (constant w.r.t. curve)
        double V_0_E = 0.0;
        for (size_t i = 0; i < numSigma; ++i) {
            double t_start = sigmaTimes[i];
            double t_end = (i + 1 < sigmaTimes.size()) ? sigmaTimes[i + 1] : 1e10;
            t_start = std::max(0.0, t_start);
            t_end = std::min(t_end, expiry);
            if (t_end <= t_start) continue;
            
            double sigma_i = baseParams.sigmaValues[i];
            if (std::abs(a) < 1e-8) {
                V_0_E += sigma_i * sigma_i * (t_end - t_start);
            } else {
                double exp_2a_end = std::exp(-2 * a * (expiry - t_end));
                double exp_2a_start = std::exp(-2 * a * (expiry - t_start));
                V_0_E += sigma_i * sigma_i * (exp_2a_end - exp_2a_start) / (2.0 * a);
            }
        }
        
        // Pre-compute B(E, T_i) values (constant)
        std::vector<double> B_E_Ti(n);
        for (size_t i = 0; i < n; ++i) {
            if (std::abs(a) < 1e-8) {
                B_E_Ti[i] = T_minus_E[i];
            } else {
                B_E_Ti[i] = (1.0 - std::exp(-a * T_minus_E[i])) / a;
            }
        }
        
        // Pre-compute sigma_P squared (constant)
        std::vector<double> sigmaP_sq(n);
        for (size_t i = 0; i < n; ++i) {
            sigmaP_sq[i] = B_E_Ti[i] * B_E_Ti[i] * V_0_E;
        }
        
        // Pre-compute step parameters (constant)
        double decay = std::exp(-a * dt);
        std::vector<double> sqrt_V_step(mcConfig.numSteps);
        for (int step = 0; step < mcConfig.numSteps; ++step) {
            double t = step * dt;
            double t_next = t + dt;
            // Single bucket approximation
            double sigma_t = baseParams.sigmaValues[stepBuckets[step]];
            double V_step;
            if (std::abs(a) < 1e-8) {
                V_step = sigma_t * sigma_t * dt;
            } else {
                V_step = sigma_t * sigma_t * (1.0 - std::exp(-2.0 * a * dt)) / (2.0 * a);
            }
            sqrt_V_step[step] = std::sqrt(V_step);
        }
        
        // Monte Carlo simulation with AD on curve-dependent quantities
        ADReal sumPayoff = ADReal(0.0);
        int effectivePaths = mcConfig.antithetic ? mcConfig.numPaths / 2 : mcConfig.numPaths;
        
        for (int path = 0; path < effectivePaths; ++path) {
            for (int anti = 0; anti < (mcConfig.antithetic ? 2 : 1); ++anti) {
                double x = 0.0;  // State x doesn't depend on curve
                
                // Path simulation (curve-independent)
                for (int step = 0; step < mcConfig.numSteps; ++step) {
                    double z = Z[path][step];
                    if (anti == 1) z = -z;
                    x = x * decay + sqrt_V_step[step] * z;
                }
                
                // Compute swap PV at expiry - CURVE DEPENDENT through df_ratios!
                // Float leg: N * (1 - P(E, T_n))
                // A(E, T_n) = df_ratio_n * exp(-0.5 * sigmaP_sq_n)
                ADReal A_n = df_ratio_n * exp(ADReal(-0.5 * sigmaP_sq[n-1]));
                ADReal P_E_Tn = A_n * exp(ADReal(-B_E_Ti[n-1] * x));
                ADReal floatPV = ADReal(swap.notional) * (ADReal(1.0) - P_E_Tn);
                
                // Fixed leg
                ADReal fixedPV = ADReal(0.0);
                for (size_t i = 0; i < n; ++i) {
                    ADReal A_i = df_ratios[i] * exp(ADReal(-0.5 * sigmaP_sq[i]));
                    ADReal P_E_Ti = A_i * exp(ADReal(-B_E_Ti[i] * x));
                    fixedPV = fixedPV + ADReal(fixedSched.accrualFactors[i] * swap.fixedRate * swap.notional) * P_E_Ti;
                }
                
                ADReal swapPV = floatPV - fixedPV;
                ADReal payoff = xad::max(swapPV, ADReal(0.0)) * xad::value(P_0_E);
                sumPayoff = sumPayoff + payoff;
            }
        }
        
        int totalPaths = effectivePaths * (mcConfig.antithetic ? 2 : 1);
        ADReal price_ad = sumPayoff / ADReal(totalPaths);
        
        // Backward pass to get dV/d(curve nodes)
        tape.registerOutput(price_ad);
        xad::derivative(price_ad) = 1.0;
        tape.computeAdjoints();
        
        // Convert dV/d(DF) to dV/d(zero rate) using chain rule:
        // dV/dr_i = dV/d(DF_i) * d(DF_i)/dr_i
        // where DF_i = exp(-r_i * t_i), so d(DF_i)/dr_i = -t_i * DF_i
        std::vector<double> V_C_direct(numCurveNodes);
        for (size_t i = 0; i < numCurveNodes; ++i) {
            double t_i = curve_.timeAt(i);
            double df_i = value(curve_.dfAt(i));
            double dDF_dr = -t_i * df_i;  // Chain rule factor
            V_C_direct[i] = xad::derivative(df_ad[i]) * dDF_dr;
        }
        
        return V_C_direct;
    }
    
    // Helper: Black swaption price with AD vol
    ADReal blackSwaptionPriceAD(
        const EuropeanSwaption& swaption,
        const DiscountCurve<CurveReal>& curve,
        ADReal vol
    ) const {
        const VanillaSwap& swap = swaption.underlying;
        double expiry = swaption.expiry;
        
        // Forward swap rate and annuity (use double values)
        double fwd = value(forwardSwapRate(swap, curve));
        double annuity = value(swapAnnuity(swap, curve));
        
        double K = swap.fixedRate;
        double sqrtT = std::sqrt(expiry);
        
        ADReal d1 = (log(ADReal(fwd / K)) + ADReal(0.5) * vol * vol * ADReal(expiry)) / (vol * ADReal(sqrtT));
        ADReal d2 = d1 - vol * ADReal(sqrtT);
        
        // Standard normal CDF approximation
        ADReal Nd1 = normalCdfAD(d1);
        ADReal Nd2 = normalCdfAD(d2);
        
        ADReal price = ADReal(annuity * swap.notional) * (ADReal(fwd) * Nd1 - ADReal(K) * Nd2);
        
        return price;
    }
    
    // Normal CDF approximation for AD types (Horner form for efficiency)
    // Uses Hart approximation which is smooth and AD-friendly
    ADReal normalCdfAD(ADReal x) const {
        // Hart approximation (smooth, no branching needed for AD)
        const double a1 =  0.254829592;
        const double a2 = -0.284496736;
        const double a3 =  1.421413741;
        const double a4 = -1.453152027;
        const double a5 =  1.061405429;
        const double p  =  0.3275911;
        
        // Work with absolute value, use symmetry at the end
        double xv = xad::value(x);
        double sign = (xv >= 0) ? 1.0 : -1.0;
        
        ADReal absX = x * ADReal(sign);  // Smooth way to get |x|
        
        ADReal t = ADReal(1.0) / (ADReal(1.0) + ADReal(p) * absX);
        ADReal t2 = t * t;
        ADReal t3 = t2 * t;
        ADReal t4 = t3 * t;
        ADReal t5 = t4 * t;
        
        ADReal poly = ADReal(a1) * t + ADReal(a2) * t2 + ADReal(a3) * t3 
                    + ADReal(a4) * t4 + ADReal(a5) * t5;
        
        const double invSqrt2Pi = 0.3989422804014327;  // 1/sqrt(2*pi)
        ADReal pdf = ADReal(invSqrt2Pi) * exp(ADReal(-0.5) * absX * absX);
        
        ADReal cdf = ADReal(1.0) - poly * pdf;
        
        // Apply symmetry: Phi(-x) = 1 - Phi(x)
        return (sign >= 0) ? cdf : (ADReal(1.0) - cdf);
    }
    
private:
    DiscountCurve<CurveReal> curve_;
    ATMVolSurface<double> volSurface_;
    std::vector<std::pair<double, double>> calibInstruments_;
    double notional_;
};

} // namespace hw1f
