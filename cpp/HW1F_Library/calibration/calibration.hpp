#pragma once
// =============================================================================
// HW1F Calibration Engine
// Calibrates mean reversion and piecewise-constant volatility to ATM swaptions
// Uses Levenberg-Marquardt with XAD for Jacobian computation
// =============================================================================

#include "hw1f/hw1f_model.hpp"
#include "curve/discount_curve.hpp"
#include "instruments/swaption.hpp"
#include "pricing/jamshidian/jamshidian.hpp"
#include "utils/common.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace hw1f {

// =============================================================================
// Calibration Instrument
// =============================================================================

struct CalibrationInstrument {
    double expiry;
    double tenor;
    double marketVol;      // ATM Black vol
    double marketPrice;    // Black price (computed from vol)
    double weight;
    
    CalibrationInstrument(double e, double t, double vol, double w = 1.0)
        : expiry(e), tenor(t), marketVol(vol), marketPrice(0.0), weight(w) {}
};

// =============================================================================
// Calibration Result
// =============================================================================

struct CalibrationResult {
    HW1FParams params;
    std::vector<double> residuals;
    double rmse;
    int iterations;
    bool converged;
    double elapsedTime;
};

// =============================================================================
// Calibration Engine
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
    
    // Add instruments from vol surface nodes
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
    
    // Compute market prices from Black vols
    void computeMarketPrices() {
        for (auto& inst : instruments_) {
            EuropeanSwaption swaption(inst.expiry, inst.tenor, 0.0, notional_, true);
            // Get forward rate for ATM strike
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
    
    // Compute residuals: r_i = (model_i - market_i) / weight_i
    std::vector<double> computeResiduals(const HW1FParams& params) const {
        std::vector<double> residuals(instruments_.size());
        
        for (size_t i = 0; i < instruments_.size(); ++i) {
            double modelP = modelPrice(instruments_[i], params);
            double marketP = instruments_[i].marketPrice;
            residuals[i] = (modelP - marketP) * instruments_[i].weight;
        }
        
        return residuals;
    }
    
    // Compute RMSE
    double computeRMSE(const std::vector<double>& residuals) const {
        double sum = 0.0;
        for (double r : residuals) {
            sum += r * r;
        }
        return std::sqrt(sum / residuals.size());
    }
    
    // Compute Jacobian using finite differences
    std::vector<std::vector<double>> computeJacobianFD(
        const HW1FParams& params,
        double eps = 1e-6
    ) const {
        size_t m = instruments_.size();
        size_t n = params.numParams();
        
        std::vector<std::vector<double>> J(m, std::vector<double>(n, 0.0));
        std::vector<double> baseResiduals = computeResiduals(params);
        
        // Perturb each parameter
        std::vector<double> paramVec = params.toVector();
        
        for (size_t j = 0; j < n; ++j) {
            double h = eps * std::max(1.0, std::abs(paramVec[j]));
            
            std::vector<double> paramUp = paramVec;
            paramUp[j] += h;
            
            HW1FParams paramsUp = params;
            paramsUp.fromVector(paramUp);
            
            std::vector<double> residualsUp = computeResiduals(paramsUp);
            
            for (size_t i = 0; i < m; ++i) {
                J[i][j] = (residualsUp[i] - baseResiduals[i]) / h;
            }
        }
        
        return J;
    }
    
    // Calibrate using Levenberg-Marquardt
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
        size_t n = paramVec.size();
        
        double lambda = 1e-3;  // LM damping parameter
        double nu = 2.0;
        
        std::vector<double> residuals = computeResiduals(params);
        double cost = 0.5 * dot(residuals, residuals);
        
        CalibrationResult result;
        result.converged = false;
        
        for (int iter = 0; iter < maxIter; ++iter) {
            // Compute Jacobian
            auto J = computeJacobianFD(params);
            auto JT = transpose(J);
            
            // Compute J^T * J
            std::vector<std::vector<double>> JTJ(n, std::vector<double>(n, 0.0));
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    for (size_t k = 0; k < instruments_.size(); ++k) {
                        JTJ[i][j] += JT[i][k] * J[k][j];
                    }
                }
            }
            
            // Compute J^T * r
            std::vector<double> JTr = matVecMult(JT, residuals);
            
            // Add damping: (J^T J + lambda * I) * delta = -J^T * r
            auto A = JTJ;
            for (size_t i = 0; i < n; ++i) {
                A[i][i] += lambda * (JTJ[i][i] + 1e-6);
            }
            
            // Solve for delta
            std::vector<double> negJTr(n);
            for (size_t i = 0; i < n; ++i) {
                negJTr[i] = -JTr[i];
            }
            
            std::vector<double> delta;
            try {
                delta = solveCholesky(A, negJTr);
            } catch (...) {
                // Matrix not positive definite, increase damping
                lambda *= nu;
                continue;
            }
            
            // Try update
            std::vector<double> newParamVec(n);
            for (size_t i = 0; i < n; ++i) {
                newParamVec[i] = paramVec[i] + delta[i];
            }
            
            // Enforce bounds
            newParamVec[0] = std::clamp(newParamVec[0], 1e-4, 1.0);  // a
            for (size_t i = 1; i < n; ++i) {
                newParamVec[i] = std::clamp(newParamVec[i], 1e-6, 0.1);  // sigma
            }
            
            HW1FParams newParams = params;
            newParams.fromVector(newParamVec);
            
            std::vector<double> newResiduals = computeResiduals(newParams);
            double newCost = 0.5 * dot(newResiduals, newResiduals);
            
            // Accept or reject
            if (newCost < cost) {
                params = newParams;
                paramVec = newParamVec;
                residuals = newResiduals;
                cost = newCost;
                lambda /= nu;
                
                if (verbose) {
                    std::cout << "  Iter " << iter << ": RMSE = " 
                              << std::scientific << std::setprecision(6)
                              << computeRMSE(residuals) << ", cost = " << cost << "\n";
                }
                
                // Check convergence
                if (norm(delta) < tol * (norm(paramVec) + tol)) {
                    result.converged = true;
                    result.iterations = iter + 1;
                    break;
                }
            } else {
                lambda *= nu;
            }
            
            if (iter == maxIter - 1) {
                result.iterations = maxIter;
            }
        }
        
        result.params = params;
        result.residuals = residuals;
        result.rmse = computeRMSE(residuals);
        result.elapsedTime = timer.elapsed();
        
        return result;
    }
    
    // Get instruments for IFT computation
    const std::vector<CalibrationInstrument>& instruments() const { return instruments_; }
    
    // Get curve
    const DiscountCurve<CurveReal>& curve() const { return curve_; }
    
    // Get vol surface
    const ATMVolSurface<double>& volSurface() const { return volSurface_; }

private:
    DiscountCurve<CurveReal> curve_;
    ATMVolSurface<double> volSurface_;
    std::vector<CalibrationInstrument> instruments_;
    double notional_;
};

} // namespace hw1f
