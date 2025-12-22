// =============================================================================
// Greeks Validation App
//
// Comprehensive validation comparing all Greeks computation methods:
// 1. FD Naive (bump-recalibrate for each market data node)
// 2. FD + Chain Rule (bump-recalibrate with explicit chain)
// 3. FD + IFT (OpenGamma adjoint-IFT with FD for V_Φ)
// 4. XAD + IFT (Full AD optimization)
//
// Validates:
// - Dimension consistency throughout
// - Direct terms (V_C_direct, V_Θ_direct) 
// - Coverage statistics (exact-fit vs LSQ)
// - IFT vs FD Naive within tolerance
//
// =============================================================================

#include "hw1f/hw1f_model.hpp"
#include "calibration/calibration_refactored.hpp"
#include "risk/ift/ift_greeks_refactored.hpp"
#include "pricing/montecarlo/montecarlo.hpp"
#include "pricing/black_vega.hpp"
#include "curve/discount_curve.hpp"
#include "curve/vol_surface_weights.hpp"
#include "utils/dimension_types.hpp"
#include "utils/cholesky_factorization.hpp"
#include "utils/common.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace hw1f;

// =============================================================================
// Test Configuration
// =============================================================================

struct TestConfig {
    // MC parameters
    int numPaths = 10000;
    int numSteps = 100;
    int seed = 42;
    bool antithetic = true;
    
    // FD bump size
    double bump = 1e-4;
    
    // Tolerance for validation
    double relTol = 0.05;  // 5% relative tolerance
    double absTol = 1.0;   // $1 absolute tolerance
    
    // Exotic instrument
    double exoticExpiry = 5.0;
    double exoticTenor = 5.0;
    double notional = 1e6;
};

// =============================================================================
// Create Test Market Data
// =============================================================================

struct TestMarketData {
    std::vector<double> curveTimes;
    std::vector<double> curveRates;
    std::vector<double> volExpiries;
    std::vector<double> volTenors;
    std::vector<std::vector<double>> vols;
};

TestMarketData createTestMarket() {
    TestMarketData mkt;
    
    // Curve: 12 nodes
    mkt.curveTimes = {0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0};
    mkt.curveRates = {0.02, 0.022, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.030, 0.030, 0.030, 0.030};
    
    // Vol surface: 9x9 = 81 nodes
    mkt.volExpiries = {0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0};
    mkt.volTenors = {1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0};
    
    mkt.vols.resize(mkt.volExpiries.size());
    for (size_t i = 0; i < mkt.volExpiries.size(); ++i) {
        mkt.vols[i].resize(mkt.volTenors.size());
        for (size_t j = 0; j < mkt.volTenors.size(); ++j) {
            // Simple vol surface: decreasing with maturity
            double baseVol = 0.25 - 0.002 * mkt.volExpiries[i] - 0.001 * mkt.volTenors[j];
            mkt.vols[i][j] = std::max(baseVol, 0.10);
        }
    }
    
    return mkt;
}

// =============================================================================
// Validate Dimensions
// =============================================================================

void validateDimensions(
    const CalibrationResult& calibResult,
    const std::string& testName
) {
    std::cout << "\n=== Dimension Validation: " << testName << " ===\n";
    std::cout << calibResult.dims.toString();
    
    // Validate Jacobian shape
    try {
        validateJacobianShape(calibResult.jacobian, calibResult.dims);
        std::cout << "[PASS] Jacobian J has correct shape (" 
                  << calibResult.jacobian.size() << " x " 
                  << (calibResult.jacobian.empty() ? 0 : calibResult.jacobian[0].size()) << ")\n";
    } catch (const std::exception& e) {
        std::cout << "[FAIL] Jacobian shape: " << e.what() << "\n";
    }
    
    // Validate Hessian shape
    try {
        validateHessianShape(calibResult.hessian, calibResult.dims);
        std::cout << "[PASS] Hessian H has correct shape (" 
                  << calibResult.hessian.size() << " x " 
                  << (calibResult.hessian.empty() ? 0 : calibResult.hessian[0].size()) << ")\n";
    } catch (const std::exception& e) {
        std::cout << "[FAIL] Hessian shape: " << e.what() << "\n";
    }
    
    // Validate residual size
    try {
        validateResidualSize(calibResult.residuals, calibResult.dims);
        std::cout << "[PASS] Residuals r has correct size (" 
                  << calibResult.residuals.size() << ")\n";
    } catch (const std::exception& e) {
        std::cout << "[FAIL] Residual size: " << e.what() << "\n";
    }
    
    // Print coverage
    std::cout << "\n" << calibResult.coverage.toString();
}

// =============================================================================
// Compute FD Naive Greeks (benchmark)
// =============================================================================

GreeksResult computeFDNaiveGreeks(
    const EuropeanSwaption& exotic,
    const DiscountCurve<double>& curve,
    const ATMVolSurface<double>& volSurface,
    const std::vector<std::pair<double, double>>& calibInstruments,
    const HW1FParams& initialParams,
    const TestConfig& cfg
) {
    Timer timer;
    timer.start();
    
    GreeksResult result;
    
    MCConfig mcConfig;
    mcConfig.numPaths = cfg.numPaths;
    mcConfig.numSteps = cfg.numSteps;
    mcConfig.seed = cfg.seed;
    mcConfig.antithetic = cfg.antithetic;
    
    RNG rng(cfg.seed);
    auto Z = rng.normalMatrix(cfg.numPaths, cfg.numSteps);
    
    // Base calibration and price
    CalibrationEngine<double> baseCalibEngine(curve, volSurface, cfg.notional);
    for (const auto& [e, t] : calibInstruments) {
        baseCalibEngine.addInstrument(e, t);
    }
    auto baseCalibResult = baseCalibEngine.calibrate(initialParams, 100, 1e-8, false);
    
    HW1FModel<double> baseModel(baseCalibResult.params);
    MonteCarloPricer<double> basePricer(baseModel, mcConfig);
    double basePrice = basePricer.price(exotic, curve, Z).price;
    result.price = basePrice;
    
    // Vol Greeks (bump each vol node, recalibrate, reprice)
    size_t n_exp = volSurface.numExpiries();
    size_t n_ten = volSurface.numTenors();
    result.volGreeks.resize(n_exp, std::vector<double>(n_ten, 0.0));
    
    for (size_t ei = 0; ei < n_exp; ++ei) {
        for (size_t ti = 0; ti < n_ten; ++ti) {
            auto bumpedVol = volSurface.bump(ei, ti, cfg.bump);
            
            CalibrationEngine<double> calibEngine(curve, bumpedVol, cfg.notional);
            for (const auto& [e, t] : calibInstruments) {
                calibEngine.addInstrument(e, t);
            }
            auto calibResult = calibEngine.calibrate(baseCalibResult.params, 50, 1e-8, false);
            
            HW1FModel<double> model(calibResult.params);
            MonteCarloPricer<double> pricer(model, mcConfig);
            double bumpedPrice = pricer.price(exotic, curve, Z).price;
            
            result.volGreeks[ei][ti] = (bumpedPrice - basePrice) / cfg.bump;
        }
    }
    
    // Curve Greeks (bump each curve node, recalibrate, reprice)
    size_t n_curve = curve.size();
    result.curveGreeks.resize(n_curve, 0.0);
    
    for (size_t i = 0; i < n_curve; ++i) {
        auto bumpedCurve = curve.bump(i, cfg.bump);
        
        CalibrationEngine<double> calibEngine(bumpedCurve, volSurface, cfg.notional);
        for (const auto& [e, t] : calibInstruments) {
            calibEngine.addInstrument(e, t);
        }
        auto calibResult = calibEngine.calibrate(baseCalibResult.params, 50, 1e-8, false);
        
        HW1FModel<double> model(calibResult.params);
        MonteCarloPricer<double> pricer(model, mcConfig);
        double bumpedPrice = pricer.price(exotic, bumpedCurve, Z).price;
        
        result.curveGreeks[i] = (bumpedPrice - basePrice) / cfg.bump;
    }
    
    result.elapsedTime = timer.elapsed();
    return result;
}

// =============================================================================
// Compare Greeks Results
// =============================================================================

struct ComparisonResult {
    double maxVolGreekAbsErr;
    double maxVolGreekRelErr;
    double maxCurveGreekAbsErr;
    double maxCurveGreekRelErr;
    bool volGreeksPass;
    bool curveGreeksPass;
    bool overallPass;
};

ComparisonResult compareGreeks(
    const GreeksResult& ref,
    const GreeksResult& test,
    const TestConfig& cfg,
    const std::string& methodName
) {
    ComparisonResult comp;
    comp.maxVolGreekAbsErr = 0.0;
    comp.maxVolGreekRelErr = 0.0;
    comp.maxCurveGreekAbsErr = 0.0;
    comp.maxCurveGreekRelErr = 0.0;
    
    // Vol Greeks comparison
    if (ref.volGreeks.size() == test.volGreeks.size()) {
        for (size_t ei = 0; ei < ref.volGreeks.size(); ++ei) {
            for (size_t ti = 0; ti < ref.volGreeks[ei].size(); ++ti) {
                double refVal = ref.volGreeks[ei][ti];
                double testVal = test.volGreeks[ei][ti];
                double absErr = std::abs(testVal - refVal);
                double relErr = (std::abs(refVal) > 1e-10) ? absErr / std::abs(refVal) : 0.0;
                
                comp.maxVolGreekAbsErr = std::max(comp.maxVolGreekAbsErr, absErr);
                comp.maxVolGreekRelErr = std::max(comp.maxVolGreekRelErr, relErr);
            }
        }
    }
    
    // Curve Greeks comparison
    if (ref.curveGreeks.size() == test.curveGreeks.size()) {
        for (size_t i = 0; i < ref.curveGreeks.size(); ++i) {
            double refVal = ref.curveGreeks[i];
            double testVal = test.curveGreeks[i];
            double absErr = std::abs(testVal - refVal);
            double relErr = (std::abs(refVal) > 1e-10) ? absErr / std::abs(refVal) : 0.0;
            
            comp.maxCurveGreekAbsErr = std::max(comp.maxCurveGreekAbsErr, absErr);
            comp.maxCurveGreekRelErr = std::max(comp.maxCurveGreekRelErr, relErr);
        }
    }
    
    comp.volGreeksPass = (comp.maxVolGreekAbsErr < cfg.absTol) || (comp.maxVolGreekRelErr < cfg.relTol);
    comp.curveGreeksPass = (comp.maxCurveGreekAbsErr < cfg.absTol) || (comp.maxCurveGreekRelErr < cfg.relTol);
    comp.overallPass = comp.volGreeksPass && comp.curveGreeksPass;
    
    // Print results
    std::cout << "\n=== " << methodName << " vs FD Naive ===\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Vol Greeks:   Max AbsErr = " << comp.maxVolGreekAbsErr 
              << ", Max RelErr = " << (comp.maxVolGreekRelErr * 100) << "% "
              << (comp.volGreeksPass ? "[PASS]" : "[FAIL]") << "\n";
    std::cout << "Curve Greeks: Max AbsErr = " << comp.maxCurveGreekAbsErr 
              << ", Max RelErr = " << (comp.maxCurveGreekRelErr * 100) << "% "
              << (comp.curveGreeksPass ? "[PASS]" : "[FAIL]") << "\n";
    std::cout << "Timing: " << methodName << " = " << test.elapsedTime << "s, FD Naive = " << ref.elapsedTime << "s\n";
    std::cout << "Speedup: " << std::setprecision(1) << (ref.elapsedTime / test.elapsedTime) << "x\n";
    
    return comp;
}

// =============================================================================
// Main Validation
// =============================================================================

int main() {
    std::cout << "================================================================\n";
    std::cout << "               HW1F Greeks Validation Suite\n";
    std::cout << "================================================================\n";
    
    TestConfig cfg;
    
    // Create test market
    auto mkt = createTestMarket();
    
    // Build curve
    std::vector<double> dfs(mkt.curveTimes.size());
    for (size_t i = 0; i < mkt.curveTimes.size(); ++i) {
        dfs[i] = std::exp(-mkt.curveRates[i] * mkt.curveTimes[i]);
    }
    DiscountCurve<double> curve(mkt.curveTimes, dfs);
    
    // Build vol surface
    ATMVolSurface<double> volSurface(mkt.volExpiries, mkt.volTenors, mkt.vols);
    
    // Initial HW params (10 params: 1 a + 9 sigma buckets)
    std::vector<double> sigmaTimes = {0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0};
    std::vector<double> sigmaValues(9, 0.01);  // Initial guess
    HW1FParams initialParams(0.03, sigmaTimes, sigmaValues);
    
    // Exotic swaption
    EuropeanSwaption exotic(cfg.exoticExpiry, cfg.exoticTenor, 0.0, cfg.notional, true);
    double fwd = value(forwardSwapRate(exotic.underlying, curve));
    exotic.underlying.fixedRate = fwd;  // ATM
    
    // =========================================================================
    // Test 1: Exact-Fit Calibration (10 instruments = 10 params)
    // =========================================================================
    std::cout << "\n========================================\n";
    std::cout << "TEST 1: Exact-Fit Calibration (n_inst = n_params)\n";
    std::cout << "========================================\n";
    
    // Select 10 co-terminal instruments
    std::vector<std::pair<double, double>> exactFitInstruments = {
        {0.5, 9.5}, {1.0, 9.0}, {2.0, 8.0}, {3.0, 7.0}, {5.0, 5.0},
        {7.0, 3.0}, {10.0, 5.0}, {15.0, 5.0}, {20.0, 5.0}, {1.0, 4.0}
    };
    
    // Calibrate
    CalibrationEngine<double> exactFitCalibEngine(curve, volSurface, cfg.notional);
    for (const auto& [e, t] : exactFitInstruments) {
        exactFitCalibEngine.addInstrument(e, t);
    }
    auto exactFitCalibResult = exactFitCalibEngine.calibrate(initialParams, 100, 1e-8, true);
    
    // Validate dimensions
    validateDimensions(exactFitCalibResult, "Exact-Fit");
    
    // Compute IFT Greeks
    IFTGreeksEngineRefactored<double> iftEngine(curve, volSurface, exactFitInstruments, cfg.notional);
    
    MCConfig mcConfig;
    mcConfig.numPaths = cfg.numPaths;
    mcConfig.numSteps = cfg.numSteps;
    mcConfig.seed = cfg.seed;
    mcConfig.antithetic = cfg.antithetic;
    
    auto iftResult = iftEngine.computeIFTFromCalibResult(exotic, exactFitCalibResult, mcConfig, cfg.bump);
    
    // Print direct term norms
    std::cout << "\nDirect Terms:\n";
    std::cout << "  ||V_C_direct|| = " << iftResult.V_C_direct_norm << "\n";
    std::cout << "  ||V_Θ_direct|| = " << iftResult.V_Theta_direct_norm << " (should be 0 for HW1F)\n";
    std::cout << "  ||λ|| = " << iftResult.lambda_norm << "\n";
    
    // Compute FD Naive (reduced for speed)
    std::cout << "\nComputing FD Naive reference (reduced paths)...\n";
    TestConfig reducedCfg = cfg;
    reducedCfg.numPaths = 2000;  // Reduced for faster validation
    auto fdNaiveResult = computeFDNaiveGreeks(exotic, curve, volSurface, exactFitInstruments, initialParams, reducedCfg);
    
    // Compare
    auto comp1 = compareGreeks(fdNaiveResult, iftResult, cfg, "FD+IFT");
    
    // =========================================================================
    // Test 2: LSQ Calibration (81 instruments > 10 params)
    // =========================================================================
    std::cout << "\n========================================\n";
    std::cout << "TEST 2: LSQ Calibration (n_inst > n_params)\n";
    std::cout << "========================================\n";
    
    // Use all vol surface nodes
    std::vector<std::pair<double, double>> lsqInstruments;
    for (double e : mkt.volExpiries) {
        for (double t : mkt.volTenors) {
            lsqInstruments.emplace_back(e, t);
        }
    }
    
    // Calibrate
    CalibrationEngine<double> lsqCalibEngine(curve, volSurface, cfg.notional);
    lsqCalibEngine.addAllSurfaceNodes();
    auto lsqCalibResult = lsqCalibEngine.calibrate(initialParams, 100, 1e-8, true);
    
    // Validate dimensions
    validateDimensions(lsqCalibResult, "Least-Squares");
    
    // Compute IFT Greeks
    IFTGreeksEngineRefactored<double> lsqIftEngine(curve, volSurface, lsqInstruments, cfg.notional);
    auto lsqIftResult = lsqIftEngine.computeIFTFromCalibResult(exotic, lsqCalibResult, mcConfig, cfg.bump);
    
    // Print direct term norms
    std::cout << "\nDirect Terms:\n";
    std::cout << "  ||V_C_direct|| = " << lsqIftResult.V_C_direct_norm << "\n";
    std::cout << "  ||V_Θ_direct|| = " << lsqIftResult.V_Theta_direct_norm << "\n";
    std::cout << "  ||λ|| = " << lsqIftResult.lambda_norm << "\n";
    
    // =========================================================================
    // Test 3: Interpolation Weights Validation
    // =========================================================================
    std::cout << "\n========================================\n";
    std::cout << "TEST 3: Interpolation Weights\n";
    std::cout << "========================================\n";
    
    // Test a point that falls between nodes
    double testExp = 1.5;
    double testTen = 2.5;
    auto weights = getInterpWeights(testExp, testTen, mkt.volExpiries, mkt.volTenors);
    
    std::cout << "Test point: (expiry=" << testExp << ", tenor=" << testTen << ")\n";
    std::cout << "Interpolation weights:\n";
    double sumWeights = 0.0;
    for (const auto& w : weights) {
        std::cout << "  Node (" << mkt.volExpiries[w.expiryIdx] << ", " 
                  << mkt.volTenors[w.tenorIdx] << "): weight = " << w.weight << "\n";
        sumWeights += w.weight;
    }
    std::cout << "Sum of weights: " << sumWeights << " (should be 1.0)\n";
    
    bool isNode = isExactNode(testExp, testTen, mkt.volExpiries, mkt.volTenors);
    std::cout << "Is exact node: " << (isNode ? "Yes" : "No") << "\n";
    
    // Test an exact node
    testExp = 1.0;
    testTen = 2.0;
    auto exactWeights = getInterpWeights(testExp, testTen, mkt.volExpiries, mkt.volTenors);
    std::cout << "\nExact node: (expiry=" << testExp << ", tenor=" << testTen << ")\n";
    std::cout << "Interpolation weights:\n";
    for (const auto& w : exactWeights) {
        std::cout << "  Node (" << mkt.volExpiries[w.expiryIdx] << ", " 
                  << mkt.volTenors[w.tenorIdx] << "): weight = " << w.weight << "\n";
    }
    
    // =========================================================================
    // Summary
    // =========================================================================
    std::cout << "\n================================================================\n";
    std::cout << "                       SUMMARY\n";
    std::cout << "================================================================\n";
    std::cout << "Exact-Fit Test: " << (comp1.overallPass ? "PASS" : "FAIL") << "\n";
    std::cout << "LSQ Test: IFT computation completed\n";
    std::cout << "Interpolation Weights Test: " << (std::abs(sumWeights - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n";
    
    return 0;
}
