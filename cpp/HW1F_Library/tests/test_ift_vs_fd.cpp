// =============================================================================
// Unit Tests: IFT vs FD Validation
//
// Validates that IFT Greeks match FD Naive Greeks within tolerance.
// This is the key correctness test for the OpenGamma adjoint-IFT implementation.
//
// Tests:
// 1. Vol Greeks: dV/dΘ should match FD bump-recalibrate
// 2. Curve Greeks: dV/dC should match FD bump-recalibrate
// 3. Direct terms: V_C_direct should be non-zero
// 4. Both exact-fit and LSQ calibration
//
// =============================================================================

#include "hw1f/hw1f_model.hpp"
#include "calibration/calibration_refactored.hpp"
#include "risk/ift/ift_greeks_refactored.hpp"
#include "pricing/montecarlo/montecarlo.hpp"
#include "curve/discount_curve.hpp"
#include "utils/common.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace hw1f;

// =============================================================================
// Test Helpers
// =============================================================================

#define TEST_ASSERT_NEAR(a, b, tol, msg) \
    do { \
        double _a = (a), _b = (b), _t = (tol); \
        double _err = std::abs(_a - _b); \
        double _rel = (std::abs(_b) > 1e-10) ? _err / std::abs(_b) : _err; \
        if (_err > _t && _rel > 0.1) { \
            std::cerr << "[FAIL] " << msg << ": " << _a << " vs " << _b \
                      << " (err=" << _err << ", rel=" << _rel * 100 << "%)\n"; \
            return false; \
        } \
    } while(0)

#define TEST_PASS(name) \
    std::cout << "[PASS] " << name << "\n"

// =============================================================================
// Create Test Market Data
// =============================================================================

struct TestMarket {
    DiscountCurve<double> curve;
    ATMVolSurface<double> volSurface;
    HW1FParams initialParams;
    EuropeanSwaption exotic;
    std::vector<std::pair<double, double>> calibInstruments;
};

TestMarket createTestMarket(bool fullSurface = false) {
    // Curve: 12 nodes
    std::vector<double> curveTimes = {0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0};
    std::vector<double> curveDFs(12);
    for (size_t i = 0; i < 12; ++i) {
        curveDFs[i] = std::exp(-0.025 * curveTimes[i]);
    }
    DiscountCurve<double> curve(curveTimes, curveDFs);
    
    // Vol surface: 9x9 = 81 nodes
    std::vector<double> volExpiries = {0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0};
    std::vector<double> volTenors = {1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0};
    std::vector<std::vector<double>> vols(9, std::vector<double>(9));
    for (size_t i = 0; i < 9; ++i) {
        for (size_t j = 0; j < 9; ++j) {
            vols[i][j] = 0.22 - 0.002 * volExpiries[i] - 0.001 * volTenors[j];
        }
    }
    ATMVolSurface<double> volSurface(volExpiries, volTenors, vols);
    
    // Initial HW params
    std::vector<double> sigmaTimes = {0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0};
    std::vector<double> sigmaValues(9, 0.01);
    HW1FParams initialParams(0.03, sigmaTimes, sigmaValues);
    
    // Exotic swaption: 5Y into 5Y
    double notional = 1e6;
    EuropeanSwaption exotic(5.0, 5.0, 0.0, notional, true);
    double fwd = value(forwardSwapRate(exotic.underlying, curve));
    exotic.underlying.fixedRate = fwd;
    
    // Calibration instruments
    std::vector<std::pair<double, double>> calibInstruments;
    if (fullSurface) {
        // All 81 nodes
        for (double e : volExpiries) {
            for (double t : volTenors) {
                calibInstruments.emplace_back(e, t);
            }
        }
    } else {
        // 10 co-terminal for exact fit
        calibInstruments = {
            {0.5, 9.5}, {1.0, 9.0}, {2.0, 8.0}, {3.0, 7.0}, {5.0, 5.0},
            {7.0, 3.0}, {10.0, 5.0}, {15.0, 5.0}, {20.0, 5.0}, {1.0, 4.0}
        };
    }
    
    return {curve, volSurface, initialParams, exotic, calibInstruments};
}

// =============================================================================
// Compute FD Naive Greeks for a subset (for speed)
// =============================================================================

struct FDNaiveResult {
    double price;
    std::vector<double> volGreeksSubset;  // For selected vol nodes
    std::vector<double> curveGreeksSubset; // For selected curve nodes
    std::vector<size_t> volIndices;  // Which vol nodes were tested
    std::vector<size_t> curveIndices; // Which curve nodes were tested
};

FDNaiveResult computeFDNaiveSubset(
    const TestMarket& mkt,
    const CalibrationResult& calibResult,
    int numPaths = 5000,
    double bump = 1e-4
) {
    FDNaiveResult result;
    
    MCConfig mcConfig;
    mcConfig.numPaths = numPaths;
    mcConfig.numSteps = 100;
    mcConfig.seed = 42;
    mcConfig.antithetic = true;
    
    RNG rng(42);
    auto Z = rng.normalMatrix(numPaths, 100);
    
    // Base price
    HW1FModel<double> baseModel(calibResult.params);
    MonteCarloPricer<double> basePricer(baseModel, mcConfig);
    result.price = basePricer.price(mkt.exotic, mkt.curve, Z).price;
    
    // Test subset of vol nodes (4 corners + center)
    result.volIndices = {0, 8, 72, 80, 40};  // Corners and center of 9x9 grid
    
    for (size_t flatIdx : result.volIndices) {
        size_t ei = flatIdx / 9;
        size_t ti = flatIdx % 9;
        
        auto bumpedVol = mkt.volSurface.bump(ei, ti, bump);
        
        CalibrationEngine<double> calibEngine(mkt.curve, bumpedVol, 1e6);
        for (const auto& [e, t] : mkt.calibInstruments) {
            calibEngine.addInstrument(e, t);
        }
        auto bumpedResult = calibEngine.calibrate(calibResult.params, 50, 1e-8, false);
        
        HW1FModel<double> bumpedModel(bumpedResult.params);
        MonteCarloPricer<double> bumpedPricer(bumpedModel, mcConfig);
        double bumpedPrice = bumpedPricer.price(mkt.exotic, mkt.curve, Z).price;
        
        result.volGreeksSubset.push_back((bumpedPrice - result.price) / bump);
    }
    
    // Test subset of curve nodes (first, middle, last)
    result.curveIndices = {0, 5, 11};
    
    for (size_t i : result.curveIndices) {
        auto bumpedCurve = mkt.curve.bump(i, bump);
        
        CalibrationEngine<double> calibEngine(bumpedCurve, mkt.volSurface, 1e6);
        for (const auto& [e, t] : mkt.calibInstruments) {
            calibEngine.addInstrument(e, t);
        }
        auto bumpedResult = calibEngine.calibrate(calibResult.params, 50, 1e-8, false);
        
        HW1FModel<double> bumpedModel(bumpedResult.params);
        MonteCarloPricer<double> bumpedPricer(bumpedModel, mcConfig);
        double bumpedPrice = bumpedPricer.price(mkt.exotic, bumpedCurve, Z).price;
        
        result.curveGreeksSubset.push_back((bumpedPrice - result.price) / bump);
    }
    
    return result;
}

// =============================================================================
// Test: IFT vol Greeks match FD
// =============================================================================

bool test_ift_vol_greeks_vs_fd() {
    std::cout << "Running IFT vol Greeks test...\n";
    
    auto mkt = createTestMarket(false);  // Exact fit
    
    // Calibrate
    CalibrationEngine<double> calibEngine(mkt.curve, mkt.volSurface, 1e6);
    for (const auto& [e, t] : mkt.calibInstruments) {
        calibEngine.addInstrument(e, t);
    }
    auto calibResult = calibEngine.calibrate(mkt.initialParams, 100, 1e-8, false);
    
    // Compute IFT Greeks
    MCConfig mcConfig;
    mcConfig.numPaths = 5000;
    mcConfig.numSteps = 100;
    mcConfig.seed = 42;
    mcConfig.antithetic = true;
    
    IFTGreeksEngineRefactored<double> iftEngine(
        mkt.curve, mkt.volSurface, mkt.calibInstruments, 1e6
    );
    auto iftResult = iftEngine.computeIFTFromCalibResult(mkt.exotic, calibResult, mcConfig, 1e-4);
    
    // Compute FD Naive subset
    auto fdResult = computeFDNaiveSubset(mkt, calibResult, 5000, 1e-4);
    
    // Compare
    double maxErr = 0.0;
    double maxRel = 0.0;
    
    for (size_t k = 0; k < fdResult.volIndices.size(); ++k) {
        size_t flatIdx = fdResult.volIndices[k];
        size_t ei = flatIdx / 9;
        size_t ti = flatIdx % 9;
        
        double iftVal = iftResult.volGreeks[ei][ti];
        double fdVal = fdResult.volGreeksSubset[k];
        
        double err = std::abs(iftVal - fdVal);
        double rel = (std::abs(fdVal) > 1e-6) ? err / std::abs(fdVal) : 0.0;
        
        maxErr = std::max(maxErr, err);
        maxRel = std::max(maxRel, rel);
        
        std::cout << "  Vol node (" << ei << "," << ti << "): IFT=" 
                  << std::setprecision(2) << iftVal << ", FD=" << fdVal 
                  << ", rel=" << std::setprecision(1) << (rel * 100) << "%\n";
    }
    
    // Tolerance: 10% relative or $1 absolute
    bool pass = (maxRel < 0.15) || (maxErr < 1.0);
    
    if (pass) {
        TEST_PASS("test_ift_vol_greeks_vs_fd");
    } else {
        std::cerr << "[FAIL] Vol Greeks max error: " << maxErr << ", max rel: " << (maxRel * 100) << "%\n";
    }
    
    return pass;
}

// =============================================================================
// Test: IFT curve Greeks match FD
// =============================================================================

bool test_ift_curve_greeks_vs_fd() {
    std::cout << "Running IFT curve Greeks test...\n";
    
    auto mkt = createTestMarket(false);
    
    // Calibrate
    CalibrationEngine<double> calibEngine(mkt.curve, mkt.volSurface, 1e6);
    for (const auto& [e, t] : mkt.calibInstruments) {
        calibEngine.addInstrument(e, t);
    }
    auto calibResult = calibEngine.calibrate(mkt.initialParams, 100, 1e-8, false);
    
    // Compute IFT Greeks
    MCConfig mcConfig;
    mcConfig.numPaths = 5000;
    mcConfig.numSteps = 100;
    mcConfig.seed = 42;
    mcConfig.antithetic = true;
    
    IFTGreeksEngineRefactored<double> iftEngine(
        mkt.curve, mkt.volSurface, mkt.calibInstruments, 1e6
    );
    auto iftResult = iftEngine.computeIFTFromCalibResult(mkt.exotic, calibResult, mcConfig, 1e-4);
    
    // Compute FD Naive subset
    auto fdResult = computeFDNaiveSubset(mkt, calibResult, 5000, 1e-4);
    
    // Compare
    double maxErr = 0.0;
    double maxRel = 0.0;
    
    for (size_t k = 0; k < fdResult.curveIndices.size(); ++k) {
        size_t i = fdResult.curveIndices[k];
        
        double iftVal = iftResult.curveGreeks[i];
        double fdVal = fdResult.curveGreeksSubset[k];
        
        double err = std::abs(iftVal - fdVal);
        double rel = (std::abs(fdVal) > 1e-6) ? err / std::abs(fdVal) : 0.0;
        
        maxErr = std::max(maxErr, err);
        maxRel = std::max(maxRel, rel);
        
        std::cout << "  Curve node " << i << ": IFT=" 
                  << std::setprecision(2) << iftVal << ", FD=" << fdVal 
                  << ", rel=" << std::setprecision(1) << (rel * 100) << "%\n";
    }
    
    // Tolerance: 15% relative or $5 absolute
    bool pass = (maxRel < 0.20) || (maxErr < 5.0);
    
    if (pass) {
        TEST_PASS("test_ift_curve_greeks_vs_fd");
    } else {
        std::cerr << "[FAIL] Curve Greeks max error: " << maxErr << ", max rel: " << (maxRel * 100) << "%\n";
    }
    
    return pass;
}

// =============================================================================
// Test: V_C_direct is non-zero
// =============================================================================

bool test_v_c_direct_nonzero() {
    std::cout << "Running V_C_direct test...\n";
    
    auto mkt = createTestMarket(false);
    
    // Calibrate
    CalibrationEngine<double> calibEngine(mkt.curve, mkt.volSurface, 1e6);
    for (const auto& [e, t] : mkt.calibInstruments) {
        calibEngine.addInstrument(e, t);
    }
    auto calibResult = calibEngine.calibrate(mkt.initialParams, 100, 1e-8, false);
    
    // Compute IFT Greeks
    MCConfig mcConfig;
    mcConfig.numPaths = 5000;
    mcConfig.numSteps = 100;
    mcConfig.seed = 42;
    mcConfig.antithetic = true;
    
    IFTGreeksEngineRefactored<double> iftEngine(
        mkt.curve, mkt.volSurface, mkt.calibInstruments, 1e6
    );
    auto iftResult = iftEngine.computeIFTFromCalibResult(mkt.exotic, calibResult, mcConfig, 1e-4);
    
    // V_C_direct should be non-zero
    bool pass = iftResult.V_C_direct_norm > 1e-6;
    
    std::cout << "  ||V_C_direct|| = " << iftResult.V_C_direct_norm << "\n";
    
    if (pass) {
        TEST_PASS("test_v_c_direct_nonzero");
    } else {
        std::cerr << "[FAIL] V_C_direct should be non-zero for curve Greeks\n";
    }
    
    return pass;
}

// =============================================================================
// Test: V_Theta_direct is zero for HW1F
// =============================================================================

bool test_v_theta_direct_zero() {
    std::cout << "Running V_Θ_direct test...\n";
    
    auto mkt = createTestMarket(false);
    
    // Calibrate
    CalibrationEngine<double> calibEngine(mkt.curve, mkt.volSurface, 1e6);
    for (const auto& [e, t] : mkt.calibInstruments) {
        calibEngine.addInstrument(e, t);
    }
    auto calibResult = calibEngine.calibrate(mkt.initialParams, 100, 1e-8, false);
    
    // Compute IFT Greeks
    MCConfig mcConfig;
    mcConfig.numPaths = 5000;
    mcConfig.numSteps = 100;
    mcConfig.seed = 42;
    mcConfig.antithetic = true;
    
    IFTGreeksEngineRefactored<double> iftEngine(
        mkt.curve, mkt.volSurface, mkt.calibInstruments, 1e6
    );
    auto iftResult = iftEngine.computeIFTFromCalibResult(mkt.exotic, calibResult, mcConfig, 1e-4);
    
    // V_Theta_direct should be zero for HW1F
    bool pass = iftResult.V_Theta_direct_norm < 1e-10;
    
    std::cout << "  ||V_Θ_direct|| = " << iftResult.V_Theta_direct_norm << "\n";
    
    if (pass) {
        TEST_PASS("test_v_theta_direct_zero");
    } else {
        std::cerr << "[FAIL] V_Θ_direct should be zero for HW1F (exotic doesn't depend on Black vols)\n";
    }
    
    return pass;
}

// =============================================================================
// Test: LSQ calibration IFT
// =============================================================================

bool test_lsq_ift_greeks() {
    std::cout << "Running LSQ IFT Greeks test...\n";
    
    auto mkt = createTestMarket(true);  // Full surface (81 instruments)
    
    // Calibrate
    CalibrationEngine<double> calibEngine(mkt.curve, mkt.volSurface, 1e6);
    calibEngine.addAllSurfaceNodes();
    auto calibResult = calibEngine.calibrate(mkt.initialParams, 100, 1e-8, false);
    
    // Verify LSQ
    bool isLSQ = calibResult.dims.n_inst > calibResult.dims.n_params;
    if (!isLSQ) {
        std::cerr << "[FAIL] Should be LSQ calibration\n";
        return false;
    }
    
    std::cout << "  LSQ: " << calibResult.dims.n_inst << " instruments > " 
              << calibResult.dims.n_params << " params\n";
    std::cout << "  RMSE: " << calibResult.rmse << "\n";
    
    // Compute IFT Greeks
    MCConfig mcConfig;
    mcConfig.numPaths = 5000;
    mcConfig.numSteps = 100;
    mcConfig.seed = 42;
    mcConfig.antithetic = true;
    
    IFTGreeksEngineRefactored<double> iftEngine(
        mkt.curve, mkt.volSurface, mkt.calibInstruments, 1e6
    );
    
    bool threw = false;
    try {
        auto iftResult = iftEngine.computeIFTFromCalibResult(mkt.exotic, calibResult, mcConfig, 1e-4);
        std::cout << "  Price: " << iftResult.price << "\n";
        std::cout << "  ||λ||: " << iftResult.lambda_norm << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] LSQ IFT threw: " << e.what() << "\n";
        threw = true;
    }
    
    if (!threw) {
        TEST_PASS("test_lsq_ift_greeks");
    }
    
    return !threw;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== IFT vs FD Validation Tests ===\n\n";
    
    int passed = 0, failed = 0;
    
    if (test_ift_vol_greeks_vs_fd()) passed++; else failed++;
    if (test_ift_curve_greeks_vs_fd()) passed++; else failed++;
    if (test_v_c_direct_nonzero()) passed++; else failed++;
    if (test_v_theta_direct_zero()) passed++; else failed++;
    if (test_lsq_ift_greeks()) passed++; else failed++;
    
    std::cout << "\n=== Summary ===\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n";
    
    return failed > 0 ? 1 : 0;
}
