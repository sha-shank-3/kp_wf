// =============================================================================
// Unit Tests: Dimension Sanity
//
// Validates that all matrices and vectors have correct dimensions:
// - J has shape (n_inst, n_params)
// - H has shape (n_params, n_params)
// - r has size n_inst
// - λ has size n_params
// - Vol Greeks have shape (n_expiries, n_tenors)
// - Curve Greeks have size n_curve_nodes
//
// =============================================================================

#include "utils/dimension_types.hpp"
#include "calibration/calibration_refactored.hpp"
#include "hw1f/hw1f_model.hpp"
#include "curve/discount_curve.hpp"
#include "utils/common.hpp"

#include <iostream>
#include <cassert>
#include <cmath>

using namespace hw1f;

// =============================================================================
// Test Helpers
// =============================================================================

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] " << msg << "\n"; \
            return false; \
        } \
    } while(0)

#define TEST_PASS(name) \
    std::cout << "[PASS] " << name << "\n"

// =============================================================================
// Test: ProblemDimensions construction and validation
// =============================================================================

bool test_problem_dimensions_construction() {
    // Valid construction
    ProblemDimensions dims(9, 10, 9, 9, 12);
    
    TEST_ASSERT(dims.n_params == 10, "n_params should be 1 + n_sigma = 10");
    TEST_ASSERT(dims.n_sigma == 9, "n_sigma should be 9");
    TEST_ASSERT(dims.n_inst == 10, "n_inst should be 10");
    TEST_ASSERT(dims.n_vol_nodes == 81, "n_vol_nodes should be 9*9=81");
    TEST_ASSERT(dims.n_curve_nodes == 12, "n_curve_nodes should be 12");
    
    // Exact-fit check
    TEST_ASSERT(dims.isExactFit(), "n_inst == n_params should be exact fit");
    TEST_ASSERT(!dims.isOverDetermined(), "Should not be over-determined");
    
    // Over-determined
    ProblemDimensions dims_lsq(9, 81, 9, 9, 12);
    TEST_ASSERT(!dims_lsq.isExactFit(), "81 > 10 should not be exact fit");
    TEST_ASSERT(dims_lsq.isOverDetermined(), "81 > 10 should be over-determined");
    
    TEST_PASS("test_problem_dimensions_construction");
    return true;
}

// =============================================================================
// Test: Jacobian shape validation
// =============================================================================

bool test_jacobian_shape_validation() {
    ProblemDimensions dims(9, 10, 9, 9, 12);
    
    // Correct shape: 10 x 10
    std::vector<std::vector<double>> J_correct(10, std::vector<double>(10, 0.0));
    try {
        validateJacobianShape(J_correct, dims);
    } catch (const std::exception& e) {
        TEST_ASSERT(false, "Valid Jacobian should not throw");
    }
    
    // Wrong number of rows: 8 x 10
    std::vector<std::vector<double>> J_wrong_rows(8, std::vector<double>(10, 0.0));
    bool threw = false;
    try {
        validateJacobianShape(J_wrong_rows, dims);
    } catch (const std::exception& e) {
        threw = true;
    }
    TEST_ASSERT(threw, "Jacobian with wrong rows should throw");
    
    // Wrong number of columns: 10 x 8
    std::vector<std::vector<double>> J_wrong_cols(10, std::vector<double>(8, 0.0));
    threw = false;
    try {
        validateJacobianShape(J_wrong_cols, dims);
    } catch (const std::exception& e) {
        threw = true;
    }
    TEST_ASSERT(threw, "Jacobian with wrong columns should throw");
    
    TEST_PASS("test_jacobian_shape_validation");
    return true;
}

// =============================================================================
// Test: Hessian shape validation
// =============================================================================

bool test_hessian_shape_validation() {
    ProblemDimensions dims(9, 10, 9, 9, 12);
    
    // Correct shape: 10 x 10
    std::vector<std::vector<double>> H_correct(10, std::vector<double>(10, 0.0));
    try {
        validateHessianShape(H_correct, dims);
    } catch (const std::exception& e) {
        TEST_ASSERT(false, "Valid Hessian should not throw");
    }
    
    // Wrong shape: 8 x 10
    std::vector<std::vector<double>> H_wrong(8, std::vector<double>(10, 0.0));
    bool threw = false;
    try {
        validateHessianShape(H_wrong, dims);
    } catch (const std::exception& e) {
        threw = true;
    }
    TEST_ASSERT(threw, "Hessian with wrong shape should throw");
    
    TEST_PASS("test_hessian_shape_validation");
    return true;
}

// =============================================================================
// Test: Residual size validation
// =============================================================================

bool test_residual_size_validation() {
    ProblemDimensions dims(9, 10, 9, 9, 12);
    
    // Correct size: 10
    std::vector<double> r_correct(10, 0.0);
    try {
        validateResidualSize(r_correct, dims);
    } catch (const std::exception& e) {
        TEST_ASSERT(false, "Valid residual should not throw");
    }
    
    // Wrong size: 8
    std::vector<double> r_wrong(8, 0.0);
    bool threw = false;
    try {
        validateResidualSize(r_wrong, dims);
    } catch (const std::exception& e) {
        threw = true;
    }
    TEST_ASSERT(threw, "Residual with wrong size should throw");
    
    TEST_PASS("test_residual_size_validation");
    return true;
}

// =============================================================================
// Test: Calibration result dimensions
// =============================================================================

bool test_calibration_result_dimensions() {
    // Create test market data
    std::vector<double> curveTimes = {0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0};
    std::vector<double> curveRates(12, 0.025);
    std::vector<double> curveDFs(12);
    for (size_t i = 0; i < 12; ++i) {
        curveDFs[i] = std::exp(-curveRates[i] * curveTimes[i]);
    }
    DiscountCurve<double> curve(curveTimes, curveDFs);
    
    std::vector<double> volExpiries = {0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0};
    std::vector<double> volTenors = {1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0};
    std::vector<std::vector<double>> vols(9, std::vector<double>(9, 0.20));
    ATMVolSurface<double> volSurface(volExpiries, volTenors, vols);
    
    // HW params: 1 a + 9 sigma = 10 params
    std::vector<double> sigmaTimes = {0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0};
    std::vector<double> sigmaValues(9, 0.01);
    HW1FParams initialParams(0.03, sigmaTimes, sigmaValues);
    
    // Calibrate with 10 instruments (exact fit)
    std::vector<std::pair<double, double>> calibInstruments = {
        {0.5, 9.5}, {1.0, 9.0}, {2.0, 8.0}, {3.0, 7.0}, {5.0, 5.0},
        {7.0, 3.0}, {10.0, 5.0}, {15.0, 5.0}, {20.0, 5.0}, {1.0, 4.0}
    };
    
    CalibrationEngine<double> calibEngine(curve, volSurface, 1e6);
    for (const auto& [e, t] : calibInstruments) {
        calibEngine.addInstrument(e, t);
    }
    
    auto result = calibEngine.calibrate(initialParams, 50, 1e-8, false);
    
    // Validate dimensions
    TEST_ASSERT(result.dims.n_params == 10, "n_params should be 10");
    TEST_ASSERT(result.dims.n_inst == 10, "n_inst should be 10");
    TEST_ASSERT(result.dims.n_sigma == 9, "n_sigma should be 9");
    TEST_ASSERT(result.dims.n_vol_nodes == 81, "n_vol_nodes should be 81");
    TEST_ASSERT(result.dims.n_curve_nodes == 12, "n_curve_nodes should be 12");
    
    // Validate Jacobian shape
    TEST_ASSERT(result.jacobian.size() == 10, "Jacobian should have 10 rows");
    TEST_ASSERT(result.jacobian[0].size() == 10, "Jacobian should have 10 columns");
    
    // Validate Hessian shape
    TEST_ASSERT(result.hessian.size() == 10, "Hessian should have 10 rows");
    TEST_ASSERT(result.hessian[0].size() == 10, "Hessian should have 10 columns");
    
    // Validate residuals size
    TEST_ASSERT(result.residuals.size() == 10, "Residuals should have 10 elements");
    
    // Validate weights size
    TEST_ASSERT(result.weights.size() == 10, "Weights should have 10 elements");
    
    // Validate IFT readiness
    TEST_ASSERT(result.isValidForIFT(), "Result should be valid for IFT");
    
    TEST_PASS("test_calibration_result_dimensions");
    return true;
}

// =============================================================================
// Test: LSQ calibration dimensions
// =============================================================================

bool test_lsq_calibration_dimensions() {
    // Create test market data
    std::vector<double> curveTimes = {0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0};
    std::vector<double> curveDFs(12);
    for (size_t i = 0; i < 12; ++i) {
        curveDFs[i] = std::exp(-0.025 * curveTimes[i]);
    }
    DiscountCurve<double> curve(curveTimes, curveDFs);
    
    std::vector<double> volExpiries = {0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0};
    std::vector<double> volTenors = {1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0};
    std::vector<std::vector<double>> vols(9, std::vector<double>(9, 0.20));
    ATMVolSurface<double> volSurface(volExpiries, volTenors, vols);
    
    std::vector<double> sigmaTimes = {0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0};
    std::vector<double> sigmaValues(9, 0.01);
    HW1FParams initialParams(0.03, sigmaTimes, sigmaValues);
    
    // Calibrate with all 81 instruments (LSQ)
    CalibrationEngine<double> calibEngine(curve, volSurface, 1e6);
    calibEngine.addAllSurfaceNodes();
    
    auto result = calibEngine.calibrate(initialParams, 50, 1e-8, false);
    
    // Validate dimensions for LSQ
    TEST_ASSERT(result.dims.n_params == 10, "n_params should be 10");
    TEST_ASSERT(result.dims.n_inst == 81, "n_inst should be 81 for full surface");
    TEST_ASSERT(result.dims.isOverDetermined(), "Should be over-determined");
    
    // Validate Jacobian shape: 81 x 10
    TEST_ASSERT(result.jacobian.size() == 81, "Jacobian should have 81 rows");
    TEST_ASSERT(result.jacobian[0].size() == 10, "Jacobian should have 10 columns");
    
    // Validate Hessian shape: 10 x 10 (NOT 81 x 81!)
    TEST_ASSERT(result.hessian.size() == 10, "Hessian H=JᵀWJ should have 10 rows");
    TEST_ASSERT(result.hessian[0].size() == 10, "Hessian H=JᵀWJ should have 10 columns");
    
    // Validate residuals size
    TEST_ASSERT(result.residuals.size() == 81, "Residuals should have 81 elements");
    
    TEST_PASS("test_lsq_calibration_dimensions");
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== Dimension Sanity Tests ===\n\n";
    
    int passed = 0, failed = 0;
    
    if (test_problem_dimensions_construction()) passed++; else failed++;
    if (test_jacobian_shape_validation()) passed++; else failed++;
    if (test_hessian_shape_validation()) passed++; else failed++;
    if (test_residual_size_validation()) passed++; else failed++;
    if (test_calibration_result_dimensions()) passed++; else failed++;
    if (test_lsq_calibration_dimensions()) passed++; else failed++;
    
    std::cout << "\n=== Summary ===\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n";
    
    return failed > 0 ? 1 : 0;
}
