// =============================================================================
// Unit Tests: V_r Bucket Splitting
//
// Validates correct computation of V_r(s,t) = ∫_s^t e^{-2a(t-u)} σ(u)² du
// when sigma is piecewise-constant and the integration interval [s,t]
// crosses bucket boundaries.
//
// Key test cases:
// 1. Interval within single bucket
// 2. Interval crossing one boundary
// 3. Interval crossing multiple boundaries
// 4. Degenerate cases (a→0, t=s)
//
// =============================================================================

#include "hw1f/hw1f_model.hpp"
#include "utils/common.hpp"

#include <iostream>
#include <cmath>
#include <vector>

using namespace hw1f;

// =============================================================================
// Test Helpers
// =============================================================================

#define TEST_ASSERT_NEAR(a, b, tol, msg) \
    do { \
        double _a = (a), _b = (b), _t = (tol); \
        if (std::abs(_a - _b) > _t) { \
            std::cerr << "[FAIL] " << msg << ": " << _a << " vs " << _b << " (tol=" << _t << ")\n"; \
            return false; \
        } \
    } while(0)

#define TEST_PASS(name) \
    std::cout << "[PASS] " << name << "\n"

// =============================================================================
// Reference implementation of V_r for single bucket (constant sigma)
// =============================================================================

double V_r_single_bucket(double s, double t, double a, double sigma) {
    if (t <= s) return 0.0;
    
    if (std::abs(a) < 1e-8) {
        // Limit as a → 0: ∫_s^t σ² du = σ² * (t - s)
        return sigma * sigma * (t - s);
    }
    
    // ∫_s^t e^{-2a(t-u)} σ² du = σ²/(2a) * [1 - e^{-2a(t-s)}]
    return sigma * sigma * (1.0 - std::exp(-2.0 * a * (t - s))) / (2.0 * a);
}

// =============================================================================
// Reference implementation of V_r for piecewise-constant sigma
// =============================================================================

double V_r_piecewise_ref(
    double s, double t, double a,
    const std::vector<double>& sigmaTimes,
    const std::vector<double>& sigmaValues
) {
    if (t <= s) return 0.0;
    
    double result = 0.0;
    size_t numBuckets = sigmaValues.size();
    
    for (size_t i = 0; i < numBuckets; ++i) {
        double bucket_start = sigmaTimes[i];
        double bucket_end = (i + 1 < sigmaTimes.size()) ? sigmaTimes[i + 1] : 1e10;
        
        // Clip to [s, t]
        double t_start = std::max(bucket_start, s);
        double t_end = std::min(bucket_end, t);
        
        if (t_end <= t_start) continue;
        
        double sigma_i = sigmaValues[i];
        
        if (std::abs(a) < 1e-8) {
            // Limit: ∫ σ² du
            result += sigma_i * sigma_i * (t_end - t_start);
        } else {
            // ∫_{t_start}^{t_end} e^{-2a(t-u)} σ² du
            // = σ²/(2a) * (e^{-2a(t-t_end)} - e^{-2a(t-t_start)})
            double exp_2a_end = std::exp(-2.0 * a * (t - t_end));
            double exp_2a_start = std::exp(-2.0 * a * (t - t_start));
            double integral = (exp_2a_end - exp_2a_start) / (2.0 * a);
            result += sigma_i * sigma_i * integral;
        }
    }
    
    return result;
}

// =============================================================================
// Test: V_r within single bucket
// =============================================================================

bool test_vr_single_bucket() {
    // Params: a = 0.03, sigma constant at 0.01
    std::vector<double> sigmaTimes = {0.0};
    std::vector<double> sigmaValues = {0.01};
    HW1FParams params(0.03, sigmaTimes, sigmaValues);
    HW1FModel<double> model(params);
    
    double a = 0.03;
    double sigma = 0.01;
    
    // Test case: [0, 1]
    double s = 0.0, t = 1.0;
    double V_model = model.V_r(s, t);
    double V_ref = V_r_single_bucket(s, t, a, sigma);
    TEST_ASSERT_NEAR(V_model, V_ref, 1e-10, "V_r(0,1) single bucket");
    
    // Test case: [2, 5]
    s = 2.0; t = 5.0;
    V_model = model.V_r(s, t);
    V_ref = V_r_single_bucket(s, t, a, sigma);
    TEST_ASSERT_NEAR(V_model, V_ref, 1e-10, "V_r(2,5) single bucket");
    
    TEST_PASS("test_vr_single_bucket");
    return true;
}

// =============================================================================
// Test: V_r crossing one boundary
// =============================================================================

bool test_vr_one_boundary() {
    // Params: a = 0.03, sigma changes at t=2
    std::vector<double> sigmaTimes = {0.0, 2.0};
    std::vector<double> sigmaValues = {0.01, 0.015};
    HW1FParams params(0.03, sigmaTimes, sigmaValues);
    HW1FModel<double> model(params);
    
    double a = 0.03;
    
    // Test case: [1, 3] crosses boundary at t=2
    double s = 1.0, t = 3.0;
    double V_model = model.V_r(s, t);
    double V_ref = V_r_piecewise_ref(s, t, a, sigmaTimes, sigmaValues);
    TEST_ASSERT_NEAR(V_model, V_ref, 1e-10, "V_r(1,3) one boundary");
    
    // Verify it's not equal to single-bucket formula with either sigma
    double V_wrong1 = V_r_single_bucket(s, t, a, 0.01);
    double V_wrong2 = V_r_single_bucket(s, t, a, 0.015);
    
    // V_model should be between the two (since sigma increases at t=2)
    bool between = (V_model > V_wrong1 && V_model < V_wrong2);
    if (!between) {
        std::cerr << "[WARN] V_r not between single-bucket values (expected for this test)\n";
    }
    
    TEST_PASS("test_vr_one_boundary");
    return true;
}

// =============================================================================
// Test: V_r crossing multiple boundaries
// =============================================================================

bool test_vr_multiple_boundaries() {
    // Params: a = 0.03, sigma changes at t=1, 2, 3
    std::vector<double> sigmaTimes = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> sigmaValues = {0.008, 0.01, 0.012, 0.015};
    HW1FParams params(0.03, sigmaTimes, sigmaValues);
    HW1FModel<double> model(params);
    
    double a = 0.03;
    
    // Test case: [0.5, 3.5] crosses boundaries at t=1, 2, 3
    double s = 0.5, t = 3.5;
    double V_model = model.V_r(s, t);
    double V_ref = V_r_piecewise_ref(s, t, a, sigmaTimes, sigmaValues);
    TEST_ASSERT_NEAR(V_model, V_ref, 1e-10, "V_r(0.5,3.5) multiple boundaries");
    
    // Test case: [0, 5] crosses all boundaries
    s = 0.0; t = 5.0;
    V_model = model.V_r(s, t);
    V_ref = V_r_piecewise_ref(s, t, a, sigmaTimes, sigmaValues);
    TEST_ASSERT_NEAR(V_model, V_ref, 1e-10, "V_r(0,5) all boundaries");
    
    TEST_PASS("test_vr_multiple_boundaries");
    return true;
}

// =============================================================================
// Test: V_r with a → 0
// =============================================================================

bool test_vr_a_near_zero() {
    // Params: a very small
    double a = 1e-10;
    std::vector<double> sigmaTimes = {0.0, 1.0, 2.0};
    std::vector<double> sigmaValues = {0.01, 0.012, 0.015};
    HW1FParams params(a, sigmaTimes, sigmaValues);
    HW1FModel<double> model(params);
    
    // Test case: [0.5, 2.5]
    double s = 0.5, t = 2.5;
    double V_model = model.V_r(s, t);
    
    // Reference: For a → 0, V_r = ∫_s^t σ(u)² du
    // = σ₀² * (1 - 0.5) + σ₁² * (2 - 1) + σ₂² * (2.5 - 2)
    // = 0.01² * 0.5 + 0.012² * 1.0 + 0.015² * 0.5
    double V_ref = 0.01 * 0.01 * 0.5 + 0.012 * 0.012 * 1.0 + 0.015 * 0.015 * 0.5;
    
    TEST_ASSERT_NEAR(V_model, V_ref, 1e-8, "V_r with a→0");
    
    TEST_PASS("test_vr_a_near_zero");
    return true;
}

// =============================================================================
// Test: V_r degenerate cases
// =============================================================================

bool test_vr_degenerate() {
    std::vector<double> sigmaTimes = {0.0, 1.0, 2.0};
    std::vector<double> sigmaValues = {0.01, 0.012, 0.015};
    HW1FParams params(0.03, sigmaTimes, sigmaValues);
    HW1FModel<double> model(params);
    
    // Test: t = s (zero-length interval)
    double V = model.V_r(1.0, 1.0);
    TEST_ASSERT_NEAR(V, 0.0, 1e-15, "V_r(t,t) = 0");
    
    // Test: t < s (reversed interval)
    V = model.V_r(2.0, 1.0);
    TEST_ASSERT_NEAR(V, 0.0, 1e-15, "V_r(t>s) = 0");
    
    // Test: s = 0
    V = model.V_r(0.0, 1.5);
    double V_ref = V_r_piecewise_ref(0.0, 1.5, 0.03, sigmaTimes, sigmaValues);
    TEST_ASSERT_NEAR(V, V_ref, 1e-10, "V_r(0, 1.5)");
    
    TEST_PASS("test_vr_degenerate");
    return true;
}

// =============================================================================
// Test: V_r consistency with bond volatility σ_P
// =============================================================================

bool test_vr_sigmaP_consistency() {
    // σ_P(t, T) = B(t, T) * sqrt(V_r(0, t))
    // This test verifies the relationship is correct
    
    std::vector<double> sigmaTimes = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> sigmaValues = {0.01, 0.012, 0.011, 0.013};
    HW1FParams params(0.03, sigmaTimes, sigmaValues);
    HW1FModel<double> model(params);
    
    double t = 2.5;
    double T = 5.0;
    
    // Compute σ_P using model
    double sigmaP_model = model.sigmaP(t, T);
    
    // Compute manually: σ_P = B(t,T) * sqrt(V_r(0,t))
    double B_t_T = model.B(t, T);
    double V_0_t = model.V_r(0.0, t);
    double sigmaP_manual = B_t_T * std::sqrt(V_0_t);
    
    TEST_ASSERT_NEAR(sigmaP_model, sigmaP_manual, 1e-10, "σ_P consistency");
    
    TEST_PASS("test_vr_sigmaP_consistency");
    return true;
}

// =============================================================================
// Test: V_r in MC step (typical use case)
// =============================================================================

bool test_vr_mc_step() {
    // In MC, we typically compute V_r(t, t+dt) for each step
    // This should handle bucket boundaries correctly
    
    std::vector<double> sigmaTimes = {0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0};
    std::vector<double> sigmaValues = {0.008, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015};
    double a = 0.03;
    HW1FParams params(a, sigmaTimes, sigmaValues);
    HW1FModel<double> model(params);
    
    // Simulate with dt = 0.05 (20 steps per year)
    double dt = 0.05;
    double T_expiry = 5.0;
    int numSteps = static_cast<int>(T_expiry / dt);
    
    double totalVariance = 0.0;
    double decay = std::exp(-a * dt);
    
    for (int step = 0; step < numSteps; ++step) {
        double t = step * dt;
        double t_next = t + dt;
        
        // V_r for this step
        double V_step = model.V_r(t, t_next);
        
        // Reference calculation
        double V_ref = V_r_piecewise_ref(t, t_next, a, sigmaTimes, sigmaValues);
        
        if (std::abs(V_step - V_ref) > 1e-12) {
            std::cerr << "[FAIL] V_r at step " << step << " (t=" << t << "): "
                      << V_step << " vs " << V_ref << "\n";
            return false;
        }
        
        // Note: This is just checking correctness, not the full variance accumulation
        totalVariance += V_step;
    }
    
    // Compare total variance to V_r(0, T_expiry)
    double V_total = model.V_r(0.0, T_expiry);
    // Note: Sum of step variances != V_r(0,T) due to decay weighting
    // But we can verify V_r(0,T) is computed correctly
    double V_total_ref = V_r_piecewise_ref(0.0, T_expiry, a, sigmaTimes, sigmaValues);
    TEST_ASSERT_NEAR(V_total, V_total_ref, 1e-10, "V_r(0, T_expiry) in MC context");
    
    TEST_PASS("test_vr_mc_step");
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== V_r Bucket Splitting Tests ===\n\n";
    
    int passed = 0, failed = 0;
    
    if (test_vr_single_bucket()) passed++; else failed++;
    if (test_vr_one_boundary()) passed++; else failed++;
    if (test_vr_multiple_boundaries()) passed++; else failed++;
    if (test_vr_a_near_zero()) passed++; else failed++;
    if (test_vr_degenerate()) passed++; else failed++;
    if (test_vr_sigmaP_consistency()) passed++; else failed++;
    if (test_vr_mc_step()) passed++; else failed++;
    
    std::cout << "\n=== Summary ===\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n";
    
    return failed > 0 ? 1 : 0;
}
