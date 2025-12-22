// =============================================================================
// HW1F Library - Greeks Comparison Application
// Compares: FD Naive, FD + Chain Rule, IFT, XAD + IFT
// =============================================================================

#include "curve/discount_curve.hpp"
#include "instruments/swaption.hpp"
#include "hw1f/hw1f_model.hpp"
#include "pricing/jamshidian/jamshidian.hpp"
#include "pricing/montecarlo/montecarlo.hpp"
#include "calibration/calibration.hpp"
#include "risk/ift/ift_greeks.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>

using namespace hw1f;

// =============================================================================
// Market Data
// =============================================================================

DiscountCurve<double> createCurve() {
    // 12 curve nodes as requested
    std::vector<double> times = {
        0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0
    };
    std::vector<double> rates = {
        0.043, 0.042, 0.041, 0.040, 0.039, 0.0385, 0.038, 0.037, 0.036, 0.035, 0.034, 0.033
    };
    return DiscountCurve<double>::fromZeroRates(times, rates);
}

ATMVolSurface<double> createVolSurface() {
    // 9x9 ATM Swaption Vol Surface (matching HullWhiteSwaption project)
    // Expiries: 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y
    // Tenors:   1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 15Y, 20Y, 30Y
    std::vector<double> expiries = {1.0/12, 3.0/12, 6.0/12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0};
    std::vector<double> tenors = {1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0};
    
    // ATM Normal Vols converted to Black vols (approximately vol_black = vol_normal / forward)
    // Using reasonable Black vol levels for current market
    std::vector<std::vector<double>> vols = {
        // 1M expiry
        {0.78, 0.65, 0.55, 0.46, 0.42, 0.39, 0.36, 0.34, 0.32},
        // 3M expiry
        {0.72, 0.60, 0.52, 0.44, 0.40, 0.38, 0.35, 0.33, 0.31},
        // 6M expiry
        {0.65, 0.55, 0.48, 0.42, 0.38, 0.36, 0.34, 0.32, 0.30},
        // 1Y expiry
        {0.55, 0.48, 0.43, 0.38, 0.35, 0.33, 0.31, 0.30, 0.28},
        // 2Y expiry
        {0.48, 0.42, 0.38, 0.34, 0.32, 0.30, 0.28, 0.27, 0.26},
        // 3Y expiry
        {0.42, 0.38, 0.35, 0.32, 0.30, 0.28, 0.27, 0.26, 0.25},
        // 5Y expiry
        {0.38, 0.35, 0.32, 0.30, 0.28, 0.27, 0.26, 0.25, 0.24},
        // 7Y expiry
        {0.35, 0.32, 0.30, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23},
        // 10Y expiry
        {0.32, 0.30, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22}
    };
    
    return ATMVolSurface<double>(expiries, tenors, vols);
}

// =============================================================================
// Main Comparison
// =============================================================================

int main() {
    std::cout << std::string(80, '=') << "\n";
    std::cout << "GREEKS COMPARISON: FD Naive vs FD+Chain vs IFT vs XAD+IFT\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    // Setup
    auto curve = createCurve();
    auto volSurface = createVolSurface();
    
    std::cout << "Market Data:\n";
    std::cout << "  Curve nodes: " << curve.size() << "\n";
    std::cout << "  Vol surface: " << volSurface.numExpiries() << " x " << volSurface.numTenors() 
              << " = " << volSurface.numNodes() << " nodes\n\n";
    
    // Calibration instruments - CO-TERMINAL style for 7Y x 20Y target
    // Using available vol surface nodes (tenors: 1,2,3,5,7,10,15,20,30Y)
    // All instruments use 20Y tenor (same as target) for consistent hedging
    // Need 10 instruments for 10 parameters (1 a + 9 sigma buckets)
    std::vector<std::pair<double, double>> calibInst = {
        // Short expiries with 20Y tenor (matches target's underlying)
        {1.0/12, 20.0},  // 1M x 20Y  -> sigma bucket [0, 1M)
        {3.0/12, 20.0},  // 3M x 20Y  -> sigma bucket [1M, 3M)
        {6.0/12, 20.0},  // 6M x 20Y  -> sigma bucket [3M, 6M)
        {1.0, 20.0},     // 1Y x 20Y  -> sigma bucket [6M, 1Y)
        {2.0, 20.0},     // 2Y x 20Y  -> sigma bucket [1Y, 2Y)
        {3.0, 20.0},     // 3Y x 20Y  -> sigma bucket [2Y, 3Y)
        {5.0, 20.0},     // 5Y x 20Y  -> sigma bucket [3Y, 5Y)
        {7.0, 20.0},     // 7Y x 20Y  -> sigma bucket [5Y, 7Y) ← TARGET
        {10.0, 20.0},    // 10Y x 20Y -> sigma bucket [7Y, inf)
        
        // Additional instrument for mean reversion identification
        {1.0, 10.0},     // 1Y x 10Y  -> different tenor helps identify 'a'
    };
    
    // 9 sigma buckets matching calibration instrument expiries
    std::vector<double> sigmaTimes = {0.0, 1.0/12, 3.0/12, 6.0/12, 1.0, 2.0, 3.0, 5.0, 7.0};  // Bucket boundaries
    std::vector<double> sigmaInit = {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01};  // Initial sigma per bucket
    
    std::cout << "Calibration instruments: " << calibInst.size() << " swaptions\n";
    std::cout << "Sigma buckets (piecewise-constant): " << sigmaTimes.size() << "\n";
    std::cout << "  Buckets: [0,1M), [1M,3M), [3M,6M), [6M,1Y), [1Y,2Y), [2Y,3Y), [3Y,5Y), [5Y,7Y), [7Y,inf)\n\n";
    
    // Calibrate
    std::cout << std::string(80, '-') << "\n";
    std::cout << "CALIBRATION (Piecewise-Constant Sigma)\n";
    std::cout << std::string(80, '-') << "\n";
    
    CalibrationEngine<double> calibEngine(curve, volSurface);
    for (const auto& [e, t] : calibInst) {
        calibEngine.addInstrument(e, t);
    }
    
    // Initialize with piecewise-constant sigma (one value per bucket)
    HW1FParams initialParams(0.03, sigmaTimes, sigmaInit);
    auto calibResult = calibEngine.calibrate(initialParams, 100, 1e-8, false);
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  a (mean reversion) = " << calibResult.params.a << "\n";
    std::cout << "  Calibrated sigma values (piecewise-constant):\n";
    for (size_t i = 0; i < calibResult.params.sigmaValues.size(); ++i) {
        double tStart = calibResult.params.sigmaTimes[i];
        double tEnd = (i + 1 < calibResult.params.sigmaTimes.size()) 
                      ? calibResult.params.sigmaTimes[i + 1] : 100.0;
        std::cout << "    sigma_" << i+1 << " [" << tStart << "Y, " << tEnd << "Y): " 
                  << std::setprecision(6) << calibResult.params.sigmaValues[i] << "\n";
    }
    std::cout << "  RMSE               = $" << std::setprecision(4) << calibResult.rmse << "\n";
    std::cout << "  Time               = " << std::setprecision(3) << calibResult.elapsedTime << "s\n\n";
    
    // Target swaption - 7Y x 20Y to get non-zero Greeks for all buckets
    double expiry = 7.0;   // All 5 sigma buckets [0,1), [1,2), [2,3), [3,5), [5,7) contribute
    double tenor = 20.0;   // Swap from year 7 to 27, many curve nodes affected
    double notional = 1e6;
    
    VanillaSwap swap(expiry, expiry + tenor, 0.0, notional, true);
    swap.fixedRate = forwardSwapRate(swap, curve);
    EuropeanSwaption swaption(expiry, swap);
    
    std::cout << "Target: " << expiry << "Y x " << tenor << "Y ATM Payer Swaption\n";
    std::cout << "  Strike (ATM) = " << std::setprecision(4) << swap.fixedRate * 100 << "%\n";
    
    // Compute Jamshidian (analytic) price for reference
    HW1FModel<double> model(calibResult.params);
    JamshidianPricer<double, double> jamPricer(model, curve);
    double jamPrice = jamPricer.price(swaption);
    std::cout << "  Jamshidian (analytic) price: $" << std::setprecision(2) << jamPrice << "\n\n";
    
    // MC Config for Greeks
    MCConfig mcConfig(5000, 50, true, 42);  // Increased paths for better convergence
    
    // ==========================================================================
    // Method 1: FD Naive (bump and recalibrate)
    // ==========================================================================
    std::cout << std::string(80, '-') << "\n";
    std::cout << "METHOD 1: FINITE DIFFERENCES (Naive - Bump & Recalibrate)\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "For each of " << volSurface.numNodes() << " vol nodes: bump -> recalibrate -> reprice\n";
    std::cout << "Total: " << volSurface.numNodes() << " recalibrations + " << volSurface.numNodes() << " MC pricings\n\n";
    
    FDGreeksEngine<double> fdEngine(curve, volSurface, calibInst);
    
    Timer timer;
    timer.start();
    auto fdNaiveGreeks = fdEngine.computeNaiveFD(swaption, calibResult.params, mcConfig);
    double fdNaiveTime = timer.elapsed();
    
    std::cout << "FD Naive Time: " << std::setprecision(3) << fdNaiveTime << "s\n";
    std::cout << "Price: $" << std::setprecision(2) << fdNaiveGreeks.price << "\n\n";
    
    // ==========================================================================
    // Method 2: FD + Chain Rule
    // ==========================================================================
    std::cout << std::string(80, '-') << "\n";
    std::cout << "METHOD 2: FD + CHAIN RULE\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "Compute dV/d(a,sigma) once with FD, then dphi/dvol via recalibration\n";
    std::cout << "dV/dvol = dV/da * da/dvol + dV/dsigma * dsigma/dvol\n\n";
    
    ChainRuleGreeksEngine<double> chainEngine(curve, volSurface, calibInst);
    
    timer.start();
    auto chainGreeks = chainEngine.computeChainRule(swaption, calibResult.params, mcConfig);
    double chainTime = timer.elapsed();
    
    std::cout << "FD + Chain Time: " << std::setprecision(3) << chainTime << "s\n";
    std::cout << "Price: $" << std::setprecision(2) << chainGreeks.price << "\n";
    std::cout << "dV/da = $" << std::setprecision(2) << chainGreeks.dVda << "\n";
    std::cout << "dV/dsigma (per bucket):\n";
    for (size_t i = 0; i < chainGreeks.dVdsigma.size(); ++i) {
        std::cout << "  dV/dsigma_" << i+1 << " = $" << chainGreeks.dVdsigma[i] << "\n";
    }
    std::cout << "\n";
    
    // ==========================================================================
    // Method 3: FD + IFT (Finite Differences + Implicit Function Theorem)
    // ==========================================================================
    std::cout << std::string(80, '-') << "\n";
    std::cout << "METHOD 3: FD + IFT (Finite Diff for dV/dphi, IFT for dphi/dm)\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "dV/dphi: 2K FD bumps (K = " << calibResult.params.numParams() << " HW params)\n";
    std::cout << "dphi/dm: IFT - NO recalibrations needed!\n\n";
    
    IFTGreeksEngine<double> iftEngine(curve, volSurface, calibInst);
    
    timer.start();
    auto iftGreeks = iftEngine.computeIFT(swaption, calibResult.params, mcConfig);
    double iftTime = timer.elapsed();
    
    std::cout << "IFT Time: " << std::setprecision(3) << iftTime << "s\n";
    std::cout << "Price: $" << std::setprecision(2) << iftGreeks.price << "\n";
    std::cout << "dV/da = $" << std::setprecision(2) << iftGreeks.dVda << "\n";
    std::cout << "dV/dsigma (per bucket):\n";
    for (size_t i = 0; i < iftGreeks.dVdsigma.size(); ++i) {
        std::cout << "  dV/dsigma_" << i+1 << " = $" << iftGreeks.dVdsigma[i] << "\n";
    }
    std::cout << "\n";
    
    // ==========================================================================
    // Method 4: XAD + IFT (Adjoint AD + Implicit Function Theorem)
    // ==========================================================================
    std::cout << std::string(80, '-') << "\n";
    std::cout << "METHOD 4: XAD + IFT (Adjoint AD for dV/dphi, IFT for dphi/dm)\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "dV/dphi: 1 AAD backward pass (ALL " << calibResult.params.numParams() << " sensitivities at once)\n";
    std::cout << "dphi/dm: IFT - NO recalibrations needed!\n";
    std::cout << "NOTE: XAD has tape overhead for MC paths - may be slower for small K\n\n";
    
    XADIFTGreeksEngine<double> xadIftEngine(curve, volSurface, calibInst);
    
    timer.start();
    auto xadIftGreeks = xadIftEngine.computeXADIFT(swaption, calibResult.params, mcConfig);
    double xadIftTime = timer.elapsed();
    
    std::cout << "XAD + IFT Time: " << std::setprecision(3) << xadIftTime << "s\n";
    std::cout << "Price: $" << std::setprecision(2) << xadIftGreeks.price << "\n";
    std::cout << "dV/da = $" << std::setprecision(2) << xadIftGreeks.dVda << "\n";
    std::cout << "dV/dsigma (per bucket - ALL computed in single backward pass!):\n";
    for (size_t i = 0; i < xadIftGreeks.dVdsigma.size(); ++i) {
        std::cout << "  dV/dsigma_" << i+1 << " = $" << xadIftGreeks.dVdsigma[i] << "\n";
    }
    std::cout << "\n";
    
    // ==========================================================================
    // Comparison Summary
    // ==========================================================================
    std::cout << std::string(80, '=') << "\n";
    std::cout << "TIMING COMPARISON\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    std::cout << std::left << std::setw(25) << "Method" 
              << std::right << std::setw(12) << "Time (s)"
              << std::setw(15) << "Speedup\n";
    std::cout << std::string(52, '-') << "\n";
    
    std::cout << std::setw(25) << "FD Naive"
              << std::setw(12) << std::setprecision(3) << fdNaiveTime
              << std::setw(15) << "1.0x (baseline)\n";
    
    std::cout << std::setw(25) << "FD + Chain Rule"
              << std::setw(12) << chainTime
              << std::setw(15) << std::setprecision(1) << fdNaiveTime / chainTime << "x\n";
    
    std::cout << std::setw(25) << "FD + IFT"
              << std::setw(12) << std::setprecision(3) << iftTime
              << std::setw(15) << std::setprecision(1) << fdNaiveTime / iftTime << "x\n";
    
    std::cout << std::setw(25) << "XAD + IFT"
              << std::setw(12) << std::setprecision(3) << xadIftTime
              << std::setw(15) << std::setprecision(1) << fdNaiveTime / xadIftTime << "x\n";
    
    // ==========================================================================
    // PRICE COMPARISON
    // ==========================================================================
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "SWAPTION PRICE COMPARISON\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    std::cout << std::left << std::setw(25) << "Method" 
              << std::right << std::setw(15) << "Price ($)"
              << std::setw(18) << "Diff vs Jam.\n";
    std::cout << std::string(58, '-') << "\n";
    
    std::cout << std::setw(25) << "Jamshidian (analytic)"
              << std::setw(15) << std::setprecision(2) << jamPrice
              << std::setw(18) << "-\n";
    
    std::cout << std::setw(25) << "FD Naive (MC)"
              << std::setw(15) << std::setprecision(2) << fdNaiveGreeks.price
              << std::setw(17) << std::setprecision(2) << std::abs(fdNaiveGreeks.price - jamPrice) / jamPrice * 100 << "%\n";
    
    std::cout << std::setw(25) << "FD + Chain Rule (MC)"
              << std::setw(15) << chainGreeks.price
              << std::setw(17) << std::abs(chainGreeks.price - jamPrice) / jamPrice * 100 << "%\n";
    
    std::cout << std::setw(25) << "FD + IFT (MC)"
              << std::setw(15) << iftGreeks.price
              << std::setw(17) << std::abs(iftGreeks.price - jamPrice) / jamPrice * 100 << "%\n";
    
    std::cout << std::setw(25) << "XAD + IFT (MC)"
              << std::setw(15) << xadIftGreeks.price
              << std::setw(17) << std::abs(xadIftGreeks.price - jamPrice) / jamPrice * 100 << "%\n";
    
    // ==========================================================================
    // SIGMA BUCKET GREEKS COMPARISON (Piecewise-Constant Volatility)
    // ==========================================================================
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "SIGMA BUCKET GREEKS (dV/d sigma_k) - PIECEWISE-CONSTANT VOLATILITY\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    std::cout << std::left << std::setw(20) << "Sigma Bucket"
              << std::right << std::setw(12) << "FD+Chain"
              << std::setw(12) << "FD+IFT"
              << std::setw(12) << "XAD+IFT\n";
    std::cout << std::string(56, '-') << "\n";
    
    for (size_t i = 0; i < calibResult.params.sigmaValues.size(); ++i) {
        double tStart = calibResult.params.sigmaTimes[i];
        double tEnd = (i + 1 < calibResult.params.sigmaTimes.size()) 
                      ? calibResult.params.sigmaTimes[i + 1] : 100.0;
        
        // Format bucket label properly for fractional years
        std::ostringstream bucketLabel;
        bucketLabel << std::fixed << std::setprecision(2);
        bucketLabel << "sigma_" << (i+1) << " [" << tStart << "," << tEnd << ")";
        std::string bucket = bucketLabel.str();
        
        double fdC = (i < chainGreeks.dVdsigma.size()) ? chainGreeks.dVdsigma[i] : 0.0;
        double ift = (i < iftGreeks.dVdsigma.size()) ? iftGreeks.dVdsigma[i] : 0.0;
        double xad = (i < xadIftGreeks.dVdsigma.size()) ? xadIftGreeks.dVdsigma[i] : 0.0;
        
        std::cout << std::setw(25) << bucket
                  << std::setw(12) << std::setprecision(2) << fdC
                  << std::setw(12) << ift
                  << std::setw(12) << xad << "\n";
    }
    std::cout << "\nNote: XAD+IFT computes ALL sigma bucket Greeks in a single backward pass!\n";
    std::cout << "      This is O(1) cost vs O(K) for other methods where K = number of buckets.\n";
    
    // ==========================================================================
    // Vol Greeks Comparison
    // ==========================================================================
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "VOL SURFACE GREEKS COMPARISON (per 1bp)\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    std::cout << std::left << std::setw(12) << "Node"
              << std::right << std::setw(12) << "FD Naive"
              << std::setw(12) << "FD+Chain"
              << std::setw(12) << "IFT"
              << std::setw(12) << "XAD+IFT\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (size_t ei = 0; ei < volSurface.numExpiries(); ++ei) {
        for (size_t ti = 0; ti < volSurface.numTenors(); ++ti) {
            std::string node = std::to_string(static_cast<int>(volSurface.expiries()[ei])) + "Yx" +
                              std::to_string(static_cast<int>(volSurface.tenors()[ti])) + "Y";
            
            double fdN = fdNaiveGreeks.volGreeks[ei][ti] * 0.0001;
            double fdC = chainGreeks.volGreeks[ei][ti] * 0.0001;
            double ift = iftGreeks.volGreeks[ei][ti] * 0.0001;
            double xad = xadIftGreeks.volGreeks[ei][ti] * 0.0001;
            
            std::cout << std::setw(12) << node
                      << std::setw(12) << std::setprecision(2) << fdN
                      << std::setw(12) << fdC
                      << std::setw(12) << ift
                      << std::setw(12) << xad << "\n";
        }
    }
    
    // ==========================================================================
    // Curve Greeks Comparison
    // ==========================================================================
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "CURVE NODE GREEKS COMPARISON (per 1bp)\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    std::cout << std::left << std::setw(12) << "Maturity"
              << std::right << std::setw(12) << "FD Naive"
              << std::setw(12) << "FD+Chain"
              << std::setw(12) << "IFT"
              << std::setw(12) << "XAD+IFT\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (size_t i = 0; i < curve.size(); ++i) {
        double fdN = fdNaiveGreeks.curveGreeks[i] * 0.0001;
        double fdC = chainGreeks.curveGreeks[i] * 0.0001;
        double ift = iftGreeks.curveGreeks[i] * 0.0001;
        double xad = xadIftGreeks.curveGreeks[i] * 0.0001;
        
        std::cout << std::setw(12) << std::setprecision(2) << curve.times()[i]
                  << std::setw(12) << std::setprecision(2) << fdN
                  << std::setw(12) << fdC
                  << std::setw(12) << ift
                  << std::setw(12) << xad << "\n";
    }
    
    // ==========================================================================
    // Method Explanation
    // ==========================================================================
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "METHOD EXPLANATION (OpenGamma Adjoint-IFT)\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    std::cout << "NOTATION (from OpenGamma paper):\n";
    std::cout << "  C = curve nodes, Theta = vol surface nodes, Phi = calibrated HW params\n";
    std::cout << "  f(C, Theta, Phi) = 0  (calibration condition)\n";
    std::cout << "  V(C, Phi(C, Theta)) = exotic price (MC swaption)\n\n";
    
    std::cout << "FD NAIVE:\n";
    std::cout << "  For each market node m_i:\n";
    std::cout << "    1. Bump m_i\n";
    std::cout << "    2. Recalibrate HW params\n";
    std::cout << "    3. Reprice swaption\n";
    std::cout << "  Cost: O(N) recalibrations + O(N) pricings\n\n";
    
    std::cout << "FD + CHAIN RULE:\n";
    std::cout << "  1. Compute V_Phi = dV/dPhi using FD (2K bumps for K params)\n";
    std::cout << "  2. For each vol node, recalibrate to get dPhi/dTheta\n";
    std::cout << "  3. Chain rule: dV/dTheta = V_Phi * dPhi/dTheta\n";
    std::cout << "  Benefit: No per-node MC repricing!\n";
    std::cout << "  Cost: 2K pricings + N recalibrations\n\n";
    
    std::cout << "FD + IFT (OpenGamma Adjoint-IFT with FD for V_Phi):\n";
    std::cout << "  Key formula: dV/dm = V_m_direct - lambda^T * f_m\n";
    std::cout << "  where lambda = solve(f_Phi^T, V_Phi)\n";
    std::cout << "  Steps:\n";
    std::cout << "    1. Compute V_Phi = dV/dPhi using FD (2K pricings)\n";
    std::cout << "    2. Build f_Phi = J^T * J (Gauss-Newton Hessian)\n";
    std::cout << "    3. Solve lambda from f_Phi^T * lambda = V_Phi (ONE solve!)\n";
    std::cout << "    4. For each vol node: f_Theta = J^T * (dr/dTheta)\n";
    std::cout << "    5. dV/dTheta = 0 - lambda^T * f_Theta\n";
    std::cout << "  Benefit: NO recalibrations! Only 1 linear solve for ALL nodes!\n";
    std::cout << "  Cost: 2K pricings + 1 solve + N matrix-vector products\n\n";
    
    std::cout << "XAD + IFT (OpenGamma Adjoint-IFT with XAD for V_Phi and f_Theta):\n";
    std::cout << "  Same formula: dV/dm = V_m_direct - lambda^T * f_m\n";
    std::cout << "  Steps:\n";
    std::cout << "    1. V_Phi: 1 AAD backward pass through MC pricing\n";
    std::cout << "    2. f_Theta: K AAD backward passes (one per calibration instrument)\n";
    std::cout << "       -> Each pass gives dr_k/dTheta for ALL " << volSurface.numExpiries() << "x" << volSurface.numTenors() << "=" << volSurface.numNodes() << " vol nodes!\n";
    std::cout << "    3. Solve lambda from f_Phi^T * lambda = V_Phi (ONE solve!)\n";
    std::cout << "    4. dV/dTheta = 0 - lambda^T * f_Theta for each node\n";
    std::cout << "  Benefit: K AAD passes instead of N FD bumps for vol Greeks!\n";
    std::cout << "  Cost: 1 + K AAD passes + 1 solve (NO recalibrations!)\n";
    std::cout << "        For N=" << volSurface.numNodes() << " vol nodes, K=" << calibInst.size() << " instruments\n\n";
    
    std::cout << std::string(80, '=') << "\n";
    std::cout << "COMPUTATION COMPLETE\n";
    std::cout << std::string(80, '=') << "\n";
    
    // Save results to file
    std::ofstream outFile("greeks_comparison_results.txt");
    if (outFile.is_open()) {
        outFile << "GREEKS COMPARISON RESULTS\n";
        outFile << "========================\n\n";
        outFile << "Timing Summary:\n";
        outFile << "FD Naive:      " << fdNaiveTime << "s\n";
        outFile << "FD + Chain:    " << chainTime << "s (speedup: " << fdNaiveTime/chainTime << "x)\n";
        outFile << "FD + IFT:      " << iftTime << "s (speedup: " << fdNaiveTime/iftTime << "x)\n";
        outFile << "XAD + IFT:     " << xadIftTime << "s (speedup: " << fdNaiveTime/xadIftTime << "x)\n";
        outFile.close();
        std::cout << "\nResults saved to greeks_comparison_results.txt\n";
    }
    
    return 0;
}
