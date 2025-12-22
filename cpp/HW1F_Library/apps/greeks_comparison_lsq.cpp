// =============================================================================
// HW1F Library - Greeks Comparison Application (Least-Squares Calibration)
// Over-determined calibration: 81 vol surface nodes -> 10 HW parameters
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
// Market Data (same as original)
// =============================================================================

DiscountCurve<double> createCurve() {
    std::vector<double> times = {
        0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0
    };
    std::vector<double> rates = {
        0.043, 0.042, 0.041, 0.040, 0.039, 0.0385, 0.038, 0.037, 0.036, 0.035, 0.034, 0.033
    };
    return DiscountCurve<double>::fromZeroRates(times, rates);
}

ATMVolSurface<double> createVolSurface() {
    // 9x9 ATM Swaption Vol Surface
    // Expiries: 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y
    // Tenors:   1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 15Y, 20Y, 30Y
    std::vector<double> expiries = {1.0/12, 3.0/12, 6.0/12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0};
    std::vector<double> tenors = {1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0};
    
    // ATM Black vols
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
// Main: Least-Squares Calibration with Full Vol Surface
// =============================================================================

int main() {
    std::cout << std::string(80, '=') << "\n";
    std::cout << "GREEKS COMPARISON: LEAST-SQUARES CALIBRATION (Over-Determined)\n";
    std::cout << "81 vol surface instruments -> 10 HW parameters\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    // Setup
    auto curve = createCurve();
    auto volSurface = createVolSurface();
    
    std::cout << "Market Data:\n";
    std::cout << "  Curve nodes: " << curve.size() << "\n";
    std::cout << "  Vol surface: " << volSurface.numExpiries() << " x " << volSurface.numTenors() 
              << " = " << volSurface.numNodes() << " nodes\n\n";
    
    // ==========================================================================
    // LEAST-SQUARES CALIBRATION
    // Use ALL 81 vol surface nodes as calibration instruments
    // This is over-determined: 81 instruments >> 10 parameters
    // ==========================================================================
    
    // Build calibration instruments from entire vol surface
    std::vector<std::pair<double, double>> calibInst;
    for (size_t ei = 0; ei < volSurface.numExpiries(); ++ei) {
        for (size_t ti = 0; ti < volSurface.numTenors(); ++ti) {
            double expiry = volSurface.expiries()[ei];
            double tenor = volSurface.tenors()[ti];
            calibInst.push_back({expiry, tenor});
        }
    }
    
    std::cout << "OVER-DETERMINED LEAST-SQUARES CALIBRATION\n";
    std::cout << "==========================================\n";
    std::cout << "  Calibration instruments: " << calibInst.size() << " (ALL vol surface nodes)\n";
    std::cout << "  HW parameters: 10 (1 mean reversion + 9 sigma buckets)\n";
    std::cout << "  Ratio: " << calibInst.size() << "/" << 10 << " = " 
              << std::setprecision(1) << calibInst.size() / 10.0 << "x over-determined\n";
    std::cout << "  Objective: min sum_i (P_HW_i - P_Black_i)^2 / " << calibInst.size() << "\n\n";
    
    // 9 sigma buckets matching vol surface expiries
    std::vector<double> sigmaTimes = {0.0, 1.0/12, 3.0/12, 6.0/12, 1.0, 2.0, 3.0, 5.0, 7.0};
    std::vector<double> sigmaInit = {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01};
    
    std::cout << "Sigma buckets (piecewise-constant): " << sigmaTimes.size() << "\n";
    std::cout << "  Buckets: [0,1M), [1M,3M), [3M,6M), [6M,1Y), [1Y,2Y), [2Y,3Y), [3Y,5Y), [5Y,7Y), [7Y,inf)\n\n";
    
    // Calibrate with all vol surface nodes
    std::cout << std::string(80, '-') << "\n";
    std::cout << "CALIBRATION (Least-Squares to Full Vol Surface)\n";
    std::cout << std::string(80, '-') << "\n";
    
    CalibrationEngine<double> calibEngine(curve, volSurface);
    for (const auto& [e, t] : calibInst) {
        calibEngine.addInstrument(e, t);
    }
    
    HW1FParams initialParams(0.03, sigmaTimes, sigmaInit);
    auto calibResult = calibEngine.calibrate(initialParams, 200, 1e-10, false);  // More iterations for LSQ
    
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
    std::cout << "  Iterations         = " << calibResult.iterations << "\n";
    std::cout << "  Time               = " << std::setprecision(3) << calibResult.elapsedTime << "s\n";
    std::cout << "\n  NOTE: RMSE > 0 because we have 81 instruments for 10 parameters\n";
    std::cout << "        This is a BEST FIT, not an exact fit!\n\n";
    
    // Target swaption - 7Y x 20Y
    double expiry = 7.0;
    double tenor = 20.0;
    double notional = 1e6;
    
    VanillaSwap swap(expiry, expiry + tenor, 0.0, notional, true);
    swap.fixedRate = forwardSwapRate(swap, curve);
    EuropeanSwaption swaption(expiry, swap);
    
    std::cout << "Target: " << expiry << "Y x " << tenor << "Y ATM Payer Swaption\n";
    std::cout << "  Strike (ATM) = " << std::setprecision(4) << swap.fixedRate * 100 << "%\n";
    
    // Jamshidian price
    HW1FModel<double> model(calibResult.params);
    JamshidianPricer<double, double> jamPricer(model, curve);
    double jamPrice = jamPricer.price(swaption);
    std::cout << "  Jamshidian (analytic) price: $" << std::setprecision(2) << jamPrice << "\n\n";
    
    // MC Config
    MCConfig mcConfig(5000, 50, true, 42);
    
    // ==========================================================================
    // Method 1: FD Naive (bump and recalibrate)
    // ==========================================================================
    std::cout << std::string(80, '-') << "\n";
    std::cout << "METHOD 1: FINITE DIFFERENCES (Naive - Bump & Recalibrate)\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "For each of " << volSurface.numNodes() << " vol nodes: bump -> recalibrate (LSQ) -> reprice\n";
    std::cout << "Total: " << volSurface.numNodes() << " least-squares calibrations + " << volSurface.numNodes() << " MC pricings\n\n";
    
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
    // Method 3: FD + IFT
    // ==========================================================================
    std::cout << std::string(80, '-') << "\n";
    std::cout << "METHOD 3: FD + IFT (Finite Diff for dV/dphi, IFT for dphi/dm)\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "IFT with OVER-DETERMINED calibration:\n";
    std::cout << "  f_Phi = J^T W J is " << calibResult.params.numParams() << "x" << calibResult.params.numParams() << " (where J is " << calibInst.size() << "x" << calibResult.params.numParams() << ")\n";
    std::cout << "  Gauss-Newton Hessian still well-conditioned for least-squares!\n\n";
    
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
    // Method 4: XAD + IFT
    // ==========================================================================
    std::cout << std::string(80, '-') << "\n";
    std::cout << "METHOD 4: XAD + IFT (Adjoint AD for dV/dphi, IFT for dphi/dm)\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "dV/dphi: 1 AAD backward pass (ALL " << calibResult.params.numParams() << " sensitivities at once)\n";
    std::cout << "f_Theta: K=" << calibInst.size() << " AAD passes for dr_k/dTheta\n";
    std::cout << "dphi/dm: IFT - NO recalibrations needed!\n\n";
    
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
    std::cout << "TIMING COMPARISON (Least-Squares with " << calibInst.size() << " instruments)\n";
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
    // Price Comparison
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
    // Sigma Bucket Greeks
    // ==========================================================================
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "SIGMA BUCKET GREEKS (dV/d sigma_k)\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    std::cout << std::left << std::setw(25) << "Sigma Bucket"
              << std::right << std::setw(12) << "FD+Chain"
              << std::setw(12) << "FD+IFT"
              << std::setw(12) << "XAD+IFT\n";
    std::cout << std::string(61, '-') << "\n";
    
    for (size_t i = 0; i < calibResult.params.sigmaValues.size(); ++i) {
        double tStart = calibResult.params.sigmaTimes[i];
        double tEnd = (i + 1 < calibResult.params.sigmaTimes.size()) 
                      ? calibResult.params.sigmaTimes[i + 1] : 100.0;
        
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
    
    // ==========================================================================
    // Vol Greeks (sample showing diverse nodes)
    // ==========================================================================
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "VOL SURFACE GREEKS (per 1bp) - SELECTED NODES\n";
    std::cout << "All " << volSurface.numNodes() << " nodes contribute to calibration!\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    std::cout << std::left << std::setw(12) << "Node"
              << std::right << std::setw(12) << "FD Naive"
              << std::setw(12) << "FD+Chain"
              << std::setw(12) << "IFT"
              << std::setw(12) << "XAD+IFT\n";
    std::cout << std::string(60, '-') << "\n";
    
    // Show a representative sample of nodes
    std::vector<std::pair<size_t, size_t>> sampleNodes = {
        {0, 0}, {0, 7},   // 1M expiry: 1Y and 20Y tenor
        {3, 0}, {3, 7},   // 1Y expiry: 1Y and 20Y tenor
        {5, 0}, {5, 7},   // 3Y expiry: 1Y and 20Y tenor
        {7, 0}, {7, 7},   // 7Y expiry: 1Y and 20Y tenor (matches target)
        {8, 0}, {8, 7}    // 10Y expiry: 1Y and 20Y tenor
    };
    
    for (const auto& [ei, ti] : sampleNodes) {
        std::ostringstream nodeLabel;
        if (volSurface.expiries()[ei] < 1.0) {
            nodeLabel << std::fixed << std::setprecision(0) 
                      << (volSurface.expiries()[ei] * 12) << "Mx";
        } else {
            nodeLabel << std::fixed << std::setprecision(0) 
                      << volSurface.expiries()[ei] << "Yx";
        }
        nodeLabel << std::fixed << std::setprecision(0) 
                  << volSurface.tenors()[ti] << "Y";
        
        double fdN = fdNaiveGreeks.volGreeks[ei][ti] * 0.0001;
        double fdC = chainGreeks.volGreeks[ei][ti] * 0.0001;
        double ift = iftGreeks.volGreeks[ei][ti] * 0.0001;
        double xad = xadIftGreeks.volGreeks[ei][ti] * 0.0001;
        
        std::cout << std::setw(12) << nodeLabel.str()
                  << std::setw(12) << std::setprecision(2) << fdN
                  << std::setw(12) << fdC
                  << std::setw(12) << ift
                  << std::setw(12) << xad << "\n";
    }
    
    std::cout << "\nNOTE: With least-squares calibration, ALL " << volSurface.numNodes() 
              << " vol nodes have non-zero Greeks!\n";
    std::cout << "      Each node contributes to the objective function.\n";
    
    // ==========================================================================
    // Curve Greeks
    // ==========================================================================
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "CURVE NODE GREEKS (per 1bp)\n";
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
    // Key Insights for Least-Squares
    // ==========================================================================
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "KEY INSIGHTS: LEAST-SQUARES vs EXACT FIT\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    std::cout << "EXACT FIT (K instruments = K parameters):\n";
    std::cout << "  - RMSE = 0 (perfect fit)\n";
    std::cout << "  - Unique solution\n";
    std::cout << "  - Greeks only non-zero for vol nodes used in calibration\n";
    std::cout << "  - Can overfit noise in market quotes\n\n";
    
    std::cout << "LEAST-SQUARES (K instruments > K parameters):\n";
    std::cout << "  - RMSE > 0 (best fit)\n";
    std::cout << "  - Smoother, more robust parameters\n";
    std::cout << "  - ALL " << volSurface.numNodes() << " vol nodes have non-zero Greeks!\n";
    std::cout << "  - Less sensitive to individual quote noise\n\n";
    
    std::cout << "IFT STILL WORKS FOR LEAST-SQUARES:\n";
    std::cout << "  - First-order optimality: f = J^T r = 0 (gradient = 0)\n";
    std::cout << "  - Gauss-Newton Hessian: f_Phi = J^T J (" << calibResult.params.numParams() << "x" << calibResult.params.numParams() << " matrix)\n";
    std::cout << "  - J is " << calibInst.size() << " x " << calibResult.params.numParams() << " (tall matrix)\n";
    std::cout << "  - J^T J is still invertible if J has full column rank!\n";
    std::cout << "  - IFT formula: dPhi/dTheta = -(J^T J)^{-1} J^T (dr/dTheta)\n\n";
    
    std::cout << std::string(80, '=') << "\n";
    std::cout << "COMPUTATION COMPLETE\n";
    std::cout << std::string(80, '=') << "\n";
    
    // Save results to file
    std::ofstream outFile("greeks_comparison_lsq_results.txt");
    if (outFile.is_open()) {
        outFile << "GREEKS COMPARISON RESULTS (LEAST-SQUARES CALIBRATION)\n";
        outFile << "=====================================================\n\n";
        outFile << "Configuration:\n";
        outFile << "  Calibration instruments: " << calibInst.size() << " (full vol surface)\n";
        outFile << "  HW parameters: " << calibResult.params.numParams() << "\n";
        outFile << "  Over-determined ratio: " << calibInst.size() / 10.0 << "x\n";
        outFile << "  Calibration RMSE: $" << calibResult.rmse << "\n\n";
        outFile << "Timing Summary:\n";
        outFile << "FD Naive:      " << fdNaiveTime << "s\n";
        outFile << "FD + Chain:    " << chainTime << "s (speedup: " << fdNaiveTime/chainTime << "x)\n";
        outFile << "FD + IFT:      " << iftTime << "s (speedup: " << fdNaiveTime/iftTime << "x)\n";
        outFile << "XAD + IFT:     " << xadIftTime << "s (speedup: " << fdNaiveTime/xadIftTime << "x)\n";
        outFile.close();
        std::cout << "\nResults saved to greeks_comparison_lsq_results.txt\n";
    }
    
    return 0;
}
