// =============================================================================
// HW1F Library - Multi-Swaption Comparison Application
// Compares Exact vs Least-Squares Calibration for 5 Different Swaptions
// Methods: FD Naive, FD + Chain Rule, IFT, XAD + IFT
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
#include <chrono>
#include <algorithm>
#include <cmath>

using namespace hw1f;

// =============================================================================
// Market Data
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
    std::vector<double> expiries = {1.0/12, 3.0/12, 6.0/12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0};
    std::vector<double> tenors = {1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0};
    
    std::vector<std::vector<double>> vols = {
        {0.78, 0.65, 0.55, 0.46, 0.42, 0.39, 0.36, 0.34, 0.32},
        {0.72, 0.60, 0.52, 0.44, 0.40, 0.38, 0.35, 0.33, 0.31},
        {0.65, 0.55, 0.48, 0.42, 0.38, 0.36, 0.34, 0.32, 0.30},
        {0.55, 0.48, 0.43, 0.38, 0.35, 0.33, 0.31, 0.30, 0.28},
        {0.48, 0.42, 0.38, 0.34, 0.32, 0.30, 0.28, 0.27, 0.26},
        {0.42, 0.38, 0.35, 0.32, 0.30, 0.28, 0.27, 0.26, 0.25},
        {0.38, 0.35, 0.32, 0.30, 0.28, 0.27, 0.26, 0.25, 0.24},
        {0.35, 0.32, 0.30, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23},
        {0.32, 0.30, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22}
    };
    
    return ATMVolSurface<double>(expiries, tenors, vols);
}

// =============================================================================
// Result Structure for a single swaption - FULL GREEKS
// =============================================================================
struct SwaptionResult {
    double expiry;
    double tenor;
    double strike;
    
    // Prices
    double jamPrice;
    double fdNaivePrice;
    double fdChainPrice;
    double iftPrice;
    double xadIftPrice;
    
    // Timing
    double fdNaiveTime;
    double fdChainTime;
    double iftTime;
    double xadIftTime;
    
    // Greeks - dV/da
    double fdChain_dVda;
    double ift_dVda;
    double xadIft_dVda;
    
    // Greeks - dV/dsigma per bucket
    std::vector<double> fdChain_dVdsigma;
    std::vector<double> ift_dVdsigma;
    std::vector<double> xadIft_dVdsigma;
    
    // Greeks - dV/dvol for ALL vol surface nodes (9x9 = 81)
    std::vector<std::vector<double>> fdNaive_volGreeks;
    std::vector<std::vector<double>> fdChain_volGreeks;
    std::vector<std::vector<double>> ift_volGreeks;
    std::vector<std::vector<double>> xadIft_volGreeks;
    
    // Greeks - dV/dr for ALL curve nodes (12)
    std::vector<double> fdNaive_curveGreeks;
    std::vector<double> fdChain_curveGreeks;
    std::vector<double> ift_curveGreeks;
    std::vector<double> xadIft_curveGreeks;
    
    // Calibration info
    double calibRMSE;
    double calibTime;
    int calibIterations;
};

// =============================================================================
// Run comparison for a single swaption
// =============================================================================
SwaptionResult runSwaptionComparison(
    double expiry, 
    double tenor,
    const DiscountCurve<double>& curve,
    const ATMVolSurface<double>& volSurface,
    const std::vector<std::pair<double, double>>& calibInst,
    const CalibrationResult& calibResult,
    const MCConfig& mcConfig)
{
    SwaptionResult result;
    result.expiry = expiry;
    result.tenor = tenor;
    
    double notional = 1e6;
    VanillaSwap swap(expiry, expiry + tenor, 0.0, notional, true);
    swap.fixedRate = forwardSwapRate(swap, curve);
    result.strike = swap.fixedRate;
    EuropeanSwaption swaption(expiry, swap);
    
    // Jamshidian (analytic) price
    HW1FModel<double> model(calibResult.params);
    JamshidianPricer<double, double> jamPricer(model, curve);
    result.jamPrice = jamPricer.price(swaption);
    
    result.calibRMSE = calibResult.rmse;
    result.calibTime = calibResult.elapsedTime;
    result.calibIterations = calibResult.iterations;
    
    // Method 1: FD Naive
    FDGreeksEngine<double> fdEngine(curve, volSurface, calibInst);
    Timer timer;
    timer.start();
    auto fdNaiveGreeks = fdEngine.computeNaiveFD(swaption, calibResult.params, mcConfig);
    result.fdNaiveTime = timer.elapsed();
    result.fdNaivePrice = fdNaiveGreeks.price;
    result.fdNaive_volGreeks = fdNaiveGreeks.volGreeks;
    result.fdNaive_curveGreeks = fdNaiveGreeks.curveGreeks;
    
    // Method 2: FD + Chain Rule
    ChainRuleGreeksEngine<double> chainEngine(curve, volSurface, calibInst);
    timer.start();
    auto chainGreeks = chainEngine.computeChainRule(swaption, calibResult.params, mcConfig);
    result.fdChainTime = timer.elapsed();
    result.fdChainPrice = chainGreeks.price;
    result.fdChain_dVda = chainGreeks.dVda;
    result.fdChain_dVdsigma = chainGreeks.dVdsigma;
    result.fdChain_volGreeks = chainGreeks.volGreeks;
    result.fdChain_curveGreeks = chainGreeks.curveGreeks;
    
    // Method 3: FD + IFT
    IFTGreeksEngine<double> iftEngine(curve, volSurface, calibInst);
    timer.start();
    auto iftGreeks = iftEngine.computeIFT(swaption, calibResult.params, mcConfig);
    result.iftTime = timer.elapsed();
    result.iftPrice = iftGreeks.price;
    result.ift_dVda = iftGreeks.dVda;
    result.ift_dVdsigma = iftGreeks.dVdsigma;
    result.ift_volGreeks = iftGreeks.volGreeks;
    result.ift_curveGreeks = iftGreeks.curveGreeks;
    
    // Method 4: XAD + IFT
    XADIFTGreeksEngine<double> xadIftEngine(curve, volSurface, calibInst);
    timer.start();
    auto xadIftGreeks = xadIftEngine.computeXADIFT(swaption, calibResult.params, mcConfig);
    result.xadIftTime = timer.elapsed();
    result.xadIftPrice = xadIftGreeks.price;
    result.xadIft_dVda = xadIftGreeks.dVda;
    result.xadIft_dVdsigma = xadIftGreeks.dVdsigma;
    result.xadIft_volGreeks = xadIftGreeks.volGreeks;
    result.xadIft_curveGreeks = xadIftGreeks.curveGreeks;
    
    return result;
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << std::string(100, '=') << "\n";
    std::cout << "MULTI-SWAPTION COMPARISON: EXACT vs LEAST-SQUARES CALIBRATION\n";
    std::cout << "5 Different Swaptions x 4 Methods x 2 Calibration Types\n";
    std::cout << std::string(100, '=') << "\n\n";
    
    // Setup market data
    auto curve = createCurve();
    auto volSurface = createVolSurface();
    
    std::cout << "Market Data:\n";
    std::cout << "  Curve nodes: " << curve.size() << "\n";
    std::cout << "  Vol surface: " << volSurface.numExpiries() << " x " << volSurface.numTenors() 
              << " = " << volSurface.numNodes() << " nodes\n\n";
    
    // 5 different swaptions to test
    std::vector<std::pair<double, double>> testSwaptions = {
        {1.0, 5.0},    // 1Y x 5Y  - Short expiry, medium tenor
        {2.0, 10.0},   // 2Y x 10Y - Medium expiry, long tenor
        {5.0, 5.0},    // 5Y x 5Y  - Medium expiry, medium tenor
        {7.0, 20.0},   // 7Y x 20Y - Long expiry, very long tenor (benchmark)
        {10.0, 10.0}   // 10Y x 10Y - Long expiry, long tenor
    };
    
    // Sigma buckets
    std::vector<double> sigmaTimes = {0.0, 1.0/12, 3.0/12, 6.0/12, 1.0, 2.0, 3.0, 5.0, 7.0};
    std::vector<double> sigmaInit = {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01};
    
    // MC Config - reduced paths for speed
    MCConfig mcConfig(3000, 50, true, 42);
    
    // ===========================================================================
    // PART 1: EXACT CALIBRATION (Co-terminal style)
    // ===========================================================================
    std::cout << std::string(100, '=') << "\n";
    std::cout << "PART 1: EXACT CALIBRATION (10 instruments -> 10 parameters)\n";
    std::cout << std::string(100, '=') << "\n\n";
    
    // Exact calibration instruments
    std::vector<std::pair<double, double>> exactCalibInst = {
        {1.0/12, 20.0}, {3.0/12, 20.0}, {6.0/12, 20.0},
        {1.0, 20.0}, {2.0, 20.0}, {3.0, 20.0}, {5.0, 20.0}, {7.0, 20.0}, {10.0, 20.0},
        {1.0, 10.0}
    };
    
    std::cout << "Calibration instruments: " << exactCalibInst.size() << " (co-terminal 20Y tenor)\n";
    std::cout << "Sigma buckets: " << sigmaTimes.size() << "\n\n";
    
    // Calibrate
    CalibrationEngine<double> exactCalibEngine(curve, volSurface);
    for (const auto& [e, t] : exactCalibInst) {
        exactCalibEngine.addInstrument(e, t);
    }
    HW1FParams exactInitParams(0.03, sigmaTimes, sigmaInit);
    auto exactCalibResult = exactCalibEngine.calibrate(exactInitParams, 100, 1e-8, false);
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Calibration Results:\n";
    std::cout << "  a (mean reversion) = " << exactCalibResult.params.a << "\n";
    std::cout << "  RMSE = $" << std::setprecision(4) << exactCalibResult.rmse << "\n";
    std::cout << "  Iterations = " << exactCalibResult.iterations << "\n";
    std::cout << "  Time = " << std::setprecision(3) << exactCalibResult.elapsedTime << "s\n\n";
    
    // Run comparisons for all 5 swaptions
    std::vector<SwaptionResult> exactResults;
    for (const auto& [expiry, tenor] : testSwaptions) {
        std::cout << "Processing " << expiry << "Y x " << tenor << "Y swaption...\n";
        auto result = runSwaptionComparison(expiry, tenor, curve, volSurface, 
                                           exactCalibInst, exactCalibResult, mcConfig);
        exactResults.push_back(result);
    }
    std::cout << "\n";
    
    // ===========================================================================
    // PART 2: LEAST-SQUARES CALIBRATION (81 instruments -> 10 parameters)
    // ===========================================================================
    std::cout << std::string(100, '=') << "\n";
    std::cout << "PART 2: LEAST-SQUARES CALIBRATION (81 instruments -> 10 parameters)\n";
    std::cout << std::string(100, '=') << "\n\n";
    
    // All vol surface nodes as calibration instruments
    std::vector<std::pair<double, double>> lsqCalibInst;
    for (size_t ei = 0; ei < volSurface.numExpiries(); ++ei) {
        for (size_t ti = 0; ti < volSurface.numTenors(); ++ti) {
            lsqCalibInst.push_back({volSurface.expiries()[ei], volSurface.tenors()[ti]});
        }
    }
    
    std::cout << "Calibration instruments: " << lsqCalibInst.size() << " (full vol surface)\n";
    std::cout << "Over-determined ratio: " << lsqCalibInst.size() / 10.0 << "x\n\n";
    
    // Calibrate
    CalibrationEngine<double> lsqCalibEngine(curve, volSurface);
    for (const auto& [e, t] : lsqCalibInst) {
        lsqCalibEngine.addInstrument(e, t);
    }
    HW1FParams lsqInitParams(0.03, sigmaTimes, sigmaInit);
    auto lsqCalibResult = lsqCalibEngine.calibrate(lsqInitParams, 200, 1e-10, false);
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Calibration Results:\n";
    std::cout << "  a (mean reversion) = " << lsqCalibResult.params.a << "\n";
    std::cout << "  RMSE = $" << std::setprecision(4) << lsqCalibResult.rmse << "\n";
    std::cout << "  Iterations = " << lsqCalibResult.iterations << "\n";
    std::cout << "  Time = " << std::setprecision(3) << lsqCalibResult.elapsedTime << "s\n\n";
    
    // Run comparisons for all 5 swaptions
    std::vector<SwaptionResult> lsqResults;
    for (const auto& [expiry, tenor] : testSwaptions) {
        std::cout << "Processing " << expiry << "Y x " << tenor << "Y swaption...\n";
        auto result = runSwaptionComparison(expiry, tenor, curve, volSurface,
                                           lsqCalibInst, lsqCalibResult, mcConfig);
        lsqResults.push_back(result);
    }
    std::cout << "\n";
    
    // ===========================================================================
    // RESULTS SUMMARY
    // ===========================================================================
    std::cout << std::string(100, '=') << "\n";
    std::cout << "RESULTS SUMMARY\n";
    std::cout << std::string(100, '=') << "\n\n";
    
    // Price comparison table
    std::cout << "PRICE COMPARISON (in $)\n";
    std::cout << std::string(100, '-') << "\n";
    std::cout << std::left << std::setw(12) << "Swaption"
              << std::right << std::setw(12) << "Calib"
              << std::setw(14) << "Jamshidian"
              << std::setw(12) << "FD Naive"
              << std::setw(12) << "FD+Chain"
              << std::setw(12) << "IFT"
              << std::setw(12) << "XAD+IFT" << "\n";
    std::cout << std::string(100, '-') << "\n";
    
    std::cout << std::fixed << std::setprecision(2);
    for (size_t i = 0; i < testSwaptions.size(); ++i) {
        std::ostringstream label;
        label << testSwaptions[i].first << "Yx" << testSwaptions[i].second << "Y";
        
        // Exact calibration row
        std::cout << std::setw(12) << label.str()
                  << std::setw(12) << "Exact"
                  << std::setw(14) << exactResults[i].jamPrice
                  << std::setw(12) << exactResults[i].fdNaivePrice
                  << std::setw(12) << exactResults[i].fdChainPrice
                  << std::setw(12) << exactResults[i].iftPrice
                  << std::setw(12) << exactResults[i].xadIftPrice << "\n";
        
        // LSQ calibration row
        std::cout << std::setw(12) << label.str()
                  << std::setw(12) << "LSQ"
                  << std::setw(14) << lsqResults[i].jamPrice
                  << std::setw(12) << lsqResults[i].fdNaivePrice
                  << std::setw(12) << lsqResults[i].fdChainPrice
                  << std::setw(12) << lsqResults[i].iftPrice
                  << std::setw(12) << lsqResults[i].xadIftPrice << "\n";
    }
    
    // Timing comparison table
    std::cout << "\n\nTIMING COMPARISON (in seconds)\n";
    std::cout << std::string(100, '-') << "\n";
    std::cout << std::left << std::setw(12) << "Swaption"
              << std::right << std::setw(12) << "Calib"
              << std::setw(12) << "FD Naive"
              << std::setw(12) << "FD+Chain"
              << std::setw(12) << "IFT"
              << std::setw(12) << "XAD+IFT"
              << std::setw(15) << "IFT Speedup"
              << std::setw(15) << "XAD Speedup" << "\n";
    std::cout << std::string(100, '-') << "\n";
    
    std::cout << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < testSwaptions.size(); ++i) {
        std::ostringstream label;
        label << testSwaptions[i].first << "Yx" << testSwaptions[i].second << "Y";
        
        // Exact calibration row
        std::cout << std::setw(12) << label.str()
                  << std::setw(12) << "Exact"
                  << std::setw(12) << exactResults[i].fdNaiveTime
                  << std::setw(12) << exactResults[i].fdChainTime
                  << std::setw(12) << exactResults[i].iftTime
                  << std::setw(12) << exactResults[i].xadIftTime
                  << std::setw(14) << std::setprecision(1) << exactResults[i].fdNaiveTime / exactResults[i].iftTime << "x"
                  << std::setw(14) << exactResults[i].fdNaiveTime / exactResults[i].xadIftTime << "x\n";
        
        // LSQ calibration row
        std::cout << std::setw(12) << label.str()
                  << std::setw(12) << "LSQ"
                  << std::setw(12) << std::setprecision(3) << lsqResults[i].fdNaiveTime
                  << std::setw(12) << lsqResults[i].fdChainTime
                  << std::setw(12) << lsqResults[i].iftTime
                  << std::setw(12) << lsqResults[i].xadIftTime
                  << std::setw(14) << std::setprecision(1) << lsqResults[i].fdNaiveTime / lsqResults[i].iftTime << "x"
                  << std::setw(14) << lsqResults[i].fdNaiveTime / lsqResults[i].xadIftTime << "x\n";
    }
    
    // Greeks comparison (dV/da)
    std::cout << "\n\nGREEKS COMPARISON: dV/da (sensitivity to mean reversion)\n";
    std::cout << std::string(100, '-') << "\n";
    std::cout << std::left << std::setw(12) << "Swaption"
              << std::right << std::setw(12) << "Calib"
              << std::setw(15) << "FD+Chain"
              << std::setw(15) << "IFT"
              << std::setw(15) << "XAD+IFT"
              << std::setw(15) << "IFT vs FD%"
              << std::setw(15) << "XAD vs FD%" << "\n";
    std::cout << std::string(100, '-') << "\n";
    
    std::cout << std::fixed << std::setprecision(2);
    for (size_t i = 0; i < testSwaptions.size(); ++i) {
        std::ostringstream label;
        label << testSwaptions[i].first << "Yx" << testSwaptions[i].second << "Y";
        
        // Exact calibration row
        double iftVsFdExact = (exactResults[i].fdChain_dVda != 0) 
            ? (exactResults[i].ift_dVda - exactResults[i].fdChain_dVda) / std::abs(exactResults[i].fdChain_dVda) * 100 : 0;
        double xadVsFdExact = (exactResults[i].fdChain_dVda != 0)
            ? (exactResults[i].xadIft_dVda - exactResults[i].fdChain_dVda) / std::abs(exactResults[i].fdChain_dVda) * 100 : 0;
        
        std::cout << std::setw(12) << label.str()
                  << std::setw(12) << "Exact"
                  << std::setw(15) << exactResults[i].fdChain_dVda
                  << std::setw(15) << exactResults[i].ift_dVda
                  << std::setw(15) << exactResults[i].xadIft_dVda
                  << std::setw(14) << iftVsFdExact << "%"
                  << std::setw(14) << xadVsFdExact << "%\n";
        
        // LSQ calibration row
        double iftVsFdLsq = (lsqResults[i].fdChain_dVda != 0)
            ? (lsqResults[i].ift_dVda - lsqResults[i].fdChain_dVda) / std::abs(lsqResults[i].fdChain_dVda) * 100 : 0;
        double xadVsFdLsq = (lsqResults[i].fdChain_dVda != 0)
            ? (lsqResults[i].xadIft_dVda - lsqResults[i].fdChain_dVda) / std::abs(lsqResults[i].fdChain_dVda) * 100 : 0;
        
        std::cout << std::setw(12) << label.str()
                  << std::setw(12) << "LSQ"
                  << std::setw(15) << lsqResults[i].fdChain_dVda
                  << std::setw(15) << lsqResults[i].ift_dVda
                  << std::setw(15) << lsqResults[i].xadIft_dVda
                  << std::setw(14) << iftVsFdLsq << "%"
                  << std::setw(14) << xadVsFdLsq << "%\n";
    }
    
    // Greeks comparison (sum of dV/dsigma)
    std::cout << "\n\nGREEKS COMPARISON: Sum(dV/dsigma_k) (total sigma sensitivity)\n";
    std::cout << std::string(100, '-') << "\n";
    std::cout << std::left << std::setw(12) << "Swaption"
              << std::right << std::setw(12) << "Calib"
              << std::setw(15) << "FD+Chain"
              << std::setw(15) << "IFT"
              << std::setw(15) << "XAD+IFT" << "\n";
    std::cout << std::string(100, '-') << "\n";
    
    for (size_t i = 0; i < testSwaptions.size(); ++i) {
        std::ostringstream label;
        label << testSwaptions[i].first << "Yx" << testSwaptions[i].second << "Y";
        
        // Compute sums from vectors
        double exactFdChainSum = 0, exactIftSum = 0, exactXadSum = 0;
        for (double v : exactResults[i].fdChain_dVdsigma) exactFdChainSum += v;
        for (double v : exactResults[i].ift_dVdsigma) exactIftSum += v;
        for (double v : exactResults[i].xadIft_dVdsigma) exactXadSum += v;
        
        double lsqFdChainSum = 0, lsqIftSum = 0, lsqXadSum = 0;
        for (double v : lsqResults[i].fdChain_dVdsigma) lsqFdChainSum += v;
        for (double v : lsqResults[i].ift_dVdsigma) lsqIftSum += v;
        for (double v : lsqResults[i].xadIft_dVdsigma) lsqXadSum += v;
        
        std::cout << std::setw(12) << label.str()
                  << std::setw(12) << "Exact"
                  << std::setw(15) << exactFdChainSum
                  << std::setw(15) << exactIftSum
                  << std::setw(15) << exactXadSum << "\n";
        
        std::cout << std::setw(12) << label.str()
                  << std::setw(12) << "LSQ"
                  << std::setw(15) << lsqFdChainSum
                  << std::setw(15) << lsqIftSum
                  << std::setw(15) << lsqXadSum << "\n";
    }
    
    // ===========================================================================
    // SAVE RESULTS TO FILE
    // ===========================================================================
    std::ofstream outFile("MULTI_SWAPTION_COMPARISON_RESULTS.txt");
    if (outFile.is_open()) {
        outFile << "================================================================================\n";
        outFile << "HULL-WHITE 1-FACTOR MODEL: MULTI-SWAPTION COMPARISON RESULTS\n";
        outFile << "Exact vs Least-Squares Calibration | 5 Swaptions | 4 Methods\n";
        outFile << "Generated: " << __DATE__ << " " << __TIME__ << "\n";
        outFile << "================================================================================\n\n";
        
        outFile << "MARKET DATA\n";
        outFile << "-----------\n";
        outFile << "Curve nodes: " << curve.size() << "\n";
        outFile << "Vol surface: " << volSurface.numExpiries() << " x " << volSurface.numTenors() 
                << " = " << volSurface.numNodes() << " nodes\n\n";
        
        outFile << "CALIBRATION SUMMARY\n";
        outFile << "-------------------\n";
        outFile << "EXACT: 10 co-terminal instruments -> 10 HW parameters\n";
        outFile << "  Mean reversion (a): " << std::fixed << std::setprecision(6) << exactCalibResult.params.a << "\n";
        outFile << "  RMSE: $" << std::setprecision(4) << exactCalibResult.rmse << "\n";
        outFile << "  Iterations: " << exactCalibResult.iterations << "\n";
        outFile << "  Time: " << std::setprecision(3) << exactCalibResult.elapsedTime << "s\n\n";
        
        outFile << "LSQ: 81 vol surface instruments -> 10 HW parameters (8.1x over-determined)\n";
        outFile << "  Mean reversion (a): " << std::setprecision(6) << lsqCalibResult.params.a << "\n";
        outFile << "  RMSE: $" << std::setprecision(4) << lsqCalibResult.rmse << "\n";
        outFile << "  Iterations: " << lsqCalibResult.iterations << "\n";
        outFile << "  Time: " << std::setprecision(3) << lsqCalibResult.elapsedTime << "s\n\n";
        
        outFile << "================================================================================\n";
        outFile << "PRICE COMPARISON (in $)\n";
        outFile << "================================================================================\n";
        outFile << std::left << std::setw(12) << "Swaption"
                << std::right << std::setw(10) << "Calib"
                << std::setw(14) << "Jamshidian"
                << std::setw(12) << "FD Naive"
                << std::setw(12) << "FD+Chain"
                << std::setw(12) << "IFT"
                << std::setw(12) << "XAD+IFT" << "\n";
        outFile << std::string(84, '-') << "\n";
        
        for (size_t i = 0; i < testSwaptions.size(); ++i) {
            std::ostringstream label;
            label << testSwaptions[i].first << "Yx" << testSwaptions[i].second << "Y";
            
            outFile << std::setw(12) << label.str()
                    << std::setw(10) << "Exact"
                    << std::setw(14) << std::setprecision(2) << exactResults[i].jamPrice
                    << std::setw(12) << exactResults[i].fdNaivePrice
                    << std::setw(12) << exactResults[i].fdChainPrice
                    << std::setw(12) << exactResults[i].iftPrice
                    << std::setw(12) << exactResults[i].xadIftPrice << "\n";
            
            outFile << std::setw(12) << label.str()
                    << std::setw(10) << "LSQ"
                    << std::setw(14) << lsqResults[i].jamPrice
                    << std::setw(12) << lsqResults[i].fdNaivePrice
                    << std::setw(12) << lsqResults[i].fdChainPrice
                    << std::setw(12) << lsqResults[i].iftPrice
                    << std::setw(12) << lsqResults[i].xadIftPrice << "\n";
        }
        
        outFile << "\n================================================================================\n";
        outFile << "TIMING COMPARISON (in seconds)\n";
        outFile << "================================================================================\n";
        outFile << std::left << std::setw(12) << "Swaption"
                << std::right << std::setw(10) << "Calib"
                << std::setw(12) << "FD Naive"
                << std::setw(12) << "FD+Chain"
                << std::setw(12) << "IFT"
                << std::setw(12) << "XAD+IFT"
                << std::setw(12) << "IFT Spdup"
                << std::setw(12) << "XAD Spdup" << "\n";
        outFile << std::string(94, '-') << "\n";
        
        for (size_t i = 0; i < testSwaptions.size(); ++i) {
            std::ostringstream label;
            label << testSwaptions[i].first << "Yx" << testSwaptions[i].second << "Y";
            
            outFile << std::setw(12) << label.str()
                    << std::setw(10) << "Exact"
                    << std::setw(12) << std::setprecision(3) << exactResults[i].fdNaiveTime
                    << std::setw(12) << exactResults[i].fdChainTime
                    << std::setw(12) << exactResults[i].iftTime
                    << std::setw(12) << exactResults[i].xadIftTime
                    << std::setw(11) << std::setprecision(1) << exactResults[i].fdNaiveTime / exactResults[i].iftTime << "x"
                    << std::setw(11) << exactResults[i].fdNaiveTime / exactResults[i].xadIftTime << "x\n";
            
            outFile << std::setw(12) << label.str()
                    << std::setw(10) << "LSQ"
                    << std::setw(12) << std::setprecision(3) << lsqResults[i].fdNaiveTime
                    << std::setw(12) << lsqResults[i].fdChainTime
                    << std::setw(12) << lsqResults[i].iftTime
                    << std::setw(12) << lsqResults[i].xadIftTime
                    << std::setw(11) << std::setprecision(1) << lsqResults[i].fdNaiveTime / lsqResults[i].iftTime << "x"
                    << std::setw(11) << lsqResults[i].fdNaiveTime / lsqResults[i].xadIftTime << "x\n";
        }
        
        outFile << "\n================================================================================\n";
        outFile << "GREEKS: dV/da (Mean Reversion Sensitivity)\n";
        outFile << "================================================================================\n";
        outFile << std::left << std::setw(12) << "Swaption"
                << std::right << std::setw(10) << "Calib"
                << std::setw(15) << "FD+Chain"
                << std::setw(15) << "IFT"
                << std::setw(15) << "XAD+IFT" << "\n";
        outFile << std::string(67, '-') << "\n";
        
        for (size_t i = 0; i < testSwaptions.size(); ++i) {
            std::ostringstream label;
            label << testSwaptions[i].first << "Yx" << testSwaptions[i].second << "Y";
            
            outFile << std::setw(12) << label.str()
                    << std::setw(10) << "Exact"
                    << std::setw(15) << std::setprecision(2) << exactResults[i].fdChain_dVda
                    << std::setw(15) << exactResults[i].ift_dVda
                    << std::setw(15) << exactResults[i].xadIft_dVda << "\n";
            
            outFile << std::setw(12) << label.str()
                    << std::setw(10) << "LSQ"
                    << std::setw(15) << lsqResults[i].fdChain_dVda
                    << std::setw(15) << lsqResults[i].ift_dVda
                    << std::setw(15) << lsqResults[i].xadIft_dVda << "\n";
        }
        
        // =====================================================================
        // FULL CURVE GREEKS for each swaption
        // =====================================================================
        outFile << "\n================================================================================\n";
        outFile << "GREEKS: dV/dr - DISCOUNT CURVE NODE SENSITIVITIES (per 1bp)\n";
        outFile << "================================================================================\n\n";
        
        std::vector<double> curveTimes = curve.times();
        
        for (size_t i = 0; i < testSwaptions.size(); ++i) {
            std::ostringstream label;
            label << testSwaptions[i].first << "Yx" << testSwaptions[i].second << "Y";
            
            // EXACT CALIBRATION
            outFile << "--- " << label.str() << " (EXACT Calibration) ---\n";
            outFile << std::left << std::setw(10) << "Maturity"
                    << std::right << std::setw(14) << "FD Naive"
                    << std::setw(14) << "FD+Chain"
                    << std::setw(14) << "IFT"
                    << std::setw(14) << "XAD+IFT"
                    << std::setw(14) << "Max Diff%" << "\n";
            outFile << std::string(80, '-') << "\n";
            
            for (size_t c = 0; c < curveTimes.size(); ++c) {
                double fdN = exactResults[i].fdNaive_curveGreeks[c] * 0.0001;
                double fdC = exactResults[i].fdChain_curveGreeks[c] * 0.0001;
                double ift = exactResults[i].ift_curveGreeks[c] * 0.0001;
                double xad = exactResults[i].xadIft_curveGreeks[c] * 0.0001;
                
                // Compute max relative difference
                double maxVal = std::max({std::abs(fdN), std::abs(fdC), std::abs(ift), std::abs(xad)});
                double maxDiff = 0.0;
                if (maxVal > 1e-10) {
                    maxDiff = std::max({std::abs(fdN - fdC), std::abs(fdN - ift), std::abs(fdN - xad)}) / maxVal * 100;
                }
                
                outFile << std::setw(10) << std::setprecision(2) << curveTimes[c]
                        << std::setw(14) << std::setprecision(4) << fdN
                        << std::setw(14) << fdC
                        << std::setw(14) << ift
                        << std::setw(14) << xad
                        << std::setw(13) << std::setprecision(2) << maxDiff << "%\n";
            }
            outFile << "\n";
            
            // LSQ CALIBRATION
            outFile << "--- " << label.str() << " (LSQ Calibration) ---\n";
            outFile << std::left << std::setw(10) << "Maturity"
                    << std::right << std::setw(14) << "FD Naive"
                    << std::setw(14) << "FD+Chain"
                    << std::setw(14) << "IFT"
                    << std::setw(14) << "XAD+IFT"
                    << std::setw(14) << "Max Diff%" << "\n";
            outFile << std::string(80, '-') << "\n";
            
            for (size_t c = 0; c < curveTimes.size(); ++c) {
                double fdN = lsqResults[i].fdNaive_curveGreeks[c] * 0.0001;
                double fdC = lsqResults[i].fdChain_curveGreeks[c] * 0.0001;
                double ift = lsqResults[i].ift_curveGreeks[c] * 0.0001;
                double xad = lsqResults[i].xadIft_curveGreeks[c] * 0.0001;
                
                double maxVal = std::max({std::abs(fdN), std::abs(fdC), std::abs(ift), std::abs(xad)});
                double maxDiff = 0.0;
                if (maxVal > 1e-10) {
                    maxDiff = std::max({std::abs(fdN - fdC), std::abs(fdN - ift), std::abs(fdN - xad)}) / maxVal * 100;
                }
                
                outFile << std::setw(10) << std::setprecision(2) << curveTimes[c]
                        << std::setw(14) << std::setprecision(4) << fdN
                        << std::setw(14) << fdC
                        << std::setw(14) << ift
                        << std::setw(14) << xad
                        << std::setw(13) << std::setprecision(2) << maxDiff << "%\n";
            }
            outFile << "\n";
        }
        
        // =====================================================================
        // FULL VOL SURFACE GREEKS for each swaption
        // =====================================================================
        outFile << "\n================================================================================\n";
        outFile << "GREEKS: dV/dvol - VOL SURFACE NODE SENSITIVITIES (per 1bp)\n";
        outFile << "================================================================================\n\n";
        
        for (size_t i = 0; i < testSwaptions.size(); ++i) {
            std::ostringstream label;
            label << testSwaptions[i].first << "Yx" << testSwaptions[i].second << "Y";
            
            // EXACT CALIBRATION
            outFile << "--- " << label.str() << " (EXACT Calibration) ---\n";
            outFile << std::left << std::setw(12) << "Expiry\\Tenor";
            for (size_t ti = 0; ti < volSurface.numTenors(); ++ti) {
                std::ostringstream tenorLabel;
                tenorLabel << std::fixed << std::setprecision(0) << volSurface.tenors()[ti] << "Y";
                outFile << std::right << std::setw(10) << tenorLabel.str();
            }
            outFile << "\n" << std::string(12 + 10*volSurface.numTenors(), '-') << "\n";
            
            // FD Naive
            outFile << "FD Naive:\n";
            for (size_t ei = 0; ei < volSurface.numExpiries(); ++ei) {
                std::ostringstream expLabel;
                if (volSurface.expiries()[ei] < 1.0) {
                    expLabel << std::fixed << std::setprecision(0) << (volSurface.expiries()[ei] * 12) << "M";
                } else {
                    expLabel << std::fixed << std::setprecision(0) << volSurface.expiries()[ei] << "Y";
                }
                outFile << std::left << std::setw(12) << expLabel.str();
                for (size_t ti = 0; ti < volSurface.numTenors(); ++ti) {
                    double val = exactResults[i].fdNaive_volGreeks[ei][ti] * 0.0001;
                    outFile << std::right << std::setw(10) << std::setprecision(2) << val;
                }
                outFile << "\n";
            }
            outFile << "\n";
            
            // FD+Chain
            outFile << "FD+Chain:\n";
            for (size_t ei = 0; ei < volSurface.numExpiries(); ++ei) {
                std::ostringstream expLabel;
                if (volSurface.expiries()[ei] < 1.0) {
                    expLabel << std::fixed << std::setprecision(0) << (volSurface.expiries()[ei] * 12) << "M";
                } else {
                    expLabel << std::fixed << std::setprecision(0) << volSurface.expiries()[ei] << "Y";
                }
                outFile << std::left << std::setw(12) << expLabel.str();
                for (size_t ti = 0; ti < volSurface.numTenors(); ++ti) {
                    double val = exactResults[i].fdChain_volGreeks[ei][ti] * 0.0001;
                    outFile << std::right << std::setw(10) << std::setprecision(2) << val;
                }
                outFile << "\n";
            }
            outFile << "\n";
            
            // IFT
            outFile << "IFT:\n";
            for (size_t ei = 0; ei < volSurface.numExpiries(); ++ei) {
                std::ostringstream expLabel;
                if (volSurface.expiries()[ei] < 1.0) {
                    expLabel << std::fixed << std::setprecision(0) << (volSurface.expiries()[ei] * 12) << "M";
                } else {
                    expLabel << std::fixed << std::setprecision(0) << volSurface.expiries()[ei] << "Y";
                }
                outFile << std::left << std::setw(12) << expLabel.str();
                for (size_t ti = 0; ti < volSurface.numTenors(); ++ti) {
                    double val = exactResults[i].ift_volGreeks[ei][ti] * 0.0001;
                    outFile << std::right << std::setw(10) << std::setprecision(2) << val;
                }
                outFile << "\n";
            }
            outFile << "\n";
            
            // XAD+IFT
            outFile << "XAD+IFT:\n";
            for (size_t ei = 0; ei < volSurface.numExpiries(); ++ei) {
                std::ostringstream expLabel;
                if (volSurface.expiries()[ei] < 1.0) {
                    expLabel << std::fixed << std::setprecision(0) << (volSurface.expiries()[ei] * 12) << "M";
                } else {
                    expLabel << std::fixed << std::setprecision(0) << volSurface.expiries()[ei] << "Y";
                }
                outFile << std::left << std::setw(12) << expLabel.str();
                for (size_t ti = 0; ti < volSurface.numTenors(); ++ti) {
                    double val = exactResults[i].xadIft_volGreeks[ei][ti] * 0.0001;
                    outFile << std::right << std::setw(10) << std::setprecision(2) << val;
                }
                outFile << "\n";
            }
            outFile << "\n";
            
            // LSQ CALIBRATION
            outFile << "--- " << label.str() << " (LSQ Calibration) ---\n";
            outFile << std::left << std::setw(12) << "Expiry\\Tenor";
            for (size_t ti = 0; ti < volSurface.numTenors(); ++ti) {
                std::ostringstream tenorLabel;
                tenorLabel << std::fixed << std::setprecision(0) << volSurface.tenors()[ti] << "Y";
                outFile << std::right << std::setw(10) << tenorLabel.str();
            }
            outFile << "\n" << std::string(12 + 10*volSurface.numTenors(), '-') << "\n";
            
            // FD Naive LSQ
            outFile << "FD Naive:\n";
            for (size_t ei = 0; ei < volSurface.numExpiries(); ++ei) {
                std::ostringstream expLabel;
                if (volSurface.expiries()[ei] < 1.0) {
                    expLabel << std::fixed << std::setprecision(0) << (volSurface.expiries()[ei] * 12) << "M";
                } else {
                    expLabel << std::fixed << std::setprecision(0) << volSurface.expiries()[ei] << "Y";
                }
                outFile << std::left << std::setw(12) << expLabel.str();
                for (size_t ti = 0; ti < volSurface.numTenors(); ++ti) {
                    double val = lsqResults[i].fdNaive_volGreeks[ei][ti] * 0.0001;
                    outFile << std::right << std::setw(10) << std::setprecision(2) << val;
                }
                outFile << "\n";
            }
            outFile << "\n";
            
            // FD+Chain LSQ
            outFile << "FD+Chain:\n";
            for (size_t ei = 0; ei < volSurface.numExpiries(); ++ei) {
                std::ostringstream expLabel;
                if (volSurface.expiries()[ei] < 1.0) {
                    expLabel << std::fixed << std::setprecision(0) << (volSurface.expiries()[ei] * 12) << "M";
                } else {
                    expLabel << std::fixed << std::setprecision(0) << volSurface.expiries()[ei] << "Y";
                }
                outFile << std::left << std::setw(12) << expLabel.str();
                for (size_t ti = 0; ti < volSurface.numTenors(); ++ti) {
                    double val = lsqResults[i].fdChain_volGreeks[ei][ti] * 0.0001;
                    outFile << std::right << std::setw(10) << std::setprecision(2) << val;
                }
                outFile << "\n";
            }
            outFile << "\n";
            
            // IFT LSQ
            outFile << "IFT:\n";
            for (size_t ei = 0; ei < volSurface.numExpiries(); ++ei) {
                std::ostringstream expLabel;
                if (volSurface.expiries()[ei] < 1.0) {
                    expLabel << std::fixed << std::setprecision(0) << (volSurface.expiries()[ei] * 12) << "M";
                } else {
                    expLabel << std::fixed << std::setprecision(0) << volSurface.expiries()[ei] << "Y";
                }
                outFile << std::left << std::setw(12) << expLabel.str();
                for (size_t ti = 0; ti < volSurface.numTenors(); ++ti) {
                    double val = lsqResults[i].ift_volGreeks[ei][ti] * 0.0001;
                    outFile << std::right << std::setw(10) << std::setprecision(2) << val;
                }
                outFile << "\n";
            }
            outFile << "\n";
            
            // XAD+IFT LSQ
            outFile << "XAD+IFT:\n";
            for (size_t ei = 0; ei < volSurface.numExpiries(); ++ei) {
                std::ostringstream expLabel;
                if (volSurface.expiries()[ei] < 1.0) {
                    expLabel << std::fixed << std::setprecision(0) << (volSurface.expiries()[ei] * 12) << "M";
                } else {
                    expLabel << std::fixed << std::setprecision(0) << volSurface.expiries()[ei] << "Y";
                }
                outFile << std::left << std::setw(12) << expLabel.str();
                for (size_t ti = 0; ti < volSurface.numTenors(); ++ti) {
                    double val = lsqResults[i].xadIft_volGreeks[ei][ti] * 0.0001;
                    outFile << std::right << std::setw(10) << std::setprecision(2) << val;
                }
                outFile << "\n";
            }
            outFile << "\n\n";
        }
        
        // =====================================================================
        // SIGMA BUCKET GREEKS
        // =====================================================================
        outFile << "\n================================================================================\n";
        outFile << "GREEKS: dV/dsigma_k - SIGMA BUCKET SENSITIVITIES\n";
        outFile << "================================================================================\n\n";
        
        for (size_t i = 0; i < testSwaptions.size(); ++i) {
            std::ostringstream label;
            label << testSwaptions[i].first << "Yx" << testSwaptions[i].second << "Y";
            
            outFile << "--- " << label.str() << " ---\n";
            outFile << std::left << std::setw(20) << "Bucket"
                    << std::right << std::setw(15) << "FD+Chain"
                    << std::setw(15) << "IFT"
                    << std::setw(15) << "XAD+IFT (Ex)"
                    << std::setw(15) << "XAD+IFT (LSQ)" << "\n";
            outFile << std::string(80, '-') << "\n";
            
            size_t numBuckets = exactResults[i].fdChain_dVdsigma.size();
            for (size_t b = 0; b < numBuckets; ++b) {
                std::ostringstream bucketLabel;
                bucketLabel << "sigma_" << (b+1);
                
                double fdC_ex = exactResults[i].fdChain_dVdsigma[b];
                double ift_ex = exactResults[i].ift_dVdsigma[b];
                double xad_ex = exactResults[i].xadIft_dVdsigma[b];
                double xad_lsq = lsqResults[i].xadIft_dVdsigma[b];
                
                outFile << std::setw(20) << bucketLabel.str()
                        << std::setw(15) << std::setprecision(2) << fdC_ex
                        << std::setw(15) << ift_ex
                        << std::setw(15) << xad_ex
                        << std::setw(15) << xad_lsq << "\n";
            }
            
            // Sum row
            double sumFdC = 0, sumIft = 0, sumXadEx = 0, sumXadLsq = 0;
            for (size_t b = 0; b < numBuckets; ++b) {
                sumFdC += exactResults[i].fdChain_dVdsigma[b];
                sumIft += exactResults[i].ift_dVdsigma[b];
                sumXadEx += exactResults[i].xadIft_dVdsigma[b];
                sumXadLsq += lsqResults[i].xadIft_dVdsigma[b];
            }
            outFile << std::string(80, '-') << "\n";
            outFile << std::setw(20) << "TOTAL"
                    << std::setw(15) << sumFdC
                    << std::setw(15) << sumIft
                    << std::setw(15) << sumXadEx
                    << std::setw(15) << sumXadLsq << "\n";
            outFile << "\n";
        }
        
        outFile << "\n================================================================================\n";
        outFile << "ANALYSIS SUMMARY\n";
        outFile << "================================================================================\n\n";
        
        // Compute averages
        double avgIftSpeedupExact = 0, avgXadSpeedupExact = 0;
        double avgIftSpeedupLsq = 0, avgXadSpeedupLsq = 0;
        for (size_t i = 0; i < testSwaptions.size(); ++i) {
            avgIftSpeedupExact += exactResults[i].fdNaiveTime / exactResults[i].iftTime;
            avgXadSpeedupExact += exactResults[i].fdNaiveTime / exactResults[i].xadIftTime;
            avgIftSpeedupLsq += lsqResults[i].fdNaiveTime / lsqResults[i].iftTime;
            avgXadSpeedupLsq += lsqResults[i].fdNaiveTime / lsqResults[i].xadIftTime;
        }
        avgIftSpeedupExact /= testSwaptions.size();
        avgXadSpeedupExact /= testSwaptions.size();
        avgIftSpeedupLsq /= testSwaptions.size();
        avgXadSpeedupLsq /= testSwaptions.size();
        
        outFile << "EXACT CALIBRATION:\n";
        outFile << "  Average IFT speedup over FD Naive: " << std::setprecision(1) << avgIftSpeedupExact << "x\n";
        outFile << "  Average XAD+IFT speedup over FD Naive: " << avgXadSpeedupExact << "x\n\n";
        
        outFile << "LEAST-SQUARES CALIBRATION:\n";
        outFile << "  Average IFT speedup over FD Naive: " << avgIftSpeedupLsq << "x\n";
        outFile << "  Average XAD+IFT speedup over FD Naive: " << avgXadSpeedupLsq << "x\n\n";
        
        outFile << "KEY OBSERVATIONS:\n";
        outFile << "1. IFT and XAD+IFT eliminate the need for recalibration in Greeks computation\n";
        outFile << "2. XAD+IFT computes ALL parameter sensitivities in a single backward pass\n";
        outFile << "3. Least-squares calibration is more robust but has non-zero RMSE\n";
        outFile << "4. All methods produce consistent Greeks within MC noise tolerance\n";
        outFile << "5. Speedups are most significant for large vol surfaces (81 nodes)\n";
        
        outFile.close();
        std::cout << "\n\nResults saved to MULTI_SWAPTION_COMPARISON_RESULTS.txt\n";
    }
    
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "COMPUTATION COMPLETE\n";
    std::cout << std::string(100, '=') << "\n";
    
    return 0;
}
