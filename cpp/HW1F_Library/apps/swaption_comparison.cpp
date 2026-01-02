// =============================================================================
// HW1F Library - Swaption Comparison Application
// Hull-White 1-Factor Model with Piecewise Constant Volatility
// 
// For 5 different swaptions, compares:
// 1. PRICING: Analytical (Jamshidian) vs Monte Carlo vs MC+XAD (time + values)
// 2. GREEKS:  Analytical vs Finite Difference vs XAD+IFT (time + values)
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
// Market Data Setup
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
// Result Structures
// =============================================================================
struct SwaptionPricingResult {
    std::string name;
    double analyticalPrice;
    double mcPrice;
    double mcXadPrice;
    double analyticalTime;
    double mcTime;
    double mcXadTime;
};

struct SwaptionGreeksResult {
    std::string name;
    // Timings
    double analyticalTime;
    double fdTime;
    double xadIftTime;
    // dV/da
    double analytical_dVda;
    double fd_dVda;
    double xadIft_dVda;
    // Sum of dV/dsigma
    double analytical_sumDVdsigma;
    double fd_sumDVdsigma;
    double xadIft_sumDVdsigma;
    // Per-bucket dV/dsigma
    std::vector<double> analytical_dVdsigma;
    std::vector<double> fd_dVdsigma;
    std::vector<double> xadIft_dVdsigma;
    // Sum of vol surface Greeks
    double analytical_sumVolGreeks;
    double fd_sumVolGreeks;
    double xadIft_sumVolGreeks;
    // Sum of curve Greeks
    double analytical_sumCurveGreeks;
    double fd_sumCurveGreeks;
    double xadIft_sumCurveGreeks;
};

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << std::fixed;
    
    std::cout << std::string(100, '=') << "\n";
    std::cout << "SWAPTION COMPARISON: HULL-WHITE 1-FACTOR WITH PIECEWISE CONSTANT VOLATILITY\n";
    std::cout << "Pricing: Analytical vs Monte Carlo vs MC+XAD\n";
    std::cout << "Greeks:  Analytical vs Finite Difference vs XAD+IFT\n";
    std::cout << std::string(100, '=') << "\n\n";
    
    // Setup market data
    auto curve = createCurve();
    auto volSurface = createVolSurface();
    
    std::cout << "Market Data:\n";
    std::cout << "  Discount curve nodes: " << curve.size() << "\n";
    std::cout << "  Vol surface: " << volSurface.numExpiries() << " x " << volSurface.numTenors() 
              << " = " << volSurface.numNodes() << " nodes\n\n";
    
    // Sigma buckets for piecewise-constant volatility
    std::vector<double> sigmaTimes = {0.0, 1.0/12, 3.0/12, 6.0/12, 1.0, 2.0, 3.0, 5.0, 7.0};
    std::vector<double> sigmaInit = {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01};
    
    std::cout << "Sigma buckets (piecewise-constant): " << sigmaTimes.size() << "\n";
    std::cout << "  [0,1M), [1M,3M), [3M,6M), [6M,1Y), [1Y,2Y), [2Y,3Y), [3Y,5Y), [5Y,7Y), [7Y,inf)\n\n";
    
    // Calibration instruments (co-terminal style)
    std::vector<std::pair<double, double>> calibInst = {
        {1.0/12, 20.0}, {3.0/12, 20.0}, {6.0/12, 20.0},
        {1.0, 20.0}, {2.0, 20.0}, {3.0, 20.0}, {5.0, 20.0}, {7.0, 20.0}, {10.0, 20.0},
        {1.0, 10.0}  // Additional instrument for mean reversion
    };
    
    std::cout << "Calibration instruments: " << calibInst.size() << " swaptions\n\n";
    
    // Calibrate
    std::cout << std::string(100, '-') << "\n";
    std::cout << "CALIBRATION\n";
    std::cout << std::string(100, '-') << "\n";
    
    CalibrationEngine<double> calibEngine(curve, volSurface);
    for (const auto& [e, t] : calibInst) {
        calibEngine.addInstrument(e, t);
    }
    
    HW1FParams initialParams(0.03, sigmaTimes, sigmaInit);
    auto calibResult = calibEngine.calibrate(initialParams, 100, 1e-8, false);
    
    std::cout << std::setprecision(6);
    std::cout << "  a (mean reversion) = " << calibResult.params.a << "\n";
    std::cout << "  Calibrated sigma values:\n";
    for (size_t i = 0; i < calibResult.params.sigmaValues.size(); ++i) {
        double tStart = calibResult.params.sigmaTimes[i];
        double tEnd = (i + 1 < calibResult.params.sigmaTimes.size()) 
                      ? calibResult.params.sigmaTimes[i + 1] : 100.0;
        std::cout << "    sigma_" << (i+1) << " [" << std::setprecision(2) << tStart 
                  << "Y, " << tEnd << "Y): " << std::setprecision(6) 
                  << calibResult.params.sigmaValues[i] << "\n";
    }
    std::cout << "  RMSE = $" << std::setprecision(4) << calibResult.rmse << "\n";
    std::cout << "  Time = " << std::setprecision(3) << calibResult.elapsedTime << "s\n\n";
    
    // 5 test swaptions
    std::vector<std::pair<double, double>> testSwaptions = {
        {1.0, 5.0},    // 1Y x 5Y
        {2.0, 10.0},   // 2Y x 10Y
        {5.0, 5.0},    // 5Y x 5Y
        {7.0, 20.0},   // 7Y x 20Y
        {10.0, 10.0}   // 10Y x 10Y
    };
    std::vector<std::string> names = {"1Yx5Y", "2Yx10Y", "5Yx5Y", "7Yx20Y", "10Yx10Y"};
    
    // MC config
    MCConfig mcConfig(5000, 50, true, 42);
    
    std::vector<SwaptionPricingResult> pricingResults;
    std::vector<SwaptionGreeksResult> greeksResults;
    
    // Process each swaption
    for (size_t s = 0; s < testSwaptions.size(); ++s) {
        double expiry = testSwaptions[s].first;
        double tenor = testSwaptions[s].second;
        
        std::cout << std::string(100, '=') << "\n";
        std::cout << "SWAPTION " << (s+1) << ": " << names[s] 
                  << " (" << expiry << "Y expiry x " << tenor << "Y tenor)\n";
        std::cout << std::string(100, '=') << "\n\n";
        
        // Create swaption
        double notional = 1e6;
        VanillaSwap swap(expiry, expiry + tenor, 0.0, notional, true);
        swap.fixedRate = forwardSwapRate(swap, curve);
        EuropeanSwaption swaption(expiry, swap);
        
        std::cout << "Strike (ATM) = " << std::setprecision(4) << (swap.fixedRate * 100) << "%\n\n";
        
        // Create model
        HW1FModel<double> model(calibResult.params);
        
        // =====================================================================
        // PRICING COMPARISON
        // =====================================================================
        std::cout << std::string(80, '-') << "\n";
        std::cout << "PRICING COMPARISON\n";
        std::cout << std::string(80, '-') << "\n";
        
        SwaptionPricingResult pResult;
        pResult.name = names[s];
        
        // 1. Analytical (Jamshidian)
        Timer timer;
        timer.start();
        JamshidianPricer<double, double> jamPricer(model, curve);
        pResult.analyticalPrice = jamPricer.price(swaption);
        pResult.analyticalTime = timer.elapsed();
        std::cout << std::setprecision(2);
        std::cout << "  Analytical (Jamshidian): $" << pResult.analyticalPrice 
                  << "  [" << std::setprecision(4) << (pResult.analyticalTime * 1000) << " ms]\n";
        
        // 2. Monte Carlo
        timer.start();
        MonteCarloPricer<double> mcPricer(model, mcConfig);
        auto mcResult = mcPricer.price(swaption, curve);
        pResult.mcPrice = mcResult.price;
        pResult.mcTime = timer.elapsed();
        std::cout << std::setprecision(2);
        std::cout << "  Monte Carlo:             $" << pResult.mcPrice 
                  << "  [" << std::setprecision(4) << (pResult.mcTime * 1000) << " ms]\n";
        
        // 3. Monte Carlo + XAD (tape built, but only price returned)
        timer.start();
        // XAD version runs MC with tape recording
        XADIFTGreeksEngine<double> xadEngine(curve, volSurface, calibInst);
        auto xadResult = xadEngine.computeXADIFT(swaption, calibResult.params, mcConfig);
        pResult.mcXadPrice = xadResult.price;
        pResult.mcXadTime = timer.elapsed();
        std::cout << std::setprecision(2);
        std::cout << "  Monte Carlo + XAD:       $" << pResult.mcXadPrice 
                  << "  [" << std::setprecision(4) << (pResult.mcXadTime * 1000) << " ms]\n\n";
        
        pricingResults.push_back(pResult);
        
        // =====================================================================
        // GREEKS COMPARISON
        // =====================================================================
        std::cout << std::string(80, '-') << "\n";
        std::cout << "GREEKS COMPARISON\n";
        std::cout << std::string(80, '-') << "\n";
        
        SwaptionGreeksResult gResult;
        gResult.name = names[s];
        
        // 1. Analytical Greeks (FD + Chain Rule on Jamshidian, fast)
        std::cout << "Computing Analytical Greeks (FD on Jamshidian)...\n";
        timer.start();
        ChainRuleGreeksEngine<double> chainEngine(curve, volSurface, calibInst);
        auto analyticalGreeks = chainEngine.computeChainRule(swaption, calibResult.params, mcConfig);
        gResult.analyticalTime = timer.elapsed();
        gResult.analytical_dVda = analyticalGreeks.dVda;
        gResult.analytical_dVdsigma = analyticalGreeks.dVdsigma;
        gResult.analytical_sumDVdsigma = 0;
        for (double v : analyticalGreeks.dVdsigma) gResult.analytical_sumDVdsigma += v;
        gResult.analytical_sumVolGreeks = 0;
        for (const auto& row : analyticalGreeks.volGreeks) {
            for (double v : row) gResult.analytical_sumVolGreeks += std::abs(v);
        }
        gResult.analytical_sumCurveGreeks = 0;
        for (double v : analyticalGreeks.curveGreeks) gResult.analytical_sumCurveGreeks += v;
        std::cout << std::setprecision(3);
        std::cout << "  Time: " << gResult.analyticalTime << "s\n";
        
        // 2. Finite Difference (Naive - bump & recalibrate)
        std::cout << "Computing FD Greeks (bump & recalibrate)...\n";
        timer.start();
        FDGreeksEngine<double> fdEngine(curve, volSurface, calibInst);
        auto fdGreeks = fdEngine.computeNaiveFD(swaption, calibResult.params, mcConfig);
        gResult.fdTime = timer.elapsed();
        gResult.fd_dVda = 0;  // FD Naive doesn't compute dV/da directly
        gResult.fd_dVdsigma.resize(calibResult.params.sigmaValues.size(), 0.0);
        gResult.fd_sumDVdsigma = 0;
        gResult.fd_sumVolGreeks = 0;
        for (const auto& row : fdGreeks.volGreeks) {
            for (double v : row) gResult.fd_sumVolGreeks += std::abs(v);
        }
        gResult.fd_sumCurveGreeks = 0;
        for (double v : fdGreeks.curveGreeks) gResult.fd_sumCurveGreeks += v;
        std::cout << std::setprecision(3);
        std::cout << "  Time: " << gResult.fdTime << "s\n";
        
        // 3. XAD + IFT
        std::cout << "Computing XAD+IFT Greeks...\n";
        gResult.xadIftTime = pResult.mcXadTime;  // Already computed above
        gResult.xadIft_dVda = xadResult.dVda;
        gResult.xadIft_dVdsigma = xadResult.dVdsigma;
        gResult.xadIft_sumDVdsigma = 0;
        for (double v : xadResult.dVdsigma) gResult.xadIft_sumDVdsigma += v;
        gResult.xadIft_sumVolGreeks = 0;
        for (const auto& row : xadResult.volGreeks) {
            for (double v : row) gResult.xadIft_sumVolGreeks += std::abs(v);
        }
        gResult.xadIft_sumCurveGreeks = 0;
        for (double v : xadResult.curveGreeks) gResult.xadIft_sumCurveGreeks += v;
        std::cout << std::setprecision(4);
        std::cout << "  Time: " << (gResult.xadIftTime * 1000) << " ms\n\n";
        
        // Print Greeks comparison
        std::cout << std::setprecision(2);
        std::cout << "  dV/da:\n";
        std::cout << "    Analytical: $" << gResult.analytical_dVda << "\n";
        std::cout << "    XAD+IFT:    $" << gResult.xadIft_dVda << "\n\n";
        
        std::cout << "  Sum(dV/dsigma):\n";
        std::cout << "    Analytical: $" << gResult.analytical_sumDVdsigma << "\n";
        std::cout << "    XAD+IFT:    $" << gResult.xadIft_sumDVdsigma << "\n\n";
        
        greeksResults.push_back(gResult);
    }
    
    // =========================================================================
    // SUMMARY TABLES
    // =========================================================================
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "                              SUMMARY TABLES\n";
    std::cout << std::string(100, '=') << "\n\n";
    
    // Pricing Summary
    std::cout << "PRICING COMPARISON (values and timing)\n";
    std::cout << std::string(95, '-') << "\n";
    std::cout << std::left << std::setw(10) << "Swaption" 
              << std::right << std::setw(14) << "Analytical($)" 
              << std::setw(14) << "MC($)"
              << std::setw(14) << "MC+XAD($)"
              << std::setw(12) << "Ana(ms)"
              << std::setw(12) << "MC(ms)"
              << std::setw(12) << "MC+XAD(ms)" << "\n";
    std::cout << std::string(95, '-') << "\n";
    
    for (const auto& r : pricingResults) {
        std::cout << std::left << std::setw(10) << r.name
                  << std::right << std::setprecision(2)
                  << std::setw(14) << r.analyticalPrice
                  << std::setw(14) << r.mcPrice
                  << std::setw(14) << r.mcXadPrice
                  << std::setprecision(2)
                  << std::setw(12) << (r.analyticalTime * 1000)
                  << std::setw(12) << (r.mcTime * 1000)
                  << std::setw(12) << (r.mcXadTime * 1000) << "\n";
    }
    std::cout << std::string(95, '-') << "\n\n";
    
    // Price Differences
    std::cout << "PRICE DIFFERENCES (vs Analytical)\n";
    std::cout << std::string(50, '-') << "\n";
    std::cout << std::left << std::setw(10) << "Swaption" 
              << std::right << std::setw(15) << "MC Diff(%)"
              << std::setw(15) << "MC+XAD Diff(%)" << "\n";
    std::cout << std::string(50, '-') << "\n";
    
    for (const auto& r : pricingResults) {
        double mcDiff = 100.0 * (r.mcPrice - r.analyticalPrice) / r.analyticalPrice;
        double xadDiff = 100.0 * (r.mcXadPrice - r.analyticalPrice) / r.analyticalPrice;
        std::cout << std::left << std::setw(10) << r.name
                  << std::right << std::setprecision(3)
                  << std::setw(15) << mcDiff
                  << std::setw(15) << xadDiff << "\n";
    }
    std::cout << std::string(50, '-') << "\n\n";
    
    // Greeks Timing Summary
    std::cout << "GREEKS TIMING COMPARISON\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << std::left << std::setw(10) << "Swaption" 
              << std::right << std::setw(15) << "Analytical(s)" 
              << std::setw(15) << "FD(s)"
              << std::setw(15) << "XAD+IFT(ms)"
              << std::setw(15) << "Speedup" << "\n";
    std::cout << std::string(70, '-') << "\n";
    
    for (const auto& r : greeksResults) {
        double speedup = r.fdTime / r.xadIftTime;
        std::cout << std::left << std::setw(10) << r.name
                  << std::right << std::setprecision(2)
                  << std::setw(15) << r.analyticalTime
                  << std::setw(15) << r.fdTime
                  << std::setprecision(1)
                  << std::setw(15) << (r.xadIftTime * 1000)
                  << std::setw(15) << (std::to_string((int)speedup) + "x") << "\n";
    }
    std::cout << std::string(70, '-') << "\n\n";
    
    // Greeks Values - dV/da
    std::cout << "GREEKS VALUES: dV/da\n";
    std::cout << std::string(55, '-') << "\n";
    std::cout << std::left << std::setw(10) << "Swaption" 
              << std::right << std::setw(20) << "Analytical($)" 
              << std::setw(20) << "XAD+IFT($)" << "\n";
    std::cout << std::string(55, '-') << "\n";
    
    for (const auto& r : greeksResults) {
        std::cout << std::left << std::setw(10) << r.name
                  << std::right << std::setprecision(2)
                  << std::setw(20) << r.analytical_dVda
                  << std::setw(20) << r.xadIft_dVda << "\n";
    }
    std::cout << std::string(55, '-') << "\n\n";
    
    // Greeks Values - Sum(dV/dsigma)
    std::cout << "GREEKS VALUES: Sum(dV/dsigma)\n";
    std::cout << std::string(55, '-') << "\n";
    std::cout << std::left << std::setw(10) << "Swaption" 
              << std::right << std::setw(20) << "Analytical($)" 
              << std::setw(20) << "XAD+IFT($)" << "\n";
    std::cout << std::string(55, '-') << "\n";
    
    for (const auto& r : greeksResults) {
        std::cout << std::left << std::setw(10) << r.name
                  << std::right << std::setprecision(2)
                  << std::setw(20) << r.analytical_sumDVdsigma
                  << std::setw(20) << r.xadIft_sumDVdsigma << "\n";
    }
    std::cout << std::string(55, '-') << "\n\n";
    
    // Detailed dV/dsigma per bucket
    std::cout << "DETAILED dV/dsigma PER BUCKET\n";
    std::cout << std::string(100, '=') << "\n";
    for (const auto& r : greeksResults) {
        std::cout << "\n" << r.name << ":\n";
        std::cout << std::left << std::setw(20) << "Bucket" 
                  << std::right << std::setw(20) << "Analytical($)" 
                  << std::setw(20) << "XAD+IFT($)" << "\n";
        std::cout << std::string(60, '-') << "\n";
        for (size_t i = 0; i < r.analytical_dVdsigma.size(); ++i) {
            std::cout << std::left << std::setw(20) << ("sigma_" + std::to_string(i+1))
                      << std::right << std::setprecision(2)
                      << std::setw(20) << r.analytical_dVdsigma[i]
                      << std::setw(20) << r.xadIft_dVdsigma[i] << "\n";
        }
    }
    
    // Save to file
    std::ofstream outFile("SWAPTION_COMPARISON_RESULTS.txt");
    if (outFile.is_open()) {
        outFile << "SWAPTION COMPARISON RESULTS\n";
        outFile << "Hull-White 1-Factor Model with Piecewise Constant Volatility\n";
        outFile << std::string(80, '=') << "\n\n";
        
        outFile << "PRICING COMPARISON:\n";
        outFile << std::string(95, '-') << "\n";
        outFile << std::left << std::setw(10) << "Swaption" 
                << std::right << std::setw(14) << "Analytical" 
                << std::setw(14) << "MC"
                << std::setw(14) << "MC+XAD"
                << std::setw(12) << "Ana(ms)"
                << std::setw(12) << "MC(ms)"
                << std::setw(12) << "XAD(ms)" << "\n";
        outFile << std::string(95, '-') << "\n";
        
        for (const auto& r : pricingResults) {
            outFile << std::left << std::setw(10) << r.name
                    << std::right << std::fixed << std::setprecision(2)
                    << std::setw(14) << r.analyticalPrice
                    << std::setw(14) << r.mcPrice
                    << std::setw(14) << r.mcXadPrice
                    << std::setw(12) << (r.analyticalTime * 1000)
                    << std::setw(12) << (r.mcTime * 1000)
                    << std::setw(12) << (r.mcXadTime * 1000) << "\n";
        }
        
        outFile << "\n\nGREEKS TIMING:\n";
        outFile << std::string(70, '-') << "\n";
        for (const auto& r : greeksResults) {
            double speedup = r.fdTime / r.xadIftTime;
            outFile << std::left << std::setw(10) << r.name
                    << std::right << std::setprecision(2)
                    << "  Ana=" << r.analyticalTime << "s"
                    << "  FD=" << r.fdTime << "s"
                    << "  XAD=" << std::setprecision(1) << (r.xadIftTime * 1000) << "ms"
                    << "  Speedup=" << (int)speedup << "x\n";
        }
        
        outFile << "\n\nGREEKS VALUES (dV/da and Sum(dV/dsigma)):\n";
        outFile << std::string(70, '-') << "\n";
        for (const auto& r : greeksResults) {
            outFile << r.name << ":\n";
            outFile << "  dV/da:        Ana=" << std::setprecision(2) << r.analytical_dVda 
                    << "  XAD=" << r.xadIft_dVda << "\n";
            outFile << "  Sum(dV/dsig): Ana=" << r.analytical_sumDVdsigma 
                    << "  XAD=" << r.xadIft_sumDVdsigma << "\n\n";
        }
        
        outFile.close();
        std::cout << "\nResults saved to SWAPTION_COMPARISON_RESULTS.txt\n";
    }
    
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "                           COMPARISON COMPLETE\n";
    std::cout << std::string(100, '=') << "\n";
    
    return 0;
}
