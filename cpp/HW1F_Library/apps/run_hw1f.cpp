// =============================================================================
// HW1F Library - Main Application (run_hw1f)
// Demonstrates calibration, pricing, and Greeks computation
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

using namespace hw1f;

// =============================================================================
// Sample Market Data (Real OIS Curve)
// =============================================================================

DiscountCurve<double> loadOISCurve() {
    // Real OIS curve data (as of 2024-12-16)
    std::vector<double> times = {
        0.003, 0.019, 0.083, 0.167, 0.25, 0.5, 0.75, 1.0,
        1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0
    };
    std::vector<double> rates = {
        0.0433, 0.0433, 0.0432, 0.0430, 0.0428, 0.0423, 0.0418, 0.0413,
        0.0405, 0.0398, 0.0387, 0.0380, 0.0376, 0.0374, 0.0377, 0.0383,
        0.0387, 0.0390
    };
    return DiscountCurve<double>::fromZeroRates(times, rates);
}

ATMVolSurface<double> loadVolSurface() {
    // ATM swaption vol surface (synthetic but realistic)
    std::vector<double> expiries = {0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0};
    std::vector<double> tenors = {1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0};
    
    std::vector<std::vector<double>> vols = {
        {0.65, 0.55, 0.50, 0.45, 0.42, 0.40, 0.38},  // 6M expiry
        {0.55, 0.48, 0.45, 0.42, 0.40, 0.38, 0.36},  // 1Y expiry
        {0.48, 0.43, 0.40, 0.38, 0.36, 0.35, 0.34},  // 2Y expiry
        {0.43, 0.40, 0.38, 0.36, 0.35, 0.34, 0.33},  // 3Y expiry
        {0.38, 0.36, 0.35, 0.34, 0.33, 0.32, 0.31},  // 5Y expiry
        {0.35, 0.34, 0.33, 0.32, 0.31, 0.30, 0.29},  // 7Y expiry
        {0.32, 0.31, 0.30, 0.29, 0.28, 0.28, 0.27}   // 10Y expiry
    };
    
    return ATMVolSurface<double>(expiries, tenors, vols);
}

// =============================================================================
// Main Application
// =============================================================================

int main() {
    std::cout << std::string(80, '=') << "\n";
    std::cout << "Hull-White 1-Factor Model: Calibration, Pricing & Greeks\n";
    std::cout << "Jamshidian Decomposition & Monte Carlo with XAD + IFT\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    // Load market data
    std::cout << "Loading market data...\n";
    auto curve = loadOISCurve();
    auto volSurface = loadVolSurface();
    
    std::cout << "  OIS Curve: " << curve.size() << " nodes\n";
    std::cout << "  Vol Surface: " << volSurface.numExpiries() << " expiries x " 
              << volSurface.numTenors() << " tenors = " 
              << volSurface.numNodes() << " nodes\n\n";
    
    // ==========================================================================
    // Calibration
    // ==========================================================================
    std::cout << std::string(80, '-') << "\n";
    std::cout << "CALIBRATION\n";
    std::cout << std::string(80, '-') << "\n\n";
    
    // Select calibration instruments (diagonal + some off-diagonal)
    std::vector<std::pair<double, double>> calibInstruments = {
        {1.0, 1.0}, {1.0, 2.0}, {1.0, 5.0},
        {2.0, 1.0}, {2.0, 2.0}, {2.0, 5.0},
        {3.0, 2.0}, {3.0, 5.0},
        {5.0, 5.0}, {5.0, 10.0}
    };
    
    CalibrationEngine<double> calibEngine(curve, volSurface);
    for (const auto& [e, t] : calibInstruments) {
        calibEngine.addInstrument(e, t);
    }
    
    std::cout << "Calibrating to " << calibInstruments.size() << " ATM swaptions...\n";
    
    // Piecewise-constant sigma with buckets matching expiry structure
    std::vector<double> sigmaTimes = {0.0, 1.0, 2.0, 3.0, 5.0};
    std::vector<double> sigmaInit = {0.01, 0.01, 0.01, 0.01, 0.01};
    HW1FParams initialParams(0.03, sigmaTimes, sigmaInit);  // Piecewise-constant sigma
    auto calibResult = calibEngine.calibrate(initialParams, 100, 1e-8, true);
    
    std::cout << "\nCalibration Results (Piecewise-Constant Sigma):\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  a (mean reversion) = " << calibResult.params.a << "\n";
    std::cout << "  Sigma buckets (piecewise-constant):\n";
    for (size_t i = 0; i < calibResult.params.sigmaValues.size(); ++i) {
        double tStart = calibResult.params.sigmaTimes[i];
        double tEnd = (i + 1 < calibResult.params.sigmaTimes.size()) 
                      ? calibResult.params.sigmaTimes[i + 1] : 100.0;
        std::cout << "    sigma_" << i+1 << " [" << tStart << "Y, " << tEnd << "Y): " 
                  << calibResult.params.sigmaValues[i] << "\n";
    }
    std::cout << "  RMSE               = $" << std::setprecision(2) << calibResult.rmse << "\n";
    std::cout << "  Iterations         = " << calibResult.iterations << "\n";
    std::cout << "  Converged          = " << (calibResult.converged ? "Yes" : "No") << "\n";
    std::cout << "  Time               = " << std::setprecision(3) << calibResult.elapsedTime << "s\n\n";
    
    // ==========================================================================
    // Target Swaption
    // ==========================================================================
    std::cout << std::string(80, '-') << "\n";
    std::cout << "TARGET SWAPTION\n";
    std::cout << std::string(80, '-') << "\n\n";
    
    double expiry = 2.0;
    double tenor = 5.0;
    double notional = 1e6;
    
    VanillaSwap swap(expiry, expiry + tenor, 0.0, notional, true);
    double fwd = forwardSwapRate(swap, curve);
    swap.fixedRate = fwd;  // ATM
    
    EuropeanSwaption swaption(expiry, swap);
    
    std::cout << std::setprecision(4);
    std::cout << "  " << expiry << "Y x " << tenor << "Y Payer Swaption\n";
    std::cout << "  ATM Strike     = " << fwd * 100 << "%\n";
    std::cout << "  Notional       = $" << std::fixed << std::setprecision(0) << notional << "\n\n";
    
    // ==========================================================================
    // Pricing: Jamshidian vs Monte Carlo
    // ==========================================================================
    std::cout << std::string(80, '-') << "\n";
    std::cout << "PRICING\n";
    std::cout << std::string(80, '-') << "\n\n";
    
    HW1FModel<double> model(calibResult.params);
    
    // Jamshidian (analytic)
    Timer timer;
    timer.start();
    JamshidianPricer<double, double> jamPricer(model, curve);
    double jamPrice = jamPricer.price(swaption);
    double jamTime = timer.elapsed();
    
    std::cout << std::setprecision(2);
    std::cout << "Jamshidian (Analytic):\n";
    std::cout << "  Price = $" << jamPrice << "\n";
    std::cout << "  Time  = " << std::setprecision(4) << jamTime * 1000 << " ms\n\n";
    
    // Monte Carlo
    MCConfig mcConfig(20000, 100, true, 12345);
    
    timer.start();
    MonteCarloPricer<double> mcPricer(model, mcConfig);
    auto mcResult = mcPricer.price(swaption, curve);
    
    std::cout << std::setprecision(2);
    std::cout << "Monte Carlo:\n";
    std::cout << "  Price     = $" << mcResult.price << "\n";
    std::cout << "  Std Error = $" << mcResult.stdError << "\n";
    std::cout << "  Paths     = " << mcConfig.numPaths << "\n";
    std::cout << "  Time      = " << std::setprecision(3) << mcResult.elapsedTime << " s\n\n";
    
    double priceDiff = std::abs(mcResult.price - jamPrice) / jamPrice * 100;
    std::cout << "MC vs Jamshidian difference: " << std::setprecision(2) << priceDiff << "%\n\n";
    
    // ==========================================================================
    // Greeks Computation
    // ==========================================================================
    std::cout << std::string(80, '-') << "\n";
    std::cout << "GREEKS\n";
    std::cout << std::string(80, '-') << "\n\n";
    
    MCConfig greeksMCConfig(5000, 50, true, 42);
    
    // IFT + XAD Greeks
    std::cout << "Computing Greeks using IFT + XAD...\n";
    IFTGreeksEngine<double> iftEngine(curve, volSurface, calibInstruments);
    auto iftGreeks = iftEngine.computeIFT(swaption, calibResult.params, greeksMCConfig);
    
    std::cout << "\nIFT + XAD Greeks (time: " << std::setprecision(3) << iftGreeks.elapsedTime << "s):\n";
    std::cout << std::setprecision(4);
    std::cout << "  dV/da     = $" << iftGreeks.dVda << " per unit a\n";
    std::cout << "  dV/dsigma = $" << iftGreeks.dVdsigma[0] << " per unit sigma\n\n";
    
    std::cout << "Curve Node Greeks (per 1bp):\n";
    std::cout << std::left << std::setw(10) << "Maturity" << std::right << std::setw(15) << "dV/dr (1bp)\n";
    std::cout << std::string(25, '-') << "\n";
    for (size_t i = 0; i < std::min(size_t(8), curve.size()); ++i) {
        std::cout << std::setw(10) << std::setprecision(3) << curve.times()[i]
                  << std::setw(15) << std::setprecision(2) << iftGreeks.curveGreeks[i] * 0.0001 << "\n";
    }
    
    std::cout << "\nVol Surface Greeks (per 1bp):\n";
    std::cout << std::left << std::setw(15) << "Node" << std::right << std::setw(15) << "dV/dvol (1bp)\n";
    std::cout << std::string(30, '-') << "\n";
    for (size_t ei = 0; ei < std::min(size_t(4), volSurface.numExpiries()); ++ei) {
        for (size_t ti = 0; ti < std::min(size_t(3), volSurface.numTenors()); ++ti) {
            std::string node = std::to_string(static_cast<int>(volSurface.expiries()[ei])) + "Y x " +
                              std::to_string(static_cast<int>(volSurface.tenors()[ti])) + "Y";
            std::cout << std::setw(15) << node
                      << std::setw(15) << std::setprecision(2) << iftGreeks.volGreeks[ei][ti] * 0.0001 << "\n";
        }
    }
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Computation Complete\n";
    std::cout << std::string(80, '=') << "\n";
    
    return 0;
}
