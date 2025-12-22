// vol_greeks_comparison.cpp
// Outputs vol surface Greeks comparison between exact-fit and LSQ calibration in clean format
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "../hw1f/hw1f_model.hpp"
#include "../calibration/calibration.hpp"
#include "../instruments/swaption.hpp"
#include "../pricing/mc_pricing.hpp"
#include "../curve/curve.hpp"
#include "../utils/vol_surface.hpp"

using namespace hw1f;

int main() {
    std::cout << std::fixed << std::setprecision(4);
    
    // Market data
    std::vector<double> curveTimes = {0.25, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30};
    std::vector<double> curveRates = {0.052, 0.050, 0.047, 0.044, 0.042, 0.040, 0.039, 0.038, 0.037, 0.036, 0.035, 0.034};
    DiscountCurve curve(curveTimes, curveRates);
    
    std::vector<double> expiries = {1.0/12, 3.0/12, 6.0/12, 1, 2, 3, 5, 7, 10};
    std::vector<double> tenors = {1, 2, 3, 5, 7, 10, 15, 20, 30};
    std::vector<std::vector<double>> vols = {
        {0.60, 0.55, 0.52, 0.48, 0.45, 0.42, 0.40, 0.38, 0.36},
        {0.55, 0.52, 0.50, 0.47, 0.44, 0.41, 0.39, 0.37, 0.35},
        {0.50, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36, 0.34},
        {0.45, 0.44, 0.43, 0.41, 0.40, 0.38, 0.37, 0.35, 0.33},
        {0.42, 0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.34, 0.32},
        {0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.31},
        {0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.32, 0.31, 0.30},
        {0.36, 0.35, 0.34, 0.33, 0.32, 0.31, 0.30, 0.29, 0.28},
        {0.34, 0.33, 0.32, 0.31, 0.30, 0.29, 0.28, 0.27, 0.26}
    };
    VolSurface volSurface(expiries, tenors, vols);
    
    // Sigma buckets
    std::vector<double> sigmaBuckets = {0, 1.0/12, 0.25, 0.5, 1, 2, 3, 5, 7};
    
    // ----- EXACT FIT Calibration (10 co-terminal instruments) -----
    CalibrationEngine exactCalib(curve, volSurface, sigmaBuckets);
    std::vector<std::pair<double, double>> exactInst = {
        {1, 20}, {2, 20}, {3, 20}, {5, 20}, {7, 20}, 
        {10, 20}, {15, 20}, {20, 20}, {1, 10}, {3, 10}
    };
    for (auto& [e, t] : exactInst) exactCalib.addInstrument(e, t);
    exactCalib.calibrate();
    auto exactParams = exactCalib.getCalibratedParams();
    
    // ----- LSQ Calibration (all 81 vol surface instruments) -----
    CalibrationEngine lsqCalib(curve, volSurface, sigmaBuckets);
    for (size_t ei = 0; ei < volSurface.numExpiries(); ++ei) {
        for (size_t ti = 0; ti < volSurface.numTenors(); ++ti) {
            lsqCalib.addInstrument(volSurface.expiries()[ei], volSurface.tenors()[ti]);
        }
    }
    lsqCalib.calibrate(200);
    auto lsqParams = lsqCalib.getCalibratedParams();
    
    // Target swaption
    double targetExpiry = 7.0, targetTenor = 20.0;
    double atmStrike = SwaptionPricer::atmStrike(curve, targetExpiry, targetTenor);
    Swaption target(targetExpiry, targetTenor, atmStrike, 1000000.0);
    
    // Setup MC pricing
    MCPricer exactMC(curve, exactParams, 100000, 42);
    MCPricer lsqMC(curve, lsqParams, 100000, 42);
    
    double exactPrice = exactMC.price(target);
    double lsqPrice = lsqMC.price(target);
    
    // Compute vol surface Greeks by FD (bump each vol node by 1bp)
    double bump = 0.0001;
    
    std::cout << "VOL SURFACE GREEKS COMPARISON (per 1bp bump)\n";
    std::cout << "============================================\n\n";
    std::cout << "Target: " << targetExpiry << "Y x " << targetTenor << "Y ATM Payer Swaption\n";
    std::cout << "Exact-fit Price: $" << exactPrice << "\n";
    std::cout << "LSQ Price: $" << lsqPrice << "\n\n";
    
    std::cout << std::left << std::setw(10) << "Node"
              << std::right << std::setw(12) << "Exact-Fit"
              << std::setw(12) << "LSQ"
              << std::setw(12) << "Diff"
              << std::setw(12) << "Diff%\n";
    std::cout << std::string(58, '-') << "\n";
    
    double totalExact = 0, totalLsq = 0;
    int nonZeroExact = 0, nonZeroLsq = 0;
    
    for (size_t ei = 0; ei < volSurface.numExpiries(); ++ei) {
        for (size_t ti = 0; ti < volSurface.numTenors(); ++ti) {
            // Bump vol surface
            VolSurface bumpedSurface = volSurface;
            bumpedSurface.bumpVol(ei, ti, bump);
            
            // Exact-fit: recalibrate and reprice
            CalibrationEngine exactBumped(curve, bumpedSurface, sigmaBuckets);
            for (auto& [e, t] : exactInst) exactBumped.addInstrument(e, t);
            exactBumped.calibrate();
            MCPricer exactBumpedMC(curve, exactBumped.getCalibratedParams(), 100000, 42);
            double exactBumpedPrice = exactBumpedMC.price(target);
            double exactGreek = (exactBumpedPrice - exactPrice) / bump * 0.0001;
            
            // LSQ: recalibrate and reprice
            CalibrationEngine lsqBumped(curve, bumpedSurface, sigmaBuckets);
            for (size_t ei2 = 0; ei2 < volSurface.numExpiries(); ++ei2) {
                for (size_t ti2 = 0; ti2 < volSurface.numTenors(); ++ti2) {
                    lsqBumped.addInstrument(bumpedSurface.expiries()[ei2], bumpedSurface.tenors()[ti2]);
                }
            }
            lsqBumped.calibrate(200);
            MCPricer lsqBumpedMC(curve, lsqBumped.getCalibratedParams(), 100000, 42);
            double lsqBumpedPrice = lsqBumpedMC.price(target);
            double lsqGreek = (lsqBumpedPrice - lsqPrice) / bump * 0.0001;
            
            // Track non-zero counts
            if (std::abs(exactGreek) > 0.001) nonZeroExact++;
            if (std::abs(lsqGreek) > 0.001) nonZeroLsq++;
            totalExact += std::abs(exactGreek);
            totalLsq += std::abs(lsqGreek);
            
            // Output
            std::string expLabel = (expiries[ei] < 1.0) ? 
                std::to_string(int(expiries[ei]*12)) + "M" :
                std::to_string(int(expiries[ei])) + "Y";
            std::string node = expLabel + "x" + std::to_string(int(tenors[ti])) + "Y";
            
            double diff = lsqGreek - exactGreek;
            double diffPct = (std::abs(exactGreek) > 0.001) ? 
                diff / exactGreek * 100 : 
                (std::abs(lsqGreek) > 0.001 ? 999.0 : 0.0);
            
            std::cout << std::setw(10) << node
                      << std::setw(12) << std::setprecision(2) << exactGreek
                      << std::setw(12) << lsqGreek
                      << std::setw(12) << diff
                      << std::setw(11) << diffPct << "%\n";
        }
    }
    
    std::cout << std::string(58, '-') << "\n";
    std::cout << "\nSUMMARY:\n";
    std::cout << "  Exact-fit non-zero Greeks: " << nonZeroExact << " / 81\n";
    std::cout << "  LSQ non-zero Greeks: " << nonZeroLsq << " / 81\n";
    std::cout << "  Total |Greeks| Exact: $" << totalExact << "\n";
    std::cout << "  Total |Greeks| LSQ: $" << totalLsq << "\n";
    
    return 0;
}
