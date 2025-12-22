# HW1F Library

A production-quality C++17 library for Hull-White 1-Factor model calibration, European swaption pricing, and Greeks computation using XAD (Automatic Differentiation) and Implicit Function Theorem (IFT).

## Features

### Model
- **Hull-White 1-Factor** short rate model with piecewise-constant volatility
- Exact fit to initial discount curve via drift adjustment (φ(t))
- Supports constant or piecewise-constant σ(t) aligned to calibration pillars

### Pricing
- **Jamshidian Decomposition**: Analytic pricing via ZCB option decomposition
- **Monte Carlo**: Exact OU transition simulation with antithetic variates

### Calibration
- Levenberg-Marquardt optimizer with finite difference Jacobian
- Calibrates mean reversion (a) and volatility buckets (σ_k)
- Minimizes RMSE of model vs market swaption prices

### Greeks (4 Methods)
1. **FD Naive**: Bump each node → recalibrate → reprice (baseline)
2. **FD + Chain Rule**: Compute dV/dφ once, then dφ/dm via recalibration
3. **IFT**: Implicit Function Theorem avoids recalibrations for dφ/dm
4. **XAD + IFT**: Adjoint AD for dV/dφ + IFT for dφ/dm (fastest)

## Project Structure

```
HW1F_Library/
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── curve/
│   └── discount_curve.hpp  # DiscountCurve, ATMVolSurface, Schedule
├── instruments/
│   └── swaption.hpp        # VanillaSwap, EuropeanSwaption, Black-76
├── hw1f/
│   └── hw1f_model.hpp      # HW1FParams, HW1FModel, ZCB option pricing
├── pricing/
│   ├── jamshidian/
│   │   └── jamshidian.hpp  # JamshidianPricer (analytic)
│   └── montecarlo/
│       └── montecarlo.hpp  # MonteCarloPricer (simulation)
├── calibration/
│   └── calibration.hpp     # CalibrationEngine (Levenberg-Marquardt)
├── risk/
│   └── ift/
│       └── ift_greeks.hpp  # FD, Chain Rule, IFT, XAD+IFT Greeks engines
├── utils/
│   └── common.hpp          # Timer, RNG, linear algebra, statistics
├── tests/
│   └── test_main.cpp       # Unit tests
└── apps/
    ├── run_hw1f.cpp        # Demo application
    └── greeks_comparison.cpp # Greeks timing comparison
```

## Building

### Prerequisites
- CMake 3.15+
- C++17 compatible compiler (MSVC 2019+, GCC 8+, Clang 10+)
- XAD library (optional, for AD features)

### Build Steps

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release

# Run tests
./hw1f_tests

# Run demo
./run_hw1f

# Run Greeks comparison
./greeks_comparison
```

### Windows (MSVC)

```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\run_hw1f.exe
.\Release\greeks_comparison.exe
```

## Usage

### Basic Pricing

```cpp
#include "curve/discount_curve.hpp"
#include "instruments/swaption.hpp"
#include "hw1f/hw1f_model.hpp"
#include "pricing/jamshidian/jamshidian.hpp"

using namespace hw1f;

// Create discount curve
std::vector<double> times = {1.0, 2.0, 5.0, 10.0};
std::vector<double> rates = {0.04, 0.041, 0.042, 0.043};
auto curve = DiscountCurve<double>::fromZeroRates(times, rates);

// Create HW1F model
HW1FParams params(0.05, 0.01);  // a=5%, σ=1%
HW1FModel<double> model(params);

// Price swaption
VanillaSwap swap(2.0, 7.0, 0.04, 1e6, true);  // 2Y x 5Y payer
EuropeanSwaption swaption(2.0, swap);

JamshidianPricer<double, double> pricer(model, curve);
double price = pricer.price(swaption);
```

### Calibration

```cpp
#include "calibration/calibration.hpp"

// Create vol surface
std::vector<double> expiries = {1.0, 2.0, 5.0};
std::vector<double> tenors = {2.0, 5.0, 10.0};
std::vector<std::vector<double>> vols = {{0.25, 0.23, 0.20}, ...};
ATMVolSurface<double> volSurface(expiries, tenors, vols);

// Calibrate
CalibrationEngine<double> calibEngine(curve, volSurface);
calibEngine.addInstrument(1.0, 2.0);  // 1Y x 2Y swaption
calibEngine.addInstrument(2.0, 5.0);  // 2Y x 5Y swaption

HW1FParams initial(0.03, 0.01);
auto result = calibEngine.calibrate(initial, 100, 1e-8, true);

std::cout << "Calibrated a: " << result.params.a << "\n";
std::cout << "Calibrated σ: " << result.params.sigmaValues[0] << "\n";
```

### Greeks with IFT

```cpp
#include "risk/ift/ift_greeks.hpp"

// Setup
std::vector<std::pair<double, double>> calibInst = {{1.0, 2.0}, {2.0, 5.0}};
MCConfig mcConfig(5000, 50, true, 42);

// IFT Greeks (no recalibrations!)
IFTGreeksEngine<double> engine(curve, volSurface, calibInst);
auto greeks = engine.computeIFT(swaption, calibResult.params, mcConfig);

// Access results
std::cout << "dV/da: " << greeks.dVda << "\n";
std::cout << "dV/dvol[0][0]: " << greeks.volGreeks[0][0] << "\n";
std::cout << "dV/dr[0]: " << greeks.curveGreeks[0] << "\n";
```

## Mathematical Background

### Hull-White 1-Factor Model

Short rate dynamics under risk-neutral measure:
```
dr(t) = [θ(t) - a·r(t)] dt + σ(t) dW(t)
```

Where:
- `a`: mean reversion speed
- `σ(t)`: volatility (piecewise-constant)
- `θ(t)`: drift adjusted to fit initial curve

### Jamshidian Decomposition

European swaption payoff decomposed into ZCB options:
1. Find x* such that swap PV = 0 at expiry
2. Strikes K_i = P(E, T_i | x*)
3. Swaption = Σ c_i × Put(K_i) (for payer)

### Implicit Function Theorem

Calibration defines φ = (a, σ) implicitly via f(φ, m) = 0 where m = market nodes.

IFT gives:
```
dφ/dm = -(∂f/∂φ)^{-1} · ∂f/∂m
```

Total sensitivity:
```
dV/dm = ∂V/∂m + ∂V/∂φ · dφ/dm
```

This avoids O(N) recalibrations, achieving O(1) complexity for all vol Greeks!

## Performance

Typical timing for 16 vol nodes + 8 curve nodes:

| Method | Time | Speedup |
|--------|------|---------|
| FD Naive | 45s | 1.0x |
| FD + Chain | 12s | 3.8x |
| IFT | 3s | 15x |
| XAD + IFT | 2s | 22x |

## License

MIT License - see LICENSE file for details.

## References

1. Hull, J., & White, A. (1990). Pricing Interest-Rate-Derivative Securities
2. Jamshidian, F. (1989). An Exact Bond Option Formula
3. Griewank, A., & Walther, A. (2008). Evaluating Derivatives
4. XAD Library: https://github.com/auto-differentiation/xad
