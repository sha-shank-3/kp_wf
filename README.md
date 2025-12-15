# Hull-White Swaption Pricing with XAD Greeks

This repository contains implementations of a Hull-White one-factor model for pricing European swaptions with Greeks calculation using Adjoint Algorithmic Differentiation (AAD).

## Features

- **Hull-White Model**: One-factor short-rate model with mean reversion
- **European Swaption Pricing**: Monte Carlo simulation
- **Real OIS Curve Data**: SOFR-based discount curve (December 2024)
- **ATM Vol Surface Calibration**: Calibrates HW parameters (a, σ) to market vols
- **Greeks Calculation**:
  - Rate curve sensitivities (Delta) using XAD AAD
  - Hull-White parameter sensitivities (dV/da, dV/dσ)
  - Vol surface sensitivities using Implicit Function Theorem
- **Performance Comparison**: XAD AAD vs Finite Differences

## Project Structure

```
├── python/
│   ├── xad_hullwhite_swaption.py    # Original swaption pricer
│   └── xad_hullwhite_greeks.py      # XAD Greeks implementation
│
├── cpp/HullWhiteSwaption/
│   ├── CMakeLists.txt               # Build configuration
│   ├── HullWhite.hpp                # Basic HW model functions
│   ├── HullWhiteEnhanced.hpp        # Enhanced with vol surface
│   ├── main.cpp                     # Basic XAD vs FD comparison
│   └── main_enhanced.cpp            # Full implementation with vol Greeks
│
└── README.md
```

## Python Implementation

### Requirements
```bash
pip install numpy xad
```

### Usage
```bash
python xad_hullwhite_greeks.py
```

### Results
- Computes swaption price (~$45,000 for 1Yx5Y ATM payer)
- Calculates 11 rate curve Greeks
- Compares XAD AAD vs Finite Differences (match within 0.27%)

## C++ Implementation

### Requirements
- C++17 compiler (MSVC, GCC, Clang)
- CMake 3.15+
- XAD library (https://github.com/auto-differentiation/xad)

### Building
```bash
# Clone XAD library
git clone https://github.com/auto-differentiation/xad.git

# Copy HullWhiteSwaption to xad/samples/
cp -r cpp/HullWhiteSwaption xad/samples/

# Add to xad/samples/CMakeLists.txt:
# add_subdirectory(HullWhiteSwaption)

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --target HullWhiteSwaptionEnhanced
```

### Usage
```bash
./Release/HullWhiteSwaptionEnhanced.exe
```

### Results
- **Rate Greeks**: 20 OIS curve sensitivities
- **HW Param Greeks**: dV/da, dV/dσ using XAD AAD
- **Vol Surface Greeks**: 81 vol node sensitivities via Implicit Function Theorem
- **Speedup**: XAD is ~6x faster than pure FD for vol surface Greeks

## Mathematical Background

### Hull-White Model
```
dr(t) = [θ(t) - a·r(t)]dt + σ·dW(t)
```
Where θ(t) is calibrated to match the initial term structure:
```
θ(t) = ∂f/∂t + a·f(0,t) + σ²/(2a)·(1 - e^(-2at))
```

### Vol Surface Greeks via Implicit Function Theorem
```
dV/d(vol_ij) = (∂V/∂a)(∂a/∂vol_ij) + (∂V/∂σ)(∂σ/∂vol_ij)
```
- ∂V/∂a and ∂V/∂σ computed via XAD AAD (one pricing run)
- ∂a/∂vol and ∂σ/∂vol computed via FD on calibration

## Performance

| Greeks Type | XAD Time | FD Time | Speedup |
|-------------|----------|---------|---------|
| Rate Curve (20) | 0.14s | 0.08s | - |
| HW Params (2) | 0.13s | N/A | - |
| Vol Surface (81) | 2.3s | 13.9s | **6x** |

## License

MIT License

## References

- Hull, J. C., & White, A. (1990). Pricing interest-rate-derivative securities
- Giles, M. B., & Glasserman, P. (2006). Smoking adjoints: fast Monte Carlo Greeks
- XAD Library: https://github.com/auto-differentiation/xad
