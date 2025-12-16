# Hull-White Model Calibration Demo

This project reconstructs a Python implementation inspired by the attached screenshots: plotting market vs model zero rates, calibrating bond prices, and visualizing calibration errors for a one-factor Hull-White (extended Vasicek) short-rate model.

## Features
- Load or generate sample market zero rates.
- Fit Hull-White parameters to match the term structure.
- Compute zero-coupon bond prices and instantaneous forward rates.
- Plot market vs model rates, and calibration error visualization.

## Quick Start

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Run the demo
```bash
python main.py
```

This will generate plots and print basic calibration info.

## Notes
- The implementation uses a simple parametric mean-reversion `a` and volatility `sigma`, with a deterministic shift to fit the initial curve.
- For production-grade calibration, replace the toy market data with actual market zero rates and maturities.
