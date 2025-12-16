from src.calibration import generate_sample_market_curve, fit_hullwhite_to_curve, compute_bond_prices
from src.plotting import plot_market_vs_model_rates, plot_bond_price_calibration

def main():
    times, rates = generate_sample_market_curve(n=12, max_T=15.0)
    model, res = fit_hullwhite_to_curve(times, rates)
    print(f"Optimized a={model.a:.6f}, sigma={model.sigma:.6f}; success={res.success}, fun={res.fun:.6e}")

    plot_market_vs_model_rates(model, times, rates)

    # Construct market bond prices from market rates (discounts from market curve)
    market_prices = model.discounts  # uses initial market curve inside model
    plot_bond_price_calibration(model, times, market_prices)

if __name__ == '__main__':
    main()
