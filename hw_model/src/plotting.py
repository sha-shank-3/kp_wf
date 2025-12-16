import numpy as np
import matplotlib.pyplot as plt

from .hullwhite import HullWhiteModel

def plot_market_vs_model_rates(model: HullWhiteModel, times, rates):
    plt.figure(figsize=(10,6))
    plt.plot(times, rates*100, 'o', markersize=6, label='Market Zero Rates', color='red')
    model_rates = np.array([model.zero_rate(T) for T in times])
    plt.plot(times, model_rates*100, '-', linewidth=2, label='Model Zero Rates (Interpolated)', color='green', alpha=0.7)
    # Instantaneous forward on fine grid
    fine = np.linspace(min(times), max(times), 100)
    fwd = np.array([model.instantaneous_forward(T) for T in fine])
    plt.plot(fine, fwd*100, '--', linewidth=1.5, label='Model Instantaneous Forward Rate', color='blue')
    plt.title('Calibration Quality: Market vs Model Rates')
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Rate (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_bond_price_calibration(model: HullWhiteModel, times, market_prices):
    plt.figure(figsize=(10,6))
    model_prices = np.array([model.bond_price(model.rates[0], 0.0, T) for T in times])
    plt.plot(times, market_prices, 'o', markersize=6, label='Market Bond Prices', color='red')
    plt.plot(times, model_prices, '-', linewidth=2, label='Model Bond Prices', color='blue')
    plt.title('Bond Price Calibration Check')
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Bond Price P(0,T)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Bar chart of relative errors
    plt.figure(figsize=(10,6))
    errors = (model_prices - market_prices) / np.maximum(market_prices, 1e-12)
    plt.bar(times, errors)
    plt.title('Calibration Error (%)')
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Relative Error (% of Market Price)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
