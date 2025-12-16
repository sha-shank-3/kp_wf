import numpy as np
from scipy.optimize import minimize

from .hullwhite import HullWhiteModel

def generate_sample_market_curve(n=10, max_T=10.0, seed=42):
    rng = np.random.default_rng(seed)
    times = np.linspace(0.5, max_T, n)
    base = 0.02 + 0.01 * (times / max_T)  # gently upward sloping
    noise = rng.normal(0.0, 0.0005, size=n)
    rates = np.maximum(base + noise, 0.0001)
    return times, rates

def fit_hullwhite_to_curve(times, rates, r0=None):
    times = np.array(times, dtype=float)
    rates = np.array(rates, dtype=float)
    if r0 is None:
        r0 = rates[0]

    def objective(params):
        a, sigma = params
        a = max(a, 1e-6)
        sigma = max(sigma, 1e-6)
        model = HullWhiteModel(a, sigma, times, rates)
        # Compare model zero rates from discounts vs market
        Tgrid = times
        model_rates = np.array([model.zero_rate(T) for T in Tgrid])
        # Calibration error: RMS
        err = np.sqrt(np.mean((model_rates - rates)**2))
        return err

    res = minimize(objective, x0=[0.1, 0.01], bounds=[(1e-6, 5.0), (1e-6, 1.0)], method='L-BFGS-B')
    a_opt, sigma_opt = float(res.x[0]), float(res.x[1])
    model = HullWhiteModel(a_opt, sigma_opt, times, rates)
    return model, res

def compute_bond_prices(model: HullWhiteModel, times):
    # Assume spot short rate r(0) ~ first rate
    r0 = model.rates[0]
    prices = []
    for T in times:
        prices.append(model.bond_price(r0, 0.0, float(T)))
    return np.array(prices)
