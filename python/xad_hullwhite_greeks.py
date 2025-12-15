# -*- coding: utf-8 -*-
"""
Hull-White Swaption Pricing with XAD-based Greeks Calculation (OPTIMIZED)

Computes sensitivities (Greeks) of a European swaption price
with respect to discount curve nodes using:
1. XAD Adjoint Algorithmic Differentiation (AAD)
2. Finite Difference (bump-and-reprice)

Key optimizations:
- Consistent interpolation between numpy and XAD versions
- Pre-computed constants where possible
- Efficient tape usage

Author: Generated from code screenshots
"""

import numpy as np
import time
import xad.adj_1st as xadj
import xad.math as xmath

# =============================================================================
# Global Parameters
# =============================================================================

# Swaption Parameters
T_option = 1.0          # Swaption expiry in years
swap_tenor = 5          # Tenor of the underlying swap in years
K_strike = 0.03         # Fixed strike rate of the swap
notional = 1_000_000
is_payer_swaption = True

# Hull-White Model Parameters
a = 0.1                 # Mean reversion speed
sigma = 0.01            # Volatility

# Monte Carlo Simulation Parameters
num_paths = 3000        # Good balance between accuracy and speed
dt = 1/26.0             # Bi-weekly time step (faster, still accurate)
T_sim = T_option

# Random seed for reproducibility
np.random.seed(42)

# Pre-generate random numbers
num_steps_global = int(T_sim / dt)
Z_matrix_global = np.random.standard_normal((num_paths, num_steps_global))

# Pre-compute constants
sqrt_dt = np.sqrt(dt)


def get_market_curve():
    """Returns sample OIS curve maturities and zero rates."""
    maturities = np.array([
        1/12, 3/12, 6/12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0
    ])
    rates = np.array([
        0.0428, 0.0422, 0.0413, 0.0403, 0.0388, 0.0383,
        0.0395, 0.0408, 0.0423, 0.0433, 0.0438
    ])
    return maturities, rates


# =============================================================================
# Consistent Interpolation Functions
# =============================================================================

def find_interp_weights(x, xs):
    """
    Find interpolation index and weight for a single point.
    Returns (idx, weight) such that result = ys[idx] * (1-w) + ys[idx+1] * w
    """
    n = len(xs)
    if x <= xs[0]:
        return 0, 0.0
    if x >= xs[-1]:
        return n - 2, 1.0
    
    # Binary search
    idx = np.searchsorted(xs, x) - 1
    idx = max(0, min(idx, n - 2))
    w = (x - xs[idx]) / (xs[idx + 1] - xs[idx])
    return idx, w


def numpy_linear_interp(x, xs, ys):
    """Linear interpolation consistent with XAD version."""
    idx, w = find_interp_weights(x, xs)
    return ys[idx] * (1 - w) + ys[idx + 1] * w


def xad_linear_interp(x, xs, ys_ad):
    """Linear interpolation with XAD active types for ys."""
    idx, w = find_interp_weights(x, xs)
    return ys_ad[idx] * (1 - w) + ys_ad[idx + 1] * w


# =============================================================================
# NumPy Pricing Functions
# =============================================================================

def np_discount(T, maturities, rates):
    """Compute discount factor P(0,T)."""
    if T <= 0:
        return 1.0
    rate = numpy_linear_interp(T, maturities, rates)
    return np.exp(-rate * T)


def np_forward_rate(t, maturities, rates, h=1e-5):
    """Compute instantaneous forward rate f(0,t)."""
    t = max(t, h)
    P_t = np_discount(t, maturities, rates)
    P_th = np_discount(t + h, maturities, rates)
    return -(np.log(P_th) - np.log(P_t)) / h


def np_theta(t, maturities, rates, h=1e-5):
    """Compute Hull-White theta(t)."""
    f_t = np_forward_rate(t, maturities, rates, h)
    f_th = np_forward_rate(t + h, maturities, rates, h)
    f_prime = (f_th - f_t) / h
    return f_prime + a * f_t + (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t))


def np_bond_price(r_t, t, T, maturities, rates):
    """Compute HW bond price P(t,T)."""
    if T <= t:
        return np.ones_like(r_t) if hasattr(r_t, '__len__') else 1.0
    
    B = (1 - np.exp(-a * (T - t))) / a
    P_T = np_discount(T, maturities, rates)
    P_t = np_discount(t, maturities, rates)
    f_t = np_forward_rate(t, maturities, rates)
    
    log_A = (np.log(P_T / P_t) + B * f_t - 
             (sigma**2 / (4 * a)) * (1 - np.exp(-2 * a * t)) * B**2)
    
    return np.exp(log_A) * np.exp(-B * r_t)


def np_simulate_paths(maturities, rates, Z):
    """Simulate paths and return terminal short rates."""
    n_steps = Z.shape[1]
    n_paths = Z.shape[0]
    t_grid = np.linspace(0, T_sim, n_steps + 1)
    
    # Pre-compute theta values
    theta_vals = np.array([np_theta(t, maturities, rates) for t in t_grid])
    
    r = np.zeros((n_paths, n_steps + 1))
    r[:, 0] = np_forward_rate(0.0, maturities, rates)
    
    for i in range(n_steps):
        r[:, i+1] = r[:, i] + (theta_vals[i] - a * r[:, i]) * dt + sigma * sqrt_dt * Z[:, i]
    
    return r[:, -1]


def np_price_swaption(maturities, rates, Z):
    """Price swaption using numpy."""
    r_T = np_simulate_paths(maturities, rates, Z)
    
    # Bond prices at expiry
    swap_times = np.arange(1, swap_tenor + 1) + T_option
    bonds = np.zeros((num_paths, len(swap_times)))
    for j, T_pay in enumerate(swap_times):
        bonds[:, j] = np_bond_price(r_T, T_option, T_pay, maturities, rates)
    
    # Fixed leg
    fixed_pv = np.sum(K_strike * bonds, axis=1)
    
    # Floating leg
    float_pv = 1.0 - bonds[:, -1]
    
    # Swap value and payoff
    swap_val = (float_pv - fixed_pv) * notional
    payoffs = np.maximum(swap_val, 0) if is_payer_swaption else np.maximum(-swap_val, 0)
    
    # Discount and average
    disc = np_discount(T_option, maturities, rates)
    return disc * np.mean(payoffs)


# =============================================================================
# XAD Pricing Functions
# =============================================================================

def xad_discount(T, maturities, rates_ad):
    """Compute discount factor using XAD types."""
    if T <= 0:
        return xadj.Real(1.0)
    rate = xad_linear_interp(T, maturities, rates_ad)
    return xmath.exp(-rate * T)


def xad_forward_rate(t, maturities, rates_ad, h=1e-5):
    """Compute instantaneous forward rate using XAD types."""
    t = max(t, h)
    P_t = xad_discount(t, maturities, rates_ad)
    P_th = xad_discount(t + h, maturities, rates_ad)
    return -(xmath.log(P_th) - xmath.log(P_t)) / h


def xad_theta(t, maturities, rates_ad, h=1e-5):
    """Compute Hull-White theta(t) using XAD types."""
    f_t = xad_forward_rate(t, maturities, rates_ad, h)
    f_th = xad_forward_rate(t + h, maturities, rates_ad, h)
    f_prime = (f_th - f_t) / h
    term3 = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t))
    return f_prime + a * f_t + term3


def xad_bond_price(r_t_ad, t, T, maturities, rates_ad, f_t_cache=None):
    """Compute HW bond price using XAD types."""
    if T <= t:
        return xadj.Real(1.0)
    
    B = (1 - np.exp(-a * (T - t))) / a
    P_T = xad_discount(T, maturities, rates_ad)
    P_t = xad_discount(t, maturities, rates_ad)
    f_t = f_t_cache if f_t_cache else xad_forward_rate(t, maturities, rates_ad)
    
    log_A = (xmath.log(P_T) - xmath.log(P_t) + B * f_t - 
             (sigma**2 / (4 * a)) * (1 - np.exp(-2 * a * t)) * B**2)
    
    return xmath.exp(log_A) * xmath.exp(-B * r_t_ad)


def xad_price_single_path(maturities, rates_ad, Z_row, theta_cache):
    """Price single path with XAD, return discounted payoff."""
    n_steps = len(Z_row)
    
    # Simulate short rate
    r_t = xad_forward_rate(0.0, maturities, rates_ad)
    for i in range(n_steps):
        r_t = r_t + (theta_cache[i] - a * r_t) * dt + sigma * sqrt_dt * Z_row[i]
    
    r_T = r_t
    
    # Bond prices at expiry
    swap_times = [float(j) + T_option for j in range(1, swap_tenor + 1)]
    f_T = xad_forward_rate(T_option, maturities, rates_ad)
    
    bonds = [xad_bond_price(r_T, T_option, T_pay, maturities, rates_ad, f_T) 
             for T_pay in swap_times]
    
    # Fixed leg
    fixed_pv = bonds[0] * K_strike
    for b in bonds[1:]:
        fixed_pv = fixed_pv + b * K_strike
    
    # Floating leg
    float_pv = xadj.Real(1.0) - bonds[-1]
    
    # Swap value
    swap_val = (float_pv - fixed_pv) * notional
    
    # Payoff
    payoff = xmath.max(swap_val, xadj.Real(0.0)) if is_payer_swaption else xmath.max(-swap_val, xadj.Real(0.0))
    
    # Discount
    disc = xad_discount(T_option, maturities, rates_ad)
    return disc * payoff


def compute_greeks_xad(maturities, rates, Z):
    """Compute Greeks using XAD AAD."""
    n_rates = len(rates)
    n_steps = Z.shape[1]
    t_grid = np.linspace(0, T_sim, n_steps + 1)
    
    acc_greeks = np.zeros(n_rates)
    acc_price = 0.0
    
    for path_idx in range(num_paths):
        with xadj.Tape() as tape:
            rates_ad = [xadj.Real(float(r)) for r in rates]
            for r_ad in rates_ad:
                tape.registerInput(r_ad)
            tape.newRecording()
            
            # Pre-compute theta on tape
            theta_cache = [xad_theta(t, maturities, rates_ad) for t in t_grid]
            
            # Price path
            payoff = xad_price_single_path(maturities, rates_ad, Z[path_idx, :], theta_cache)
            
            tape.registerOutput(payoff)
            payoff.derivative = 1.0
            tape.computeAdjoints()
            
            acc_price += payoff.value
            for i, r_ad in enumerate(rates_ad):
                acc_greeks[i] += r_ad.derivative
    
    return acc_price / num_paths, acc_greeks / num_paths


def compute_greeks_fd(maturities, rates, Z, bump=1e-4):
    """Compute Greeks using finite differences."""
    base_price = np_price_swaption(maturities, rates, Z)
    
    greeks = np.zeros(len(rates))
    for i in range(len(rates)):
        rates_up = rates.copy()
        rates_up[i] += bump
        price_up = np_price_swaption(maturities, rates_up, Z)
        
        rates_dn = rates.copy()
        rates_dn[i] -= bump
        price_dn = np_price_swaption(maturities, rates_dn, Z)
        
        greeks[i] = (price_up - price_dn) / (2 * bump)
    
    return base_price, greeks


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Hull-White Swaption Greeks: XAD (AAD) vs Finite Differences")
    print("=" * 70)
    
    maturities, rates = get_market_curve()
    n_rates = len(rates)
    Z = Z_matrix_global
    
    print(f"\nMarket Curve ({n_rates} nodes):")
    for i, (m, r) in enumerate(zip(maturities, rates)):
        print(f"  Node {i:2d}: T={m:6.3f}Y, Rate={r*100:.3f}%")
    
    print(f"\nSwaption: {T_option}Y expiry × {swap_tenor}Y swap, Strike={K_strike*100:.2f}%")
    print(f"Notional: ${notional:,}, Type: {'Payer' if is_payer_swaption else 'Receiver'}")
    print(f"Hull-White: a={a}, σ={sigma}")
    print(f"Monte Carlo: {num_paths} paths, {num_steps_global} steps (dt={dt:.4f})")
    
    # XAD Greeks
    print("\n" + "-" * 70)
    print("Computing Greeks with XAD (AAD)...")
    t0 = time.time()
    price_xad, greeks_xad = compute_greeks_xad(maturities, rates, Z)
    time_xad = time.time() - t0
    print(f"XAD completed in {time_xad:.3f}s")
    
    # FD Greeks
    print("\n" + "-" * 70)
    print("Computing Greeks with Finite Differences...")
    t0 = time.time()
    price_fd, greeks_fd = compute_greeks_fd(maturities, rates, Z)
    time_fd = time.time() - t0
    print(f"FD completed in {time_fd:.3f}s")
    
    # Results
    print("\n" + "=" * 70)
    print("PRICE COMPARISON")
    print("=" * 70)
    print(f"  XAD Price:    ${price_xad:,.2f}")
    print(f"  FD Price:     ${price_fd:,.2f}")
    print(f"  Difference:   ${abs(price_xad - price_fd):,.4f} ({abs(price_xad - price_fd)/price_fd*100:.6f}%)")
    
    print("\n" + "=" * 70)
    print("GREEKS COMPARISON (dPrice/dRate)")
    print("=" * 70)
    print(f"{'Node':<5} {'Mat':>6} {'XAD':>14} {'FD':>14} {'Diff':>12} {'Rel%':>10}")
    print("-" * 70)
    
    max_rel = 0.0
    for i in range(n_rates):
        g_xad, g_fd = greeks_xad[i], greeks_fd[i]
        diff = abs(g_xad - g_fd)
        rel = diff / max(abs(g_fd), 1e-10) * 100
        max_rel = max(max_rel, rel) if abs(g_fd) > 1 else max_rel
        print(f"{i:<5} {maturities[i]:>6.3f} {g_xad:>14.2f} {g_fd:>14.2f} {diff:>12.2f} {rel:>9.4f}%")
    
    print("-" * 70)
    print(f"Max Relative Difference (significant Greeks): {max_rel:.4f}%")
    
    print("\n" + "=" * 70)
    print("TIMING ANALYSIS")
    print("=" * 70)
    print(f"  XAD Time:     {time_xad:.3f}s ({num_paths} tape recordings)")
    print(f"  FD Time:      {time_fd:.3f}s ({2*n_rates+1} pricings)")
    print(f"  Ratio:        XAD is {time_xad/time_fd:.1f}x slower (Python overhead)")
    print(f"\n  Note: In C++, XAD typically achieves 2-5x speedup over FD for many inputs.")
    
    return price_xad, price_fd, greeks_xad, greeks_fd


if __name__ == '__main__':
    main()
