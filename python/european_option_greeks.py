"""
European Option Pricing with Monte Carlo Simulation
Greeks Calculation: XAD (AAD) vs Finite Differences Comparison

This script demonstrates:
1. European call/put option pricing via Monte Carlo
2. Greeks calculation using XAD Adjoint Algorithmic Differentiation
3. Greeks calculation using Finite Differences
4. Timing comparison between XAD and FD approaches

For pathwise AAD to work with discontinuous payoffs,
we use soft-max approximation: max(x,0) ≈ (x + sqrt(x² + ε²)) / 2
"""

import numpy as np
import time
from typing import Dict

# Try to import XAD
try:
    import xad.adj_1st as xadj
    import xad.math as xmath
    XAD_AVAILABLE = True
except ImportError:
    print("Warning: XAD not installed. Run: pip install xad")
    XAD_AVAILABLE = False
    xmath = None


# =============================================================================
# Black-Scholes Monte Carlo Pricing (Standard Python/NumPy)
# =============================================================================

def monte_carlo_european_option(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool = True,
    n_paths: int = 100000,
    seed: int = 42
) -> float:
    """
    Price European option using Monte Carlo simulation.
    Uses GBM: S(T) = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
    """
    np.random.seed(seed)
    Z = np.random.standard_normal(n_paths)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    S_T = S0 * np.exp(drift + diffusion)
    
    if is_call:
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)
    
    return np.exp(-r * T) * np.mean(payoffs)


# =============================================================================
# Finite Difference Greeks
# =============================================================================

def compute_greeks_fd(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool = True,
    n_paths: int = 100000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute Greeks using central finite differences.
    """
    dS = S0 * 0.01
    dsigma = 0.01
    dT = 1/365
    dr = 0.0001
    
    V0 = monte_carlo_european_option(S0, K, T, r, sigma, is_call, n_paths, seed)
    
    # Delta and Gamma
    V_up = monte_carlo_european_option(S0 + dS, K, T, r, sigma, is_call, n_paths, seed)
    V_down = monte_carlo_european_option(S0 - dS, K, T, r, sigma, is_call, n_paths, seed)
    delta = (V_up - V_down) / (2 * dS)
    gamma = (V_up - 2*V0 + V_down) / (dS**2)
    
    # Vega
    V_sigma_up = monte_carlo_european_option(S0, K, T, r, sigma + dsigma, is_call, n_paths, seed)
    V_sigma_down = monte_carlo_european_option(S0, K, T, r, sigma - dsigma, is_call, n_paths, seed)
    vega = (V_sigma_up - V_sigma_down) / (2 * dsigma)
    
    # Theta: (V(T-dt) - V(T)) is the change when one day passes
    # Theta is typically expressed as daily P&L, so we DON'T divide by dT
    if T > dT:
        V_T_down = monte_carlo_european_option(S0, K, T - dT, r, sigma, is_call, n_paths, seed)
        theta = V_T_down - V0  # Change in value when 1 day passes
    else:
        theta = 0.0
    
    # Rho
    V_r_up = monte_carlo_european_option(S0, K, T, r + dr, sigma, is_call, n_paths, seed)
    V_r_down = monte_carlo_european_option(S0, K, T, r - dr, sigma, is_call, n_paths, seed)
    rho = (V_r_up - V_r_down) / (2 * dr)
    
    return {
        'price': V0,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }


# =============================================================================
# XAD (AAD) Greeks - Using Soft-Max Approximation
# =============================================================================

def compute_greeks_xad(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool = True,
    n_paths: int = 20000,
    seed: int = 42,
    smoothing: float = 0.001
) -> Dict[str, float]:
    """
    Compute all Greeks in a single Monte Carlo simulation using XAD AAD.
    
    Uses soft-max approximation for differentiability:
    max(x, 0) ≈ (x + sqrt(x² + ε²)) / 2
    
    This computes Delta, Vega, Theta, Rho in ONE backward pass.
    Gamma requires FD on Delta (2 additional simulations).
    """
    if not XAD_AVAILABLE:
        raise ImportError("XAD library not available")
    
    np.random.seed(seed)
    Z = np.random.standard_normal(n_paths)
    eps = smoothing * K
    
    # Create tape and activate
    tape = xadj.Tape()
    tape.activate()
    
    # Create active variables for all inputs we want sensitivities to
    S0_ad = xadj.Real(S0)
    T_ad = xadj.Real(T)
    r_ad = xadj.Real(r)
    sigma_ad = xadj.Real(sigma)
    
    # Register inputs BEFORE calling newRecording
    tape.registerInput(S0_ad)
    tape.registerInput(T_ad)
    tape.registerInput(r_ad)
    tape.registerInput(sigma_ad)
    
    # Start recording operations
    tape.newRecording()
    
    # Pre-compute common terms
    drift = (r_ad - sigma_ad * sigma_ad * 0.5) * T_ad
    sqrt_T = xmath.sqrt(T_ad)
    
    # Accumulate payoffs over all paths
    total_payoff = xadj.Real(0.0)
    
    for z in Z:
        S_T = S0_ad * xmath.exp(drift + sigma_ad * sqrt_T * z)
        
        if is_call:
            x = S_T - K
        else:
            x = K - S_T
        
        # Smooth max approximation
        payoff = (x + xmath.sqrt(x * x + eps * eps)) * 0.5
        total_payoff = total_payoff + payoff
    
    # Discount and average
    discount = xmath.exp(-r_ad * T_ad)
    price = discount * total_payoff / n_paths
    
    # Register output and compute adjoints (backward pass)
    tape.registerOutput(price)
    price.derivative = 1.0
    tape.computeAdjoints()
    
    # Extract Greeks
    price_val = price.value
    delta = S0_ad.derivative
    vega = sigma_ad.derivative
    theta_dT = T_ad.derivative  # dV/dT
    rho = r_ad.derivative
    
    tape.deactivate()
    
    # Theta: as time passes, T decreases, so theta = dV/dT * (-1/365)
    theta = -theta_dT / 365
    
    # Gamma via finite difference on delta
    dS = S0 * 0.01
    delta_up = compute_delta_xad(S0 + dS, K, T, r, sigma, is_call, n_paths, seed, smoothing)
    delta_down = compute_delta_xad(S0 - dS, K, T, r, sigma, is_call, n_paths, seed, smoothing)
    gamma = (delta_up - delta_down) / (2 * dS)
    
    return {
        'price': price_val,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }


def compute_delta_xad(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool = True,
    n_paths: int = 20000,
    seed: int = 42,
    smoothing: float = 0.001
) -> float:
    """Compute only delta for gamma calculation."""
    if not XAD_AVAILABLE:
        return 0.0
    
    np.random.seed(seed)
    Z = np.random.standard_normal(n_paths)
    eps = smoothing * K
    
    tape = xadj.Tape()
    tape.activate()
    
    S0_ad = xadj.Real(S0)
    T_ad = xadj.Real(T)
    r_ad = xadj.Real(r)
    sigma_ad = xadj.Real(sigma)
    
    tape.registerInput(S0_ad)
    tape.newRecording()
    
    drift = (r_ad - sigma_ad * sigma_ad * 0.5) * T_ad
    sqrt_T = xmath.sqrt(T_ad)
    
    total_payoff = xadj.Real(0.0)
    
    for z in Z:
        S_T = S0_ad * xmath.exp(drift + sigma_ad * sqrt_T * z)
        
        if is_call:
            x = S_T - K
        else:
            x = K - S_T
        
        payoff = (x + xmath.sqrt(x * x + eps * eps)) * 0.5
        total_payoff = total_payoff + payoff
    
    discount = xmath.exp(-r_ad * T_ad)
    price = discount * total_payoff / n_paths
    
    tape.registerOutput(price)
    price.derivative = 1.0
    tape.computeAdjoints()
    
    delta = S0_ad.derivative
    tape.deactivate()
    
    return delta


# =============================================================================
# Black-Scholes Analytical (for validation)
# =============================================================================

def black_scholes_analytical(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool = True
) -> Dict[str, float]:
    """Black-Scholes analytical solution for European options."""
    from scipy.stats import norm
    
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if is_call:
        price = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
    
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    vega = S0 * norm.pdf(d1) * np.sqrt(T)
    
    theta_call = (-S0 * norm.pdf(d1) * sigma / (2*np.sqrt(T)) 
                  - r * K * np.exp(-r*T) * norm.cdf(d2))
    theta_put = (-S0 * norm.pdf(d1) * sigma / (2*np.sqrt(T)) 
                 + r * K * np.exp(-r*T) * norm.cdf(-d2))
    theta = (theta_call if is_call else theta_put) / 365
    
    if is_call:
        rho = K * T * np.exp(-r*T) * norm.cdf(d2)
    else:
        rho = -K * T * np.exp(-r*T) * norm.cdf(-d2)
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }


# =============================================================================
# Main Comparison
# =============================================================================

def run_comparison():
    """Run full comparison of XAD vs Finite Differences."""
    
    print("=" * 80)
    print("European Option Greeks: XAD (AAD) vs Finite Differences Comparison")
    print("=" * 80)
    
    # Option parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.20
    is_call = True
    n_paths_fd = 50000
    n_paths_xad = 20000  # Smaller for XAD due to tape memory
    seed = 42
    
    print(f"\nOption Parameters:")
    print(f"  Spot (S0):     ${S0:.2f}")
    print(f"  Strike (K):    ${K:.2f}")
    print(f"  Maturity (T):  {T:.2f} years")
    print(f"  Rate (r):      {r*100:.2f}%")
    print(f"  Volatility:    {sigma*100:.2f}%")
    print(f"  Option Type:   {'Call' if is_call else 'Put'}")
    print(f"  FD MC Paths:   {n_paths_fd:,}")
    print(f"  XAD MC Paths:  {n_paths_xad:,}")
    
    # Analytical Solution
    print("\n" + "-" * 80)
    print("Analytical Black-Scholes (Benchmark)")
    print("-" * 80)
    
    try:
        bs = black_scholes_analytical(S0, K, T, r, sigma, is_call)
        print(f"  Price:  ${bs['price']:.4f}")
        print(f"  Delta:   {bs['delta']:.6f}")
        print(f"  Gamma:   {bs['gamma']:.6f}")
        print(f"  Vega:    {bs['vega']:.4f}")
        print(f"  Theta:  ${bs['theta']:.4f}/day")
        print(f"  Rho:     {bs['rho']:.4f}")
        has_analytical = True
    except ImportError:
        print("  scipy not available")
        bs = None
        has_analytical = False
    
    # Finite Differences
    print("\n" + "-" * 80)
    print("Finite Differences Method")
    print("-" * 80)
    
    start_time = time.time()
    fd_results = compute_greeks_fd(S0, K, T, r, sigma, is_call, n_paths_fd, seed)
    fd_time = time.time() - start_time
    
    print(f"  Price:  ${fd_results['price']:.4f}")
    print(f"  Delta:   {fd_results['delta']:.6f}")
    print(f"  Gamma:   {fd_results['gamma']:.6f}")
    print(f"  Vega:    {fd_results['vega']:.4f}")
    print(f"  Theta:  ${fd_results['theta']:.4f}/day")
    print(f"  Rho:     {fd_results['rho']:.4f}")
    print(f"\n  Time:    {fd_time:.4f} seconds")
    print(f"  MC runs: 9 (base + 2 per Greek)")
    
    # XAD AAD
    if XAD_AVAILABLE:
        print("\n" + "-" * 80)
        print("XAD Adjoint Algorithmic Differentiation")
        print("-" * 80)
        
        start_time = time.time()
        xad_results = compute_greeks_xad(S0, K, T, r, sigma, is_call, n_paths_xad, seed)
        xad_time = time.time() - start_time
        
        print(f"  Price:  ${xad_results['price']:.4f}")
        print(f"  Delta:   {xad_results['delta']:.6f}")
        print(f"  Gamma:   {xad_results['gamma']:.6f}")
        print(f"  Vega:    {xad_results['vega']:.4f}")
        print(f"  Theta:  ${xad_results['theta']:.4f}/day")
        print(f"  Rho:     {xad_results['rho']:.4f}")
        print(f"\n  Time:    {xad_time:.4f} seconds")
        print(f"  MC runs: 3 (1 base with 4 Greeks + 2 for gamma)")
    else:
        xad_results = None
        xad_time = None
    
    # Comparison Table
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    print("\n┌─────────────┬────────────────┬────────────────┬────────────────┬─────────────┐")
    print("│   Greek     │   Analytical   │      FD        │      XAD       │  XAD Error  │")
    print("├─────────────┼────────────────┼────────────────┼────────────────┼─────────────┤")
    
    greeks = ['price', 'delta', 'gamma', 'vega', 'theta', 'rho']
    labels = ['Price', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
    
    for greek, label in zip(greeks, labels):
        analytical_val = bs[greek] if has_analytical else float('nan')
        fd_val = fd_results[greek]
        xad_val = xad_results[greek] if xad_results else float('nan')
        
        if has_analytical and abs(analytical_val) > 1e-10:
            error_pct = abs((xad_val - analytical_val) / analytical_val) * 100
            error_str = f"{error_pct:.2f}%"
        else:
            error_str = "N/A"
        
        print(f"│ {label:11} │ {analytical_val:14.6f} │ {fd_val:14.6f} │ {xad_val:14.6f} │ {error_str:11} │")
    
    print("└─────────────┴────────────────┴────────────────┴────────────────┴─────────────┘")
    
    # Timing Analysis
    print("\n" + "-" * 80)
    print("TIMING ANALYSIS")
    print("-" * 80)
    
    print(f"\n  Finite Differences:  {fd_time:.4f} seconds ({n_paths_fd:,} paths)")
    if xad_time:
        print(f"  XAD AAD:             {xad_time:.4f} seconds ({n_paths_xad:,} paths)")
        
        # Normalized comparison (per 10k paths)
        fd_per_10k = fd_time / (n_paths_fd / 10000) * 9  # 9 simulations
        xad_per_10k = xad_time / (n_paths_xad / 10000) * 3  # 3 simulations
        
        print(f"\n  Normalized (per 10k paths):")
        print(f"    FD:  {fd_per_10k:.4f}s for 9 simulations")
        print(f"    XAD: {xad_per_10k:.4f}s for 3 simulations")
        
        print(f"\n  Key Advantages of XAD AAD:")
        print(f"    • Computes Delta, Vega, Theta, Rho in ONE backward pass")
        print(f"    • O(1) cost regardless of number of input sensitivities")
        print(f"    • For 100+ Greeks (e.g., curve nodes), XAD is ~10-50x faster")
        print(f"    • Exact derivatives (no truncation error from bump size)")
    
    # Accuracy Analysis
    if has_analytical and xad_results:
        print("\n" + "-" * 80)
        print("ACCURACY COMPARISON (vs Analytical)")
        print("-" * 80)
        
        print("\n  ┌───────────┬──────────────┬──────────────┐")
        print("  │   Greek   │   FD Error   │  XAD Error   │")
        print("  ├───────────┼──────────────┼──────────────┤")
        
        for greek, label in zip(greeks, labels):
            if abs(bs[greek]) > 1e-10:
                fd_error = abs((fd_results[greek] - bs[greek]) / bs[greek]) * 100
                xad_error = abs((xad_results[greek] - bs[greek]) / bs[greek]) * 100
                print(f"  │ {label:9} │ {fd_error:10.2f}% │ {xad_error:10.2f}% │")
        
        print("  └───────────┴──────────────┴──────────────┘")
        
        print("\n  Notes:")
        print("    • XAD uses soft-max approximation (small smoothing bias)")
        print("    • FD Theta error due to discrete time step")
        print("    • Both methods have MC sampling noise")
    
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)
    
    return fd_results, xad_results


if __name__ == "__main__":
    run_comparison()
