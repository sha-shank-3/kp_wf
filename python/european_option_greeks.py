"""
European Option Pricing with Monte Carlo Simulation
Greeks Calculation: XAD (AAD) vs Finite Differences Comparison

This script demonstrates:
1. European call/put option pricing via Monte Carlo
2. Greeks calculation using XAD Adjoint Algorithmic Differentiation
3. Greeks calculation using Finite Differences
4. Timing comparison between XAD and FD approaches

Key: Each MC path uses its own tape, with adjoints computed per-path
and accumulated across all paths (following the swaption pattern).
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
    n_paths: int = 10000,
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
    n_paths: int = 10000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute Greeks using central finite differences.
    Each Greek requires 2 additional MC simulations (up and down bump).
    Total: 9 MC simulations for 5 Greeks (using base for Gamma center).
    """
    dS = S0 * 0.01
    dsigma = 0.01
    dT = 1/365
    dr = 0.0001
    
    # Base price
    V0 = monte_carlo_european_option(S0, K, T, r, sigma, is_call, n_paths, seed)
    
    # Delta and Gamma (2 simulations)
    V_up = monte_carlo_european_option(S0 + dS, K, T, r, sigma, is_call, n_paths, seed)
    V_down = monte_carlo_european_option(S0 - dS, K, T, r, sigma, is_call, n_paths, seed)
    delta = (V_up - V_down) / (2 * dS)
    gamma = (V_up - 2*V0 + V_down) / (dS**2)
    
    # Vega (2 simulations)
    V_sigma_up = monte_carlo_european_option(S0, K, T, r, sigma + dsigma, is_call, n_paths, seed)
    V_sigma_down = monte_carlo_european_option(S0, K, T, r, sigma - dsigma, is_call, n_paths, seed)
    vega = (V_sigma_up - V_sigma_down) / (2 * dsigma)
    
    # Theta (1 simulation - forward difference)
    if T > dT:
        V_T_down = monte_carlo_european_option(S0, K, T - dT, r, sigma, is_call, n_paths, seed)
        theta = V_T_down - V0  # Daily P&L as time passes
    else:
        theta = 0.0
    
    # Rho (2 simulations)
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
# XAD (AAD) Greeks - Per-Path Tape Pattern
# =============================================================================

def price_single_path_xad(
    S0_ad,      # XAD active Real
    T_ad,       # XAD active Real  
    r_ad,       # XAD active Real
    sigma_ad,   # XAD active Real
    K: float,
    z: float,   # Single random number for this path
    is_call: bool,
    smoothing: float = 0.001
):
    """
    Price a single path using XAD active types.
    Returns discounted payoff as XAD Real.
    
    Uses soft-max approximation: max(x,0) ≈ (x + sqrt(x² + ε²)) / 2
    """
    eps = smoothing * K
    
    # GBM simulation
    drift = (r_ad - sigma_ad * sigma_ad * 0.5) * T_ad
    sqrt_T = xmath.sqrt(T_ad)
    S_T = S0_ad * xmath.exp(drift + sigma_ad * sqrt_T * z)
    
    # Payoff with soft-max
    if is_call:
        x = S_T - K
    else:
        x = K - S_T
    
    # Smooth max approximation for differentiability
    payoff = (x + xmath.sqrt(x * x + eps * eps)) * 0.5
    
    # Discount
    discount = xmath.exp(-r_ad * T_ad)
    
    return discount * payoff


def compute_greeks_xad(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool = True,
    n_paths: int = 10000,
    seed: int = 42,
    smoothing: float = 0.001
) -> Dict[str, float]:
    """
    Compute Greeks using XAD AAD with per-path tape pattern.
    
    For each path:
    1. Create a new tape
    2. Create fresh active variables
    3. Register inputs, compute price, compute adjoints
    4. Accumulate derivatives
    
    This gives us Delta, Vega, Theta, Rho in n_paths forward+backward passes.
    Gamma requires FD on Delta (2 additional full computations).
    """
    if not XAD_AVAILABLE:
        raise ImportError("XAD library not available")
    
    np.random.seed(seed)
    Z = np.random.standard_normal(n_paths)
    
    # Accumulators for price and Greeks
    acc_price = 0.0
    acc_delta = 0.0
    acc_vega = 0.0
    acc_theta = 0.0  # dV/dT
    acc_rho = 0.0
    
    # Process each path with its own tape
    for path_idx in range(n_paths):
        with xadj.Tape() as tape:
            # Create fresh active variables for this path
            S0_ad = xadj.Real(S0)
            T_ad = xadj.Real(T)
            r_ad = xadj.Real(r)
            sigma_ad = xadj.Real(sigma)
            
            # Register all inputs
            tape.registerInput(S0_ad)
            tape.registerInput(T_ad)
            tape.registerInput(r_ad)
            tape.registerInput(sigma_ad)
            
            # Start recording operations
            tape.newRecording()
            
            # Compute discounted payoff for this path
            payoff = price_single_path_xad(
                S0_ad, T_ad, r_ad, sigma_ad, K, Z[path_idx], is_call, smoothing
            )
            
            # Register output and compute adjoints
            tape.registerOutput(payoff)
            payoff.derivative = 1.0
            tape.computeAdjoints()
            
            # Accumulate
            acc_price += payoff.value
            acc_delta += S0_ad.derivative
            acc_vega += sigma_ad.derivative
            acc_theta += T_ad.derivative
            acc_rho += r_ad.derivative
    
    # Average across paths
    price = acc_price / n_paths
    delta = acc_delta / n_paths
    vega = acc_vega / n_paths
    theta_dT = acc_theta / n_paths
    rho = acc_rho / n_paths
    
    # Convert theta: dV/dT → daily P&L (as time passes, T decreases)
    theta = -theta_dT / 365
    
    # Gamma via finite difference on delta
    dS = S0 * 0.01
    delta_up = compute_delta_xad(S0 + dS, K, T, r, sigma, is_call, n_paths, seed, smoothing)
    delta_down = compute_delta_xad(S0 - dS, K, T, r, sigma, is_call, n_paths, seed, smoothing)
    gamma = (delta_up - delta_down) / (2 * dS)
    
    return {
        'price': price,
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
    n_paths: int = 10000,
    seed: int = 42,
    smoothing: float = 0.001
) -> float:
    """Compute only delta for gamma calculation (per-path tape pattern)."""
    if not XAD_AVAILABLE:
        return 0.0
    
    np.random.seed(seed)
    Z = np.random.standard_normal(n_paths)
    
    acc_delta = 0.0
    
    for path_idx in range(n_paths):
        with xadj.Tape() as tape:
            S0_ad = xadj.Real(S0)
            T_ad = xadj.Real(T)
            r_ad = xadj.Real(r)
            sigma_ad = xadj.Real(sigma)
            
            tape.registerInput(S0_ad)
            tape.newRecording()
            
            payoff = price_single_path_xad(
                S0_ad, T_ad, r_ad, sigma_ad, K, Z[path_idx], is_call, smoothing
            )
            
            tape.registerOutput(payoff)
            payoff.derivative = 1.0
            tape.computeAdjoints()
            
            acc_delta += S0_ad.derivative
    
    return acc_delta / n_paths


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
    n_paths = 10000   # MC paths for both methods
    seed = 42
    
    print(f"\nOption Parameters:")
    print(f"  Spot (S0):     ${S0:.2f}")
    print(f"  Strike (K):    ${K:.2f}")
    print(f"  Maturity (T):  {T:.2f} years")
    print(f"  Rate (r):      {r*100:.2f}%")
    print(f"  Volatility:    {sigma*100:.2f}%")
    print(f"  Option Type:   {'Call' if is_call else 'Put'}")
    print(f"  MC Paths:      {n_paths:,}")
    
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
    fd_results = compute_greeks_fd(S0, K, T, r, sigma, is_call, n_paths, seed)
    fd_time = time.time() - start_time
    
    print(f"  Price:  ${fd_results['price']:.4f}")
    print(f"  Delta:   {fd_results['delta']:.6f}")
    print(f"  Gamma:   {fd_results['gamma']:.6f}")
    print(f"  Vega:    {fd_results['vega']:.4f}")
    print(f"  Theta:  ${fd_results['theta']:.4f}/day")
    print(f"  Rho:     {fd_results['rho']:.4f}")
    print(f"\n  Time:    {fd_time:.4f} seconds")
    print(f"  MC runs: 9 simulations × {n_paths:,} paths each")
    
    # XAD AAD
    if XAD_AVAILABLE:
        print("\n" + "-" * 80)
        print("XAD Adjoint Algorithmic Differentiation (Per-Path Tape)")
        print("-" * 80)
        
        start_time = time.time()
        xad_results = compute_greeks_xad(S0, K, T, r, sigma, is_call, n_paths, seed)
        xad_time = time.time() - start_time
        
        print(f"  Price:  ${xad_results['price']:.4f}")
        print(f"  Delta:   {xad_results['delta']:.6f}")
        print(f"  Gamma:   {xad_results['gamma']:.6f}")
        print(f"  Vega:    {xad_results['vega']:.4f}")
        print(f"  Theta:  ${xad_results['theta']:.4f}/day")
        print(f"  Rho:     {xad_results['rho']:.4f}")
        print(f"\n  Time:    {xad_time:.4f} seconds")
        print(f"  MC runs: 3 simulations × {n_paths:,} paths (1 base + 2 for gamma)")
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
    
    print(f"\n  Finite Differences:  {fd_time:.4f} seconds")
    if xad_time:
        print(f"  XAD AAD:             {xad_time:.4f} seconds")
        
        if fd_time > xad_time:
            speedup = fd_time / xad_time
            print(f"\n  ✓ XAD is {speedup:.2f}x FASTER than Finite Differences")
        else:
            ratio = xad_time / fd_time
            print(f"\n  FD is {ratio:.1f}x faster (XAD per-path tape has overhead)")
        
        print(f"\n  Computation breakdown:")
        print(f"    FD:  9 simulations × {n_paths:,} paths = {9*n_paths:,} path evaluations")
        print(f"    XAD: 3 simulations × {n_paths:,} paths = {3*n_paths:,} path evaluations")
        print(f"         (1 base gives Delta,Vega,Theta,Rho + 2 for Gamma)")
        
        print(f"\n  XAD advantages for complex models:")
        print(f"    • For 100+ Greeks (curve nodes), XAD scales O(1) vs FD O(n)")
        print(f"    • Exact derivatives (no bump size selection)")
        print(f"    • All first-order sensitivities in one backward pass")
    
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
        print("    • Both methods use same random numbers for fair comparison")
        print("    • MC sampling noise is the main error source")
    
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)
    
    return fd_results, xad_results


if __name__ == "__main__":
    run_comparison()
