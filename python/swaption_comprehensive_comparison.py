# -*- coding: utf-8 -*-
"""
Comprehensive Swaption Comparison: Pricing and Greeks
======================================================

This script compares 5 different swaptions across:

PRICING:
1. Analytical (Jamshidian Decomposition)
2. Monte Carlo
3. Monte Carlo + XAD

GREEKS:
1. Analytical (closed-form)
2. Finite Difference (bump-and-reprice)
3. XAD + AAD

Greeks computed:
- Discount curve sensitivities (DV01 by node)
- Hull-White parameters (a, sigma)
- ATM vol surface nodes

Uses Least Squares calibration for Hull-White parameters.

Author: Swaption Comparison Framework
Date: January 2026
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import xad.adj_1st as xadj
import xad.math as xmath
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Global Configuration
# =============================================================================

np.random.seed(42)

@dataclass
class SwaptionSpec:
    """Specification for a swaption"""
    name: str
    T_option: float  # Option expiry
    swap_tenor: int  # Swap tenor in years
    K_strike: Optional[float]  # Strike (None = ATM)
    notional: float = 1_000_000
    is_payer: bool = True

# 5 Different Swaptions to compare
SWAPTIONS = [
    SwaptionSpec("1Yx5Y", 1.0, 5, None),
    SwaptionSpec("2Yx3Y", 2.0, 3, None),
    SwaptionSpec("3Yx7Y", 3.0, 7, None),
    SwaptionSpec("5Yx10Y", 5.0, 10, None),
    SwaptionSpec("10Yx5Y", 10.0, 5, None),
]

# Monte Carlo Parameters
MC_PATHS = 20000
MC_DT = 0.005  # Finer time step for better convergence

# =============================================================================
# Market Data: OIS Curve
# =============================================================================

def get_ois_curve() -> Tuple[np.ndarray, np.ndarray]:
    """Real SOFR OIS curve from December 2024"""
    maturities = np.array([
        1/360, 7/360, 1/12, 2/12, 3/12, 6/12, 9/12,
        1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0
    ])
    rates = np.array([
        0.0433, 0.0432, 0.0430, 0.0426, 0.0422, 0.0413, 0.0407,
        0.0403, 0.0394, 0.0388, 0.0383, 0.0387, 0.0395, 0.0408,
        0.0423, 0.0427, 0.0430, 0.0433, 0.0436, 0.0438
    ])
    return maturities, rates


# =============================================================================
# ATM Vol Surface (Normal vols in bps)
# =============================================================================

def get_vol_surface() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ATM Normal vol surface - December 2024"""
    expiries = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0])
    tenors = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])
    
    # Normal vols in bps
    vols_bps = np.array([
        [45, 48, 51, 55, 58, 62, 65, 67, 70],   # 3M
        [48, 52, 55, 59, 62, 66, 69, 71, 74],   # 6M
        [52, 56, 59, 63, 66, 70, 73, 75, 78],   # 1Y
        [58, 62, 65, 69, 72, 76, 79, 81, 84],   # 2Y
        [62, 66, 69, 73, 76, 80, 83, 85, 88],   # 3Y
        [68, 72, 75, 79, 82, 86, 89, 91, 94],   # 5Y
        [72, 76, 79, 83, 86, 90, 93, 95, 98],   # 7Y
        [78, 82, 85, 89, 92, 96, 99, 101, 104], # 10Y
        [82, 86, 89, 93, 96, 100, 103, 105, 108] # 20Y
    ], dtype=float)
    
    vols = vols_bps / 10000.0  # Convert to absolute
    return expiries, tenors, vols


# =============================================================================
# Interpolation Utilities
# =============================================================================

def linear_interp(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    """Linear interpolation with extrapolation"""
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    idx = np.searchsorted(xs, x) - 1
    idx = max(0, min(idx, len(xs) - 2))
    w = (x - xs[idx]) / (xs[idx + 1] - xs[idx])
    return float(ys[idx] * (1 - w) + ys[idx + 1] * w)


def xad_linear_interp(x: float, xs: np.ndarray, ys_ad: list):
    """XAD-compatible linear interpolation"""
    if x <= xs[0]:
        return ys_ad[0]
    if x >= xs[-1]:
        return ys_ad[-1]
    idx = int(np.searchsorted(xs, x)) - 1
    idx = max(0, min(idx, len(xs) - 2))
    w = (x - xs[idx]) / (xs[idx + 1] - xs[idx])
    return ys_ad[idx] * (1 - w) + ys_ad[idx + 1] * w


def bilinear_interp(x: float, y: float, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> float:
    """Bilinear interpolation for vol surface"""
    # Find indices
    ix = max(0, min(np.searchsorted(xs, x) - 1, len(xs) - 2))
    iy = max(0, min(np.searchsorted(ys, y) - 1, len(ys) - 2))
    
    wx = (x - xs[ix]) / (xs[ix + 1] - xs[ix]) if xs[ix + 1] != xs[ix] else 0
    wy = (y - ys[iy]) / (ys[iy + 1] - ys[iy]) if ys[iy + 1] != ys[iy] else 0
    
    wx = max(0, min(wx, 1))
    wy = max(0, min(wy, 1))
    
    z00 = zs[ix, iy]
    z01 = zs[ix, iy + 1]
    z10 = zs[ix + 1, iy]
    z11 = zs[ix + 1, iy + 1]
    
    return (1 - wx) * (1 - wy) * z00 + wx * (1 - wy) * z10 + (1 - wx) * wy * z01 + wx * wy * z11


# =============================================================================
# Curve Functions
# =============================================================================

def discount(T: float, mats: np.ndarray, rates: np.ndarray) -> float:
    """Discount factor P(0, T)"""
    if T <= 0:
        return 1.0
    rate = linear_interp(T, mats, rates)
    return np.exp(-rate * T)


def forward_rate(t: float, mats: np.ndarray, rates: np.ndarray, h: float = 1e-5) -> float:
    """Instantaneous forward rate f(0, t)"""
    t = max(t, h)
    P_t = discount(t, mats, rates)
    P_th = discount(t + h, mats, rates)
    return -(np.log(P_th) - np.log(P_t)) / h


def swap_annuity(start: float, tenor: int, mats: np.ndarray, rates: np.ndarray) -> float:
    """Sum of discount factors for fixed leg payments"""
    annuity = 0.0
    for i in range(1, tenor + 1):
        annuity += discount(start + i, mats, rates)
    return annuity


def forward_swap_rate(start: float, tenor: int, mats: np.ndarray, rates: np.ndarray) -> float:
    """Forward swap rate"""
    df_start = discount(start, mats, rates)
    df_end = discount(start + tenor, mats, rates)
    A = swap_annuity(start, tenor, mats, rates)
    return (df_start - df_end) / A if A > 0 else 0.0


# =============================================================================
# Black (Bachelier) Swaption Pricing
# =============================================================================

def black_swaption_price(forward: float, strike: float, expiry: float,
                          sigma_n: float, annuity: float, notional: float,
                          is_payer: bool) -> float:
    """Bachelier (Normal vol) swaption price"""
    if expiry <= 0 or sigma_n <= 0:
        return 0.0
    
    sqrt_T = np.sqrt(expiry)
    d = (forward - strike) / (sigma_n * sqrt_T)
    omega = 1.0 if is_payer else -1.0
    
    price = annuity * sigma_n * sqrt_T * (omega * d * norm.cdf(omega * d) + norm.pdf(d))
    return notional * price


# =============================================================================
# Hull-White Model Functions
# =============================================================================

def HW_B(a: float, t: float, T: float) -> float:
    """Hull-White B(t, T) function"""
    tau = T - t
    if abs(a) < 1e-10:
        return tau
    return (1.0 - np.exp(-a * tau)) / a


def HW_logA(a: float, sigma: float, t: float, T: float, 
            mats: np.ndarray, rates: np.ndarray) -> float:
    """Hull-White log(A(t, T)) function"""
    B = HW_B(a, t, T)
    P_T = discount(T, mats, rates)
    P_t = discount(t, mats, rates)
    f_t = forward_rate(t, mats, rates)
    
    sigma2 = sigma * sigma
    term = sigma2 / (4.0 * a) * (1.0 - np.exp(-2.0 * a * t)) * B * B
    
    return np.log(P_T) - np.log(P_t) + B * f_t - term


def HW_bond_price(r_t: float, t: float, T: float, a: float, sigma: float,
                  mats: np.ndarray, rates: np.ndarray) -> float:
    """Hull-White zero-coupon bond price P(t, T) given short rate r_t"""
    if T <= t:
        return 1.0
    B = HW_B(a, t, T)
    logA = HW_logA(a, sigma, t, T, mats, rates)
    return np.exp(logA - B * r_t)


def HW_bond_price_vec(r_t: np.ndarray, t: float, T: float, a: float, sigma: float,
                       mats: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Vectorized Hull-White bond price"""
    if T <= t:
        return np.ones_like(r_t)
    B = HW_B(a, t, T)
    logA = HW_logA(a, sigma, t, T, mats, rates)
    return np.exp(logA - B * r_t)


def HW_theta(t: float, a: float, sigma: float, mats: np.ndarray, rates: np.ndarray) -> float:
    """Hull-White theta(t) for calibration to term structure"""
    h = 1e-5
    f_t = forward_rate(t, mats, rates, h)
    f_th = forward_rate(t + h, mats, rates, h)
    f_prime = (f_th - f_t) / h
    term3 = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t))
    return f_prime + a * f_t + term3


# =============================================================================
# Jamshidian Decomposition for Analytical Swaption Pricing
# =============================================================================

def find_critical_rate(a: float, sigma: float, T_option: float,
                        payment_times: List[float], coupons: List[float],
                        mats: np.ndarray, rates: np.ndarray) -> float:
    """Find critical rate r* for Jamshidian decomposition using Newton-Raphson"""
    r_star = forward_swap_rate(T_option, len(payment_times), mats, rates)
    
    for _ in range(50):
        f = -1.0
        df = 0.0
        
        for i, (T_pay, c) in enumerate(zip(payment_times, coupons)):
            P = HW_bond_price(r_star, T_option, T_pay, a, sigma, mats, rates)
            B = HW_B(a, T_option, T_pay)
            f += c * P
            df -= c * P * B
        
        if abs(df) < 1e-15:
            break
        delta = f / df
        r_star -= delta
        
        if abs(delta) < 1e-12:
            break
    
    return r_star


def HW_bond_option(a: float, sigma: float, t: float, T: float, S: float,
                    K: float, is_call: bool, mats: np.ndarray, rates: np.ndarray) -> float:
    """Bond option price under Hull-White (closed-form)"""
    if T <= t:
        return 0.0
    
    P_T = discount(T, mats, rates)
    P_S = discount(S, mats, rates)
    
    B_TS = HW_B(a, T, S)
    sigma_p = sigma * np.sqrt((1.0 - np.exp(-2.0 * a * (T - t))) / (2.0 * a)) * B_TS
    
    if sigma_p < 1e-10:
        return max(P_S - K * P_T, 0.0) if is_call else max(K * P_T - P_S, 0.0)
    
    d1 = np.log(P_S / (K * P_T)) / sigma_p + 0.5 * sigma_p
    d2 = d1 - sigma_p
    
    if is_call:
        return P_S * norm.cdf(d1) - K * P_T * norm.cdf(d2)
    else:
        return K * P_T * norm.cdf(-d2) - P_S * norm.cdf(-d1)


def analytical_swaption_price_jamshidian(a: float, sigma: float, T_option: float,
                                          swap_tenor: int, strike: float,
                                          notional: float, is_payer: bool,
                                          mats: np.ndarray, rates: np.ndarray) -> float:
    """Jamshidian decomposition for Hull-White swaption pricing"""
    # Payment times and coupons
    payment_times = [T_option + i for i in range(1, swap_tenor + 1)]
    coupons = [strike] * (swap_tenor - 1) + [1.0 + strike]
    
    # Find critical rate r*
    r_star = find_critical_rate(a, sigma, T_option, payment_times, coupons, mats, rates)
    
    # Strike prices for bond options
    K_bonds = [HW_bond_price(r_star, T_option, T_pay, a, sigma, mats, rates) 
               for T_pay in payment_times]
    
    # Sum of bond options
    price = 0.0
    for i, T_pay in enumerate(payment_times):
        opt = HW_bond_option(a, sigma, 0.0, T_option, T_pay, K_bonds[i],
                              not is_payer, mats, rates)
        price += coupons[i] * opt
    
    return notional * price


# =============================================================================
# Monte Carlo Swaption Pricing
# =============================================================================

def simulate_short_rate_paths(a: float, sigma: float, T_sim: float,
                               mats: np.ndarray, rates: np.ndarray,
                               num_paths: int, dt: float,
                               Z: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate short rate paths using Hull-White model"""
    num_steps = int(T_sim / dt)
    t_grid = np.linspace(0, T_sim, num_steps + 1)
    
    if Z is None:
        Z = np.random.standard_normal((num_paths, num_steps))
    
    r_paths = np.zeros((num_paths, num_steps + 1))
    r_paths[:, 0] = forward_rate(0.0, mats, rates)
    
    sqrt_dt = np.sqrt(dt)
    
    for i in range(num_steps):
        theta_t = HW_theta(t_grid[i], a, sigma, mats, rates)
        dr = (theta_t - a * r_paths[:, i]) * dt + sigma * sqrt_dt * Z[:, i]
        r_paths[:, i + 1] = r_paths[:, i] + dr
    
    return t_grid, r_paths


def mc_swaption_price(a: float, sigma: float, T_option: float, swap_tenor: int,
                       strike: float, notional: float, is_payer: bool,
                       mats: np.ndarray, rates: np.ndarray,
                       num_paths: int = MC_PATHS, dt: float = MC_DT,
                       Z: Optional[np.ndarray] = None) -> float:
    """Monte Carlo swaption pricing"""
    # Simulate paths
    _, r_paths = simulate_short_rate_paths(a, sigma, T_option, mats, rates, num_paths, dt, Z)
    r_T = r_paths[:, -1]
    
    # Calculate swap value at expiry
    payment_times = [T_option + i for i in range(1, swap_tenor + 1)]
    
    # Fixed leg PV
    fixed_pv = np.zeros(num_paths)
    for T_pay in payment_times:
        bonds = HW_bond_price_vec(r_T, T_option, T_pay, a, sigma, mats, rates)
        fixed_pv += strike * bonds
    
    # Floating leg PV = 1 - P(T_option, T_final)
    final_bonds = HW_bond_price_vec(r_T, T_option, payment_times[-1], a, sigma, mats, rates)
    float_pv = 1.0 - final_bonds
    
    # Swap value
    swap_values = (float_pv - fixed_pv) * notional
    
    # Payoff
    if is_payer:
        payoffs = np.maximum(swap_values, 0)
    else:
        payoffs = np.maximum(-swap_values, 0)
    
    # Discount and average
    disc = discount(T_option, mats, rates)
    return disc * np.mean(payoffs)


# =============================================================================
# Least Squares Calibration
# =============================================================================

def calibrate_hull_white_least_squares(mats: np.ndarray, rates: np.ndarray,
                                        vol_expiries: np.ndarray, vol_tenors: np.ndarray,
                                        vol_surface: np.ndarray,
                                        calib_instruments: List[Tuple[int, int]]) -> Tuple[float, float, float]:
    """Calibrate Hull-White (a, sigma) by minimizing squared price errors"""
    
    def objective(params):
        a, sigma = params
        if a <= 0 or sigma <= 0:
            return 1e10
        
        sse = 0.0
        for ei, ti in calib_instruments:
            expiry = vol_expiries[ei]
            tenor = int(vol_tenors[ti])
            
            # Market price from Black formula
            sigma_n = vol_surface[ei, ti]
            F = forward_swap_rate(expiry, tenor, mats, rates)
            A = swap_annuity(expiry, tenor, mats, rates)
            market_price = black_swaption_price(F, F, expiry, sigma_n, A, 1.0, True)
            
            # Model price from Jamshidian
            model_price = analytical_swaption_price_jamshidian(a, sigma, expiry, tenor, F,
                                                                1.0, True, mats, rates)
            
            diff = model_price - market_price
            sse += diff * diff
        
        return sse
    
    # Optimize
    result = minimize(objective, x0=[0.05, 0.01], 
                      bounds=[(0.001, 0.5), (0.001, 0.05)],
                      method='L-BFGS-B')
    
    a_opt, sigma_opt = result.x
    rmse = np.sqrt(result.fun / len(calib_instruments))
    
    return a_opt, sigma_opt, rmse


def compute_vol_surface_greeks_ift(a: float, sigma: float, T_option: float, swap_tenor: int,
                                    strike: float, notional: float, is_payer: bool,
                                    mats: np.ndarray, rates: np.ndarray,
                                    vol_expiries: np.ndarray, vol_tenors: np.ndarray,
                                    vol_surface: np.ndarray,
                                    calib_instruments: List[Tuple[int, int]],
                                    bump: float = 1e-5) -> np.ndarray:
    """
    Compute swaption price sensitivity to ATM vol surface nodes using IFT.
    
    The Implicit Function Theorem gives:
    dP/d(vol_i) = dP/d(theta) * dtheta/d(vol_i)
    
    where theta = (a, sigma) are calibrated parameters.
    
    At the calibrated optimum:
    dg/dtheta = 0 (first order condition)
    
    So dtheta/d(vol_i) = -(d²g/dtheta²)^(-1) * (d²g/dtheta d(vol_i))
    
    where g is the calibration objective function.
    """
    n_vols = len(calib_instruments)
    
    # Compute dP/da and dP/dsigma (price Greeks w.r.t. HW params)
    dP_da, dP_dsigma = analytical_hw_greeks(a, sigma, T_option, swap_tenor,
                                             strike, notional, is_payer, mats, rates)
    
    # Compute Hessian of calibration objective w.r.t. (a, sigma)
    h = bump
    
    def calib_obj(params):
        a_, s_ = params
        sse = 0.0
        for ei, ti in calib_instruments:
            expiry = vol_expiries[ei]
            tenor = int(vol_tenors[ti])
            sigma_n = vol_surface[ei, ti]
            F = forward_swap_rate(expiry, tenor, mats, rates)
            A = swap_annuity(expiry, tenor, mats, rates)
            market_price = black_swaption_price(F, F, expiry, sigma_n, A, 1.0, True)
            model_price = analytical_swaption_price_jamshidian(a_, s_, expiry, tenor, F, 1.0, True, mats, rates)
            sse += (model_price - market_price) ** 2
        return sse
    
    # Hessian via FD
    f0 = calib_obj([a, sigma])
    f_ap = calib_obj([a + h, sigma])
    f_am = calib_obj([a - h, sigma])
    f_sp = calib_obj([a, sigma + h])
    f_sm = calib_obj([a, sigma - h])
    f_apsp = calib_obj([a + h, sigma + h])
    f_amsm = calib_obj([a - h, sigma - h])
    f_amsp = calib_obj([a - h, sigma + h])
    f_apsm = calib_obj([a + h, sigma - h])
    
    H_aa = (f_ap - 2*f0 + f_am) / (h**2)
    H_ss = (f_sp - 2*f0 + f_sm) / (h**2)
    H_as = (f_apsp - f_amsp - f_apsm + f_amsm) / (4 * h**2)
    
    # Hessian matrix
    H = np.array([[H_aa, H_as], [H_as, H_ss]])
    
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.zeros(n_vols)
    
    # Compute d²g/(dtheta d(vol_i)) for each vol node
    vol_greeks = np.zeros(n_vols)
    vol_bump = 1e-5
    
    for k, (ei, ti) in enumerate(calib_instruments):
        # Create bumped vol surface
        vol_bumped = vol_surface.copy()
        vol_bumped[ei, ti] += vol_bump
        
        # Gradient of objective w.r.t. theta at bumped vol
        def grad_at_bumped(params):
            a_, s_ = params
            sse = 0.0
            for ej, tj in calib_instruments:
                expiry = vol_expiries[ej]
                tenor = int(vol_tenors[tj])
                sigma_n = vol_bumped[ej, tj]
                F = forward_swap_rate(expiry, tenor, mats, rates)
                A = swap_annuity(expiry, tenor, mats, rates)
                market_price = black_swaption_price(F, F, expiry, sigma_n, A, 1.0, True)
                model_price = analytical_swaption_price_jamshidian(a_, s_, expiry, tenor, F, 1.0, True, mats, rates)
                sse += (model_price - market_price) ** 2
            return sse
        
        # Gradient at optimum with bumped vol
        grad_a_bumped = (grad_at_bumped([a + h, sigma]) - grad_at_bumped([a - h, sigma])) / (2 * h)
        grad_s_bumped = (grad_at_bumped([a, sigma + h]) - grad_at_bumped([a, sigma - h])) / (2 * h)
        
        # Gradient at optimum with original vol (should be ~0)
        grad_a_orig = (calib_obj([a + h, sigma]) - calib_obj([a - h, sigma])) / (2 * h)
        grad_s_orig = (calib_obj([a, sigma + h]) - calib_obj([a, sigma - h])) / (2 * h)
        
        # Cross derivative
        d2g_da_dvol = (grad_a_bumped - grad_a_orig) / vol_bump
        d2g_ds_dvol = (grad_s_bumped - grad_s_orig) / vol_bump
        
        cross_grad = np.array([d2g_da_dvol, d2g_ds_dvol])
        
        # IFT: dtheta/dvol = -H^{-1} * cross_grad
        dtheta_dvol = -H_inv @ cross_grad
        
        # Chain rule: dP/dvol = dP/dtheta * dtheta/dvol
        vol_greeks[k] = dP_da * dtheta_dvol[0] + dP_dsigma * dtheta_dvol[1]
    
    return vol_greeks


def fd_vol_surface_greeks(a: float, sigma: float, T_option: float, swap_tenor: int,
                           strike: float, notional: float, is_payer: bool,
                           mats: np.ndarray, rates: np.ndarray,
                           vol_expiries: np.ndarray, vol_tenors: np.ndarray,
                           vol_surface: np.ndarray,
                           calib_instruments: List[Tuple[int, int]],
                           bump: float = 1e-4) -> np.ndarray:
    """
    Compute vol surface Greeks via full recalibration (brute-force FD).
    """
    n_vols = len(calib_instruments)
    vol_greeks = np.zeros(n_vols)
    
    # Base price with current calibration
    base_price = analytical_swaption_price_jamshidian(a, sigma, T_option, swap_tenor,
                                                       strike, notional, is_payer, mats, rates)
    
    for k, (ei, ti) in enumerate(calib_instruments):
        # Bump up
        vol_up = vol_surface.copy()
        vol_up[ei, ti] += bump
        a_up, sigma_up, _ = calibrate_hull_white_least_squares(mats, rates, vol_expiries, vol_tenors,
                                                                 vol_up, calib_instruments)
        price_up = analytical_swaption_price_jamshidian(a_up, sigma_up, T_option, swap_tenor,
                                                         strike, notional, is_payer, mats, rates)
        
        # Bump down
        vol_dn = vol_surface.copy()
        vol_dn[ei, ti] -= bump
        a_dn, sigma_dn, _ = calibrate_hull_white_least_squares(mats, rates, vol_expiries, vol_tenors,
                                                                 vol_dn, calib_instruments)
        price_dn = analytical_swaption_price_jamshidian(a_dn, sigma_dn, T_option, swap_tenor,
                                                         strike, notional, is_payer, mats, rates)
        
        vol_greeks[k] = (price_up - price_dn) / (2 * bump)
    
    return vol_greeks


# =============================================================================
# XAD-based Monte Carlo Greeks
# =============================================================================

def xad_discount(T: float, mats: np.ndarray, rates_ad: list):
    """XAD discount factor"""
    if T <= 0:
        return xadj.Real(1.0)
    rate = xad_linear_interp(T, mats, rates_ad)
    return xmath.exp(-rate * T)


def xad_forward_rate(t: float, mats: np.ndarray, rates_ad: list, h: float = 1e-5):
    """XAD forward rate"""
    t = max(t, h)
    P_t = xad_discount(t, mats, rates_ad)
    P_th = xad_discount(t + h, mats, rates_ad)
    return -(xmath.log(P_th) - xmath.log(P_t)) / h


def xad_theta(t: float, a: float, sigma: float, mats: np.ndarray, rates_ad: list):
    """XAD theta function"""
    h = 1e-5
    f_t = xad_forward_rate(t, mats, rates_ad, h)
    f_th = xad_forward_rate(t + h, mats, rates_ad, h)
    f_prime = (f_th - f_t) / h
    term3 = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t))
    return f_prime + a * f_t + term3


def xad_bond_price(r_t_ad, t: float, T: float, a: float, sigma: float,
                    mats: np.ndarray, rates_ad: list, f_t_cache=None):
    """XAD bond price"""
    if T <= t:
        return xadj.Real(1.0)
    
    B = (1 - np.exp(-a * (T - t))) / a
    P_T = xad_discount(T, mats, rates_ad)
    P_t = xad_discount(t, mats, rates_ad)
    f_t = f_t_cache if f_t_cache else xad_forward_rate(t, mats, rates_ad)
    
    log_A = (xmath.log(P_T) - xmath.log(P_t) + B * f_t - 
             (sigma**2 / (4 * a)) * (1 - np.exp(-2 * a * t)) * B * B)
    
    return xmath.exp(log_A) * xmath.exp(-B * r_t_ad)


def xad_mc_price_and_greeks(a: float, sigma: float, T_option: float, swap_tenor: int,
                             strike: float, notional: float, is_payer: bool,
                             mats: np.ndarray, rates: np.ndarray,
                             num_paths: int, dt: float,
                             Z: np.ndarray) -> Tuple[float, np.ndarray]:
    """XAD Monte Carlo pricing with curve Greeks"""
    num_steps = Z.shape[1]
    n_rates = len(rates)
    t_grid = np.linspace(0, T_option, num_steps + 1)
    sqrt_dt = np.sqrt(dt)
    
    acc_price = 0.0
    acc_greeks = np.zeros(n_rates)
    
    payment_times = [T_option + i for i in range(1, swap_tenor + 1)]
    
    for path_idx in range(num_paths):
        with xadj.Tape() as tape:
            # Register rates as inputs
            rates_ad = [xadj.Real(float(r)) for r in rates]
            for r_ad in rates_ad:
                tape.registerInput(r_ad)
            tape.newRecording()
            
            # Pre-compute theta values
            theta_cache = [xad_theta(t, a, sigma, mats, rates_ad) for t in t_grid]
            
            # Simulate short rate
            r_t = xad_forward_rate(0.0, mats, rates_ad)
            for i in range(num_steps):
                r_t = r_t + (theta_cache[i] - a * r_t) * dt + sigma * sqrt_dt * Z[path_idx, i]
            
            r_T = r_t
            f_T = xad_forward_rate(T_option, mats, rates_ad)
            
            # Bond prices at expiry
            bonds = [xad_bond_price(r_T, T_option, T_pay, a, sigma, mats, rates_ad, f_T)
                     for T_pay in payment_times]
            
            # Fixed leg
            fixed_pv = xadj.Real(0.0)
            for b in bonds:
                fixed_pv = fixed_pv + b * strike
            
            # Floating leg
            float_pv = xadj.Real(1.0) - bonds[-1]
            
            # Swap value
            swap_val = (float_pv - fixed_pv) * notional
            
            # Payoff
            if is_payer:
                payoff = xmath.max(swap_val, xadj.Real(0.0))
            else:
                payoff = xmath.max(-swap_val, xadj.Real(0.0))
            
            # Discount
            disc = xad_discount(T_option, mats, rates_ad)
            price = disc * payoff
            
            tape.registerOutput(price)
            price.derivative = 1.0
            tape.computeAdjoints()
            
            acc_price += price.value
            for i, r_ad in enumerate(rates_ad):
                acc_greeks[i] += r_ad.derivative
    
    return acc_price / num_paths, acc_greeks / num_paths


# =============================================================================
# Finite Difference Greeks
# =============================================================================

def fd_curve_greeks(a: float, sigma: float, T_option: float, swap_tenor: int,
                     strike: float, notional: float, is_payer: bool,
                     mats: np.ndarray, rates: np.ndarray,
                     num_paths: int, dt: float, Z: np.ndarray,
                     bump: float = 1e-4) -> Tuple[float, np.ndarray]:
    """Finite difference curve Greeks using Monte Carlo"""
    base_price = mc_swaption_price(a, sigma, T_option, swap_tenor, strike, notional,
                                    is_payer, mats, rates, num_paths, dt, Z)
    
    greeks = np.zeros(len(rates))
    for i in range(len(rates)):
        rates_up = rates.copy()
        rates_up[i] += bump
        price_up = mc_swaption_price(a, sigma, T_option, swap_tenor, strike, notional,
                                      is_payer, mats, rates_up, num_paths, dt, Z)
        
        rates_dn = rates.copy()
        rates_dn[i] -= bump
        price_dn = mc_swaption_price(a, sigma, T_option, swap_tenor, strike, notional,
                                      is_payer, mats, rates_dn, num_paths, dt, Z)
        
        greeks[i] = (price_up - price_dn) / (2 * bump)
    
    return base_price, greeks


def fd_hw_greeks(a: float, sigma: float, T_option: float, swap_tenor: int,
                  strike: float, notional: float, is_payer: bool,
                  mats: np.ndarray, rates: np.ndarray,
                  num_paths: int, dt: float, Z: np.ndarray,
                  bump: float = 1e-5) -> Tuple[float, float]:
    """Finite difference Greeks for Hull-White parameters"""
    base = mc_swaption_price(a, sigma, T_option, swap_tenor, strike, notional,
                              is_payer, mats, rates, num_paths, dt, Z)
    
    # dP/da
    price_a_up = mc_swaption_price(a + bump, sigma, T_option, swap_tenor, strike, notional,
                                    is_payer, mats, rates, num_paths, dt, Z)
    price_a_dn = mc_swaption_price(a - bump, sigma, T_option, swap_tenor, strike, notional,
                                    is_payer, mats, rates, num_paths, dt, Z)
    greek_a = (price_a_up - price_a_dn) / (2 * bump)
    
    # dP/dsigma
    price_s_up = mc_swaption_price(a, sigma + bump, T_option, swap_tenor, strike, notional,
                                    is_payer, mats, rates, num_paths, dt, Z)
    price_s_dn = mc_swaption_price(a, sigma - bump, T_option, swap_tenor, strike, notional,
                                    is_payer, mats, rates, num_paths, dt, Z)
    greek_sigma = (price_s_up - price_s_dn) / (2 * bump)
    
    return greek_a, greek_sigma


def analytical_curve_greeks(a: float, sigma: float, T_option: float, swap_tenor: int,
                             strike: float, notional: float, is_payer: bool,
                             mats: np.ndarray, rates: np.ndarray,
                             bump: float = 1e-4) -> Tuple[float, np.ndarray]:
    """Finite difference curve Greeks using analytical pricing"""
    base_price = analytical_swaption_price_jamshidian(a, sigma, T_option, swap_tenor,
                                                       strike, notional, is_payer, mats, rates)
    
    greeks = np.zeros(len(rates))
    for i in range(len(rates)):
        rates_up = rates.copy()
        rates_up[i] += bump
        price_up = analytical_swaption_price_jamshidian(a, sigma, T_option, swap_tenor,
                                                         strike, notional, is_payer, mats, rates_up)
        
        rates_dn = rates.copy()
        rates_dn[i] -= bump
        price_dn = analytical_swaption_price_jamshidian(a, sigma, T_option, swap_tenor,
                                                         strike, notional, is_payer, mats, rates_dn)
        
        greeks[i] = (price_up - price_dn) / (2 * bump)
    
    return base_price, greeks


def analytical_hw_greeks(a: float, sigma: float, T_option: float, swap_tenor: int,
                          strike: float, notional: float, is_payer: bool,
                          mats: np.ndarray, rates: np.ndarray,
                          bump: float = 1e-5) -> Tuple[float, float]:
    """Finite difference HW parameter Greeks using analytical pricing"""
    # dP/da
    price_a_up = analytical_swaption_price_jamshidian(a + bump, sigma, T_option, swap_tenor,
                                                       strike, notional, is_payer, mats, rates)
    price_a_dn = analytical_swaption_price_jamshidian(a - bump, sigma, T_option, swap_tenor,
                                                       strike, notional, is_payer, mats, rates)
    greek_a = (price_a_up - price_a_dn) / (2 * bump)
    
    # dP/dsigma
    price_s_up = analytical_swaption_price_jamshidian(a, sigma + bump, T_option, swap_tenor,
                                                       strike, notional, is_payer, mats, rates)
    price_s_dn = analytical_swaption_price_jamshidian(a, sigma - bump, T_option, swap_tenor,
                                                       strike, notional, is_payer, mats, rates)
    greek_sigma = (price_s_up - price_s_dn) / (2 * bump)
    
    return greek_a, greek_sigma


# =============================================================================
# XAD Greeks for HW Parameters
# =============================================================================

def xad_hw_greeks(a: float, sigma: float, T_option: float, swap_tenor: int,
                   strike: float, notional: float, is_payer: bool,
                   mats: np.ndarray, rates: np.ndarray,
                   num_paths: int, dt: float, Z: np.ndarray) -> Tuple[float, float, float]:
    """XAD Greeks for Hull-White parameters a and sigma"""
    num_steps = Z.shape[1]
    t_grid = np.linspace(0, T_option, num_steps + 1)
    sqrt_dt = np.sqrt(dt)
    
    acc_price = 0.0
    acc_greek_a = 0.0
    acc_greek_sigma = 0.0
    
    payment_times = [T_option + i for i in range(1, swap_tenor + 1)]
    
    for path_idx in range(num_paths):
        with xadj.Tape() as tape:
            a_ad = xadj.Real(float(a))
            sigma_ad = xadj.Real(float(sigma))
            tape.registerInput(a_ad)
            tape.registerInput(sigma_ad)
            tape.newRecording()
            
            # Pre-compute theta values (with AD types)
            theta_cache = []
            for t in t_grid:
                h = 1e-5
                t_val = max(t, h)
                P_t = discount(t_val, mats, rates)
                P_th = discount(t_val + h, mats, rates)
                f_t = -(np.log(P_th) - np.log(P_t)) / h
                f_th_val = forward_rate(t_val + h, mats, rates)
                f_prime = (f_th_val - f_t) / h
                term3 = (sigma_ad * sigma_ad / (xadj.Real(2.0) * a_ad)) * (xadj.Real(1.0) - xmath.exp(xadj.Real(-2.0) * a_ad * t))
                theta_cache.append(f_prime + a_ad * f_t + term3)
            
            # Simulate short rate
            r_t = xadj.Real(forward_rate(0.0, mats, rates))
            for i in range(num_steps):
                r_t = r_t + (theta_cache[i] - a_ad * r_t) * dt + sigma_ad * sqrt_dt * Z[path_idx, i]
            
            r_T = r_t
            
            # Bond prices at expiry
            bonds = []
            for T_pay in payment_times:
                tau = T_pay - T_option
                B = (xadj.Real(1.0) - xmath.exp(-a_ad * tau)) / a_ad
                P_T = discount(T_pay, mats, rates)
                P_t = discount(T_option, mats, rates)
                f_t = forward_rate(T_option, mats, rates)
                sigma2_4a = sigma_ad * sigma_ad / (xadj.Real(4.0) * a_ad)
                exp_neg2at = xmath.exp(xadj.Real(-2.0) * a_ad * T_option)
                log_A = np.log(P_T) - np.log(P_t) + B * f_t - sigma2_4a * (xadj.Real(1.0) - exp_neg2at) * B * B
                bond = xmath.exp(log_A) * xmath.exp(-B * r_T)
                bonds.append(bond)
            
            # Fixed leg
            fixed_pv = xadj.Real(0.0)
            for b in bonds:
                fixed_pv = fixed_pv + b * strike
            
            # Floating leg
            float_pv = xadj.Real(1.0) - bonds[-1]
            
            # Swap value
            swap_val = (float_pv - fixed_pv) * notional
            
            # Payoff
            if is_payer:
                payoff = xmath.max(swap_val, xadj.Real(0.0))
            else:
                payoff = xmath.max(-swap_val, xadj.Real(0.0))
            
            # Discount
            disc = discount(T_option, mats, rates)
            price = payoff * disc
            
            tape.registerOutput(price)
            price.derivative = 1.0
            tape.computeAdjoints()
            
            acc_price += price.value
            acc_greek_a += a_ad.derivative
            acc_greek_sigma += sigma_ad.derivative
    
    return acc_price / num_paths, acc_greek_a / num_paths, acc_greek_sigma / num_paths


# =============================================================================
# Main Comparison Function
# =============================================================================

def run_comprehensive_comparison() -> str:
    """Run full comparison and return results as formatted string"""
    results = []
    results.append("=" * 100)
    results.append("COMPREHENSIVE SWAPTION COMPARISON: PRICING AND GREEKS")
    results.append("=" * 100)
    results.append(f"Date: January 2, 2026")
    results.append(f"Monte Carlo Paths: {MC_PATHS}")
    results.append(f"Time Step: {MC_DT}")
    results.append("")
    
    # Get market data
    mats, rates = get_ois_curve()
    vol_expiries, vol_tenors, vol_surface = get_vol_surface()
    
    results.append("-" * 100)
    results.append("OIS CURVE (SOFR-based, December 2024)")
    results.append("-" * 100)
    for i, (m, r) in enumerate(zip(mats, rates)):
        results.append(f"  Node {i:2d}: T={m:7.4f}Y, Rate={r*100:.3f}%")
    results.append("")
    
    # Calibrate Hull-White
    results.append("-" * 100)
    results.append("HULL-WHITE CALIBRATION (Least Squares on Prices)")
    results.append("-" * 100)
    
    # Use a subset of vol surface for calibration
    calib_instruments = [(2, 3), (3, 3), (4, 3), (5, 3), (6, 4), (7, 4)]  # Various expiry/tenor combos
    
    t0 = time.time()
    a_calib, sigma_calib, rmse = calibrate_hull_white_least_squares(
        mats, rates, vol_expiries, vol_tenors, vol_surface, calib_instruments
    )
    calib_time = time.time() - t0
    
    results.append(f"  Calibrated a     = {a_calib:.6f}")
    results.append(f"  Calibrated sigma = {sigma_calib:.6f}")
    results.append(f"  Price RMSE       = {rmse:.6f}")
    results.append(f"  Calibration Time = {calib_time:.3f}s")
    results.append("")
    
    # Store all comparison data
    all_pricing_data = []
    all_greeks_data = []
    
    # Process each swaption
    for swaption in SWAPTIONS:
        results.append("=" * 100)
        results.append(f"SWAPTION: {swaption.name} ({swaption.T_option}Y x {swaption.swap_tenor}Y)")
        results.append("=" * 100)
        
        # Get ATM strike
        strike = forward_swap_rate(swaption.T_option, swaption.swap_tenor, mats, rates)
        results.append(f"  ATM Strike: {strike*100:.4f}%")
        results.append(f"  Notional: ${swaption.notional:,}")
        results.append(f"  Type: {'Payer' if swaption.is_payer else 'Receiver'}")
        results.append("")
        
        # Generate random numbers for consistent MC comparison
        num_steps = int(swaption.T_option / MC_DT)
        Z = np.random.standard_normal((MC_PATHS, num_steps))
        
        # =====================================================================
        # PRICING COMPARISON
        # =====================================================================
        results.append("-" * 100)
        results.append("PRICING COMPARISON")
        results.append("-" * 100)
        
        # 1. Analytical (Jamshidian)
        t0 = time.time()
        price_analytical = analytical_swaption_price_jamshidian(
            a_calib, sigma_calib, swaption.T_option, swaption.swap_tenor,
            strike, swaption.notional, swaption.is_payer, mats, rates
        )
        time_analytical = time.time() - t0
        
        # 2. Monte Carlo
        t0 = time.time()
        price_mc = mc_swaption_price(
            a_calib, sigma_calib, swaption.T_option, swaption.swap_tenor,
            strike, swaption.notional, swaption.is_payer, mats, rates,
            MC_PATHS, MC_DT, Z
        )
        time_mc = time.time() - t0
        
        # 3. Monte Carlo + XAD (just pricing, no Greeks yet)
        t0 = time.time()
        price_xad, _ = xad_mc_price_and_greeks(
            a_calib, sigma_calib, swaption.T_option, swaption.swap_tenor,
            strike, swaption.notional, swaption.is_payer, mats, rates,
            min(MC_PATHS, 1000), MC_DT, Z[:min(MC_PATHS, 1000), :]  # Reduced for XAD
        )
        time_xad_price = time.time() - t0
        
        results.append(f"  {'Method':<25} {'Price':>15} {'Time (s)':>12}")
        results.append(f"  {'-'*25} {'-'*15} {'-'*12}")
        results.append(f"  {'Analytical (Jamshidian)':<25} ${price_analytical:>13,.2f} {time_analytical:>11.6f}")
        results.append(f"  {'Monte Carlo':<25} ${price_mc:>13,.2f} {time_mc:>11.6f}")
        results.append(f"  {'MC + XAD':<25} ${price_xad:>13,.2f} {time_xad_price:>11.6f}")
        results.append("")
        results.append(f"  Price Difference (MC vs Analytical): ${abs(price_mc - price_analytical):,.2f} ({abs(price_mc - price_analytical)/price_analytical*100:.4f}%)")
        results.append(f"  Price Difference (XAD vs Analytical): ${abs(price_xad - price_analytical):,.2f} ({abs(price_xad - price_analytical)/price_analytical*100:.4f}%)")
        results.append("")
        
        all_pricing_data.append({
            'swaption': swaption.name,
            'analytical': price_analytical,
            'mc': price_mc,
            'xad': price_xad,
            'time_analytical': time_analytical,
            'time_mc': time_mc,
            'time_xad': time_xad_price
        })
        
        # =====================================================================
        # GREEKS COMPARISON
        # =====================================================================
        results.append("-" * 100)
        results.append("GREEKS COMPARISON")
        results.append("-" * 100)
        
        # Reduce paths for Greeks comparison to make it faster
        reduced_paths = min(MC_PATHS, 2000)
        Z_reduced = Z[:reduced_paths, :]
        
        # 1. Analytical + FD for curve Greeks
        t0 = time.time()
        _, curve_greeks_analytical = analytical_curve_greeks(
            a_calib, sigma_calib, swaption.T_option, swaption.swap_tenor,
            strike, swaption.notional, swaption.is_payer, mats, rates
        )
        time_analytical_curve = time.time() - t0
        
        t0 = time.time()
        hw_greek_a_analytical, hw_greek_sigma_analytical = analytical_hw_greeks(
            a_calib, sigma_calib, swaption.T_option, swaption.swap_tenor,
            strike, swaption.notional, swaption.is_payer, mats, rates
        )
        time_analytical_hw = time.time() - t0
        
        # 2. Finite Difference (MC-based)
        t0 = time.time()
        _, curve_greeks_fd = fd_curve_greeks(
            a_calib, sigma_calib, swaption.T_option, swaption.swap_tenor,
            strike, swaption.notional, swaption.is_payer, mats, rates,
            reduced_paths, MC_DT, Z_reduced
        )
        time_fd_curve = time.time() - t0
        
        t0 = time.time()
        hw_greek_a_fd, hw_greek_sigma_fd = fd_hw_greeks(
            a_calib, sigma_calib, swaption.T_option, swaption.swap_tenor,
            strike, swaption.notional, swaption.is_payer, mats, rates,
            reduced_paths, MC_DT, Z_reduced
        )
        time_fd_hw = time.time() - t0
        
        # 3. XAD AAD
        t0 = time.time()
        _, curve_greeks_xad = xad_mc_price_and_greeks(
            a_calib, sigma_calib, swaption.T_option, swaption.swap_tenor,
            strike, swaption.notional, swaption.is_payer, mats, rates,
            min(reduced_paths, 500), MC_DT, Z_reduced[:min(reduced_paths, 500), :]
        )
        time_xad_curve = time.time() - t0
        
        t0 = time.time()
        _, hw_greek_a_xad, hw_greek_sigma_xad = xad_hw_greeks(
            a_calib, sigma_calib, swaption.T_option, swaption.swap_tenor,
            strike, swaption.notional, swaption.is_payer, mats, rates,
            min(reduced_paths, 500), MC_DT, Z_reduced[:min(reduced_paths, 500), :]
        )
        time_xad_hw = time.time() - t0
        
        # Report curve Greeks
        results.append("")
        results.append("  DISCOUNT CURVE SENSITIVITIES (dP/dRate):")
        results.append(f"  {'Node':<6} {'Mat':>8} {'Analytical':>14} {'FD (MC)':>14} {'XAD':>14} {'FD-Anly Diff%':>14}")
        results.append(f"  {'-'*6} {'-'*8} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
        
        for i in range(len(rates)):
            g_anly = curve_greeks_analytical[i]
            g_fd = curve_greeks_fd[i]
            g_xad = curve_greeks_xad[i]
            diff_pct = abs(g_fd - g_anly) / max(abs(g_anly), 1) * 100 if abs(g_anly) > 1 else 0
            results.append(f"  {i:<6} {mats[i]:>8.4f} {g_anly:>14.2f} {g_fd:>14.2f} {g_xad:>14.2f} {diff_pct:>13.2f}%")
        
        results.append("")
        results.append("  HULL-WHITE PARAMETER SENSITIVITIES:")
        results.append(f"  {'Parameter':<15} {'Analytical':>14} {'FD (MC)':>14} {'XAD':>14}")
        results.append(f"  {'-'*15} {'-'*14} {'-'*14} {'-'*14}")
        results.append(f"  {'dP/da':<15} {hw_greek_a_analytical:>14.2f} {hw_greek_a_fd:>14.2f} {hw_greek_a_xad:>14.2f}")
        results.append(f"  {'dP/dsigma':<15} {hw_greek_sigma_analytical:>14.2f} {hw_greek_sigma_fd:>14.2f} {hw_greek_sigma_xad:>14.2f}")
        
        results.append("")
        results.append("  TIMING COMPARISON:")
        results.append(f"  {'Method':<30} {'Curve Greeks':>15} {'HW Greeks':>15}")
        results.append(f"  {'-'*30} {'-'*15} {'-'*15}")
        results.append(f"  {'Analytical (FD-based)':<30} {time_analytical_curve:>14.4f}s {time_analytical_hw:>14.4f}s")
        results.append(f"  {'Finite Difference (MC)':<30} {time_fd_curve:>14.4f}s {time_fd_hw:>14.4f}s")
        results.append(f"  {'XAD AAD':<30} {time_xad_curve:>14.4f}s {time_xad_hw:>14.4f}s")
        
        # =====================================================================
        # VOL SURFACE GREEKS (via IFT and FD with recalibration)
        # =====================================================================
        results.append("")
        results.append("-" * 100)
        results.append("VOL SURFACE GREEKS (Sensitivity to ATM Vol Nodes via Calibration)")
        results.append("-" * 100)
        
        # Compute via IFT
        t0 = time.time()
        vol_greeks_ift = compute_vol_surface_greeks_ift(
            a_calib, sigma_calib, swaption.T_option, swaption.swap_tenor,
            strike, swaption.notional, swaption.is_payer, mats, rates,
            vol_expiries, vol_tenors, vol_surface, calib_instruments
        )
        time_vol_ift = time.time() - t0
        
        # Compute via FD (brute force recalibration)
        t0 = time.time()
        vol_greeks_fd = fd_vol_surface_greeks(
            a_calib, sigma_calib, swaption.T_option, swaption.swap_tenor,
            strike, swaption.notional, swaption.is_payer, mats, rates,
            vol_expiries, vol_tenors, vol_surface, calib_instruments
        )
        time_vol_fd = time.time() - t0
        
        results.append("")
        results.append(f"  {'Calib Instr':<15} {'Expiry':>8} {'Tenor':>6} {'IFT':>14} {'FD Recalib':>14}")
        results.append(f"  {'-'*15} {'-'*8} {'-'*6} {'-'*14} {'-'*14}")
        
        for k, (ei, ti) in enumerate(calib_instruments):
            exp = vol_expiries[ei]
            ten = vol_tenors[ti]
            results.append(f"  {k:<15} {exp:>8.2f} {ten:>6.0f} {vol_greeks_ift[k]:>14.2f} {vol_greeks_fd[k]:>14.2f}")
        
        results.append("")
        results.append(f"  IFT Time:           {time_vol_ift:.4f}s")
        results.append(f"  FD Recalib Time:    {time_vol_fd:.4f}s")
        results.append(f"  Speedup:            {time_vol_fd/time_vol_ift:.1f}x" if time_vol_ift > 0 else "  Speedup:            N/A")
        results.append("")
        
        all_greeks_data.append({
            'swaption': swaption.name,
            'curve_greeks_analytical': curve_greeks_analytical,
            'curve_greeks_fd': curve_greeks_fd,
            'curve_greeks_xad': curve_greeks_xad,
            'hw_a_analytical': hw_greek_a_analytical,
            'hw_a_fd': hw_greek_a_fd,
            'hw_a_xad': hw_greek_a_xad,
            'hw_sigma_analytical': hw_greek_sigma_analytical,
            'hw_sigma_fd': hw_greek_sigma_fd,
            'hw_sigma_xad': hw_greek_sigma_xad,
            'vol_greeks_ift': vol_greeks_ift,
            'vol_greeks_fd': vol_greeks_fd,
            'time_analytical_curve': time_analytical_curve,
            'time_fd_curve': time_fd_curve,
            'time_xad_curve': time_xad_curve,
            'time_analytical_hw': time_analytical_hw,
            'time_fd_hw': time_fd_hw,
            'time_xad_hw': time_xad_hw,
            'time_vol_ift': time_vol_ift,
            'time_vol_fd': time_vol_fd
        })
    
    # =====================================================================
    # SUMMARY STATISTICS
    # =====================================================================
    results.append("")
    results.append("=" * 100)
    results.append("SUMMARY: PRICING COMPARISON ACROSS ALL SWAPTIONS")
    results.append("=" * 100)
    results.append("")
    results.append(f"{'Swaption':<12} {'Analytical':>15} {'Monte Carlo':>15} {'MC+XAD':>15} {'MC-Anly %':>12} {'Time Ratio':>12}")
    results.append(f"{'-'*12} {'-'*15} {'-'*15} {'-'*15} {'-'*12} {'-'*12}")
    
    for data in all_pricing_data:
        diff_pct = abs(data['mc'] - data['analytical']) / data['analytical'] * 100
        time_ratio = data['time_mc'] / data['time_analytical'] if data['time_analytical'] > 0 else 0
        results.append(f"{data['swaption']:<12} ${data['analytical']:>13,.2f} ${data['mc']:>13,.2f} ${data['xad']:>13,.2f} {diff_pct:>11.4f}% {time_ratio:>11.1f}x")
    
    results.append("")
    results.append("=" * 100)
    results.append("SUMMARY: TIMING COMPARISON FOR GREEKS CALCULATION")
    results.append("=" * 100)
    results.append("")
    results.append(f"{'Swaption':<12} {'Anly Curve':>12} {'FD Curve':>12} {'XAD Curve':>12} {'Anly HW':>12} {'FD HW':>12} {'XAD HW':>12}")
    results.append(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for data in all_greeks_data:
        results.append(f"{data['swaption']:<12} {data['time_analytical_curve']:>11.4f}s {data['time_fd_curve']:>11.4f}s {data['time_xad_curve']:>11.4f}s {data['time_analytical_hw']:>11.4f}s {data['time_fd_hw']:>11.4f}s {data['time_xad_hw']:>11.4f}s")
    
    results.append("")
    results.append("=" * 100)
    results.append("SUMMARY: VOL SURFACE GREEKS TIMING (IFT vs FD Recalibration)")
    results.append("=" * 100)
    results.append("")
    results.append(f"{'Swaption':<12} {'IFT Time':>12} {'FD Recalib':>12} {'Speedup':>10}")
    results.append(f"{'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    
    for data in all_greeks_data:
        speedup = data['time_vol_fd'] / data['time_vol_ift'] if data['time_vol_ift'] > 0 else 0
        results.append(f"{data['swaption']:<12} {data['time_vol_ift']:>11.4f}s {data['time_vol_fd']:>11.4f}s {speedup:>9.1f}x")
    
    results.append("")
    results.append("=" * 100)
    results.append("ANALYSIS NOTES")
    results.append("=" * 100)
    results.append("")
    results.append("1. PRICING:")
    results.append("   - Analytical (Jamshidian) is fastest as it's closed-form")
    results.append("   - Monte Carlo converges to analytical with sufficient paths")
    results.append("   - MC+XAD provides same prices with derivative computation capability")
    results.append("")
    results.append("2. GREEKS (Discount Curve & HW Parameters):")
    results.append("   - Analytical + FD: Fastest for closed-form pricing, each bump is cheap")
    results.append("   - FD on MC: Requires 2N+1 pricings for N curve nodes")
    results.append("   - XAD AAD: Single tape recording per path, efficient for many Greeks")
    results.append("   - Note: Python XAD has overhead; C++ implementation is much faster")
    results.append("")
    results.append("3. VOL SURFACE GREEKS (via Calibration):")
    results.append("   - IFT Method: Uses Implicit Function Theorem to avoid recalibration")
    results.append("   - FD Recalib: Brute-force bump-and-recalibrate for each vol node")
    results.append("   - IFT is significantly faster as it avoids repeated optimizations")
    results.append("")
    results.append("4. ACCURACY:")
    results.append("   - FD and XAD curve Greeks converge with more MC paths")
    results.append("   - Analytical Greeks (via FD on closed-form) are most accurate")
    results.append("   - XAD Greeks match FD Greeks when using same random numbers")
    results.append("   - IFT vol Greeks are approximations valid near the optimum")
    results.append("")
    results.append("=" * 100)
    
    return "\n".join(results)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    print("Running comprehensive swaption comparison...")
    print("This may take a few minutes...")
    print("")
    
    results = run_comprehensive_comparison()
    print(results)
    
    # Save to file
    output_file = "SWAPTION_COMPARISON_RESULTS.txt"
    with open(output_file, 'w') as f:
        f.write(results)
    
    print(f"\nResults saved to {output_file}")
