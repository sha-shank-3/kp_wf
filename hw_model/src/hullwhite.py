import numpy as np

# One-factor Hull-White (extended Vasicek) helpers
# We use a deterministic shift to fit the initial curve, with parameters (a, sigma).

class HullWhiteModel:
    def __init__(self, a: float, sigma: float, zero_curve_times: np.ndarray, zero_curve_rates: np.ndarray):
        self.a = float(a)
        self.sigma = float(sigma)
        self.times = np.array(zero_curve_times, dtype=float)
        self.rates = np.array(zero_curve_rates, dtype=float)
        # Build continuously compounded discount factors from market zero rates
        self.discounts = np.exp(-self.rates * self.times)
        # Interpolators for discount curve
        self._build_interpolators()

    def _build_interpolators(self):
        # Simple log-linear interpolation on discounts to ensure monotonicity
        # x: time, y: log(discount)
        x = self.times
        y = np.log(np.maximum(self.discounts, 1e-12))
        self._x = x
        self._y = y

    def discount(self, T: float) -> float:
        # Piecewise-linear interpolation in log space
        x, y = self._x, self._y
        if T <= x[0]:
            return float(np.exp(np.interp(T, [0.0, x[0]], [0.0, y[0]])))
        if T >= x[-1]:
            return float(np.exp(np.interp(T, [x[-1], T], [y[-1], y[-1]])))
        yi = np.interp(T, x, y)
        return float(np.exp(yi))

    def zero_rate(self, T: float) -> float:
        if T <= 1e-8:
            return self.rates[0]
        P = self.discount(T)
        return -np.log(P) / T

    def instantaneous_forward(self, T: float, h: float = 1e-4) -> float:
        # f(T) = -d/dT ln P(0,T)
        T1 = max(T - h, 1e-8)
        T2 = T + h
        P1, P2 = self.discount(T1), self.discount(T2)
        return -(np.log(P2) - np.log(P1)) / (T2 - T1)

    def B(self, t: float, T: float) -> float:
        a = self.a
        return (1.0 - np.exp(-a * (T - t))) / a

    def A(self, t: float, T: float) -> float:
        # Extended Vasicek/Hull-White A-factor under deterministic shift to fit curve
        # Using initial discount curve P(0,.) via discount(T)
        B = self.B(t, T)
        sigma = self.sigma
        a = self.a
        # Variance term
        var = (sigma**2 / (2*a**3)) * (1 - np.exp(-a*(T-t)))**2 * (1 - np.exp(-2*a*t))
        P0T = self.discount(T)
        P0t = self.discount(t)
        # A(t,T) ensures model matches initial curve: P(t,T) = P(0,T)/P(0,t) * exp(something)
        # Here we approximate the Hull-White adjustment via variance term
        return (P0T / P0t) * np.exp(var)

    def bond_price(self, r_t: float, t: float, T: float) -> float:
        # P(t,T) = A(t,T) * exp(-B(t,T) * r_t)
        return float(self.A(t, T) * np.exp(-self.B(t, T) * r_t))
