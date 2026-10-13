"""
Eisdorfer, Goyal & Zhdanov (2018), "Equity Misvaluation and Default Options" -
valuation engine, plus the inversions used in this thesis:

    observed price  ->  mu_Q  ->  R_A  ->  cost of equity / ERP

Equation numbers refer to the paper (Appendix A, eqs. A1-A9; Section 2.2, eqs. 10-11).
All money inputs must be in the same units (e.g. Compustat $ millions), and
price = equity / shares with shares in millions.

Key simplification used here: in the post-refinancing (year-5) problem the
ratio a/K = r/(r - mu) does not depend on the coupon or on fixed costs, so the
default-to-distress ratio u_d = x_d / x* depends only on (r, mu_Q, sigma, eta).
The 4x4 non-linear system (A5) therefore reduces to one 1-D root per mu_Q, and
every coupon / terminal node after that is closed form.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq

TAU = 0.35      # corporate tax rate (paper)
ETA = 0.15      # distress cost (paper)
ALPHA = 0.15    # bankruptcy cost (paper sets alpha = eta = 15%)


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
def pi_coef(gm_ratio, csr, dsr, tau=TAU):
    """Cash flow per unit of gross margin: (tau*DSR - CSR)/GM + (1 - tau)  (A3)."""
    return (tau * dsr - csr) / gm_ratio + (1.0 - tau)


def x_star(c, F, pi, tau=TAU):
    """Distress threshold where the cash flow to equity is zero (A2)."""
    return (1.0 - tau) * (c + F) / pi


def kappa(gm_ratio, csr, dsr, tau=TAU):
    """mu_P = kappa * R_A  (eq. 10)."""
    return csr / ((1.0 - tau) * gm_ratio + tau * dsr)


def roots(sigma, mu, r):
    """Positive and negative roots of 0.5*sigma^2*b*(b-1) + mu*b - r = 0."""
    a = 0.5 * sigma ** 2
    b = mu - 0.5 * sigma ** 2
    disc = math.sqrt(b * b + 4.0 * a * r)
    return (-b + disc) / (2.0 * a), (-b - disc) / (2.0 * a)


# ----------------------------------------------------------------------------
# Year-5 post-refinancing structural block (A4-A6)
# ----------------------------------------------------------------------------
class StructuralBlock:
    """
    Solves (A5) once for given (r, mu_Q, sigma, eta). With u = x/x* and
    K = (1-tau)(c+F)/r, equity is E = K * e(u), where
        e(u) = b u^b2 + g(u)                        u >= 1
        e(u) = c1 u^b1 + d2 u^b2 + (1+eta) g(u)     u_d <= u < 1
        e(u) = 0                                    u < u_d   (defaulted)
    and g(u) = r u/(r-mu) - 1.
    """

    def __init__(self, r, mu, sigma, eta=ETA, tau=TAU, alpha=ALPHA):
        if mu >= r:
            raise ValueError(f"mu_Q ({mu:.4f}) must be below r ({r:.4f})")
        self.r, self.mu, self.sigma = r, mu, sigma
        self.eta, self.tau, self.alpha = eta, tau, alpha
        self.b1, self.b2 = roots(sigma, mu, r)
        self.k = r / (r - mu)  # a/K
        self.u_d = self._solve_ud()
        self.cc, self.dd, self.bb = self._coefs(self.u_d)

    def _coefs(self, u):
        # value matching and smooth pasting (x * E'(x)) at u_d -> C', D'
        b1, b2, k, eta = self.b1, self.b2, self.k, self.eta
        rhs1 = -(1 + eta) * (k * u - 1.0)        # P + Q
        rhs2 = -(1 + eta) * k * u                # b1 P + b2 Q
        P = (rhs2 - b2 * rhs1) / (b1 - b2)
        Q = rhs1 - P
        cc, dd = P * u ** (-b1), Q * u ** (-b2)
        # value matching at u = 1 -> B'
        bb = cc + dd + eta * (k - 1.0)
        return cc, dd, bb

    def _residual(self, log_u):
        # smooth pasting at u = 1 (the remaining equation of A5)
        cc, dd, bb = self._coefs(math.exp(log_u))
        return self.b2 * bb + self.k - self.b1 * cc - self.b2 * dd - (1 + self.eta) * self.k

    def _solve_ud(self):
        hi = -1e-10
        lo = -1.0
        while self._residual(lo) * self._residual(hi) > 0:
            lo *= 2.0
            if lo < -200:
                raise RuntimeError("could not bracket default threshold u_d")
        return math.exp(brentq(self._residual, lo, hi, xtol=1e-14))

    def e_dimless(self, u):
        u = np.asarray(u, dtype=float)
        g = self.k * u - 1.0
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            normal = self.bb * u ** self.b2 + g
            distress = self.cc * u ** self.b1 + self.dd * u ** self.b2 + (1 + self.eta) * g
        out = np.where(u >= 1.0, normal, np.where(u > self.u_d, distress, 0.0))
        return np.maximum(out, 0.0)

    # --- claims in money units ---------------------------------------------
    def equity(self, x, c, F, pi):
        """Post-refinancing equity E(x) (A4). c = 0 gives the unlevered value V_U."""
        x = np.asarray(x, dtype=float)
        if pi <= 0:
            return np.zeros_like(x)
        if c + F <= 0:
            return pi * x / (self.r - self.mu)
        K = (1 - self.tau) * (c + F) / self.r
        return K * self.e_dimless(x / x_star(c, F, pi, self.tau))

    def default_threshold(self, c, F, pi):
        return self.u_d * x_star(c, F, pi, self.tau) if c + F > 0 else 0.0

    def debt(self, x, c, F, pi):
        """Perpetual debt value (A6), with the recovery read as (1-alpha)*V_U(x_d) - c/r."""
        x = np.asarray(x, dtype=float)
        if c <= 0:
            return np.zeros_like(x)
        xd = self.default_threshold(c, F, pi)
        vu_xd = float(self.equity(xd, 0.0, F, pi))
        with np.errstate(divide="ignore", over="ignore"):
            val = c / self.r + (x / xd) ** self.b2 * ((1 - self.alpha) * vu_xd - c / self.r)
        return np.where(x > xd, val, (1 - self.alpha) * self.equity(x, 0.0, F, pi))

    def coupon_for_leverage(self, x5, F, pi, target_lev):
        """Coupon c such that D/(D+E) = target industry leverage. Returns 0 if none."""
        if target_lev <= 0 or pi <= 0:
            return 0.0
        # largest coupon for which the firm is still alive at x5
        c_max = x5 * pi / ((1 - self.tau) * self.u_d) - F
        if c_max <= 0:
            return 0.0

        def f(c):
            d = float(self.debt(x5, c, F, pi))
            e = float(self.equity(x5, c, F, pi))
            return (d / (d + e) if d + e > 0 else 1.0) - target_lev

        lo, hi = c_max * 1e-9, c_max * (1 - 1e-9)
        if f(lo) * f(hi) > 0:
            return 0.0
        return brentq(f, lo, hi, xtol=1e-10 * max(1.0, c_max))

    def pre_refi_equity(self, x5, F, pi, target_lev, coupons_out=None):
        """E'(x5) = E(x5) + D(x5)  (A7): the new debt proceeds go to shareholders."""
        x5 = np.atleast_1d(np.asarray(x5, dtype=float))
        out = np.empty_like(x5)
        for i, x in enumerate(x5):
            c = self.coupon_for_leverage(x, F, pi, target_lev)
            if coupons_out is not None:
                coupons_out.append(c)
            out[i] = float(self.equity(x, c, F, pi)) + float(self.debt(x, c, F, pi))
        return out


# ----------------------------------------------------------------------------
# Years 0-5: binomial lattice in y = log x (A8-A9)
# ----------------------------------------------------------------------------
@dataclass
class FirmInputs:
    gm: float            # gross margin x0 = Sales - COGS  ($)
    gm_ratio: float      # GM = x / Sales
    csr: float           # industry avg capex/sales (3y)
    dsr: float           # industry avg depreciation/sales (3y)
    F: float             # fixed costs (XSGA, $)
    sigma: float         # annualised quarterly sales volatility
    ltd: float           # long-term debt (DLTT, $), repaid at T = 5
    std: float           # short-term debt (DLC, $), repaid at t = 1
    coupon_rate: float   # yield by CHS quintile: AAA / BBB / BBB+2% (decimal)
    st_rate: float       # interest rate on short-term debt (decimal)
    target_lev: float    # industry market leverage used at refinancing
    r: float             # risk-free rate (paper: avg of 3m T-bill and 10y)
    shares: float = 1.0  # shares outstanding (so that equity/shares = price)


def equity_value(mu_q, f: FirmInputs, T=5.0, N=50, eta=ETA, tau=TAU, alpha=ALPHA, trace=None):
    """
    Model equity value at t = 0 for a given risk-neutral drift mu_Q.
    Pass a dict as `trace` to collect intermediate values (see `print_trace`).
    """
    if f.gm <= 0:
        raise ValueError("gross margin must be positive (paper excludes x0 <= 0)")
    dt = T / N
    dy = f.sigma * math.sqrt(dt)
    p_u = 0.5 + (mu_q - 0.5 * f.sigma ** 2) * math.sqrt(dt) / (2 * f.sigma)
    if not 0.0 <= p_u <= 1.0:
        raise ValueError(f"p_u = {p_u:.3f} outside [0,1]; reduce dt or check sigma/mu_Q")
    p_d = 1.0 - p_u
    disc = math.exp(-f.r * dt)
    pi = pi_coef(f.gm_ratio, f.csr, f.dsr, tau)
    y0 = math.log(f.gm)
    m_std = int(round(1.0 / dt))  # node where short-term debt is repaid

    # terminal nodes j = 0..N, y = y0 + (2j - N) dy
    blk = StructuralBlock(f.r, mu_q, f.sigma, eta, tau, alpha)
    xT = np.exp(y0 + (2 * np.arange(N + 1) - N) * dy)
    coupons = [] if trace is not None else None
    E_pre = blk.pre_refi_equity(xT, f.F, pi, f.target_lev, coupons)
    E = np.maximum(E_pre - f.ltd, 0.0)
    if trace is not None:
        trace.update(mu_q=mu_q, dt=dt, dy=dy, p_u=p_u, pi=pi, beta1=blk.b1, beta2=blk.b2,
                     u_d=blk.u_d, x_T_min=xT.min(), x_T_max=xT.max(),
                     coupon_T_min=min(coupons), coupon_T_max=max(coupons),
                     unlevered_T_share=float(np.mean(np.array(coupons) == 0)),
                     default_T_share=float(np.mean(E == 0)),
                     x_star_today=x_star(f.coupon_rate * f.ltd + f.st_rate * f.std, f.F, pi, tau)
                     if pi > 0 else np.nan, defaulted_nodes=0, total_nodes=N + 1)

    for m in range(N - 1, -1, -1):
        x = np.exp(y0 + (2 * np.arange(m + 1) - m) * dy)
        interest = f.coupon_rate * f.ltd + (f.st_rate * f.std if m < m_std else 0.0)
        cf = pi * x - (1 - tau) * (interest + f.F)
        # (A1): distress cost scales the whole negative cash flow by (1 + eta)
        cf = np.where(cf < 0, (1 + eta) * cf, cf) * dt
        principal = f.std if m == m_std else 0.0
        E = np.maximum(disc * (p_u * E[1:] + p_d * E[:-1]) + cf - principal, 0.0)
        if trace is not None:
            trace["defaulted_nodes"] += int(np.sum(E == 0))
            trace["total_nodes"] += m + 1
            if m == m_std:
                trace["default_share_year1"] = float(np.mean(E == 0))
    if trace is not None:
        trace["equity_0"] = float(E[0])
    return float(E[0])


def print_trace(t):
    """Readable summary of an `equity_value(..., trace=t)` run."""
    print(f"    lattice      : dt={t['dt']:.2f}  dy={t['dy']:.4f}  p_u={t['p_u']:.4f}")
    print(f"    cash flow    : Pi={t['pi']:.4f} per $ of gross margin; "
          f"x* today={t['x_star_today']:.2f} (cash flow < 0 below this)")
    print(f"    year-5 block : beta1={t['beta1']:.3f}  beta2={t['beta2']:.3f}  x_d/x*={t['u_d']:.4f}")
    print(f"    year-5 nodes : x in [{t['x_T_min']:.1f}, {t['x_T_max']:.1f}]  "
          f"coupon in [{t['coupon_T_min']:.2f}, {t['coupon_T_max']:.2f}]  "
          f"no new debt possible={t['unlevered_T_share']:.0%}  "
          f"equity=0 after LTD repayment={t['default_T_share']:.0%}")
    print(f"    default      : {t['defaulted_nodes']}/{t['total_nodes']} lattice nodes at zero equity; "
          f"year-1 nodes in default={t.get('default_share_year1', float('nan')):.0%}")
    print(f"    equity today : {t['equity_0']:.2f}")


def model_price(mu_q, f: FirmInputs, **kw):
    return equity_value(mu_q, f, **kw) / f.shares


def implied_mu_q(price, f: FirmInputs, lo=-0.30, hi=None, n_scan=25, verbose=False, **kw):
    """
    Risk-neutral drift that equates model and observed price. Scans for a sign
    change first (the objective is deterministic, so this is reliable) and then
    applies Brent. Returns np.nan if no root lies in [lo, hi].
    """
    hi = f.r - 1e-4 if hi is None else min(hi, f.r - 1e-4)

    def diff(mu):
        try:
            return model_price(mu, f, **kw) - price
        except ValueError:
            return np.nan

    grid = np.linspace(lo, hi, n_scan)
    vals = np.array([diff(m) for m in grid])
    if verbose:
        print(f"    mu_Q scan (observed price {price:.2f}):")
        for m, v in zip(grid, vals):
            mp = "  n/a (p_u outside [0,1])" if not np.isfinite(v) else f"{v + price:12.2f}"
            print(f"      mu_Q={m:+.4f}  model price={mp}")
    for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb <= 0:
            root = brentq(diff, a, b, xtol=1e-6)
            if verbose:
                print(f"    root bracket [{a:+.4f}, {b:+.4f}] -> mu_Q = {root:+.5f}")
            return root
    if verbose:
        fin = vals[np.isfinite(vals)]
        if fin.size == 0:
            print("    NO ROOT: model price undefined on the whole grid")
        elif np.all(fin > 0):
            print("    NO ROOT: model price is above the observed price even at the lowest mu_Q")
        else:
            print("    NO ROOT: model price stays below the observed price even as mu_Q -> r")
    return np.nan


# ----------------------------------------------------------------------------
# From mu_Q to discount rates (eqs. 10-11)
# ----------------------------------------------------------------------------
def implied_asset_return(mu_q, r, gm_ratio, csr, dsr, tau=TAU):
    """
    Eq. 11 with eq. 10 substituted: mu_Q = r - R_A + kappa*R_A, hence
    R_A = (r - mu_Q) / (1 - kappa). No outside assumption about mu_P is needed.
    """
    return (r - mu_q) / (1.0 - kappa(gm_ratio, csr, dsr, tau))


def implied_cost_of_equity(R_A, cod, w_e, w_d, tau_d=0.0):
    """R_A = w_e*CoE + w_d*(1 - tau_d)*CoD solved for CoE (tau_d = 0: paper's WACC)."""
    return (R_A - w_d * (1.0 - tau_d) * cod) / w_e


def market_erp(df):
    """
    Market ERP per date from firm-level results (columns: date, coe, me, rf).
    The market portfolio has beta = 1, so ERP = value-weighted CoE - r.
    """
    import pandas as pd
    d = df.dropna(subset=["coe", "me", "rf"])
    d = d[d["me"] > 0]

    def one(g):
        w = g["me"] / g["me"].sum()
        coe = float((w * g["coe"]).sum())
        r = float(g["rf"].iloc[0])
        return pd.Series({"n_firms": len(g), "coe_vw": coe, "rf": r, "erp": coe - r,
                          "coe_median": float(g["coe"].median())})

    return d.groupby("date").apply(one).reset_index()
