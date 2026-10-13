import math

import numpy as np
import pytest

from egz_model import (FirmInputs, StructuralBlock, equity_value, implied_asset_return,
                       implied_mu_q, kappa, model_price, pi_coef, x_star)

R, MU, SIG = 0.05, 0.01, 0.25
PI = pi_coef(0.35, 0.066, 0.048)


def firm(**kw):
    base = dict(gm=1000.0, gm_ratio=0.35, csr=0.066, dsr=0.048, F=600.0, sigma=0.25,
                ltd=800.0, std=100.0, coupon_rate=0.08, st_rate=0.06, target_lev=0.25,
                r=0.05, shares=50.0)
    base.update(kw)
    return FirmInputs(**base)


def test_A5_boundary_conditions_hold():
    blk = StructuralBlock(R, MU, SIG)
    c, F = 50.0, 200.0
    xs, xd = x_star(c, F, PI), blk.default_threshold(c, F, PI)
    E = lambda x: float(blk.equity(x, c, F, PI))
    h = 1e-6 * xs
    assert xd < xs
    assert abs(E(xs * (1 + 1e-9)) - E(xs * (1 - 1e-9))) < 1e-4          # value matching at x*
    dl = (E(xs) - E(xs - h)) / h
    dr = (E(xs + h) - E(xs)) / h
    assert abs(dl - dr) < 1e-4 * max(1, abs(dr))                          # smooth pasting at x*
    assert abs(E(xd * (1 + 1e-9))) < 1e-6                                 # E(x_d) = 0
    assert abs((E(xd + h) - E(xd)) / h) < 1e-3                            # E'(x_d) = 0


def test_equity_exceeds_forced_operation_value():
    # the default option is worth >= 0 (eq. 4)
    blk = StructuralBlock(R, MU, SIG)
    x = np.linspace(100, 3000, 50)
    c, F = 50.0, 200.0
    dcf = PI * x / (R - MU) - (1 - 0.35) * (c + F) / R
    assert np.all(blk.equity(x, c, F, PI) >= dcf - 1e-8)


def test_debt_bounded_by_riskless_perpetuity():
    blk = StructuralBlock(R, MU, SIG)
    c = 60.0
    d = blk.debt(np.linspace(500, 5000, 20), c, 200.0, PI)
    assert np.all(d <= c / R + 1e-9) and np.all(d > 0)


def test_coupon_matches_target_leverage():
    blk = StructuralBlock(R, MU, SIG)
    x5, F = 1500.0, 300.0
    c = blk.coupon_for_leverage(x5, F, PI, 0.3)
    d, e = float(blk.debt(x5, c, F, PI)), float(blk.equity(x5, c, F, PI))
    assert c > 0 and abs(d / (d + e) - 0.3) < 1e-8


def test_lattice_reproduces_gordon_value_without_frictions():
    # no debt, no fixed cost -> equity = PI*x0/(r - mu) up to discretisation
    f = firm(F=0.0, ltd=0.0, std=0.0, target_lev=0.0, shares=1.0)
    mu = 0.02
    exact = PI * f.gm / (f.r - mu)
    assert equity_value(mu, f) == pytest.approx(exact, rel=1e-3)  # dt = 0.1 discretisation


def test_price_increasing_in_mu_q():
    f = firm()
    p = [model_price(m, f) for m in np.linspace(-0.10, 0.04, 8)]
    assert np.all(np.diff(p) > 0)


def test_round_trip_inversion():
    f = firm()
    true_mu = 0.012
    mu_hat = implied_mu_q(model_price(true_mu, f), f)
    assert mu_hat == pytest.approx(true_mu, abs=1e-5)


def test_asset_return_inversion_consistent_with_eq_10_11():
    R_A, r = 0.09, 0.04
    k = kappa(0.35, 0.066, 0.048)
    mu_q = r - R_A + k * R_A
    assert implied_asset_return(mu_q, r, 0.35, 0.066, 0.048) == pytest.approx(R_A)


def test_market_erp_value_weights_firm_coe():
    import pandas as pd
    from egz_model import market_erp
    df = pd.DataFrame({"date": [1, 1], "coe": [0.10, 0.06], "me": [300.0, 100.0], "rf": [0.04, 0.04]})
    out = market_erp(df).iloc[0]
    assert out["coe_vw"] == pytest.approx(0.09)
    assert out["erp"] == pytest.approx(0.05)


def test_cost_of_equity_uses_paper_wacc_without_tax_shield():
    from egz_model import implied_cost_of_equity
    # R_A = 0.6*CoE + 0.4*CoD  ->  CoE = (0.08 - 0.4*0.07)/0.6
    assert implied_cost_of_equity(0.08, 0.07, 0.6, 0.4) == pytest.approx((0.08 - 0.028) / 0.6)
