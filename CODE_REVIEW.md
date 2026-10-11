# Code review: `RestartFromScratch_EGZ.py`

Goal of the script: take the Eisdorfer, Goyal & Zhdanov (2018, JF) "Equity Misvaluation and
Default Options" valuation model and run it backwards. Instead of computing a fundamental
price, solve for the risk-neutral drift `mu_Q` that makes the model price equal the observed
price. Then value-weight `mu_Q` across the S&P 500 and back out the ERP from
`r_A = mu_P - mu_Q + r_f` and `r_A = w_e*CoE + w_d*(1-tau)*CoD`.

Line numbers refer to the committed file. Items are ordered by how much they distort the
estimate of `mu_Q`.

## A. Bugs that stop the pipeline from producing a valid `mu_Q`

1. **The root search fails on every firm-year, and the error is hidden** (L1165/1173, L1028-1046).
   `model_diff_revival` returns a tuple `(diff, B, C, D, xd)`. `brentq` needs a scalar
   function. Given a tuple, it raises `ValueError: truth value of an array ... ambiguous`
   (verified). `get_frmyr_miu_q_revival` catches `ValueError` and prints
   *"guess needs diff signs"*, so every firm looks like a bracketing problem. On top of that,
   `brentq` returns a float, so `mu_q, B, C, D, xd = brentq(...)` would fail anyway.
   Fix: the objective returns only `diff`. After the root is found, call the model once more at
   that `mu_q` to get B, C, D, xd.

2. **`sys_of_eq_revival` has no `return` statement** (L678-722). `fsolve` gets `None`.
   It also has only 2 equations for 4 unknowns. `compute_leverageclaude` uses `prc`, which is
   not one of its arguments (a `NameError`). Its bare `except:` swallows both errors, so
   `find_c_to_match_leverage_revivalclaude` always returns `(Vu, 0, 0, None, ...)`. Every
   terminal node therefore gets the *unlevered* value. Leverage matching never runs.

3. **The 5-year simulation is thrown away** (L246-254). The loop appends `gm`, the starting
   value, instead of `gm_temp`. All 51 terminal gross margins equal today's GM (verified: one
   distinct value). The positivity check also tests `gm` instead of `gm_temp`.

4. **The tree mixes Monte Carlo paths with a binomial lattice.**
   * Terminal nodes come from 51 *random, unsorted* simulated paths. Intermediate cash flows
     (`cf_node`, L971) come from yet another random path per node, and that path doesn't
     depend on the node index `n`. In EGZ, node (m, j) is the lattice state
     `y = y0 + (2j - m)·σ√dt`, so its cash flow is deterministic. With random draws the
     objective is noisy in `mu_Q`, which breaks Brent's method (no stable sign change). Your
     commented-out code `x_5 = exp(y_0 + n*dy)` in `check_cf` was the right idea.
   * Indexing (L1122-1157). At m=49, n=0 reads `E[-1, 50]`, which wraps to row 50 (verified).
     Row 50 is never filled for m<50, so `E[50,1]` stays 0 and feeds the root node. The return
     cell `E[49,0]` is arbitrary. Standard layout: node j = 0..m at time m, with
     `up = E[j+1, m+1]` and `down = E[j, m+1]`, returning `E[0, 0]`.
   * `p_u = 0.5 + α√dt/(2σ)` falls outside [0,1] when σ is small relative to |α|. For example,
     σ=0.03 and α=0.08 gives p_u ≈ 0.92, and σ=0.02 gives more than 1. Add a guard or skip
     the firm.

5. **Debt principal is subtracted at every node** (L1153: `pv_eq + cf - (ltd + std)`).
   Over 50 steps this removes the debt 50 times and drives most nodes to the `max(·, 0)` floor.
   Interest is already in `cf_node` (`couponrate*ltd`). Principal should come out once: STD
   (plus its interest) at the year-1 node, and LTD at T through `tv_eq - debt` (already at
   L1112).

6. **`find_Vu_xd_revival2` operator precedence** (L628).
   `cashflow_coef * gm_5 / rf - g` should be `cashflow_coef * gm_5 / (rf - g)`.

7. **`x_star` formula** (L502). The code computes `(1-τ + τ·dsr - csr)/xgm`. The zero-cash-flow
   point is `x* = (1-τ)(c+f) / [(1-τ) + (τ·dsr - csr)/xgm]`. Only the capex/depreciation term
   should be divided by `xgm`, matching the `fill` term used everywhere else.

8. **`execute_values(cur, sql, insert)`** (L1329/1390). The function expects a list of tuples:
   `[tuple(map(float_or_int, insert))]`. Convert numpy scalars, or psycopg2 may reject them.

## B. Model or specification issues in the terminal (year-5) structural block

9. **The terminal system should be 4 equations in (B, C, D, x_d), with no observed price in it.**
   With Π = (1-τ) + (τ·dsr - csr)/xgm and K = (1-τ)(c+f)/r:
   * normal region (x ≥ x*): `E = B x^β2 + Π x/(r-μ) - K`
   * distress region (x_d ≤ x < x*): `E = C x^β1 + D x^β2 + (1+η)(Π x/(r-μ) - K)`
   * the conditions are value-matching and smooth-pasting at x* (2 equations), plus
     `E(x_d)=0` and `E'(x_d)=0` (2 equations).

   The current code puts `prc` in place of `B·x*^β2` and uses `b2 * prc**(-1)` as a derivative.
   That mixes today's per-share market price into a year-5 firm-level boundary problem.
   Given x_d, the three conditions are linear in B, C, D, so a 1-D root search on x_d
   is far more stable than a 4-D `fsolve`.

10. **Debt value uses `Vu(gm_5)` instead of `Vu(x_d)`** (L813/878). In Leland-type models
    the recovery is `(1-α)·Vu(x_d)`.

11. **No guard for `mu_Q ≥ r`.** The bracket is [-0.1, 0.1], but r_f is 1-5% for much of
    your sample. Above r_f the Gordon terms `/(rf - mu_q)` change sign or blow up, and
    `roots()` is undefined at the boundary. Set the upper bracket to something like
    `min(0.1, rf - 1e-4)`.

12. **`coupon_rate` quintile logic** (L582-587). `if frm_quint == 0 or 1:` is always True.
    Use `in (0, 1)` / `in (2, 3)`. Check units too: `spread = aaa` is the AAA *yield level*
    (FRED reports it in percent, and the code never divides by 100). That makes
    `rf + aaa` either about 100x too large, or a double-counted rf if it was already
    decimal. The fallback at L432 has the same problem.

## C. ERP inversion (`find_yr_erp_revival`)

13. `mu_P = 0.04` is a constant, so all time variation in the ERP comes from `mu_Q`, and the
    ERP level moves one-for-one with that 4% guess. Consider an estimated physical drift of
    aggregate gross margin (e.g. trailing or forecast growth of S&P GM) and report
    sensitivity to it.
14. Maturity mismatch: CoD uses the 10-year yield plus OAS, but `rf` in the identity is
    the short rate from `df_rf['rf']`. Use one consistent rate.
15. The identity `mu_P - mu_Q = r_A - r_f` applies to the *unlevered* state variable (gross
    margin). Using the after-tax WACC as `r_A` is consistent only because Π already
    nets out taxes. This deserves a sentence in the thesis.

## D. Smaller issues

* L91: `df_dbaa = df_daaa.dropna()` is a typo (unused, but misleading).
* L394: `df_fundy['TWOSIC']` should be `'twosic'` (the KeyError is swallowed).
* L282-283: the industry-σ fallback takes the std of pooled changes, not the mean of
  per-firm std devs as the comment says.
* L318-324: `csr_dsr` recomputes rolling means over the whole dataframe on every call.
  Compute once at load time; it will be a large speedup with 12 workers.
* `pct_change` on raw quarterly sales picks up seasonality. EGZ-style σ is usually taken on
  year-over-year quarterly changes or deseasonalized sales.
* DB credentials: the password is now read from the `PGPASSWORD` environment variable.
  The original hard-coded password should be rotated if it was ever shared.

## Suggested order of work
1 → 3 → 4 → 5 → 6/7 → 9 → 11 → 12, then re-run single-firm tests (gvkey 1010, 1983)
and plot `model_price(mu_Q)` over the bracket. It should be smooth and monotone before
you hand it to `brentq`.
