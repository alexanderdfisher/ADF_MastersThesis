# Running the EGZ inversion on local data

The data are too large to upload and are licensed WRDS extracts, so this runs on the
Windows PC. **Never commit data files.**

## Setup
- Branch `claude/equity-risk-premium-reverse-h9oa63`.
- Data folder: `C:\Users\Fisher\Desktop\Thesis\Main Data`. This is the default `DATA_DIR`;
  override it with the `EGZ_DATA_DIR` environment variable.
- `pip install pandas numpy scipy openpyxl psycopg2-binary matplotlib tabulate pytest`

## Steps
1. `python -m pytest -q test_egz_model.py`. All 8 tests should pass.
2. **Unit checks before trusting any output:**
   - CRSP `SHROUT` in `All/dailyCRSP_CHSdta.csv`: the CRSP convention is thousands, and
     `get_vars_revival` divides by 1000 to get millions. Check PRC*SHROUT for a known large
     firm against its actual market cap. If SHROUT is already in millions, remove the /1000.
   - Compustat sale, cogs, xsga, dltt and dlc should be in $ millions.
   - AAA/BAA yields (`CORP_BOND_DTA.xlsx`) and the `rf`/`10y` columns
     (`Rf_SP500ret_annual.xlsx`): percent or decimal? Is `rf` a T-bill rate? The paper uses
     the average of the 3-month T-bill and 10-year yields.
3. `python RestartFromScratch_EGZ.py --gvkey 1010 --date 19831231 --out results/egz_1010.csv`
   - Show all inputs (sigma, xgm, gm, f, couponrate, csr, dsr, std, ltd, lev, rf, prc,
     shares, sti) and the outputs mu_q and r_a.
   - Sanity ranges: sigma 0.05-1, lev in (0,1), couponrate about 0.08-0.16 in 1983,
     mu_q < rf, R_A about 5-20%.
   - If mu_q is NaN, compare `egz.model_price` at mu_Q in {-0.3, -0.1, 0, rf-0.001}
     with prc to see why there is no root.
4. Run about 20 S&P 500 constituents for 19831231 (`sp500_constituents.xlsx`, sheet
   `constituents`; pass several gvkeys to `--gvkey`). Summarise mu_q and R_A, and list each
   failure with its reason.
5. Fix only clear, local bugs in the input functions (e.g. the `df_fundy['TWOSIC']` vs
   `'twosic'` KeyError). List anything bigger. Commit code only, never data or `results/`.

Background: see `CODE_REVIEW.md` section 0. The old `*_revival` valuation functions are
known to be broken; use `get_frmyr_miu_q_egz`, which calls `egz_model.py`.
