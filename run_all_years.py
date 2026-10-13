"""
Run the EGZ inversion for every S&P 500 constituent in every year of sp500_constituents.xlsx.

    python run_all_years.py --workers 6            # all years, resumable
    python run_all_years.py --years 1983 1984      # selected years

Each worker loads the data once (importing RestartFromScratch_EGZ) and then values whole years.
Writes results/years/egz_<year>.csv (valued firms) and results/years/fail_<year>.csv (reasons);
a year whose egz_<year>.csv exists is skipped. Concatenate with summarize_results.py.
"""
import argparse
import multiprocessing as mp
import os
import time
import warnings

OUT_DIR = os.path.join("results", "years")


def _init_worker():
    warnings.filterwarnings("ignore")
    global m, full_dsf
    import RestartFromScratch_EGZ as m
    full_dsf = m.df_dsf


def _run_year(key):
    import pandas as pd
    year = key // 10000
    # Prices and shares only need the latest observation within the year: slicing CRSP makes the
    # per-firm lookups ~100x cheaper. A firm that did not trade during the year gets no price.
    m.df_dsf = full_dsf[(full_dsf["DATE"] >= pd.Timestamp(year, 1, 1)) & (full_dsf["DATE"] <= pd.Timestamp(year, 12, 31))]
    gvkeys = [int(g) for g in m.sp500_dict[key] if not pd.isna(g)]
    rows, fails, t0 = [], [], time.time()
    for gv in gvkeys:
        try:
            res = m.get_frmyr_miu_q_egz(gv, key, verbose=False)
            if pd.isna(res["mu_q"]):
                fails.append({"gvkey": gv, "date": key, "reason": "no mu_Q root"})
            rows.append(res)
        except Exception as e:
            fails.append({"gvkey": gv, "date": key, "reason": f"{type(e).__name__}: {e}"})
    pd.DataFrame(fails, columns=["gvkey", "date", "reason"]).to_csv(os.path.join(OUT_DIR, f"fail_{year}.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"egz_{year}.csv"), index=False)
    n_ok = sum(not pd.isna(r["mu_q"]) for r in rows)
    return f"{year}: {n_ok}/{len(gvkeys)} valued in {time.time() - t0:.0f}s"


if __name__ == "__main__":
    import pandas as pd
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--years", type=int, nargs="*")
    cli = p.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    cols = pd.read_excel(os.path.join(os.environ.get("EGZ_DATA_DIR", r"C:\Users\Fisher\Desktop\Thesis\Main Data"),
                                      "sp500_constituents.xlsx"), "constituents", nrows=0).columns
    keys = [int(c) for c in cols if (not cli.years or int(c) // 10000 in cli.years)
            and not os.path.exists(os.path.join(OUT_DIR, f"egz_{int(c) // 10000}.csv"))]
    print(f"{len(keys)} years to run with {cli.workers} workers", flush=True)
    with mp.Pool(cli.workers, initializer=_init_worker) as pool:
        for msg in pool.imap_unordered(_run_year, keys):
            print(msg, flush=True)
    print("done", flush=True)
