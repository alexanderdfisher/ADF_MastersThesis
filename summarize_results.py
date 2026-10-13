"""
Summary tables, figures and an HTML report from the run_all_years.py output.

    python summarize_results.py            # reads results/years, writes results/summary

Everything written goes under results/ (git-ignored): it is derived from licensed data.
"""
import base64
import glob
import io
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import egz_model as egz

warnings.filterwarnings("ignore")
DATA_DIR = os.environ.get("EGZ_DATA_DIR", r"C:\Users\Fisher\Desktop\Thesis\Main Data")
IN_DIR, OUT_DIR = os.path.join("results", "years"), os.path.join("results", "summary")

# validated reference palette (dataviz skill), first three categorical slots + neutrals
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 110, "savefig.bbox": "tight",
})
SECTORS = [(1, 9, "Agriculture"), (10, 14, "Mining & oil/gas"), (15, 17, "Construction"),
           (20, 39, "Manufacturing"), (40, 48, "Transport & telecom"), (49, 49, "Utilities"),
           (50, 51, "Wholesale"), (52, 59, "Retail"), (60, 69, "Financials"), (70, 89, "Services"),
           (90, 99, "Other")]


def sector(s2):
    for lo, hi, name in SECTORS:
        if lo <= s2 <= hi:
            return name
    return "Unknown"


def wavg(x, w):
    ok = x.notna() & w.notna() & (w > 0)
    return float((x[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan


def reason_group(r):
    r = str(r)
    if "financial" in r: return "Financial (excluded)"
    if "Pi <= 0" in r: return "Pi <= 0 (capex > margin)"
    if "gross margin" in r: return "Gross margin <= 0"
    if "no mu_Q root" in r: return "No mu_Q root"
    return "Missing input data"


# ----------------------------------------------------------------------------- load
def load():
    res = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(IN_DIR, "egz_*.csv")))], ignore_index=True)
    fails = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(IN_DIR, "fail_*.csv")))], ignore_index=True)
    res["year"] = res["date"] // 10000
    fails["year"] = fails["date"] // 10000
    ok = res[res["mu_q"].notna()].copy()

    fy = pd.read_csv(os.path.join(DATA_DIR, "Sample", "fundamentals_pull_csv.csv"), usecols=["gvkey", "fyear", "sic"])
    fy["sic2"] = fy["sic"].astype(int).astype(str).str.zfill(4).str[:2].astype(int)
    fy = fy.sort_values("fyear").drop_duplicates(["gvkey", "fyear"], keep="last").rename(columns={"fyear": "year"})

    def add_sic(df):
        out = pd.merge_asof(df.sort_values("year"), fy[["gvkey", "year", "sic2"]].sort_values("year"),
                            on="year", by="gvkey", direction="backward")
        out["sector"] = out["sic2"].apply(lambda s: sector(s) if pd.notna(s) else "Unknown")
        return out

    ok, fails = add_sic(ok), add_sic(fails)
    ok["sales"] = ok["gm"] / ok["xgm"]
    ok["f_sales"] = ok["f"] / ok["sales"]
    ok["kappa"] = egz.kappa(ok["xgm"], ok["csr"], ok["dsr"])
    ok["pi"] = egz.pi_coef(ok["xgm"], ok["csr"], ok["dsr"])
    ok["erp_i"] = ok["coe"] - ok["rf"]
    ok["mkt_lev"] = (ok["std"] + ok["ltd"]) / (ok["std"] + ok["ltd"] + ok["me"])
    ok["bucket"] = np.select([ok["couponrate"] <= ok.groupby("year")["couponrate"].transform("min") + 1e-9,
                              ok["couponrate"] >= ok.groupby("year")["couponrate"].transform("max") - 1e-9],
                             ["AAA (CHS q1-2)", "BBB+2% (CHS q5)"], "BBB (CHS q3-4)")
    ok["size_q"] = ok.groupby("year")["me"].transform(lambda s: pd.qcut(s, 5, labels=[1, 2, 3, 4, 5])).astype(int)
    # model-fragility flags
    ok["flag_own_csr"] = ok["csr_source"].eq("own")
    ok["flag_f0"] = ok["f_imputed"].astype(bool)
    ok["flag_near_r"] = (ok["rf"] - ok["mu_q"]) < 0.01
    ok["flag_coe_lt_rf"] = ok["coe"] < ok["rf"]
    ok["flag_coe_gt_40"] = ok["coe"] > 0.40
    ok["flag_any"] = ok[["flag_own_csr", "flag_f0", "flag_near_r", "flag_coe_lt_rf", "flag_coe_gt_40"]].any(axis=1)

    # market caps for every constituent (valued or not) at the last trading day of the year
    dsf = pd.read_csv(os.path.join(DATA_DIR, "All", "dailyCRSP_CHSdta.csv"), usecols=["DATE", "GVKEY", "PRC", "SHROUT"],
                      na_values=["C", "B"])
    dsf["year"] = dsf["DATE"] // 10000
    me = dsf.sort_values("DATE").groupby(["GVKEY", "year"]).tail(1)
    me = me.assign(me_all=me["PRC"].abs() * me["SHROUT"] / 1e3)[["GVKEY", "year", "me_all"]].rename(columns={"GVKEY": "gvkey"})
    fails = fails.merge(me, on=["gvkey", "year"], how="left")
    fails["group"] = fails["reason"].apply(reason_group)

    rf = pd.read_excel(os.path.join(DATA_DIR, "Rf_SP500ret_annual.xlsx"))
    rf["year"] = pd.to_datetime(rf["Date"], format="%Y%m%d").dt.year
    rf = rf.set_index("year")[["3mo", "10y"]] / 100
    dam = pd.read_excel(os.path.join(DATA_DIR, "damodaran.xls"), "Import").set_index("Year")
    rec = pd.read_excel(os.path.join(DATA_DIR, "recession_periods.xls"), header=0)
    rec.columns = ["date", "rec"]
    rec["date"] = pd.to_datetime(rec["date"])
    return ok, fails, rf, dam, rec


# ----------------------------------------------------------------------------- tables
def annual_table(ok, rf, dam):
    g = ok.groupby("year")
    t = pd.DataFrame({
        "n_valued": g.size(),
        "r (avg 3m,10y)": g["rf"].first(),
        "CoE VW": g.apply(lambda d: wavg(d["coe"], d["me"])),
        "CoE EW": g["coe"].mean(),
        "CoE median": g["coe"].median(),
        "R_A VW": g.apply(lambda d: wavg(d["r_a"], d["me"])),
        "mu_Q VW": g.apply(lambda d: wavg(d["mu_q"], d["me"])),
    })
    t["ERP VW"] = t["CoE VW"] - t["r (avg 3m,10y)"]
    t["ERP median"] = g["erp_i"].median()
    t["ERP VW vs 10y"] = t["CoE VW"] - rf["10y"].reindex(t.index)
    clean = ok[~ok["flag_any"]].groupby("year")
    t["ERP VW, unflagged firms"] = clean.apply(lambda d: wavg(d["coe"], d["me"])) - t["r (avg 3m,10y)"]
    t["Damodaran implied ERP (FCFE)"] = dam["Implied ERP (FCFE)"].reindex(t.index)
    return t


def coverage_table(ok, fails):
    v = ok.groupby("year").agg(valued=("gvkey", "size"), me_valued=("me", "sum"))
    f = fails.pivot_table(index="year", columns="group", values="gvkey", aggfunc="count", fill_value=0)
    fme = fails.pivot_table(index="year", columns="group", values="me_all", aggfunc="sum", fill_value=0)
    t = v.join(f).fillna(0)
    t["constituents"] = t["valued"] + f.sum(axis=1).reindex(t.index).fillna(0)
    me_tot = t["me_valued"] + fme.sum(axis=1).reindex(t.index).fillna(0)
    fin_me = fme.get("Financial (excluded)", 0)
    t["% firms valued"] = 100 * t["valued"] / t["constituents"]
    t["% mkt cap valued"] = 100 * t["me_valued"] / me_tot
    t["% mkt cap valued, ex-financials"] = 100 * t["me_valued"] / (me_tot - fin_me)
    t["% mkt cap financial"] = 100 * fin_me / me_tot
    cols = ["constituents", "valued", "% firms valued", "% mkt cap valued", "% mkt cap valued, ex-financials",
            "% mkt cap financial"] + list(f.columns)
    return t[cols], fme.div(me_tot, axis=0) * 100


def pooled_stats(ok):
    spec = [("sigma", "Sales volatility sigma", 0.396), ("xgm", "Gross margin / sales (GM)", 0.253),
            ("f_sales", "Fixed costs (XSGA) / sales", 0.351), ("csr", "Capex / sales (CSR)", 0.108),
            ("dsr", "Depreciation / sales (DSR)", 0.079), ("couponrate", "Coupon rate (CoD)", 0.0835),
            ("rf", "Risk-free r", 0.0522), ("lev", "Industry target leverage", 0.265),
            ("mkt_lev", "Firm market leverage", 0.265), ("pi", "Pi (cash flow per $ GM)", np.nan),
            ("kappa", "kappa = mu_P / R_A", np.nan), ("mu_q", "Implied mu_Q", np.nan),
            ("r_a", "Implied R_A", 0.0939), ("coe", "Implied CoE", np.nan), ("erp_i", "Firm CoE - r", np.nan),
            ("me", "Market equity ($M)", np.nan)]
    rows = []
    for col, label, egz_mean in spec:
        x = ok[col].astype(float)
        q = x.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        rows.append({"variable": label, "N": x.notna().sum(), "mean": x.mean(), "sd": x.std(), "p1": q[0.01],
                     "p25": q[0.25], "median": q[0.5], "p75": q[0.75], "p99": q[0.99], "EGZ Table 1 mean": egz_mean})
    return pd.DataFrame(rows).set_index("variable")


def group_table(ok, by, label):
    g = ok.groupby(by)
    t = pd.DataFrame({
        "firm-years": g.size(),
        "share of mkt cap %": 100 * g["me"].sum() / ok["me"].sum(),
        "CoE median": g["coe"].median(),
        "R_A median": g["r_a"].median(),
        "CoE - r median": g["erp_i"].median(),
        "CoE - r VW": g.apply(lambda d: wavg(d["erp_i"], d["me"])),
        "mu_Q median": g["mu_q"].median(),
        "book D/(D+E) median": g["w_d"].median(),
        "% own CSR": 100 * g["flag_own_csr"].mean(),
        "% CoE < r": 100 * g["flag_coe_lt_rf"].mean(),
        "% mu_Q within 1pp of r": 100 * g["flag_near_r"].mean(),
    })
    t.index.name = label
    return t


def period_table(annual):
    bins = [(1983, 1989), (1990, 1999), (2000, 2009), (2010, 2022), (1983, 2022)]
    cols = ["r (avg 3m,10y)", "CoE VW", "R_A VW", "ERP VW", "ERP median", "ERP VW vs 10y",
            "ERP VW, unflagged firms", "Damodaran implied ERP (FCFE)"]
    rows = {f"{a}-{b}": annual.loc[a:b, cols].mean() for a, b in bins}
    t = pd.DataFrame(rows).T
    t["ERP VW sd (annual)"] = [annual.loc[a:b, "ERP VW"].std() for a, b in bins]
    t["corr(ERP VW, Damodaran)"] = [annual.loc[a:b, ["ERP VW", "Damodaran implied ERP (FCFE)"]].corr().iloc[0, 1]
                                    for a, b in bins]
    return t


def flag_table(ok):
    cols = {"flag_own_csr": "% own CSR/DSR (industry Pi <= 0)", "flag_f0": "% F = 0 (no XSGA)",
            "flag_near_r": "% mu_Q within 1pp of r", "flag_coe_lt_rf": "% CoE < r", "flag_coe_gt_40": "% CoE > 40%",
            "flag_any": "% any flag"}
    t = 100 * ok.groupby("year")[list(cols)].mean().rename(columns=cols)
    mc = ok.assign(**{c: ok[c] * ok["me"] for c in cols}).groupby("year")
    t["% mkt cap flagged"] = 100 * mc["flag_any"].sum() / mc["me"].sum()
    return t


# ----------------------------------------------------------------------------- figures
def shade_recessions(ax, rec):
    r = rec.assign(year=rec["date"].dt.year).groupby("year")["rec"].max()
    for y, v in r.items():
        if v and 1983 <= y <= 2022:
            ax.axvspan(y - 0.5, y + 0.5, color=SHADE, zorder=0, lw=0)


def label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.plot([x], [y], "o", color=color, ms=4)


def fig_erp(annual, rec):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    shade_recessions(ax, rec)
    x = annual.index
    for col, c, lab in [("ERP VW", BLUE, "EGZ-implied ERP (VW)"), ("ERP median", AQUA, "EGZ-implied ERP (median firm)"),
                        ("Damodaran implied ERP (FCFE)", ORANGE, "Damodaran implied ERP")]:
        y = 100 * annual[col]
        ax.plot(x, y, color=c, label=lab)
        label_end(ax, x[-1], y.iloc[-1], f"{y.iloc[-1]:.1f}%", c)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("percent")
    ax.set_title("Implied equity risk premium, S&P 500, 1983-2022")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.text(0, -0.24, "Shaded: NBER recession years. EGZ ERP = value-weighted CoE minus r (avg of 3m bill and 10y). "
                      "Damodaran is measured against the 10y bond.", transform=ax.transAxes, fontsize=8, color=INK2)
    return fig


def fig_components(annual, rf, rec):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    shade_recessions(ax, rec)
    x = annual.index
    for y, c, lab in [(annual["CoE VW"], BLUE, "Cost of equity (VW)"), (annual["R_A VW"], AQUA, "Return on assets R_A (VW)"),
                      (annual["r (avg 3m,10y)"], ORANGE, "Risk-free r")]:
        ax.plot(x, 100 * y, color=c, label=lab)
        label_end(ax, x[-1], 100 * y.iloc[-1], f"{100 * y.iloc[-1]:.1f}%", c)
    ax.set_ylabel("percent")
    ax.set_ylim(bottom=0)
    ax.set_title("Discount-rate components: the ERP is the gap between the blue and orange lines")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    return fig


def fig_coverage(cov, fme):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = cov.index
    order = ["Financial (excluded)", "Pi <= 0 (capex > margin)", "Missing input data", "Gross margin <= 0", "No mu_Q root"]
    colors = [ORANGE, AQUA, "#9a9893", "#c9c7c1", "#5f5d58"]
    ax.bar(x, cov["% mkt cap valued"], color=BLUE, width=0.8, label="Valued", edgecolor="white", linewidth=1)
    bottom = cov["% mkt cap valued"].values.copy()
    for g, c in zip(order, colors):
        if g in fme.columns:
            v = fme[g].reindex(x).fillna(0).values
            ax.bar(x, v, bottom=bottom, color=c, width=0.8, label=g, edgecolor="white", linewidth=1)
            bottom += v
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of S&P 500 market cap")
    ax.set_title("Sample coverage by market cap, and why the rest drops out")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8)
    return fig


def fig_box(ok):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    years = sorted(ok["year"].unique())
    data = [100 * ok.loc[ok["year"] == y, "erp_i"].values for y in years]
    bp = ax.boxplot(data, positions=years, widths=0.6, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=INK2, lw=1),
                    capprops=dict(color=INK2, lw=1), boxprops=dict(facecolor="#b7d3f6", edgecolor=BLUE, lw=1),
                    whis=(5, 95))
    ax.axhline(0, color=ORANGE, lw=1)
    ax.set_xticks(years[::3])
    ax.set_ylabel("firm CoE - r, percent")
    ax.set_title("Cross-section of firm-level premia (boxes: 25-75%, whiskers: 5-95%)")
    return fig


def fig_leverage(ok):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for b, c in [("AAA (CHS q1-2)", BLUE), ("BBB (CHS q3-4)", AQUA), ("BBB+2% (CHS q5)", ORANGE)]:
        d = ok[ok["bucket"] == b]
        ax.scatter(100 * d["w_d"], 100 * d["erp_i"].clip(-0.3, 0.5), s=9, alpha=0.35, color=c, label=b, linewidths=0)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("book debt weight w_d, percent")
    ax.set_ylabel("firm CoE - r, percent (clipped)")
    ax.set_title("Leverage amplifies firm premia: CoE = (R_A - w_d*CoD)/(1 - w_d) fans out as w_d rises")
    ax.legend(loc="upper left", markerscale=2)
    return fig


def fig_flags(flags):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for col, c in [("% own CSR/DSR (industry Pi <= 0)", BLUE), ("% CoE < r", ORANGE), ("% mu_Q within 1pp of r", AQUA)]:
        ax.plot(flags.index, flags[col], color=c, label=col)
    ax.set_ylabel("% of valued firms")
    ax.set_title("Model-fragility flags over time")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8)
    return fig


def fig_sector(sec):
    sec = sec.drop(index=[s for s in ["Unknown", "Financials"] if s in sec.index]).sort_values("CoE - r median")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(sec))
    ax.barh(y, 100 * sec["CoE - r median"], color=BLUE, height=0.6)
    ax.set_yticks(y, sec.index)
    for i, (v, own, n) in enumerate(zip(sec["CoE - r median"], sec["% own CSR"], sec["firm-years"])):
        ax.annotate(f"{100 * v:.1f}%  (n={n:,}, own CSR {own:.0f}%)", (100 * v, i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("median firm CoE - r, percent")
    ax.set_title("Median firm premium by sector, pooled 1983-2022")
    ax.grid(axis="y", visible=False)
    return fig


# ----------------------------------------------------------------------------- findings
def seasonality():
    q = pd.read_csv(os.path.join(DATA_DIR, "Sample", "Qtrly_Sales_COMP.csv")).dropna(subset=["saleq", "sic"])
    q["retail"] = q["sic"].astype(int).astype(str).str.zfill(4).str[:2].astype(int).between(52, 59)
    g = q.sort_values(["gvkey", "datadate"]).groupby("gvkey")["saleq"]
    q["qoq"], q["yoy"] = g.pct_change(), g.pct_change(4)
    return q.groupby("retail").agg(as_coded=("qoq", lambda v: v.clip(-1, 1).std() * 2),
                                   yoy=("yoy", lambda v: v.clip(-1, 1).std()))


def findings_html(ok, annual, periods, cov, flags, sec, size, buck, src, seas):
    P = lambda v: f"{100 * v:.1f}%"
    full = periods.loc["1983-2022"]
    worst = annual["ERP VW"].idxmin()
    top = ok[ok["year"] == worst].nlargest(3, "me")
    top_txt = ", ".join(f"gvkey {int(r.gvkey)} (R_A {P(r.r_a)}, CoE {P(r.coe)})" for r in top.itertuples())
    util = sec.loc["Utilities"]
    f0 = ok[ok["flag_f0"]]
    f0_util = 100 * ok.loc[ok["sector"] == "Utilities", "flag_f0"].mean()
    s_ret, s_oth = seas.loc[True], seas.loc[False]
    head = (
        f"<p><b>Headline.</b> Over 1983-2022 the value-weighted EGZ-implied ERP averages <b>{P(full['ERP VW'])}</b> "
        f"(median firm {P(full['ERP median'])}; {P(full['ERP VW vs 10y'])} measured against the 10-year yield, the "
        f"basis Damodaran uses; his implied ERP averages {P(full['Damodaran implied ERP (FCFE)'])}). The series tracks "
        f"Damodaran closely in 1990-2009 (correlation {periods.loc['1990-1999', 'corr(ERP VW, Damodaran)']:.2f} and "
        f"{periods.loc['2000-2009', 'corr(ERP VW, Damodaran)']:.2f}) and spikes in 2008 like his, but diverges after "
        f"2010: EGZ falls back to about 3% while Damodaran rises to 5-6%. Sub-period means: 1980s "
        f"{P(periods.loc['1983-1989', 'ERP VW'])}, 1990s {P(periods.loc['1990-1999', 'ERP VW'])}, 2000s "
        f"{P(periods.loc['2000-2009', 'ERP VW'])}, 2010-22 {P(periods.loc['2010-2022', 'ERP VW'])}.</p>"
        f"<p><b>Sample.</b> {cov['valued'].sum():,.0f} firm-years valued out of {cov['constituents'].sum():,.0f} "
        f"S&amp;P 500 constituent-years: {cov['% firms valued'].mean():.0f}% of firms and "
        f"{cov['% mkt cap valued'].mean():.0f}% of market cap on average, "
        f"{cov['% mkt cap valued, ex-financials'].mean():.0f}% of non-financial market cap. Financials (excluded) are "
        f"{cov['% mkt cap financial'].mean():.0f}% of market cap on average (range "
        f"{cov['% mkt cap financial'].min():.0f}-{cov['% mkt cap financial'].max():.0f}%), so the ERP is that of the "
        f"non-financial S&amp;P 500.</p>")
    pit = [
        ("Capex proportional to sales forever (the Pi problem)",
         f"Capex = CSR x sales and depreciation = DSR x sales in every future state, with no link to the growth they "
         f"buy beyond mu_P = kappa R_A. Where industry capex/sales is close to the firm's gross margin ratio, cash flow "
         f"per $ of gross margin (Pi) is near or below zero and the firm is worth ~0 at any mu_Q. With the plain "
         f"industry mean this hit 29% of S&amp;P firm-years; after winsorizing and industry x distress averaging, "
         f"{flags['% own CSR/DSR (industry Pi <= 0)'].mean():.0f}% are valued with their own CSR/DSR "
         f"({src.loc['own', 'share of mkt cap %']:.1f}% of market cap) and 611 firm-years (3.7%) have no value even "
         f"then: utilities, oil &amp; gas, airlines. Firms on their own CSR show a median CoE - r of "
         f"{P(src.loc['own', 'CoE - r median'])} vs {P(src.loc['industry', 'CoE - r median'])}. Utilities get a "
         f"median CoE - r of {P(util['CoE - r median'])}, implausible for a low-beta sector: high kappa amplifies "
         f"R_A = (r - mu_Q)/(1 - kappa), and {util['% mu_Q within 1pp of r']:.0f}% of their roots sit within 1pp of r."),
        ("Growth ceiling: mu_Q must stay below r",
         f"The model needs mu_Q &lt; r. Firms priced for growth above that can only be matched with mu_Q just under "
         f"r, so R_A = (r - mu_Q)/(1 - kappa) collapses toward zero and CoE falls below r. This dominates the "
         f"value-weighted ERP in {worst} ({P(annual.loc[worst, 'ERP VW'])}): the three largest firms are {top_txt}. "
         f"{flags.loc[worst, '% mkt cap flagged']:.0f}% of that year's valued market cap carries a flag; excluding "
         f"flagged firms gives {P(annual.loc[worst, 'ERP VW, unflagged firms'])}. In the low-rate era (r near 1%) the "
         f"ceiling binds harder, which likely explains part of the post-2010 gap to Damodaran. Value-weighted mu_Q "
         f"is negative from 2002 on."),
        ("Cost of debt and book weights push CoE below r",
         f"CoE = (R_A - w_d CoD)/(1 - w_d). CoD is a bucket yield (AAA/BBB/BBB+2%), not firm-specific, and w_d mixes "
         f"book debt with market equity (EGZ's WACC has no tax shield). When the bucket yield exceeds the implied R_A, "
         f"leverage pushes CoE below R_A and often below r: {flags['% CoE < r'].mean():.0f}% of firm-years on average, "
         f"up to {flags['% CoE < r'].max():.0f}% ({flags['% CoE < r'].idxmax()}), and "
         f"{buck.loc['BBB+2% (CHS q5)', '% CoE < r']:.0f}% in the most distressed bucket. Leverage also widens the "
         f"spread of firm CoE (see the scatter)."),
        ("Small-firm problem",
         f"Industry averages of ratios are dominated by firms with near-zero sales (1983, SIC 67: mean capex/sales 5.74 "
         f"vs median 0.00). This is fixed by winsorizing at 5/95 and averaging within industry x CHS quintile, but "
         f"inputs stay sensitive to that choice. Within the S&amp;P, the smallest size quintile has a VW CoE - r of "
         f"{P(size.loc[1, 'CoE - r VW'])} vs {P(size.loc[5, 'CoE - r VW'])} for the largest, with higher book leverage "
         f"({size.loc[1, 'book D/(D+E) median']:.2f} vs {size.loc[5, 'book D/(D+E) median']:.2f}): partly a size "
         f"premium, partly leverage amplification. The paper's full CRSP/Compustat universe would be much more exposed "
         f"than this S&amp;P sample."),
        ("Financial firms",
         f"Gross margin is not a meaningful state variable for banks, insurers and investment trusts (gvkey 1062, an "
         f"SIC 6799 trust, gives R_A = 0.7%). They are excluded, which removes "
         f"{cov['% mkt cap financial'].mean():.0f}% of market cap on average and up to "
         f"{cov['% mkt cap financial'].max():.0f}%. The paper keeps all common stocks and reports robustness to "
         f"excluding financials."),
        ("Missing SG&amp;A treated as zero fixed costs",
         f"{100 * len(f0) / len(ok):.0f}% of valued firm-years ({100 * f0['me'].sum() / ok['me'].sum():.0f}% of "
         f"market cap) never report XSGA and get F = 0, assuming Compustat includes it in COGS; that is "
         f"{f0_util:.0f}% of utility firm-years. Zero F removes operating leverage, which lowers the value of the "
         f"default option."),
        ("Sales volatility includes seasonality",
         f"sigma is the std of quarter-on-quarter sales growth x 2, which picks up seasonal patterns. For retailers "
         f"this gives {s_ret['as_coded']:.2f} vs {s_ret['yoy']:.2f} with year-on-year growth ({s_oth['as_coded']:.2f} "
         f"vs {s_oth['yoy']:.2f} for other firms). Utilities' median sigma is also high, plausibly seasonal demand. Higher sigma raises "
         f"the option value and shifts mu_Q. Fix: year-on-year growth or seasonally adjusted sales."),
        ("Fragile roots",
         f"{flags['% mu_Q within 1pp of r'].mean():.1f}% of firm-years have mu_Q within 1pp of r. R_A is then driven "
         f"by the small difference r - mu_Q, so small changes in r or other inputs move it a lot."),
        ("Definitions and timing",
         f"r is the average of the 3-month and 10-year yields (EGZ), below the 10-year bond Damodaran uses, which adds "
         f"about {100 * (full['ERP VW'] - full['ERP VW vs 10y']):.1f}pp to the ERP. r and the bond yields are "
         f"calendar-year averages, not point-in-time; Compustat fiscal years are matched to December prices; "
         f"quarterly dates assume a December year-end; industry codes are header SICs; BAA is one annual value for "
         f"1983-85; CHS starts in 1983, so a few firms fall back to the AAA coupon. The market ERP is VW CoE - r over "
         f"valued non-financial firms, i.e. it treats this portfolio as the beta-1 market."),
    ]
    return head + "<ol>" + "".join(f"<li><b>{h}.</b> {b}</li>" for h, b in pit) + "</ol>"


# ----------------------------------------------------------------------------- report
def fig_b64(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"))
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def html_table(df, pct_cols=(), dec=3):
    fmt = df.copy()
    for c in fmt.columns:
        if fmt[c].dtype.kind in "fc":
            fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else (f"{100 * v:.2f}%" if c in pct_cols else
                                                                 f"{v:,.{dec}f}" if abs(v) < 1000 else f"{v:,.0f}"))
    return fmt.to_html(classes="t", border=0)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    ok, fails, rf, dam, rec = load()
    annual = annual_table(ok, rf, dam)
    cov, fme = coverage_table(ok, fails)
    pooled = pooled_stats(ok)
    periods = period_table(annual)
    flags = flag_table(ok)
    sec = group_table(ok, "sector", "sector")
    size = group_table(ok, "size_q", "S&P size quintile (1 = smallest)")
    buck = group_table(ok, "bucket", "coupon bucket")
    src = group_table(ok, "csr_source", "CSR/DSR source")
    fail_sector = fails.pivot_table(index="sector", columns="group", values="gvkey", aggfunc="count", fill_value=0)
    for name, t in [("annual_erp", annual), ("coverage", cov), ("pooled_stats", pooled), ("periods", periods),
                    ("flags", flags), ("by_sector", sec), ("by_size", size), ("by_coupon_bucket", buck),
                    ("by_csr_source", src), ("failures_by_sector", fail_sector)]:
        t.to_csv(os.path.join(OUT_DIR, f"{name}.csv"))
    ok.to_csv(os.path.join(OUT_DIR, "firm_years.csv"), index=False)

    figs = {"erp": fig_erp(annual, rec), "components": fig_components(annual, rf, rec), "coverage": fig_coverage(cov, fme),
            "box": fig_box(ok), "leverage": fig_leverage(ok), "flags": fig_flags(flags), "sector": fig_sector(sec)}
    img = {k: fig_b64(f, k) for k, f in figs.items()}
    pct_annual = [c for c in annual.columns if c != "n_valued"]
    seas = seasonality()
    seas.to_csv(os.path.join(OUT_DIR, "sigma_seasonality.csv"))
    sections = [
        ("Key findings and model pitfalls", findings_html(ok, annual, periods, cov, flags, sec, size, buck, src, seas)),
        ("Implied ERP over time", f'<img src="data:image/png;base64,{img["erp"]}">'
                                  f'<img src="data:image/png;base64,{img["components"]}">'),
        ("Sub-period averages", html_table(periods, pct_cols=[c for c in periods.columns if not c.startswith("corr")])),
        ("Annual results", html_table(annual, pct_cols=pct_annual)),
        ("Coverage", f'<img src="data:image/png;base64,{img["coverage"]}">' + html_table(cov, dec=1)),
        ("Pooled summary statistics (valued firm-years)", html_table(pooled)),
        ("Cross-section", f'<img src="data:image/png;base64,{img["box"]}">'
                          f'<img src="data:image/png;base64,{img["leverage"]}">'),
        ("By sector", f'<img src="data:image/png;base64,{img["sector"]}">' + html_table(sec, dec=3)),
        ("By size quintile", html_table(size)), ("By coupon bucket", html_table(buck)),
        ("By CSR/DSR source", html_table(src)),
        ("Failures by sector (firm-years)", html_table(fail_sector)),
        ("Fragility flags", f'<img src="data:image/png;base64,{img["flags"]}">' + html_table(flags, dec=1)),
    ]
    css = ("body{font-family:Segoe UI,system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#0b0b0b;"
           "background:#fcfcfb}img{max-width:100%;display:block;margin:12px 0}table.t{border-collapse:collapse;"
           "font-size:12px;margin:8px 0 24px;display:block;overflow-x:auto}table.t td,table.t th{padding:3px 8px;"
           "border-bottom:1px solid #e4e3df;text-align:right;white-space:nowrap}table.t th{color:#52514e}"
           "h2{margin-top:36px;border-bottom:1px solid #e4e3df}li{margin:8px 0;line-height:1.45}p{line-height:1.5}")
    html = (f"<!doctype html><meta charset='utf-8'><title>EGZ ERP results</title><style>{css}</style>"
            f"<h1>EGZ-implied ERP: S&amp;P 500, 1983-2022</h1>"
            + "".join(f"<h2>{h}</h2>{b}" for h, b in sections))
    with open(os.path.join(OUT_DIR, "report.html"), "w", encoding="utf-8") as fh:
        fh.write(html)
    print(periods.round(4).to_string())
    print(f"\nwrote {OUT_DIR}/report.html, {len(figs)} figures, tables as CSV")
