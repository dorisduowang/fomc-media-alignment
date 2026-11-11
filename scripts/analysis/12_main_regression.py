"""
Main lead-lag regression and the full lag profile.

The dependent variable is post-meeting realized S&P 500 volatility. The regressor
of interest is pre-meeting media-FOMC textual similarity two days before the
meeting, sim_{-2}. We estimate a progressive set of specifications on the GDELT
sample, the only corpus with complete pre-meeting coverage at a common lag, and
report both heteroskedasticity-robust (HC1) and Newey-West HAC standard errors.

Outputs:
    results/tables/leadlag_profile.csv     Pearson r by source and lag
    results/tables/main_regression.tex     progressive OLS table (GDELT)
    results/tables/main_regression.csv     machine-readable coefficients
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"
DATA = ROOT / "data_reference"


def leadlag_profile():
    pm = pd.read_parquet(DATA / "per_meeting.parquet")
    rows = []
    for src, d in pm.groupby("source"):
        for k in sorted(d["lag_days"].unique()):
            sub = d[d["lag_days"] == k].dropna(subset=["cosine_sim", "vol_post"])
            if len(sub) >= 5:
                r, p = pearsonr(sub["cosine_sim"], sub["vol_post"])
                rows.append(
                    {
                        "source": src,
                        "lag": int(k),
                        "n": len(sub),
                        "mean_cosine": sub["cosine_sim"].mean(),
                        "pearson_r": r,
                        "p_value": p,
                    }
                )
    prof = pd.DataFrame(rows)
    prof.to_csv(TAB / "leadlag_profile.csv", index=False)
    return prof


def _fit(df, rhs):
    formula = "vol_post ~ " + " + ".join(rhs)
    ols = smf.ols(formula, data=df).fit()
    hc1 = ols.get_robustcov_results(cov_type="HC1")
    hac = ols.get_robustcov_results(cov_type="HAC", maxlags=2, use_correction=True)
    return ols, hc1, hac


def _fmt(coef, se, p):
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    return f"{coef:.3f}{stars}", f"({se:.3f})"


SPECS = [
    ("(1)", ["sim_m2"]),
    ("(2)", ["sim_m2", "vol_pre"]),
    ("(3)", ["sim_m2", "vol_pre", "vix_pre"]),
    ("(4)", ["sim_m2", "vol_pre", "vix_pre", "sent_m2"]),
    ("(5)", ["sim_m2", "vol_pre", "vix_pre", "sent_m2", "mps_orth"]),
    ("(6)", ["sim_m2", "vol_pre", "vix_pre", "sent_m2", "mps_orth", "C(year)"]),
]

# rows of control indicators for the table body
CONTROL_ROWS = [
    ("Pre-meeting volatility", "vol_pre"),
    ("Pre-meeting VIX", "vix_pre"),
    ("Pre-meeting sentiment", "sent_m2"),
    ("Policy surprise (orth.)", "mps_orth"),
    ("Year fixed effects", "C(year)"),
]


def main_regression():
    panel = pd.read_csv(TAB / "panel_meeting.csv", parse_dates=["meeting_date"])
    g = panel[panel["source"] == "gdelt"].copy()
    g = g.dropna(subset=["sim_m2", "vol_post", "vol_pre", "vix_pre", "sent_m2", "mps_orth"])
    g["year"] = g["meeting_date"].dt.year

    records = []
    tex_cols = {}
    for name, rhs in SPECS:
        ols, hc1, hac = _fit(g, rhs)
        idx = hc1.model.exog_names.index("sim_m2")
        b, se_hc, p_hc = hc1.params[idx], hc1.bse[idx], hc1.pvalues[idx]
        se_hac, p_hac = hac.bse[idx], hac.pvalues[idx]
        records.append(
            {
                "spec": name, "beta_sim_m2": b, "se_HC1": se_hc, "p_HC1": p_hc,
                "se_HAC": se_hac, "p_HAC": p_hac, "r2": ols.rsquared,
                "n": int(ols.nobs), "controls": "+".join(rhs),
            }
        )
        c, s = _fmt(b, se_hc, p_hc)
        _, s_hac = _fmt(b, se_hac, p_hac)
        tex_cols[name] = (c, s, s_hac, f"{ols.rsquared:.3f}", str(int(ols.nobs)))

    out = pd.DataFrame(records)
    out.to_csv(TAB / "main_regression.csv", index=False)
    print(out.to_string(index=False))

    ncol = len(SPECS)
    align = "l" + "c" * ncol
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule"]
    lines.append(" & " + " & ".join(name for name, _ in SPECS) + r" \\")
    lines.append(r"\midrule")
    row_b, row_hc, row_hac = r"$\mathrm{sim}_{-2}$", r"\quad HC1 SE", r"\quad HAC SE"
    for name, _ in SPECS:
        c, s, s_hac, _, _ = tex_cols[name]
        row_b += f" & {c}"; row_hc += f" & {s}"; row_hac += f" & {s_hac}"
    lines += [row_b + r" \\", row_hc + r" \\", row_hac + r" \\", r"\midrule"]
    for label, key in CONTROL_ROWS:
        marks = "".join(
            " & \\checkmark" if key in rhs else " & " for _, rhs in SPECS
        )
        lines.append(label + marks + r" \\")
    lines.append(r"\midrule")
    r2row, nrow = r"$R^2$", r"$N$"
    for name, _ in SPECS:
        _, _, _, r2, n = tex_cols[name]
        r2row += f" & {r2}"; nrow += f" & {n}"
    lines += [r2row + r" \\", nrow + r" \\", r"\bottomrule", r"\end{tabular}"]
    (TAB / "main_regression.tex").write_text("\n".join(lines))
    print(f"\n[main] wrote {TAB / 'main_regression.tex'}")
    return out


def alt_window_robustness():
    """Bivariate and controlled sim_-2 slope at 3, 5, 7, 10-day vol windows."""
    panel = pd.read_csv(TAB / "panel_meeting.csv", parse_dates=["meeting_date"])
    g = panel[panel["source"] == "gdelt"].copy()
    rows = []
    for w in (3, 5, 7, 10):
        dv = f"vol_post_{w}"
        d = g.dropna(subset=["sim_m2", dv, "vol_pre", "vix_pre"])
        biv = smf.ols(f"{dv} ~ sim_m2", data=d).fit(cov_type="HC1")
        ctr = smf.ols(f"{dv} ~ sim_m2 + vol_pre + vix_pre", data=d).fit(cov_type="HC1")
        rows.append(
            {
                "window": w, "n": int(biv.nobs),
                "beta_bivariate": biv.params["sim_m2"], "p_bivariate": biv.pvalues["sim_m2"],
                "beta_controlled": ctr.params["sim_m2"], "p_controlled": ctr.pvalues["sim_m2"],
            }
        )
    res = pd.DataFrame(rows)
    res.to_csv(TAB / "altwindow.csv", index=False)
    print("\n[altwindow]\n", res.to_string(index=False))
    return res


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        from temporal_validation import regression_cli
        regression_cli()
        raise SystemExit(0)
    prof = leadlag_profile()
    print("[leadlag] GDELT profile:")
    print(prof[prof.source == "gdelt"].to_string(index=False))
    print()
    main_regression()
    alt_window_robustness()
