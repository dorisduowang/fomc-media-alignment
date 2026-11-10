"""
Identification battery: permutation placebo, Granger causality, VAR, bootstrap,
and subsample robustness.

The unconditional pre-meeting similarity signal is strong, but the paper does not
claim a causal media-to-volatility channel. This script implements the checks
that discipline that interpretation.

    1. Permutation placebo: randomly re-pair sim_{-2} with post-meeting volatility
       and locate the observed statistic in the permutation null.
    2. Granger causality: test whether pre-meeting similarity Granger-causes
       post-meeting volatility across the chronologically ordered GDELT meetings.
    3. VAR(2): reduced-form dynamics of (similarity, volatility, VIX) with an
       impulse-response summary.
    4. Bootstrap: pairs-bootstrap confidence interval for the sim_{-2} coefficient
       in the bivariate and fully controlled specifications.
    5. Subsamples: Yellen vs Powell, high vs low pre-meeting VIX, and a
       leave-one-out check that drops each meeting in turn.

Outputs land in results/tables/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"
DATA = ROOT / "data_reference"
RNG = np.random.default_rng(30113)


def gdelt_sample():
    panel = pd.read_csv(TAB / "panel_meeting.csv", parse_dates=["meeting_date"])
    g = panel[panel["source"] == "gdelt"].copy()
    g = g.dropna(subset=["sim_m2", "vol_post", "vol_pre", "vix_pre", "sent_m2"])
    return g.sort_values("meeting_date").reset_index(drop=True)


def permutation_placebo(g, n_perm=20000):
    """Randomization null for the sim_{-2} to volatility association."""
    x = g["sim_m2"].to_numpy()
    y = g["vol_post"].to_numpy()
    r_obs = pearsonr(x, y)[0]
    # controlled statistic: partial correlation given vol_pre and vix_pre
    Z = sm.add_constant(g[["vol_pre", "vix_pre"]].to_numpy())
    rx = x - sm.OLS(x, Z).fit().predict(Z)
    ry = y - sm.OLS(y, Z).fit().predict(Z)
    rp_obs = pearsonr(rx, ry)[0]

    r_null = np.empty(n_perm)
    rp_null = np.empty(n_perm)
    for i in range(n_perm):
        perm = RNG.permutation(len(y))
        r_null[i] = pearsonr(x, y[perm])[0]
        rp_null[i] = pearsonr(rx, ry[perm])[0]

    res = pd.DataFrame(
        {
            "statistic": ["raw_corr", "partial_corr_given_vol_vix"],
            "observed": [r_obs, rp_obs],
            "perm_p_two_sided": [
                np.mean(np.abs(r_null) >= abs(r_obs)),
                np.mean(np.abs(rp_null) >= abs(rp_obs)),
            ],
            "null_mean": [r_null.mean(), rp_null.mean()],
            "null_sd": [r_null.std(), rp_null.std()],
        }
    )
    res.to_csv(TAB / "placebo_permutation.csv", index=False)
    print("[placebo]\n", res.to_string(index=False))
    return res


def granger(g, maxlag=3):
    """Granger test: does similarity precede volatility across ordered meetings."""
    series = g[["vol_post", "sim_m2"]].to_numpy()  # test sim_m2 -> vol_post
    out = grangercausalitytests(series, maxlag=maxlag, verbose=False)
    rows = []
    for lag in range(1, maxlag + 1):
        f = out[lag][0]["ssr_ftest"]
        rows.append({"lag": lag, "F": f[0], "p_value": f[1], "df_num": f[3], "df_den": f[2]})
    res = pd.DataFrame(rows)
    res.to_csv(TAB / "granger.csv", index=False)
    print("[granger] sim_-2 -> vol_post\n", res.to_string(index=False))
    return res


def var_model(g):
    """Reduced-form VAR(2) on standardized (similarity, volatility, VIX)."""
    d = g[["sim_m2", "vol_post", "vix_pre"]].copy()
    d = (d - d.mean()) / d.std()
    model = VAR(d)
    fit = model.fit(2)
    irf = fit.irf(6)
    # cumulative response of volatility to a similarity shock
    resp = irf.irfs[:, 1, 0]  # response of var index 1 (vol) to shock in index 0 (sim)
    res = pd.DataFrame({"horizon": np.arange(len(resp)), "vol_resp_to_sim_shock": resp})
    res.to_csv(TAB / "var_irf.csv", index=False)
    with open(TAB / "var_summary.txt", "w") as fh:
        fh.write(str(fit.summary()))
    print("[var] cumulative vol response to sim shock:", resp.cumsum()[-1].round(3))
    return res


def bootstrap(g, n_boot=10000):
    """Pairs bootstrap for the sim_{-2} coefficient."""
    def beta(df, rhs):
        return smf.ols("vol_post ~ " + " + ".join(rhs), data=df).fit().params["sim_m2"]

    specs = {
        "bivariate": ["sim_m2"],
        "controlled": ["sim_m2", "vol_pre", "vix_pre"],
    }
    rows = []
    n = len(g)
    for label, rhs in specs.items():
        b0 = beta(g, rhs)
        draws = np.empty(n_boot)
        for i in range(n_boot):
            idx = RNG.integers(0, n, n)
            draws[i] = beta(g.iloc[idx], rhs)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append(
            {
                "spec": label,
                "beta": b0,
                "boot_se": draws.std(),
                "ci_lo": lo,
                "ci_hi": hi,
                "share_positive": np.mean(draws > 0),
            }
        )
    res = pd.DataFrame(rows)
    res.to_csv(TAB / "bootstrap.csv", index=False)
    print("[bootstrap]\n", res.to_string(index=False))
    return res


def subsamples(g):
    rows = []

    def one(df, label, rhs):
        if len(df) < len(rhs) + 3:
            return
        fit = smf.ols("vol_post ~ " + " + ".join(rhs), data=df).fit(
            cov_type="HC1"
        )
        rows.append(
            {
                "subsample": label,
                "n": int(fit.nobs),
                "beta_sim_m2": fit.params["sim_m2"],
                "se": fit.bse["sim_m2"],
                "p": fit.pvalues["sim_m2"],
                "controls": "+".join(rhs),
            }
        )

    biv = ["sim_m2"]
    ctrl = ["sim_m2", "vol_pre", "vix_pre"]
    one(g, "full (bivariate)", biv)
    one(g, "full (controlled)", ctrl)
    one(g[g.chair == "Yellen"], "Yellen (bivariate)", biv)
    one(g[g.chair == "Powell"], "Powell (bivariate)", biv)
    med = g["vix_pre"].median()
    one(g[g.vix_pre >= med], "high VIX (bivariate)", biv)
    one(g[g.vix_pre < med], "low VIX (bivariate)", biv)
    # leave-one-out range for the bivariate slope
    slopes = []
    for i in range(len(g)):
        slopes.append(smf.ols("vol_post ~ sim_m2", data=g.drop(g.index[i])).fit().params["sim_m2"])
    rows.append(
        {
            "subsample": "leave-one-out range (bivariate)",
            "n": len(g),
            "beta_sim_m2": np.mean(slopes),
            "se": np.std(slopes),
            "p": np.nan,
            "controls": f"min={min(slopes):.3f}, max={max(slopes):.3f}",
        }
    )
    res = pd.DataFrame(rows)
    res.to_csv(TAB / "subsamples.csv", index=False)
    print("[subsamples]\n", res.to_string(index=False))
    return res


def main():
    g = gdelt_sample()
    print(f"[identification] GDELT estimation sample: n={len(g)}, "
          f"{g.meeting_date.min().date()} to {g.meeting_date.max().date()}\n")
    permutation_placebo(g)
    print()
    granger(g)
    print()
    var_model(g)
    print()
    bootstrap(g)
    print()
    subsamples(g)


if __name__ == "__main__":
    main()
