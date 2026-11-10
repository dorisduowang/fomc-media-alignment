"""
Build the meeting-level analysis panel and diagnose sample coverage.

This script assembles the estimation panel used throughout the paper from three
inputs that the distributed pipeline produces:

    data_reference/per_meeting.parquet  one row per (meeting, lag, source) with
                                        cosine similarity, mean sentiment, and
                                        pre/post realized volatility
    data_reference/sentiment.parquet    document-level hawkish-dovish scores
    data_reference/market/VIX_2014_2025.csv   daily CBOE VIX from FRED

It produces two artifacts:

    results/tables/panel_meeting.csv    the wide meeting x source panel with the
                                        pre-meeting similarity regressor sim_{-2},
                                        controls, fixed-effect keys, and regime
                                        indicators
    results/tables/coverage_diagnostic.csv  the by-source, by-lag observation
                                        counts that document the coverage
                                        structure of each corpus

The coverage diagnostic is the reproducible answer to the "sample-size" question:
it shows exactly which meetings each corpus reaches and at which lags, so that
every downstream sample restriction is transparent rather than silent.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_reference"
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

# Chair tenure. Yellen chaired through the 31 January 2018 meeting; Powell from
# the March 2018 meeting onward. Meetings are assigned by date.
POWELL_START = pd.Timestamp("2018-02-05")

# Regime windows used for the heterogeneity tests.
COVID_START, COVID_END = pd.Timestamp("2020-03-01"), pd.Timestamp("2020-12-31")
HIKE_START, HIKE_END = pd.Timestamp("2022-03-01"), pd.Timestamp("2023-07-31")


def load_similarity_panel():
    """Return the long (meeting, lag, source) panel with real volatility."""
    pm = pd.read_parquet(DATA / "per_meeting.parquet")
    pm["meeting_date"] = pd.to_datetime(pm["meeting_date"])
    return pm


def coverage_diagnostic(pm):
    """Observation counts by source and lag, plus a per-source summary."""
    counts = (
        pm.dropna(subset=["cosine_sim"])
        .groupby(["source", "lag_days"])
        .size()
        .unstack("source")
        .sort_index()
    )
    summary = (
        pm.dropna(subset=["cosine_sim"])
        .groupby("source")
        .agg(
            n_meetings=("meeting_date", "nunique"),
            first_meeting=("meeting_date", "min"),
            last_meeting=("meeting_date", "max"),
            n_obs=("cosine_sim", "size"),
        )
    )
    return counts, summary


def add_vix(panel):
    """Merge the pre-meeting VIX level (last close strictly before the meeting)."""
    vix = pd.read_csv(DATA / "market" / "VIX_2014_2025.csv")
    vix.columns = ["date", "vix"]
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix.dropna(subset=["vix"]).sort_values("date").reset_index(drop=True)

    def pre_vix(meeting):
        prior = vix[vix["date"] < meeting]
        return prior["vix"].iloc[-1] if len(prior) else np.nan

    panel["vix_pre"] = panel["meeting_date"].map(pre_vix)
    return panel


def build_wide(pm):
    """Collapse the long panel to one row per (meeting, source).

    The pre-meeting regressor is the two-day-ahead cosine similarity, sim_{-2},
    the pre-specified baseline lag. We also retain the meeting-day similarity and
    the mean of the three pre-meeting lags as alternative regressors, and carry
    the sentiment measured over the same pre-meeting window.
    """
    def at_lag(df, k, col):
        s = df[df["lag_days"] == k].set_index("meeting_date")[col]
        return s

    rows = []
    for src, d in pm.groupby("source"):
        meetings = sorted(d["meeting_date"].unique())
        sim_m2 = at_lag(d, -2, "cosine_sim")
        sim_0 = at_lag(d, 0, "cosine_sim")
        sent_m2 = at_lag(d, -2, "mean_net_per_1k")
        # pre-window average similarity over lags -3, -2, -1 where available
        pre = (
            d[d["lag_days"].isin([-3, -2, -1])]
            .groupby("meeting_date")["cosine_sim"]
            .mean()
        )
        vol = d.groupby("meeting_date")[["vol_pre", "vol_post"]].first()
        for m in meetings:
            rows.append(
                {
                    "meeting_date": m,
                    "source": src,
                    "sim_m2": sim_m2.get(m, np.nan),
                    "sim_0": sim_0.get(m, np.nan),
                    "sim_pre_mean": pre.get(m, np.nan),
                    "sent_m2": sent_m2.get(m, np.nan),
                    "vol_pre": vol.loc[m, "vol_pre"] if m in vol.index else np.nan,
                    "vol_post": vol.loc[m, "vol_post"] if m in vol.index else np.nan,
                }
            )
    wide = pd.DataFrame(rows)
    wide["meeting_date"] = pd.to_datetime(wide["meeting_date"])
    return wide


def add_mps(panel):
    """Merge the Bauer-Swanson (2023) high-frequency monetary policy surprise.

    MPS is the surprise in the expected path of the funds rate over the coming
    year; MPS_ORTH is the component orthogonal to public information. Each meeting
    is matched to the surprise observation within one calendar day of its date.
    """
    mps = pd.read_csv(DATA / "market" / "mps_fomc.csv")
    mps["Date"] = pd.to_datetime(mps["Date"])
    mps = mps.dropna(subset=["MPS"]).sort_values("Date")

    def nearest(meeting):
        diff = (mps["Date"] - meeting).abs()
        j = diff.idxmin()
        if diff.loc[j] <= pd.Timedelta(days=1):
            return mps.loc[j, "MPS"], mps.loc[j, "MPS_ORTH"]
        return np.nan, np.nan

    vals = panel["meeting_date"].map(lambda m: nearest(m))
    panel["mps"] = [v[0] for v in vals]
    panel["mps_orth"] = [v[1] for v in vals]
    panel["abs_mps"] = panel["mps"].abs()
    return panel


def add_alt_window_vol(panel):
    """Recompute realized volatility at 3, 5, 7, and 10 trading-day windows.

    Volatility is the annualized standard deviation of daily log returns over the
    trading days strictly after (pre: strictly before) the meeting, computed from
    the daily S&P 500 close.
    """
    spx = pd.read_csv(DATA / "market" / "SPX_daily_2014_2025.csv")
    spx["date"] = pd.to_datetime(spx["date"])
    spx = spx.dropna(subset=["spx"]).sort_values("date").reset_index(drop=True)
    spx["logret"] = np.log(spx["spx"]).diff()
    rets = spx.dropna(subset=["logret"]).reset_index(drop=True)

    def rv(meeting, w, post=True):
        if post:
            sub = rets[rets["date"] > meeting]["logret"].iloc[:w]
        else:
            sub = rets[rets["date"] < meeting]["logret"].iloc[-w:]
        if len(sub) < max(2, w - 1):
            return np.nan
        return float(sub.std(ddof=1) * np.sqrt(252))

    for w in (3, 5, 7, 10):
        panel[f"vol_post_{w}"] = panel["meeting_date"].map(lambda m: rv(m, w, True))
        panel[f"vol_pre_{w}"] = panel["meeting_date"].map(lambda m: rv(m, w, False))
    return panel


def add_fixed_effects(panel):
    panel["year"] = panel["meeting_date"].dt.year
    panel["chair"] = np.where(panel["meeting_date"] >= POWELL_START, "Powell", "Yellen")
    d = panel["meeting_date"]
    panel["covid"] = ((d >= COVID_START) & (d <= COVID_END)).astype(int)
    panel["hike_cycle"] = ((d >= HIKE_START) & (d <= HIKE_END)).astype(int)
    return panel


def main():
    pm = load_similarity_panel()

    counts, summary = coverage_diagnostic(pm)
    counts.to_csv(OUT / "coverage_diagnostic.csv")
    summary.to_csv(OUT / "coverage_summary.csv")
    print("[panel] coverage by source:")
    print(summary.to_string())
    print("\n[panel] observation counts by source and lag:")
    print(counts.to_string())

    wide = build_wide(pm)
    wide = add_vix(wide)
    wide = add_mps(wide)
    wide = add_alt_window_vol(wide)
    wide = add_fixed_effects(wide)

    # High-VIX regime: top quartile of pre-meeting VIX within the estimation
    # sample of each source is computed downstream; here we tag the global
    # top-quartile threshold for reference.
    thr = wide["vix_pre"].quantile(0.75)
    wide["high_vix"] = (wide["vix_pre"] >= thr).astype(int)

    wide = wide.sort_values(["source", "meeting_date"]).reset_index(drop=True)
    wide.to_csv(OUT / "panel_meeting.csv", index=False)
    print(f"\n[panel] wrote {OUT / 'panel_meeting.csv'} with {len(wide)} rows")

    # A compact estimation sample per source: complete cases on the baseline
    # specification (sim_{-2}, vol_post, vol_pre, vix_pre).
    est = wide.dropna(subset=["sim_m2", "vol_post", "vol_pre", "vix_pre"])
    print("\n[panel] complete-case estimation sample sizes (sim_-2 model):")
    print(est.groupby("source").size().to_string())


if __name__ == "__main__":
    main()
