"""
Publication figures for the manuscript.

Figures use a vivid, colorblind-safe qualitative palette (Okabe and Ito) for
categorical marks and the perceptually uniform viridis map for the coverage
heatmap. Each figure is written as both PDF (vector, for the manuscript) and PNG
(for preview), and every figure is generated from the reproduced result tables and
the real panel.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
DATA = ROOT / "data_reference"
FIG.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-safe palette.
BLUE = "#0072B2"
VERM = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
INK = "#222222"
MUTE = "#c7c7c7"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 10,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.dpi": 150,
    }
)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[fig] {name}")


def fig_leadlag_gdelt():
    prof = pd.read_csv(TAB / "leadlag_profile.csv")
    g = prof[prof.source == "gdelt"].sort_values("lag")
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    colors = [BLUE if p < 0.05 else MUTE for p in g.p_value]
    ax.bar(g.lag, g.pearson_r, color=colors, edgecolor=INK, linewidth=0.7, width=0.7)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.axvline(-2, color=VERM, linestyle=":", linewidth=1.3)
    ax.set_xlabel("Lag $k$ (days relative to FOMC meeting)")
    ax.set_ylabel("Pearson $r$ with post-meeting volatility")
    ax.set_xticks(range(-3, 4))
    peak = g[g.lag == -2].iloc[0]
    ax.annotate(
        f"$r={peak.pearson_r:.3f}$\n$p={peak.p_value:.3f}$",
        xy=(-2, peak.pearson_r), xytext=(-1.7, peak.pearson_r + 0.06), fontsize=8.5,
    )
    handles = [
        mpl.patches.Patch(facecolor=BLUE, edgecolor=INK, label="$p<0.05$"),
        mpl.patches.Patch(facecolor=MUTE, edgecolor=INK, label="not significant"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper right")
    save(fig, "fig1_leadlag_gdelt")


def fig_leadlag_sources():
    prof = pd.read_csv(TAB / "leadlag_profile.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    styles = {
        "gdelt": (BLUE, "o", "-", "GDELT"),
        "guardian": (VERM, "s", "--", "The Guardian"),
        "nyt": (GREEN, "^", ":", "New York Times"),
    }
    for src, (c, mk, ls, lab) in styles.items():
        d = prof[prof.source == src].sort_values("lag")
        ax.plot(d.lag, d.pearson_r, color=c, marker=mk, linestyle=ls,
                markersize=4, linewidth=1.3, label=lab, alpha=0.9)
        sig = d[d.p_value < 0.05]
        ax.scatter(sig.lag, sig.pearson_r, facecolor=c, edgecolor=INK,
                   s=48, zorder=5, linewidth=0.8)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xlabel("Lag $k$ (days relative to FOMC meeting)")
    ax.set_ylabel("Pearson $r$ with post-meeting volatility")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    save(fig, "fig2_leadlag_sources")


def fig_coef_path():
    reg = pd.read_csv(TAB / "main_regression.csv")
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    x = np.arange(len(reg))
    ci_lo = reg.beta_sim_m2 - 1.96 * reg.se_HC1
    ci_hi = reg.beta_sim_m2 + 1.96 * reg.se_HC1
    sig = reg.p_HC1 < 0.05
    ax.errorbar(x, reg.beta_sim_m2,
                yerr=[reg.beta_sim_m2 - ci_lo, ci_hi - reg.beta_sim_m2],
                fmt="none", ecolor=INK, capsize=3, linewidth=1.0, zorder=2)
    ax.scatter(x[sig], reg.beta_sim_m2[sig], color=BLUE, s=55, zorder=3,
               edgecolor=INK, linewidth=0.7, label="$p<0.05$")
    ax.scatter(x[~sig], reg.beta_sim_m2[~sig], color=MUTE, s=55, zorder=3,
               edgecolor=INK, linewidth=0.7, label="not significant")
    ax.axhline(0, color=VERM, linewidth=1.2, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(reg.spec)
    ax.set_xlabel("Specification (controls added left to right)")
    ax.set_ylabel(r"Coefficient on $\mathrm{sim}_{-2}$")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    save(fig, "fig3_coefficient_path")


def fig_placebo():
    panel = pd.read_csv(TAB / "panel_meeting.csv")
    g = panel[panel.source == "gdelt"].dropna(
        subset=["sim_m2", "vol_post", "vol_pre", "vix_pre"]
    )
    rng = np.random.default_rng(30113)
    x = g.sim_m2.to_numpy(); y = g.vol_post.to_numpy()
    r_obs = pearsonr(x, y)[0]
    null = np.array([pearsonr(x, rng.permutation(y))[0] for _ in range(20000)])
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.hist(null, bins=45, color=SKY, edgecolor="white", linewidth=0.3, alpha=0.9)
    ax.axvline(r_obs, color=VERM, linewidth=2.0)
    ax.annotate(f"observed\n$r={r_obs:.3f}$", xy=(r_obs, 0),
                xytext=(r_obs - 0.02, ax.get_ylim()[1] * 0.72),
                ha="right", fontsize=8.5, color=VERM)
    ax.set_xlabel("Pearson $r$ under random re-pairing (permutation null)")
    ax.set_ylabel("Frequency")
    save(fig, "fig4_placebo_null")


def fig_scatter():
    panel = pd.read_csv(TAB / "panel_meeting.csv")
    g = panel[panel.source == "gdelt"].dropna(subset=["sim_m2", "vol_post"])
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.scatter(g.sim_m2, g.vol_post, facecolor=BLUE, edgecolor=INK,
               s=38, linewidth=0.6, alpha=0.85)
    b1, b0 = np.polyfit(g.sim_m2, g.vol_post, 1)
    xs = np.linspace(g.sim_m2.min(), g.sim_m2.max(), 50)
    ax.plot(xs, b0 + b1 * xs, color=VERM, linewidth=1.8)
    ax.set_xlabel(r"Pre-meeting similarity $\mathrm{sim}_{-2}$ (GDELT)")
    ax.set_ylabel("Post-meeting realized volatility")
    save(fig, "fig5_scatter_sim_vol")


def fig_hawkish_timeline():
    cmp = pd.read_csv(TAB / "sentiment_comparison.csv")
    cmp["meeting_date"] = pd.to_datetime(cmp["meeting_date"])
    f = cmp.sort_values("meeting_date")
    mean_net = f.lm_net_per_1k.mean()
    dev = f.lm_net_per_1k - mean_net
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.fill_between(f.meeting_date, mean_net + dev, mean_net, where=dev >= 0,
                    color=VERM, alpha=0.55, interpolate=True,
                    label="above-average tone")
    ax.fill_between(f.meeting_date, mean_net + dev, mean_net, where=dev < 0,
                    color=BLUE, alpha=0.55, interpolate=True,
                    label="below-average tone")
    ax.plot(f.meeting_date, f.lm_net_per_1k, color=INK, linewidth=1.0)
    ax.axhline(mean_net, color=INK, linewidth=0.8, linestyle="--")
    ax.set_xlabel("FOMC meeting date")
    ax.set_ylabel("Net market-dictionary tone per 1,000 tokens")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    save(fig, "fig6_hawkish_timeline")


def fig_sentiment_agreement():
    cmp = pd.read_csv(TAB / "sentiment_comparison.csv")
    d = cmp.dropna(subset=["lm_neg_per_1k", "vix_pre"])
    fig, ax = plt.subplots(figsize=(4.9, 3.4))
    ax.scatter(d.lm_neg_per_1k, d.vix_pre, facecolor=GREEN,
               edgecolor=INK, s=38, linewidth=0.6, alpha=0.85)
    b1, b0 = np.polyfit(d.lm_neg_per_1k, d.vix_pre, 1)
    xs = np.linspace(d.lm_neg_per_1k.min(), d.lm_neg_per_1k.max(), 50)
    ax.plot(xs, b0 + b1 * xs, color=VERM, linewidth=1.8)
    r = pearsonr(d.lm_neg_per_1k, d.vix_pre)[0]
    ax.set_xlabel("Minutes negative tone, Loughran-McDonald (per 1,000 tokens)")
    ax.set_ylabel("Pre-meeting VIX")
    ax.annotate(f"$r={r:.2f}$", xy=(0.06, 0.9), xycoords="axes fraction", fontsize=9)
    save(fig, "fig7_sentiment_agreement")


def fig_coverage():
    cov = pd.read_csv(TAB / "coverage_diagnostic.csv", index_col=0)[["gdelt", "guardian", "nyt"]]
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    vmax = np.nanmax(cov.values)
    im = ax.imshow(cov.T.values, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    ax.set_yticks(range(3)); ax.set_yticklabels(["GDELT", "Guardian", "NYT"])
    ax.set_xticks(range(len(cov.index))); ax.set_xticklabels(cov.index, fontsize=8)
    ax.set_xlabel("Lag $k$ (days)")
    for i in range(cov.shape[1]):
        for j in range(cov.shape[0]):
            v = cov.values[j, i]
            if not np.isnan(v):
                ax.text(j, i, int(v), ha="center", va="center", fontsize=7,
                        color="white" if v < vmax * 0.55 else INK)
    fig.colorbar(im, ax=ax, shrink=0.7, label="meetings with coverage")
    save(fig, "fig8_coverage")


def fig_altwindow():
    a = pd.read_csv(TAB / "altwindow.csv")
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    x = np.arange(len(a)); w = 0.38
    ax.bar(x - w / 2, a.beta_bivariate, w, color=BLUE, edgecolor=INK,
           linewidth=0.6, label="bivariate")
    ax.bar(x + w / 2, a.beta_controlled, w, color=ORANGE, edgecolor=INK,
           linewidth=0.6, label="controlled (vol., VIX)")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{int(v)}-day" for v in a.window])
    ax.set_xlabel("Post-meeting volatility window")
    ax.set_ylabel(r"Coefficient on $\mathrm{sim}_{-2}$")
    ax.legend(frameon=False, fontsize=8.5)
    save(fig, "fig9_altwindow")


def main():
    fig_leadlag_gdelt()
    fig_leadlag_sources()
    fig_coef_path()
    fig_placebo()
    fig_scatter()
    fig_hawkish_timeline()
    fig_sentiment_agreement()
    fig_coverage()
    fig_altwindow()


if __name__ == "__main__":
    main()
