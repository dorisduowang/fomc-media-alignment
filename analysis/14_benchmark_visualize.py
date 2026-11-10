"""
MACS 30113
Yuzhao (Jerry) Jin; CNETID: yuzhaojin1
Duo (Doris) Wang; CNETID: dorisduowang

This file generates initial figures for initial lead-lag analysis and benchmark differences: 
  Figure 1: Lead-lag Pearson r vs S&P 500 volatility by text sources
  Figure 2: Benchmark bar chart (Python vs NumPy vs Numba)

Utilizing course contents:
- Week 3: matplotlib figure generation and S3 upload 
- Week 5: reading S3 parquet outputs from prior pipeline stages via boto3
"""
import io
import os
from pathlib import Path

import boto3
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
BUCKET = os.environ.get("BUCKET", "")

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Load data from S3
def load_csv_from_s3(bucket, key):
    s3 = boto3.client("s3")
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_csv(io.BytesIO(body))

def upload_figure(bucket, local_path, s3_key):
    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, s3_key)
    print(f"[s3] uploaded {s3_key}")

# Figure 1: Lead-lag Pearson r by source
def plot_lead_lag(df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    sources   = ["gdelt", "guardian", "nyt"]
    titles    = ["GDELT (2,436,994 records)", "The Guardian (~18,900 articles)",
                 "New York Times (~7,800 articles)"]
    colors    = ["#2196F3", "#4CAF50", "#FF5722"]
    sig_alpha = 0.90    # shade significant points

    for ax, src, title, color in zip(axes, sources, titles, colors):
        sub = df[df["source"] == src].sort_values("lag_days")
        lags = sub["lag_days"].values
        r    = sub["pearson_r_vol"].values
        pval = sub["p_value"].values
        n    = sub["n"].values

        # Plot line
        ax.plot(lags, r, color=color, linewidth=2, zorder=3)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axvline(0, color="gray",  linewidth=0.8, linestyle=":",  alpha=0.7,
                   label="Meeting day")

        # Significant points (p < 0.05) filled, non-significant open
        for lag_i, r_i, p_i, n_i in zip(lags, r, pval, n):
            if p_i < 0.05:
                ax.scatter(lag_i, r_i, color=color, s=80, zorder=5,
                           edgecolors="black", linewidths=0.8)
                ax.annotate(f"r={r_i:.2f}\np={p_i:.3f}",
                            xy=(lag_i, r_i),
                            xytext=(5, 6), textcoords="offset points",
                            fontsize=7, color="black")
            else:
                ax.scatter(lag_i, r_i, color="white", s=60, zorder=5,
                           edgecolors=color, linewidths=1.2)

        ax.set_xlabel("Lag (days relative to FOMC meeting)", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_xticks(sorted(sub["lag_days"].unique()))
        ax.set_xlim(lags.min() - 0.5, lags.max() + 0.5)
        ax.set_ylim(-0.8, 0.85)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Pearson r (cosine-sim vs realised volatility)", fontsize=10)

    # Shared legend
    sig_patch   = mpatches.Patch(color="gray", label="● p < 0.05 (significant)")
    insig_patch = mpatches.Patch(color="white", ec="gray", label="○ p ≥ 0.05")
    vline_patch = mpatches.Patch(color="gray",  linestyle=":", label="FOMC meeting day")
    fig.legend(handles=[sig_patch, insig_patch], loc="lower center",
               ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Lead–Lag Correlation: US Media Cosine Similarity vs S&P 500 Realised Volatility\n"
        "Around 88 FOMC Meetings, 2015–2025  |  Filled = p < 0.05",
        fontsize=11, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved {out_path}")

# Figure 2: Benchmark bar chart
def plot_benchmark(df, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))

    impls  = df["impl"].tolist()
    times  = df["seconds"].tolist()
    colors = ["#EF5350", "#42A5F5", "#66BB6A"]   # red, blue, green
    bars   = ax.bar(impls, times, color=colors, width=0.5,
                    edgecolor="black", linewidth=0.7)

    # Speedup labels
    baseline = times[0]
    for bar, t, impl in zip(bars, times, impls):
        speedup = baseline / t
        label = f"{t:.3f}s\n({speedup:.1f}×)" if impl != "python" else f"{t:.3f}s\n(baseline)"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Time (seconds) — lower is better", fontsize=11)
    ax.set_title("Sentiment Scorer Performance: Python vs NumPy vs Numba JIT\n"
                 "(5 million token workload)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(times) * 1.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticklabels(["Pure Python", "NumPy\nvectorised", "Numba JIT\n@jit(parallel=True)"],
                       fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved {out_path}")

# Main
def main():
    if not BUCKET:
        raise EnvironmentError("BUCKET not set in .env")

    print("[viz] loading results from S3 ...")
    summary_df   = load_csv_from_s3(BUCKET, "processed/lead_lag/summary.csv")
    benchmark_df = load_csv_from_s3(BUCKET, "processed/benchmarks/benchmarks.csv")

    # Figure 1
    fig1_path = FIGURES_DIR / "lead_lag_correlation.png"
    plot_lead_lag(summary_df, fig1_path)
    upload_figure(BUCKET, fig1_path, "processed/figures/lead_lag_correlation.png")

    # Figure 2
    fig2_path = FIGURES_DIR / "benchmark.png"
    plot_benchmark(benchmark_df, fig2_path)
    upload_figure(BUCKET, fig2_path, "processed/figures/benchmark.png")

    print("[done] figures saved to analysis/figures/ and uploaded to S3")

if __name__ == "__main__":
    main()