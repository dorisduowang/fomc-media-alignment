"""
Descriptive statistics for the broad-corpus estimation sample.

The table summarizes the variables that enter the main regression on the 33 GDELT
meetings with complete pre-meeting coverage: the pre-meeting similarity regressor,
the realized-volatility outcome and its pre-meeting counterpart, the pre-meeting
VIX, the pre-meeting media tone, and the orthogonalized high-frequency policy
surprise. It is the Table 1 a reader expects before the estimates, and every later
sample size can be read against the count reported here.

Outputs:
    results/tables/summary_stats.csv
    results/tables/tab_summary.tex
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"

VARS = [
    ("sim_m2", r"Pre-meeting similarity $\mathrm{sim}_{-2}$"),
    ("vol_post", "Post-meeting realized volatility"),
    ("vol_pre", "Pre-meeting realized volatility"),
    ("vix_pre", "Pre-meeting VIX"),
    ("sent_m2", "Pre-meeting media tone"),
    ("mps_orth", "Policy surprise (orthogonalized)"),
]


def main():
    panel = pd.read_csv(TAB / "panel_meeting.csv")
    g = panel[panel["source"] == "gdelt"].dropna(
        subset=["sim_m2", "vol_post", "vol_pre", "vix_pre"]
    )

    rows = []
    for col, _ in VARS:
        s = g[col].dropna()
        rows.append(
            {
                "variable": col,
                "n": int(s.size),
                "mean": s.mean(),
                "sd": s.std(ddof=1),
                "min": s.min(),
                "median": s.median(),
                "max": s.max(),
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(TAB / "summary_stats.csv", index=False)

    label = dict(VARS)
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & $N$ & Mean & SD & Min & Median & Max \\",
        r"\midrule",
    ]
    for _, r in stats.iterrows():
        lines.append(
            f"{label[r['variable']]} & {int(r['n'])} & {r['mean']:.3f} & "
            f"{r['sd']:.3f} & {r['min']:.3f} & {r['median']:.3f} & {r['max']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_summary.tex").write_text("\n".join(lines) + "\n")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
