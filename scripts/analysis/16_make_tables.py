"""
Render the result CSVs into LaTeX tables for the manuscript and appendix.

Reads the tables produced by the analysis scripts and writes \\input-ready .tex
fragments into results/tables/. The main regression table is written by
12_main_regression.py; this script produces the remaining tables.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def coverage_table():
    s = pd.read_csv(TAB / "coverage_summary.csv")
    s = s.rename(columns={s.columns[0]: "source"})
    order = {"gdelt": "GDELT", "guardian": "The Guardian", "nyt": "New York Times"}
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Corpus & Meetings & First & Last & Obs. \\", r"\midrule"]
    for key, lab in order.items():
        r = s[s.source == key].iloc[0]
        lines.append(
            f"{lab} & {int(r.n_meetings)} & {r.first_meeting} & {r.last_meeting} & {int(r.n_obs)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_coverage.tex").write_text("\n".join(lines))


def leadlag_table():
    prof = pd.read_csv(TAB / "leadlag_profile.csv")
    g = prof[prof.source == "gdelt"].sort_values("lag")
    lines = [r"\begin{tabular}{rrrrr}", r"\toprule",
             r"Lag $k$ & $N$ & Mean cosine & Pearson $r$ & $p$-value \\", r"\midrule"]
    for _, r in g.iterrows():
        lines.append(
            f"{int(r.lag):+d} & {int(r.n)} & {r.mean_cosine:.3f} & "
            f"{r.pearson_r:.3f}{stars(r.p_value)} & {r.p_value:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_leadlag.tex").write_text("\n".join(lines))


def identification_table():
    plc = pd.read_csv(TAB / "placebo_permutation.csv")
    gr = pd.read_csv(TAB / "granger.csv")
    bs = pd.read_csv(TAB / "bootstrap.csv")
    lines = [r"\begin{tabular}{llr}", r"\toprule",
             r"Check & Statistic & Value \\", r"\midrule"]
    raw = plc[plc.statistic == "raw_corr"].iloc[0]
    par = plc[plc.statistic == "partial_corr_given_vol_vix"].iloc[0]
    lines.append(f"Permutation placebo & raw corr., perm. $p$ & {raw.observed:.3f} ($p={raw.perm_p_two_sided:.4f}$) \\\\")
    lines.append(f" & partial corr. $\\mid$ VIX, vol., perm. $p$ & {par.observed:.3f} ($p={par.perm_p_two_sided:.3f}$) \\\\")
    for _, r in gr.iterrows():
        lines.append(f"Granger $\\text{{sim}}\\to\\text{{vol}}$ & lag {int(r.lag)}, $F$-test $p$ & {r.F:.2f} ($p={r.p_value:.3f}$) \\\\")
    biv = bs[bs.spec == "bivariate"].iloc[0]
    ctr = bs[bs.spec == "controlled"].iloc[0]
    lines.append(f"Bootstrap (bivariate) & 95\\% CI for $\\beta$ & [{biv.ci_lo:.2f}, {biv.ci_hi:.2f}] \\\\")
    lines.append(f"Bootstrap (controlled) & 95\\% CI for $\\beta$ & [{ctr.ci_lo:.2f}, {ctr.ci_hi:.2f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_identification.tex").write_text("\n".join(lines))


def subsample_table():
    ss = pd.read_csv(TAB / "subsamples.csv")
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Subsample & $N$ & $\hat\beta_{\mathrm{sim}_{-2}}$ & $p$ \\", r"\midrule"]
    for _, r in ss.iterrows():
        if "leave-one-out" in r.subsample:
            lines.append(f"{r.subsample} & {int(r.n)} & {r.controls} & \\\\")
        else:
            p = "" if pd.isna(r.p) else f"{r.p:.3f}"
            b = f"{r.beta_sim_m2:.3f}"
            b += stars(r.p) if not pd.isna(r.p) else ""
            lines.append(f"{r.subsample} & {int(r.n)} & {b} & {p} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_subsamples.tex").write_text("\n".join(lines))


def benchmark_table():
    b = pd.read_csv(TAB / "benchmarks.csv")
    lines = [r"\begin{tabular}{lrr}", r"\toprule",
             r"Implementation & Time (s) & Speedup \\", r"\midrule"]
    names = {"python": "Pure Python loop", "numpy": "NumPy searchsorted",
             "numba": "Numba parallel JIT"}
    for _, r in b.iterrows():
        lines.append(f"{names.get(r.impl, r.impl)} & {r.seconds:.3f} & {r.speedup:.1f}$\\times$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_benchmark.tex").write_text("\n".join(lines))


def coverage_full_table():
    cov = pd.read_csv(TAB / "coverage_diagnostic.csv", index_col=0)[
        ["gdelt", "guardian", "nyt"]
    ]
    lines = [r"\begin{tabular}{rrrr}", r"\toprule",
             r"Lag $k$ & GDELT & Guardian & NYT \\", r"\midrule"]
    for k, row in cov.iterrows():
        f = lambda v: "" if pd.isna(v) else str(int(v))
        lines.append(f"{int(k):+d} & {f(row.gdelt)} & {f(row.guardian)} & {f(row.nyt)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_coverage_full.tex").write_text("\n".join(lines))


def var_irf_table():
    irf = pd.read_csv(TAB / "var_irf.csv")
    lines = [r"\begin{tabular}{rr}", r"\toprule",
             r"Horizon (meetings) & Vol. response to sim. shock \\", r"\midrule"]
    for _, r in irf.iterrows():
        lines.append(f"{int(r.horizon)} & {r.vol_resp_to_sim_shock:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_var_irf.tex").write_text("\n".join(lines))


def granger_table():
    gr = pd.read_csv(TAB / "granger.csv")
    lines = [r"\begin{tabular}{rrrr}", r"\toprule",
             r"Lag & $F$ & df & $p$-value \\", r"\midrule"]
    for _, r in gr.iterrows():
        lines.append(f"{int(r.lag)} & {r.F:.3f} & ({int(r.df_num)},{int(r.df_den)}) & {r.p_value:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_granger.tex").write_text("\n".join(lines))


def altwindow_table():
    a = pd.read_csv(TAB / "altwindow.csv")
    lines = [r"\begin{tabular}{rrcccc}", r"\toprule",
             r" & & \multicolumn{2}{c}{Bivariate} & \multicolumn{2}{c}{Controlled} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Window (days) & $N$ & $\hat\beta$ & $p$ & $\hat\beta$ & $p$ \\", r"\midrule"]
    for _, r in a.iterrows():
        bb = f"{r.beta_bivariate:.3f}" + stars(r.p_bivariate)
        bc = f"{r.beta_controlled:.3f}" + stars(r.p_controlled)
        lines.append(f"{int(r.window)} & {int(r.n)} & {bb} & {r.p_bivariate:.3f} & {bc} & {r.p_controlled:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_altwindow.tex").write_text("\n".join(lines))


def main():
    coverage_table()
    coverage_full_table()
    altwindow_table()
    leadlag_table()
    identification_table()
    granger_table()
    var_irf_table()
    subsample_table()
    benchmark_table()
    print("[tables] wrote coverage, coverage_full, leadlag, identification, "
          "granger, var_irf, subsamples, benchmark (.tex)")


if __name__ == "__main__":
    main()
