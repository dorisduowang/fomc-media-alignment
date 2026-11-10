# Reproducible pipeline for "Anticipation, Not Reaction".
#
# The CPU stages below reproduce every number and figure in the manuscript from
# the committed meeting aggregates and the local inputs in data_reference/. The distributed
# ingestion, Spark sentiment/similarity, and transformer-embedding stages are
# declared for completeness but are gated on cluster or GPU resources and are not
# part of the default target; run them explicitly with the paths for your
# environment. See README for the split between reproducible and resource-gated
# stages.

TABLES = "results/tables"
FIGDIR = "results/figures"

rule all:
    input:
        f"{TABLES}/panel_meeting.csv",
        f"{TABLES}/main_regression.tex",
        f"{TABLES}/placebo_permutation.csv",
        f"{TABLES}/sentiment_agreement.csv",
        f"{TABLES}/benchmarks.csv",
        f"{TABLES}/tab_identification.tex",
        f"{FIGDIR}/fig1_leadlag_gdelt.pdf",

rule panel:
    input:
        "data_reference/per_meeting.parquet",
        "data_reference/sentiment.parquet",
        "data_reference/market/VIX_2014_2025.csv",
    output:
        f"{TABLES}/panel_meeting.csv",
        f"{TABLES}/coverage_diagnostic.csv",
    shell:
        "python scripts/analysis/00_build_panel.py"

rule main_regression:
    input:
        f"{TABLES}/panel_meeting.csv",
    output:
        f"{TABLES}/main_regression.tex",
        f"{TABLES}/leadlag_profile.csv",
    shell:
        "python scripts/analysis/12_main_regression.py"

rule identification:
    input:
        f"{TABLES}/panel_meeting.csv",
    output:
        f"{TABLES}/placebo_permutation.csv",
        f"{TABLES}/granger.csv",
        f"{TABLES}/bootstrap.csv",
        f"{TABLES}/subsamples.csv",
        f"{TABLES}/var_irf.csv",
    shell:
        "python scripts/analysis/13_identification.py"

rule sentiment_lm:
    output:
        f"{TABLES}/sentiment_agreement.csv",
        f"{TABLES}/sentiment_comparison.csv",
    shell:
        "python scripts/sentiment/07b_lm_dictionary.py"

rule benchmark:
    output:
        f"{TABLES}/benchmarks.csv",
    shell:
        "python scripts/benchmark/13_benchmark.py"

rule latex_tables:
    input:
        f"{TABLES}/leadlag_profile.csv",
        f"{TABLES}/placebo_permutation.csv",
        f"{TABLES}/subsamples.csv",
        f"{TABLES}/benchmarks.csv",
        f"{TABLES}/sentiment_agreement.csv",
    output:
        f"{TABLES}/tab_identification.tex",
        f"{TABLES}/tab_leadlag.tex",
        f"{TABLES}/tab_subsamples.tex",
        f"{TABLES}/tab_benchmark.tex",
        f"{TABLES}/tab_coverage.tex",
    shell:
        "python scripts/analysis/16_make_tables.py"

rule figures:
    input:
        f"{TABLES}/leadlag_profile.csv",
        f"{TABLES}/main_regression.csv",
        f"{TABLES}/sentiment_comparison.csv",
    output:
        f"{FIGDIR}/fig1_leadlag_gdelt.pdf",
    shell:
        "python scripts/figures/make_figures.py"
