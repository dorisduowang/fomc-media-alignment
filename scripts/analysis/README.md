# Temporal validation workflow

The versioned entrypoints below read the Spark meeting-window output, rebuild the market outcomes from frozen local prices, and estimate the holdout before secondary samples. The analysis protocol is at `MakkaPakka update/analysis_protocol_extension.md`. Existing calls to the numbered scripts without arguments retain the earlier baseline workflow.

Use a new `ANALYSIS_OUT` for each frozen run. Supply the local Spark output directory as `PER_MEETING`, and the directory containing the frozen `SPX_daily_2014_2025.csv` and `VIX_2014_2025.csv` as `MARKET_DIR`.

```bash
python scripts/analysis/00_build_panel.py \
  --per-meeting "$PER_MEETING" --market-dir "$MARKET_DIR" --output-dir "$ANALYSIS_OUT"
python scripts/analysis/12_main_regression.py \
  --input-dir "$ANALYSIS_OUT" --output-dir "$ANALYSIS_OUT"
python scripts/analysis/13_identification.py \
  --input-dir "$ANALYSIS_OUT" --output-dir "$ANALYSIS_OUT"
```

The equivalent scripts under `MakkaPakka update/scripts/analysis/` route these arguments to the same implementation. Each stage is written to a temporary directory and published after completion. A completed stage contains a manifest with artifact, code, and protocol hashes. Reusing a completed stage requires a new output directory.

The panel retains `source`, `corpus`, and `weighting` in its keys. It rebuilds all 3-, 5-, 7-, and 10-return strict and event-inclusive windows and saves their exact trading dates. Realized variance, mean squared return, and log-volatility companions are included. Zero volatility is missing only for log outcomes. VIX uses the last available observation strictly before each event. Coverage reports the actual observations, including gaps, by sample, corpus, weighting, and lag.

The default analysis selects GDELT, direct-Fed, domain-equal weighting and `vol_post_strict_5`. Discovery is March 18, 2015 through March 20, 2019. Holdout is May 1, 2019 through December 10, 2025. The two known archive gaps are checked explicitly. Archive availability bounds the possible sample, and the effective sample comes from the validated observations.

The regression stage reports Pearson correlation and three OLS specifications. The latter progressively add matching pre-meeting volatility and pre-meeting VIX. Each model selects its own complete cases and saves its meeting list. The outputs include HC3 confidence intervals, one-event-lag HAC sensitivity, standardized coefficients, partial R-squared, and the outcome change associated with one predictor standard deviation. Use `--outcome vol_post_event_10`, for example, to pair that outcome with `vol_pre_10`.

The inference stage uses a chronological moving-block bootstrap with seed 30113 and 2,000 draws. The default block length is the ceiling of the cube root of the model sample size. `--block-length` and `--bootstrap-draws` make these choices explicit. Every draw, including singular draws, is recorded, and at least 90% must be estimable.

The 15-lag family uses common complete-case support. HC3 t-statistics are compared with the maximum absolute statistic from every nonidentity circular shift of the reduced-model residuals, controlling for pre-volatility and VIX. The full shift distribution and the common-support meeting list are saved. The corrected p-value has resolution `1 / n`. This residual-shift procedure is a time-dependence sensitivity analysis, and its assumptions differ from an exchangeable independent-event permutation.

After primary holdout estimates have been saved, run the regression and inference commands with `--sample discovery` or `--sample pooled` for secondary estimates. The gate verifies the holdout's exact panel, protocol, implementation, selection, and result hashes. Companion outcomes also follow the primary holdout.

MPS remains pending selection of a USMPD definition with Jerry. To enable the secondary surprise specifications later, supply a normalized CSV with unique `meeting_date` and `mps` columns together with `--mps-definition usmpd_statement` or `--mps-definition usmpd_monetary_event` when building the panel. Keep one series and definition for the whole period. The MPS models and their no-surprise bridge use identical meetings. Missing surprises leave the primary models' samples intact.

The panel also writes `pseudo_event_requests.csv`. It lists same-year/month, same-weekday non-FOMC dates with sufficient frozen market coverage, the corresponding real meeting whose minutes are the comparison target, and the required lag -2 media date. Computing those predictors needs the matching media inputs and the same frozen TF-IDF model. A supplied `--pseudo-similarity` CSV must contain `anchor_meeting_date`, `pseudo_date`, `predictor_date`, `source`, `corpus`, `weighting`, `cosine_sim`, `model_fit_cutoff`, `filter_version`, and `representation_type`. It must use the real meeting's minutes as the comparison target. The code verifies matching dates and provenance, rebuilds pseudo-event outcomes, and selects one candidate per event reproducibly. The real and pseudo models use common events. Their estimates are reported as a matched-date sensitivity check.

The stage manifests record whether matched-date similarity was supplied. Corpus relevance coding and cloud deployment validation remain separate from the statistical implementation. The versioned inference entrypoint runs the block bootstrap and lag-family procedure. The earlier VAR and Granger routines remain available through the baseline entrypoint.
