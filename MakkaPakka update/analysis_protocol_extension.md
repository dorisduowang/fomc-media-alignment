STATUS: FROZEN

# Temporal holdout protocol for the 2015-2025 extension

This document must be reviewed by Doris and Jerry before any 2019-2025 similarity or market-regression result is inspected. It is a prospective temporal-holdout protocol, not a retrospective preregistration. Change the first line to `STATUS: FROZEN` only after every field below is accepted. Record the freeze time, names, and SHA-256 immediately afterward.

Freeze time UTC: 2026-11-01

Approved by Doris: Doris Duo Wang

Approved by Jerry: Jerry Yuzhao Jin

## Research question

Test whether pre-meeting alignment between GDELT GKG metadata and the eventual FOMC minutes is associated with realized equity-market volatility after FOMC meetings. The design does not identify a causal media effect and does not treat the later-released minutes as information available in real time.

## Samples

- Discovery sample: meetings from 2015-03-18 through 2019-03-20 with an available lag -2 GDELT archive day. Expected n is 33.
- Temporal holdout: meetings from 2019-05-01 through 2025-12-10 with an available lag -2 GDELT archive day. The 2025-06-18 meeting is unavailable because the official GDELT archive has a gap. Expected n is 53.
- Pooled sample: discovery plus holdout. Expected n is 86.
- The 2015-01-28 meeting is unavailable because GDELT 2.0 begins after its lag -2 day.
- The holdout is evaluated and saved before any pooled-sample specification is inspected.

## Primary corpus

The primary GDELT corpus is the frozen direct-Fed filter. A record must contain an explicit Federal Reserve, FOMC, Federal Reserve Bank, Board of Governors, or named Federal Reserve official signal together with monetary-policy context. Exact or suffix domain matching is required. Substring matching is prohibited.

The original broad macro-finance filter is a robustness corpus. It cannot be labeled the population of Federal Reserve news. Its inclusion rules and outlet allowlist must be stored in versioned machine-readable files with SHA-256 hashes.

## Text representation

The primary representation is the concatenation of GKG themes, organizations, and persons fields. It is labeled `gkg_themes_orgs_persons`. It is metadata, not article full text. The manuscript will use the phrase `GKG metadata-to-minutes lexical alignment` unless a separately licensed full-text validation sample is added.

## TF-IDF model

- Fit tokenizer, stop-word remover, hashing TF, IDF, and L2 normalization on discovery-period documents only.
- The fit cutoff is 2019-03-23, which includes the complete plus-or-minus-three-day window around the final discovery meeting.
- Transform both discovery and holdout with the frozen discovery-fitted model.
- Save the fitted model, cutoff, parameters, stop-word list, feature dimension, code hash, and input hashes.
- Domain-equal weighting is primary. Article weighting is a robustness specification.
- The legacy unnormalized TF-IDF measure is robustness only.

## Primary predictor

The primary predictor is cosine similarity at calendar lag -2 between the daily GDELT direct-Fed domain-equal metadata vector and the FOMC minutes vector. Lag -2 was selected in the discovery sample and is tested once in the untouched holdout.

The complete lag family from -7 through +7 is descriptive and must receive a max-absolute-statistic permutation correction. No alternative lag becomes primary after the holdout is inspected.

## Market outcomes

The primary outcome is annualized standard deviation of the first five S&P 500 close-to-close log returns whose trading dates are strictly greater than the meeting date.

Required companion outcomes are:

- Event-inclusive five-return volatility using the meeting date when it is a trading day and the following trading dates.
- Strictly post-meeting volatility using 3, 7, and 10 returns.
- Event-inclusive volatility using 3, 7, and 10 returns.
- Mean squared return or realized variance over the same windows.
- Log volatility with a documented zero-handling rule.

For every meeting and window, save the exact pre and post trading dates. Pre and post return-date sets must be disjoint. Pre-volatility controls use the matching window length.

The frozen local S&P 500 file is used. Runtime market-data downloads are prohibited.

## Primary models

Estimate and report the holdout first:

1. Pearson correlation between lag -2 similarity and strict five-return post-meeting volatility.
2. OLS of strict five-return post-meeting volatility on lag -2 similarity.
3. OLS adding matching five-return pre-meeting volatility.
4. OLS adding pre-meeting VIX.

Report raw and standardized coefficients, confidence intervals, partial R-squared, and the outcome change associated with a one-standard-deviation predictor change. Use two-sided tests.

The primary uncertainty estimate is HC3 because the holdout is small. Report one-event-lag HAC and a chronological moving-block bootstrap as sensitivity checks. Bootstrap observations in time order; do not use an unrestricted independent-pairs bootstrap as the only uncertainty estimate.

The pooled and discovery estimates are secondary and must be labeled accordingly.

## Monetary-policy surprises

The policy-surprise variable is not a mandatory primary control because it occurs after the lag -2 media measure and can lie on an event-day pathway to volatility. It is a secondary identification and mechanism specification.

Use one consistent SF Fed USMPD definition across the available 2015-2025 sample. The primary surprise choice must be frozen before results are inspected. Prefer either the statement window or the combined monetary-event window and label it exactly. Do not silently splice the legacy Bauer-Swanson series to USMPD.

Specifications without a surprise use their full available sample. Surprise specifications use their own available sample and must be paired with a no-surprise bridge model estimated on exactly that same restricted sample. Report signed, absolute, and squared surprise versions as separate predeclared robustness tests.

## Corpus validation and dependence

- Draw a stratified random audit sample of at least 200 direct-Fed and 200 broad-macro records across years and high-volume domains.
- Two coders independently classify relevance.
- Report precision, agreement, Cohen's kappa, disagreement categories, and adjudication.
- Report duplicate document IDs, duplicate normalized URLs, syndication clusters, top-domain shares, and effective domain counts.
- Report article-weighted and domain-equal-weighted results.
- Compare GDELT, NYT, and Guardian only on common-support meetings and with coefficient-difference tests. Different text fields and sample coverage prevent a causal breadth interpretation.

## Multiple testing and placebo design

- Correct the -7 through +7 lag family with a max-absolute-t permutation statistic.
- Preserve chronological dependence with circular shifts or block permutations.
- Add weekday-and-month-matched non-FOMC pseudo-event dates.
- Save the full permutation distribution and random seed.
- Do not call a single unadjusted lag permutation result confirmatory evidence.

## VAR and Granger analysis

VAR, Granger, and impulse-response results are excluded from the main paper unless the final sample supports predeclared stationarity, stability, residual, lag-order, and confidence-interval diagnostics. Otherwise they move to an explicitly exploratory appendix or are removed.

## Interpretation and decision rules

- The main claim is associational.
- Pre-meeting media alignment with later-released minutes is retrospective alignment, not a real-time forecast based on information available to market participants.
- A non-significant controlled coefficient is not proof of no effect.
- Any equivalence claim requires bounds fixed before testing.
- If the holdout fails to support the discovery association, report that result as the central temporal-validation finding and rewrite the title, abstract, and conclusion.
- No filters, lags, outcomes, transformations, controls, exclusions, or weighting rules are changed after holdout inspection without labeling the change exploratory and reserving a new validation sample.

## Reproducibility record

Archive the download plan, official source URLs, source hashes, code hashes, environment lock, EMR release, Spark version, cluster configuration, logs, shard manifests, fitted model, random seeds, quality-control tables, and final artifact hashes. Generate all final tables, figures, manuscript statistics, and the Word/PDF exports from one frozen run.
