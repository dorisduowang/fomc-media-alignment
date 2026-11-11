"""Versioned panels and inference for the frozen temporal validation protocol."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "MakkaPakka update" / "analysis_protocol_extension.md"
KEYS = ["meeting_date", "source", "corpus", "weighting"]
PRIMARY_OUTCOME = "vol_post_strict_5"
LAGS = tuple(range(-7, 8))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def lag_column(lag):
    return f"sim_lag_{'m' if lag < 0 else 'p'}{abs(lag)}"


def sample_labels(dates):
    return np.select(
        [dates.between("2015-03-18", "2019-03-20"), dates.between("2019-05-01", "2025-12-10")],
        ["discovery", "holdout"], default="excluded",
    )


@contextmanager
def output_stage(output, name):
    """Publish a complete stage by renaming its directory after success."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / name
    if destination.exists():
        raise FileExistsError(f"Stage already exists: {destination}. Choose a new output directory.")
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=output) as temporary:
        stage = Path(temporary) / name
        stage.mkdir()
        yield stage
        stage.rename(destination)


def write_manifest(stage, metadata):
    metadata = dict(metadata, created_utc=datetime.now(timezone.utc).isoformat(),
                    protocol_sha256=digest(PROTOCOL), code_sha256=digest(__file__),
                    packages={name: importlib.metadata.version(name)
                              for name in ("numpy", "pandas", "scipy", "statsmodels")})
    metadata["files"] = {p.name: digest(p) for p in stage.iterdir() if p.is_file()}
    (stage / "manifest.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def volatility_module():
    path = ROOT / "analysis" / "12_lead_lag_volatility.py"
    spec = importlib.util.spec_from_file_location("frozen_volatility", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def market_outcomes(dates, spx_file):
    frame = volatility_module().fetch_sp500_volatility(sorted(pd.unique(dates)), spx_file)
    frame["meeting_date"] = pd.to_datetime(frame["meeting_date"])
    for width in (3, 5, 7, 10):
        for timing in ("strict", "event"):
            column = f"vol_post_{timing}_{width}"
            frame[f"log_{column}"] = np.log(frame[column].where(frame[column] > 0))
            frame[f"msr_post_{timing}_{width}"] = frame[f"rv_post_{timing}_{width}"] / width
    return frame


def pre_event_vix(dates, path):
    frame = pd.read_csv(path)
    if len(frame.columns) != 2:
        raise ValueError("VIX input must contain a date and a value column")
    frame.columns = ["date", "vix_pre"]
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["vix_pre"] = pd.to_numeric(frame["vix_pre"], errors="coerce")
    if frame["date"].duplicated().any():
        raise ValueError("VIX dates must be unique")
    left = pd.DataFrame({"meeting_date": sorted(pd.unique(dates))})
    return pd.merge_asof(left, frame.dropna().sort_values("date"), left_on="meeting_date",
                         right_on="date", direction="backward", allow_exact_matches=False).drop(columns="date")


def pseudo_requests(calendar):
    """List same-month, same-weekday dates and their required lag -2 inputs."""
    events = set(calendar)
    requests = []
    for anchor in sorted(events):
        if sample_labels(pd.Series([anchor]))[0] == "excluded":
            continue
        month = pd.date_range(anchor.replace(day=1), anchor + pd.offsets.MonthEnd(0))
        for day in month:
            if day.weekday() == anchor.weekday() and day not in events:
                requests.append(dict(anchor_meeting_date=anchor, pseudo_date=day,
                                     predictor_date=day - pd.Timedelta(days=2)))
    return pd.DataFrame(requests)


def build_pseudo_panel(path, calendar, spx_file, vix_file, filter_version):
    pseudo = pd.read_csv(path, parse_dates=["anchor_meeting_date", "pseudo_date", "predictor_date"])
    required = {"anchor_meeting_date", "pseudo_date", "predictor_date", "source", "corpus", "weighting",
                "cosine_sim", "model_fit_cutoff", "filter_version", "representation_type"}
    if required - set(pseudo):
        raise ValueError(f"Pseudo-event input is missing {sorted(required - set(pseudo))}")
    anchor, day = pseudo["anchor_meeting_date"], pseudo["pseudo_date"]
    valid = (anchor.isin(calendar) & ~day.isin(calendar) & (anchor.dt.weekday == day.dt.weekday)
             & (anchor.dt.to_period("M") == day.dt.to_period("M"))
             & (pseudo["predictor_date"] == day - pd.Timedelta(days=2)))
    if not valid.all() or pseudo.duplicated(["anchor_meeting_date", "pseudo_date", "source", "corpus", "weighting"]).any():
        raise ValueError("Pseudo-events require unique matched non-FOMC dates and lag -2 predictors")
    if (not pd.to_datetime(pseudo["model_fit_cutoff"]).eq(pd.Timestamp("2019-03-23")).all()
            or not pseudo["filter_version"].eq(filter_version).all()
            or not pseudo["representation_type"].eq("gkg_themes_orgs_persons").all()
            or not pd.to_numeric(pseudo["cosine_sim"], errors="raise").between(0, 1).all()):
        raise ValueError("Pseudo-event predictors must use the same frozen GKG model and filter")
    market = market_outcomes(day, spx_file).merge(pre_event_vix(day, vix_file), on="meeting_date", validate="one_to_one")
    market = market.rename(columns={"meeting_date": "pseudo_date"})
    pseudo = pseudo.merge(market, on="pseudo_date", validate="many_to_one")
    pseudo["sim_m2"] = pseudo["cosine_sim"]
    pseudo["sample"] = sample_labels(pseudo["anchor_meeting_date"])
    return pseudo


def validate_long(frame):
    required = {*KEYS, "lag_days", "cosine_sim", "model_fit_cutoff", "representation_type", "filter_version"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Versioned similarity input is missing {sorted(missing)}")
    frame = frame.copy()
    frame["meeting_date"] = pd.to_datetime(frame["meeting_date"], errors="raise").dt.normalize()
    if frame[KEYS + ["lag_days"]].isna().any().any() or frame.duplicated(KEYS + ["lag_days"]).any():
        raise ValueError("Similarity keys must be complete and unique")
    if not frame["lag_days"].isin(LAGS).all():
        raise ValueError("Expected integer lags from -7 through +7")
    scores = pd.to_numeric(frame["cosine_sim"], errors="raise")
    if not scores.dropna().between(0, 1).all():
        raise ValueError("Cosine scores must be finite values between zero and one")
    frame["cosine_sim"] = scores
    if not pd.to_datetime(frame["model_fit_cutoff"], errors="raise").eq(pd.Timestamp("2019-03-23")).all():
        raise ValueError("Model fit cutoff must be 2019-03-23")
    gdelt = frame[frame["source"] == "gdelt"]
    if (not gdelt["representation_type"].eq("gkg_themes_orgs_persons").all()
            or gdelt["filter_version"].isna().any() or gdelt["filter_version"].nunique() != 1):
        raise ValueError("GDELT input must identify one metadata representation and filter version")
    if not gdelt["corpus"].isin(["direct_fed", "broad_macro"]).all() or not gdelt["weighting"].isin(["article", "domain_equal"]).all():
        raise ValueError("Unknown GDELT corpus or weighting")
    gap = gdelt["meeting_date"].isin(pd.to_datetime(["2015-01-28", "2025-06-18"]))
    if (gap & gdelt["lag_days"].eq(-2) & gdelt["cosine_sim"].notna()).any():
        raise ValueError("Unexpected lag -2 observation inside an official archive gap")
    return frame


def build_wide(frame):
    panel = frame.pivot(index=KEYS, columns="lag_days", values="cosine_sim").reindex(columns=LAGS)
    panel.columns = [lag_column(lag) for lag in panel.columns]
    panel = panel.reset_index()
    panel["sim_m2"] = panel[lag_column(-2)]
    panel["sim_0"] = panel[lag_column(0)]
    panel["sim_pre_mean"] = panel[[lag_column(k) for k in (-3, -2, -1)]].mean(axis=1)
    metadata = [c for c in ("n_docs", "n_urls", "n_domains", "mean_native_tone") if c in frame]
    auxiliary = frame.loc[frame["lag_days"] == -2, KEYS + metadata]
    auxiliary = auxiliary.rename(columns={c: f"{c}_m2" for c in metadata})
    return panel.merge(auxiliary, on=KEYS, how="left", validate="one_to_one")


def panel_cli():
    parser = argparse.ArgumentParser(description="Build versioned meeting panels from frozen local inputs")
    parser.add_argument("--per-meeting", type=Path, required=True)
    parser.add_argument("--market-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mps-file", type=Path, help="Normalized CSV with meeting_date and mps")
    parser.add_argument("--mps-definition", choices=["usmpd_statement", "usmpd_monetary_event"])
    parser.add_argument("--pseudo-similarity", type=Path, help="Matched-date lag -2 similarity CSV")
    args = parser.parse_args()
    if not PROTOCOL.read_text().startswith("STATUS: FROZEN"):
        parser.error("The temporal validation protocol must be frozen")
    if bool(args.mps_file) != bool(args.mps_definition):
        parser.error("Supply --mps-file and --mps-definition together")
    long = validate_long(pd.read_parquet(args.per_meeting))
    inputs = [p for p in sorted(args.per_meeting.rglob("*.parquet"))] if args.per_meeting.is_dir() else [args.per_meeting]
    spx = args.market_dir / "SPX_daily_2014_2025.csv"
    vix = args.market_dir / "VIX_2014_2025.csv"
    inputs.extend([spx, vix])
    market = market_outcomes(long["meeting_date"], spx).merge(
        pre_event_vix(long["meeting_date"], vix), on="meeting_date", validate="one_to_one")
    # Recompute market outcomes even when an older input carries columns with the same names.
    long = long.drop(columns=[c for c in market if c != "meeting_date" and c in long])
    long = long.merge(market, on="meeting_date", validate="many_to_one")
    long["sample"] = sample_labels(long["meeting_date"])
    panel = build_wide(long).merge(market, on="meeting_date", validate="many_to_one")
    panel["sample"] = sample_labels(panel["meeting_date"])
    panel["mps"] = np.nan
    if args.mps_file:
        surprises = pd.read_csv(args.mps_file, parse_dates=["meeting_date"])[["meeting_date", "mps"]]
        surprises["mps"] = pd.to_numeric(surprises["mps"], errors="raise")
        if surprises["meeting_date"].isna().any() or surprises["meeting_date"].duplicated().any():
            raise ValueError("Surprise CSV requires unique meeting dates")
        if not np.isfinite(surprises["mps"].dropna()).all():
            raise ValueError("Surprise values must be finite")
        panel = panel.drop(columns="mps").merge(surprises, on="meeting_date", how="left", validate="many_to_one")
        inputs.append(args.mps_file)
    panel["abs_mps"] = panel["mps"].abs()
    panel["squared_mps"] = panel["mps"] ** 2
    if not np.isfinite(panel["squared_mps"].dropna()).all():
        raise ValueError("Squared surprises overflowed; check the surprise series and units")
    calendar_file = ROOT / "MakkaPakka update" / "data_reference" / "meetings_all88.csv"
    calendar = pd.to_datetime(pd.read_csv(calendar_file)["meeting_date"])
    inputs.append(calendar_file)
    pseudo = None
    if args.pseudo_similarity:
        pseudo = build_pseudo_panel(args.pseudo_similarity, calendar, spx, vix,
                                    long.loc[long["source"] == "gdelt", "filter_version"].iloc[0])
        inputs.append(args.pseudo_similarity)
    coverage = (long.groupby(["source", "corpus", "weighting", "sample", "lag_days"], dropna=False)
                .agg(n_rows=("cosine_sim", "size"), n_observed=("cosine_sim", "count")).reset_index())
    with output_stage(args.output_dir, "panel") as stage:
        panel.sort_values(KEYS).to_csv(stage / "panel_meeting.csv", index=False)
        long.sort_values(KEYS + ["lag_days"]).to_parquet(stage / "panel_long.parquet", index=False)
        coverage.to_csv(stage / "coverage.csv", index=False)
        market.to_csv(stage / "market_window_audit.csv", index=False)
        requests = pseudo_requests(calendar)
        trades = volatility_module().load_frozen_sp500(spx)["date"].to_numpy()
        days = requests["pseudo_date"].to_numpy()
        pre_count = np.searchsorted(trades, days, side="left")
        post_count = len(trades) - np.searchsorted(trades, days, side="right")
        requests[(pre_count >= 10) & (post_count >= 10)].to_csv(stage / "pseudo_event_requests.csv", index=False)
        if pseudo is not None:
            pseudo.to_csv(stage / "pseudo_panel.csv", index=False)
        write_manifest(stage, {"stage": "panel", "mps_definition": args.mps_definition,
                               "volatility_code_sha256": digest(ROOT / "analysis/12_lead_lag_volatility.py"),
                               "zero_volatility_rule": "exclude from log outcomes only",
                               "has_pseudo_similarity": pseudo is not None,
                               "inputs": {p.name + f"#{i}": digest(p) for i, p in enumerate(inputs)}})
    print(f"Wrote {len(panel)} meeting/corpus/weighting rows to {args.output_dir / 'panel'}")


def load_panel(input_dir):
    stage = Path(input_dir) / "panel"
    metadata = json.loads((stage / "manifest.json").read_text())
    if metadata["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Panel protocol hash differs from the current frozen protocol")
    for name in metadata["files"]:
        if digest(stage / name) != metadata["files"][name]:
            raise ValueError(f"Panel artifact changed after completion: {name}")
    return pd.read_csv(stage / "panel_meeting.csv", parse_dates=["meeting_date"]), metadata


def analysis_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample", choices=["holdout", "discovery", "pooled"], default="holdout")
    parser.add_argument("--source", default="gdelt")
    parser.add_argument("--corpus", default="direct_fed")
    parser.add_argument("--weighting", default="domain_equal")
    parser.add_argument("--outcome", default=PRIMARY_OUTCOME)
    return parser


def matching_pre(outcome):
    if not re.fullmatch(r"(?:log_)?vol_post_(?:strict|event)_(?:3|5|7|10)|(?:rv|msr)_post_(?:strict|event)_(?:3|5|7|10)", outcome):
        raise ValueError("Unknown outcome definition")
    return f"vol_pre_{outcome.rsplit('_', 1)[1]}"


def selection(args, metadata):
    return dict(panel_sha256=metadata["files"]["panel_meeting.csv"], source=args.source,
                corpus=args.corpus, weighting=args.weighting, outcome=args.outcome)


def select_sample(panel, args):
    samples = ["discovery", "holdout"] if args.sample == "pooled" else [args.sample]
    selected = panel[(panel["source"] == args.source) & (panel["corpus"] == args.corpus)
                     & (panel["weighting"] == args.weighting) & panel["sample"].isin(samples)]
    return selected.sort_values("meeting_date").reset_index(drop=True)


def holdout_gate(args, metadata):
    if args.sample == "holdout" and args.outcome == PRIMARY_OUTCOME:
        return
    path = args.output_dir / f"regression_holdout_{PRIMARY_OUTCOME}" / "manifest.json"
    recorded = json.loads(path.read_text())
    expected = dict(selection(args, metadata), outcome=PRIMARY_OUTCOME)
    if recorded["selection"] != expected or recorded["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Save the primary holdout results for this exact panel and selection first")
    if recorded["code_sha256"] != digest(__file__):
        raise ValueError("Holdout results were produced by a different analysis implementation")
    if any(digest(path.parent / name) != checksum for name, checksum in recorded["files"].items()):
        raise ValueError("Holdout results changed after completion")


def complete_cases(frame, columns):
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=columns).sort_values("meeting_date")


def fit_model(frame, outcome, controls):
    columns = ["sim_m2", *controls]
    data = complete_cases(frame, [outcome, *columns])
    x = sm.add_constant(data[columns].astype(float), has_constant="add")
    if len(data) < len(x.columns) + 3 or np.linalg.matrix_rank(x) < len(x.columns):
        raise ValueError("Insufficient observations or a rank-deficient model")
    fit = sm.OLS(data[outcome].astype(float), x).fit()
    robust = fit.get_robustcov_results(cov_type="HC3", use_t=True)
    hac = fit.get_robustcov_results(cov_type="HAC", maxlags=1, use_correction=True, use_t=True)
    index = list(x.columns).index("sim_m2")
    sx, sy = data["sim_m2"].std(ddof=1), data[outcome].std(ddof=1)
    reduced = sm.OLS(data[outcome], x.drop(columns="sim_m2")).fit()
    lo, hi = robust.conf_int()[index]
    record = dict(n=len(data), beta=float(robust.params[index]), se_hc3=float(robust.bse[index]),
                  p_hc3=float(robust.pvalues[index]), ci_lo=float(lo), ci_hi=float(hi),
                  se_hac1=float(hac.bse[index]), p_hac1=float(hac.pvalues[index]),
                  beta_standardized=float(robust.params[index] * sx / sy),
                  ci_standardized_lo=float(lo * sx / sy), ci_standardized_hi=float(hi * sx / sy),
                  effect_per_sd=float(robust.params[index] * sx),
                  effect_per_sd_ci_lo=float(lo * sx), effect_per_sd_ci_hi=float(hi * sx),
                  partial_r2=float((reduced.ssr - fit.ssr) / reduced.ssr),
                  first_meeting=str(data["meeting_date"].min().date()),
                  last_meeting=str(data["meeting_date"].max().date()))
    if not np.isfinite([record[k] for k in ("beta", "se_hc3", "p_hc3", "partial_r2")]).all():
        raise ValueError("Model has undefined estimates or uncertainty")
    return record, data, fit


def regression_cli():
    args = analysis_parser("Estimate the holdout before discovery and pooled models").parse_args()
    panel, metadata = load_panel(args.input_dir)
    pre = matching_pre(args.outcome)
    holdout_gate(args, metadata)
    sample = select_sample(panel, args)
    specs = [("bivariate", []), ("pre_vol", [pre]), ("pre_vol_vix", [pre, "vix_pre"])]
    records, membership = [], []
    if metadata["mps_definition"]:
        mps_sample = complete_cases(sample, [args.outcome, "sim_m2", pre, "vix_pre", "mps"])
        specs.extend((name, [pre, "vix_pre", *extra]) for name, extra in (
            ("mps_bridge", []), ("mps_signed", ["mps"]), ("mps_absolute", ["abs_mps"]), ("mps_squared", ["squared_mps"])))
    for name, controls in specs:
        data = mps_sample if name.startswith("mps_") else sample
        try:
            record, used, _ = fit_model(data, args.outcome, controls)
            records.append(dict(spec=name, status="estimated", outcome=args.outcome,
                                controls="+".join(controls), **record))
            membership.extend(dict(spec=name, meeting_date=str(d.date())) for d in used["meeting_date"])
        except ValueError as exc:
            records.append(dict(spec=name, status=str(exc), outcome=args.outcome))
    if not all(r["status"] == "estimated" for r in records[:3]):
        raise ValueError("All three primary OLS models must be estimable before saving this sample")
    data = complete_cases(sample, [args.outcome, "sim_m2"])
    correlation = pearsonr(data["sim_m2"], data[args.outcome])
    ci = correlation.confidence_interval()
    with output_stage(args.output_dir, f"regression_{args.sample}_{args.outcome}") as stage:
        pd.DataFrame(records).to_csv(stage / "models.csv", index=False)
        pd.DataFrame(membership).to_csv(stage / "model_samples.csv", index=False)
        pd.DataFrame([dict(n=len(data), r=correlation.statistic, p=correlation.pvalue,
                           ci_lo=ci.low, ci_hi=ci.high)]).to_csv(stage / "correlation.csv", index=False)
        write_manifest(stage, {"stage": "regression", "sample": args.sample,
                               "selection": selection(args, metadata), "mps_definition": metadata["mps_definition"]})
    print(f"Saved {args.sample} regression estimates and model-specific membership")


def moving_block_indices(n, length, rng):
    if not 1 <= length < n:
        raise ValueError("Block length must be at least one and smaller than the sample")
    starts = rng.integers(0, n - length + 1, size=int(np.ceil(n / length)))
    return np.concatenate([np.arange(start, start + length) for start in starts])[:n]


def block_bootstrap(frame, outcome, controls, draws, length, rng):
    _, data, fit = fit_model(frame, outcome, controls)
    x = fit.model.exog
    y = fit.model.endog
    length = length or int(np.ceil(len(data) ** (1 / 3)))
    results = []
    for draw in range(draws):
        indices = moving_block_indices(len(data), length, rng)
        valid = np.linalg.matrix_rank(x[indices]) == x.shape[1]
        beta = float(np.linalg.lstsq(x[indices], y[indices], rcond=None)[0][1]) if valid else np.nan
        results.append(dict(draw=draw, beta=beta))
    valid = np.asarray([r["beta"] for r in results])
    valid = valid[np.isfinite(valid)]
    if len(valid) < max(2, int(np.ceil(0.9 * draws))):
        raise ValueError("Fewer than 90% of block bootstrap draws were estimable")
    lo, hi = np.quantile(valid, [0.025, 0.975])
    summary = dict(n=len(data), block_length=length, valid_draws=len(valid), requested_draws=draws,
                   boot_se=float(valid.std(ddof=1)), ci_lo=float(lo), ci_hi=float(hi))
    return summary, results


def lag_family_test(frame, outcome, controls):
    columns = [lag_column(k) for k in LAGS]
    data = complete_cases(frame, [outcome, *controls, *columns])
    z = sm.add_constant(data[controls].to_numpy(dtype=float), has_constant="add")
    if len(data) < z.shape[1] + 5 or np.linalg.matrix_rank(z) < z.shape[1]:
        raise ValueError("Lag-family inference requires complete common-support observations")
    y = data[outcome].to_numpy(dtype=float)
    matrices = [np.column_stack([z, data[c].to_numpy(dtype=float)]) for c in columns]
    if any(np.linalg.matrix_rank(x) < x.shape[1] for x in matrices):
        raise ValueError("A lag-family model is rank deficient")
    null_fit = sm.OLS(y, z).fit()

    def statistics(outcomes):
        return np.asarray([sm.OLS(outcomes, x).fit(cov_type="HC3").tvalues[-1] for x in matrices])

    observed = statistics(y)
    distribution = []
    for shift in range(1, len(data)):
        outcomes = null_fit.fittedvalues + np.roll(null_fit.resid, shift)
        values = statistics(outcomes)
        distribution.append(dict(shift=shift, max_abs_t=float(np.abs(values).max()),
                                 **{f"t_{k}": float(t) for k, t in zip(LAGS, values)}))
    maxima = np.asarray([r["max_abs_t"] for r in distribution])
    if not np.isfinite(np.r_[observed, maxima]).all():
        raise ValueError("Lag-family statistics are undefined")
    table = [dict(lag=k, n=len(data), t_hc3=float(t),
                  p_max_abs_t=float((1 + np.count_nonzero(maxima >= abs(t))) / len(data)))
             for k, t in zip(LAGS, observed)]
    return table, distribution, data


def matched_placebo(frame, pseudo, outcome, controls, seed):
    """Use one reproducible matched date per event and report both samples."""
    rng = np.random.default_rng(seed)
    selected = []
    for _, candidates in pseudo.sort_values(["anchor_meeting_date", "pseudo_date"]).groupby("anchor_meeting_date"):
        selected.append(candidates.iloc[rng.integers(len(candidates))])
    if not selected:
        raise ValueError("No matched pseudo-events are available for the selected sample")
    chosen = pd.DataFrame(selected)
    eligible = complete_cases(chosen, [outcome, "sim_m2", *controls])["anchor_meeting_date"]
    real = complete_cases(frame[frame["meeting_date"].isin(eligible)], [outcome, "sim_m2", *controls])
    chosen = chosen[chosen["anchor_meeting_date"].isin(real["meeting_date"])].copy()
    chosen["meeting_date"] = chosen["anchor_meeting_date"]
    records = []
    for name, data in (("fomc_common_support", real), ("matched_non_fomc", chosen)):
        record, _, _ = fit_model(data, outcome, controls)
        records.append(dict(event_type=name, **record))
    return records, chosen[["anchor_meeting_date", "pseudo_date", "predictor_date"]]


def inference_cli():
    parser = analysis_parser("Moving-block bootstrap and lag-family circular-shift inference")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--block-length", type=int)
    parser.add_argument("--seed", type=int, default=30113)
    args = parser.parse_args()
    if args.bootstrap_draws < 2:
        parser.error("--bootstrap-draws must be at least two")
    panel, metadata = load_panel(args.input_dir)
    holdout_gate(args, metadata)
    regression = args.output_dir / f"regression_{args.sample}_{args.outcome}" / "manifest.json"
    recorded = json.loads(regression.read_text())
    if (recorded["selection"] != selection(args, metadata)
            or recorded["code_sha256"] != digest(__file__)
            or recorded["protocol_sha256"] != digest(PROTOCOL)):
        raise ValueError("Save matching regression estimates before inference")
    if any(digest(regression.parent / name) != checksum for name, checksum in recorded["files"].items()):
        raise ValueError("Regression results changed after completion")
    sample = select_sample(panel, args)
    pre = matching_pre(args.outcome)
    rng = np.random.default_rng(args.seed)
    summaries, all_draws = [], []
    for name, controls in (("bivariate", []), ("pre_vol", [pre]), ("pre_vol_vix", [pre, "vix_pre"])):
        summary, draws = block_bootstrap(sample, args.outcome, controls, args.bootstrap_draws,
                                         args.block_length, rng)
        summaries.append(dict(spec=name, **summary))
        all_draws.extend(dict(spec=name, **r) for r in draws)
    family, distribution, common = lag_family_test(sample, args.outcome, [pre, "vix_pre"])
    placebo = None
    if metadata["has_pseudo_similarity"]:
        pseudo = pd.read_csv(args.input_dir / "panel" / "pseudo_panel.csv",
                             parse_dates=["anchor_meeting_date", "pseudo_date", "predictor_date"])
        placebo = matched_placebo(sample, select_sample(pseudo.assign(meeting_date=pseudo["anchor_meeting_date"]), args),
                                  args.outcome, [pre, "vix_pre"], args.seed)
    with output_stage(args.output_dir, f"inference_{args.sample}_{args.outcome}") as stage:
        pd.DataFrame(summaries).to_csv(stage / "bootstrap.csv", index=False)
        pd.DataFrame(all_draws).to_csv(stage / "bootstrap_draws.csv", index=False)
        pd.DataFrame(family).to_csv(stage / "lag_family.csv", index=False)
        pd.DataFrame(distribution).to_csv(stage / "circular_shift_distribution.csv", index=False)
        common[["meeting_date"]].to_csv(stage / "lag_family_sample.csv", index=False)
        if placebo is not None:
            pd.DataFrame(placebo[0]).to_csv(stage / "matched_non_fomc.csv", index=False)
            placebo[1].to_csv(stage / "matched_non_fomc_dates.csv", index=False)
        write_manifest(stage, {"stage": "inference", "sample": args.sample, "selection": selection(args, metadata),
                               "seed": args.seed, "bootstrap_draws": args.bootstrap_draws,
                               "block_length_rule": "explicit value or ceiling of n**(1/3)",
                               "matched_non_fomc": "estimated" if placebo is not None else "awaiting matched-date similarity input",
                               "lag_family": "HC3 max-absolute-t on common support; circular shifts of reduced-model residuals"})
    print(f"Saved {args.sample} block bootstrap and lag-family correction")
