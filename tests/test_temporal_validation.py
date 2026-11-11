"""Regression checks for sample selection, temporal inference, and stage integrity."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("temporal_validation", ROOT / "scripts/analysis/temporal_validation.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    rng = np.random.default_rng(30113)
    calendar = pd.to_datetime(pd.read_csv(ROOT / "MakkaPakka update/data_reference/meetings_all88.csv")["meeting_date"])
    meetings = calendar[~calendar.isin(pd.to_datetime(["2015-01-28", "2025-06-18"]))]
    rows = []
    for day in meetings:
        for corpus in ("direct_fed", "broad_macro"):
            for weighting in ("article", "domain_equal"):
                for lag in analysis.LAGS:
                    rows.append(dict(meeting_date=day, source="gdelt", corpus=corpus, weighting=weighting,
                                     lag_days=lag, cosine_sim=rng.uniform(0.05, 0.8),
                                     model_fit_cutoff="2019-03-23", representation_type="gkg_themes_orgs_persons",
                                     filter_version="synthetic-rules", n_docs=20, n_urls=19, n_domains=4,
                                     mean_native_tone=rng.normal(), vol_post=-999.0))
    long = pd.DataFrame(rows)
    source = tmp_path / "similarity.parquet"
    long.to_parquet(source, index=False)
    market = tmp_path / "market"
    market.mkdir()
    dates = pd.bdate_range("2014-12-01", "2026-02-01")
    prices = 100 * np.exp(rng.normal(0, 0.01, len(dates)).cumsum())
    pd.DataFrame(dict(date=dates, spx=prices)).to_csv(market / "SPX_daily_2014_2025.csv", index=False)
    pd.DataFrame(dict(observation_date=dates, VIXCLS=rng.uniform(10, 40, len(dates)))).to_csv(market / "VIX_2014_2025.csv", index=False)
    mps = tmp_path / "surprises.csv"
    pd.DataFrame(dict(meeting_date=meetings, mps=np.where(np.arange(len(meetings)) % 2 == 0,
                                                        rng.normal(size=len(meetings)), np.nan))).to_csv(mps, index=False)
    output = tmp_path / "analysis"
    monkeypatch.setattr(sys, "argv", ["panel", "--per-meeting", str(source), "--market-dir", str(market),
                                     "--output-dir", str(output), "--mps-file", str(mps),
                                     "--mps-definition", "usmpd_statement"])
    analysis.panel_cli()
    return output, long


def test_panel_retains_corpus_weighting_and_rebuilds_outcomes(synthetic_run):
    output, _ = synthetic_run
    panel, _ = analysis.load_panel(output)
    assert len(panel) == 86 * 4
    assert not panel.duplicated(analysis.KEYS).any()
    assert panel["vol_post"].gt(0).all()
    np.testing.assert_allclose(panel["vol_post"], panel["vol_post_strict_5"])
    assert panel.groupby("sample").size().to_dict() == {"discovery": 33 * 4, "holdout": 53 * 4}
    audit = pd.read_csv(output / "panel/market_window_audit.csv")
    for _, row in audit.iterrows():
        for width in (3, 5, 7, 10):
            assert set(row[f"pre_dates_{width}"].split("|")).isdisjoint(row[f"strict_dates_{width}"].split("|"))


def test_mps_missingness_does_not_shrink_primary_models(synthetic_run, monkeypatch):
    output, _ = synthetic_run
    monkeypatch.setattr(sys, "argv", ["regression", "--input-dir", str(output), "--output-dir", str(output)])
    analysis.regression_cli()
    folder = output / "regression_holdout_vol_post_strict_5"
    models = pd.read_csv(folder / "models.csv").set_index("spec")
    assert models.loc[["bivariate", "pre_vol", "pre_vol_vix"], "n"].eq(53).all()
    assert models.loc["mps_bridge", "n"] < 53
    assert models.loc[["mps_bridge", "mps_signed", "mps_absolute", "mps_squared"], "n"].nunique() == 1
    membership = pd.read_csv(folder / "model_samples.csv")
    mps_sets = [set(membership.loc[membership["spec"] == name, "meeting_date"])
                for name in ("mps_bridge", "mps_signed", "mps_absolute", "mps_squared")]
    assert all(s == mps_sets[0] for s in mps_sets)


def test_secondary_requires_completed_unchanged_holdout(synthetic_run, monkeypatch):
    output, _ = synthetic_run
    argv = ["regression", "--input-dir", str(output), "--output-dir", str(output)]
    monkeypatch.setattr(sys, "argv", [*argv, "--sample", "pooled"])
    with pytest.raises(FileNotFoundError):
        analysis.regression_cli()
    monkeypatch.setattr(sys, "argv", argv)
    analysis.regression_cli()
    monkeypatch.setattr(sys, "argv", [*argv, "--sample", "pooled"])
    analysis.regression_cli()
    models = output / "regression_holdout_vol_post_strict_5/models.csv"
    models.write_text(models.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after completion"):
        analysis.regression_cli()


def test_hc3_and_window_control_match_independent_fit(synthetic_run):
    output, _ = synthetic_run
    panel, _ = analysis.load_panel(output)
    panel = panel.query("source == 'gdelt' and corpus == 'direct_fed' and weighting == 'domain_equal' and sample == 'holdout'")
    outcome = "vol_post_event_10"
    pre = analysis.matching_pre(outcome)
    assert pre == "vol_pre_10"
    record, data, _ = analysis.fit_model(panel, outcome, [pre, "vix_pre"])
    independent = sm.OLS(data[outcome], sm.add_constant(data[["sim_m2", pre, "vix_pre"]])).fit(cov_type="HC3", use_t=True)
    assert record["beta"] == pytest.approx(independent.params["sim_m2"])
    assert record["se_hc3"] == pytest.approx(independent.bse["sim_m2"])


def test_circular_family_is_ordered_and_corrects_all_lags(synthetic_run):
    output, _ = synthetic_run
    panel, _ = analysis.load_panel(output)
    panel = panel.query("corpus == 'direct_fed' and weighting == 'domain_equal' and sample == 'holdout'")
    observed, distribution, common = analysis.lag_family_test(panel, analysis.PRIMARY_OUTCOME, ["vol_pre_5", "vix_pre"])
    shuffled, shifted, _ = analysis.lag_family_test(panel.sample(frac=1, random_state=4), analysis.PRIMARY_OUTCOME, ["vol_pre_5", "vix_pre"])
    assert observed == shuffled and distribution == shifted
    assert len(distribution) == len(common) - 1
    maxima = np.asarray([row["max_abs_t"] for row in distribution])
    for row in observed:
        single = np.asarray([r[f"t_{row['lag']}"] for r in distribution])
        assert row["p_max_abs_t"] == pytest.approx((1 + np.count_nonzero(maxima >= abs(row["t_hc3"]))) / len(common))
        assert row["p_max_abs_t"] >= (1 + np.count_nonzero(abs(single) >= abs(row["t_hc3"]))) / len(common)


def test_blocks_and_stage_rollback(tmp_path):
    index = analysis.moving_block_indices(13, 4, np.random.default_rng(2))
    assert len(index) == 13
    for start in range(0, 13, 4):
        assert np.all(np.diff(index[start:start + 4]) == 1)
    with pytest.raises(RuntimeError):
        with analysis.output_stage(tmp_path, "failed") as stage:
            (stage / "partial.csv").write_text("partial")
            raise RuntimeError("simulated failure")
    assert not (tmp_path / "failed").exists()


def test_duplicate_and_mixed_model_inputs_are_rejected(synthetic_run):
    _, long = synthetic_run
    with pytest.raises(ValueError, match="unique"):
        analysis.validate_long(pd.concat([long, long.iloc[:1]]))
    long.loc[0, "model_fit_cutoff"] = "2025-01-01"
    with pytest.raises(ValueError, match="cutoff"):
        analysis.validate_long(long)


def test_pseudo_calendar_excludes_real_events_and_matches_month():
    dates = pd.to_datetime(["2019-03-20", "2020-03-15", "2025-12-10"])
    requests = analysis.pseudo_requests(dates)
    assert not requests["pseudo_date"].isin(dates).any()
    assert requests["pseudo_date"].dt.weekday.equals(requests["anchor_meeting_date"].dt.weekday)
    assert requests["pseudo_date"].dt.to_period("M").equals(requests["anchor_meeting_date"].dt.to_period("M"))
    assert requests["predictor_date"].equals(requests["pseudo_date"] - pd.Timedelta(days=2))


def test_matched_placebo_uses_same_events_and_rejects_fomc_dates(synthetic_run):
    output, _ = synthetic_run
    panel, _ = analysis.load_panel(output)
    real = panel.query("corpus == 'direct_fed' and weighting == 'domain_equal' and sample == 'holdout'")
    calendar = pd.to_datetime(pd.read_csv(ROOT / "MakkaPakka update/data_reference/meetings_all88.csv")["meeting_date"])
    requests = analysis.pseudo_requests(calendar)
    requests = requests[requests["anchor_meeting_date"].isin(real["meeting_date"])].groupby("anchor_meeting_date").first().reset_index()
    requests["source"], requests["corpus"], requests["weighting"] = "gdelt", "direct_fed", "domain_equal"
    requests["cosine_sim"] = np.random.default_rng(13).uniform(0.05, 0.8, len(requests))
    requests["model_fit_cutoff"] = "2019-03-23"
    requests["filter_version"] = "synthetic-rules"
    requests["representation_type"] = "gkg_themes_orgs_persons"
    source = output.parent / "pseudo.csv"
    requests.to_csv(source, index=False)
    market = output.parent / "market"
    pseudo = analysis.build_pseudo_panel(source, calendar, market / "SPX_daily_2014_2025.csv",
                                         market / "VIX_2014_2025.csv", "synthetic-rules")
    pseudo["meeting_date"] = pseudo["anchor_meeting_date"]
    records, chosen = analysis.matched_placebo(real, pseudo, analysis.PRIMARY_OUTCOME, ["vol_pre_5", "vix_pre"], 30113)
    assert records[0]["n"] == records[1]["n"] == len(chosen) == 53
    requests.loc[0, "pseudo_date"] = requests.loc[0, "anchor_meeting_date"]
    requests.to_csv(source, index=False)
    with pytest.raises(ValueError, match="matched non-FOMC"):
        analysis.build_pseudo_panel(source, calendar, market / "SPX_daily_2014_2025.csv",
                                    market / "VIX_2014_2025.csv", "synthetic-rules")


def test_versioned_cli_runs_through_notebook_entrypoints(synthetic_run):
    prepared, _ = synthetic_run
    folder = prepared.parent
    output = folder / "cli_run"
    commands = [
        ["00_build_panel.py", "--per-meeting", str(folder / "similarity.parquet"),
         "--market-dir", str(folder / "market"), "--output-dir", str(output)],
        ["12_main_regression.py", "--input-dir", str(output), "--output-dir", str(output)],
        ["13_identification.py", "--input-dir", str(output), "--output-dir", str(output), "--bootstrap-draws", "40"],
    ]
    for script, *args in commands:
        subprocess.run([sys.executable, str(ROOT / "MakkaPakka update/scripts/analysis" / script), *args],
                       check=True, capture_output=True, text=True)
    models = pd.read_csv(output / "regression_holdout_vol_post_strict_5/models.csv")
    assert models["n"].eq(53).all() and len(models) == 3
    infer = output / "inference_holdout_vol_post_strict_5"
    assert len(pd.read_csv(infer / "lag_family.csv")) == 15
    assert len(pd.read_csv(infer / "bootstrap_draws.csv")) == 3 * 40
    manifest = json.loads((infer / "manifest.json").read_text())
    assert manifest["matched_non_fomc"] == "awaiting matched-date similarity input"
