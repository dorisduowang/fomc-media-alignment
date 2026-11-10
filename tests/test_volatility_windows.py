"""Regression tests for non-overlapping FOMC volatility windows.

These tests target ``volatility_windows`` in
``analysis/12_lead_lag_volatility.py``.  They guard against the legacy bug
where a post-meeting volatility measure was built from rolling statistics
that still contained pre-meeting returns.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKET_MODULE_PATH = PROJECT_ROOT / "analysis" / "12_lead_lag_volatility.py"


def _load_market_module():
    assert MARKET_MODULE_PATH.is_file(), (
        f"Market analysis module missing at {MARKET_MODULE_PATH}. "
        "Expected layout: ROOT/analysis/ and ROOT/tests/."
    )
    spec = importlib.util.spec_from_file_location(
        "lead_lag_volatility", MARKET_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {MARKET_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def volatility_windows():
    module = _load_market_module()
    function = getattr(module, "volatility_windows", None)
    assert callable(function), (
        "analysis/12_lead_lag_volatility.py must expose a callable "
        "volatility_windows(frame, meeting, window=5)"
    )
    return function


@pytest.fixture
def market_frame():
    dates = pd.bdate_range("2026-01-05", periods=15)
    return pd.DataFrame(
        {
            "date": dates,
            "logret": np.arange(1, len(dates) + 1, dtype=float) / 1_000,
        }
    )


def _split_dates(value):
    return [pd.Timestamp(item) for item in value.split("|")]


def test_trading_day_windows_use_the_expected_returns(
    volatility_windows, market_frame
):
    meeting = pd.Timestamp("2026-01-12")
    result = volatility_windows(market_frame, meeting)

    expected_pre = market_frame.loc[market_frame["date"] < meeting].tail(5)
    expected_strict = market_frame.loc[market_frame["date"] > meeting].head(5)
    expected_event = market_frame.loc[market_frame["date"] >= meeting].head(5)
    annualization = np.sqrt(252.0)

    assert result["meeting_date"] == meeting
    assert _split_dates(result["pre_dates"]) == expected_pre["date"].tolist()
    assert _split_dates(result["strict_dates"]) == expected_strict["date"].tolist()
    assert _split_dates(result["event_dates"]) == expected_event["date"].tolist()
    assert result["vol_pre_5"] == pytest.approx(
        expected_pre["logret"].std(ddof=1) * annualization
    )
    assert result["vol_post_strict_5"] == pytest.approx(
        expected_strict["logret"].std(ddof=1) * annualization
    )
    assert result["vol_post_event_5"] == pytest.approx(
        expected_event["logret"].std(ddof=1) * annualization
    )


def test_strict_post_window_excludes_meeting_day(volatility_windows, market_frame):
    meeting = pd.Timestamp("2026-01-12")
    result = volatility_windows(market_frame, meeting)

    pre_dates = set(_split_dates(result["pre_dates"]))
    strict_dates = set(_split_dates(result["strict_dates"]))
    event_dates = _split_dates(result["event_dates"])

    assert pre_dates.isdisjoint(strict_dates)
    assert meeting not in strict_dates
    assert event_dates[0] == meeting


def test_nontrading_meeting_has_identical_strict_and_event_windows(
    volatility_windows, market_frame
):
    meeting = pd.Timestamp("2026-01-10")  # Saturday
    result = volatility_windows(market_frame, meeting)

    assert result["strict_dates"] == result["event_dates"]
    assert result["vol_post_strict_5"] == pytest.approx(
        result["vol_post_event_5"]
    )


@pytest.mark.parametrize(
    "meeting",
    [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-23")],
)
def test_incomplete_window_is_rejected(volatility_windows, market_frame, meeting):
    with pytest.raises(AssertionError):
        volatility_windows(market_frame, meeting)


def test_strict_volatility_unaffected_by_pre_meeting_outlier(
    volatility_windows, market_frame
):
    meeting = pd.Timestamp("2026-01-12")
    baseline = volatility_windows(market_frame, meeting)
    perturbed = market_frame.copy()
    perturbed.loc[perturbed["date"] < meeting, "logret"] *= 100
    after = volatility_windows(perturbed, meeting)
    assert after["vol_post_strict_5"] == pytest.approx(
        baseline["vol_post_strict_5"]
    )
    assert after["vol_pre_5"] != pytest.approx(baseline["vol_pre_5"])


def test_three_return_window(volatility_windows, market_frame):
    meeting = pd.Timestamp("2026-01-12")
    result = volatility_windows(market_frame, meeting, window=3)
    assert len(_split_dates(result["strict_dates"])) == 3
    assert "vol_post_strict_3" in result


def test_frozen_sp500_accepts_project_csv(tmp_path):
    csv_path = tmp_path / "SPX_daily_2014_2025.csv"
    csv_path.write_text("date,spx\n2025-01-02,100\n2025-01-03,102\n")
    returns = _load_market_module().load_frozen_sp500(csv_path)
    assert returns["date"].tolist() == [pd.Timestamp("2025-01-03")]
    assert returns["logret"].tolist() == pytest.approx([np.log(1.02)])
