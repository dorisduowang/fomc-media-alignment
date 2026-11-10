"""
MACS 30113
Yuzhao (Jerry) Jin; CNETID: yuzhaojin1
Duo (Doris) Wang; CNETID: dorisduowang

This file presents the final answer to the research question. Takes the GPU-computed
cosine similarities and the Numba-computed hawkish/dovish sentiment, joins
them with S&P 500 realised volatility around each FOMC meeting, and writes
two summary tables back to S3:
    processed/lead_lag/per_meeting.parquet -- one row per (meeting, lag, source)
    processed/lead_lag/summary.csv -- lead vs lag aggregates

1. statsmodels OLS for "does media sentiment at lag k predict S&P 500
    volatility on day 0?" (one model per lag, with cluster-robust SEs)
2. scipy.stats.pearsonr for the headline lead-lag plot

Utilizing course contents:
  * Use the frozen local S&P 500 file for reproducible volatility outcomes;
    runtime market-data downloads are not permitted.
  * Reuses Numba and CuPy outputs already in S3
"""

import argparse
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import boto3
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
ENV_BUCKET = os.environ.get("BUCKET", "")

def _s3_parquet(bucket, key):
    s3 = boto3.client("s3")
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_parquet(io.BytesIO(body))

def _s3_all_parquets(bucket, prefix):
    s3 = boto3.client("s3")
    pag = s3.get_paginator("list_objects_v2")
    frames = []
    for page in pag.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                b = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                frames.append(pd.read_parquet(io.BytesIO(b)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def volatility_windows(frame, meeting, window=5):
    """Compute non-overlapping annualized realized-volatility windows.

    ``frame`` must contain trading dates (a ``date`` column or a DatetimeIndex)
    and precomputed close-to-close daily ``logret`` values.  For a meeting on
    date d, pre returns have trading dates < d, strict post returns have dates
    > d, and event-inclusive returns have dates >= d.  On a non-trading meeting
    day the two post windows coincide.  The return date is the date of the
    closing price at the *end* of the log-return interval.

    Incomplete windows, missing returns, and duplicate trading dates are
    rejected rather than silently shortening the sample or leaking returns.
    """
    assert isinstance(window, int) and window >= 2, "window must be an integer >= 2"
    assert "logret" in frame.columns, "frame must have a 'logret' column"
    dates = frame["date"] if "date" in frame.columns else frame.index
    trades = pd.DataFrame({
        "date": pd.to_datetime(dates, errors="coerce").to_numpy(),
        "logret": pd.to_numeric(frame["logret"], errors="coerce").to_numpy(),
    })
    assert trades["date"].notna().all(), "Missing or invalid trading dates"
    trades["date"] = trades["date"].dt.normalize()
    assert not trades["date"].duplicated().any(), "Duplicate trading dates"
    trades = trades.sort_values("date").reset_index(drop=True)

    meeting_date = pd.Timestamp(meeting).normalize()
    assert not pd.isna(meeting_date), "Invalid meeting date"
    pre = trades.loc[trades["date"] < meeting_date].tail(window)
    strict = trades.loc[trades["date"] > meeting_date].head(window)
    event = trades.loc[trades["date"] >= meeting_date].head(window)

    for name, selected in (("pre", pre), ("strict", strict), ("event", event)):
        assert len(selected) == window, (
            f"Incomplete {name} window for meeting {meeting_date.date()}: "
            f"expected {window}, found {len(selected)}"
        )
        assert np.isfinite(selected["logret"].to_numpy()).all(), (
            f"Nonfinite log returns in {name} window for {meeting_date.date()}"
        )
    assert set(pre["date"]).isdisjoint(set(strict["date"])), (
        "Pre and strict post return dates must not overlap"
    )

    annualization = np.sqrt(252.0)
    def _dates(selected):
        return "|".join(d.strftime("%Y-%m-%d") for d in selected["date"])
    def _vol(selected):
        return float(selected["logret"].std(ddof=1) * annualization)
    def _rv(selected):
        return float(np.square(selected["logret"].to_numpy()).sum())

    n = str(window)
    return {
        "meeting_date": meeting_date,
        "pre_dates": _dates(pre),
        "strict_dates": _dates(strict),
        "event_dates": _dates(event),
        f"vol_pre_{n}": _vol(pre),
        f"vol_post_strict_{n}": _vol(strict),
        f"vol_post_event_{n}": _vol(event),
        f"rv_pre_{n}": _rv(pre),
        f"rv_post_strict_{n}": _rv(strict),
        f"rv_post_event_{n}": _rv(event),
    }


def load_frozen_sp500(csv_path=None):
    """Load frozen S&P 500 daily closing prices (no runtime downloads)."""
    if csv_path is None:
        csv_path = (
            Path(__file__).resolve().parent.parent
            / "MakkaPakka update" / "data_reference" / "market"
            / "SPX_daily_2014_2025.csv"
        )
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Frozen S&P 500 file not found: {csv_path}. "
            "Supply --spx-csv with the local frozen market CSV path."
        )
    raw = pd.read_csv(csv_path)
    by_lower = {str(c).strip().lower(): c for c in raw.columns}
    date_col = next((by_lower[c] for c in ("date", "datetime") if c in by_lower), None)
    assert date_col is not None, f"No date column in {csv_path}"
    close_col = next((by_lower[c] for c in (
        "close", "adj close", "adj_close", "spx", "spx_close", "sp500_close"
    ) if c in by_lower), None)
    ret_col = by_lower.get("logret")
    assert close_col is not None or ret_col is not None, (
        f"No Close or logret column in frozen file {csv_path}; "
        f"available columns: {list(raw.columns)}"
    )
    data = pd.DataFrame({"date": pd.to_datetime(raw[date_col], errors="coerce")})
    assert data["date"].notna().all(), "Frozen S&P 500 file has invalid dates"
    data["date"] = data["date"].dt.normalize()
    assert not data["date"].duplicated().any(), "Duplicate S&P 500 trading dates"
    if close_col is not None:
        prices = pd.to_numeric(
            raw[close_col].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )
        assert (prices.notna() & (prices > 0)).all(), (
            "Frozen S&P 500 close prices must be positive and nonmissing"
        )
        data["close"] = prices
    else:
        data["logret"] = pd.to_numeric(raw[ret_col], errors="coerce")
    data = data.sort_values("date").reset_index(drop=True)
    if close_col is not None:
        data["logret"] = np.log(data["close"]).diff()
    # Drop the one unavailable return at the start of a close-price series.
    if close_col is not None:
        data = data.iloc[1:].reset_index(drop=True)
    assert np.isfinite(data["logret"].to_numpy()).all(), (
        "Invalid log returns in frozen S&P 500 file"
    )
    return data[["date", "logret"]]


def fetch_sp500_volatility(meeting_dates, csv_path=None):
    """Construct explicit pre/strict-post/event volatility from frozen data."""
    spx = load_frozen_sp500(csv_path)
    rows = []
    for meeting in meeting_dates:
        meeting_day = pd.Timestamp(meeting).normalize()
        record = {"meeting_date": str(meeting_day.date())}
        for width in (3, 5, 7, 10):
            result = volatility_windows(spx, meeting_day, window=width)
            for key, value in result.items():
                if key == "meeting_date":
                    continue
                if key in ("pre_dates", "strict_dates", "event_dates"):
                    # Keep the exact return dates for all tested horizons.
                    record[f"{key}_{width}"] = value
                else:
                    record[key] = value
        # Backwards-compatible column names, with corrected strict outcome.
        record["vol_pre"] = record["vol_pre_5"]
        record["vol_post"] = record["vol_post_strict_5"]
        rows.append(record)
    return pd.DataFrame(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default=ENV_BUCKET,
                   help="S3 lake bucket (default: from .env BUCKET)")
    p.add_argument("--spx-csv", default=None,
                   help="Path to the frozen S&P 500 CSV; no network market-data download")
    args = p.parse_args()

    print("[analysis] loading similarity, sentiment, FOMC dates ...")
    sim = _s3_parquet(args.bucket, "processed/similarity/cosine.parquet")
    sent = _s3_parquet(args.bucket, "processed/sentiment/sentiment.parquet")

    meetings = pd.to_datetime(sim["meeting_date"].unique())
    print(f"[analysis] {len(meetings)} FOMC meetings under analysis")

    vol = fetch_sp500_volatility(meetings, csv_path=args.spx_csv)
    print(f"[analysis] S&P 500 volatility for {len(vol)} meetings")

    # Per-source mean sentiment in the same 7-day lag windows the CuPy job used
    sent["doc_date"] = pd.to_datetime(sent["doc_date"])
    sent_by_day = (sent.groupby(["source", "doc_date"], as_index=False)
                       ["net_per_1k"].mean())

    rows = []
    for md in meetings:
        for lag in range(-7, 8):
            day = md + pd.Timedelta(days=lag)
            for src in ("nyt", "guardian", "gdelt"):
                sub = sent_by_day[(sent_by_day["source"] == src) &
                                  (sent_by_day["doc_date"] == day)]
                rows.append({
                    "meeting_date": str(md.date()),
                    "lag_days": lag,
                    "source": src,
                    "mean_net_per_1k": float(sub["net_per_1k"].mean())
                                        if len(sub) else np.nan,
                })
    sent_panel = pd.DataFrame(rows)

    # Join everything
    panel = (sim.merge(sent_panel, on=["meeting_date", "lag_days", "source"],
                       how="left")
                .merge(vol, on="meeting_date", how="left"))

    # Save per-meeting panel
    out = io.BytesIO()
    panel.to_parquet(out, index=False)
    out.seek(0)
    s3 = boto3.client("s3")
    s3.put_object(Bucket=args.bucket,
                  Key="processed/lead_lag/per_meeting.parquet",
                  Body=out.read())

    # At each lag, mean similarity and Pearson r with vol_post
    from scipy.stats import pearsonr
    summary_rows = []
    for src in ("nyt", "guardian", "gdelt"):
        for lag in range(-7, 8):
            sub = panel[(panel["source"] == src) & (panel["lag_days"] == lag)]
            sub = sub.dropna(subset=["cosine_sim", "vol_post"])
            if len(sub) < 5:
                continue
            r, pval = pearsonr(sub["cosine_sim"], sub["vol_post"])
            summary_rows.append({
                "source": src, "lag_days": lag,
                "n": int(len(sub)),
                "mean_cosine":   float(sub["cosine_sim"].mean()),
                "mean_sent":     float(sub["mean_net_per_1k"].mean(skipna=True)),
                "pearson_r_vol": float(r),
                "p_value": float(pval),
            })
    summary = pd.DataFrame(summary_rows).sort_values(["source", "lag_days"])
    print("\n=== Lead-lag summary ===")
    print(summary.to_string(index=False))

    csv_buf = io.StringIO()
    summary.to_csv(csv_buf, index=False)
    s3.put_object(Bucket=args.bucket,
                  Key="processed/lead_lag/summary.csv",
                  Body=csv_buf.getvalue().encode("utf-8"))
    print(f"\n[done] s3://{args.bucket}/processed/lead_lag/")

if __name__ == "__main__":
    main()