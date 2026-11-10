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
  * Week 5: pull market data via Boto3-free yfinance, store result in S3
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

def fetch_sp500_volatility(meeting_dates):
    """5-day realised vol of log returns around each FOMC meeting."""
    import yfinance as yf
    spans = (min(meeting_dates) - pd.Timedelta(days=30),
             max(meeting_dates) + pd.Timedelta(days=30))
    spx = yf.download("^GSPC", start=spans[0], end=spans[1], progress=False)
    spx["logret"] = np.log(spx["Close"]).diff()
    spx["vol5"] = spx["logret"].rolling(5).std() * np.sqrt(252)

    rows = []
    for d in meeting_dates:
        # find the next trading day on/after d
        post = spx.index[spx.index >= pd.Timestamp(d)][:5]
        pre  = spx.index[spx.index <  pd.Timestamp(d)][-5:]
        if len(post) and len(pre):
            rows.append({
                "meeting_date": str(pd.Timestamp(d).date()),
                "vol_pre":  float(spx.loc[pre,  "vol5"].mean()),
                "vol_post": float(spx.loc[post, "vol5"].mean()),
            })
    return pd.DataFrame(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default=ENV_BUCKET,
                   help="S3 lake bucket (default: from .env BUCKET)")
    args = p.parse_args()

    print("[analysis] loading similarity, sentiment, FOMC dates ...")
    sim = _s3_parquet(args.bucket, "processed/similarity/cosine.parquet")
    sent = _s3_parquet(args.bucket, "processed/sentiment/sentiment.parquet")

    meetings = pd.to_datetime(sim["meeting_date"].unique())
    print(f"[analysis] {len(meetings)} FOMC meetings under analysis")

    vol = fetch_sp500_volatility(meetings)
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