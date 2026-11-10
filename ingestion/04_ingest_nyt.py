"""
MACS 30113
Yuzhao (Jerry) Jin; CNETID: yuzhaojin1
Duo (Doris) Wang; CNETID: dorisduowang

This file is for fetching NYT Article Search API results for "Federal Reserve"
(2015-2025) and upload one JSON per article to S3.

We learned from initial "04_setup_stepfunctions_nyt.py" that the original 
approach, using AWS Lambda + Step Functions is failed because:
    1. start_sync_execution has a 5-minute hard AWS limit — our 132-task
       execution takes 15-20 minutes, causing ReadTimeoutError.
    2. Lambda invocation rate limits (429 TooManyRequestsException) caused
       the Step Functions execution to FAIL even with MaxConcurrency=2.

This version uses local ThreadPoolExecutor(max_workers=2), exactly the
same pattern as 05_ingest_guardian.py, which runs reliably.
The S3 catalog Lambda from 02_lambda_s3_trigger_setup.py still fires
automatically on every S3 PUT — the event-driven catalog is preserved.

Multi-key support:
NYT_KEYS in .env = comma-separated list of API keys.
  
Vocareum compliance:
  - No Lambda invocations (ThreadPoolExecutor runs locally)
  - No EC2 instances
  - Catalog Lambda (02) still auto-fires on S3 PUTs, capped at 5
  - Total concurrent Lambda from this script: up to 2 catalog Lambda
    invocations (one per active worker uploading) < 10 limit 

Utilizing course contents: 
- Week 2: ThreadPoolExecutor for I/O-bound parallel API calls
- Week 4-5: Boto3 + S3 data lake
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import boto3
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
ENV_BUCKET = os.environ.get("BUCKET", "")
_nyt_raw = os.environ.get("NYT_KEYS", "")
ENV_NYT_KEYS = [k.strip() for k in _nyt_raw.split(",") if k.strip()]

# NYT API Search setup
BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
QUERY = "Federal Reserve"
SORT = "newest"
SLEEP_TIME = 12 # seconds between pages — 5 requests/minute rate limit
MAX_PAGES = 10 # max 100 articles per month
TIMEOUT = 30 # request timeout

def make_session():
    """Requests session with User-Agent from the team's working notebook."""
    s = requests.Session()
    s.headers.update({"User-Agent": "nyt-articlesearch-scraper/1.0"})
    return s

def fetch_month(api_key, year, month, session):
    """
    Fetch all articles for one (year, month) batch.
    Mirrors the notebook's fetch logic exactly.

    end_date = first day of NEXT month (exclusive) — notebook pattern:
        START_DATE = "20210101"
        END_DATE   = "20220101"  (exclusive upper bound)
    """
    if month == 12:
        next_year, next_month = year + 1, 1
    else:
        next_year, next_month = year, month + 1

    begin = f"{year:04d}{month:02d}01"
    end   = f"{next_year:04d}{next_month:02d}01"

    articles = []
    for page in range(MAX_PAGES):
        params = {
            "q":          QUERY,
            "begin_date": begin,
            "end_date":   end,
            "sort":       SORT,
            "page":       page,
            "api-key":    api_key,
        }
        try:
            r = session.get(BASE_URL, params=params, timeout=TIMEOUT)
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                print(f"  [{year}-{month:02d}] p{page}: 429 rate limit — backing off 60s")
                time.sleep(60)
                try:
                    r = session.get(BASE_URL, params=params, timeout=TIMEOUT)
                    r.raise_for_status()
                except Exception:
                    break
            else:
                break
        except Exception:
            break

        docs = r.json().get("response", {}).get("docs", [])
        hits = r.json().get("response", {}).get("meta", {}).get("hits", 0)
        if not docs:
            break

        articles.extend(docs)

        total_pages = min((hits + 9) // 10, MAX_PAGES)
        if page >= total_pages - 1:
            break

        time.sleep(SLEEP_TIME)   # respect 5 requests/minute rate limit

    return articles

def upload_articles(s3, bucket, docs):
    """Upload each article as a JSON object to S3."""
    written = 0
    for article in docs:
        pub    = article.get("pub_date", "")[:10]
        raw_id = article.get("uri", article.get("_id", ""))
        doc_id = "nyt_" + raw_id.split("/")[-1]
        key    = f"raw/nyt/{pub}/{doc_id}.json"

        payload = {
            "source":      "nyt",
            "doc_id":      doc_id,
            "doc_date":    pub,
            "url":         article.get("web_url", ""),
            "headline":    (article.get("headline") or {}).get("main", ""),
            "abstract":    article.get("abstract", ""),
            "snippet":     article.get("snippet", ""),
            "keywords":    ", ".join(
                kw.get("value", "") for kw in article.get("keywords", [])),
            "news_desk":   article.get("news_desk", ""),
            "section":     article.get("section_name", ""),
            "word_count":  article.get("word_count", 0),
            "text": " ".join(filter(None, [
                (article.get("headline") or {}).get("main", ""),
                article.get("abstract", ""),
                article.get("snippet", ""),
            ])),
            "ingested_at": datetime.utcnow().isoformat(),
        }
        s3.put_object(
            Bucket=bucket, Key=key,
            Body=json.dumps(payload).encode("utf-8"),
            ContentType="application/json",
        )
        written += 1
    return written

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket",     default=ENV_BUCKET,
                   help="S3 lake bucket (default: from .env BUCKET)")
    p.add_argument("--nyt-keys",   default=",".join(ENV_NYT_KEYS),
                   help="Comma-separated NYT API keys (default: from .env NYT_KEYS)")
    p.add_argument("--start-year", type=int, default=2015)
    p.add_argument("--end-year",   type=int, default=2025)
    p.add_argument("--workers",    type=int, default=2,
                   help="Parallel threads (default 2 — keeps Lambda invocations low)")
    args = p.parse_args()

    if not args.bucket:
        p.error("--bucket is required, or set BUCKET in .env")
    keys = [k.strip() for k in args.nyt_keys.split(",") if k.strip()]
    if not keys:
        p.error("--nyt-keys is required, or set NYT_KEYS in .env")

    # Build (year, month) task list with round-robin key assignment
    pairs = [
        (y, m)
        for y in range(args.start_year, args.end_year + 1)
        for m in range(1, 13)
    ]
    tasks = [
        (keys[i % len(keys)], y, m)
        for i, (y, m) in enumerate(pairs)
    ]

    print(f"[nyt] bucket     : {args.bucket}")
    print(f"[nyt] query      : {QUERY}  (all sections, no fq filter)")
    print(f"[nyt] period     : {args.start_year}–{args.end_year}")
    print(f"[nyt] tasks      : {len(tasks)} months  workers={args.workers}")
    print(f"[nyt] keys       : {len(keys)} key(s) — round-robin assigned")
    for ki, k in enumerate(keys):
        count = sum(1 for (key, y, m) in tasks if key == k)
        print(f"         key {ki+1} ({k[:8]}...) → {count} tasks"
              f" (<= {count * MAX_PAGES} requests, limit 500/day)")
    print(f"[nyt] Vocareum   : local threads — 0 Lambda invocations from fetching")
    print(f"                   S3 PUTs trigger catalog Lambda (capped at 5) ✅")

    s3      = boto3.client("s3")
    session = make_session()
    total   = 0
    t0      = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(fetch_month, api_key, y, m, session): (y, m)
            for (api_key, y, m) in tasks
        }
        for fut in as_completed(futs):
            y, m = futs[fut]
            try:
                docs = fut.result()
                n    = upload_articles(s3, args.bucket, docs)
                total += n
                if n > 0:
                    print(f"[ok]  {y:04d}-{m:02d}  +{n} articles")
            except Exception as e:
                print(f"[err] {y:04d}-{m:02d}: {e}")

    elapsed = time.time() - t0
    print(f"\n[done] {total:,} NYT articles in {elapsed:.1f}s")
    print(f"       query=\"{QUERY}\"  all sections")
    print(f"       s3://{args.bucket}/raw/nyt/")

if __name__ == "__main__":
    main()