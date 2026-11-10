"""
MACS 30113
Yuzhao (Jerry) Jin; CNETID: yuzhaojin1
Duo (Doris) Wang; CNETID: dorisduowang

This file is for scrapping Guardian US-edition coverage of the Federal Reserve 
2015-2025 from the Guardian Open Platform's free content endpoint, in parallel.

API reference:  https://open-platform.theguardian.com/documentation/search
Endpoint:       https://content.guardianapis.com/search
Rate limit:     5,000 requests/day (free developer tier)
Max page-size:  200

Utilizing Course Contents: 
- Week 2: ThreadPoolExecutor for I/O-bound parallel API calls
- Week 4-5: Boto3 + S3 sink
"""
import argparse
import calendar
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import boto3
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
ENV_BUCKET  = os.environ.get("BUCKET",  "")
ENV_GDN_KEY = os.environ.get("GDN_KEY", "")

API   = "https://content.guardianapis.com/search"
QUERY = '"Federal Reserve"'   # exact phrase — inner quotes are essential

def fetch_month(api_key, year, month):
    """
    Fetch all Guardian articles matching the exact phrase "Federal Reserve"
    in the business or us-news section for one calendar month.

    Pagination follows the official documentation pattern:
        read data["response"]["pages"] from first response,
        loop until current_page > total_pages.
    """
    last_day = calendar.monthrange(year, month)[1]

    params = {
        "api-key": api_key,
        "q": QUERY, # exact phrase "Federal Reserve"
        "section": "business|us-news",  # OR across two sections
        "from-date": f"{year:04d}-{month:02d}-01",
        "to-date": f"{year:04d}-{month:02d}-{last_day:02d}",
        "page-size": 200, # API maximum
        "order-by": "relevance", # most on-topic first
        "show-fields": "bodyText,body,headline,trailText,byline",
        "show-tags": "keyword", # editorial tags for Spark filtering
        "production-office": "us", # US office only (lowercase per docs)
    }

    all_docs     = []
    current_page = 1
    total_pages  = 1   # updated after first response

    while current_page <= total_pages:
        params["page"] = current_page
        try:
            r = requests.get(API, params=params, timeout=30)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"[warn] {year}-{month:02d} p{current_page}: {e}")
            break

        resp        = r.json().get("response", {})
        total_pages = resp.get("pages", 1)
        results     = resp.get("results", [])
        all_docs.extend(results)

        if not results:
            break
        current_page += 1
        time.sleep(0.2)

    return all_docs


def _extract_text(fields):
    """
    Best-available plain text from a Guardian article's fields.
    Priority: bodyText (plain text) > body (HTML, stripped) > trailText.
    """
    body_text = fields.get("bodyText", "").strip()
    if body_text:
        return body_text[:50000]

    body_html = fields.get("body", "").strip()
    if body_html:
        return BeautifulSoup(body_html, "html.parser").get_text(" ", strip=True)[:50000]

    return fields.get("trailText", "")[:5000]

def upload(s3, bucket, docs):
    n = 0
    for d in docs:
        pub    = d.get("webPublicationDate", "")[:10]
        doc_id = "gdn_" + d.get("id", "").replace("/", "_")[:120]
        key    = f"raw/guardian/{pub}/{doc_id}.json"
        fields = d.get("fields", {})
        text   = _extract_text(fields)

        # Extract keyword tags returned by show-tags=keyword
        tags = [t.get("webTitle", "") for t in d.get("tags", [])]

        s3.put_object(
            Bucket=bucket, Key=key,
            Body=json.dumps({
                "source":      "guardian",
                "doc_id":      doc_id,
                "doc_date":    pub,
                "url":         d.get("webUrl"),
                "headline":    d.get("webTitle"),
                "section":     d.get("sectionId"),
                "tags":        tags,
                "text":        text,
                "ingested_at": datetime.utcnow().isoformat(),
            }).encode("utf-8"),
            ContentType="application/json",
        )
        n += 1
    return n

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket",       default=ENV_BUCKET,
                   help="S3 lake bucket (default: from .env BUCKET)")
    p.add_argument("--guardian-key", default=ENV_GDN_KEY,
                   help="Guardian API key (default: from .env GDN_KEY)")
    p.add_argument("--start-year",   type=int, default=2015)
    p.add_argument("--end-year",     type=int, default=2025)
    p.add_argument("--workers",      type=int, default=4,
                   help="parallel threads (default 4 — keeps catalog Lambda under 5 concurrent)")
    args = p.parse_args()

    if not args.bucket:
        p.error("--bucket is required, or set BUCKET in your .env file")
    if not args.guardian_key:
        p.error("--guardian-key is required, or set GDN_KEY in your .env file")

    s3    = boto3.client("s3")
    tasks = [
        (y, m)
        for y in range(args.start_year, args.end_year + 1)
        for m in range(1, 13)
    ]

    print(f'[guardian] query         : {QUERY}')
    print(f'[guardian] section filter: business|us-news')
    print(f'[guardian] order-by      : relevance')
    print(f'[guardian] page-size     : 200 (API max)')
    print(f'[guardian] tasks         : {len(tasks)} months  workers={args.workers}')

    total = 0
    t0    = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(fetch_month, args.guardian_key, y, m): (y, m)
            for y, m in tasks
        }
        for fut in as_completed(futs):
            y, m = futs[fut]
            try:
                docs = fut.result()
                n    = upload(s3, args.bucket, docs)
                total += n
                print(f"[ok]  {y:04d}-{m:02d}  +{n}")
            except Exception as e:
                print(f"[err] {y:04d}-{m:02d}: {e}")

    print(f"\n[done] {total:,} Guardian articles in {time.time()-t0:.1f}s")
    print(f'       query="{QUERY}"  section=business|us-news')
    print(f"       s3://{args.bucket}/raw/guardian/")

if __name__ == "__main__":
    main()