"""
MACS 30113
Yuzhao (Jerry) Jin; CNETID: yuzhaojin1
Duo (Doris) Wang; CNETID: dorisduowang

This file is for craping FOMC meeting minutes from the Federal Reserve website
(2015-2025) and upload each as JSON to s3://<bucket>/raw/fomc/.

This scraping process utilizes User-Agent format, URL discovery regex, div#content 
text extraction, 2-second sleep, and KNOWN_FOMC_DATES safety-net pass for any gaps
to ensure that scrapped data are uploaded to boto3 S3

URL format confirmed from the Fed website:
  https://www.federalreserve.gov/monetarypolicy/fomcminutes<YYYYMMDD>.htm
  e.g. https://www.federalreserve.gov/monetarypolicy/fomcminutes20210127.htm
       https://www.federalreserve.gov/monetarypolicy/fomcminutes20150128.htm

Using the following course content: 
- Week 4 & 5: Boto3 and S3 data lake.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import boto3
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
ENV_BUCKET = os.environ.get("BUCKET", "")

BASE     = "https://www.federalreserve.gov"
CAL_URL  = f"{BASE}/monetarypolicy/fomccalendars.htm"
HIST_URL = f"{BASE}/monetarypolicy/fomchistorical{{year}}.htm"
MIN_URL  = f"{BASE}/monetarypolicy/fomcminutes{{date}}.htm"
HEADERS  = {"User-Agent": "MACS30113-MakkaPakka-Academic/1.0"}

# We found istorical archive pages only exist for completed/archived years up to 2020.
# Year 2021 and onward appear only on the main calendar page.
HIST_CUTOFF_YEAR = 2020

# Full FOMC meeting schedule 2015-2025 (8 meetings/year).
# Used in Pass 3 as a safety net to catch any links missed by HTML scraping.
KNOWN_FOMC_DATES = [
    # 2015
    "20150128","20150318","20150429","20150617",
    "20150729","20150917","20151028","20151216",
    # 2016
    "20160127","20160316","20160427","20160615",
    "20160727","20160921","20161102","20161214",
    # 2017
    "20170201","20170315","20170503","20170614",
    "20170726","20170920","20171101","20171213",
    # 2018
    "20180131","20180321","20180502","20180613",
    "20180801","20180926","20181108","20181219",
    # 2019
    "20190130","20190320","20190501","20190619",
    "20190731","20190918","20191030","20191211",
    # 2020
    "20200129","20200311","20200415","20200610",
    "20200729","20200916","20201105","20201216",
    # 2021
    "20210127","20210317","20210428","20210616",
    "20210728","20210922","20211103","20211215",
    # 2022
    "20220126","20220316","20220504","20220615",
    "20220727","20220921","20221102","20221214",
    # 2023
    "20230201","20230322","20230503","20230614",
    "20230726","20230920","20231101","20231213",
    # 2024
    "20240131","20240320","20240501","20240612",
    "20240731","20240918","20241107","20241218",
    # 2025 (published so far)
    "20250129","20250319","20250507","20250618",
]

# URL Discovery
def _safe_get(url, max_retries=3):
    """
    GET with retry on transient errors only.
    Returns Response on 200, None on 404 (no retry),
    retries up to max_retries on 5xx or network errors.
    """
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            print(f"[retry] network error: {e}; sleep {wait}s")
            time.sleep(wait)
            continue

        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return None   # permanent — page does not exist, do not retry
        # 5xx transient error
        wait = 2 ** attempt
        print(f"[retry] {url} status={r.status_code}; sleep {wait}s")
        time.sleep(wait)

    return None

def _extract_links(soup, start_year, end_year):
    """Pull fomcminutes<YYYYMMDD>.htm links from a BeautifulSoup object."""
    found = []
    for a in soup.find_all("a", href=True):
        m = re.search(r"fomcminutes(\d{8})\.htm", a["href"], re.IGNORECASE)
        if not m:
            continue
        try:
            date = datetime.strptime(m.group(1), "%Y%m%d").date()
        except ValueError:
            continue
        if start_year <= date.year <= end_year:
            href = a["href"]
            url  = BASE + href if href.startswith("/") else href
            found.append((str(date), url))
    return found

def discover_minutes_urls(start_year, end_year):
    """Three-pass discovery of all FOMC minutes URLs."""
    found = {}   # date_str -> url

    # Pass 1: main calendar page
    print("[fomc] Pass 1 — calendar page ...")
    r = _safe_get(CAL_URL)
    if r:
        links = _extract_links(
            BeautifulSoup(r.text, "html.parser"), start_year, end_year)
        for d, u in links:
            found[d] = u
        print(f"[fomc]   {len(found)} links found")
    time.sleep(1)

    # Pass 2: historical archive pages (2015-2020 only)
    hist_end = min(end_year, HIST_CUTOFF_YEAR)
    for year in range(max(start_year, 2015), hist_end + 1):
        r = _safe_get(HIST_URL.format(year=year))
        if r:
            links = _extract_links(
                BeautifulSoup(r.text, "html.parser"), start_year, end_year)
            new = [(d, u) for d, u in links if d not in found]
            for d, u in new:
                found[d] = u
            if new:
                print(f"[fomc]   historical {year}: +{len(new)}  total={len(found)}")
        time.sleep(0.5)

    # Pass 3: probe known dates not yet found
    print("[fomc] Pass 3 — probing known dates ...")
    added = 0
    for stamp in KNOWN_FOMC_DATES:
        try:
            date = datetime.strptime(stamp, "%Y%m%d").date()
        except ValueError:
            continue
        if not (start_year <= date.year <= end_year):
            continue
        date_str = str(date)
        if date_str in found:
            continue
        url = MIN_URL.format(date=stamp)
        r   = _safe_get(url)
        if r:
            found[date_str] = url
            print(f"[fomc]   Pass 3 found: {date_str}")
            added += 1
        time.sleep(0.5)
    print(f"[fomc]   Pass 3 added {added} dates")

    return sorted(found.items())

# Text extraction
def parse_minutes(html):
    """Extract main text body from the minutes HTML page."""
    soup    = BeautifulSoup(html, "html.parser")
    article = soup.find(id="article") or soup
    paras   = [p.get_text(" ", strip=True) for p in article.find_all("p")]
    return "\n\n".join(p for p in paras if len(p) > 20)

# S3 Upload
def upload(s3_client, bucket, date_str, url, text):
    doc_id = "fomc_" + date_str.replace("-", "")
    key    = f"raw/fomc/{date_str}/{doc_id}.json"
    s3_client.put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps({
            "source":      "fomc",
            "doc_id":      doc_id,
            "doc_date":    date_str,
            "url":         url,
            "text":        text,
            "n_chars":     len(text),
            "ingested_at": datetime.utcnow().isoformat(),
        }).encode("utf-8"),
        ContentType="application/json",
    )
    return key

# Main Functions for final executions
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket",     default=ENV_BUCKET,
                   help="S3 lake bucket (default: BUCKET from .env)")
    p.add_argument("--start-year", type=int, default=2015)
    p.add_argument("--end-year",   type=int, default=2025)
    p.add_argument("--sleep",      type=float, default=1.0,
                   help="seconds between page fetches")
    args = p.parse_args()

    if not args.bucket:
        p.error("--bucket is required, or set BUCKET in your .env file")

    s3_client = boto3.client("s3")
    urls      = discover_minutes_urls(args.start_year, args.end_year)
    print(f"\n[fomc] {len(urls)} unique minutes pages to download\n")

    uploaded = 0
    skipped  = 0
    for date_str, url in urls:
        r = _safe_get(url)
        if r is None:
            print(f"[skip] {date_str} — not published yet or not found")
            skipped += 1
            time.sleep(args.sleep)
            continue
        try:
            text = parse_minutes(r.text)
            if len(text) < 500:
                print(f"[skip] {date_str} — text too short ({len(text)} chars)")
                skipped += 1
                continue
            key = upload(s3_client, args.bucket, date_str, url, text)
            print(f"[ok]   {date_str}  {len(text):>8,} chars  →  {key}")
            uploaded += 1
        except Exception as e:
            print(f"[err]  {date_str}: {e}")
            skipped += 1
        time.sleep(args.sleep)

    print(f"\n[done] uploaded={uploaded}  skipped={skipped}")
    print(f"       s3://{args.bucket}/raw/fomc/")


if __name__ == "__main__":
    main()
