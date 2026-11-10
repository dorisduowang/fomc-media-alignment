"""
MACS 30113
Yuzhao (Jerry) Jin; CNETID: yuzhaojin1
Duo (Doris) Wang; CNETID: dorisduowang

This file is intended to be used as Lambda worker that fetches one (year, month) 
batch of NYT articles about the Federal Reserve and uploads JSON files to S3.
However it failed due to the "04_setup_stepfuctions_nyt.py" being requesting too
frequently as specified in the file. 

Fixes based from NYT-API-articlesearch-product.yaml:
  - sort="newest" (from notebook)
  - end_date = first day of next month (exclusive, from notebook)
  - User-Agent header (from notebook)
  - 429 rate-limit handling with backoff
  - Better per-page error logging in return value

Vocareum compliance: no changes to Lambda concurrency (stays at 4).

Utilizing course contents:
- Week 4: Lambda handler from event_driven.ipynb
- Week 6: Step Functions Map from step_functions.ipynb
"""

import json
import os
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime
import boto3

s3     = boto3.client("s3")
BUCKET = os.environ["BUCKET"]

# API configuration
BASE_API_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
SEARCH_QUERY = "Federal Reserve"   # q param — searches body, headline, byline
SORT        = "newest"   # from notebook: SORT = "newest"
SLEEP_TIME  = 12         # from notebook: SLEEP_TIME = 12 (5 req/min rate limit)
MAX_PAGES   = 10         # max pages per Lambda (10 × 10 = 100 articles/month)
TIMEOUT     = 30         # increased from 10 to 30 for Lambda network latency

# Setup session header: 
# SESSION.headers.update({"User-Agent": "nyt-articlesearch-scraper/1.0"})
HEADERS = {"User-Agent": "nyt-articlesearch-scraper/1.0"}


def _api_call(year, month, page, nyt_key):
    """
    Call the NYT Article Search API for one month-page.
    Mirrors notebook's fetch_articles_page() exactly.

    end_date = first day of NEXT month (exclusive) — from notebook:
        START_DATE = "20210101"
        END_DATE   = "20220101"  # exclusive upper bound
    """
    if month == 12:
        next_year, next_month = year + 1, 1
    else:
        next_year, next_month = year, month + 1

    begin_date = f"{year:04d}{month:02d}01"
    end_date   = f"{next_year:04d}{next_month:02d}01"

    params = {
        "q":          SEARCH_QUERY,   # "Federal Reserve" across all sections
        "begin_date": begin_date,
        "end_date":   end_date,
        "sort":       SORT,
        "page":       page,
        "api-key":    nyt_key,
        # NO fq filter — captures all sections
    }

    url = BASE_API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def extract_article_data(article):
    """
    Extract fields — directly from notebook's extract_article_data().
    """
    return {
        "abstract":        article.get("abstract", ""),
        "byline":          (article.get("byline") or {}).get("original", ""),
        "document_type":   article.get("document_type", ""),
        "title":           (article.get("headline") or {}).get("main", ""),
        "print_headline":  (article.get("headline") or {}).get("print_headline", ""),
        "keywords":        ", ".join([kw.get("value", "") for kw in article.get("keywords", [])]),
        "news_desk":       article.get("news_desk", ""),
        "pub_date":        article.get("pub_date", ""),
        "section_name":    article.get("section_name", ""),
        "snippet":         article.get("snippet", ""),
        "source":          article.get("source", ""),
        "subsection_name": article.get("subsection_name", ""),
        "type_of_material":article.get("type_of_material", ""),
        "uri":             article.get("uri", ""),
        "web_url":         article.get("web_url", ""),
        "word_count":      article.get("word_count", 0),
    }


def lambda_handler(event, context):
    year    = int(event["year"])
    month   = int(event["month"])
    nyt_key = event["nyt_api_key"]

    written  = 0
    errors   = []

    for page in range(0, MAX_PAGES):
        try:
            resp = _api_call(year, month, page, nyt_key)

        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Rate limited — back off and retry once
                errors.append(f"p{page}: 429 rate-limit, backing off 30s")
                time.sleep(30)
                try:
                    resp = _api_call(year, month, page, nyt_key)
                except Exception as e2:
                    errors.append(f"p{page}: retry failed: {e2}")
                    break
            else:
                errors.append(f"p{page}: HTTP {e.code} {e.reason}")
                break

        except Exception as e:
            errors.append(f"p{page}: {type(e).__name__}: {e}")
            break

        docs = resp.get("response", {}).get("docs", [])
        hits = resp.get("response", {}).get("meta", {}).get("hits", 0)

        if not docs:
            break

        for article in docs:
            data   = extract_article_data(article)
            pub    = data["pub_date"][:10] if data["pub_date"] else f"{year:04d}-{month:02d}-01"
            raw_id = article.get("uri", article.get("_id", f"{year}{month:02d}_{page}_{written}"))
            doc_id = "nyt_" + raw_id.split("/")[-1]
            key    = f"raw/nyt/{pub}/{doc_id}.json"

            s3.put_object(
                Bucket=BUCKET, Key=key,
                Body=json.dumps({
                    "source":      "nyt",
                    "doc_id":      doc_id,
                    "doc_date":    pub,
                    "url":         data["web_url"],
                    "headline":    data["title"],
                    "abstract":    data["abstract"],
                    "snippet":     data["snippet"],
                    "keywords":    data["keywords"],
                    "news_desk":   data["news_desk"],
                    "section":     data["section_name"],
                    "word_count":  data["word_count"],
                    "text": " ".join(filter(None, [
                        data["title"],
                        data["abstract"],
                        data["snippet"],
                    ])),
                }).encode("utf-8"),
                ContentType="application/json",
            )
            written += 1

        # Check if we have reached the last page
        total_pages = min((hits + 9) // 10, MAX_PAGES)
        if page >= total_pages - 1:
            break

        # Rate limit: 5 requests/minute = 12s sleep (from notebook)
        time.sleep(SLEEP_TIME)

    result = {"year": year, "month": month, "written": written}
    if errors:
        result["errors"] = errors
    return result