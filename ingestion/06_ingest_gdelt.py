"""
MACS 30113
Yuzhao (Jerry) Jin; CNETID: yuzhaojin1
Duo (Doris) Wang; CNETID: dorisduowang

This file is for streaming GDELT 2.0 GKG v2.1 records for FOMC-meeting windows from
the GDELT S3 bucket, filter to Fed/economy-relevant US English articles,and write 
JSONL shards to s3://<bucket>/raw/gdelt/.

Utilizing course contents: 
- Week 2: ThreadPoolExecutor for parallel I/O
- Week 4-5: Boto3 S3 (requester-pays bucket with signed credentials)
- Week 6: catalog Lambda triggered by every S3 PUT
AI use: none.
"""

import argparse
import csv
import io
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

# GDELT GKG rows can contain fields larger than Python csv's default 128 KB.
csv.field_size_limit(10 * 1024 * 1024)   # raise limit to 10 MB

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
ENV_BUCKET = os.environ.get("BUCKET", "")

# GDELT S3 bucket 
# Per the AWS Open Data Registry (https://registry.opendata.aws/gdelt/):
#   Access: aws s3 ls --no-sign-request s3://gdelt-open-data/
#   ARN:    arn:aws:s3:::gdelt-open-data
#   Region: us-east-1

GDELT_BUCKET     = "gdelt-open-data"
GDELT_GKG_PREFIX = "v2/gkg/"

S3_GDELT = boto3.client(
    "s3",
    region_name="us-east-1",
    config=Config(signature_version=UNSIGNED),   # --no-sign-request per registry
)

# US domain filter
US_DOMAIN_KEYWORDS = {
    # Major US newspapers
    "nytimes", "wsj", "washingtonpost", "usatoday", "latimes",
    "chicagotribune", "nypost", "nydailynews", "bostonglobe",
    # Financial / business news
    "bloomberg", "reuters", "cnbc", "marketwatch", "barrons",
    "ft", "seekingalpha", "thestreet", "benzinga", "zerohedge",
    "investopedia", "kiplinger", "fool", "money", "fortune",
    "businessinsider", "forbes", "inc",
    # Broadcast / wire services
    "cnn", "foxnews", "foxbusiness", "nbcnews", "abcnews",
    "cbsnews", "npr", "pbs", "apnews", "ap",
    # Online / political
    "axios", "politico", "theatlantic", "newyorker", "slate",
    "vox", "motherjones", "thedailybeast", "huffpost",
    # Fed-specific
    "federalreserve",
}

def is_us_domain(domain):
    """Return True if domain matches any US news/financial outlet keyword."""
    if not domain:
        return False
    d = domain.lower()
    return any(kw in d for kw in US_DOMAIN_KEYWORDS)

# Federal Reserve / economy relevance filters
# Signal sources (any one is sufficient to pass the filter):
#   1. V1THEMES contains a verified GDELT economy/Fed theme code
#   2. V1ORGANIZATIONS contains a Fed/financial institution name
#   3. V1PERSONS contains a Fed-related person
#   4. V2DOCUMENTIDENTIFIER URL contains a Fed-related keyword
#   5. V1COUNTS contains a CAMEO economic event code (02xx-06xx)

# Verified real GDELT GKG theme codes from LOOKUP-GKGTHEMES.TXT
FED_THEME_CODES = {
    # EPU taxonomy (Economic Policy Uncertainty)
    "EPU_ECONOMY",
    "EPU_ECONOMY_HISTORIC",
    "EPU_POLICY",
    "EPU_POLICY_BUDGET",
    "EPU_POLICY_TAX",
    "EPU_POLICY_GOVERNMENT",
    "EPU_POLICY_POLITICAL",
    "EPU_POLICY_CONGRESS",       # Congressional action (CAMEO: 04xx Consult)
    "EPU_CATS_REGULATION",
    "EPU_CATS_TAXES",
    # ECON taxonomy
    "ECON_STOCKMARKET",
    "ECON_WORLDCURRENCIES",      # exchange rates
    "ECON_TAXATION",
    "ECON_BANKRUPTCY",
    "ECON_FREETRADE",            # tariffs/trade (CAMEO: 05xx diplomatic coop)
    "ECON_ELECTRICALGENERATION", # energy prices (Fed inflation input)
    # TAX economic
    "TAX_ECON_PRICE",            # price levels / inflation signals
    # ENV oil (supply shocks relevant to Fed)
    "ENV_OIL",
    # World Bank taxonomy
    "WB_1920_FINANCIAL_SECTOR_DEVELOPMENT",
    "WB_1104_MACROECONOMIC_VULNERABILITY_AND_DEBT",
    "WB_855_LABOR_MARKETS",      # employment — Fed dual mandate
    "WB_405_BUSINESS_CLIMATE",
    "WB_2530_BUSINESS_ENVIRONMENT",
    "WB_698_TRADE",
    "WB_337_INSURANCE",
    "WB_346_COMPETITIVE_INDUSTRIES",
    "WB_695_POVERTY",            # inequality — Fed equity mandate
    "WB_2936_GOLD",              # gold (monetary policy signal)
    "WB_1973_FINANCIAL_RISK_REDUCTION",
    # General
    "GENERAL_GOVERNMENT",
    "LEGISLATION",
    "ELECTION",
}

# CAMEO economic event root codes (from CAMEO_eventcodes.txt)
# These appear as CountType values in the GKG V1COUNTS field.
# Root codes 02-06 cover economic cooperation, appeals, and consultations:
#   02xx = APPEAL (0211 econ coop, 0231 econ aid, 0254 ease sanctions)
#   03xx = EXPRESS INTENT (0311 econ coop, 0354 ease sanctions)
#   04xx = CONSULT (economic policy consultations)
#   05xx = DIPLOMATIC COOPERATION (economic diplomacy)
#   06xx = MATERIAL COOPERATION (economic aid/cooperation)
CAMEO_ECONOMIC_ROOT_CODES = {"02", "03", "04", "05", "06"}

# Organization keywords — expanded with CAMEO_knowngroup.txt institutions
FED_ORG_KEYWORDS = (
    # Federal Reserve direct
    "federal reserve",
    "federal open market committee",
    "fomc",
    "federal reserve bank",
    "board of governors",
    # Fed chairs / vice chairs
    "jerome powell", "powell",
    "janet yellen", "yellen",
    "ben bernanke", "bernanke",
    "alan greenspan", "greenspan",
    "lael brainard",
    "richard clarida",
    # US financial regulators
    "treasury department",
    "department of the treasury",
    "u.s. treasury",
    "securities and exchange commission",
    "sec ",
    "commodity futures trading commission",
    "cftc",
    "fdic",
    "federal deposit insurance",
    "office of the comptroller",
    "consumer financial protection bureau",
    "cfpb",
    # International financial institutions (CAMEO_knowngroup.txt)
    "bank for international settlements",  # BIS
    "international monetary fund", "imf",  # IMF
    "world bank",                          # WBK
    "world trade organization",            # WTO
    "world economic forum",                # WEF
    "group of seven", "g-7", "g7",        # GOS
    "group of eight", "g-8", "g8",        # GOE
    "g20",
    "paris club",                          # PRC (creditor consortium)
    "organization of petroleum exporting countries", "opec",  # OPC
    "european central bank", "ecb",
    "european bank for reconstruction",    # EBR
    # Major US banks (Fed oversight, systemic importance)
    "jpmorgan", "goldman sachs", "morgan stanley",
    "bank of america", "wells fargo", "citigroup", "citibank",
    "blackrock", "vanguard",
    # US Congress committees
    "senate banking committee",
    "house financial services",
    "congressional budget office", "cbo",
    # Rating agencies (financial system signals)
    "moody", "standard and poor", "s&p", "fitch",
)

# Person keywords
FED_PERSON_KEYWORDS = (
    "powell", "yellen", "bernanke", "greenspan",
    "brainard", "clarida", "waller", "williams",
    "draghi", "lagarde",                    # ECB (global monetary spillovers)
    "mnuchin", "bessent",                   # US Treasury Secretaries
    "jamie dimon",
)


def is_fed_relevant(themes_str, orgs_str, persons_str, url, counts_str):
    """
    Return True if the article is relevant to the Federal Reserve,
    US financial system, or macroeconomy.  Any one signal is sufficient.

    Five signal sources:
      1. V1THEMES — verified GDELT theme codes
      2. V1ORGANIZATIONS — Fed/financial institution names
      3. V1PERSONS — Fed-related persons
      4. URL — Fed-related path keywords
      5. V1COUNTS — CAMEO economic event codes (02xx-06xx from
                    CAMEO_eventcodes.txt)
    """
    # Signal 1: GKG themes
    if themes_str:
        for entry in themes_str.split(";"):
            code = entry.strip().split(",")[0].upper()
            if code in FED_THEME_CODES:
                return True

    # Signal 2: organizations (CAMEO_knowngroup.txt institutions)
    if orgs_str:
        orgs_l = orgs_str.lower()
        for kw in FED_ORG_KEYWORDS:
            if kw in orgs_l:
                return True

    # Signal 3: persons
    if persons_str:
        pers_l = persons_str.lower()
        for kw in FED_PERSON_KEYWORDS:
            if kw in pers_l:
                return True

    # Signal 4: URL path keywords
    if url:
        url_l = url.lower()
        for kw in ("federal-reserve", "fomc", "federal_reserve",
                   "federalreserve", "interest-rate", "interest_rate",
                   "monetary-policy", "monetary_policy", "rate-hike",
                   "rate-cut", "quantitative-easing", "inflation"):
            if kw in url_l:
                return True

    # Signal 5: CAMEO economic event codes in V1COUNTS
    # V1COUNTS format: CountType#Count#ObjectType#...;CountType#Count#...
    # CAMEO_eventcodes.txt root codes 02-06 = economic cooperation/appeals
    if counts_str:
        for block in counts_str.split(";"):
            parts = block.split("#")
            if parts:
                ct = parts[0].strip()
                if ct[:2] in CAMEO_ECONOMIC_ROOT_CODES:
                    return True

    return False

# Tone, theme, org, person, location parsers
def parse_tone(tone_str):
    """Parse V1.5TONE into 7-component dict (from GKG codebook)."""
    if not tone_str:
        return {}
    parts = tone_str.split(",")
    try:
        return {
            "tone":                   float(parts[0]) if len(parts) > 0 else None,
            "positive_score":         float(parts[1]) if len(parts) > 1 else None,
            "negative_score":         float(parts[2]) if len(parts) > 2 else None,
            "polarity":               float(parts[3]) if len(parts) > 3 else None,
            "activity_ref_density":   float(parts[4]) if len(parts) > 4 else None,
            "self_group_ref_density": float(parts[5]) if len(parts) > 5 else None,
            "word_count":    int(float(parts[6])) if len(parts) > 6 else None,
        }
    except (ValueError, IndexError):
        return {"raw_tone": tone_str}

def parse_themes(themes_str):
    if not themes_str:
        return []
    return [e.strip().split(",")[0] for e in themes_str.split(";") if e.strip()]

def parse_orgs(orgs_str):
    if not orgs_str:
        return []
    return [e.strip().split(",")[0] for e in orgs_str.split(";") if e.strip()]

def parse_persons(persons_str):
    if not persons_str:
        return []
    return [e.strip().split(",")[0] for e in persons_str.split(";") if e.strip()]

def parse_locations(locations_str):
    """Parse V1LOCATIONS — semicolon blocks, pound-delimited fields."""
    if not locations_str:
        return []
    locs = []
    for block in locations_str.split(";"):
        parts = block.split("#")
        if len(parts) >= 3:
            locs.append({
                "type":         parts[0].strip(),
                "full_name":    parts[1].strip(),
                "country_code": parts[2].strip(),
                "adm1_code":    parts[3].strip() if len(parts) > 3 else "",
            })
    return locs

# S3 key discovery — list actual GDELT files for each date in the window

def discover_gdelt_keys_for_date(date_obj):
    """
    List actual GKG zip files that exist in the GDELT bucket for one date.
    Uses RequestPayer='requester' because gdelt-open-data is requester-pays.
    """
    prefix = GDELT_GKG_PREFIX + date_obj.strftime("%Y%m%d")
    keys = []
    try:
        pag = S3_GDELT.get_paginator("list_objects_v2")
        for page in pag.paginate(
            Bucket=GDELT_BUCKET,
            Prefix=prefix,
            PaginationConfig={"PageSize": 200},
        ):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if k.endswith(".gkg.csv"):   # files are plain CSV, not zipped
                    keys.append(k)
    except Exception as e:
        if not getattr(discover_gdelt_keys_for_date, "_err_logged", False):
            discover_gdelt_keys_for_date._err_logged = True
            print(f"\n[debug] S3 list error for {prefix}: {e}\n")
    return keys

def discover_all_keys(fomc_dates, pad_days):
    """Collect actual GDELT S3 keys for all FOMC meeting windows."""
    all_keys = set()
    dates_to_list = set()
    for d in fomc_dates:
        for offset in range(-pad_days, pad_days + 1):
            dates_to_list.add(d + timedelta(days=offset))

    print(f"[gdelt] Listing GDELT files for {len(dates_to_list)} calendar days ...")
    for i, date_obj in enumerate(sorted(dates_to_list)):
        all_keys.update(discover_gdelt_keys_for_date(date_obj))
        if (i + 1) % 100 == 0:
            print(f"[gdelt]   {i+1}/{len(dates_to_list)} days  keys: {len(all_keys):,}")

    return sorted(all_keys)

# Shard fetch and filter
def fetch_and_filter_shard(src_key):
    """Download one GKG CSV, apply all pipeline filters, return records.
    Retries up to 5 times with 30-second backoff to survive brief internet
    disconnects (ConnectionError, ReadTimeoutError, etc.)."""
    import time as _time

    MAX_RETRIES = 5
    RETRY_WAIT  = 30   # seconds to wait between retries

    body = None
    for attempt in range(MAX_RETRIES):
        try:
            body = S3_GDELT.get_object(
                Bucket=GDELT_BUCKET,
                Key=src_key,
            )["Body"].read()
            break   # success — exit retry loop
        except Exception as e:
            err = str(e)
            if "NoSuchKey" in err or "404" in err:
                return []   # file genuinely absent — skip, don't retry
            # Network or timeout error — wait and retry
            if attempt < MAX_RETRIES - 1:
                if not getattr(fetch_and_filter_shard, "_retry_logged", False):
                    fetch_and_filter_shard._retry_logged = True
                    print(f"\n[retry] network error: {err[:80]}"
                          f" — waiting {RETRY_WAIT}s then retrying ...")
                _time.sleep(RETRY_WAIT)
            else:
                if not getattr(fetch_and_filter_shard, "_dl_err", False):
                    fetch_and_filter_shard._dl_err = True
                    print(f"\n[debug] Download failed after {MAX_RETRIES} "
                          f"attempts on {src_key}: {err}\n")
                return []

    if body is None:
        return []

    # Files are plain uncompressed CSV (not zip):
    raw = body.decode("latin-1", errors="ignore")

    # One-time diagnostic: structure of first successful shard
    if not getattr(fetch_and_filter_shard, "_raw_ok", False):
        fetch_and_filter_shard._raw_ok = True
        lines = [l for l in raw.split("\n") if l.strip()]
        print(f"\n[debug] === First shard: {src_key} ===")
        print(f"[debug] Lines: {len(lines)}")
        if lines:
            f0 = lines[0].split("\t")
            print(f"[debug] Columns: {len(f0)}")
            print(f"[debug] col[0] GKGID    : {f0[0][:60]!r}")
            print(f"[debug] col[3] domain   : {f0[3][:50] if len(f0)>3 else 'MISSING'!r}")
            print(f"[debug] col[5] counts   : {f0[5][:80] if len(f0)>5 else 'MISSING'!r}")
            print(f"[debug] col[7] themes   : {f0[7][:100] if len(f0)>7 else 'MISSING'!r}")
            print(f"[debug] col[13] orgs    : {f0[13][:100] if len(f0)>13 else 'MISSING'!r}")
            wide = sum(1 for l in lines if len(l.split("\t")) >= 16)
            print(f"[debug] Rows >=16 cols  : {wide}/{len(lines)}")
            domains = sorted({l.split("\t")[3] for l in lines[:300]
                              if len(l.split("\t")) >= 4})[:10]
            print(f"[debug] Sample domains  : {domains}")
            us_ct = sum(1 for l in lines[:300] if len(l.split("\t")) >= 4
                        and is_us_domain(l.split("\t")[3]))
            print(f"[debug] US domain hits  : {us_ct}/300")
        print()

    records = []
    for row in csv.reader(io.StringIO(raw), delimiter="\t"):
        if len(row) < 16:
            continue

        # English only: "-T" in GKGRECORDID = GDELT Translingual
        if "-T" in row[0]:
            continue

        # US domain filter
        if not is_us_domain(row[3].strip()):
            continue

        themes_str  = row[7].strip()
        orgs_str    = row[13].strip()
        persons_str = row[11].strip()
        url         = row[4].strip()
        counts_str  = row[5].strip()   # V1COUNTS — CAMEO event codes

        # Fed/economy relevance (5 signals)
        if not is_fed_relevant(themes_str, orgs_str, persons_str,
                               url, counts_str):
            continue

        try:
            doc_date = datetime.strptime(row[1][:8], "%Y%m%d").date().isoformat()
        except (ValueError, IndexError):
            continue

        themes    = parse_themes(themes_str)
        orgs      = parse_orgs(orgs_str)
        persons   = parse_persons(persons_str)
        locations = parse_locations(row[9].strip())
        tone      = parse_tone(row[15].strip())

        text = " ".join(filter(None, [
            " ".join(themes),
            " ".join(orgs),
            " ".join(persons),
        ]))

        records.append({
            "source":        "gdelt",
            "doc_id":        row[0],
            "doc_date":      doc_date,
            "source_domain": row[3].strip(),
            "url":           url,
            "themes":        themes,
            "organizations": orgs,
            "persons":       persons,
            "locations":     locations[:20],
            "tone":          tone,
            "text":          text,
        })

    return records

# FOMC date discovery
def load_fomc_dates(s3_signed, bucket):
    dates = []
    pag = s3_signed.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=bucket, Prefix="raw/fomc/"):
        for obj in page.get("Contents", []):
            parts = obj["Key"].split("/")
            if len(parts) >= 4:
                try:
                    dates.append(datetime.strptime(parts[2], "%Y-%m-%d").date())
                except ValueError:
                    pass
    return sorted(set(dates))

# S3 upload to our lake bucket
def upload_batch(s3_signed, bucket, stamp, records):
    if not records:
        return
    body      = "\n".join(json.dumps(r) for r in records).encode("utf-8")
    date_part = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
    key       = f"raw/gdelt/{date_part}/gdelt_{stamp}.jsonl"
    s3_signed.put_object(
        Bucket=bucket, Key=key, Body=body,
        ContentType="application/x-jsonlines",
    )

# Main Function for Executions
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket",   default=ENV_BUCKET)
    p.add_argument("--pad-days", type=int, default=3)
    p.add_argument("--workers",  type=int, default=4)
    args = p.parse_args()

    if not args.bucket:
        p.error("--bucket required, or set BUCKET in .env")

    s3_signed = boto3.client("s3")

    print("[gdelt] Pipeline filters (5 signal sources):")
    print(f"         1. Themes   : {len(FED_THEME_CODES)} verified GDELT codes "
          f"(EPU_, ECON_, WB_, ENV_OIL, ...)")
    print(f"         2. Orgs     : {len(FED_ORG_KEYWORDS)} keywords incl. CAMEO "
          f"knowngroups (BIS,IMF,WBK,WEF,G7,...)")
    print(f"         3. Persons  : {len(FED_PERSON_KEYWORDS)} Fed-related "
          f"(Powell,Yellen,Bernanke,...)")
    print(f"         4. URL      : Federal Reserve / rate / inflation path keywords")
    print(f"         5. V1COUNTS : CAMEO economic event root codes 02-06 "
          f"(from CAMEO_eventcodes.txt)")
    print(f"         Domain   : US news/financial outlets (keyword match)")
    print(f"         English  : GKGRECORDID without '-T' (no Translingual)")
    print(f"         S3 access: UNSIGNED anonymous (--no-sign-request per AWS Open Data Registry)")

    fomc_dates = load_fomc_dates(s3_signed, args.bucket)
    if not fomc_dates:
        print("[gdelt] No FOMC dates found — run 03_ingest_fomc.py first")
        return
    print(f"[gdelt] Anchoring on {len(fomc_dates)} FOMC meetings  pad_days={args.pad_days}")

    # ── Startup diagnostic: verify GDELT bucket access ─────────────────────
    print("\n[gdelt] Running bucket access diagnostic ...")
    print("[gdelt] Registry: aws s3 ls --no-sign-request s3://gdelt-open-data/")
    try:
        resp = S3_GDELT.list_objects_v2(
            Bucket=GDELT_BUCKET,
            MaxKeys=5,
            Delimiter="/",
        )
        prefixes = [p["Prefix"] for p in resp.get("CommonPrefixes", [])]
        top = [o["Key"] for o in resp.get("Contents", [])]
        print(f"[debug] Top-level prefixes  : {prefixes}")
        print(f"[debug] Top-level objects   : {top}")
    except Exception as e:
        print(f"[debug] Top-level list error: {e}")

    try:
        resp2 = S3_GDELT.list_objects_v2(
            Bucket=GDELT_BUCKET,
            Prefix="v2/gkg/",
            MaxKeys=3,
        )
        gkg_keys = [o["Key"] for o in resp2.get("Contents", [])]
        print(f"[debug] v2/gkg/ first keys  : {gkg_keys}")
        if gkg_keys:
            print(f"[debug] ACCESS CONFIRMED — UNSIGNED works ✓")
        else:
            print(f"[debug] v2/gkg/ returned empty — bucket may be restructured")
    except Exception as e:
        print(f"[debug] v2/gkg/ list error  : {e}")
    print()

    keys = discover_all_keys(fomc_dates, args.pad_days)
    if not keys:
        print("[gdelt] No GDELT files found. Run this CLI check first:")
        print("        aws s3 ls --no-sign-request s3://gdelt-open-data/v2/gkg/ | head -5")
        print("        If empty: aws s3 ls --no-sign-request s3://gdelt-open-data/")
        return
    print(f"[gdelt] {len(keys):,} real GDELT shards  workers={args.workers}")

    # ── Resume logic: skip shards already uploaded to S3 ───────────────────
    # On restart, list what is already in raw/gdelt/ and skip those stamps.
    # This means a restart only downloads the remaining shards, not everything.
    print("[gdelt] Checking for already-uploaded shards (resume support) ...")
    done_stamps = set()
    try:
        pag = s3_signed.get_paginator("list_objects_v2")
        for page in pag.paginate(Bucket=args.bucket, Prefix="raw/gdelt/"):
            for obj in page.get("Contents", []):
                # key format: raw/gdelt/YYYY-MM-DD/gdelt_YYYYMMDDHHMMSS.jsonl
                m = re.search(r"gdelt_(\d{14})\.jsonl", obj["Key"])
                if m:
                    done_stamps.add(m.group(1))
    except Exception as e:
        print(f"[gdelt] Could not check resume state: {e}")

    if done_stamps:
        before = len(keys)
        keys = [k for k in keys
                if not any(s in k for s in done_stamps)]
        print(f"[gdelt] Resuming: {len(done_stamps):,} shards already done  "
              f"→ {len(keys):,} remaining (skipping {before - len(keys):,})")
    else:
        print(f"[gdelt] Fresh start — processing all {len(keys):,} shards")

    if not keys:
        print("[done] All shards already uploaded — nothing to do.")
        return

    total = 0
    done  = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut_to_key = {ex.submit(fetch_and_filter_shard, k): k for k in keys}
        for fut in as_completed(fut_to_key):
            key  = fut_to_key[fut]
            done += 1
            try:
                records = fut.result()
                if records:
                    m     = re.search(r"(\d{14})", key)
                    stamp = m.group(1) if m else key[-26:-12]
                    upload_batch(s3_signed, args.bucket, stamp, records)
                    total += len(records)
            except Exception as e:
                print(f"[warn] {key}: {e}")
            if done % 100 == 0:
                print(f"[gdelt] {done:,}/{len(keys):,} done  "
                      f"Fed-relevant US records this run: {total:,}")

    print(f"\n[done] {total:,} Fed-relevant US GDELT records")
    print(f"       s3://{args.bucket}/raw/gdelt/")


if __name__ == "__main__":
    main()
