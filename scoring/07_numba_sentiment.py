"""
MACS 30113
Yuzhao (Jerry) Jin; CNETID: yuzhaojin1
Duo (Doris) Wang; CNETID: dorisduowang

This file is to score every document in the lake for hawkish/dovish forward-guidance
sentiment using a dictionary lookup that has been JIT-compiled with Numba
@jit(nopython=True, parallel=True), exactly the pattern from
2M_CPU_Parallelism.ipynb.

Utilizing course contents:
- Week 1: Numba JIT (@jit, nopython=True)
- Week 2: prange + parallel=True multithreading

"""

import argparse
import io
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import boto3
from numba import jit, prange
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
ENV_BUCKET = os.environ.get("BUCKET", "")


# ---------------------------------------------------------------------------
# Hawkish / dovish lexicons
# ---------------------------------------------------------------------------
HAWKISH = [
    "tighten", "tightening", "tightened", "tight",
    "restrictive", "restraint", "restrain",
    "inflation", "inflationary", "overheating", "overheated",
    "hike", "hikes", "raise", "raising", "raised",
    "elevate", "elevated", "above-target",
    "firm", "firmer", "robust",
    "vigilant", "vigilance", "anchored",
    "withdraw", "withdrawal", "taper", "tapering", "tapered",
    "hawkish",
]
DOVISH = [
    "accommodate", "accommodative", "accommodation",
    "ease", "easing", "eased", "easy",
    "cut", "cuts", "cutting", "lowered", "lowering", "lower",
    "stimulus", "stimulative", "stimulate",
    "support", "supportive", "supporting",
    "patient", "patiently", "patience",
    "subdued", "soft", "softening", "softened", "weak", "weaker", "weakness",
    "downside", "uncertain", "uncertainty",
    "dovish",
]

def _hash(tok: str) -> int:
    """Stable 64-bit hash; matches what Spark will use upstream."""
    return np.int64(hash(tok) & 0x7FFFFFFFFFFFFFFF)

def build_lexicon_arrays():
    pairs = ([(_hash(w), +1.0) for w in HAWKISH] +
             [(_hash(w), -1.0) for w in DOVISH])
    pairs.sort(key=lambda p: p[0])  # sort so we can binary-search inside JIT
    hashes = np.array([p[0] for p in pairs], dtype=np.int64)
    weights = np.array([p[1] for p in pairs], dtype=np.float32)
    return hashes, weights

# JIT kernels — the two functions below are the actual hot path.
@jit(nopython=True, cache=True)
def _bsearch(arr, x):
    """Numba-friendly binary search returning index or -1."""
    lo, hi = 0, arr.size
    while lo < hi:
        mid = (lo + hi) // 2
        v = arr[mid]
        if v < x:
            lo = mid + 1
        elif v > x:
            hi = mid
        else:
            return mid
    return -1

@jit(nopython=True, parallel=True, cache=True)
def score_documents(doc_starts, doc_lens, token_hashes, lex_hashes, lex_w):
    """
    Vectorised dictionary scoring.

    Parameters: 
    - doc_starts : int64[N] -- start index of each document inside token_hashes
    - doc_lens : int64[N] -- number of tokens per document
    - token_hashes : int64[T] -- flat array of all tokens hashed
    - lex_hashes : int64[L] -- sorted hawk/dove vocabulary hashes
    - lex_w : float32[L] -- +1.0 (hawk) or -1.0 (dove)

    Returns: 
    - net: float32[N] -- sum of weights per doc
    - hawk_n : int64[N] -- count of hawkish hits
    - dove_n : int64[N] -- count of dovish hits
    - tokens : int64[N] -- total tokens per doc (echoed for normalisation)
    """
    n = doc_starts.size
    net    = np.zeros(n, dtype=np.float32)
    hawk_n = np.zeros(n, dtype=np.int64)
    dove_n = np.zeros(n, dtype=np.int64)
    tokens = np.zeros(n, dtype=np.int64)

    for i in prange(n):
        s = doc_starts[i]
        L = doc_lens[i]
        h = 0
        d = 0
        ssum = np.float32(0.0)
        for j in range(L):
            idx = _bsearch(lex_hashes, token_hashes[s + j])
            if idx >= 0:
                w = lex_w[idx]
                ssum += w
                if w > 0:
                    h += 1
                else:
                    d += 1
        net[i] = ssum
        hawk_n[i] = h
        dove_n[i] = d
        tokens[i] = L
    return net, hawk_n, dove_n, tokens

# I/O glue: read the Spark-emitted tokens Parquet from s3://.../curated/tokens/
def load_tokens(bucket, prefix):
    """Load every Parquet shard under prefix and return one combined DataFrame.
    The Spark stage writes one row per document with columns:
        source, doc_id, doc_date, tokens (array<string>)
    """
    s3 = boto3.client("s3")
    pag = s3.get_paginator("list_objects_v2")
    dfs = []
    for page in pag.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith(".parquet"):
                continue
            buf = io.BytesIO(s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read())
            dfs.append(pd.read_parquet(buf))
    if not dfs:
        raise RuntimeError(f"no parquet under s3://{bucket}/{prefix}")
    return pd.concat(dfs, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default=ENV_BUCKET,
                   help="S3 lake bucket (default: from .env BUCKET)")
    p.add_argument("--tokens-prefix", default="curated/tokens/")
    p.add_argument("--out-prefix", default="processed/sentiment/")
    args = p.parse_args()

    print("[numba] loading tokens parquet from S3 ...")
    df = load_tokens(args.bucket, args.tokens_prefix)
    print(f"[numba] {len(df):,} documents loaded")

    # Flatten tokens into a single int64 array and a (start, len) index.
    starts, lens = [], []
    flat = []
    cursor = 0
    for toks in df["tokens"]:
        starts.append(cursor)
        lens.append(len(toks))
        flat.extend(_hash(t) for t in toks)
        cursor += len(toks)
    token_hashes = np.array(flat, dtype=np.int64)
    doc_starts = np.array(starts, dtype=np.int64)
    doc_lens   = np.array(lens, dtype=np.int64)
    lex_h, lex_w = build_lexicon_arrays()

    print(f"[numba] scoring {token_hashes.size:,} tokens across "
          f"{doc_starts.size:,} docs ...")
    t0 = time.time()
    net, hawk_n, dove_n, tokens = score_documents(
        doc_starts, doc_lens, token_hashes, lex_h, lex_w)
    dt = time.time() - t0
    rate = token_hashes.size / dt
    print(f"[numba] done in {dt:.2f}s  ({rate/1e6:.1f} M tokens/s)")

    df["net_score"] = net
    df["hawk_count"] = hawk_n
    df["dove_count"] = dove_n
    df["n_tokens"] = tokens
    df["net_per_1k"] = (net / tokens.clip(min=1)) * 1000.0

    out = io.BytesIO()
    df.drop(columns=["tokens"]).to_parquet(out, index=False)
    out.seek(0)
    s3 = boto3.client("s3")
    out_key = args.out_prefix.rstrip("/") + "/sentiment.parquet"
    s3.put_object(Bucket=args.bucket, Key=out_key, Body=out.read())
    print(f"[done] s3://{args.bucket}/{out_key}")

if __name__ == "__main__":
    main()