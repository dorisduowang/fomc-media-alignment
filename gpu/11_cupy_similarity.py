"""
MACS 30113
Yuzhao (Jerry) Jin; CNETID: yuzhaojin1
Duo (Doris) Wang; CNETID: dorisduowang
This file is for computing cosine similarity between every FOMC meeting TF-IDF vector
and the mean TF-IDF vector of US media at each lag k in {-7..+7} days.

Notes on AWS account constraints from Vocareum README:
  Supported EC2 instance types: nano, micro, small, medium, large ONLY.
  The smallest GPU instance on AWS is g4dn.xlarge (an "xlarge" type) which
  is NOT supported and will be terminated immediately if launched.
  Therefore, this script runs on CPU using NumPy by default on AWS Academy.

  A CuPy GPU path is included for reference and will activate automatically
  if CuPy is importable. On AWS, the script falls
  back to NumPy gracefully with no code changes needed.

  The --benchmark flag times both CPU (NumPy) execution paths when GPU
  is unavailable, or CPU vs GPU when CuPy is available.

Utilizing course contents:
- Week 3: CuPy GPU arrays, .get() back to host; float32 for memory efficiency; 
          cupyx.profiler.benchmark for timing
"""

import argparse
import io
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import boto3
from dotenv import load_dotenv

# GPU availability check — graceful fallback to NumPy on AWS Academy
try:
    import cupy as cp
    from cupyx.profiler import benchmark as cupy_benchmark
    # Must trigger actual kernel compilation (nvrtc), not just device init.
    # cp.zeros() and getDeviceCount() succeed even without nvrtc.dll.
    # An elementwise comparison forces nvrtc to compile a CUDA kernel —
    # this is exactly the operation that fails if nvrtc*.dll is missing.
    _test = cp.array([1.0, 0.0], dtype=cp.float32)
    _dummy = (_test == 0).sum()   # triggers nvrtc kernel compilation
    del _test, _dummy
    GPU_AVAILABLE = True
    print("[device] CuPy detected — GPU path enabled")
except Exception:
    GPU_AVAILABLE = False
    print("[device] CuPy not available — running on CPU (NumPy)")
    print("[device] Note: AWS Academy only supports instance types up to 'large'.")
    print("[device]       GPU instances (g4dn.xlarge) are 'xlarge' and not supported.")

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
ENV_BUCKET = os.environ.get("BUCKET", "")

# S3 loading helpers
def _s3_parquets(bucket, prefix):
    """Read flat (non-partitioned) Parquet files from S3."""
    s3 = boto3.client("s3")
    pag = s3.get_paginator("list_objects_v2")
    frames = []
    for page in pag.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                frames.append(pd.read_parquet(io.BytesIO(body)))
    if not frames:
        raise RuntimeError(f"no parquet under s3://{bucket}/{prefix}")
    return pd.concat(frames, ignore_index=True)

def _s3_parquets_partitioned(bucket, prefix):
    """
    Read Hive-partitioned Parquet written by Spark's .partitionBy().

    Spark encodes partition columns (meeting_date, lag_days) in the directory
    path rather than inside the parquet files:
        curated/meeting_windows/meeting_date=2015-01-28/lag_days=-7/part-XXXX.parquet

    """
    s3 = boto3.client("s3")
    pag = s3.get_paginator("list_objects_v2")
    frames = []
    for page in pag.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue
            # Extract partition values from path segments
            partition_vals = {}
            for segment in key.split("/"):
                if "=" in segment:
                    k, v = segment.split("=", 1)
                    partition_vals[k] = v
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            df   = pd.read_parquet(io.BytesIO(body))
            # Inject partition columns
            for col, val in partition_vals.items():
                df[col] = val
            if "lag_days" in df.columns:
                df["lag_days"] = df["lag_days"].astype(int)
            frames.append(df)
    if not frames:
        raise RuntimeError(f"no parquet under s3://{bucket}/{prefix}")
    return pd.concat(frames, ignore_index=True)


def vec_to_dense(v, dim=1 << 18):
    if isinstance(v, dict):
        out = np.zeros(dim, dtype=np.float32)
        idx = np.asarray(v.get("indices", []), dtype=np.int64)
        val = np.asarray(v.get("values", []),  dtype=np.float32)
        out[idx] = val
        return out
    arr = np.asarray(v, dtype=np.float32)
    if arr.size == dim:
        return arr
    raise TypeError(f"unrecognised vec shape {arr.shape}")

# Similarity kernels — GPU (CuPy) and CPU (NumPy) versions
def _normalize(X):
    """L2-normalise rows. Works on both NumPy and CuPy arrays."""
    xp    = cp if GPU_AVAILABLE and isinstance(X, cp.ndarray) else np
    norms = xp.linalg.norm(X, axis=1, keepdims=True)
    norms = xp.where(norms == 0, xp.float32(1.0), norms)
    return X / norms


def cosine_similarities_gpu(fomc_gpu, win_gpu, valid_fomc_idx):
    """All-pairs cosine using CuPy — O(M*V) batched matmul on GPU."""
    fomc_n   = _normalize(fomc_gpu)
    win_n    = _normalize(win_gpu)
    valid_f  = fomc_n[valid_fomc_idx]
    sims     = cp.sum(valid_f * win_n, axis=1)
    return cp.asnumpy(sims)


def cosine_similarities_cpu(fomc_cpu, win_cpu, valid_fomc_idx):
    """Row-wise cosine using NumPy — runs on any machine."""
    fomc_n  = _normalize(fomc_cpu)
    win_n   = _normalize(win_cpu)
    valid_f = fomc_n[valid_fomc_idx]
    sims    = np.sum(valid_f * win_n, axis=1)
    return sims

# Benchmark helper
def run_benchmark(fomc_arr, win_arr, valid_fomc_idx):
    """
    Times CPU execution. If GPU available, also times GPU and prints speedup.
    On AWS Academy (no GPU), prints CPU timing only — still demonstrates
    the algorithm's performance characteristics for the README.
    """
    print("\n[bench] CPU NumPy cosine similarity:")
    t0 = time.time()
    for _ in range(5):
        cosine_similarities_cpu(fomc_arr, win_arr, valid_fomc_idx)
    cpu_t = (time.time() - t0) / 5
    print(f"[bench]   mean CPU time : {cpu_t:.3f}s")

    if GPU_AVAILABLE:
        fomc_gpu = cp.asarray(fomc_arr)
        win_gpu  = cp.asarray(win_arr)
        valid_gpu = cp.asarray(valid_fomc_idx)
        cp.cuda.Stream.null.synchronize()
        print("[bench] GPU CuPy cosine similarity:")
        gpu_result = cupy_benchmark(
            cosine_similarities_gpu,
            (fomc_gpu, win_gpu, valid_gpu),
            n_repeat=5,
        )
        print(gpu_result)
    else:
        print("[bench] GPU not available on this machine.")
        print("[bench] To run GPU benchmark: use Midway 3 GPU partition with")
        print("[bench] 'module load cuda/12.1' and 'pip install cupy-cuda12x'")

# Main Function for executions
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket",    default=ENV_BUCKET,
                   help="S3 lake bucket (default: from .env BUCKET)")
    p.add_argument("--dim",       type=int, default=1 << 18)
    p.add_argument("--benchmark", action="store_true")
    args = p.parse_args()

    if not args.bucket:
        p.error("--bucket is required, or set BUCKET in your .env file")

    print("[sim] loading FOMC TF-IDF vectors from s3://.../curated/tfidf/ ...")
    fomc_df = _s3_parquets(args.bucket, "curated/tfidf/")
    fomc_df = fomc_df[fomc_df["source"] == "fomc"].reset_index(drop=True)

    # Use _s3_parquets_partitioned so partition column values are reconstructed from the S3 
    # key path rather than missing from the DataFrame.
    print("[sim] loading meeting-window vectors from s3://.../curated/meeting_windows/ ...")
    print("[sim] (note: dense 262144-dim vectors — expect ~4 GB RAM usage)")
    win_df = _s3_parquets_partitioned(args.bucket, "curated/meeting_windows/")
    print(f"[sim] {len(fomc_df)} FOMC docs, {len(win_df)} (meeting,lag,source) windows")
    print(f"[sim] win_df columns: {list(win_df.columns)}")

    # Build dense arrays
    fomc_arr = np.vstack([vec_to_dense(v, args.dim) for v in fomc_df["features"]])
    win_arr  = np.vstack([vec_to_dense(
                    v.get("values") if isinstance(v, dict) else v, args.dim)
                  for v in win_df["mean_features"]])

    fomc_index    = {row["doc_date"]: i for i, row in fomc_df.iterrows()}
    row_fomc_idx  = np.array(
        [fomc_index.get(d, -1) for d in win_df["meeting_date"]], dtype=np.int64)
    valid_mask    = row_fomc_idx >= 0
    valid_fomc_idx = row_fomc_idx[valid_mask]
    valid_win_arr  = win_arr[valid_mask]

    # Run similarity
    t0 = time.time()
    if GPU_AVAILABLE:
        fomc_gpu = cp.asarray(fomc_arr, dtype=cp.float32)
        win_gpu  = cp.asarray(valid_win_arr, dtype=cp.float32)
        vidx_gpu = cp.asarray(valid_fomc_idx)
        sims_cpu = cosine_similarities_gpu(fomc_gpu, win_gpu, vidx_gpu)
        device   = "GPU (CuPy)"
    else:
        sims_cpu = cosine_similarities_cpu(fomc_arr, valid_win_arr, valid_fomc_idx)
        device   = "CPU (NumPy)"

    elapsed = time.time() - t0
    print(f"[sim] {sims_cpu.size:,} similarities via {device} in {elapsed:.2f}s")

    # Save results
    out = win_df.loc[valid_mask].copy()
    out["cosine_sim"] = sims_cpu
    out_cols = ["meeting_date", "lag_days", "source", "cosine_sim"]
    out = out[out_cols].sort_values(["meeting_date", "source", "lag_days"])

    buf = io.BytesIO()
    out.to_parquet(buf, index=False)
    buf.seek(0)
    s3 = boto3.client("s3")
    s3.put_object(Bucket=args.bucket,
                  Key="processed/similarity/cosine.parquet",
                  Body=buf.read())
    print(f"[done] s3://{args.bucket}/processed/similarity/cosine.parquet")

    if args.benchmark:
        run_benchmark(fomc_arr, valid_win_arr, valid_fomc_idx)

if __name__ == "__main__":
    main()