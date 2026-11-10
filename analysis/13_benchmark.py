"""
MACS 30113
Yuzhao (Jerry) Jin; CNETID: yuzhaojin1
Duo (Doris) Wang; CNETID: dorisduowang

This file is for reproducible benchmarks that compares: 
  (a) pure-Python loop sentiment scoring (baseline)
  (b) NumPy vectorized (CPU vector processing)
  (c) Numba @jit(nopython=True, parallel=True)
  (d) Spark on EMR (counts via spark-submit, timed externally)
  (e) CuPy GPU all-pairs cosine vs NumPy CPU

All numbers are written to processed/benchmarks/benchmarks.csv

Utilizing course contents: 
- Week 1-2: numba JIT vs Python vs NumPy
- Week 3: CuPy GPU vs NumPy CPU
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

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
ENV_BUCKET = os.environ.get("BUCKET", "")


# Reuse the JIT kernel from the sentiment scorer
import sys
from importlib import import_module
_scoring_dir = str(Path(__file__).resolve().parent.parent / "scoring")
if _scoring_dir not in sys.path:
    sys.path.insert(0, _scoring_dir)
sent = import_module("07_numba_sentiment")     # noqa: E402


def py_score(token_hashes, lex_hashes, lex_w):
    """Pure-Python reference implementation."""
    lex = dict(zip(lex_hashes.tolist(), lex_w.tolist()))
    s = 0.0
    for t in token_hashes:
        if t in lex:
            s += lex[t]
    return s


def numpy_score(token_hashes, lex_hashes, lex_w):
    """Vectorised baseline: searchsorted is O(T log L) with no Python loop."""
    idx = np.searchsorted(lex_hashes, token_hashes)
    idx_clip = np.clip(idx, 0, lex_hashes.size - 1)
    hits = lex_hashes[idx_clip] == token_hashes
    return float((lex_w[idx_clip] * hits).sum())


def cupy_cosine(V, A_rows, B_rows):
    """All-pairs cosine on GPU vs CPU at a given problem size."""
    import cupy as cp
    A_cpu = np.random.rand(A_rows, V).astype(np.float32)
    B_cpu = np.random.rand(B_rows, V).astype(np.float32)

    # CPU
    t0 = time.time()
    An = A_cpu / (np.linalg.norm(A_cpu, axis=1, keepdims=True) + 1e-9)
    Bn = B_cpu / (np.linalg.norm(B_cpu, axis=1, keepdims=True) + 1e-9)
    _ = An @ Bn.T
    cpu_t = time.time() - t0

    # GPU
    A_gpu = cp.asarray(A_cpu); B_gpu = cp.asarray(B_cpu)
    cp.cuda.Stream.null.synchronize()
    t0 = time.time()
    An = A_gpu / (cp.linalg.norm(A_gpu, axis=1, keepdims=True) + 1e-9)
    Bn = B_gpu / (cp.linalg.norm(B_gpu, axis=1, keepdims=True) + 1e-9)
    _ = (An @ Bn.T).get()
    cp.cuda.Stream.null.synchronize()
    gpu_t = time.time() - t0
    return cpu_t, gpu_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default=ENV_BUCKET,
                   help="S3 lake bucket (default: from .env BUCKET)")
    p.add_argument("--corpus-size", type=int, default=5_000_000,
                   help="number of synthetic tokens for sentiment benchmark")
    args = p.parse_args()

    print(f"[bench] generating {args.corpus_size:,} synthetic tokens ...")
    lex_h, lex_w = sent.build_lexicon_arrays()
    rng = np.random.default_rng(0)
    # 90% random hashes (misses), 10% drawn from the lexicon (hits)
    tokens = np.where(
        rng.random(args.corpus_size) < 0.1,
        lex_h[rng.integers(0, lex_h.size, size=args.corpus_size)],
        rng.integers(0, 1 << 62, size=args.corpus_size, dtype=np.int64),
    )

    # (a) pure-Python
    t0 = time.time()
    py_score(tokens[: min(200_000, args.corpus_size)], lex_h, lex_w)
    py_t = (time.time() - t0) * (args.corpus_size / 200_000)   # scale to full size
    print(f"[bench] python (extrapolated) ~ {py_t:.2f}s")

    # (b) numpy
    t0 = time.time()
    numpy_score(tokens, lex_h, lex_w)
    np_t = time.time() - t0
    print(f"[bench] numpy   {np_t:.3f}s")

    # (c) numba — call once to compile, then time
    doc_starts = np.array([0], dtype=np.int64)
    doc_lens   = np.array([tokens.size], dtype=np.int64)
    sent.score_documents(doc_starts, doc_lens, tokens[:8], lex_h, lex_w)   # warm
    t0 = time.time()
    sent.score_documents(doc_starts, doc_lens, tokens, lex_h, lex_w)
    nb_t = time.time() - t0
    print(f"[bench] numba   {nb_t:.3f}s   (parallel JIT)")

    rows = [
        {"task": "sentiment-5M-tokens", "impl": "python", "seconds": py_t},
        {"task": "sentiment-5M-tokens", "impl": "numpy",  "seconds": np_t},
        {"task": "sentiment-5M-tokens", "impl": "numba",  "seconds": nb_t},
    ]

    # (e) CuPy vs NumPy for all-pairs cosine
    try:
        for V in (1 << 14, 1 << 16, 1 << 18):
            cpu_t, gpu_t = cupy_cosine(V, 256, 1024)
            rows.append({"task": f"cosine-256x1024@V={V}",
                         "impl": "numpy", "seconds": cpu_t})
            rows.append({"task": f"cosine-256x1024@V={V}",
                         "impl": "cupy",  "seconds": gpu_t})
            print(f"[bench] cosine V={V}  cpu={cpu_t:.3f}s  gpu={gpu_t:.3f}s  "
                  f"speedup={cpu_t/gpu_t:.1f}x")
    except Exception as e:
        print(f"[bench] CuPy skipped: {e}")

    df = pd.DataFrame(rows)
    s3 = boto3.client("s3")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=args.bucket,
                  Key="processed/benchmarks/benchmarks.csv",
                  Body=buf.getvalue().encode("utf-8"))
    print("\n[markdown]")
    print(df.to_markdown(index=False))
    print(f"\n[done] s3://{args.bucket}/processed/benchmarks/benchmarks.csv")


if __name__ == "__main__":
    main()
