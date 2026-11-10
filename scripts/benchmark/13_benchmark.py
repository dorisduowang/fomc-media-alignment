"""
Scalable-computing benchmark: sentiment scoring throughput.

Times three implementations of the hawkish-dovish token scorer on a synthetic
five-million-token corpus: a pure-Python dictionary loop, a NumPy searchsorted
vectorization, and a Numba parallel JIT kernel. The GPU cosine-similarity
benchmark lives in gpu/11_cupy_similarity.py and activates when CUDA is present;
on a CPU-only host it reports the NumPy path only.

Outputs:
    results/tables/benchmarks.csv
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

N_TOKENS = 5_000_000
LEXICON_SIZE = 200
RNG = np.random.default_rng(30113)


def make_corpus():
    lex_hashes = np.sort(RNG.integers(0, 10_000, size=LEXICON_SIZE))
    lex_scores = RNG.choice([-1, 1], size=LEXICON_SIZE).astype(np.int64)
    # 10 percent of tokens are lexicon hits
    hits = RNG.integers(0, LEXICON_SIZE, size=int(0.1 * N_TOKENS))
    noise = RNG.integers(10_000, 1_000_000, size=N_TOKENS - len(hits))
    tokens = np.concatenate([lex_hashes[hits], noise])
    RNG.shuffle(tokens)
    return tokens.astype(np.int64), lex_hashes, lex_scores


def score_python(tokens, lex_hashes, lex_scores):
    table = {int(h): int(s) for h, s in zip(lex_hashes, lex_scores)}
    total = 0
    for t in tokens:
        s = table.get(int(t))
        if s is not None:
            total += s
    return total


def score_numpy(tokens, lex_hashes, lex_scores):
    idx = np.searchsorted(lex_hashes, tokens)
    idx = np.clip(idx, 0, len(lex_hashes) - 1)
    hit = lex_hashes[idx] == tokens
    return int(lex_scores[idx][hit].sum())


@njit(parallel=True, cache=True)
def _score_numba(tokens, lex_hashes, lex_scores):
    total = 0
    n = len(lex_hashes)
    for i in prange(len(tokens)):
        lo, hi = 0, n - 1
        t = tokens[i]
        while lo <= hi:
            mid = (lo + hi) // 2
            if lex_hashes[mid] == t:
                total += lex_scores[mid]
                break
            elif lex_hashes[mid] < t:
                lo = mid + 1
            else:
                hi = mid - 1
    return total


def timeit(fn, *args, repeat=3):
    fn(*args)  # warm up (JIT compile / cache)
    best = min(_run(fn, *args) for _ in range(repeat))
    return best


def _run(fn, *args):
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0


def main():
    tokens, lex_hashes, lex_scores = make_corpus()
    rows = []
    t_py = timeit(score_python, tokens, lex_hashes, lex_scores)
    t_np = timeit(score_numpy, tokens, lex_hashes, lex_scores)
    t_nb = timeit(_score_numba, tokens, lex_hashes, lex_scores)
    rows = [
        {"task": "sentiment-5M-tokens", "impl": "python", "seconds": t_py, "speedup": 1.0},
        {"task": "sentiment-5M-tokens", "impl": "numpy", "seconds": t_np, "speedup": t_py / t_np},
        {"task": "sentiment-5M-tokens", "impl": "numba", "seconds": t_nb, "speedup": t_py / t_nb},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "benchmarks.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
