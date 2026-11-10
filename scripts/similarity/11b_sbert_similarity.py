"""
Sentence-BERT media-FOMC similarity (contextual-embedding robustness).

This is the primary similarity measure in the final robustness design. It embeds
each document and each FOMC minute with all-mpnet-base-v2 (768 dimensions) and
computes cosine similarity in the dense embedding space, which captures negation
and context that TF-IDF cannot. Embedding the full media corpus is GPU-bound, so
this job is intended for a GPU node (personal AWS GPU, Colab, or Midway); on the
88 FOMC minutes and the curated outlets it also runs on CPU.

    python scripts/similarity/11b_sbert_similarity.py \
        --media curated/media_text.parquet --fomc curated/fomc_text.parquet \
        --output processed/similarity_sbert.parquet --batch-size 64
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def embed(texts, model, batch_size):
    return model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--media", required=True)
    p.add_argument("--fomc", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)

    media = pd.read_parquet(args.media)   # meeting_date, lag_days, source, text
    fomc = pd.read_parquet(args.fomc)     # meeting_date, text

    fomc_emb = embed(fomc["text"], model, args.batch_size)
    fomc_map = {d: e for d, e in zip(fomc["meeting_date"], fomc_emb)}

    media_emb = embed(media["text"], model, args.batch_size)
    media = media.assign(_row=np.arange(len(media)))

    # Mean embedding per (meeting, lag, source), then cosine against the meeting.
    rows = []
    for (m, k, s), grp in media.groupby(["meeting_date", "lag_days", "source"]):
        if m not in fomc_map:
            continue
        v = media_emb[grp["_row"].to_numpy()].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-12)
        f = fomc_map[m]
        rows.append(
            {
                "meeting_date": m,
                "lag_days": k,
                "source": s,
                "cosine_sim": float(v @ f),
            }
        )
    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"[sbert] wrote {len(out)} similarities to {args.output}")


if __name__ == "__main__":
    main()
