"""
FinBERT tone inference (contextual-sentiment robustness).

FinBERT (yiyanghkust/finbert-tone) classifies financial text as positive,
negative, or neutral tone. This is the higher-capacity robustness check on the
custom hawkish-dovish dictionary and the Loughran-McDonald baseline. Running it
over the full media corpus is GPU-bound; on the FOMC minutes it also runs on CPU.
Long documents are scored by chunking to the 512-token model limit and averaging
the softmax tone probabilities.

    python scripts/sentiment/07c_finbert_inference.py \
        --input curated/fomc_text.parquet --text-col text \
        --output processed/finbert_tone.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def chunk_tokens(ids, max_len, stride):
    step = max_len - stride
    for start in range(0, max(len(ids), 1), step):
        yield ids[start : start + max_len]
        if start + max_len >= len(ids):
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--text-col", default="text")
    p.add_argument("--model", default="yiyanghkust/finbert-tone")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--stride", type=int, default=64)
    args = p.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()
    # finbert-tone label order: 0 neutral, 1 positive, 2 negative
    labels = ["neutral", "positive", "negative"]

    df = pd.read_parquet(args.input)

    @torch.no_grad()
    def score(text):
        ids = tok.encode(str(text), add_special_tokens=False)
        probs = []
        for chunk in chunk_tokens(ids, args.max_len - 2, args.stride):
            enc = tok.prepare_for_model(
                chunk, max_length=args.max_len, truncation=True, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy()[0])
        mean = np.mean(probs, axis=0)
        return mean  # neutral, positive, negative

    recs = []
    for _, row in df.iterrows():
        m = score(row[args.text_col])
        recs.append(
            {
                **{c: row[c] for c in df.columns if c != args.text_col},
                "finbert_neutral": m[0],
                "finbert_positive": m[1],
                "finbert_negative": m[2],
                "finbert_tone": float(m[1] - m[2]),
            }
        )
    out = pd.DataFrame(recs)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"[finbert] scored {len(out)} documents to {args.output}")


if __name__ == "__main__":
    main()
