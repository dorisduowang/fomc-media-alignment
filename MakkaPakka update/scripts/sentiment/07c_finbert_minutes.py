"""
FinBERT tone of the FOMC minutes as a contextual-embedding robustness check on the
Loughran-McDonald market-dictionary tone.

The dictionary tone in scripts/sentiment/07b_lm_dictionary.py counts words in fixed
sentiment categories; a transformer that was fine-tuned on financial text instead
reads each passage in context and returns a positive, negative, or neutral
probability, recovering negation and phrasing that a bag-of-words count misses. This
script scores the cached minutes with ProsusAI/finbert-tone, forms a net tone as the
positive minus negative probability averaged over the 512-token chunks of each
minute, and asks whether that contextual measure reproduces the two facts the
dictionary measure established: that negative tone rises with the pre-meeting VIX
and forecasts post-meeting realized volatility.

Outputs:
    results/tables/finbert_minutes.csv      per-meeting FinBERT tone and market data
    results/tables/finbert_validation.csv   tone-market correlations
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_reference"
TAB = ROOT / "results" / "tables"
MODEL = "yiyanghkust/finbert-tone"
MAX_LEN, STRIDE = 512, 64


def main():
    minutes = pd.read_csv(DATA / "fomc_minutes_raw.csv", dtype={"date": str})
    minutes["meeting_date"] = pd.to_datetime(minutes["date"], format="%Y%m%d")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)
    model.eval()
    # finbert-tone label order: 0 neutral, 1 positive, 2 negative.
    idx_pos, idx_neg = 1, 2

    @torch.no_grad()
    def score(text):
        # The tokenizer splits a long minute into overlapping 512-token windows,
        # each already carrying the special tokens and padding, so the model reads
        # the whole document in batches rather than only its first window.
        enc = tok(
            str(text),
            max_length=MAX_LEN,
            truncation=True,
            stride=STRIDE,
            return_overflowing_tokens=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
        }
        probs = model(**inputs).logits.softmax(-1).cpu().numpy()
        p = probs.mean(axis=0)
        return p[idx_pos], p[idx_neg]

    rows = []
    for _, r in minutes.iterrows():
        p_pos, p_neg = score(r["text"])
        rows.append(
            {
                "meeting_date": r["meeting_date"],
                "finbert_pos": p_pos,
                "finbert_neg": p_neg,
                "finbert_net": p_pos - p_neg,
            }
        )
    fb = pd.DataFrame(rows)

    lm = pd.read_csv(TAB / "sentiment_comparison.csv")
    lm["meeting_date"] = pd.to_datetime(lm["meeting_date"])
    merged = fb.merge(lm, on="meeting_date", how="left").sort_values("meeting_date")
    merged.to_csv(TAB / "finbert_minutes.csv", index=False)

    pairs = [
        ("finbert_neg_vs_pre_vix", "finbert_neg", "vix_pre"),
        ("finbert_neg_vs_post_vol", "finbert_neg", "vol_post"),
        ("finbert_net_vs_post_vol", "finbert_net", "vol_post"),
        ("finbert_neg_vs_abs_surprise", "finbert_neg", "abs_mps"),
        ("finbert_net_vs_lm_net", "finbert_net", "lm_net_per_1k"),
        ("finbert_neg_vs_lm_neg", "finbert_neg", "lm_neg_per_1k"),
    ]
    recs = []
    for name, a, b in pairs:
        d = merged.dropna(subset=[a, b])
        r_p, p_p = pearsonr(d[a], d[b])
        recs.append(
            {"relation": name, "n": len(d), "pearson_r": r_p, "pearson_p": p_p}
        )
    val = pd.DataFrame(recs)
    val.to_csv(TAB / "finbert_validation.csv", index=False)
    pd.set_option("display.width", 140)
    print(val.to_string(index=False))


if __name__ == "__main__":
    main()
