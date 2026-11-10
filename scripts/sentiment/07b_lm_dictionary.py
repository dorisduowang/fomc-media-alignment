"""
Loughran-McDonald sentiment on the FOMC minutes and agreement with the custom
hawkish-dovish dictionary.

The distributed pipeline scores tone with a transparent, auditable hawkish-dovish
lexicon (see scripts/sentiment/07_spark_sentiment.py). This script validates that
baseline against the standard finance-domain dictionary of Loughran and McDonald
(2011). For every FOMC minute for which the cleaned text is available locally we
compute the LM polarity score and compare it with the custom net tone.

Agreement between an independently constructed, peer-reviewed dictionary and our
lexicon is the reproducible evidence that the tone measure is not an artifact of
one particular word list.

Outputs:
    results/tables/sentiment_comparison.csv   per-meeting LM and custom scores
    results/tables/sentiment_agreement.csv    correlation summary
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pysentiment2 as ps
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"
DATA = ROOT / "data_reference"
MINUTES = (
    ROOT
    / "112 - final-project-makkapakka-main"
    / "FOMC Cleaning and Initial Data Analysis"
    / "fomc_minutes_cleaned.csv"
)


def lm_polarity(text, lm):
    tokens = lm.tokenize(str(text))
    score = lm.get_score(tokens)
    return score["Polarity"], score["Positive"], score["Negative"], len(tokens)


def main():
    lm = ps.LM()
    minutes = pd.read_csv(MINUTES)
    minutes["date"] = pd.to_datetime(minutes["date"])

    recs = []
    for _, row in minutes.iterrows():
        pol, pos, neg, ntok = lm_polarity(row["cleaned_text"], lm)
        recs.append(
            {
                "meeting_date": row["date"],
                "lm_polarity": pol,
                "lm_pos_per_1k": 1000 * pos / max(ntok, 1),
                "lm_neg_per_1k": 1000 * neg / max(ntok, 1),
                "lm_net_per_1k": 1000 * (pos - neg) / max(ntok, 1),
            }
        )
    lm_df = pd.DataFrame(recs)

    sent = pd.read_parquet(DATA / "sentiment.parquet")
    fomc = sent[sent["source"] == "fomc"].copy()
    fomc["meeting_date"] = pd.to_datetime(fomc["doc_date"])
    custom = fomc[["meeting_date", "net_per_1k"]].rename(
        columns={"net_per_1k": "custom_net_per_1k"}
    )

    merged = lm_df.merge(custom, on="meeting_date", how="inner").sort_values(
        "meeting_date"
    )
    merged.to_csv(TAB / "sentiment_comparison.csv", index=False)

    r_p, p_p = pearsonr(merged["lm_net_per_1k"], merged["custom_net_per_1k"])
    r_s, p_s = spearmanr(merged["lm_net_per_1k"], merged["custom_net_per_1k"])
    sign_agree = np.mean(
        np.sign(merged["lm_net_per_1k"]) == np.sign(merged["custom_net_per_1k"])
    )
    agree = pd.DataFrame(
        [
            {
                "n_meetings": len(merged),
                "pearson_r": r_p,
                "pearson_p": p_p,
                "spearman_rho": r_s,
                "spearman_p": p_s,
                "sign_agreement": sign_agree,
                "first_meeting": merged["meeting_date"].min().date(),
                "last_meeting": merged["meeting_date"].max().date(),
            }
        ]
    )
    agree.to_csv(TAB / "sentiment_agreement.csv", index=False)
    print(agree.to_string(index=False))


if __name__ == "__main__":
    main()
