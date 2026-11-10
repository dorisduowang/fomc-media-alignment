"""
Distributed hawkish-dovish sentiment scoring with a Spark pandas UDF.

This replaces the single-machine Numba scorer with a distributed job. The
hawkish-dovish lexicon is broadcast to every executor, and a grouped pandas UDF
scores documents partition-by-partition using vectorized NumPy set membership, so
the same tone measure the paper reports is produced across the whole corpus
without pulling text back to the driver.

    spark-submit scripts/sentiment/07_spark_sentiment.py \
        --input  s3://<bucket>/curated/{gdelt,nyt,guardian,fomc} \
        --output s3://<bucket>/processed/sentiment
"""

import argparse

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F, types as T

# A compact, auditable hawkish-dovish lexicon. Hawkish terms signal tightening or
# inflation concern; dovish terms signal accommodation or growth concern. The
# full list mirrors scoring/07_numba_sentiment.py.
HAWKISH = [
    "inflation", "inflationary", "tightening", "tighten", "restrictive", "hawkish",
    "overheating", "hike", "raise", "raising", "elevated", "persistent", "upside",
    "firmer", "robust", "strong", "resilient", "accelerate", "pressures",
]
DOVISH = [
    "accommodative", "accommodation", "easing", "ease", "dovish", "cut", "lower",
    "lowering", "cooling", "softening", "weak", "weakening", "downside", "slack",
    "unemployment", "recession", "sluggish", "moderating", "subdued",
]


def build_spark(app="spark-sentiment"):
    return SparkSession.builder.appName(app).getOrCreate()


def make_udf(bc_hawk, bc_dove):
    schema = T.StructType(
        [
            T.StructField("hawk_count", T.LongType()),
            T.StructField("dove_count", T.LongType()),
            T.StructField("n_tokens", T.LongType()),
            T.StructField("net_per_1k", T.DoubleType()),
        ]
    )

    @F.pandas_udf(schema)
    def score(text: pd.Series) -> pd.DataFrame:
        hawk = bc_hawk.value
        dove = bc_dove.value
        rows = []
        for doc in text.fillna(""):
            toks = doc.lower().split()
            n = len(toks)
            h = sum(1 for t in toks if t in hawk)
            d = sum(1 for t in toks if t in dove)
            net = 1000.0 * (h - d) / n if n else 0.0
            rows.append((h, d, n, net))
        return pd.DataFrame(rows, columns=["hawk_count", "dove_count", "n_tokens", "net_per_1k"])

    return score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--text-col", default="text")
    args = p.parse_args()

    spark = build_spark()
    bc_hawk = spark.sparkContext.broadcast(set(HAWKISH))
    bc_dove = spark.sparkContext.broadcast(set(DOVISH))

    df = spark.read.parquet(args.input)
    score = make_udf(bc_hawk, bc_dove)
    scored = df.withColumn("s", score(F.col(args.text_col))).select(
        "*",
        F.col("s.hawk_count").alias("hawk_count"),
        F.col("s.dove_count").alias("dove_count"),
        F.col("s.n_tokens").alias("n_tokens"),
        F.col("s.net_per_1k").alias("net_per_1k"),
    ).drop("s")

    scored.write.mode("overwrite").parquet(args.output)
    print(f"[spark-sentiment] wrote scores to {args.output}")
    spark.stop()


if __name__ == "__main__":
    main()
