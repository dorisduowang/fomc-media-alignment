"""
Distributed media-FOMC cosine similarity with Spark.

This replaces the single-machine CuPy/NumPy similarity step. The FOMC minute
TF-IDF vectors (one per meeting, 2^18 dimensions) are small enough to broadcast;
the media vectors are the large side and stay distributed. Each media document is
aggregated into its (meeting, lag, source) window, and cosine similarity against
the broadcast FOMC vector is computed on the workers. Nothing is pulled back to a
single machine, which is the point of moving off the GPU box.

    spark-submit scripts/similarity/11_spark_similarity.py \
        --media  s3://<bucket>/curated/tfidf \
        --fomc   s3://<bucket>/curated/tfidf_fomc \
        --output s3://<bucket>/processed/similarity
"""

import argparse

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.linalg import SparseVector, Vectors
from pyspark.sql import types as T


def build_spark(app="spark-similarity"):
    return SparkSession.builder.appName(app).getOrCreate()


def cosine(a, b):
    """Cosine similarity between two pyspark.ml sparse vectors."""
    if a is None or b is None:
        return 0.0
    dot = float(a.dot(b))
    na = float(a.norm(2))
    nb = float(b.norm(2))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--media", required=True, help="TF-IDF vectors per media document")
    p.add_argument("--fomc", required=True, help="TF-IDF vectors per FOMC minute")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    spark = build_spark()

    media = spark.read.parquet(args.media)  # meeting_date, lag_days, source, tfidf
    fomc = spark.read.parquet(args.fomc)    # meeting_date, tfidf_fomc

    # Mean media TF-IDF vector per (meeting, lag, source). Averaging sparse
    # vectors is done through a small serializer-friendly aggregation.
    def mean_vectors(vs):
        acc = {}
        n = 0
        size = 0
        for v in vs:
            n += 1
            size = v.size
            for i, x in zip(v.indices, v.values):
                acc[i] = acc.get(i, 0.0) + x
        if n == 0:
            return None
        idx = sorted(acc)
        return SparseVector(size, idx, [acc[i] / n for i in idx])

    # Aggregate media vectors into (meeting, lag, source) windows, then compute
    # cosine similarity against the broadcast FOMC vector on the workers.
    windowed = (
        media.groupBy("meeting_date", "lag_days", "source")
        .agg(F.collect_list("tfidf").alias("vecs"))
    )

    joined = windowed.join(F.broadcast(fomc), on="meeting_date", how="inner")

    cos_udf = F.udf(
        lambda vecs, fv: cosine(mean_vectors(vecs), fv), T.DoubleType()
    )
    out = joined.withColumn(
        "cosine_sim", cos_udf(F.col("vecs"), F.col("tfidf_fomc"))
    ).select("meeting_date", "lag_days", "source", "cosine_sim")

    out.write.mode("overwrite").parquet(args.output)
    print(f"[spark-similarity] wrote cosine similarities to {args.output}")
    spark.stop()


if __name__ == "__main__":
    main()
