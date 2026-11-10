"""
Distributed media-FOMC cosine similarity with Spark.

For versioned runs, compare each pre-aggregated meeting window with its FOMC
minutes vector. The legacy document-vector interface is also available.

    spark-submit scripts/similarity/11_spark_similarity.py \
        --windows s3://<bucket>/runs/<id>/curated/meeting_windows \
        --tfidf   s3://<bucket>/runs/<id>/curated/tfidf \
        --output  s3://<bucket>/runs/<id>/processed/per_meeting
"""

import argparse

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.linalg import SparseVector
from pyspark.sql import types as T


def build_spark(app="spark-similarity"):
    return SparkSession.builder.appName(app).getOrCreate()


def cosine(a, b):
    """Cosine similarity between two Spark ML vectors."""
    if a is None or b is None:
        return None
    dot = float(a.dot(b))
    na = float(a.norm(2))
    nb = float(b.norm(2))
    return min(1.0, dot / (na * nb)) if na > 0 and nb > 0 else None


def versioned_similarity(spark, args):
    windows = spark.read.parquet(args.windows).withColumn("meeting_date", F.to_date("meeting_date"))
    fomc = (spark.read.parquet(args.tfidf).filter(F.col("source") == "fomc")
            .select(F.to_date("doc_date").alias("meeting_date"), "model_fit_cutoff",
                    F.col("features").alias("fomc_features")))
    keys = ["meeting_date", "model_fit_cutoff"]
    if fomc.groupBy(*keys).count().filter(F.col("count") > 1).limit(1).count():
        raise ValueError("Expected one FOMC vector per meeting and fit cutoff")
    joined = windows.join(F.broadcast(fomc), on=keys, how="left")
    if joined.filter(F.col("fomc_features").isNull()).limit(1).count():
        raise ValueError("Meeting windows are missing a matching FOMC vector")
    return (joined.withColumn("cosine_sim", F.udf(cosine, T.DoubleType())("mean_features", "fomc_features"))
            .drop("mean_features", "fomc_features"))


def legacy_similarity(spark, args):
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
        lambda vecs, fv: cosine(mean_vectors(vecs), fv) or 0.0, T.DoubleType()
    )
    return joined.withColumn(
        "cosine_sim", cos_udf(F.col("vecs"), F.col("tfidf_fomc"))
    ).select("meeting_date", "lag_days", "source", "cosine_sim")


def main():
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--windows", help="Versioned, pre-aggregated meeting windows")
    source.add_argument("--media", help="Legacy TF-IDF vectors per media document")
    p.add_argument("--tfidf", help="Versioned TF-IDF containing the FOMC minutes")
    p.add_argument("--fomc", help="Legacy TF-IDF vectors per FOMC minute")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.windows and (not args.tfidf or args.fomc) or args.media and (not args.fomc or args.tfidf):
        p.error("Use --windows with --tfidf, or --media with --fomc")

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    try:
        out = versioned_similarity(spark, args) if args.windows else legacy_similarity(spark, args)
        out.write.mode("overwrite").parquet(args.output)
        print(f"[spark-similarity] wrote cosine similarities to {args.output}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
