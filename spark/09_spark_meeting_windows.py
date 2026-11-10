"""
MACS 30113
Yuzhao (Jerry) Jin; CNETID: yuzhaojin1
Duo (Doris) Wang; CNETID: dorisduowang

For each FOMC meeting m and lag k in {-7..+7} days, emits the mean TF-IDF
vector of every article from that (source, lag day) into:
    curated/meeting_windows/  (partitioned by meeting_date, lag_days)

The CuPy/NumPy similarity script (11_cupy_similarity.py) reads these as
(meeting, lag, source) tensors and computes cosine similarity vs FOMC vectors.

Utilizing course concepts:
- Week 7: Spark SQL cross-join + filter, window functions, groupBy + agg
- Week 7: User-defined functions (UDF) with VectorUDT
"""

import argparse
import sys
import numpy as np
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.stat import Summarizer

@F.udf(VectorUDT())
def mean_vec(vecs):
    """
    Average a list of Spark ML sparse/dense vectors into one dense vector.
    Used to compute the mean TF-IDF for each (meeting, lag, source) group.
    """
    if not vecs:
        return Vectors.sparse(1 << 18, [], [])
    import numpy as np
    acc = np.zeros(vecs[0].size, dtype=np.float32)
    for v in vecs:
        acc += v.toArray()
    return Vectors.dense((acc / len(vecs)).astype(float))

def main(bucket):
    spark = (
        SparkSession.builder
        .appName("makkapakka_meeting_windows")
        # Match memory config of 08 — same m5.large cluster
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.executor.memory", "5g")
        .config("spark.executor.cores", "2")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.adaptive.enabled", "true")
        # Auto-broadcast FOMC (88 rows, tiny) in the crossJoin
        .config("spark.sql.autoBroadcastJoinThreshold", "50m")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    tfidf_path = f"s3://{bucket}/curated/tfidf/"

    # 1) FOMC meeting dates + TF-IDF vectors (88 rows — Spark will broadcast)
    fomc = (
        spark.read.parquet(tfidf_path)
        .filter(F.col("source") == "fomc")
        .select(
            F.col("doc_date").alias("meeting_date"),
            F.col("doc_id").alias("fomc_id"),
            F.col("features").alias("fomc_vec"),
        )
        .distinct()
    )
    n_meetings = fomc.count()
    print(f"[spark] {n_meetings} FOMC meeting dates loaded")

    # 2) All media documents (NYT + Guardian + GDELT)
    media = (
        spark.read.parquet(tfidf_path)
        .filter(F.col("source") != "fomc")
        .cache()   # cache before crossJoin to avoid re-reading from S3
    )
    n_media = media.count()
    print(f"[spark] {n_media:,} media documents")

    # 3) Cross-join media × FOMC, then filter to |lag| <= 7 days.
    #    FOMC (88 rows) is auto-broadcast — Spark converts the crossJoin
    #    into a broadcast hash join, which is memory-efficient.
    crossed = (
        media.alias("a")
        .crossJoin(fomc.alias("f"))
        .withColumn(
            "lag_days",
            F.datediff(F.col("a.doc_date"), F.col("f.meeting_date"))
        )
        .filter(F.abs("lag_days") <= 7)
    )

    # 4) Mean TF-IDF per (meeting, lag, source)
    grouped = (
        crossed
        .groupBy("f.meeting_date", "lag_days", "a.source")
        .agg(F.collect_list("a.features").alias("vecs"))
        .withColumn("mean_features", mean_vec("vecs"))
        .drop("vecs")
    )

    out_path = f"s3://{bucket}/curated/meeting_windows/"
    (grouped
     .write
     .mode("overwrite")
     .partitionBy("meeting_date", "lag_days")
     .parquet(out_path))

    print(f"[done] wrote {out_path}")
    spark.stop()


def daily_windows(features):
    gdelt_corpora = F.array(F.lit("broad_macro"), F.when(F.col("direct_fed"), F.lit("direct_fed")))
    normalized_url = F.regexp_replace(F.trim("url"), r"(?i)^https?://(?:www\.)?|#.*$|/$", "")
    media = (features.filter(F.col("source") != "fomc")
             .withColumn("corpus", F.explode(F.when(F.col("source") == "gdelt", gdelt_corpora)
                                            .otherwise(F.array(F.lit("all")))))
             .filter(F.col("corpus").isNotNull())
             .withColumn("normalized_url", F.when(F.length(normalized_url) > 0, normalized_url)))
    keys = ["doc_date", "source", "corpus", "representation_type", "model_fit_cutoff"]
    # Distinct aggregates introduce null placeholders in Spark's expand step,
    # which the vector summarizer cannot deserialize.
    coverage = (media.groupBy(*keys)
                .agg(F.countDistinct("normalized_url").alias("n_urls"),
                     F.countDistinct("source_domain").alias("n_domains")))
    article = (media.groupBy(*keys)
               .agg(Summarizer.mean(F.col("features")).alias("mean_features"),
                    F.count("*").alias("n_docs"),
                    F.avg("gkg_tone.tone").alias("mean_native_tone"),
                    F.first("filter_version").alias("filter_version"))
               .join(coverage, on=keys))
    domain = (media.groupBy(*keys, "source_domain")
              .agg(Summarizer.mean(F.col("features")).alias("domain_features"),
                   F.avg("gkg_tone.tone").alias("domain_tone"))
              .groupBy(*keys)
              .agg(Summarizer.mean(F.col("domain_features")).alias("mean_features"),
                   F.avg("domain_tone").alias("mean_native_tone")))
    domain = domain.join(article.drop("mean_features", "mean_native_tone"), on=keys)
    return (article.withColumn("weighting", F.lit("article"))
            .unionByName(domain.withColumn("weighting", F.lit("domain_equal"))))


def versioned_main():
    parser = argparse.ArgumentParser(description="Build article- and domain-weighted meeting windows")
    parser.add_argument("--tfidf", required=True)
    parser.add_argument("--meetings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-lag", type=int, default=7)
    args = parser.parse_args()
    if args.max_lag < 0:
        parser.error("--max-lag must be nonnegative")
    # Sort aggregation processes each high-dimensional vector group in turn.
    # The default object hash aggregation holds many dense summarizer states in memory.
    spark = (SparkSession.builder.appName("fomc_versioned_meeting_windows")
             .config("spark.sql.execution.useObjectHashAggregateExec", "false").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    try:
        calendar = (spark.read.option("header", True).csv(args.meetings)
                    .select(F.to_date("meeting_date").alias("meeting_date")))
        if calendar.filter(F.col("meeting_date").isNull()).limit(1).count() or not calendar.take(1):
            raise ValueError("Meeting calendar must contain valid meeting dates")
        meetings = calendar.distinct().withColumn("doc_date", F.explode(F.sequence(
            F.date_sub("meeting_date", args.max_lag), F.date_add("meeting_date", args.max_lag))))
        windows = (daily_windows(spark.read.parquet(args.tfidf)).join(F.broadcast(meetings), on="doc_date")
                   .withColumn("lag_days", F.datediff("doc_date", "meeting_date")))
        (windows.write.mode("overwrite").partitionBy("meeting_date", "lag_days").parquet(args.output))
        print(f"[meeting-windows] wrote {args.output}")
    finally:
        spark.stop()


if __name__ == "__main__":
    if len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        main(sys.argv[1])
    else:
        versioned_main()
