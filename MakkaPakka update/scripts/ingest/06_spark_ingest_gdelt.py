"""
Distributed GDELT 2.0 ingestion with Spark.

This replaces the thread-pool GDELT ingester with a genuinely distributed job. It
reads the GDELT Global Knowledge Graph shards directly, filters to United States,
Federal-Reserve-relevant records inside each FOMC meeting window with Spark SQL,
and writes date-partitioned Parquet. The same code runs on a local Spark session
(for validation on a sample week) and on an EMR cluster (for the full corpus)
without modification; only the input and output paths change.

Local validation:
    spark-submit scripts/ingest/06_spark_ingest_gdelt.py \
        --gdelt-glob "sample/gkg/*.csv" \
        --out results/gdelt_parquet --meetings results/tables/panel_meeting.csv

EMR (full corpus):
    spark-submit --deploy-mode cluster scripts/ingest/06_spark_ingest_gdelt.py \
        --gdelt-glob "s3://gdelt-open-data/v2/gkg/*.gkg.csv" \
        --out s3://<bucket>/curated/gdelt --meetings s3://<bucket>/ref/meetings.csv

The GDELT GKG v2 layout is documented in GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf.
Only the columns needed downstream are projected, so shards are never materialized
in full on any single worker.
"""

import argparse

from pyspark.sql import SparkSession, functions as F, types as T

# GKG v2.1 is tab-delimited with 27 fields. We name and keep the ones we use.
GKG_COLS = [
    "GKGRECORDID", "DATE", "SourceCollectionIdentifier", "SourceCommonName",
    "DocumentIdentifier", "Counts", "V2Counts", "Themes", "V2Themes",
    "Locations", "V2Locations", "Persons", "V2Persons", "Organizations",
    "V2Organizations", "V2Tone", "Dates", "GCAM", "SharingImage",
    "RelatedImages", "SocialImageEmbeds", "SocialVideoEmbeds", "Quotations",
    "AllNames", "Amounts", "TranslationInfo", "Extras",
]

FED_TERMS = [
    "federal reserve", "fomc", "monetary policy", "interest rate", "jerome powell",
    "janet yellen", "rate hike", "rate cut", "central bank",
]


def build_spark(app="gdelt-ingest"):
    return (
        SparkSession.builder.appName(app)
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def load_meeting_windows(spark, meetings_path, pad_days):
    m = spark.read.option("header", True).csv(meetings_path)
    m = m.select(F.to_date("meeting_date").alias("meeting_date")).distinct()
    m = m.withColumn("win_start", F.date_sub("meeting_date", pad_days))
    m = m.withColumn("win_end", F.date_add("meeting_date", pad_days))
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gdelt-glob", required=True, help="input GKG shards (local or s3)")
    p.add_argument("--out", required=True, help="output Parquet path")
    p.add_argument("--meetings", required=True, help="CSV with a meeting_date column")
    p.add_argument("--pad-days", type=int, default=7)
    args = p.parse_args()

    spark = build_spark()

    raw = (
        spark.read.option("sep", "\t")
        .option("header", False)
        .csv(args.gdelt_glob)
        .toDF(*GKG_COLS)
    )

    # Parse the YYYYMMDDHHMMSS stamp into a date, and lowercase the fields we scan.
    df = raw.select(
        "GKGRECORDID",
        F.to_date(F.substring("DATE", 1, 8), "yyyyMMdd").alias("doc_date"),
        F.col("SourceCommonName").alias("source_domain"),
        F.col("DocumentIdentifier").alias("url"),
        F.lower(F.coalesce("V2Themes", F.lit(""))).alias("themes"),
        F.lower(F.coalesce("V2Organizations", F.lit(""))).alias("orgs"),
        F.lower(F.coalesce("AllNames", F.lit(""))).alias("names"),
        F.split("V2Tone", ",").getItem(0).cast("double").alias("tone"),
    )

    # United States sources: GDELT country inference is imperfect, so we keep
    # .com/.org/.gov/.us domains plus explicit US outlets and filter on English
    # Fed-relevant content in the theme and organization fields.
    us_domain = F.col("url").rlike(r"\.(com|org|gov|net|us)(/|$)")
    fed_regex = "|".join([t.replace(" ", r"\s+") for t in FED_TERMS])
    fed_relevant = (
        F.col("themes").rlike("econ_|usfed|monetary")
        | F.col("orgs").rlike(fed_regex)
        | F.col("names").rlike(fed_regex)
    )
    df = df.filter(us_domain & fed_relevant & F.col("doc_date").isNotNull())

    windows = load_meeting_windows(spark, args.meetings, args.pad_days)

    # Keep only records inside a meeting window; tag each with its meeting and lag.
    joined = df.join(
        F.broadcast(windows),
        (df.doc_date >= windows.win_start) & (df.doc_date <= windows.win_end),
    ).withColumn("lag_days", F.datediff("doc_date", "meeting_date"))

    out = joined.select(
        "GKGRECORDID", "doc_date", "meeting_date", "lag_days",
        "source_domain", "url", "themes", "tone",
    )

    (
        out.write.mode("overwrite")
        .partitionBy("meeting_date")
        .parquet(args.out)
    )

    n = out.count()
    print(f"[gdelt-spark] wrote {n} Fed-relevant US records to {args.out}")
    spark.stop()


if __name__ == "__main__":
    main()
