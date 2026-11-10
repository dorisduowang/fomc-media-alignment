"""
MACS 30113
Yuzhao (Jerry) Jin; CNETID: yuzhaojin1
Duo (Doris) Wang; CNETID: dorisduowang

The purpose of this file is for distributed preprocessing of the entire Spark 
file lake (FOMC + NYT + Guardian + GDELT) on an EMR Spark cluster.  
This Writes three curated artifacts:
    curated/tokens/ one row per doc with cleaned tokens
    curated/tfidf/  one row per doc with TF-IDF sparse vec 
    curated/ngrams/ bigram frequency tables 

Submit via launch_emr_and_run.py (which handles spark-submit config).

Utilizing course contents:
- Week 7: Spark execution model, RDD/DataFrame, lazy evaluation
- Week 7: pyspark.ml.feature  RegexTokenizer / StopWordsRemover / HashingTF / IDF
- Week 7: NGram extraction

"""
import argparse
import hashlib
import sys
from datetime import date
from pathlib import Path

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, NGram, HashingTF, IDF, Normalizer

# Unified schema — only the 5 fields every source shares
SCHEMA_DOC = T.StructType([
    T.StructField("source",   T.StringType(), True),
    T.StructField("doc_id",   T.StringType(), True),
    T.StructField("doc_date", T.StringType(), True),
    T.StructField("text",     T.StringType(), True),
    T.StructField("url",      T.StringType(), True),
])

GKG_TONE_SCHEMA = T.StructType([
    T.StructField(name, T.DoubleType()) for name in (
        "tone", "positive_score", "negative_score", "polarity",
        "activity_ref_density", "self_group_ref_density", "word_count",
    )
])
GKG_TONE_SCHEMA.add("raw_tone", T.StringType())
VERSIONED_SCHEMA = T.StructType([
    *SCHEMA_DOC.fields,
    T.StructField("metadata_text", T.StringType()),
    T.StructField("source_domain", T.StringType()),
    T.StructField("representation_type", T.StringType()),
    T.StructField("filter_version", T.StringType()),
    T.StructField("direct_fed", T.BooleanType()),
    T.StructField("broad_macro", T.BooleanType()),
    T.StructField("gkg_tone", GKG_TONE_SCHEMA),
])


def read_all_sources(spark, bucket):
    """
    Read every source into one unified DataFrame with columns:
        source, doc_id, doc_date, text, url

    Data formats confirmed from uploaded sample files:
      FOMC — one JSON per file, "text" = full minutes text
      NYT — one JSON per file, "text" = headline + abstract + snippet
      Guardian — one JSON per file, "text" = full article body
      GDELT — JSONL (multiple records per file), "text" = pre-built
              space-joined themes + organizations + persons string
    """
    # FOMC / NYT / Guardian
    # All single-JSON-per-file. Schema projection picks only the 5 fields we
    # need and ignores extra fields (n_chars, ingested_at, headline, etc.).
    docs = spark.read.json(
        [
            f"s3://{bucket}/raw/fomc/*/*.json",
            f"s3://{bucket}/raw/nyt/*/*.json",
            f"s3://{bucket}/raw/guardian/*/*.json",
        ],
        schema=SCHEMA_DOC,
    )

    # GDELT
    # Path A: Pre-filtered JSONL shards from 06_ingest_gdelt.py.
    # The JSONL has these fields (confirmed from gdelt_20150316000000.jsonl):
    #   source, doc_id, doc_date, url, source_domain
    #   themes        : Array[String]   ← NOT a string, cannot concat_ws directly
    #   organizations : Array[String]
    #   persons       : Array[String]
    #   locations     : Array[Struct]
    #   tone          : Struct{tone, positive_score, ...}
    #   text          : String   ← already space-joined themes+orgs+persons
    try:
        gdelt_raw = spark.read.json(f"s3://{bucket}/raw/gdelt/*/*.jsonl")
        # Use limit(1) not count() — avoids full scan of 2M records
        if gdelt_raw.limit(1).count() == 0:
            raise RuntimeError("empty lake shards — using direct path")
        print("[spark] GDELT: reading from lake shards (raw/gdelt/)")
        gdelt = gdelt_raw.select(
            F.lit("gdelt").alias("source"),
            F.col("doc_id"),
            F.col("doc_date"),
            F.col("text"),    # pre-built space-joined string 
            F.col("url"),     # real URL field 
        )

    except Exception as e:
        # Path B: direct read from the public GDELT S3 bucket.
        # Files are plain uncompressed .gkg.csv
        print(f"[spark] GDELT lake empty ({e}) — reading direct from gdelt-open-data")
        GKG_COLS = [
            "GKGRECORDID", "DATE", "SourceCollectionIdentifier",
            "SourceCommonName", "DocumentIdentifier", "Counts", "V2Counts",
            "V2Themes", "V2EnhancedThemes", "V2Locations", "V2EnhancedLocations",
            "V2Persons", "V2EnhancedPersons", "V2Organizations",
            "V2EnhancedOrganizations", "V2Tone", "V2EnhancedDates",
            "V2GCAM", "V2SharingImage", "V2RelatedImages", "V2SocialImageEmbeds",
            "V2SocialVideoEmbeds", "V2Quotations", "V2AllNames",
            "V2Amounts", "V2TranslationInfo", "V2Extras",
        ]
        lines = spark.sparkContext.textFile(
            # Corrected extension: .gkg.csv not .gkg.csv.zip
            "s3://gdelt-open-data/v2/gkg/201[5-9]*.gkg.csv,"
            "s3://gdelt-open-data/v2/gkg/202[0-5]*.gkg.csv"
        )
        parts  = lines.map(lambda l: l.split("\t"))
        wide   = parts.filter(lambda r: len(r) >= 16)
        # Build a named DataFrame from the first 27 columns
        gdelt_rdd = wide.map(
            lambda r: (r[0], r[1], r[3], r[4], r[7], r[11], r[13])
        )
        gdelt_raw2 = spark.createDataFrame(
            gdelt_rdd,
            ["GKGRECORDID", "DATE", "SourceCommonName",
             "DocumentIdentifier", "V2Themes", "V2Persons", "V2Organizations"]
        )
        gdelt = gdelt_raw2.select(
            F.lit("gdelt").alias("source"),
            F.col("GKGRECORDID").alias("doc_id"),
            F.to_date(F.col("DATE").substr(1, 8), "yyyyMMdd")
             .cast("string").alias("doc_date"),
            F.concat_ws(" ",
                F.coalesce(F.col("V2Themes"),        F.lit("")),
                F.coalesce(F.col("V2Persons"),       F.lit("")),
                F.coalesce(F.col("V2Organizations"), F.lit(""))
            ).alias("text"),
            F.col("DocumentIdentifier").alias("url"),
        )

    docs = docs.unionByName(gdelt)
    # Filter out nulls and empty text. Threshold is 20 chars (not 100) so
    # short GDELT records (themes+orgs+persons can be brief) are kept.
    return docs.filter(
        F.col("text").isNotNull() & (F.length(F.col("text")) > 20)
    )

# ML Pipeline: tokenise → strip stopwords → bigram → TF → IDF
def build_text_pipeline(normalize=False):
    """
    5-stage pipeline (from pyspark_ml.ipynb course notebook):
      RegexTokenizer → StopWordsRemover → NGram → HashingTF → IDF
    """
    tokenizer = RegexTokenizer(
        inputCol="text", outputCol="raw_tokens",
        pattern=r"[^a-zA-Z]+", minTokenLength=3, toLowercase=True,
    )
    # English stopwords plus FOMC boilerplate that would dominate TF-IDF
    stop = StopWordsRemover(
        inputCol="raw_tokens", outputCol="tokens",
        stopWords=StopWordsRemover.loadDefaultStopWords("english") + [
            "committee", "meeting", "participants", "members", "staff",
            "federal", "reserve", "open", "market", "monetary", "policy",
            "would", "could", "noted", "indicated", "discussed",
            "also", "including", "reported", "according",
        ],
    )
    bigram = NGram(n=2, inputCol="tokens", outputCol="bigrams")
    tf     = HashingTF(
        inputCol="tokens", outputCol="rawFeatures",
        numFeatures=1 << 18,   # 262,144 — matches 11_cupy_similarity.py
    )
    idf = IDF(inputCol="rawFeatures", outputCol="tfidf_raw" if normalize else "features",
              minDocFreq=5)
    stages = [tokenizer, stop, bigram, tf, idf]
    if normalize:
        stop.setLocale("en_US")
        stages.append(Normalizer(inputCol="tfidf_raw", outputCol="features", p=2.0))
    return Pipeline(stages=stages)


def read_versioned_sources(spark, args):
    # The broad partition already includes the direct records. Reading both would duplicate them.
    gdelt = (spark.read.schema(VERSIONED_SCHEMA)
             .option("recursiveFileLookup", "true").option("pathGlobFilter", "*.jsonl")
             .json(f"{args.gdelt_path.rstrip('/')}/broad_macro"))
    invalid = (F.col("source").isNull() | (F.col("source") != "gdelt")
               | F.col("metadata_text").isNull() | F.col("source_domain").isNull()
               | F.col("representation_type").isNull() | F.col("filter_version").isNull()
               | F.col("direct_fed").isNull() | F.col("broad_macro").isNull()
               | ~F.col("broad_macro"))
    if gdelt.filter(invalid).limit(1).count():
        raise ValueError("GDELT input must be the versioned broad_macro partition")
    versions = gdelt.select("filter_version").distinct().take(2)
    if len(versions) != 1:
        raise ValueError("GDELT input must contain records from one filter version")

    docs = gdelt.withColumn("text", F.col("metadata_text"))
    for source, path, domain, representation in (
        ("fomc", args.fomc_path, "federalreserve.gov", "fomc_minutes"),
        ("nyt", args.nyt_path, "nytimes.com", "nyt_api_text"),
        ("guardian", args.guardian_path, "theguardian.com", "guardian_article_body"),
    ):
        if path:
            other = (spark.read.schema(SCHEMA_DOC).json(path)
                     .withColumn("source", F.lit(source))
                     .withColumn("source_domain", F.lit(domain))
                     .withColumn("representation_type", F.lit(representation)))
            docs = docs.unionByName(other, allowMissingColumns=True)

    docs = docs.withColumn("doc_date", F.to_date("doc_date"))
    if docs.filter(F.col("doc_date").isNull() | F.col("doc_id").isNull()).limit(1).count():
        raise ValueError("Every document needs a valid date and ID")
    docs = docs.filter(F.length(F.trim("text")) > 0).dropDuplicates(["source", "doc_id"])
    return docs, versions[0].filter_version


def versioned_main():
    parser = argparse.ArgumentParser(description="Fit discovery TF-IDF and transform a versioned run")
    parser.add_argument("--fomc-path", required=True)
    parser.add_argument("--gdelt-path", required=True, help="Ingestion output root containing broad_macro")
    parser.add_argument("--nyt-path")
    parser.add_argument("--guardian-path")
    parser.add_argument("--tokens-out")
    parser.add_argument("--tfidf-out", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--fit-start", type=date.fromisoformat, default=date(2015, 3, 15))
    parser.add_argument("--fit-cutoff", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    if args.fit_start > args.fit_cutoff:
        parser.error("--fit-start must be on or before --fit-cutoff")

    spark = SparkSession.builder.appName("fomc_versioned_tfidf").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        docs, filter_version = read_versioned_sources(spark, args)
        docs = docs.cache()
        discovery = docs.filter(
            F.col("doc_date").between(args.fit_start, args.fit_cutoff)
            & ((F.col("source") == "fomc") | ((F.col("source") == "gdelt") & F.col("direct_fed")))
        )
        fit_counts = {row.source: row["count"] for row in discovery.groupBy("source").count().collect()}
        if set(fit_counts) != {"fomc", "gdelt"}:
            raise ValueError("Discovery fit requires both FOMC minutes and direct-Fed GDELT records")
        model = build_text_pipeline(normalize=True).fit(discovery)
        model.write().overwrite().save(args.model_out)
        fit_info = dict(fit_start=str(args.fit_start), fit_cutoff=str(args.fit_cutoff),
                        n_fomc=fit_counts["fomc"], n_gdelt=fit_counts["gdelt"],
                        filter_version=filter_version,
                        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        spark.createDataFrame([fit_info]).coalesce(1).write.mode("overwrite").json(
            f"{args.model_out.rstrip('/')}/fit_info")
        features = model.transform(docs).withColumn("model_fit_cutoff", F.lit(args.fit_cutoff))
        if args.tokens_out:
            (features.select("source", "doc_id", "doc_date", "tokens")
             .filter(F.size("tokens") > 10).write.mode("overwrite").parquet(args.tokens_out))
        (features.select("source", "doc_id", "doc_date", "source_domain", "url", "representation_type",
                         "filter_version", "direct_fed", "broad_macro", "gkg_tone", "model_fit_cutoff", "features")
         .write.mode("overwrite").parquet(args.tfidf_out))
        print(f"[tfidf] fit on {fit_counts} through {args.fit_cutoff}")
    finally:
        spark.stop()

# Main
def main(bucket):
    spark = (
        SparkSession.builder
        .appName("makkapakka_preprocess_tfidf")
        # Tuned for m5.large (8 GB RAM, 2 vCPUs per node, 4 core nodes)
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.executor.memory", "5g")
        .config("spark.executor.cores", "2")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.adaptive.enabled", "true")
        # Raise broadcast threshold so Spark auto-broadcasts FOMC (88 rows)
        .config("spark.sql.autoBroadcastJoinThreshold", "50m")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"[spark] reading lake from s3://{bucket}/raw/")
    docs = read_all_sources(spark, bucket).cache()
    n_docs = docs.count()
    n_src  = docs.select("source").distinct().count()
    print(f"[spark] {n_docs:,} documents across {n_src} sources")
    docs.groupBy("source").count().show()

    pipeline = build_text_pipeline()
    print("[spark] fitting TF-IDF pipeline ...")
    model = pipeline.fit(docs)
    feats = model.transform(docs)

    # 1) tokens — for Numba scorer
    (feats
     .select("source", "doc_id", "doc_date", "tokens")
     .filter(F.size("tokens") > 10)
     .write.mode("overwrite")
     .parquet(f"s3://{bucket}/curated/tokens/"))
    print("[spark] wrote curated/tokens/")

    # 2) TF-IDF vectors for cosine similarity
    (feats
     .select("source", "doc_id", "doc_date", "features")
     .write.mode("overwrite")
     .parquet(f"s3://{bucket}/curated/tfidf/"))
    print("[spark] wrote curated/tfidf/")

    # 3) Bigram frequencies for exploratory analysis
    bigram_freq = (
        feats
        .select("source", "doc_date", F.explode("bigrams").alias("bg"))
        .groupBy("source", "doc_date", "bg")
        .agg(F.count("*").alias("n"))
        .filter(F.col("n") >= 3)
    )
    bigram_freq.write.mode("overwrite").parquet(
        f"s3://{bucket}/curated/ngrams/bigrams.parquet"
    )

    # Top-100 bigrams per source
    from pyspark.sql.window import Window
    w = Window.partitionBy("source").orderBy(F.desc("total"))
    (bigram_freq
     .groupBy("source", "bg").agg(F.sum("n").alias("total"))
     .withColumn("rk", F.row_number().over(w))
     .filter(F.col("rk") <= 100)
     .write.mode("overwrite")
     .parquet(f"s3://{bucket}/curated/ngrams/top100.parquet"))
    print("[spark] wrote curated/ngrams/")

    spark.stop()
    print("[done]")

if __name__ == "__main__":
    if len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        main(sys.argv[1])
    else:
        versioned_main()
