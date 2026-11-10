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
import sys
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, NGram, HashingTF, IDF

# Unified schema — only the 5 fields every source shares
SCHEMA_DOC = T.StructType([
    T.StructField("source",   T.StringType(), True),
    T.StructField("doc_id",   T.StringType(), True),
    T.StructField("doc_date", T.StringType(), True),
    T.StructField("text",     T.StringType(), True),
    T.StructField("url",      T.StringType(), True),
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
def build_text_pipeline():
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
    idf    = IDF(inputCol="rawFeatures", outputCol="features",
                 minDocFreq=5)
    return Pipeline(stages=[tokenizer, stop, bigram, tf, idf])

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
    if len(sys.argv) < 2:
        raise SystemExit("Usage: 08_spark_preprocess_tfidf.py <s3-bucket>")
    main(sys.argv[1])
