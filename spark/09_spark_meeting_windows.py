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

import sys
import numpy as np
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors, VectorUDT

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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: 09_spark_meeting_windows.py <s3-bucket>")
    main(sys.argv[1])