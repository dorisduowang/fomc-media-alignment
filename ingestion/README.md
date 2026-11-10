# GDELT archive ingestion

The acquisition script consumes the official GDELT HTTPS shard plan, checks compressed size and MD5, and records shard status for restartable processing. The manifest counts malformed rows separately. An archive with zero valid rows fails the shard gate. The rule JSON records the direct-Fed and broad-macro filters.

From the repository root, run `python -m pytest -q tests`. The tests use simulated HTTP and S3 for acquisition and synthetic market inputs for volatility. The [extension notebook](../MakkaPakka%20update/Jerry_GDELT_2015_2025_workflow.ipynb) also documents the S3 and EMR paths.

The [handoff directory](../MakkaPakka%20update/handoff/gdelt_2015_2025_q1_v1/) records the pilot plan, aggregate coverage, volatility audit, and original full-plan checksum. The notebook can regenerate the full plan from GDELT's public index. The code hashes identify the original handoff files before public packaging, and the package inventory has local paths removed. Use the Git commit to identify the public source.

## Local three-day pilot

Set `PILOT_DIR` to a dedicated local directory outside the repository. Stage the frozen FOMC minutes under `$PILOT_DIR/raw/fomc/` as one JSON file per meeting, with `doc_id`, `doc_date` in ISO format, `text`, and `url`, as shown in the notebook. Use the project's Python environment and a working Java installation.

```bash
PLAN='MakkaPakka update/handoff/gdelt_2015_2025_q1_v1/gdelt_pilot_plan.csv'
python ingestion/06_ingest_gdelt.py --local-root "$PILOT_DIR" --plan "$PLAN" \
  --out-prefix raw/gdelt --manifest-prefix manifests/gdelt --workers 6 --resume

spark-submit --master 'local[4]' spark/08_spark_preprocess_tfidf.py \
  --gdelt-path "$PILOT_DIR/raw/gdelt" --fomc-path "$PILOT_DIR/raw/fomc/*.json" \
  --tfidf-out "$PILOT_DIR/curated/tfidf" --model-out "$PILOT_DIR/models/tfidf" \
  --fit-cutoff 2019-03-23
spark-submit --master 'local[4]' spark/09_spark_meeting_windows.py \
  --tfidf "$PILOT_DIR/curated/tfidf" \
  --meetings 'MakkaPakka update/data_reference/meetings_all88.csv' \
  --output "$PILOT_DIR/curated/meeting_windows" --max-lag 7
spark-submit --master 'local[4]' 'MakkaPakka update/scripts/similarity/11_spark_similarity.py' \
  --windows "$PILOT_DIR/curated/meeting_windows" --tfidf "$PILOT_DIR/curated/tfidf" \
  --output "$PILOT_DIR/processed/per_meeting"
```

Check `$PILOT_DIR/manifests/gdelt/run_manifest.jsonl` for 288 distinct shards and successful MD5 checks before proceeding to Spark. The plan covers March 18, 2019, March 13, 2020, and December 8, 2025 in Eastern time. Spark reads `broad_macro` once and derives the direct-Fed subset from its flags. The model uses L2-normalized TF-IDF fitted on direct-Fed GDELT records and FOMC minutes from March 15, 2015 through March 23, 2019. The output keeps both article and domain-equal weighting, along with document, URL, domain, and native-tone summaries. Pilot fits reflect the pilot's limited discovery coverage. The notebook describes the separate manual relevance review.
