# GDELT archive ingestion

The acquisition script consumes the official GDELT HTTPS shard plan, checks compressed size and MD5, and records shard status for restartable processing. The rule JSON records the direct-Fed and broad-macro filters. Review its inclusion rules before a confirmatory run.

From the repository root, run `python -m pytest -q tests`. The tests use simulated HTTP and S3 for acquisition and synthetic market inputs for volatility. For cloud preflight and the three-day pilot, use the [extension notebook](../MakkaPakka%20update/Jerry_GDELT_2015_2025_workflow.ipynb) with a limited IAM identity and a run-specific bucket and prefix.

The [handoff directory](../MakkaPakka%20update/handoff/gdelt_2015_2025_q1_v1/) records the pilot plan, aggregate coverage, volatility audit, and original full-plan checksum. The notebook can regenerate the full plan from GDELT's public index. The code hashes identify the original handoff files before public packaging, and the package inventory has local paths removed. Use the Git commit to identify the public source.

The Spark preprocessing, meeting windows, and similarity stages still need an end-to-end integration test with the new schema and run paths before a full run.
