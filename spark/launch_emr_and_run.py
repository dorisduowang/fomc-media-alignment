"""
MACS 30113
Yuzhao (Jerry) Jin; CNETID: yuzhaojin1
Duo (Doris) Wang; CNETID: dorisduowang

This file builds a one-command driver that
    1. uploads the Spark scripts to s3://<bucket>/code/
    2. launches a Spark-enabled EMR cluster (using the team's existing
       launch_spark_cluster.py helper from the lab),
    3. adds the two preprocess + meeting-window steps to the cluster,
    4. waits for completion and terminates the cluster.

Utilizing course contenst:
- Week 7-8: Spark on EMR end-to-end automation
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
ENV_BUCKET = os.environ.get("BUCKET", "")

SPARK_SCRIPTS = [
    ("spark/08_spark_preprocess_tfidf.py",  "preprocess+tfidf"),
    ("spark/09_spark_meeting_windows.py",   "meeting windows"),
]

def upload_code(s3, bucket, root):
    """Ship every Spark script to s3://<bucket>/code/."""
    keys = []
    for rel, _ in SPARK_SCRIPTS:
        full = os.path.join(root, rel)
        key = "code/" + os.path.basename(rel)
        s3.upload_file(full, bucket, key)
        keys.append(key)
        print(f"[s3] uploaded code/{os.path.basename(rel)}")
    return keys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default=ENV_BUCKET,
                   help="S3 lake bucket (default: from .env BUCKET)")
    p.add_argument("--cores", type=int, default=4,
                   help="core node count using m5.large (2 vCPUs each, default 4)")
    p.add_argument("--instance-type", type=str, default="m4.large",
                   help="EC2 instance type — Vocareum only supports up to 'large'. "
                        "m5.large is NOT supported in this environment; "
                        "use m4.large (Vocareum README tip) (default: m4.large, 2 vCPUs, 8 GB)")
    p.add_argument("--keep-alive", action="store_true",
                   help="keep cluster up after steps finish (debugging)")
    args = p.parse_args()

    if not args.bucket:
        p.error("--bucket is required (or set BUCKET in your .env file)")

    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, ".."))

    s3 = boto3.client("s3")
    upload_code(s3, args.bucket, project_root)

    # Fixed based on Vocareum README: supported instance types are 
    # nano/micro/small/medium/large ONLY.

    # m5.large raises ValidationException: "Instance type 'm5.large' is not supported"
    # in this Vocareum environment. The README tip says: "try using the m4.large
    # instance type." m4.large has identical specs: 2 vCPUs, 8 GB RAM.

    # 1 primary + 4 core = 5 instances × 2 vCPUs = 10 vCPUs (under 32 limit)
    # 5 instances total (under 9 limit)
    cmd = [
        "python", os.path.join(project_root, "spark", "launch_spark_cluster.py"),
        "--s3_bucket", args.bucket,
        "--primary_count", "1",
        "--core_count", str(args.cores),
        "--instance_type", args.instance_type,
        "--cluster_name", "makkapakka-spark",
    ]
    print("[emr] launching cluster ...")
    subprocess.check_call(cmd)

    # Find the cluster ID we just launched (most recent STARTING/WAITING/RUNNING)
    emr = boto3.client("emr")
    listing = emr.list_clusters(
        ClusterStates=["STARTING", "BOOTSTRAPPING", "RUNNING", "WAITING"]
    )["Clusters"]
    listing = [c for c in listing if c["Name"] == "makkapakka-spark"]
    cluster_id = listing[0]["Id"]
    print(f"[emr] cluster {cluster_id}")

    # Submit both Spark steps
    steps = []
    for i, (rel, name) in enumerate(SPARK_SCRIPTS):
        steps.append({
            "Name": name,
            "ActionOnFailure": "CANCEL_AND_WAIT",
            "HadoopJarStep": {
                "Jar": "command-runner.jar",
                "Args": [
                    "spark-submit",
                    "--deploy-mode", "cluster",
                    "--executor-memory", "4g",
                    "--executor-cores", "2",
                    "--driver-memory",  "4g",
                    f"s3://{args.bucket}/code/{os.path.basename(rel)}",
                    args.bucket,
                ],
            },
        })

    resp = emr.add_job_flow_steps(JobFlowId=cluster_id, Steps=steps)
    step_ids = resp["StepIds"]
    print(f"[emr] submitted steps {step_ids}")

    # Block on each step in turn.
    from botocore.waiter import WaiterError
    waiter_cfg = {"Delay": 60, "MaxAttempts": 180}   # 180 min max

    for sid in step_ids:
        print(f"[emr] waiting on step {sid} (up to 3 hours) ...")
        try:
            emr.get_waiter("step_complete").wait(
                ClusterId=cluster_id,
                StepId=sid,
                WaiterConfig=waiter_cfg,
            )
        except WaiterError:
            # Waiter ran out of time — check actual step state
            pass

        s     = emr.describe_step(ClusterId=cluster_id, StepId=sid)["Step"]
        state = s["Status"]["State"]
        print(f"[emr]   {sid} {state}")

        if state == "FAILED":
            reason = (s["Status"].get("StateChangeReason") or {}).get("Message", "")
            print(f"[emr] Step FAILED: {reason}")
            if not args.keep_alive:
                print("[emr] terminating cluster after failure")
                emr.terminate_job_flows(JobFlowIds=[cluster_id])
            raise SystemExit(f"Step {sid} FAILED — see EMR console for logs.")

        if state not in ("COMPLETED",):
            print(f"[emr] Step still in state {state} after 3 hours — "
                  f"check EMR console. Cluster NOT terminated.")
            raise SystemExit(f"Step {sid} did not complete within timeout.")

    if not args.keep_alive:
        print("[emr] terminating cluster")
        emr.terminate_job_flows(JobFlowIds=[cluster_id])


if __name__ == "__main__":
    main()
