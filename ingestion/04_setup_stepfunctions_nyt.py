"""
MACS 30113
Yuzhao (Jerry) Jin; CNETID: yuzhaojin1
Duo (Doris) Wang; CNETID: dorisduowang

This is file is a tryout for deploying the NYT-fetch Lambda and an EXPRESS 
Step Functions state machine that scatters (year, month) tasks across 
concurrent Lambda workers. But this file eventually failed due to too frequent
requests as specified in the 04_ingest_nyt.py

The NYT Article Search API free tier allows 500 requests/day per key
(= 5,000 articles/day).  With 120 month-batches needing up to 10 pages
each (1,200 requests total) a single key is insufficient for a one-session
run. We assign --nyt-keys (comma-separated list) and assign keys to tasks 
by round-robin so each key handles a separate slice:
    Key 1 -> months 1, 4, 7, ...   (40 tasks -> 400 requests max)
    Key 2 -> months 2, 5, 8, ...   (40 tasks -> 400 requests max)
    Key 3 -> months 3, 6, 9, ...   (40 tasks -> 400 requests max)
Each key stays well under its 500-request/day limit.

The API key is now passed PER TASK inside the Step Functions event payload
rather than as a Lambda environment variable, so each Lambda invocation
uses exactly the key assigned to its (year, month) batch.

Utilizing course contents: 
- Week 4: serverless Lambda via Boto3
- Week 6: Step Functions Map state — scatter-gather pattern from
          step_functions.ipynb applied to a time-series API batch
"""

import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
ENV_BUCKET   = os.environ.get("BUCKET", "")
_nyt_raw     = os.environ.get("NYT_KEYS", "")
ENV_NYT_KEYS = [k.strip() for k in _nyt_raw.split(",") if k.strip()]

def _lambda_call_with_retry(fn, *args, max_retries=5, **kwargs):
    """
    Call a Lambda API function with exponential backoff on 429.
    The Lambda control-plane API (create, update, put_concurrency) has its
    own rate limit separate from function invocations.  Calling several
    Lambda API operations in quick succession triggers TooManyRequestsException.
    """
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except boto3.client("lambda").exceptions.TooManyRequestsException:
            wait = 2 ** attempt
            print(f"[lambda] 429 rate limit on API call; sleeping {wait}s ...")
            time.sleep(wait)
        except Exception as e:
            # Re-raise non-rate-limit errors immediately
            raise e
    raise RuntimeError(f"Lambda API still rate-limited after {max_retries} retries")

def package_handler(handler_path):
    """Pack the handler module into a Lambda zip in memory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        with open(handler_path) as f:
            z.writestr("lambda_function.py", f.read())
    buf.seek(0)
    return buf.read()

def upsert_lambda(client, role_arn, name, env, zip_bytes):
    try:
        client.create_function(
            FunctionName=name,
            Runtime="python3.10",
            Role=role_arn,
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": zip_bytes},
            Timeout=300,
            MemorySize=512,
            Environment={"Variables": env},
        )
        print(f"[lambda] created {name}")
    except client.exceptions.ResourceConflictException:
        # Update existing function — wrap each call with backoff retry
        # because rapid successive Lambda API calls trigger 429
        for attempt in range(5):
            try:
                client.update_function_code(FunctionName=name, ZipFile=zip_bytes)
                break
            except Exception as e:
                if "TooManyRequests" in str(e):
                    wait = 2 ** attempt
                    print(f"[lambda] 429 on update_function_code; sleep {wait}s")
                    time.sleep(wait)
                else:
                    raise
        client.get_waiter("function_updated_v2").wait(FunctionName=name)
        for attempt in range(5):
            try:
                client.update_function_configuration(
                    FunctionName=name, Environment={"Variables": env})
                break
            except Exception as e:
                if "TooManyRequests" in str(e):
                    wait = 2 ** attempt
                    print(f"[lambda] 429 on update_function_configuration; sleep {wait}s")
                    time.sleep(wait)
                else:
                    raise
        print(f"[lambda] updated {name}")
    client.get_waiter("function_active_v2").wait(FunctionName=name)
    return client.get_function(FunctionName=name)["Configuration"]["FunctionArn"]

def make_sm_def(lambda_arn):
    """
    Map state — one Lambda per (year, month) batch.

    MaxConcurrency reduced from 4 to 2:
      With MaxConcurrency=4, Step Functions dispatched Lambda invocations
      faster than the Vocareum account could handle, causing persistent
      TooManyRequestsException 429 errors that exhausted retries and
      failed the entire execution (status=FAILED in 75.3 seconds).
      MaxConcurrency=2 limits to 2 NYT + 2 catalog = 4 total concurrent
      Lambda invocations — well under the 10-instance Vocareum limit.

    Retry policy strengthened:
      IntervalSeconds: 2 -> 15  (more breathing room between retries)
      MaxAttempts: 3 -> 5   (more chances before giving up on a task)
    """
    return {
        "Comment": "Parallel NYT ingestion — one Lambda per year-month batch",
        "StartAt": "Map",
        "States": {
            "Map": {
                "Type": "Map",
                "End": True,
                "MaxConcurrency": 2,    # reduced from 4 to avoid Lambda 429 rate limit
                "Iterator": {
                    "StartAt": "Invoke",
                    "States": {
                        "Invoke": {
                            "Type": "Task",
                            "Resource": "arn:aws:states:::lambda:invoke",
                            "OutputPath": "$.Payload",
                            "Parameters": {
                                "Payload.$": "$",
                                "FunctionName": lambda_arn,
                            },
                            "Retry": [{
                                "ErrorEquals": [
                                    "Lambda.ServiceException",
                                    "Lambda.AWSLambdaException",
                                    "Lambda.SdkClientException",
                                    "Lambda.TooManyRequestsException",
                                    "States.TaskFailed",
                                    "Lambda.Unknown",
                                ],
                                "IntervalSeconds": 15,   # increased from 2
                                "MaxAttempts":     5,    # increased from 3
                                "BackoffRate":     2,
                            }],
                            "End": True,
                        }
                    }
                }
            }
        }
    }

def build_task_list(start_year, end_year, keys):
    """
    Build the (year, month) task list with round-robin key assignment.

    With 3 keys and 120 tasks the distribution is:
        Key 0 -> tasks 0, 3, 6, ...  (40 tasks -> <=400 requests)
        Key 1 -> tasks 1, 4, 7, ...  (40 tasks -> <=400 requests)
        Key 2 -> tasks 2, 5, 8, ...  (40 tasks -> <=400 requests)
    Each key stays under the 500-request/day free-tier limit.
    """
    pairs = [
        (y, m)
        for y in range(start_year, end_year + 1)
        for m in range(1, 13)
    ]
    tasks = [
        {
            "year": y,
            "month": m,
            "nyt_api_key": keys[i % len(keys)],   # round-robin assignment
        }
        for i, (y, m) in enumerate(pairs)
    ]
    # Print key distribution for transparency
    for ki, k in enumerate(keys):
        count = sum(1 for t in tasks if t["nyt_api_key"] == k)
        print(f"[keys]   key {ki+1} ({k[:8]}...) -> {count} tasks "
              f"(<= {count * 10} requests, limit 500/day)")
    return tasks

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default=ENV_BUCKET,
                   help="S3 lake bucket (default: from .env BUCKET)")
    p.add_argument("--nyt-keys", default=",".join(ENV_NYT_KEYS),
                   help="Comma-separated NYT API keys (default: from .env NYT_KEYS)")
    p.add_argument("--start-year", type=int, default=2015)
    p.add_argument("--end-year",   type=int, default=2025)
    p.add_argument("--function",   default="makkapakka_nyt_fetch")
    p.add_argument("--state-machine", default="makkapakka_nyt_sm")
    args = p.parse_args()

    # Validate required values
    if not args.bucket:
        p.error("--bucket is required (or set BUCKET in .env)")
    keys = [k.strip() for k in args.nyt_keys.split(",") if k.strip()]
    if not keys:
        p.error("--nyt-keys is required (or set NYT_KEYS in .env)")

    print(f"[nyt] bucket : {args.bucket}")
    print(f"[nyt] keys   : {len(keys)} key(s) provided")
    print(f"[nyt] period : {args.start_year}–{args.end_year}")

    iam   = boto3.client("iam")
    aws_l = boto3.client("lambda")
    sfn   = boto3.client("stepfunctions")

    role_arn = iam.get_role(RoleName="LabRole")["Role"]["Arn"]
    handler_path = os.path.join(
        os.path.dirname(__file__), "lambda_packages", "nyt_fetch_handler.py")
    zip_bytes = package_handler(handler_path)

    # Deploy Lambda.
    # Each task carries its own key in the event payload.
    lambda_arn = upsert_lambda(
        aws_l, role_arn, args.function,
        env={"BUCKET": args.bucket, "NYT_QUERY": "Federal Reserve"},
        zip_bytes=zip_bytes,
    )

    # Reduced to 2: MaxConcurrency=2 means at most 2 NYT Lambda invocations
    # at once. 2 NYT + 2 catalog Lambda (triggered by S3 PUTs) = 4 total
    # concurrent — avoids the Lambda invocation rate limit that caused 429
    # TooManyRequestsException failures at MaxConcurrency=4.
    time.sleep(3)
    for attempt in range(5):
        try:
            aws_l.put_function_concurrency(
                FunctionName=args.function, ReservedConcurrentExecutions=2)
            break
        except Exception as e:
            if "TooManyRequests" in str(e):
                wait = 2 ** attempt
                print(f"[lambda] 429 on put_function_concurrency; sleep {wait}s")
                time.sleep(wait)
            else:
                raise

    # Create / update the Step Functions state machine
    sm_def = make_sm_def(lambda_arn)
    try:
        sm_arn = sfn.create_state_machine(
            name=args.state_machine,
            definition=json.dumps(sm_def),
            roleArn=role_arn,
            type="EXPRESS",
        )["stateMachineArn"]
    except sfn.exceptions.StateMachineAlreadyExists:
        listing = sfn.list_state_machines()["stateMachines"]
        sm_arn = next(sm["stateMachineArn"] for sm in listing
                      if sm["name"] == args.state_machine)
        sfn.update_state_machine(
            stateMachineArn=sm_arn,
            definition=json.dumps(sm_def),
            roleArn=role_arn,
        )
    print(f"[sfn] state machine {sm_arn}")

    # Build task list with round-robin key assignment
    tasks = build_task_list(args.start_year, args.end_year, keys)
    print(f"[sfn] query          : Federal Reserve")
    print(f"[sfn] fq filter      : section.name:\"Business\"")
    print(f"[sfn] scattering {len(tasks)} tasks across <=2 concurrent Lambdas")
    print(f"[sfn] Vocareum check : 2 NYT + 2 catalog = 4 concurrent < 10 limit")

    t0 = time.time()
    resp = sfn.start_sync_execution(
        stateMachineArn=sm_arn,
        name=f"nyt_run_{int(t0)}",
        input=json.dumps(tasks),
    )
    elapsed = time.time() - t0
    print(f"[sfn] done in {elapsed:.1f}s   status={resp.get('status')}")
    if resp.get("status") != "SUCCEEDED":
        print(resp.get("cause"))
        sys.exit(1)

    out   = json.loads(resp["output"])
    total = sum(r.get("written", 0) for r in out)
    print(f"[done] wrote {total} NYT articles to s3://{args.bucket}/raw/nyt/")

if __name__ == "__main__":
    main()