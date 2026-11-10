"""
MACS 30113
Yuzhao (Jerry) Jin; CNETID: yuzhaojin1
Duo (Doris) Wang; CNETID: dorisduowang

This file is for wiring S3 PUTs on raw/* into a Lambda function that could write 
a catalog row to DynamoDB and enqueues an SQS message that fans out a 
re-processing job.
Served as a trigger re-runs via AWS Lambda on new S3 uploads. 

Utilizing the following course content: 
- Week 4: serverless Lambda via Boto3
- Week 6: Ingestion (S3 -> Lambda -> SQS -> DynamoDB)
"""

import argparse
import io
import json
import os
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
ENV_BUCKET = os.environ.get("BUCKET", "")


# Imbedded lambda handler
LAMBDA_SOURCE = r'''
import json, os, urllib.parse, boto3, datetime

dynamodb = boto3.resource("dynamodb")
sqs      = boto3.client("sqs")
table    = dynamodb.Table(os.environ["CATALOG_TABLE"])
queue    = os.environ["REPROCESS_QUEUE_URL"]


def lambda_handler(event, context):
    """Catalog every S3 object uploaded under raw/* and fan-out to SQS."""
    written = 0
    for rec in event.get("Records", []):
        bucket = rec["s3"]["bucket"]["name"]
        key    = urllib.parse.unquote_plus(rec["s3"]["object"]["key"])

        # Pattern: raw/<source>/<YYYY-MM-DD>/<doc_id>.<ext>
        parts = key.split("/")
        if len(parts) < 4 or parts[0] != "raw":
            continue
        source, date, fname = parts[1], parts[2], parts[3]
        doc_id = os.path.splitext(fname)[0]

        # Catalog row for join key on FOMC-window analytics.
        table.put_item(Item={
            "source":   source,
            "doc_id":   doc_id,
            "doc_date": date,
            "s3_key":   key,
            "ingested_at": datetime.datetime.utcnow().isoformat(),
            "size_bytes": rec["s3"]["object"].get("size", 0),
        })

        # Continue for dowstream scoring on fresh data
        sqs.send_message(
            QueueUrl=queue,
            MessageBody=json.dumps({
                "source": source, "doc_date": date,
                "doc_id": doc_id, "s3_key": key, "bucket": bucket,
            }),
        )
        written += 1

    return {"cataloged": written}
'''

def build_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("lambda_function.py", LAMBDA_SOURCE)
    buf.seek(0)
    return buf.read()

def upsert_lambda(aws_lambda, role_arn, name, env, zip_bytes):
    try:
        aws_lambda.create_function(
            FunctionName=name,
            Runtime="python3.10",
            Role=role_arn,
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": zip_bytes},
            Timeout=60,
            MemorySize=256,
            Environment={"Variables": env},
        )
        print(f"[lambda] created {name}")
    except aws_lambda.exceptions.ResourceConflictException:
        aws_lambda.update_function_code(FunctionName=name, ZipFile=zip_bytes)
        aws_lambda.get_waiter("function_updated_v2").wait(FunctionName=name)
        aws_lambda.update_function_configuration(
            FunctionName=name, Environment={"Variables": env})
        print(f"[lambda] updated {name}")
    # Final wait on ensuring the function is fully active before registration
    # S3 bucket notification invokation
    aws_lambda.get_waiter("function_active_v2").wait(FunctionName=name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket",   default=ENV_BUCKET,
                   help="S3 lake bucket (default: from .env BUCKET)")
    p.add_argument("--table",    default="makkapakka_catalog")
    p.add_argument("--function", default="makkapakka_s3_catalog")
    p.add_argument("--queue",    default="makkapakka_reprocess")
    args = p.parse_args()

    if not args.bucket:
        p.error("--bucket is required (or set BUCKET in your .env file)")

    iam   = boto3.client("iam")
    s3    = boto3.client("s3")
    sqs   = boto3.client("sqs")
    aws_l = boto3.client("lambda")

    role_arn = iam.get_role(RoleName="LabRole")["Role"]["Arn"]

    # SQS queue downstream of the Lambda
    try:
        queue_url = sqs.create_queue(QueueName=args.queue)["QueueUrl"]
    except sqs.exceptions.QueueNameExists:
        queue_url = sqs.get_queue_url(QueueName=args.queue)["QueueUrl"]
    print(f"[sqs] queue {queue_url}")

    # Lambda function with environmental variables pointing at 
    # each queue with relevant catalog
    zip_bytes = build_zip()
    upsert_lambda(
        aws_l, role_arn, args.function,
        env={"CATALOG_TABLE": args.table, "REPROCESS_QUEUE_URL": queue_url},
        zip_bytes=zip_bytes,
    )
    lambda_arn = aws_l.get_function(FunctionName=args.function)["Configuration"]["FunctionArn"]

    # Specifically cap concurrent invocations at 5 to avoid aws deactivation.
    aws_l.put_function_concurrency(
        FunctionName=args.function, ReservedConcurrentExecutions=5)
    print(f"[lambda] concurrency capped at 5")

    # 3. Allow S3 to invoke this Lambda for our specific bucket
    try:
        aws_l.add_permission(
            FunctionName=args.function,
            StatementId="s3invoke",
            Action="lambda:InvokeFunction",
            Principal="s3.amazonaws.com",
            SourceArn=f"arn:aws:s3:::{args.bucket}",
        )
    except aws_l.exceptions.ResourceConflictException:
        pass

    # Register the S3 with Lambda notification on raw/* prefix
    s3.put_bucket_notification_configuration(
        Bucket=args.bucket,
        NotificationConfiguration={
            "LambdaFunctionConfigurations": [{
                "LambdaFunctionArn": lambda_arn,
                "Events": ["s3:ObjectCreated:*"],
                "Filter": {"Key": {"FilterRules": [{"Name": "prefix", "Value": "raw/"}]}},
            }]
        },
    )
    print("[done] S3 -> Lambda -> SQS/DynamoDB trigger is live")

if __name__ == "__main__":
    main()