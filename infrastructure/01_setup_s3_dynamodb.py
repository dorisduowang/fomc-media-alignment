"""
MACS 30113
Yuzhao (Jerry) Jin; CNETID: yuzhaojin1
Duo (Doris) Wang; CNETID: dorisduowang

This file is for provision the S3 data-lake bucket and the DynamoDB catalog table.

Utilizing the following course contents: 
- Week 4 & 5: Boto3; S3 data-lake architectures
- Week 5: DynamoDB as a low-latency keyed lookup store
"""
import argparse
import json
import os
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
ENV_BUCKET = os.environ.get("BUCKET", "")
ENV_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

PREFIXES = [
    "raw/fomc/", "raw/nyt/", "raw/guardian/", "raw/gdelt/",
    "curated/tokens/", "curated/tfidf/", "curated/ngrams/",
    "processed/sentiment/", "processed/similarity/",
    "processed/lead_lag/", "logs/",
]

def create_bucket(s3, bucket, region):
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        print(f"[s3] created bucket {bucket} in {region}")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            print(f"[s3] bucket {bucket} already exists; reusing")
        else:
            raise

    for p in PREFIXES:
        s3.put_object(Bucket=bucket, Key=p)
        print(f"[s3]   prefix {p}")

def create_catalog_table(dynamodb, table_name):
    try:
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "source", "KeyType": "HASH"},
                {"AttributeName": "doc_id",  "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "source",   "AttributeType": "S"},
                {"AttributeName": "doc_id",   "AttributeType": "S"},
                {"AttributeName": "doc_date", "AttributeType": "S"},
            ],
            GlobalSecondaryIndexes=[{
                "IndexName": "by_date",
                "KeySchema": [
                    {"AttributeName": "source",   "KeyType": "HASH"},
                    {"AttributeName": "doc_date", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
                "ProvisionedThroughput": {
                    "ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
            }],
            ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
        )
        table.meta.client.get_waiter("table_exists").wait(TableName=table_name)
        print(f"[dynamodb] created {table_name}")
    except dynamodb.meta.client.exceptions.ResourceInUseException:
        print(f"[dynamodb] table {table_name} already exists; reusing")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default=ENV_BUCKET,
                   help="S3 bucket name (default: BUCKET from .env)")
    p.add_argument("--table",  default="makkapakka_catalog")
    p.add_argument("--region", default=ENV_REGION)
    args = p.parse_args()

    if not args.bucket:
        p.error("--bucket is required, or set BUCKET in your .env file")

    s3       = boto3.client("s3", region_name=args.region)
    dynamodb = boto3.resource("dynamodb", region_name=args.region)

    create_bucket(s3, args.bucket, args.region)
    create_catalog_table(dynamodb, args.table)

    s3.put_object(
        Bucket=args.bucket, Key="config/lake_info.json",
        Body=json.dumps({
            "bucket": args.bucket, "catalog_table": args.table,
            "region": args.region, "created_at": int(time.time()),
        }).encode("utf-8"),
    )
    print("[done] infrastructure ready")

if __name__ == "__main__":
    main()