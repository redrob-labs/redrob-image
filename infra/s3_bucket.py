#!/usr/bin/env python3
"""Create (or verify) the private training bucket in Seoul.

Usage:
    python3 infra/s3_bucket.py --bucket NAME [--create] [--profile PROFILE]

The bucket name is passed on the command line or via REDROB_IMAGE_BUCKET; it is not
stored in this public repository. Verification fails unless all four public-access
blocks are on, default encryption is set, and the region is ap-northeast-2.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REGION = "ap-northeast-2"


def aws(args: list[str], profile: str | None) -> dict:
    cmd = ["aws", *args, "--region", REGION, "--output", "json"]
    if profile:
        cmd += ["--profile", profile]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return json.loads(out) if out.strip() else {}


def create(bucket: str, profile: str | None) -> None:
    aws(["s3api", "create-bucket", "--bucket", bucket,
         "--create-bucket-configuration", f"LocationConstraint={REGION}"], profile)
    aws(["s3api", "put-public-access-block", "--bucket", bucket,
         "--public-access-block-configuration",
         "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"], profile)
    aws(["s3api", "put-bucket-encryption", "--bucket", bucket,
         "--server-side-encryption-configuration",
         json.dumps({"Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]})], profile)
    # Old checkpoints expire; paper/ and data/ are kept.
    aws(["s3api", "put-bucket-lifecycle-configuration", "--bucket", bucket,
         "--lifecycle-configuration", json.dumps({"Rules": [{
             "ID": "expire-old-checkpoints", "Status": "Enabled",
             "Filter": {"Prefix": "checkpoints/"},
             "Expiration": {"Days": 30}}]})], profile)


def verify(bucket: str, profile: str | None) -> list[str]:
    problems = []
    loc = aws(["s3api", "get-bucket-location", "--bucket", bucket], profile).get("LocationConstraint")
    if loc != REGION:
        problems.append(f"region is {loc}, want {REGION}")
    pab = aws(["s3api", "get-public-access-block", "--bucket", bucket], profile)
    conf = pab.get("PublicAccessBlockConfiguration", {})
    for key in ("BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets"):
        if not conf.get(key):
            problems.append(f"{key} is off")
    try:
        aws(["s3api", "get-bucket-encryption", "--bucket", bucket], profile)
    except subprocess.CalledProcessError:
        problems.append("default encryption is off")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=os.environ.get("REDROB_IMAGE_BUCKET"))
    ap.add_argument("--profile")
    ap.add_argument("--create", action="store_true")
    a = ap.parse_args()
    if not a.bucket:
        ap.error("--bucket or REDROB_IMAGE_BUCKET is required")
    if a.create:
        create(a.bucket, a.profile)
    problems = verify(a.bucket, a.profile)
    for p in problems:
        print(f"s3_bucket: {p}", file=sys.stderr)
    print("ok" if not problems else "FAILED")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
