#!/usr/bin/env python3
"""Refuse to commit things that must not reach this public repository.

Checks staged files (or the paths given) for: AI Hub data files, large binaries,
AWS account IDs / access keys, and private bucket URIs.

Usage:
    python3 scripts/check_public.py            # staged files
    python3 scripts/check_public.py PATH ...   # explicit paths
Install as a pre-commit hook:
    ln -s ../../scripts/check_public.py .git/hooks/pre-commit
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024
BLOCKED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".zip", ".tar", ".safetensors", ".pt", ".ckpt", ".bin", ".npy", ".npz"}
# Small images under these prefixes are allowed (docs figures, font previews made by us).
IMAGE_ALLOWED_PREFIXES = ("docs/figures/", "synth/fonts/previews/")
PATTERNS = [
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key id"),
    (re.compile(r"(?<![0-9])[0-9]{12}(?![0-9])"), "possible AWS account id"),
    (re.compile(r"aws_secret_access_key", re.I), "AWS secret key name"),
    (re.compile(r"aihub[^\s]*\.(?:zip|tar)", re.I), "AI Hub archive path"),
    (re.compile(r"/(?:aihub_raw|aihub_cache)/", re.I), "AI Hub raw path"),
]
# Files whose numbers are expected (CSV costs, lockfiles) skip the account-id check.
ACCOUNT_ID_EXEMPT = ("paper/cost.csv", "synth/fonts/manifest.json")  # manifest: sha256 hex digests
# Files that name the patterns on purpose.
SELF_EXEMPT = (".gitignore", "scripts/check_public.py")


def staged() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        check=True, capture_output=True, text=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def check(path: str) -> list[str]:
    p = Path(path)
    if not p.is_file():
        return []
    problems = []
    suffix = p.suffix.lower()
    if suffix in BLOCKED_SUFFIXES and not path.startswith(IMAGE_ALLOWED_PREFIXES):
        problems.append(f"{path}: binary/data file type {suffix} is not allowed")
    if p.stat().st_size > MAX_BYTES:
        problems.append(f"{path}: larger than {MAX_BYTES // (1024 * 1024)} MiB")
    if suffix in BLOCKED_SUFFIXES or path in SELF_EXEMPT:
        return problems
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return problems
    for rx, label in PATTERNS:
        if label == "possible AWS account id" and path in ACCOUNT_ID_EXEMPT:
            continue
        if rx.search(text):
            problems.append(f"{path}: {label}")
    return problems


def main(argv: list[str]) -> int:
    paths = argv[1:] or staged()
    problems = [msg for path in paths for msg in check(path)]
    for msg in problems:
        print(f"check_public: {msg}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
