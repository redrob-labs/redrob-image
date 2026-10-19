#!/bin/bash
# Guard against aihubshell's merge bug: it escapes non-ASCII names with printf %q, so its find
# matches nothing, writes a 0-byte zip and then deletes the .part files. Hard-link every part
# as soon as tar creates it, so the delete only drops one name and the data survives.
#   bash infra/aihub_part_guard.sh <dataset dir>
D=$1
SAFE=$D/.parts-safe
mkdir -p "$SAFE"
cd "$D" || exit 1
while true; do
  find . -path ./.parts-safe -prune -o -type f -name '*.part[0-9]*' -links 1 -print0 |
    while IFS= read -r -d '' f; do cp -l --parents -n "$f" "$SAFE/" 2>/dev/null; done
  sleep 2
done
