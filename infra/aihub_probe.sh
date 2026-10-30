#!/bin/bash
# List AI Hub datasets and probe whether this key may download each (first ~64 KiB only).
#   bash infra/aihub_probe.sh <datasetkey...>
# Writes ~/aihub/<ds>/files.txt (name | size | filekey). Prints the HTTP code of a probe on the smallest file.
set -uo pipefail
ROOT=${AIHUB_ROOT:-$HOME/aihub}
. "$ROOT/.key.env"
for DS in "$@"; do
  mkdir -p "$ROOT/$DS"
  bash "$ROOT/aihubshell" -mode l -datasetkey "$DS" 2>/dev/null | grep ' | ' | sed 's/^[ │├└─]*//' > "$ROOT/$DS/files.txt"
  n=$(wc -l < "$ROOT/$DS/files.txt")
  small=$(awk -F' \\| ' '{s=$2; v=s+0; if (s ~ /GB/) v*=1024; else if (s ~ /KB/) v/=1024; print v"\t"$3}' "$ROOT/$DS/files.txt" | sort -n | head -1 | cut -f2)
  code=$(curl -sS -L -r 0-65535 -m 30 -o "$ROOT/$DS/.probe" -H "apikey:$AIHUB_APIKEY" -w '%{http_code}' \
         "https://api.aihub.or.kr/down/0.6/$DS.do?fileSn=$small" 2>/dev/null || true)
  echo "$DS files=$n probe_filekey=$small http=$code body=$(head -c 120 "$ROOT/$DS/.probe" | tr -d '\0' | tr '\n' ' ' | LC_ALL=C sed 's/[^[:print:]]//g' | head -c 120)"
  rm -f "$ROOT/$DS/.probe"
done
