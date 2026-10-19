#!/bin/bash
# Download AI Hub datasets one file at a time with the official aihubshell (resumable, logged).
#   bash infra/aihub_download.sh <datasetkey> [filekey ...]   (no filekeys = every file in the tree)
# Needs: ~/aihub/.key.env (export AIHUB_APIKEY=..., mode 600), ~/aihub/aihubshell, approval on aihub.or.kr.
# AI Hub terms: data must stay on servers in Korea (ap-northeast-2). Never into git or public storage.
set -euo pipefail
DS=$1; shift
ROOT=${AIHUB_ROOT:-$HOME/aihub}
SHELL_BIN=$ROOT/aihubshell
. "$ROOT/.key.env"
mkdir -p "$ROOT/$DS/.done"
cd "$ROOT/$DS"
if [ $# -eq 0 ]; then
  # filekeys = last field of "name | size | key" rows; labels (small) first
  mapfile -t KEYS < <(bash "$SHELL_BIN" -mode l -datasetkey "$DS" | awk -F'|' 'NF==3{gsub(/ /,"",$3); s=$2; print (s ~ /MB|KB/ ? 0 : 1), $3}' | sort -k1,1n -s | awk '{print $2}')
else
  KEYS=("$@")
fi
echo "dataset $DS: ${#KEYS[@]} files"
for k in "${KEYS[@]}"; do
  [ -f ".done/$k" ] && { echo "skip $k"; continue; }
  echo "== $k $(date -u +%FT%TZ)"
  out=$(bash "$SHELL_BIN" -mode d -datasetkey "$DS" -filekey "$k" 2>&1 | grep -v -E '^ *[0-9]+ +[0-9.]+[kMG]? ' | tail -5)
  echo "$out"
  if grep -q '병합이 완료' <<<"$out"; then touch ".done/$k"; else echo "FAILED $k"; exit 1; fi
done
echo "ALL DONE $DS $(date -u +%FT%TZ)"
