#!/bin/bash
# Full download queue after the pilot: rest of AI Hub 105, then the useful parts of 81.
# Order: validation first (eval), then one zip of each sign type / book set for variety, then the rest.
# 81: text-in-the-wild scenes and printed glyphs; handwriting and the 140 GB augmentation set are skipped.
#   setsid nohup bash infra/aihub_queue.sh > ~/aihub/queue.log 2>&1 &
# Each fetch reads its script fresh, so deploy a fixed fetcher under a new name while a queue runs.
set -uo pipefail
D=$(dirname "$0")
F=${AIHUB_FETCH:-$HOME/aihub/aihub_fetch.sh}
[ -f "$F" ] || F=$D/aihub_fetch.sh
# wait for an earlier queue (pid in $1) to finish so two downloads never split the home uplink
if [ -n "${1:-}" ]; then while [ -d "/proc/$1" ]; do sleep 60; done; fi
if [ -z "${SKIP_81:-}" ]; then
  bash "$F" 81 67943 67939 67940 67941 67942 255442 255439 255440 255441 || echo "81 queue stopped"
fi
bash "$F" 105 68073 37321 37318 37319 37320 \
  37306 37307 37309 37311 37312 37313 37304 37315 37300 68071 \
  37301 37302 37303 37305 37308 37310 37316 || echo "105 queue stopped"
echo "QUEUE DONE $(date -u +%FT%TZ)"
