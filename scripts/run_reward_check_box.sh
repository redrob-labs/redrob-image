#!/bin/bash
# Plan C stage 0 on the gpu-box: one-jamo right/wrong pairs, then rank them with the reward.
#   setsid nohup bash scripts/run_reward_check_box.sh > ~/runs/reward-check.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$HOME/hf}
PY=${PY:-$HOME/redrob-image-venv/bin/python}
OUT=${OUT:-$HOME/runs/reward-check}
step() { echo "== $* $(date -u +%FT%TZ)"; "$@" || { echo "FAILED: $*"; exit 1; }; }

step $PY scripts/make_jamo_pairs.py --n-prompts "${N:-100}" --per "${PER:-5}" --out "$OUT"
step $PY scripts/reward.py --check "$OUT"
echo "REWARD CHECK CHAIN DONE $(date -u +%FT%TZ)"
