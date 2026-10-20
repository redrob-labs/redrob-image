#!/bin/bash
# Held-out set for the stable validation loss (configs/*.yaml val:), built on the gpu-box:
#   256 AI Hub 81 text-in-the-wild samples (a dataset the pilot never trains on), captioned like T5,
#   + 256 synth images from a seed the training set did not use (99), both pre-computed at 512 px.
# Then, if $1/$2 name the running pilot chain and trainer pids, stop them cleanly (the trainer checkpoints on
# SIGTERM) and restart the chain, which resumes the same run with validation switched on.
#   setsid nohup bash scripts/prep_val_box.sh <chain-pid> <trainer-pid> > ~/runs/prep-val.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$HOME/hf}
PY=${PY:-$HOME/redrob-image-venv/bin/python}
V=$HOME/data
step() { echo "== $* $(date -u +%FT%TZ)"; "$@" || { echo "FAILED: $*"; exit 1; }; }

[ -n "$(ls "$V"/synth-val/shards/*.tar 2>/dev/null)" ] || \
  step $PY synth/generate.py --count 256 --seed 99 --n 0 --shard-size 256 --workers 4 --out "$V/synth-val"
[ -s "$V/val81-cap/captions.composed.jsonl" ] || \
  step $PY scripts/caption.py --shards "$HOME/aihub/81/shards-v1" --out "$V/val81-cap" --n 256 --seed 1 --batch 8 --name val81
step $PY scripts/precompute.py --res 512 --out "$V/precomp-val" \
  --aihub "$HOME/aihub/81/shards-v1:$V/val81-cap/captions.composed.jsonl:aihub81" --synth "$V/synth-val/shards"

if [ -n "${1:-}" ] && [ -n "${2:-}" ]; then
  kill "$1"
  kill -TERM "$2"
  while [ -d "/proc/$2" ]; do sleep 5; done
  echo "== trainer stopped $(date -u +%FT%TZ); restarting pilot chain"
  setsid nohup bash scripts/run_pilot_box.sh </dev/null >> "$HOME/runs/pilot-chain.log" 2>&1 &
  echo "chain pid $!"
fi
echo "PREP VAL DONE $(date -u +%FT%TZ)"
