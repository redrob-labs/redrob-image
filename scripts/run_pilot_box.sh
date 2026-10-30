#!/bin/bash
# Pilot chain on the gpu-box, one GPU job at a time:
#   wait for an earlier job (pid in $1) -> pre-compute 512 (all) and 1024 (3 shards per source)
#   -> LoRA r32 -> score its eval snapshots -> Turbo + r32 eval -> LoRA r64 -> same evals.
# Every step is resumable, so re-running this script after a crash or reboot continues where it stopped.
#   setsid nohup bash scripts/run_pilot_box.sh <pid-to-wait-for> > ~/runs/pilot-chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$HOME/hf}
PY=${PY:-$HOME/redrob-image-venv/bin/python}
CAP=${CAP:-$HOME/aihub/105/captions-20k/captions.composed.jsonl}
SHARDS=${SHARDS:-$HOME/aihub/105/shards-v1}
SYNTH=${SYNTH:-$HOME/data/synth-20k/shards}
PRE=${PRE:-$HOME/data/precomp}
RUNS=${RUNS:-$HOME/runs}
if [ -n "${1:-}" ]; then while [ -d "/proc/$1" ]; do sleep 60; done; fi
[ -s "$CAP" ] || { echo "no captions at $CAP"; exit 1; }
step() { echo "== $* $(date -u +%FT%TZ)"; "$@" || { echo "FAILED: $*"; exit 1; }; }

step $PY scripts/precompute.py --res 512 --out "$PRE" --aihub "$SHARDS:$CAP" --synth "$SYNTH"
step $PY scripts/precompute.py --res 1024 --out "$PRE" --aihub "$SHARDS:$CAP" --synth "$SYNTH" --max-per-source 3000 --vae-batch 2

for R in 32 64; do
  RUN=$RUNS/pilot-r$R
  step $PY scripts/train_lora.py --config configs/g6e.yaml --run "$RUN" --rank $R
  for d in "$RUN"/eval/step-*; do
    [ -f "$d/report.json" ] || step $PY scripts/eval_baseline.py --model Tongyi-MAI/Z-Image --steps 28 --guidance 4 \
        --subset 40 --ocr qwen3vl-8b --out "$d"
  done
  for m in Tongyi-MAI/Z-Image-Turbo:9:0 Tongyi-MAI/Z-Image:28:4; do
    IFS=: read -r model steps g <<< "$m"
    tag=$(basename "$model")
    [ -f "$RUN/final-eval-$tag/report.json" ] || step $PY scripts/eval_baseline.py --model "$model" --steps "$steps" \
        --guidance "$g" --prompts eval/prompts.jsonl eval/prompts_unseen_syllables.jsonl --ocr qwen3vl-8b \
        --lora "$RUN/final/pytorch_lora_weights.safetensors" --out "$RUN/final-eval-$tag"
  done
done
echo "PILOT CHAIN DONE $(date -u +%FT%TZ)"
