#!/bin/bash
# Pilot chain v2 on the gpu-box, one GPU job at a time. Replaces run_pilot_box.sh after r32:
#   wait for the r32 trainer (pid in $1) -> score r32's eval snapshots + final evals (as run_pilot_box.sh)
#   -> pre-compute v2 with text masks (512 all, 1024 6000 per source) -> LoRA r64 v2 (configs/g6e-r64v2.yaml)
#   -> score its snapshots + final evals.
# Every step is resumable; re-running continues where it stopped.
#   setsid nohup bash scripts/run_pilot_v2_box.sh <r32-trainer-pid> >> ~/runs/pilot-chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$HOME/hf}
PY=${PY:-$HOME/redrob-image-venv/bin/python}
CAP=${CAP:-$HOME/aihub/105/captions-20k/captions.composed.jsonl}
SHARDS=${SHARDS:-$HOME/aihub/105/shards-v1}
SYNTH=${SYNTH:-$HOME/data/synth-20k/shards}
PRE2=${PRE2:-$HOME/data/precomp-v2}
RUNS=${RUNS:-$HOME/runs}
if [ -n "${1:-}" ]; then while [ -d "/proc/$1" ]; do sleep 60; done; fi
step() { echo "== $* $(date -u +%FT%TZ)"; "$@" || { echo "FAILED: $*"; exit 1; }; }

evals() {  # $1 = run dir: qwen3vl scoring of in-training snapshots, then Turbo/Base + final LoRA on all prompts
  local RUN=$1
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
}

[ -f "$RUNS/pilot-r32/final/pytorch_lora_weights.safetensors" ] && evals "$RUNS/pilot-r32"
step $PY scripts/precompute.py --res 512 --out "$PRE2" --aihub "$SHARDS:$CAP" --synth "$SYNTH"
step $PY scripts/precompute.py --res 1024 --out "$PRE2" --aihub "$SHARDS:$CAP" --synth "$SYNTH" --max-per-source 6000 --vae-batch 2
step $PY scripts/train_lora.py --config configs/g6e-r64v2.yaml --run "$RUNS/pilot-r64v2"
evals "$RUNS/pilot-r64v2"
echo "PILOT CHAIN V2 DONE $(date -u +%FT%TZ)"
