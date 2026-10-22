#!/bin/bash
# Glyph pilot chain on the gpu-box (docs/glyph-pilot-plan.md), one GPU job at a time:
#   wait for the r64 v2 chain (pid in $1) -> glyph pre-compute (val, 512, 1024)
#   -> reaction check: Base with the glyph and no LoRA on 10 prompts
#   -> 100-step smoke -> pilot (configs/g6e-glyph-v1.yaml) -> qwen3vl scoring of snapshots + final evals.
# Every step is resumable; re-running continues where it stopped.
#   setsid nohup bash scripts/run_glyph_box.sh <v2-chain-pid> >> ~/runs/glyph-chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$HOME/hf}
PY=${PY:-$HOME/redrob-image-venv/bin/python}
CAP=${CAP:-$HOME/aihub/105/captions-20k/captions.composed.jsonl}
SHARDS=${SHARDS:-$HOME/aihub/105/shards-v1}
SYNTH=${SYNTH:-$HOME/data/synth-20k/shards}
V=$HOME/data
PRE=${PRE:-$V/precomp-glyph}
RUNS=${RUNS:-$HOME/runs}
RUN=$RUNS/pilot-glyph-v1
if [ -n "${1:-}" ]; then while [ -d "/proc/$1" ]; do sleep 60; done; fi
step() { echo "== $* $(date -u +%FT%TZ)"; "$@" || { echo "FAILED: $*"; exit 1; }; }

step $PY scripts/precompute.py --glyph --res 512 --out "$V/precomp-val-glyph" \
  --aihub "$HOME/aihub/81/shards-v1:$V/val81-cap/captions.composed.jsonl:aihub81" --synth "$V/synth-val/shards"
step $PY scripts/precompute.py --glyph --res 512 --out "$PRE" --aihub "$SHARDS:$CAP" --synth "$SYNTH"
step $PY scripts/precompute.py --glyph --res 1024 --out "$PRE" --aihub "$SHARDS:$CAP" --synth "$SYNTH" \
  --max-per-source 6000 --vae-batch 2
# Base was not trained with a glyph-only condition; see whether it reacts at all before training
[ -f "$RUNS/glyph-react-base/report.json" ] || step $PY scripts/eval_baseline.py --model Tongyi-MAI/Z-Image --steps 28 \
  --guidance 4 --subset 10 --glyph --ocr qwen3vl-8b --out "$RUNS/glyph-react-base"
[ -f "$RUNS/glyph-smoke/summary.json" ] || step $PY scripts/train_lora.py --config configs/g6e-glyph-v1.yaml \
  --run "$RUNS/glyph-smoke" --max-steps 100 --no-eval
step $PY scripts/train_lora.py --config configs/g6e-glyph-v1.yaml --run "$RUN"

for d in "$RUN"/eval/step-*; do
  [ -f "$d/report.json" ] || step $PY scripts/eval_baseline.py --model Tongyi-MAI/Z-Image --steps 28 --guidance 4 \
      --subset 40 --glyph --ocr qwen3vl-8b --out "$d"
done
for m in Tongyi-MAI/Z-Image-Turbo:9:0 Tongyi-MAI/Z-Image:28:4; do
  IFS=: read -r model steps g <<< "$m"
  tag=$(basename "$model")
  [ -f "$RUN/final-eval-$tag/report.json" ] || step $PY scripts/eval_baseline.py --model "$model" --steps "$steps" \
      --guidance "$g" --prompts eval/prompts.jsonl eval/prompts_unseen_syllables.jsonl --glyph --ocr qwen3vl-8b \
      --lora "$RUN/final/pytorch_lora_weights.safetensors" --out "$RUN/final-eval-$tag"
done
echo "GLYPH CHAIN DONE $(date -u +%FT%TZ)"
