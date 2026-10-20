#!/bin/bash
# Full-size eval of one pilot checkpoint: all 220 eval prompts, seed 42, Base + LoRA, scored with PaddleOCR, next to
# Base's own 220 seed-42 baseline images scored the same way. The GPU cannot hold the trainer and a second pipeline,
# so with <chain-pid> <trainer-pid> the chain and trainer are stopped first (the trainer checkpoints on SIGTERM) and
# the chain is restarted afterwards, resuming the run.
#   setsid nohup bash scripts/full_eval_box.sh ~/runs/pilot-r32 step-5001 <chain-pid> <trainer-pid> > ~/runs/full-eval.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=$1; CK=$2
export HF_HOME=${HF_HOME:-$HOME/hf}
export REDROB_PADDLE_PY=${REDROB_PADDLE_PY:-$HOME/paddle-venv/bin/python}
PY=${PY:-$HOME/redrob-image-venv/bin/python}
BASE_IMGS=${BASE_IMGS:-$HOME/redrob-image-eval/repo/outputs/baseline-base/images}
P="eval/prompts.jsonl eval/prompts_unseen_syllables.jsonl"
step() { echo "== $* $(date -u +%FT%TZ)"; "$@" || echo "FAILED: $*"; }

if [ -n "${3:-}" ] && [ -n "${4:-}" ]; then
  kill "$3"; kill -TERM "$4"
  while [ -d "/proc/$4" ]; do sleep 5; done
  echo "== trainer stopped $(date -u +%FT%TZ)"
fi
OUT=$RUN/full-eval-$CK
[ -f "$OUT/report.json" ] || step $PY scripts/eval_baseline.py --model Tongyi-MAI/Z-Image --steps 28 --guidance 4 \
    --prompts $P --seeds 42 --ocr paddleocr --lora "$RUN/ckpt/$CK/pytorch_lora_weights.safetensors" --out "$OUT"
B=$RUN/full-eval-base
mkdir -p "$B/images"
for f in "$BASE_IMGS"/*.png; do ln -sf "$f" "$B/images/$(basename "$f")"; done
[ -f "$B/report.json" ] || step $PY scripts/eval_baseline.py --model Tongyi-MAI/Z-Image --steps 28 --guidance 4 \
    --prompts $P --seeds 42 --ocr paddleocr --out "$B"
if [ -n "${3:-}" ]; then
  setsid nohup bash scripts/run_pilot_box.sh </dev/null >> "$HOME/runs/pilot-chain.log" 2>&1 &
  echo "== pilot chain restarted, pid $!"
fi
echo "FULL EVAL DONE $(date -u +%FT%TZ)"
