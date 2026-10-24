#!/bin/bash
# Plan B trial on the gpu-box: render + harmonize 20 prompts, then dual-OCR score every config.
#   setsid nohup bash scripts/run_render_harmonize_box.sh > ~/runs/render-harmonize-trial.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$HOME/hf}
PY=${PY:-$HOME/redrob-image-venv/bin/python}
OUT=${OUT:-$HOME/runs/render-harmonize-trial}
step() { echo "== $* $(date -u +%FT%TZ)"; "$@" || { echo "FAILED: $*"; exit 1; }; }
step $PY scripts/render_harmonize.py --prompts eval/prompts.jsonl eval/prompts_unseen_syllables.jsonl \
  --subset "${N:-20}" --out "$OUT"
for d in "$OUT"/plain "$OUT"/b*-s*; do
  [ -f "$d/report.json" ] || step $PY scripts/eval_baseline.py --model Tongyi-MAI/Z-Image --steps 28 --guidance 4 \
      --prompts "$OUT/prompts.jsonl" --ocr dual --out "$d"
done
echo "RENDER HARMONIZE TRIAL DONE $(date -u +%FT%TZ)"
