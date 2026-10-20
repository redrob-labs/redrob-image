#!/bin/bash
# Quick PaddleOCR score of an in-training eval snapshot next to the Base baseline on the same 40 prompts,
# while the trainer keeps the GPU (PaddleOCR needs ~2 GB). Writes to <run>/quick-paddle/{step-N,base}/ so the
# pilot chain's own qwen3vl scoring of <run>/eval/step-N is untouched.
#   bash scripts/quick_score_box.sh ~/runs/pilot-r32 2500
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=$1; STEP=$2
PY=${PY:-$HOME/redrob-image-venv/bin/python}
BASE_IMGS=${BASE_IMGS:-$HOME/redrob-image-eval/repo/outputs/baseline-base/images}
export REDROB_PADDLE_PY=${REDROB_PADDLE_PY:-$HOME/paddle-venv/bin/python}
Q=$RUN/quick-paddle
for tag in "step-$STEP" base; do
  src=$RUN/eval/step-$STEP/images; [ "$tag" = base ] && src=$BASE_IMGS
  mkdir -p "$Q/$tag/images"
  for f in "$RUN/eval/step-$STEP/images"/*.png; do
    ln -sf "$src/$(basename "$f")" "$Q/$tag/images/$(basename "$f")"
  done
  [ -f "$Q/$tag/report.json" ] || $PY scripts/eval_baseline.py --model Tongyi-MAI/Z-Image --steps 28 --guidance 4 \
      --subset 40 --ocr paddleocr --out "$Q/$tag"
done
echo "QUICK SCORE DONE"
