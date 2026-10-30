#!/bin/bash
# Plan B second trial on the gpu-box: re-draw the pasted text (Z-Image+LoRA inpaint, Qwen-Image-Edit-2511),
# then score every config two ways: full image (eval_baseline --ocr dual) and text box only (score_crops.py).
#   setsid nohup bash scripts/run_rerender_box.sh > ~/runs/rerender-trial.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$HOME/hf}
PY=${PY:-$HOME/redrob-image-venv/bin/python}
TRIAL=${TRIAL:-$HOME/runs/render-harmonize-trial}
OUT=${OUT:-$HOME/runs/rerender-trial}
LORA=${LORA:-$HOME/runs/pilot-r64v2/final/pytorch_lora_weights.safetensors}
step() { echo "== $* $(date -u +%FT%TZ)"; "$@" || { echo "FAILED: $*"; exit 1; }; }

step $PY scripts/rerender.py --trial "$TRIAL" --out "$OUT" --mode zimage --lora "$LORA"
# the Qwen weights download in parallel (fetch_qwen_edit_box.sh); wait for it
until grep -q "FETCH DONE" ~/runs/fetch-qwen-edit.log 2>/dev/null; do
  grep -q Traceback ~/runs/fetch-qwen-edit.log && { echo "FAILED: qwen fetch"; exit 1; }
  sleep 60
done
$PY scripts/rerender.py --trial "$TRIAL" --out "$OUT" --mode qwen \
  || step $PY scripts/rerender.py --trial "$TRIAL" --out "$OUT" --mode qwen --offload
for d in "$OUT"/zr* "$OUT"/qwen; do
  [ -f "$d/report.json" ] || step $PY scripts/eval_baseline.py --model Tongyi-MAI/Z-Image --steps 28 --guidance 4 \
      --prompts "$OUT/prompts.jsonl" --ocr dual --out "$d"
done
step $PY scripts/score_crops.py --prompts "$OUT/prompts.jsonl" --dirs "$TRIAL"/plain "$TRIAL"/b*-s* "$OUT"/zr* "$OUT"/qwen
echo "RERENDER TRIAL DONE $(date -u +%FT%TZ)"
