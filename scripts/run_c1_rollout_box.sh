#!/usr/bin/env bash
# Plan C stage 1 rollout: r64 v2 LoRA on Base, 4 seeds per training prompt, PaddleOCR boxes, classifier reward.
# N=200 is the smoke; N=0 runs all 2,000 (resumes: images already drawn are kept).
set -euo pipefail
cd "$HOME/redrob-image-t4"
PY="$HOME/redrob-image-venv/bin/python"
export HF_HOME=${HF_HOME:-$HOME/hf}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N="${N:-200}"
OUT="$HOME/runs/c1-rollout"
LORA="$HOME/runs/pilot-r64v2/final/pytorch_lora_weights.safetensors"
echo "rollout N=$N $(date -u +%FT%TZ)"
$PY scripts/eval_baseline.py --model Tongyi-MAI/Z-Image --steps 28 --guidance 4 \
  --prompts eval/train_prompts_c1.jsonl --subset "$N" --seeds 1 2 3 4 --ocr paddleocr \
  --lora "$LORA" --review-frac 0 --out "$OUT"
echo "reward $(date -u +%FT%TZ)"
$PY scripts/reward_cls.py score --images "$OUT" --model "$HOME/runs/syl-cls-v2/model.pt"
echo "C1 ROLLOUT DONE N=$N $(date -u +%FT%TZ)"
