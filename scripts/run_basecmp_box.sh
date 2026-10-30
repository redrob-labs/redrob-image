#!/usr/bin/env bash
# Base-model comparison (after plan C was dropped): score other open models on the same 220-prompt eval,
# no training, same dual OCR + syllable-classifier reward as r64 v2. Ungated models only for now.
#   setsid -f bash scripts/run_basecmp_box.sh > ~/runs/basecmp-chain.log 2>&1 < /dev/null
set -euo pipefail
cd "$HOME/redrob-image-t4"
PY="$HOME/redrob-image-venv/bin/python"
export HF_HOME=${HF_HOME:-$HOME/hf}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; }
PROMPTS=(eval/prompts.jsonl eval/prompts_unseen_syllables.jsonl)

# Qwen-Image-2512 (Apache-2.0, 20B). Model card settings: 50 steps, true CFG 4.0.
M=Qwen/Qwen-Image-2512
step $PY -c "from huggingface_hub import snapshot_download as s; print('FETCH DONE', s('$M', max_workers=8))"
R="$HOME/runs/basecmp/qwen-image-2512"
gen() { $PY scripts/eval_baseline.py --model "$M" --steps 50 --guidance 4 --prompts "${PROMPTS[@]}" "$@"; }
# 8-prompt smoke first: catches load/OOM problems and gives the seconds per image
[ -f "$R-smoke/report.json" ] || step gen --subset 8 --ocr paddleocr --out "$R-smoke" \
    || step gen --subset 8 --ocr paddleocr --out "$R-smoke" --offload
[ -f "$R/report.json" ] || step gen --ocr dual --out "$R" || step gen --ocr dual --out "$R" --offload
[ -f "$R/rewards_cls.jsonl" ] || step $PY scripts/reward_cls.py score --images "$R" --model "$HOME/runs/syl-cls-v2/model.pt"
echo "BASECMP DONE $(date -u +%FT%TZ)"
