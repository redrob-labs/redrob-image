#!/bin/bash
# Glyph channel pilot chain on the gpu-box (configs/g6e-glyph-ch1.yaml), one GPU job at a time:
#   50-step smoke -> pilot -> dual-OCR scoring of every in-run eval -> final evals (Base + LoRA, Turbo + LoRA).
# Reuses ~/data/precomp-glyph and ~/data/precomp-val-glyph. Every step is resumable.
#   setsid nohup bash scripts/run_glyph_ch_box.sh >> ~/runs/glyph-ch-chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$HOME/hf}
PY=${PY:-$HOME/redrob-image-venv/bin/python}
RUNS=${RUNS:-$HOME/runs}
RUN=$RUNS/pilot-glyph-ch1
CFG=configs/g6e-glyph-ch1.yaml
step() { echo "== $* $(date -u +%FT%TZ)"; "$@" || { echo "FAILED: $*"; exit 1; }; }

[ -f "$RUNS/glyph-ch-smoke/summary.json" ] || step $PY scripts/train_lora.py --config $CFG \
  --run "$RUNS/glyph-ch-smoke" --max-steps 50 --no-eval
step $PY scripts/train_lora.py --config $CFG --run "$RUN"

for d in "$RUN"/eval/step-*; do
  # images already exist (written in-run), so this only scores them
  [ -f "$d/report.json" ] || step $PY scripts/eval_baseline.py --model Tongyi-MAI/Z-Image --steps 28 --guidance 4 \
      --subset 40 --glyph --glyph-mode channel --ocr dual --lora "$RUN/final/pytorch_lora_weights.safetensors" --out "$d"
done
for m in Tongyi-MAI/Z-Image:28:4 Tongyi-MAI/Z-Image-Turbo:9:0; do
  IFS=: read -r model steps g <<< "$m"
  tag=$(basename "$model")
  [ -f "$RUN/final-eval-$tag/report.json" ] || step $PY scripts/eval_baseline.py --model "$model" --steps "$steps" \
      --guidance "$g" --prompts eval/prompts.jsonl eval/prompts_unseen_syllables.jsonl --glyph --glyph-mode channel \
      --ocr dual --lora "$RUN/final/pytorch_lora_weights.safetensors" --out "$RUN/final-eval-$tag"
done
echo "GLYPH CH CHAIN DONE $(date -u +%FT%TZ)"
