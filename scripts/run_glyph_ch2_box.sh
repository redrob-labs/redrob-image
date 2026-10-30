#!/bin/bash
# Glyph channel pilot ch2 on the gpu-box (configs/g6e-glyph-ch2.yaml), one GPU job at a time:
#   masked-caption embeddings (512, 1024) -> 50-step smoke -> pilot -> swap test -> dual-OCR scoring -> final evals.
# Every step is resumable.
#   setsid nohup bash scripts/run_glyph_ch2_box.sh >> ~/runs/glyph-ch2-chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$HOME/hf}
PY=${PY:-$HOME/redrob-image-venv/bin/python}
RUNS=${RUNS:-$HOME/runs}
RUN=$RUNS/pilot-glyph-ch2
CFG=configs/g6e-glyph-ch2.yaml
PRE=${PRE:-$HOME/data/precomp-glyph}
CAP=${CAP:-$HOME/aihub/105/captions-20k/captions.composed.jsonl}
SYNTH=${SYNTH:-$HOME/data/synth-20k/shards}
step() { echo "== $* $(date -u +%FT%TZ)"; "$@" || { echo "FAILED: $*"; exit 1; }; }

for res in 512 1024; do
  step $PY scripts/precompute_mtxt.py --precomp "$PRE" --res $res --aihub-captions "$CAP" --synth "$SYNTH"
done
[ -f "$RUNS/glyph-ch2-smoke/summary.json" ] || step $PY scripts/train_lora.py --config $CFG \
  --run "$RUNS/glyph-ch2-smoke" --max-steps 50 --no-eval
step $PY scripts/train_lora.py --config $CFG --run "$RUN"

LORA="$RUN/final/pytorch_lora_weights.safetensors"
[ -f "$RUNS/glyph-ch2-swap/report.json" ] || RUN="$RUN" OUT="$RUNS/glyph-ch2-swap" GLYPH_MODE=channel \
  step bash scripts/glyph_swap_box.sh
for d in "$RUN"/eval/step-*; do
  [ -f "$d/report.json" ] || step $PY scripts/eval_baseline.py --model Tongyi-MAI/Z-Image --steps 28 --guidance 4 \
      --subset 40 --glyph --glyph-mode channel --ocr dual --lora "$LORA" --out "$d"
done
for m in Tongyi-MAI/Z-Image:28:4 Tongyi-MAI/Z-Image-Turbo:9:0; do
  IFS=: read -r model steps g <<< "$m"
  tag=$(basename "$model")
  [ -f "$RUN/final-eval-$tag/report.json" ] || step $PY scripts/eval_baseline.py --model "$model" --steps "$steps" \
      --guidance "$g" --prompts eval/prompts.jsonl eval/prompts_unseen_syllables.jsonl --glyph --glyph-mode channel \
      --ocr dual --lora "$LORA" --out "$RUN/final-eval-$tag"
done
echo "GLYPH CH2 CHAIN DONE $(date -u +%FT%TZ)"
