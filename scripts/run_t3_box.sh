#!/bin/sh
# T3 end-to-end on one GPU box: OCR reference set -> OCR engine comparison -> baselines.
# Resumable: each stage writes a .DONE marker under outputs/logs and is skipped on re-run.
#   sh scripts/run_t3_box.sh
set -eu
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$HOME/hf}"
PY="${PY:-$HOME/redrob-image-venv/bin/python}"
export REDROB_PADDLE_PY="${REDROB_PADDLE_PY:-$HOME/paddle-venv/bin/python}"
L=outputs/logs
mkdir -p "$L"
stage() { [ -f "$L/$1.DONE" ] && { echo "skip $1"; return 1; }; echo "== $1 $(date -u +%FT%TZ)"; }
done_() { touch "$L/$1.DONE"; }

if stage refset; then
  $PY scripts/make_ocr_refset.py --download --fonts data/fonts --out data/ocr_refset --n 600; done_ refset
fi
if stage compare; then
  $PY scripts/compare_ocr.py --refset data/ocr_refset --engines paddleocr easyocr qwen3vl-8b --out paper/ocr; done_ compare
fi
# Engine with the lowest overall CER on the reference set.
OCR=$($PY -c "import json;s=json.load(open('paper/ocr/ocr-compare.json'))['engines'];print(min(s,key=lambda e:s[e]['overall']['cer_nospace']))")
echo "ocr=$OCR"
P="eval/prompts.jsonl eval/prompts_unseen_syllables.jsonl"
if stage turbo; then
  $PY scripts/eval_baseline.py --model Tongyi-MAI/Z-Image-Turbo --steps 9 --guidance 0 --ocr "$OCR" \
    --prompts $P --seeds 42 --out outputs/baseline-turbo; done_ turbo
fi
if stage base; then
  $PY scripts/eval_baseline.py --model Tongyi-MAI/Z-Image --steps 50 --guidance 4 --cfg-normalization --ocr "$OCR" \
    --prompts $P --seeds 42 --out outputs/baseline-base; done_ base
fi
echo "ALL DONE $(date -u +%FT%TZ)"
