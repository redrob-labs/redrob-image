#!/bin/bash
# Glyph swap test: does the model draw what the GLYPH says or what the PROMPT says?
# Each record keeps prompt A's caption (text A in quotes) but sets target_text = B, so the glyph map shows B.
# Scores are against B (the glyph); the OCR lines in scores show whether A or B came out.
# Stops the glyph chain + trainer (trainer checkpoints on SIGTERM), runs 10 images with the latest LoRA ckpt,
# then restarts the chain, which resumes training.
#   setsid nohup bash scripts/glyph_swap_box.sh <chain-pid> <trainer-pid> > ~/runs/glyph-swap.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$HOME/hf}
PY=${PY:-$HOME/redrob-image-venv/bin/python}
RUN=${RUN:-$HOME/runs/pilot-glyph-v1}
OUT=${OUT:-$HOME/runs/glyph-swap}
mkdir -p "$OUT"
# WAIT_EVAL=<dir>: wait until the trainer's in-run eval has written all its images before stopping it.
if [ -n "${WAIT_EVAL:-}" ]; then
  while [ "$(ls "$WAIT_EVAL/images" 2>/dev/null | wc -l)" -lt "${WAIT_N:-40}" ]; do sleep 30; done
  sleep 60
  echo "== eval $WAIT_EVAL complete $(date -u +%FT%TZ)"
fi
$PY - "$OUT/prompts_swap.jsonl" <<'EOF'
import json, sys
recs = {}
for l in open("eval/prompts.jsonl", encoding="utf-8"):
    r = json.loads(l)
    recs.setdefault(r["target_text"], r)
pairs = [("약국", "편의점"), ("편의점", "약국"), ("퉁소", "꼴뚜기"), ("꼴뚜기", "퉁소"),
         ("주차장", "도깨비불"), ("도깨비불", "주차장"), ("세일", "커피"), ("커피", "세일"),
         ("신선한 과일 채소", "도깨비방망이가게"), ("도깨비방망이가게", "신선한 과일 채소")]
with open(sys.argv[1], "w", encoding="utf-8") as f:
    for a, b in pairs:
        if a not in recs:
            print("missing prompt", a); continue
        r = dict(recs[a]); r["id"] = f"swap-{len(a)}-{len(b)}-{r['id']}"
        r["prompt_text"] = a; r["target_text"] = b
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
EOF
wc -l "$OUT/prompts_swap.jsonl"
if [ -n "${1:-}" ] && [ -n "${2:-}" ]; then
  kill "$1"; kill -TERM "$2"
  while [ -d "/proc/$2" ]; do sleep 5; done
  echo "== trainer stopped $(date -u +%FT%TZ)"
fi
CK=$(ls -d "$RUN"/ckpt/step-* | sort -t- -k2 -n | tail -1)
echo "== ckpt $CK"
$PY scripts/eval_baseline.py --model Tongyi-MAI/Z-Image --steps 28 --guidance 4 --glyph --glyph-mode "${GLYPH_MODE:-omni}" --ocr dual \
  --prompts "$OUT/prompts_swap.jsonl" --lora "$CK/pytorch_lora_weights.safetensors" --out "$OUT" \
  || echo "FAILED: swap eval"
if [ -n "${1:-}" ]; then
  setsid nohup bash "${CHAIN:-scripts/run_glyph_box.sh}" </dev/null >> "${CHAIN_LOG:-$HOME/runs/glyph-chain.log}" 2>&1 &
  echo "== glyph chain restarted, pid $!"
fi
echo "SWAP DONE $(date -u +%FT%TZ)"
