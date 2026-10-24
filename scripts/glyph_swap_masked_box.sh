#!/bin/bash
# Masked-caption swap test for the glyph-channel ch2 pilot: the caption's quoted text is replaced with "*"
# (as in ch2 training), and the glyph map shows text B. If the model draws B, it reads the glyph.
# Reuses the prompt file of an earlier swap run (SRC) for the same 10 pairs.
#   setsid nohup bash scripts/glyph_swap_masked_box.sh <chain-pid> <trainer-pid> > ~/runs/glyph-ch2-swap-masked.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$HOME/hf}
PY=${PY:-$HOME/redrob-image-venv/bin/python}
RUN=${RUN:-$HOME/runs/pilot-glyph-ch2}
SRC=${SRC:-$HOME/runs/glyph-ch2-swap-mid/prompts_swap.jsonl}
OUT=${OUT:-$HOME/runs/glyph-ch2-swap-masked}
mkdir -p "$OUT"
$PY - "$SRC" "$OUT/prompts_swap.jsonl" <<'EOF'
import json, re, sys
q = re.compile(r'"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’')
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for l in open(sys.argv[1], encoding="utf-8"):
        r = json.loads(l)
        for k in ("prompt_ko", "prompt_en"):
            r[k] = q.sub(lambda m: m.group(0)[0] + "*" + m.group(0)[-1], r[k])
        r["id"] = r["id"].replace("swap-", "swapm-", 1)
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
EOF
head -2 "$OUT/prompts_swap.jsonl"
if [ -n "${1:-}" ] && [ -n "${2:-}" ]; then
  kill "$1"; kill -TERM "$2"
  while [ -d "/proc/$2" ]; do sleep 5; done
  echo "== trainer stopped $(date -u +%FT%TZ)"
fi
CK=$(ls -d "$RUN"/ckpt/step-* | sort -t- -k2 -n | tail -1)
echo "== ckpt $CK"
$PY scripts/eval_baseline.py --model Tongyi-MAI/Z-Image --steps 28 --guidance 4 --glyph --glyph-mode channel --ocr dual \
  --prompts "$OUT/prompts_swap.jsonl" --lora "$CK/pytorch_lora_weights.safetensors" --out "$OUT" \
  || echo "FAILED: masked swap eval"
if [ -n "${1:-}" ]; then
  setsid nohup bash scripts/run_glyph_ch2_box.sh </dev/null >> "$HOME/runs/glyph-ch2-chain.log" 2>&1 &
  echo "== glyph ch2 chain restarted, pid $!"
fi
echo "SWAP MASKED DONE $(date -u +%FT%TZ)"
