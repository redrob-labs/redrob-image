#!/usr/bin/env bash
# Plan C stage 1 chain on the box: wait for the full rollout -> pairs -> (human gate) -> region DPO -> final eval.
# The DPO run starts only after ~/runs/dpo-go exists: it is created once the classifier's verdicts on the
# r64 v2 human-check sheet agree with the human labels (paper/runs/human-check-r64v2).
#   setsid -f bash scripts/run_dpo_box.sh > ~/runs/dpo-chain.log 2>&1 < /dev/null
set -euo pipefail
cd "$HOME/redrob-image-t4"
PY="$HOME/redrob-image-venv/bin/python"
export HF_HOME=${HF_HOME:-$HOME/hf}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROLL="$HOME/runs/c1-rollout"
PAIRS="$HOME/data/dpo-c1"
RUN="$HOME/runs/dpo-c1"
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; }

until grep -q "C1 ROLLOUT DONE N=0" "$HOME/runs/c1-rollout-full.log" 2>/dev/null; do sleep 300; done
echo "rollout done $(date -u +%FT%TZ)"
[ -f "$PAIRS/empty.safetensors" ] || step $PY scripts/dpo_pairs.py --rollout "$ROLL" --out "$PAIRS"
# smoke before the gate, so a trainer bug shows up while the human check is still pending
[ -f "$HOME/runs/dpo-smoke/summary.json" ] || step $PY scripts/train_dpo.py --config configs/g6e-dpo-c1.yaml \
    --run "$HOME/runs/dpo-smoke" --max-steps 2
echo "DPO SMOKE OK $(date -u +%FT%TZ)"

echo "waiting for ~/runs/dpo-go (human check) $(date -u +%FT%TZ)"
until [ -f "$HOME/runs/dpo-go" ]; do sleep 300; done
[ -f "$RUN/summary.json" ] || step $PY scripts/train_dpo.py --config configs/g6e-dpo-c1.yaml --run "$RUN"

E="$RUN/final-eval-Z-Image"
[ -f "$E/report.json" ] || step $PY scripts/eval_baseline.py --model Tongyi-MAI/Z-Image --steps 28 --guidance 4 \
    --prompts eval/prompts.jsonl eval/prompts_unseen_syllables.jsonl --ocr dual \
    --lora "$RUN/final/pytorch_lora_weights.safetensors" --out "$E"
# the same classifier reward on the DPO eval and on the r64 v2 eval it started from
for d in "$E" "$HOME/runs/pilot-r64v2/final-eval-Z-Image"; do
    [ -f "$d/rewards_cls.jsonl" ] || step $PY scripts/reward_cls.py score --images "$d" --model "$HOME/runs/syl-cls-v2/model.pt"
done
echo "DPO CHAIN DONE $(date -u +%FT%TZ)"
