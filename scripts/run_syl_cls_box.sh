#!/usr/bin/env bash
# Syllable classifier chain on the GPU box: fonts -> render cells -> photo patches -> train -> stage-0 check.
# RUN names the run dir (~/runs/<RUN>) and the check tag.
set -euo pipefail
cd "$HOME/redrob-image-t4"
PY="$HOME/redrob-image-venv/bin/python"
RUN="${RUN:-syl-cls}"
TAG="${TAG:-v6cls}"
DATA="$HOME/data/syl-cls"
echo "fonts $(date -u +%FT%TZ)"
bash scripts/fetch_ko_fonts.sh "$HOME/.local/share/fonts/ko-train"
echo "render $(date -u +%FT%TZ)"
[ -f "$DATA/meta.json" ] || $PY scripts/syllable_cls.py render --fonts "$HOME/.local/share/fonts/ko-train" --out "$DATA"
if [ "${BG:-0}" = 1 ] && [ ! -f "$DATA/bg.npy" ]; then
  echo "patches $(date -u +%FT%TZ)"
  $PY scripts/syllable_cls.py patches --src "$HOME/aihub/105/extracted" --out "$DATA"
fi
echo "train $(date -u +%FT%TZ)"
[ -f "$HOME/runs/$RUN/done" ] || { $PY scripts/syllable_cls.py train --data "$DATA" --out "$HOME/runs/$RUN" && touch "$HOME/runs/$RUN/done"; }
echo "check $(date -u +%FT%TZ)"
$PY scripts/syllable_cls.py check --model "$HOME/runs/$RUN/model.pt" --pairs "$HOME/runs/reward-check" --tag "$TAG"
echo "SYL-CLS CHAIN DONE $(date -u +%FT%TZ)"
