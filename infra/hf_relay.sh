#!/bin/bash
# Relay Hugging Face dataset files to the GPU box one at a time: download here (direct internet),
# copy to the box over the tailnet, delete the local copy. Needed while the box routes its internet
# through the Korean exit node for AI Hub (home upload speed). Resumable: done files are skipped.
#   bash infra/hf_relay.sh <user@box> <remote_dir> <repo_id> <path...>
# Example:
#   bash infra/hf_relay.sh ubuntu@gpu-box data/nvidia-ocr-ko nvidia/OCR-Synthetic-Multilingual-v1 ko/train/train_000.h5
set -euo pipefail
BOX=$1; RDIR=$2; REPO=$3; shift 3
LOCAL=${RELAY_DIR:-${KIROCREW_SCRATCH:-$HOME/.cache}/hf-relay}
SSH=(ssh -F /dev/null -o StrictHostKeyChecking=yes -o NumberOfPasswordPrompts=0 -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no)
mkdir -p "$LOCAL"
"${SSH[@]}" "$BOX" "mkdir -p $RDIR/.done"
for p in "$@"; do
  key=${p//\//__}
  if "${SSH[@]}" "$BOX" "test -f $RDIR/.done/$key"; then echo "skip $p"; continue; fi
  echo "== $p $(date -u +%FT%TZ)"
  f="$LOCAL/$key"
  curl -sS -L -C - --retry 5 -o "$f" "https://huggingface.co/datasets/$REPO/resolve/main/$p"
  "${SSH[@]}" "$BOX" "mkdir -p \$(dirname $RDIR/$p)"
  rsync -a --partial -e "${SSH[*]}" "$f" "$BOX:$RDIR/$p.part"
  "${SSH[@]}" "$BOX" "mv $RDIR/$p.part $RDIR/$p && touch $RDIR/.done/$key"
  rm -f "$f"
  echo "done $p $(du -h --apparent-size "$LOCAL" | tail -1 | cut -f1) local"
done
echo "RELAY DONE $REPO $(date -u +%FT%TZ)"
