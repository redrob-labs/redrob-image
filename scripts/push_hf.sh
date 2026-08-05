#!/usr/bin/env bash
# Upload Redrob Image open-weight package to Hugging Face.
#
# Ships both:
#   - redrob-image.safetensors          (ComfyUI UNET)
#   - transformer/                      (Diffusers ZImageTransformer2DModel)
#
# From this directory (with weights present or WEIGHT= pointing at GPU path):
#   HF_REPO=your-org/redrob-image bash scripts/push_hf.sh
#
# Optional:
#   WEIGHT=/path/to/redrob-image.safetensors
#   TRANSFORMER_DIR=/path/to/transformer   # skip conversion if already built
#   PYTHON=python3

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF_REPO="${HF_REPO:?Set HF_REPO e.g. savagemanage/redrob-image}"
WEIGHT="${WEIGHT:-$ROOT/redrob-image.safetensors}"
TRANSFORMER_DIR="${TRANSFORMER_DIR:-}"
COMFY_VENV="${COMFY_VENV:-$HOME/ComfyUI/venv}"
HF_BIN="${HF_BIN:-$COMFY_VENV/bin/hf}"
PYTHON="${PYTHON:-}"

if [[ ! -f "$WEIGHT" ]]; then
  echo "missing weight: $WEIGHT" >&2
  echo "hint: WEIGHT=/opt/dlami/nvme/comfyui-models/diffusion_models/redrob-image.safetensors $0" >&2
  exit 1
fi

if [[ ! -x "$HF_BIN" ]]; then
  HF_BIN="$(command -v hf || true)"
fi
if [[ -z "$HF_BIN" ]]; then
  echo "hf CLI not found" >&2
  exit 1
fi

if [[ -z "$PYTHON" ]]; then
  if [[ -x "$COMFY_VENV/bin/python" ]]; then
    PYTHON="$COMFY_VENV/bin/python"
  else
    PYTHON="$(command -v python3 || command -v python || true)"
  fi
fi
if [[ -z "$PYTHON" ]]; then
  echo "python not found (needed to build Diffusers transformer/)" >&2
  exit 1
fi

echo "[auth]"
"$HF_BIN" auth whoami

echo "[repo] $HF_REPO"
"$HF_BIN" repo create "$HF_REPO" --type model --exist-ok || true

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

cp -f "$ROOT/README.md" "$STAGING/README.md"
cp -f "$ROOT/README.ko.md" "$STAGING/README.ko.md"
cp -f "$ROOT/LICENSE" "$STAGING/LICENSE"
cp -f "$ROOT/NOTICE" "$STAGING/NOTICE"
mkdir -p "$STAGING/workflows"
cp -f "$ROOT/workflows/redrob-image-api.json" "$STAGING/workflows/redrob-image-api.json"
ln -f "$WEIGHT" "$STAGING/redrob-image.safetensors" 2>/dev/null \
  || cp -f "$WEIGHT" "$STAGING/redrob-image.safetensors"

if [[ -n "$TRANSFORMER_DIR" ]]; then
  if [[ ! -f "$TRANSFORMER_DIR/diffusion_pytorch_model.safetensors" && ! -f "$TRANSFORMER_DIR/diffusion_pytorch_model.safetensors.index.json" ]]; then
    echo "TRANSFORMER_DIR missing diffusion_pytorch_model.safetensors: $TRANSFORMER_DIR" >&2
    exit 1
  fi
  if [[ ! -f "$TRANSFORMER_DIR/config.json" ]]; then
    echo "TRANSFORMER_DIR missing config.json: $TRANSFORMER_DIR" >&2
    exit 1
  fi
  echo "[transformer] copy $TRANSFORMER_DIR"
  mkdir -p "$STAGING/transformer"
  cp -f "$TRANSFORMER_DIR/config.json" "$STAGING/transformer/config.json"
  # Copy weight file(s); support single-file or sharded layouts.
  shopt -s nullglob
  for f in "$TRANSFORMER_DIR"/diffusion_pytorch_model*.safetensors \
           "$TRANSFORMER_DIR"/diffusion_pytorch_model.safetensors.index.json; do
    cp -f "$f" "$STAGING/transformer/$(basename "$f")"
  done
  shopt -u nullglob
else
  echo "[transformer] convert from Comfy UNET"
  "$PYTHON" "$ROOT/scripts/comfy_to_diffusers_zimage.py" \
    --input "$WEIGHT" \
    --output-dir "$STAGING/transformer"
fi

echo "[upload] comfy=$(du -h "$STAGING/redrob-image.safetensors" | cut -f1) transformer=$(du -sh "$STAGING/transformer" | cut -f1)"
"$HF_BIN" upload "$HF_REPO" "$STAGING" . --repo-type model

echo "[done] https://huggingface.co/$HF_REPO"
