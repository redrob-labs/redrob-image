#!/bin/bash
# Fetch Qwen-Image-Edit-2511 (Apache-2.0) into the gpu-box HF cache for the plan B re-render trial.
#   setsid nohup bash scripts/fetch_qwen_edit_box.sh > ~/runs/fetch-qwen-edit.log 2>&1 &
set -uo pipefail
export HF_HOME=${HF_HOME:-$HOME/hf}
PY=${PY:-$HOME/redrob-image-venv/bin/python}
$PY - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen-Image-Edit-2511", max_workers=8)
print("FETCH DONE", p, flush=True)
EOF
