#!/usr/bin/env python3
"""OCR worker for engines that must live in their own venv (PaddlePaddle pins CUDA wheels that
conflict with torch's). Protocol: one image path per stdin line -> one JSON list of lines per
stdout line. Started by ocr_engines.SubprocessEngine.

  <venv>/bin/python eval/ocr_worker.py paddleocr
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["REDROB_OCR_INPROC"] = "1"

from PIL import Image  # noqa: E402

import ocr_engines  # noqa: E402

eng = ocr_engines.make(sys.argv[1])
out = sys.stdout
sys.stdout = sys.stderr  # library chatter must not corrupt the protocol
print("READY", file=out, flush=True)
for line in sys.stdin:
    p = line.rstrip("\n")
    if not p:
        continue
    try:
        res = eng(Image.open(p))
    except Exception as e:  # report, keep serving
        print(f"worker error on {p}: {e!r}", file=sys.stderr, flush=True)
        res = []
    print(json.dumps(res, ensure_ascii=False), file=out, flush=True)
