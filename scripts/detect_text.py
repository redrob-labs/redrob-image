#!/usr/bin/env python3
"""Text DETECTION only (no recognition) for datasets whose labels cover part of the text in an image.

AI Hub 71553 (menu boards) labels menu names but not prices or other text. The parser must know where that
other text is, so a crop with it inside can be dropped instead of teaching a caption that leaves it out. This
writes one <out>/<image stem>.json per image: {"size": [w, h], "boxes": [[x0, y0, x1, y1], ...]} in source
pixels. Recognition is deliberately not run: an OCR reading would be a guess, and guesses never become labels.

Runs in the PaddleOCR venv (paddle-venv on the gpu-box):
    ~/paddle-venv/bin/python scripts/detect_text.py --images ~/aihub/71553/extracted --out ~/aihub/71553/det --limit 1000
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png"}


def boxes_of(det, im, min_score: float) -> list[list[float]]:
    import numpy as np

    out = []
    for res in det.predict(np.asarray(im)[:, :, ::-1].copy()):
        r = res.json["res"] if hasattr(res, "json") else res
        for poly, s in zip(r.get("dt_polys", []), r.get("dt_scores", [])):
            if s < min_score:
                continue
            xs, ys = [float(q[0]) for q in poly], [float(q[1]) for q in poly]
            out.append([round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="PP-OCRv5_server_det")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-score", type=float, default=0.5)
    a = ap.parse_args()
    from paddleocr import TextDetection
    from PIL import Image, ImageOps

    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted(Path(dp) / f for dp, _, fs in os.walk(a.images.expanduser()) for f in fs
                  if Path(f).suffix.lower() in IMG_EXT)
    if a.limit:
        imgs = imgs[: a.limit]
    det = TextDetection(model_name=a.model, device=a.device)
    t0, done = time.time(), 0
    for p in imgs:
        dst = out / f"{p.stem}.json"
        if dst.exists():
            continue
        try:
            im = Image.open(p)
            rot = im.getexif().get(0x0112, 1) in (5, 6, 7, 8)
            im = im.convert("RGB")
        except Exception:
            continue
        rec = {"size": list(im.size), "boxes": boxes_of(det, im, a.min_score)}
        if rot:
            # labels are drawn on either the stored or the EXIF-rotated orientation; keep both so the parser
            # can use whichever matches the label's image size
            t = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            rec["rotated"] = {"size": list(t.size), "boxes": boxes_of(det, t, a.min_score)}
        dst.write_text(json.dumps(rec), encoding="utf-8")
        done += 1
        if done % 100 == 0:
            print(f"{done}/{len(imgs)} {done / (time.time() - t0):.2f} img/s", flush=True)
    print(f"done {done} images in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
