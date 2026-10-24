#!/usr/bin/env python3
"""Dual-OCR score of the text box only.

Full-image OCR mixes in any text the scene model drew by itself, so a correct pasted word scored wrong
("가져 파 개몸 ... 이곳은 금연 구역입니다"). This crops each image to the glyph.layout() boxes the text was
placed in (plus a margin), reads only the crop with both engines, and scores it like eval_baseline.py --ocr dual.

  python scripts/score_crops.py --prompts ~/runs/rerender-trial/prompts.jsonl \
      --dirs ~/runs/render-harmonize-trial/plain ~/runs/rerender-trial/qwen ...
writes <dir>/crop_report.json and <dir>/crop_scores.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "eval"))
import glyph as G  # noqa: E402
import metrics  # noqa: E402

DUAL = ["paddleocr", "qwen3vl-8b"]


def crop(img, r: dict, margin: float):
    W, H = img.size
    bs = [b["bbox_xyxy"] for b in G.layout(r["target_text"], r.get("scene", ""), W, H)]
    x0, y0 = min(b[0] for b in bs), min(b[1] for b in bs)
    x1, y1 = max(b[2] for b in bs), max(b[3] for b in bs)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return img.crop((max(0, int(x0 - mx)), max(0, int(y0 - my)), min(W, int(x1 + mx)), min(H, int(y1 + my))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, type=Path)
    ap.add_argument("--dirs", nargs="+", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--margin", type=float, default=0.15)
    a = ap.parse_args()

    import ocr_engines
    from PIL import Image

    pp = Path.home() / "paddle-venv/bin/python"
    if pp.exists():
        os.environ.setdefault("REDROB_PADDLE_PY", str(pp))
    rows = [json.loads(l) for l in a.prompts.expanduser().read_text(encoding="utf-8").splitlines() if l.strip()]
    engs = {n: ocr_engines.make(n) for n in DUAL}
    for d in a.dirs:
        d = d.expanduser()
        scored = []
        for r in rows:
            p = d / "images" / f"{r['id']}_s{a.seed}.png"
            if not p.exists():
                continue
            c = crop(Image.open(p).convert("RGB"), r, a.margin)
            per = {n: e(c) for n, e in engs.items()}
            sc = {n: metrics.score_one(r["target_text"], ls) for n, ls in per.items()}
            row = dict(r, image_id=p.stem, **metrics.combine(*sc.values()))
            for n in DUAL:
                row[f"lines_{n}"], row[f"score_{n}"] = per[n], sc[n]
            scored.append(row)
        if not scored:
            print(f"{d}: no images", flush=True)
            continue
        with (d / "crop_scores.jsonl").open("w", encoding="utf-8") as f:
            for row in scored:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        agg = metrics.aggregate(scored)
        (d / "crop_report.json").write_text(json.dumps(dict(margin=a.margin, ocr="dual-crop", **agg), indent=1,
                                                       ensure_ascii=False))
        o = agg["overall"]
        print(f"{d.name}: n={o['n']} crop CER {o['cer']:.3f} EM {o['em']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
