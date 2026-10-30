#!/usr/bin/env python3
"""Side-by-side sheets for the plan B re-render trial (human naturalness check).

  RUNS=~/runs python scripts/rerender_sheet.py   # writes $RUNS/rerender-trial/sheet-full.png and sheet-zoom.png
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import glyph as G  # noqa: E402

RUNS = Path(__import__("os").environ.get("RUNS", ROOT / "paper/runs")).expanduser()  # RUNS=~/runs on the gpu-box
COLS = [("plain", RUNS / "render-harmonize-trial/plain"), ("b16-s0.3", RUNS / "render-harmonize-trial/b16-s0.3"),
        ("zr24-s0.35", RUNS / "rerender-trial/zr24-s0.35"), ("zr24-s0.5", RUNS / "rerender-trial/zr24-s0.5"),
        ("qwen", RUNS / "rerender-trial/qwen")]
IDS = ["seen-1-2-01", "seen-3-5-01", "seen-6-10-01", "unseen-1-2-01", "unseen-3-5-00", "unseen-6-10-00",
       "unseen-11+-01", "usyl-02"]


def main():
    rows = {json.loads(l)["id"]: json.loads(l) for l in (RUNS / "rerender-trial/prompts.jsonl").read_text().splitlines()}
    for kind, cw, ch in (("full", 300, 300), ("zoom", 360, 180)):
        sheet = Image.new("RGB", (cw * len(COLS), 24 + ch * len(IDS)), "white")
        d = ImageDraw.Draw(sheet)
        for j, (name, _) in enumerate(COLS):
            d.text((j * cw + 6, 6), name, fill="black")
        for i, iid in enumerate(IDS):
            r = rows[iid]
            for j, (_, p) in enumerate(COLS):
                im = Image.open(p / "images" / f"{iid}_s42.png").convert("RGB")
                if kind == "zoom":
                    W, H = im.size
                    bs = [b["bbox_xyxy"] for b in G.layout(r["target_text"], r.get("scene", ""), W, H)]
                    x0, y0 = min(b[0] for b in bs), min(b[1] for b in bs)
                    x1, y1 = max(b[2] for b in bs), max(b[3] for b in bs)
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    hw = max(x1 - x0, 2 * (y1 - y0)) * 0.65
                    im = im.crop((int(max(0, cx - hw)), int(max(0, cy - hw / 2)), int(min(W, cx + hw)),
                                  int(min(H, cy + hw / 2))))
                im.thumbnail((cw - 4, ch - 4))
                sheet.paste(im, (j * cw + 2, 24 + i * ch + 2))
        out = RUNS / f"rerender-trial/sheet-{kind}.png"
        sheet.save(out)
        print(out, sheet.size)


if __name__ == "__main__":
    main()
