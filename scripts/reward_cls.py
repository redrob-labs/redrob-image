#!/usr/bin/env python3
"""Plan C reward for MODEL-DRAWN images: PaddleOCR finds the text lines, the syllable classifier reads cells.

Stage 0 knew where each syllable was (it drew them). A model-drawn image does not tell us, so:
  1. PaddleOCR detection gives line boxes (its recognised text only decides how many cells a line has).
  2. Each line box is cut into cells along its long side: Hangul 1, space 0.35, other characters 0.6.
     If the recognised text and the box shape disagree on the syllable count by more than one, the box shape
     (width / height) decides, so a misread line still gets the right number of cells.
  3. A monotone alignment matches the target syllables to the drawn cells, maximising the summed
     log P(target syllable | cell). A target syllable with no cell costs MISS, a drawn cell with no target
     syllable (extra text) costs EXTRA.
  reward = aligned sum / number of target syllables     (0 is perfect, more negative is worse)

  python scripts/reward_cls.py score --images <eval dir> --model ~/runs/syl-cls-v2/model.pt
      reads <dir>/scores_dual.jsonl (PaddleOCR boxes already there) or runs PaddleOCR, writes rewards_cls.jsonl
  python scripts/reward_cls.py sheet --images <eval dir> --n 20 --out <dir>
      human check sheet (target only, no reward shown) + key.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "eval"))

MISS, EXTRA = -12.0, -3.0


def is_syl(c: str) -> bool:
    return 0xAC00 <= ord(c) <= 0xD7A3


def line_cells(img, line: dict) -> list:
    """Cells (PIL) of one detected line, left to right (top to bottom when the box is tall)."""
    x0, y0, x1, y1 = line["box"]
    w, h = x1 - x0, y1 - y0
    if w < 4 or h < 4:
        return []
    vertical = h > 1.6 * w
    long_, short = (h, w) if vertical else (w, h)
    t = line.get("text", "").strip()
    wts = [1.0 if is_syl(c) else 0.35 if c.isspace() else 0.6 for c in t]
    n_rec = sum(1 for c in t if is_syl(c))
    n_shape = max(1, round(long_ / short))
    if n_rec == 0 or abs(n_rec - n_shape) > 1:
        wts, kinds = [1.0] * n_shape, [True] * n_shape
    else:
        kinds = [is_syl(c) for c in t]
    tot = sum(wts)
    out, pos = [], 0.0
    for wt, k in zip(wts, kinds):
        a, b = pos / tot * long_, (pos + wt) / tot * long_
        pos += wt
        if not k:
            continue
        pad = 0.08 * (b - a)
        if vertical:
            box = (x0 - 0.08 * w, y0 + a - pad, x1 + 0.08 * w, y0 + b + pad)
        else:
            box = (x0 + a - pad, y0 - 0.08 * h, x0 + b + pad, y1 + 0.08 * h)
        W, H = img.size
        box = (max(0, int(box[0])), max(0, int(box[1])), min(W, int(box[2])), min(H, int(box[3])))
        if box[2] - box[0] >= 3 and box[3] - box[1] >= 3:
            out.append(img.crop(box))
    return out


def order(lines: list[dict]) -> list[dict]:
    """Reading order: rows top to bottom (a row = boxes whose vertical centres are close), left to right."""
    ls = sorted((l for l in lines if l.get("box")), key=lambda l: (l["box"][1] + l["box"][3]) / 2)
    rows: list[list[dict]] = []
    for l in ls:
        cy, h = (l["box"][1] + l["box"][3]) / 2, l["box"][3] - l["box"][1]
        if rows and abs(cy - rows[-1][0]["_cy"]) < 0.5 * h:
            rows[-1].append(l)
        else:
            rows.append([l])
        l["_cy"] = rows[-1][0].get("_cy", cy)
    return [l for r in rows for l in sorted(r, key=lambda l: l["box"][0])]


def align(lp) -> tuple[float, list]:
    """lp[i][j] = log P(target j | cell i). Monotone alignment, cells and targets may be skipped."""
    n, m = len(lp), len(lp[0]) if lp else 0
    NEG = -1e9
    D = [[NEG] * (m + 1) for _ in range(n + 1)]
    B = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == j == 0:
                continue
            best, arg = NEG, None
            if i and j and D[i - 1][j - 1] + lp[i - 1][j - 1] > best:
                best, arg = D[i - 1][j - 1] + lp[i - 1][j - 1], "m"
            if i and D[i - 1][j] + EXTRA > best:
                best, arg = D[i - 1][j] + EXTRA, "x"
            if j and D[i][j - 1] + MISS > best:
                best, arg = D[i][j - 1] + MISS, "s"
            D[i][j], B[i][j] = best, arg
    path, i, j = [], n, m
    while i or j:
        a = B[i][j]
        path.append((a, i - 1 if a in "mx" else None, j - 1 if a in "ms" else None))
        i, j = (i - 1, j - 1) if a == "m" else (i - 1, j) if a == "x" else (i, j - 1)
    return D[n][m], path[::-1]


class ClsReward:
    def __init__(self, model: Path, paddle=None):
        from syllable_cls import SylCls

        self.cls = SylCls(model)
        self._paddle = paddle

    @property
    def paddle(self):
        if self._paddle is None:
            import os
            import ocr_engines

            pp = Path.home() / "paddle-venv/bin/python"
            if pp.exists():
                os.environ.setdefault("REDROB_PADDLE_PY", str(pp))
            self._paddle = ocr_engines.make("paddleocr")
        return self._paddle

    def __call__(self, img, target: str, lines: list[dict] | None = None) -> dict:
        import torch

        if lines is None:
            lines = self.paddle(img)
        tgt = [c for c in target if is_syl(c)]
        cells = [c for l in order([dict(l) for l in lines]) for c in line_cells(img, l)]
        if not cells or not tgt:
            return dict(reward=MISS, n_cells=len(cells), read="", path=[])
        cells = cells[:4 * len(tgt) + 8]
        lp_all = self._logp_all(cells)  # [cells, 11172]
        idx = torch.tensor([ord(c) - 0xAC00 for c in tgt])
        lp = lp_all[:, idx].tolist()
        tot, path = align(lp)
        read = "".join(chr(0xAC00 + int(k)) for k in lp_all.argmax(1))
        return dict(reward=tot / len(tgt), n_cells=len(cells), read=read,
                    matched=sum(1 for a, _, _ in path if a == "m"), extra=sum(1 for a, _, _ in path if a == "x"),
                    missing=sum(1 for a, _, _ in path if a == "s"))

    def _logp_all(self, cells):
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from syllable_cls import SIZE, normalise

        x = np.stack([np.asarray(c.convert("L").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255
                      for c in cells])
        x = normalise(torch.from_numpy(x)[:, None].to(self.cls.dev))
        with torch.no_grad():
            return F.log_softmax(self.cls.net(x).float(), 1).cpu()


def load_rows(d: Path) -> list[dict]:
    """scores_dual.jsonl (eval, lines_paddleocr) or scores.jsonl from an --ocr paddleocr rollout (lines)."""
    p = d / "scores_dual.jsonl"
    if not p.exists():
        p = d / "scores.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r.setdefault("lines_paddleocr", r.get("lines"))
    return rows


def score(d: Path, model: Path) -> None:
    from PIL import Image

    rw = ClsReward(model)
    rows = load_rows(d)
    with (d / "rewards_cls.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            img = Image.open(d / "images" / f"{r['image_id']}.png").convert("RGB")
            res = rw(img, r["target_text"], r.get("lines_paddleocr"))
            f.write(json.dumps(dict(image_id=r["image_id"], id=r["id"], seed=r.get("seed"), split=r["split"],
                                    target_text=r["target_text"], dual_em=bool(r.get("em")), dual_cer=r.get("cer"),
                                    **res), ensure_ascii=False) + "\n")
    print("REWARD-CLS SCORE DONE", flush=True)


def sheet(d: Path, n: int, out: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(d)
    rng = random.Random(7)
    by = {}
    for r in rows:
        by.setdefault(r["split"], []).append(r)
    quota = {"seen": 7, "unseen": 7, "unseen_syllable": 6}
    pick = []
    for s, k in quota.items():
        pick += rng.sample(by.get(s, []), min(k, len(by.get(s, []))))
    rng.shuffle(pick)
    pick = pick[:n]
    font = ImageFont.truetype(str(Path("~/.local/share/fonts/NotoSansKR-Bold.ttf").expanduser()), 26)
    cw, ch, cols = 420, 470, 5
    canvas = Image.new("RGB", (cw * cols, ch * ((len(pick) + cols - 1) // cols)), "white")
    dr = ImageDraw.Draw(canvas)
    key = []
    for k, r in enumerate(pick):
        img = Image.open(d / "images" / f"{r['image_id']}.png").convert("RGB")
        img.thumbnail((cw - 20, cw - 20))
        x, y = (k % cols) * cw, (k // cols) * ch
        canvas.paste(img, (x + 10, y + 10))
        dr.text((x + 10, y + cw - 5), f"{k + 1}. {r['target_text']}"[:24], fill="black", font=font)
        key.append(dict(n=k + 1, image_id=r["image_id"], split=r["split"], target_text=r["target_text"]))
    canvas.save(out / "sheet.png")
    (out / "key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    print(f"sheet {len(pick)} -> {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--images", type=Path, required=True)
    s.add_argument("--model", type=Path, default=Path("~/runs/syl-cls-v2/model.pt"))
    h = sub.add_parser("sheet")
    h.add_argument("--images", type=Path, required=True)
    h.add_argument("--n", type=int, default=20)
    h.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "score":
        score(a.images.expanduser(), a.model.expanduser())
    else:
        sheet(a.images.expanduser(), a.n, a.out.expanduser())
    return 0


if __name__ == "__main__":
    sys.exit(main())
