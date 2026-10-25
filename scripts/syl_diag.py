#!/usr/bin/env python3
"""Why does the classifier misread clean stage-0 composite cells? Geometry vs appearance numbers."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path.home() / "redrob-image-t4/scripts"))
from reward import syllable_cells  # noqa: E402
import syllable_cls as S  # noqa: E402

d = Path.home() / "runs/reward-check"
pairs = {json.loads(l)["id"]: json.loads(l) for l in (d / "pairs.jsonl").read_text().splitlines() if l.strip()}
cls = S.SylCls(Path.home() / "runs/syl-cls/model.pt")
out = Path.home() / "runs/syl-cls/diag"
out.mkdir(exist_ok=True)
stats = []
tiles = []
for k, (pid, r) in enumerate(list(pairs.items())[:40]):
    img = Image.open(d / "images/good" / f"{pid}.png").convert("RGB")
    for c, s in syllable_cells(img, r):
        g = np.asarray(c.convert("L").resize((64, 64)), dtype=np.float32) / 255
        t = g.mean()
        ink = g > t if (g > t).mean() < 0.5 else g < t
        edge = np.concatenate([ink[:, :6], ink[:, -6:]], 1).mean()
        top = cls.top(c, 1)[0]
        stats.append(dict(id=pid, s=s, read=top[0], p=top[1], w=c.width, h=c.height,
                          contrast=float(g.std()), ink=float(ink.mean()), edge=float(edge)))
        if len(tiles) < 80:
            tiles.append((g * 255).astype(np.uint8))
ok = [x for x in stats if x["read"] == x["s"]]
bad = [x for x in stats if x["read"] != x["s"]]
for name, rs in (("ok", ok), ("bad", bad)):
    if rs:
        print(name, len(rs), {k: round(float(np.median([x[k] for x in rs])), 3) for k in ("w", "h", "contrast", "ink", "edge")})
for x in bad[:15]:
    print(x)
while len(tiles) < 80:
    tiles.append(np.zeros((64, 64), np.uint8))
Image.fromarray(np.concatenate([np.concatenate(tiles[i * 10:(i + 1) * 10], 1) for i in range(8)], 0)).save(out / "cells.png")
