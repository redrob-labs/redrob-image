#!/usr/bin/env python3
"""How short text is shown to the trainer: per Hangul-length bucket, how many text blocks there are and how big
they are at the trained resolution. Reads only the <key>.json metadata inside webdataset shards (parser or
synth output), so it runs on CPU next to training. Numbers only, safe to commit.

A sample is trained at area ~= res^2 (precompute.py buckets), so a block's short side in trained pixels is
min(w, h) * res / sqrt(W * H) of the sample.

    python scripts/length_stats.py --shards ~/aihub/105/shards-v1/train --name aihub105 --out paper/data-stats/length-aihub105.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import tarfile
from pathlib import Path


def hangul(s: str) -> int:
    return sum("\uac00" <= c <= "\ud7a3" for c in s)


def bucket(n: int) -> str:
    return "0" if n == 0 else "1-2" if n <= 2 else "3-5" if n <= 5 else "6-10" if n <= 10 else "11+"


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))], 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    px = collections.defaultdict(list)          # bucket -> short side in trained px
    area = collections.defaultdict(list)        # bucket -> block area share of the sample
    alone = collections.Counter()               # samples whose only text block is in this bucket
    samples, blocks = 0, collections.Counter()
    for t in sorted(a.shards.expanduser().glob("*.tar")):
        with tarfile.open(t) as tf:
            for m in tf:
                if not m.name.endswith(".json"):
                    continue
                d = json.load(tf.extractfile(m))
                W, H = d.get("width") or d.get("size", [0, 0])[0], d.get("height") or d.get("size", [0, 0])[1]
                tb = d.get("text_blocks") or []
                if not W or not H or not tb:
                    continue
                samples += 1
                s = a.res / math.sqrt(W * H)
                for b in tb:
                    if "bbox_xyxy" in b:
                        x0, y0, x1, y1 = b["bbox_xyxy"]
                    else:  # synth/generate.py stores a polygon
                        xs, ys = [p[0] for p in b["polygon"]], [p[1] for p in b["polygon"]]
                        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                    k = bucket(hangul(b.get("text", "")))
                    blocks[k] += 1
                    px[k].append(min(x1 - x0, y1 - y0) * s)
                    area[k].append((x1 - x0) * (y1 - y0) / (W * H))
                if len(tb) == 1:
                    alone[bucket(hangul(tb[0].get("text", "")))] += 1
    rep = {"name": a.name, "res": a.res, "samples": samples, "buckets": {}}
    for k in ("1-2", "3-5", "6-10", "11+", "0"):
        rep["buckets"][k] = {"blocks": blocks[k], "samples_with_only_this": alone[k],
                             "short_side_px_p10": pct(px[k], 0.1), "short_side_px_p50": pct(px[k], 0.5),
                             "short_side_px_p90": pct(px[k], 0.9),
                             "area_share_p50": round(pct([x * 1000 for x in area[k]], 0.5) / 1000, 4)}
    js = json.dumps(rep, indent=1, ensure_ascii=False)
    print(js)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(js + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
