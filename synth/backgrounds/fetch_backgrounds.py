#!/usr/bin/env python3
"""Download CC0 photo/texture backgrounds from Poly Haven into data/backgrounds/.

Poly Haven publishes every asset under CC0 1.0 (https://polyhaven.com/license): no attribution
required, commercial use and ML training allowed. We still record the asset id, authors, source
URL and md5 per file in synth/backgrounds/manifest.json (committed; images are not).

Two kinds:
  texture  - the 2k diffuse (albedo) JPG of a PBR texture set (walls, wood, metal, fabric, ground)
  hdri     - the tonemapped JPG of an HDRI panorama; we keep 3 crops from the horizon band
             (where buildings, rooms and landscapes are), resized to 2048x1024

Usage:
    .venv/bin/python synth/backgrounds/fetch_backgrounds.py --textures 330 --hdris 60
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "backgrounds"
MANIFEST = Path(__file__).resolve().parent / "manifest.json"
API = "https://api.polyhaven.com"
UA = {"User-Agent": "redrob-image-synth/1.0 (research dataset build)"}
LICENSE = {"id": "CC0-1.0", "url": "https://polyhaven.com/license",
           "statement": "All assets on Poly Haven are CC0: free for any use, including commercial, no attribution required."}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def get_json(url: str):
    return json.loads(get(url))


def do_texture(aid: str, meta: dict) -> list[dict]:
    dst = OUT / f"ph_tex_{aid}.jpg"
    files = get_json(f"{API}/files/{aid}")
    d = files.get("Diffuse", {})
    ent = (d.get("2k") or d.get("1k") or {}).get("jpg")
    if not ent:
        return []
    if not dst.exists():
        data = get(ent["url"])
        if hashlib.md5(data).hexdigest() != ent["md5"]:
            raise RuntimeError(f"md5 mismatch {aid}")
        dst.write_bytes(data)
    return [{"file": dst.name, "kind": "texture", "asset": aid, "name": meta.get("name"),
             "authors": list(meta.get("authors", {})), "categories": meta.get("categories", []),
             "source_url": ent["url"], "asset_page": f"https://polyhaven.com/a/{aid}", "md5": ent["md5"],
             "license": LICENSE["id"]}]


def do_hdri(aid: str, meta: dict) -> list[dict]:
    names = [OUT / f"ph_hdri_{aid}_{i}.jpg" for i in range(3)]
    files = get_json(f"{API}/files/{aid}")
    ent = files.get("tonemapped")
    if not ent:
        return []
    if not all(n.exists() for n in names):
        data = get(ent["url"])
        if hashlib.md5(data).hexdigest() != ent["md5"]:
            raise RuntimeError(f"md5 mismatch {aid}")
        im = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = im.size
        # horizon band: middle 50% of height; three non-overlapping-ish crops of 2:1 aspect
        bh = h // 2
        bw = bh * 2
        y0 = h // 4
        for i, n in enumerate(names):
            x0 = int((w - bw) * (i + 0.5) / 3)
            im.crop((x0, y0, x0 + bw, y0 + bh)).resize((2048, 1024), Image.LANCZOS).save(n, quality=92)
    return [{"file": n.name, "kind": "hdri_crop", "asset": aid, "name": meta.get("name"),
             "authors": list(meta.get("authors", {})), "categories": meta.get("categories", []),
             "source_url": ent["url"], "asset_page": f"https://polyhaven.com/a/{aid}", "md5": ent["md5"],
             "crop": i, "license": LICENSE["id"]} for i, n in enumerate(names)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--textures", type=int, default=330)
    ap.add_argument("--hdris", type=int, default=60)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    tex = get_json(f"{API}/assets?t=textures")
    hdr = get_json(f"{API}/assets?t=hdris")
    tex_ids = sorted(tex)
    hdr_ids = sorted(hdr)
    rng.shuffle(tex_ids)
    rng.shuffle(hdr_ids)
    jobs = [(do_texture, i, tex[i]) for i in tex_ids[: a.textures]] + [(do_hdri, i, hdr[i]) for i in hdr_ids[: a.hdris]]
    recs: list[dict] = []
    with ThreadPoolExecutor(a.threads) as ex:
        for res in ex.map(lambda j: _safe(*j), jobs):
            recs.extend(res)
    recs.sort(key=lambda r: r["file"])
    MANIFEST.write_text(json.dumps({"source": "Poly Haven", "license": LICENSE, "count": len(recs),
                                    "items": recs}, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(recs)} backgrounds ({sum(r['kind'] == 'texture' for r in recs)} textures, "
          f"{sum(r['kind'] == 'hdri_crop' for r in recs)} hdri crops) -> {MANIFEST}")
    return 0


def _safe(fn, aid, meta):
    try:
        return fn(aid, meta)
    except Exception as e:  # network hiccup on one asset should not abort the batch
        print(f"skip {aid}: {e}", file=sys.stderr)
        return []


if __name__ == "__main__":
    sys.exit(main())
