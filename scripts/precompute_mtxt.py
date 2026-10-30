"""Masked-caption embeddings for the glyph-channel pilot (configs/g6e-glyph-ch2.yaml).

The ch1 swap test drew the PROMPT text 10/10 times: prompt text and glyph always agreed in training, so the model
never needed the glyph. Here every quoted text span in a caption is replaced with a placeholder ("*"), as AnyText
does, so on those batches the glyph is the only place the text can come from.

Reads an existing precompute dir (scripts/precompute.py) and writes, next to every <name>.safetensors, a sidecar
<name>.mtxt.safetensors holding <i>.mtxt.<variant>. Latents and glyphs are untouched. Resumable per shard.
  HF_HOME=~/hf python scripts/precompute_mtxt.py --precomp ~/data/precomp-glyph --res 512 \
      --aihub-captions ~/aihub/105/captions-20k/captions.composed.jsonl --synth ~/data/synth-20k/shards
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PLACEHOLDER = "*"
QUOTED = re.compile(r'"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’')


def mask_caption(c: str) -> tuple[str, int]:
    out, n = QUOTED.subn(lambda m: m.group(0)[0] + PLACEHOLDER + m.group(0)[-1], c)
    return out, n


def synth_captions(tar: Path) -> dict:
    out = {}
    with tarfile.open(tar) as tf:
        for m in tf.getmembers():
            if m.name.endswith(".json"):
                meta = json.load(tf.extractfile(m))
                out[m.name[:-5]] = meta.get("captions") or {}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Tongyi-MAI/Z-Image")
    ap.add_argument("--precomp", required=True, type=Path)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--aihub-captions", type=Path, help="captions jsonl for the aihub105 rows")
    ap.add_argument("--synth", type=Path, help="synth shard dir (captions inside each json)")
    ap.add_argument("--txt-batch", type=int, default=32)
    a = ap.parse_args()
    d = a.precomp.expanduser() / str(a.res)

    aihub = {}
    if a.aihub_captions:
        for ln in open(a.aihub_captions.expanduser(), encoding="utf-8"):
            r = json.loads(ln)
            aihub[r["key"]] = r["captions"]

    from safetensors.torch import save_file

    from precompute import Encoders

    enc = Encoders(a.model)
    stats = {"rows": 0, "texts": 0, "no_quote": 0, "missing_caption": 0}
    synth_cache: dict = {}
    for js in sorted(d.glob("*.json")):
        if js.name == "speed.json":
            continue
        name = js.name[:-5]
        dst = d / f"{name}.mtxt.safetensors"
        if dst.exists():
            continue
        t0 = time.time()
        rows = json.loads(js.read_text(encoding="utf-8"))
        texts = []
        for r in rows:
            if r["source"] == "synth":
                if r["shard"] not in synth_cache:
                    synth_cache.clear()  # one tar at a time; rows of a shard file come from one tar
                    synth_cache[r["shard"]] = synth_captions(a.synth.expanduser() / r["shard"])
                caps = synth_cache[r["shard"]].get(r["key"])
            else:
                caps = aihub.get(r["key"])
            if not caps:
                stats["missing_caption"] += 1
                continue
            stats["rows"] += 1
            for v in r["variants"]:
                if not caps.get(v):
                    continue
                m, n = mask_caption(caps[v])
                stats["texts"] += 1
                if n == 0:  # nothing to hide; the plain caption already says no text, keep it as it is
                    stats["no_quote"] += 1
                texts.append((r["i"], v, m))
        tensors = {}
        for s in range(0, len(texts), a.txt_batch):
            chunk = texts[s: s + a.txt_batch]
            for (i, v, _), e in zip(chunk, enc.text([c for _, _, c in chunk])):
                tensors[f"{i}.mtxt.{v}"] = e.contiguous()
        tmp = d / f"{name}.mtxt.safetensors.part"
        save_file(tensors, str(tmp))
        tmp.rename(dst)
        print(f"{name}: {len(tensors)} masked captions {time.time() - t0:.0f}s", flush=True)
    print(json.dumps(stats), flush=True)
    if texts_example := [mask_caption(c)[0] for c in list((next(iter(aihub.values()), {}) or {}).values())[:1]]:
        print("example:", texts_example[0], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
