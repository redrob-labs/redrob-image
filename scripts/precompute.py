#!/usr/bin/env python3
"""T7: pre-compute VAE latents and Qwen3-4B caption embeddings so the DiT trainer never loads either.

Sources (webdataset shards, <key>.jpg + <key>.json):
  --aihub DIR:CAPTIONS   AI Hub 105 shards + captions.composed.jsonl from scripts/caption.py (only captioned keys)
  --synth DIR            synth/generate.py shards (captions inside each json)
Each image is resized to cover the nearest aspect bucket at --res (area ~= res^2, sides multiple of 16, the same
aspects the crop search uses) and centre-cropped; every caption variant is embedded exactly as ZImagePipeline
does (chat template, hidden_states[-2], padding tokens dropped).

Output per input shard: <out>/<res>/<source>-<shard>.safetensors with "<i>.lat" (16, h/8, w/8) and
"<i>.txt.<variant>" (tokens, 2560), all bf16, latents already shifted and scaled for the DiT, plus
<out>/<res>/index.jsonl (one row per sample), <out>/empty.safetensors (embedding of "" for CFG dropout)
and <out>/<res>/speed.json. Finished shards are skipped, so a re-run resumes.

Usage (on the box):
    HF_HOME=~/hf python scripts/precompute.py --res 512 --out ~/data/precomp \
        --aihub ~/aihub/105/shards-v1:~/aihub/105/captions-20k/captions.composed.jsonl \
        --synth ~/data/synth-20k/shards
"""
from __future__ import annotations

import argparse
import io
import json
import math
import tarfile
import time
from collections import defaultdict
from pathlib import Path

ASPECTS = (1.0, 4 / 3, 3 / 4, 16 / 9, 9 / 16, 2.0, 0.5)
VARIANTS = ("ko_long", "ko_short", "en")


def buckets(res: int) -> list[tuple[int, int]]:
    out = []
    for ar in ASPECTS:
        w = round(res * math.sqrt(ar) / 16) * 16
        h = round(res / math.sqrt(ar) / 16) * 16
        out.append((w, h))
    return out


def nearest(w: int, h: int, bks):
    ar = w / h
    return min(bks, key=lambda b: abs(math.log(ar / (b[0] / b[1]))))


def fit(img, bw: int, bh: int):
    from PIL import Image

    s = max(bw / img.width, bh / img.height)
    nw, nh = max(bw, round(img.width * s)), max(bh, round(img.height * s))
    img = img.resize((nw, nh), Image.LANCZOS)
    x, y = (nw - bw) // 2, (nh - bh) // 2
    return img.crop((x, y, x + bw, y + bh))


def samples(tar: Path, captions: dict | None):
    from PIL import Image

    with tarfile.open(tar) as tf:
        members = {m.name: m for m in tf.getmembers()}
        for name in sorted(members):
            if not name.endswith(".json"):
                continue
            key = name[:-5]
            meta = json.load(tf.extractfile(members[name]))
            caps = captions.get(key) if captions is not None else meta.get("captions")
            if not caps:
                continue
            img = Image.open(io.BytesIO(tf.extractfile(members[key + ".jpg"]).read())).convert("RGB")
            yield key, meta, img, {v: caps[v] for v in VARIANTS if caps.get(v)}


class Encoders:
    def __init__(self, model: str):
        import torch
        from diffusers import AutoencoderKL
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.vae = AutoencoderKL.from_pretrained(model, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda").eval()
        self.tok = AutoTokenizer.from_pretrained(model, subfolder="tokenizer")
        self.te = AutoModel.from_pretrained(model, subfolder="text_encoder", torch_dtype=torch.bfloat16).to("cuda").eval()
        self.shift, self.scale = self.vae.config.shift_factor, self.vae.config.scaling_factor

    def latents(self, imgs):
        import numpy as np

        t = self.torch
        x = t.from_numpy(np.stack([np.asarray(i, dtype=np.float32) for i in imgs])).permute(0, 3, 1, 2)
        x = (x / 127.5 - 1.0).to("cuda", t.bfloat16)
        with t.inference_mode():
            z = self.vae.encode(x).latent_dist.mode()
        return ((z - self.shift) * self.scale).to(t.bfloat16).cpu()

    def text(self, prompts: list[str], max_len: int = 512):
        t = self.torch
        chat = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True) for p in prompts]
        enc = self.tok(chat, padding="longest", max_length=max_len, truncation=True, return_tensors="pt")
        ids, mask = enc.input_ids.to("cuda"), enc.attention_mask.to("cuda").bool()
        with t.inference_mode():
            h = self.te(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-2]
        return [h[i][mask[i]].to(t.bfloat16).cpu() for i in range(len(prompts))]


def text_boxes(meta: dict) -> list[tuple[float, float, float, float]]:
    """Text block boxes (x0, y0, x1, y1) in the stored image's pixels: parser shards carry bbox_xyxy, synth shards
    a polygon."""
    out = []
    for b in meta.get("text_blocks") or []:
        if "bbox_xyxy" in b:
            out.append(tuple(float(v) for v in b["bbox_xyxy"]))
        elif b.get("polygon"):
            xs, ys = [p[0] for p in b["polygon"]], [p[1] for p in b["polygon"]]
            out.append((min(xs), min(ys), max(xs), max(ys)))
    return out


def fit_boxes(boxes, w: int, h: int, bw: int, bh: int):
    """The same cover-resize + centre-crop as fit(), applied to boxes; drops boxes that end up outside."""
    s = max(bw / w, bh / h)
    nw, nh = max(bw, round(w * s)), max(bh, round(h * s))
    ox, oy = (nw - bw) // 2, (nh - bh) // 2
    out = []
    for x0, y0, x1, y1 in boxes:
        x0, x1 = max(0.0, x0 * s - ox), min(float(bw), x1 * s - ox)
        y0, y1 = max(0.0, y0 * s - oy), min(float(bh), y1 * s - oy)
        if x1 > x0 and y1 > y0:
            out.append((x0, y0, x1, y1))
    return out


def text_mask(boxes, bw: int, bh: int, f: int = 8):
    """uint8 (bh/f, bw/f) latent-grid mask, 1 on every latent cell a text box touches (the trainer up-weights the
    loss there)."""
    import torch

    m = torch.zeros(bh // f, bw // f, dtype=torch.uint8)
    for x0, y0, x1, y1 in boxes:
        m[int(y0 // f): int(math.ceil(y1 / f)), int(x0 // f): int(math.ceil(x1 / f))] = 1
    return m


def run_shard(enc: Encoders, src: str, tar: Path, captions, out_dir: Path, res: int, vae_batch: int, txt_batch: int):
    from safetensors.torch import save_file

    bks = buckets(res)
    rows, tensors = [], {}
    by_bucket = defaultdict(list)
    for key, meta, img, caps in samples(tar, captions):
        b = nearest(img.width, img.height, bks)
        i = len(rows)
        fb = fit_boxes(text_boxes(meta), img.width, img.height, *b)
        tensors[f"{i}.mask"] = text_mask(fb, *b).contiguous()
        rows.append({"i": i, "key": key, "source": src, "shard": tar.name, "bucket": list(b),
                     "kind": meta.get("kind", meta.get("scene", "")), "variants": list(caps),
                     # smallest text short side in trained pixels; the trainer can keep small text out of low-res stages
                     "min_text_px": round(min((min(x1 - x0, y1 - y0) for x0, y0, x1, y1 in fb), default=0.0), 1)})
        by_bucket[b].append((i, fit(img, *b), caps))
    for b, items in by_bucket.items():
        for s in range(0, len(items), vae_batch):
            chunk = items[s: s + vae_batch]
            for (i, _, _), z in zip(chunk, enc.latents([im for _, im, _ in chunk])):
                tensors[f"{i}.lat"] = z.contiguous()
    texts = [(i, v, c) for b in by_bucket.values() for i, _, caps in b for v, c in caps.items()]
    for s in range(0, len(texts), txt_batch):
        chunk = texts[s: s + txt_batch]
        for (i, v, _), e in zip(chunk, enc.text([c for _, _, c in chunk])):
            tensors[f"{i}.txt.{v}"] = e.contiguous()
    name = f"{src}-{tar.stem}"
    tmp = out_dir / f"{name}.safetensors.part"
    save_file(tensors, str(tmp))
    tmp.rename(out_dir / f"{name}.safetensors")
    for r in rows:
        r["file"] = f"{name}.safetensors"
    (out_dir / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return len(rows), sum(t.numel() * t.element_size() for t in tensors.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Tongyi-MAI/Z-Image")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--aihub", action="append", default=[], help="SHARD_DIR:CAPTIONS_JSONL[:SOURCE_NAME] (source defaults to aihub105)")
    ap.add_argument("--synth", action="append", default=[], help="SHARD_DIR")
    ap.add_argument("--vae-batch", type=int, default=8)
    ap.add_argument("--txt-batch", type=int, default=32)
    ap.add_argument("--max-shards", type=int, default=0, help="per source, for a speed test")
    ap.add_argument("--max-per-source", type=int, default=0, help="stop a source once this many samples are cached")
    a = ap.parse_args()
    out = a.out.expanduser() / str(a.res)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for spec in a.aihub:
        d, cap, *name = spec.split(":", 2)
        name = name[0] if name else "aihub105"
        caps = {}
        for ln in open(Path(cap).expanduser(), encoding="utf-8"):
            r = json.loads(ln)
            caps[r["key"]] = r["captions"]
        tars = sorted({Path(d).expanduser() / "train" / r for r in
                       {json.loads(ln)["shard"] for ln in open(Path(cap).expanduser(), encoding="utf-8")}})
        jobs += [(name, t, caps) for t in (tars[: a.max_shards] if a.max_shards else tars)]
    for d in a.synth:
        tars = sorted(Path(d).expanduser().glob("*.tar"))
        jobs += [("synth", t, None) for t in (tars[: a.max_shards] if a.max_shards else tars)]

    enc = Encoders(a.model)
    from safetensors.torch import save_file

    empty = a.out.expanduser() / "empty.safetensors"
    if not empty.exists():
        save_file({"txt": enc.text([""])[0].contiguous()}, str(empty))

    speed = {"res": a.res, "shards": 0, "samples": 0, "bytes": 0, "seconds": 0.0}
    per_src: dict = {}
    for js in out.glob("*.json"):  # samples already cached, so --max-per-source holds across resumes
        if js.name != "speed.json":
            src = js.name.split("-", 1)[0]
            per_src[src] = per_src.get(src, 0) + len(json.loads(js.read_text(encoding="utf-8")))
    for src, tar, caps in jobs:
        if (out / f"{src}-{tar.stem}.safetensors").exists():
            continue
        if a.max_per_source and per_src.get(src, 0) >= a.max_per_source:
            continue
        t0 = time.time()
        n, nbytes = run_shard(enc, src, tar, caps, out, a.res, a.vae_batch, a.txt_batch)
        per_src[src] = per_src.get(src, 0) + n
        dt = time.time() - t0
        speed.update(shards=speed["shards"] + 1, samples=speed["samples"] + n, bytes=speed["bytes"] + nbytes,
                     seconds=round(speed["seconds"] + dt, 1))
        print(f"{src}-{tar.stem}: {n} samples {nbytes / 2**20:.0f} MiB {dt:.0f}s", flush=True)

    with open(out / "index.jsonl", "w", encoding="utf-8") as f:
        for js in sorted(out.glob("*.json")):
            if js.name == "speed.json":
                continue
            for r in json.loads(js.read_text(encoding="utf-8")):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if speed["samples"]:
        speed["samples_per_sec"] = round(speed["samples"] / max(speed["seconds"], 1e-6), 2)
        speed["mib_per_sample"] = round(speed["bytes"] / speed["samples"] / 2**20, 3)
        (out / "speed.json").write_text(json.dumps(speed, indent=2), encoding="utf-8")
    print(json.dumps(speed), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
