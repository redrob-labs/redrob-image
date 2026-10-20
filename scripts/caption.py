#!/usr/bin/env python3
"""T5 captioning for AI Hub 105 shards with Qwen3-VL (Apache-2.0).

The VLM never writes the text content. It is told only WHERE each text region is and returns a scene
description plus a per-region style (color, weight, typeface). The captions are then composed in code with
the OCR label text inserted verbatim, so a misread by the VLM cannot reach a caption. Scene strings that
still contain a label text (>= 2 syllables) or a quote are replaced by a plain category phrase and counted.

Variants per sample (same keys as synth/generate.py): ko_long, ko_short, en (text stays Korean).

Outputs stay on the box (AI Hub derived): <out>/captions.jsonl, <out>/preview.html (thumbnails inline).
Numbers only for git: paper/data-stats/<name>.json.

Usage:
    ~/redrob-image-venv/bin/python scripts/caption.py --shards ~/aihub/105/shards-v1 --out ~/aihub/105/captions-1k --n 1000
"""
from __future__ import annotations

import argparse
import base64
import collections
import html
import io
import json
import random
import re
import tarfile
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "paper" / "data-stats"

POS_WORDS = [[("왼쪽 위", "top left"), ("위쪽 가운데", "top center"), ("오른쪽 위", "top right")],
             [("왼쪽", "middle left"), ("가운데", "center"), ("오른쪽", "middle right")],
             [("왼쪽 아래", "bottom left"), ("아래쪽 가운데", "bottom center"), ("오른쪽 아래", "bottom right")]]
CATEGORY = {"간판": ("간판", "a Korean signboard"), "책표지": ("책 표지", "a Korean book cover")}
SIZE_WORDS = [(0.25, "아주 큰", "very large"), (0.12, "큰", "large"), (0.05, "중간 크기", "medium"), (0.0, "작은", "small")]

PROMPT = """You write descriptions of photos for training an image generator.
This photo shows {cat_en}. It contains {n} text region(s), listed by id, position and size:
{regions}
Return ONLY a JSON object, no markdown:
{{"scene_ko": "<one Korean sentence: the object, background, colors, lighting, camera angle>",
 "scene_short_ko": "<short Korean noun phrase, e.g. 파란 식당 간판>",
 "scene_en": "<the same as scene_ko, in English>",
 "regions": [{{"id": 0, "style_ko": "<color, weight and typeface of that text in Korean, e.g. 흰색 굵은 고딕체>", "style_en": "<the same in English, e.g. bold white sans-serif>"}}]}}
Rules: never write, quote, translate or guess the words of any text in the image; describe how text looks, not what it says.
Give one regions entry for every id."""


def josa_rago(t: str) -> str:
    """'라고' after a vowel, '이라고' after a final consonant (or a digit/Latin that reads with one, kept simple)."""
    ch = t.rstrip()[-1:] or "a"
    if "\uac00" <= ch <= "\ud7a3":
        return "이라고" if (ord(ch) - 0xAC00) % 28 else "라고"
    return "라고"


def pos_words(b, W, H):
    cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
    col = 0 if cx < 0.36 else (2 if cx > 0.64 else 1)
    row = 0 if cy < 0.36 else (2 if cy > 0.64 else 1)
    return POS_WORDS[row][col]


def size_words(b, W, H):
    r = min(b[2] - b[0], b[3] - b[1]) / min(W, H)
    for lo, ko, en in SIZE_WORDS:
        if r >= lo:
            return ko, en
    return SIZE_WORDS[-1][1:]


def pick(shards: Path, n: int, seed: int) -> list[tuple[str, str]]:
    """Stratified pick over (category, kind) from the train shards; returns (tar, key)."""
    groups = collections.defaultdict(list)
    for tar in sorted(shards.glob("train/*.tar")):
        with tarfile.open(tar) as tf:
            for m in tf.getmembers():
                if m.name.endswith(".json"):
                    meta = json.load(tf.extractfile(m))
                    groups[(meta["category"], meta["kind"])].append((str(tar), m.name[:-5]))
    rng = random.Random(seed)
    for g in groups.values():
        rng.shuffle(g)
    out, keys = [], sorted(groups)
    while len(out) < n and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


def load(tar: str, key: str):
    with tarfile.open(tar) as tf:
        meta = json.load(tf.extractfile(f"{key}.json"))
        img = Image.open(io.BytesIO(tf.extractfile(f"{key}.jpg").read())).convert("RGB")
    return meta, img


def parse_json(s: str) -> dict | None:
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def clean(s, fallback: str, texts: list[str], hangul_ok: bool, why: collections.Counter, tag: str) -> str:
    s = (s or "").strip().strip(".").strip()
    bad = None
    if not s:
        bad = "empty"
    elif any(q in s for q in "\"'“”‘’「」『』"):
        bad = "quote"
    elif any(len(t) >= 2 and t.replace(" ", "") in s.replace(" ", "") for t in texts):
        bad = "label_text"
    elif not hangul_ok and re.search(r"[\uac00-\ud7a3]", s):
        bad = "hangul_in_en"
    elif len(s) > 300:
        bad = "too_long"
    if bad:
        why[f"{tag}:{bad}"] += 1
        return fallback
    return s


def compose(meta: dict, vlm: dict | None, why: collections.Counter) -> dict:
    W, H = meta["width"], meta["height"]
    cat_ko, cat_en = CATEGORY.get(meta["category"], ("장면", "a scene"))
    blocks = sorted(meta["text_blocks"], key=lambda b: -((b["bbox_xyxy"][2] - b["bbox_xyxy"][0]) * (b["bbox_xyxy"][3] - b["bbox_xyxy"][1])))
    texts = [b["text"] for b in blocks]
    vlm = vlm or {}
    if not vlm:
        why["vlm:unparsed"] += 1
    styles = {r.get("id"): r for r in vlm.get("regions", []) if isinstance(r, dict)}
    scene_ko = clean(vlm.get("scene_ko"), f"{cat_ko} 사진", texts, True, why, "scene_ko")
    short_ko = clean(vlm.get("scene_short_ko"), cat_ko, texts, True, why, "scene_short_ko")
    scene_en = clean(vlm.get("scene_en"), f"A photo of {cat_en}", texts, False, why, "scene_en")
    ko, en = [], []
    for b in blocks:
        t = b["text"]
        r = styles.get(b["index"], {})
        sko = clean(r.get("style_ko"), "", texts, True, why, "style_ko")
        sen = clean(r.get("style_en"), "", texts, False, why, "style_en")
        pko, pen = pos_words(b["bbox_xyxy"], W, H)
        zko, zen = size_words(b["bbox_xyxy"], W, H)
        sko = sko.replace(", ", " ").replace(",", " ")
        lead = (zen + " " + sen).strip()
        ko.append(f"{pko}에 {zko} {sko + ' ' if sko else ''}글씨로 \"{t}\"{josa_rago(t)} 적혀 있다.")
        en.append(f"{lead[:1].upper() + lead[1:]} Korean text at the {pen} reads \"{t}\".")
    return {"ko_long": f"{scene_ko}. " + " ".join(ko),
            "ko_short": f"{short_ko}, 문구: " + ", ".join(f'"{t}"' for t in texts),
            "en": f"{scene_en}. " + " ".join(en)}


class VLM:
    def __init__(self, model_id: str, max_side: int):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch, self.max_side = torch, max_side
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.proc.tokenizer.padding_side = "left"

    def __call__(self, items: list[tuple[Image.Image, str]]) -> list[str]:
        msgs = []
        for img, prompt in items:
            if max(img.size) > self.max_side:
                s = self.max_side / max(img.size)
                img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
            msgs.append([{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}])
        inputs = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=400, do_sample=False)
        return self.proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def prompt_for(meta: dict) -> str:
    W, H = meta["width"], meta["height"]
    lines = []
    for b in meta["text_blocks"]:
        _, pen = pos_words(b["bbox_xyxy"], W, H)
        _, zen = size_words(b["bbox_xyxy"], W, H)
        lines.append(f"{b['index']}: {pen}, {zen}")
    cat_en = CATEGORY.get(meta["category"], ("", "a scene"))[1]
    return PROMPT.format(cat_en=cat_en, n=len(lines), regions="\n".join(lines))


def thumb(img: Image.Image, side=384) -> str:
    im = img.copy()
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--preview", type=int, default=200)
    ap.add_argument("--name", default="captions-1k")
    a = ap.parse_args()
    shards, out = a.shards.expanduser(), a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    picks = pick(shards, a.n, a.seed)
    done = {}
    cap_path = out / "captions.jsonl"
    if cap_path.exists():  # resume
        for ln in open(cap_path, encoding="utf-8"):
            r = json.loads(ln)
            done[r["key"]] = r
    todo = [p for p in picks if p[1] not in done]
    print(f"{len(picks)} picked, {len(done)} already captioned, {len(todo)} to go", flush=True)

    why = collections.Counter()
    t0 = time.time()
    if todo:
        vlm = VLM(a.model, a.max_side)
        with open(cap_path, "a", encoding="utf-8") as f:
            for s in range(0, len(todo), a.batch):
                chunk = [(p, *load(*p)) for p in todo[s: s + a.batch]]
                raws = vlm([(img, prompt_for(meta)) for _, meta, img in chunk])
                for (p, meta, img), raw in zip(chunk, raws):
                    r = {"key": p[1], "shard": Path(p[0]).name, "kind": meta["kind"], "category": meta["category"],
                         "vlm_raw": raw, "vlm": parse_json(raw)}
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    done[p[1]] = r
                f.flush()
                print(f"{min(s + a.batch, len(todo))}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    elapsed = time.time() - t0

    # compose captions (always from the current code, so the template can change without re-running the VLM)
    rows = []
    for tar, key in picks:
        meta, img = load(tar, key)
        r = done[key]
        r["captions"] = compose(meta, r["vlm"], why)
        r["texts"] = [b["text"] for b in meta["text_blocks"]]
        rows.append((r, img))
    with open(out / "captions.composed.jsonl", "w", encoding="utf-8") as f:
        for r, _ in rows:
            f.write(json.dumps({k: r[k] for k in ("key", "shard", "kind", "category", "captions")}, ensure_ascii=False) + "\n")

    # every label text must appear verbatim, inside quotes, in every variant
    missing = sum(1 for r, _ in rows for v in r["captions"].values() for t in r["texts"] if f'"{t}"' not in v)

    parts = ["<!doctype html><meta charset=utf-8><title>caption preview</title>",
             "<style>body{font:14px sans-serif;margin:16px}.r{display:flex;gap:12px;border-bottom:1px solid #ccc;padding:8px 0}"
             "img{width:384px;height:auto;flex:none}p{margin:4px 0}b{color:#555}</style>",
             f"<h1>Caption preview ({min(a.preview, len(rows))} of {len(rows)})</h1>",
             "<p>Check: does the scene sentence match the photo, and does each style match its text?</p>"]
    for r, img in rows[: a.preview]:
        c = r["captions"]
        parts.append(f"<div class=r><img alt='{html.escape(r['key'])}' src='data:image/jpeg;base64,{thumb(img)}'><div>"
                     f"<p><b>{html.escape(r['category'])} / {r['kind']}</b> {html.escape(r['key'])}</p>"
                     + "".join(f"<p><b>{k}</b> {html.escape(v)}</p>" for k, v in c.items()) + "</div></div>")
    (out / "preview.html").write_text("\n".join(parts), encoding="utf-8")

    lens = {k: sorted(len(r["captions"][k]) for r, _ in rows) for k in ("ko_long", "ko_short", "en")}
    stats = {"name": a.name, "model": a.model, "samples": len(rows),
             "by_kind": dict(collections.Counter(r["kind"] for r, _ in rows)),
             "by_category": dict(collections.Counter(r["category"] for r, _ in rows)),
             "vlm_json_parsed": sum(1 for r, _ in rows if r["vlm"]),
             "fallbacks": dict(sorted(why.items())),
             "label_text_missing_from_caption": missing,
             "caption_chars_median": {k: v[len(v) // 2] for k, v in lens.items()},
             "vlm_seconds_this_run": round(elapsed, 1),
             "vlm_images_this_run": len(todo),
             "vlm_seconds_per_image": round(elapsed / len(todo), 2) if todo else None,
             "settings": {"batch": a.batch, "max_side": a.max_side, "seed": a.seed}}
    STATS.mkdir(parents=True, exist_ok=True)
    (STATS / f"{a.name}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
