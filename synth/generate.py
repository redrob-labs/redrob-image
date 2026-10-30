#!/usr/bin/env python3
"""SynthDoG-style Korean text image generator (signs, posters, book covers, labels, menus, notices).

Two phases, both deterministic for a given --seed:
  1. plan   (main process): scene, canvas size and text blocks for every image, then a
             dataset-level coverage pass so every KS X 1001 syllable occurs >= --n times
             (and, with --stage 2, every other non-held-out syllable >= --n2 times).
  2. render (--workers processes): background, panels, text with font/size/color/stroke/shadow,
             curve + perspective per block, global warp/blur/noise/JPEG, then JSON metadata
             with polygons and three captions (Korean long, Korean short, English).
Outputs (gitignored): synth/out/flat/<key>.jpg|.json, synth/out/shards/synth-XXXXX.tar
(webdataset: <key>.jpg + <key>.json), synth/out/plan.json, synth/out/run.json.

Usage:
    .venv/bin/python synth/generate.py --count 10000 --workers 16 --seed 6 --stage 2
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import math
import random
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_source import FILLER, KS, KS_SET, SCENES, TextSource, coverage_block_text  # noqa: E402
from unseen import UnseenFilter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "synth" / "out"
FONT_DIR = ROOT / "data" / "fonts"
BG_DIR = ROOT / "data" / "backgrounds"
FONT_MANIFEST = ROOT / "synth" / "fonts" / "manifest.json"
BG_MANIFEST = ROOT / "synth" / "backgrounds" / "manifest.json"

SCENE_WORDS = {
    "sign": ("간판", "storefront sign"), "poster": ("포스터", "poster"), "book_cover": ("책 표지", "book cover"),
    "product_label": ("상품 라벨", "product label"), "menu": ("메뉴판", "menu board"), "notice": ("안내문", "printed notice"),
}
STYLE_WORDS = {
    "gothic": ("고딕체", "sans-serif"), "myeongjo": ("명조체", "serif"), "rounded": ("둥근 글씨체", "rounded"),
    "handwriting": ("손글씨체", "handwritten"), "brush": ("붓글씨체", "brush-script"), "display": ("굵은 장식 글씨체", "bold display"),
}
COLOR_NAMES = [
    ((20, 20, 20), "검은색", "black"), ((245, 245, 245), "흰색", "white"), ((200, 30, 35), "빨간색", "red"),
    ((235, 120, 20), "주황색", "orange"), ((245, 210, 40), "노란색", "yellow"), ((40, 150, 60), "초록색", "green"),
    ((30, 90, 200), "파란색", "blue"), ((25, 35, 90), "남색", "navy"), ((120, 50, 160), "보라색", "purple"),
    ((235, 120, 170), "분홍색", "pink"), ((120, 75, 40), "갈색", "brown"), ((128, 128, 128), "회색", "gray"),
    ((200, 165, 60), "금색", "gold"), ((120, 190, 235), "하늘색", "sky blue"), ((20, 140, 140), "청록색", "teal"),
    ((225, 205, 170), "베이지색", "beige"), ((110, 20, 30), "자주색", "maroon"), ((60, 60, 60), "진회색", "dark gray"),
]
POS_WORDS = [[("왼쪽 위", "top left"), ("위쪽 가운데", "top center"), ("오른쪽 위", "top right")],
             [("왼쪽", "middle left"), ("가운데", "center"), ("오른쪽", "middle right")],
             [("왼쪽 아래", "bottom left"), ("아래쪽 가운데", "bottom center"), ("오른쪽 아래", "bottom right")]]
BG_CAT_WORDS = [("wood", "나무", "wood"), ("brick", "벽돌", "brick"), ("concrete", "콘크리트", "concrete"),
                ("plaster", "회벽", "plaster"), ("metal", "금속", "metal"), ("fabric", "천", "fabric"),
                ("rock", "돌", "stone"), ("sand", "모래", "sand"), ("asphalt", "아스팔트", "asphalt"),
                ("tiles", "타일", "tile"), ("terrain", "흙바닥", "ground"), ("roofing", "지붕", "roofing"),
                ("bark", "나무껍질", "bark"), ("floor", "바닥", "floor"), ("wall", "벽", "wall")]


# ============================================================ plan
def syllable_counts(plan: list[dict]) -> collections.Counter:
    c = collections.Counter()
    for item in plan:
        for b in item["blocks"]:
            for ch in b["text"]:
                if "가" <= ch <= "힣":
                    c[ch] += 1
    return c


def make_plan(n: int, seed: int, n_ks: int, stage: int, n2: int, ts: TextSource) -> tuple[list[dict], dict]:
    plan = []
    for i in range(n):
        rng = random.Random(f"{seed}:plan:{i}")
        scene = rng.choice(SCENES)
        r = rng.random()
        size = (1024, 1024) if r < 0.7 else ((1024, 768) if r < 0.85 else (768, 1024))
        if scene == "book_cover" and rng.random() < 0.6:
            size = (768, 1024)
        plan.append({"key": f"{i:06d}", "index": i, "scene": scene, "size": list(size),
                     "blocks": ts.scene_blocks(scene, rng)})
    natural = syllable_counts(plan)
    # A syllable that is itself a held-out word (삯, 홰) can never be rendered: no target for it.
    targets = {s: n_ks for s in KS if ts.f.ok(s)}
    if stage >= 2:
        for c in range(0xAC00, 0xD7A4):
            s = chr(c)
            if s not in KS_SET and not ts.f.bad_syllable(s):
                targets[s] = n2
    rng = random.Random(f"{seed}:coverage")
    order = list(range(n))
    rng.shuffle(order)
    cursor = 0
    added_words = 0
    for _round in range(6):
        counts = syllable_counts(plan)
        deficit = {s: t - counts.get(s, 0) for s, t in targets.items() if counts.get(s, 0) < t}
        if not deficit:
            break
        words = ts.coverage_words(deficit, rng, FILLER)
        print(f"coverage round {_round}: {len(deficit)} syllables short, {len(words)} words", flush=True)
        added_words += len(words)
        groups, j = [], 0
        while j < len(words):
            k = rng.choice([1, 2, 2, 3])
            g = [words[j]]
            j += 1
            # grow the group only while the joined text stays clean (two clean words can
            # form a held-out word across the boundary once spaces are ignored)
            while len(g) < k and j < len(words) and ts.f.ok(" ".join(g + [words[j]])):
                g.append(words[j])
                j += 1
            groups.append(g)
        for chunk in groups:
            for _try in range(4 * n):
                item = plan[order[cursor % n]]
                cursor += 1
                role, text = coverage_block_text(item["scene"], chunk, rng)
                if not ts.f.ok(text):
                    # context words formed a held-out word across a boundary: keep the bare words
                    role, text = "cov_tag", " ".join(chunk)
                if not ts.f.ok(text):
                    continue
                covs = [b for b in item["blocks"] if b.get("coverage")]
                if not covs:
                    blk = {"role": role, "text": text, "coverage": True}
                    if len(item["blocks"]) >= 4:
                        item["blocks"][-1] = blk  # the last block is always an optional one
                    else:
                        item["blocks"].append(blk)
                    break
                merged = covs[0]["text"] + "\n" + text
                if covs[0]["text"].count("\n") < 5 and ts.f.ok(merged):
                    covs[0]["text"] = merged
                    break
            else:
                raise RuntimeError("could not place coverage words")
    final = syllable_counts(plan)
    info = {"natural_counts_min_ks": min(natural.get(s, 0) for s in KS),
            "ks_below_n_before_coverage": sum(1 for s in KS if natural.get(s, 0) < n_ks),
            "coverage_words_added": added_words,
            "ks_held_out_as_words": [s for s in KS if not ts.f.ok(s)],
            "ks_below_n_after": sum(1 for s in KS if final.get(s, 0) < n_ks and ts.f.ok(s)),
            "stage2_below_n2_after": sum(1 for s, t in targets.items() if s not in KS_SET and final.get(s, 0) < t)}
    return plan, info


# ============================================================ render helpers
_FONTS: list[dict] | None = None
_BGS: list[dict] | None = None
_FILTER: UnseenFilter | None = None


def init_worker():
    global _FONTS, _BGS, _FILTER
    cv2.setNumThreads(1)
    man = json.loads(FONT_MANIFEST.read_text())
    _FONTS = []
    for fam in man["families"]:
        for f in fam["files"]:
            p = FONT_DIR / f["file"]
            cmap = frozenset(TTFont(p, lazy=True).getBestCmap())
            _FONTS.append({"path": str(p), "file": f["file"], "family": fam["family"], "style": fam["style"],
                           "license": fam["license"], "variable": f["variable"], "cmap": cmap})
    _BGS = json.loads(BG_MANIFEST.read_text())["items"]
    _FILTER = UnseenFilter.load()


def luminance(c) -> float:
    def ch(v):
        v = v / 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * ch(c[0]) + 0.7152 * ch(c[1]) + 0.0722 * ch(c[2])


def contrast(a, b) -> float:
    la, lb = luminance(a), luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def color_name(c) -> tuple[str, str]:
    best = min(COLOR_NAMES, key=lambda x: sum((x[0][i] - c[i]) ** 2 for i in range(3)))
    return best[1], best[2]


def rand_color(rng: random.Random) -> tuple[int, int, int]:
    if rng.random() < 0.6:
        base = rng.choice(COLOR_NAMES)[0]
        return tuple(int(min(255, max(0, v + rng.randint(-25, 25)))) for v in base)
    return (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))


def pick_text_color(rng, bg_mean, min_ratio=4.5):
    for _ in range(25):
        c = rand_color(rng)
        if contrast(c, bg_mean) >= min_ratio:
            return c
    return (15, 15, 15) if contrast((15, 15, 15), bg_mean) > contrast((245, 245, 245), bg_mean) else (245, 245, 245)


# ---------------- backgrounds
def smooth_noise(rng: np.random.Generator, h: int, w: int, scale: int) -> np.ndarray:
    small = rng.random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def procedural_bg(kind: str, w: int, h: int, rng: random.Random, nrng: np.random.Generator) -> np.ndarray:
    c1 = np.array(rand_color(rng), np.float32)
    c2 = np.array(rand_color(rng), np.float32)
    if kind == "gradient":
        ang = rng.uniform(0, math.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        if rng.random() < 0.3:
            t = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (0.75 * max(w, h))
        else:
            t = (xx * math.cos(ang) + yy * math.sin(ang)) / (w * abs(math.cos(ang)) + h * abs(math.sin(ang)) + 1e-6)
        t = np.clip(t, 0, 1)[..., None]
        img = c1 * (1 - t) + c2 * t
    elif kind == "paper":
        base = np.array(rng.choice([(245, 240, 228), (238, 232, 215), (250, 250, 247), (232, 222, 200), (240, 236, 225)]), np.float32)
        n = smooth_noise(nrng, h, w, 64) * 18 + smooth_noise(nrng, h, w, 8) * 10 + nrng.normal(0, 3, (h, w)).astype(np.float32)
        img = base[None, None, :] - n[..., None]
        fib = (nrng.random((h, w)) > 0.995).astype(np.float32)
        fib = cv2.GaussianBlur(cv2.resize(fib, (w, h)), (0, 0), 1.0) * 60
        img -= fib[..., None]
    elif kind == "wood":
        base = np.array(rng.choice([(160, 110, 65), (190, 140, 90), (120, 80, 45), (205, 165, 115), (95, 60, 35)]), np.float32)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        warp = smooth_noise(nrng, h, w, 128) * 40
        rings = np.sin((yy + warp) / rng.uniform(5, 14) + smooth_noise(nrng, h, w, 32) * 3)
        n = rings * 18 + nrng.normal(0, 4, (h, w)).astype(np.float32)
        img = base[None, None, :] + n[..., None] * np.array([1.0, 0.8, 0.6], np.float32)
    elif kind == "metal":
        base = np.array(rng.choice([(170, 172, 175), (200, 200, 205), (120, 122, 128), (190, 170, 120), (150, 150, 150)]), np.float32)
        n = nrng.normal(0, 1, (h, w)).astype(np.float32)
        n = cv2.blur(n, (61, 1)) * 30 + smooth_noise(nrng, h, w, 200) * 25
        img = base[None, None, :] + n[..., None]
    else:  # solid
        img = np.broadcast_to(c1, (h, w, 3)).astype(np.float32) + nrng.normal(0, 2.0, (h, w, 1)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def photo_bg(w: int, h: int, rng: random.Random) -> tuple[np.ndarray, dict]:
    rec = rng.choice(_BGS)
    im = cv2.imread(str(BG_DIR / rec["file"]), cv2.IMREAD_COLOR)[:, :, ::-1]
    H, W = im.shape[:2]
    scale = rng.uniform(0.45, 1.0)
    cw = int(min(W, H * w / h) * scale)
    ch = int(cw * h / w)
    cw, ch = max(64, min(cw, W)), max(64, min(ch, H))
    x0, y0 = rng.randint(0, W - cw), rng.randint(0, H - ch)
    crop = im[y0:y0 + ch, x0:x0 + cw]
    if rng.random() < 0.5:
        crop = crop[:, ::-1]
    out = cv2.resize(np.ascontiguousarray(crop), (w, h), interpolation=cv2.INTER_AREA if cw > w else cv2.INTER_CUBIC)
    if rng.random() < 0.4:  # light colour jitter
        out = np.clip(out.astype(np.float32) * rng.uniform(0.8, 1.15) + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
    return out, rec


def bg_desc(meta: dict) -> tuple[str, str]:
    if meta["type"] == "procedural":
        return {"gradient": ("그라데이션 배경", "a smooth gradient background"), "paper": ("종이 질감 배경", "a paper texture background"),
                "wood": ("나무결 배경", "a wood grain background"), "metal": ("금속 질감 배경", "a brushed metal background"),
                "solid": ("단색 배경", "a plain solid-color background")}[meta["kind"]]
    cats = meta.get("categories", [])
    if meta["kind"] == "hdri_crop":
        if "indoor" in cats or "studio" in cats:
            return "실내 사진 배경", "an indoor photo background"
        if "urban" in cats:
            return "도시 풍경 사진 배경", "an urban street photo background"
        if "skies" in cats:
            return "하늘 풍경 사진 배경", "a sky landscape photo background"
        return "야외 풍경 사진 배경", "an outdoor landscape photo background"
    for key, ko, en in BG_CAT_WORDS:
        if any(key in c for c in cats):
            return f"{ko} 질감 사진 배경", f"a {en} texture photo background"
    return "질감 있는 사진 배경", "a textured photo background"


# ---------------- text layer
_FONT_BYTES: dict[str, bytes] = {}


def load_font(f: dict, size: int, weight: int | None):
    data = _FONT_BYTES.get(f["path"])
    if data is None:
        data = _FONT_BYTES[f["path"]] = Path(f["path"]).read_bytes()
    font = ImageFont.truetype(io.BytesIO(data), size)
    if weight is not None:
        font.set_variation_by_axes([weight])
    return font


def pick_weight(f: dict, rng: random.Random) -> int | None:
    if not f["variable"]:
        return None
    ax = load_font(f, 20, None).get_variation_axes()[0]
    return rng.randint(int(ax["minimum"]), int(ax["maximum"]))


def choose_font(text: str, style_hint: str | None, rng: random.Random) -> dict:
    need = {ord(c) for c in text if not c.isspace()}
    ok = [f for f in _FONTS if need <= f["cmap"]]
    if not ok:
        raise RuntimeError(f"no font covers {text!r}")
    if style_hint and rng.random() < 0.5:
        hinted = [f for f in ok if f["style"] == style_hint]
        if hinted:
            ok = hinted
    # pick family uniformly, then a file of that family
    fams = sorted({f["family"] for f in ok})
    fam = rng.choice(fams)
    return rng.choice([f for f in ok if f["family"] == fam])


def wrap_balanced(line: str, parts: int) -> list[str]:
    words = line.split(" ")
    if parts <= 1 or len(words) < 2:
        return [line]
    parts = min(parts, len(words))
    total = len(line)
    out, cur = [], []
    for w in words:
        cur.append(w)
        if len(out) < parts - 1 and len(" ".join(cur)) >= total / parts:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def fit_text(text: str, fdesc: dict, slot_w: int, slot_h: int, rng: random.Random, min_size: int = 18):
    """Return (font, size, lines, weight). Picks the largest size that fits; wraps long lines if
    the unwrapped text would fall below min_size."""
    weight = pick_weight(fdesc, rng)
    ref = load_font(fdesc, 100, weight)
    base_lines = text.split("\n")
    best = None
    for wrap in (1, 2, 3):
        lines = [x for ln in base_lines for x in (wrap_balanced(ln, wrap) if len(ln) > 8 else [ln])]
        wmax100 = max(ref.getlength(ln) for ln in lines) or 1.0
        size = int(min(slot_h / (len(lines) * 1.22), slot_w * 100 / wmax100 * 0.97, 260))
        if best is None or size > best[0]:
            best = (size, lines)
        if size >= min_size:
            break
    size, lines = max(best[0], 12), best[1]
    font = load_font(fdesc, size, weight)
    while size > 12 and max(font.getlength(ln) for ln in lines) > slot_w:
        size -= 2
        font = load_font(fdesc, size, weight)
    return font, size, lines, weight


def render_block(lines, font, fill, align, stroke_w, stroke_fill, shadow, rng):
    """Draw lines on an RGBA layer. Returns (layer, line_boxes[x0,y0,x1,y1])."""
    spacing = int(font.size * rng.uniform(0.12, 0.3))
    asc, desc = font.getmetrics()
    lh = asc + desc
    widths = [font.getlength(ln) for ln in lines]
    pad = stroke_w + (shadow["blur"] * 3 + max(abs(shadow["dx"]), abs(shadow["dy"])) if shadow else 0) + 8
    W = int(max(widths) + 2 * pad)
    H = int(len(lines) * lh + (len(lines) - 1) * spacing + 2 * pad)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(layer)
    dm = ImageDraw.Draw(mask)
    boxes = []
    for i, ln in enumerate(lines):
        if align == "left":
            x = pad
        elif align == "right":
            x = W - pad - widths[i]
        else:
            x = (W - widths[i]) / 2
        y = pad + i * (lh + spacing)
        kw = dict(font=font, stroke_width=stroke_w)
        if stroke_w:
            d.text((x, y), ln, fill=fill + (255,), stroke_fill=stroke_fill + (255,), **kw)
        else:
            d.text((x, y), ln, fill=fill + (255,), font=font)
        dm.text((x, y), ln, fill=255, **kw)
        bb = dm.textbbox((x, y), ln, **kw)
        boxes.append(list(bb))
    if shadow:
        sh = Image.new("RGBA", (W, H), shadow["color"] + (0,))
        a = mask.filter(ImageFilter.GaussianBlur(shadow["blur"])) if shadow["blur"] else mask
        a = a.point(lambda v: int(v * shadow["alpha"]))
        sh.putalpha(a)
        base = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        base.paste(sh, (shadow["dx"], shadow["dy"]))
        base.alpha_composite(layer)
        layer = base
    return layer, boxes


def quad_from_box(b, curve_pts: int):
    x0, y0, x1, y1 = b
    if curve_pts <= 2:
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    xs = np.linspace(x0, x1, curve_pts)
    return [(x, y0) for x in xs] + [(x, y1) for x in xs[::-1]]


def warp_layer(layer: Image.Image, polys: list[list[tuple]], rng: random.Random, strength: float):
    """Optional arc curve + rotation/perspective. Returns (rgba ndarray, transformed polys)."""
    arr = np.array(layer)
    h, w = arr.shape[:2]
    if rng.random() < 0.18 * strength + 0.05:
        amp = rng.uniform(0.03, 0.12) * w * rng.choice([-1, 1]) * 0.35
        A = abs(int(amp)) + 2
        canvas = np.zeros((h + 2 * A, w, 4), np.uint8)
        canvas[A:A + h] = arr
        xs = np.arange(w, dtype=np.float32)
        off = (amp * np.sin(np.pi * xs / max(1, w - 1))).astype(np.float32)
        map_x = np.tile(xs, (h + 2 * A, 1))
        map_y = (np.arange(h + 2 * A, dtype=np.float32)[:, None] - off[None, :])
        arr = cv2.remap(canvas, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        polys = [[(x, y + A + amp * math.sin(math.pi * x / max(1, w - 1))) for x, y in p] for p in polys]
        h = arr.shape[0]
    ang = math.radians(rng.uniform(-7, 7) * strength) if rng.random() < 0.6 else 0.0
    j = 0.08 * strength if rng.random() < 0.5 else 0.0
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    cx, cy = w / 2, h / 2
    dst = []
    for x, y in src:
        x2 = x + rng.uniform(-j, j) * w
        y2 = y + rng.uniform(-j, j) * h
        dx, dy = x2 - cx, y2 - cy
        dst.append([cx + dx * math.cos(ang) - dy * math.sin(ang), cy + dx * math.sin(ang) + dy * math.cos(ang)])
    dst = np.float32(dst)
    mn = dst.min(0)
    dst -= mn
    ow, oh = int(math.ceil(dst[:, 0].max())) + 1, int(math.ceil(dst[:, 1].max())) + 1
    M = cv2.getPerspectiveTransform(src, dst)
    arr = cv2.warpPerspective(arr, M, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    out = []
    for p in polys:
        pts = cv2.perspectiveTransform(np.float32(p).reshape(-1, 1, 2), M).reshape(-1, 2)
        out.append([tuple(map(float, q)) for q in pts])
    return arr, out


def alpha_paste(dst: np.ndarray, src_rgba: np.ndarray, x: int, y: int):
    H, W = dst.shape[:2]
    h, w = src_rgba.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    s = src_rgba[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = s[..., 3:4] / 255.0
    d = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (s[..., :3] * a + d * (1 - a)).astype(np.uint8)


def draw_panel(img: np.ndarray, rect, rng: random.Random, nrng, scene: str) -> dict:
    """Paint a sign/label/cover/menu/notice panel onto img in place."""
    x, y, w, h = rect
    kind = rng.choices(["solid", "gradient", "paper", "wood", "metal", "photo"], weights=[5, 2, 2, 1, 1, 1])[0]
    if scene == "notice":
        kind = rng.choice(["paper", "solid", "paper"])
    if scene == "menu" and rng.random() < 0.5:
        kind = "chalk"
    if kind == "chalk":
        tex = procedural_bg("solid", w, h, rng, nrng)
        tex[:] = rng.choice([(30, 50, 40), (25, 25, 28), (40, 40, 35)])
        tex = np.clip(tex.astype(np.float32) + nrng.normal(0, 6, (h, w, 1)), 0, 255).astype(np.uint8)
        meta = {"kind": "chalkboard"}
    elif kind == "photo":
        tex, rec = photo_bg(w, h, rng)
        meta = {"kind": "photo", "file": rec["file"], "license": rec["license"], "source_url": rec["source_url"]}
    else:
        tex = procedural_bg(kind, w, h, rng, nrng)
        meta = {"kind": kind}
    shape = rng.choices(["rect", "round", "ellipse"], weights=[5, 4, 1 if scene == "product_label" else 0])[0]
    mask = np.zeros((h, w), np.uint8)
    if shape == "rect":
        mask[:] = 255
    elif shape == "round":
        r = int(min(w, h) * rng.uniform(0.04, 0.18))
        cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
        cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
        for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
            cv2.circle(mask, (cx, cy), r, 255, -1)
    else:
        cv2.ellipse(mask, (w // 2, h // 2), (w // 2 - 1, h // 2 - 1), 0, 0, 360, 255, -1)
    if rng.random() < 0.5:  # border
        bw = max(2, int(min(w, h) * rng.uniform(0.008, 0.03)))
        inner = cv2.erode(mask, np.ones((bw * 2 + 1, bw * 2 + 1), np.uint8))
        bc = np.array(rand_color(rng), np.uint8)
        tex[(mask > 0) & (inner == 0)] = bc
    # drop shadow under the panel
    H, W = img.shape[:2]
    if rng.random() < 0.6:
        sm = np.zeros((H, W), np.float32)
        ox, oy = rng.randint(3, 14), rng.randint(3, 14)
        xs0, ys0 = max(0, x + ox), max(0, y + oy)
        xs1, ys1 = min(W, x + ox + w), min(H, y + oy + h)
        if xs1 > xs0 and ys1 > ys0:
            sm[ys0:ys1, xs0:xs1] = mask[ys0 - y - oy:ys1 - y - oy, xs0 - x - ox:xs1 - x - ox] / 255.0
            sm = cv2.GaussianBlur(sm, (0, 0), 8) * 0.45
            img[:] = (img.astype(np.float32) * (1 - sm[..., None])).astype(np.uint8)
    rgba = np.dstack([tex, mask])
    alpha_paste(img, rgba, x, y)
    meta["shape"] = shape
    return meta


def layout(scene: str, W: int, H: int, nblocks: int, rng: random.Random):
    """Return (panel rect or None, container rect) for the text stack."""
    m = int(min(W, H) * 0.04)
    if scene == "sign":
        pw, ph = int(W * rng.uniform(0.75, 0.96)), int(H * rng.uniform(0.25, 0.30 + 0.12 * nblocks))
        return (rng.randint(m, W - pw - m) if W - pw - 2 * m > 0 else m, rng.randint(m, max(m, H - ph - m)), pw, ph)
    if scene == "book_cover":
        pw, ph = int(W * rng.uniform(0.62, 0.88)), int(H * rng.uniform(0.78, 0.94))
        pw = min(pw, int(ph * 0.75))
        return ((W - pw) // 2 + rng.randint(-m, m), (H - ph) // 2 + rng.randint(-m, m) // 2, pw, ph)
    if scene == "product_label":
        pw, ph = int(W * rng.uniform(0.5, 0.85)), int(H * rng.uniform(0.35, 0.65))
        return ((W - pw) // 2 + rng.randint(-2 * m, 2 * m), (H - ph) // 2 + rng.randint(-2 * m, 2 * m), pw, ph)
    if scene in ("menu", "notice"):
        pw, ph = int(W * rng.uniform(0.6, 0.9)), int(H * rng.uniform(0.7, 0.93))
        return ((W - pw) // 2 + rng.randint(-m, m), (H - ph) // 2 + rng.randint(-m, m) // 2, pw, ph)
    # poster: no panel, or a translucent band
    return None


def pos_words(poly, W, H):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    cx, cy = (min(xs) + max(xs)) / 2 / W, (min(ys) + max(ys)) / 2 / H
    col = 0 if cx < 0.36 else (2 if cx > 0.64 else 1)
    row = 0 if cy < 0.36 else (2 if cy > 0.64 else 1)
    return POS_WORDS[row][col]


def size_words(size: int, H: int) -> tuple[str, str]:
    r = size / H
    if r > 0.09:
        return "아주 큰", "very large"
    if r > 0.055:
        return "큰", "large"
    if r > 0.032:
        return "중간 크기", "medium"
    return "작은", "small"


def captions(scene: str, bgd: tuple[str, str], blocks: list[dict], W: int, H: int) -> dict:
    order = sorted(blocks, key=lambda b: (-b["size"], b["index"]))
    sko, sen = SCENE_WORDS[scene]
    ko_parts, en_parts, quoted = [], [], []
    for b in order:
        q = ", ".join(f'"{ln}"' for ln in b["lines_text"])
        quoted.append(q)
        pko, pen = pos_words(b["polygon"], W, H)
        zko, zen = size_words(b["size"], H)
        cko, cen = color_name(b["color"])
        tko, ten = STYLE_WORDS[b["font_style"]]
        extra_ko = " 테두리가 있는" if b["stroke_width"] else ""
        extra_en = " outlined" if b["stroke_width"] else ""
        ko_parts.append(f"{pko}에{extra_ko} {zko} {cko} {tko} 글씨: {q}.")
        en_parts.append(f"{zen.capitalize()}{extra_en} {cen} {ten} Korean text at the {pen}: {q}.")
    ko_long = f"{bgd[0]} 위의 {sko} 이미지. " + " ".join(ko_parts)
    ko_short = f"{sko}, 문구: " + ", ".join(quoted)
    en = f"A {sen} on {bgd[1]}. " + " ".join(en_parts)
    return {"ko_long": ko_long, "ko_short": ko_short, "en": en}


def stroke_and_shadow(rng, fill, bg_mean, textured: bool):
    stroke_w, stroke_fill, shadow = 0, None, None
    if rng.random() < (0.55 if textured else 0.2):
        stroke_fill = (15, 15, 15) if luminance(fill) > 0.4 else (245, 245, 245)
        if rng.random() < 0.3:
            stroke_fill = pick_text_color(rng, fill, 3.0)
        stroke_w = -1  # filled in relative to size
    if rng.random() < (0.45 if textured else 0.2):
        shadow = {"dx": rng.randint(-6, 6) or 2, "dy": rng.randint(1, 8), "blur": rng.choice([0, 1, 2, 4]),
                  "alpha": rng.uniform(0.4, 0.85), "color": (0, 0, 0) if luminance(fill) > 0.2 else (255, 255, 255)}
    return stroke_w, stroke_fill, shadow


def render(item: dict, seed: int) -> dict:
    rng = random.Random(f"{seed}:render:{item['index']}")
    nrng = np.random.default_rng(seed * 1000003 + item["index"])
    W, H = item["size"]
    scene = item["scene"]
    # ---- background
    if rng.random() < 0.65:
        img, rec = photo_bg(W, H, rng)
        bgmeta = {"type": "photo", "kind": rec["kind"], "file": rec["file"], "asset": rec["asset"],
                  "categories": rec["categories"], "source": "Poly Haven", "source_url": rec["source_url"],
                  "asset_page": rec["asset_page"], "license": rec["license"]}
    else:
        kind = rng.choice(["gradient", "paper", "wood", "metal", "solid"])
        img = procedural_bg(kind, W, H, rng, nrng)
        bgmeta = {"type": "procedural", "kind": kind, "source": "procedural (synth/generate.py)", "license": "CC0-1.0"}
    img = np.ascontiguousarray(img)
    blocks_in = item["blocks"]
    rect = layout(scene, W, H, len(blocks_in), rng)
    panel_meta = None
    if rect is not None:
        x, y, pw, ph = rect
        x, y = max(0, min(x, W - pw)), max(0, min(y, H - ph))
        rect = (x, y, pw, ph)
        panel_meta = draw_panel(img, rect, rng, nrng, scene)
        pad = int(min(pw, ph) * rng.uniform(0.07, 0.12))
        if panel_meta.get("shape") == "ellipse":
            pad = int(min(pw, ph) * 0.2)
        cont = (x + pad, y + pad, pw - 2 * pad, ph - 2 * pad)
    else:
        m = int(min(W, H) * rng.uniform(0.06, 0.1))
        cont = (m, m, W - 2 * m, H - 2 * m)
    # ---- vertical stack of slots
    weights = []
    for i, b in enumerate(blocks_in):
        nl = b["text"].count("\n") + 1
        if i == 0:
            wgt = 2.6 if nl == 1 else 1.0 * nl
        else:
            wgt = 0.9 * nl if nl > 1 else rng.uniform(0.8, 1.3)
        weights.append(wgt)
    gap = int(cont[3] * 0.04)
    total_h = cont[3] - gap * (len(blocks_in) - 1)
    order = list(range(len(blocks_in)))
    if scene == "book_cover" and len(order) > 2 and rng.random() < 0.5:
        order = [0] + order[2:] + [1]  # author at the bottom
    hs = {i: total_h * weights[i] / sum(weights) for i in order}
    ycur = cont[1]
    style_hint = {"menu": "handwriting", "notice": "gothic", "book_cover": "myeongjo", "sign": "display",
                  "poster": "display", "product_label": "rounded"}.get(scene)
    strength = rng.uniform(0.3, 1.0)
    out_blocks = []
    for i in order:
        b = blocks_in[i]
        sh = int(hs[i])
        slot = (cont[0], int(ycur), cont[2], sh)
        ycur += sh + gap
        region = img[max(0, slot[1]):slot[1] + max(1, slot[3]), max(0, slot[0]):slot[0] + max(1, slot[2])]
        bg_mean = tuple(float(v) for v in region.reshape(-1, 3).mean(0)) if region.size else (128, 128, 128)
        textured = bool(region.size and region.reshape(-1, 3).std(0).mean() > 28)
        fdesc = choose_font(b["text"], style_hint if i == 0 else None, rng)
        nl = b["text"].count("\n") + 1
        align = "left" if (nl > 2 and scene in ("menu", "notice") and rng.random() < 0.7) else rng.choices(["center", "left", "right"], [6, 2, 1])[0]
        sw_scale = rng.uniform(0.75, 1.0) if i == 0 else rng.uniform(0.45, 1.0)
        slot_h = max(12, int(slot[3] * 0.92))
        if out_blocks:  # secondary text stays visibly smaller than the headline block
            slot_h = min(slot_h, int(out_blocks[0]["size"] * rng.uniform(0.45, 0.75) * 1.22 * nl))
        font, size, lines, weight = fit_text(b["text"], fdesc, int(slot[2] * sw_scale), slot_h, rng)
        fill = pick_text_color(rng, bg_mean, 4.5 if not textured else 3.0)
        stroke_w, stroke_fill, shadow = stroke_and_shadow(rng, fill, bg_mean, textured)
        if stroke_w == -1:
            stroke_w = max(1, int(size * rng.uniform(0.03, 0.08)))
        layer, boxes = render_block(lines, font, fill, align, stroke_w, stroke_fill, shadow, rng)
        cpts = 6  # points per long edge, so an arc-curved block keeps a tight polygon
        polys = [quad_from_box([min(bb[0] for bb in boxes), min(bb[1] for bb in boxes),
                                max(bb[2] for bb in boxes), max(bb[3] for bb in boxes)], cpts)]
        polys += [quad_from_box(bb, cpts) for bb in boxes]
        arr, polys = warp_layer(layer, polys, rng, strength)
        lh, lw = arr.shape[:2]
        if align == "left":
            px = slot[0]
        elif align == "right":
            px = slot[0] + slot[2] - lw
        else:
            px = slot[0] + (slot[2] - lw) // 2
        py = slot[1] + (slot[3] - lh) // 2
        px = int(max(0, min(px, W - lw))) if lw <= W else (W - lw) // 2
        py = int(max(0, min(py, H - lh))) if lh <= H else (H - lh) // 2
        alpha_paste(img, arr, px, py)
        polys = [[(round(x + px, 1), round(y + py, 1)) for x, y in p] for p in polys]
        out_blocks.append({
            "index": i, "role": b["role"], "coverage": bool(b.get("coverage")), "text": b["text"],
            "lines_text": lines, "polygon": polys[0], "line_polygons": polys[1:],
            "font_family": fdesc["family"], "font_file": fdesc["file"], "font_style": fdesc["style"],
            "font_license": fdesc["license"], "font_weight_axis": weight, "size": size,
            "color": list(fill), "stroke_width": stroke_w, "stroke_color": list(stroke_fill) if stroke_fill else None,
            "shadow": bool(shadow), "align": align,
        })
    # ---- global effects
    effects = {}
    if rng.random() < 0.25:
        j = rng.uniform(0.02, 0.07)
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst = np.float32([[rng.uniform(0, j) * W, rng.uniform(0, j) * H], [W - rng.uniform(0, j) * W, rng.uniform(0, j) * H],
                          [W - rng.uniform(0, j) * W, H - rng.uniform(0, j) * H], [rng.uniform(0, j) * W, H - rng.uniform(0, j) * H]])
        # scale up about the centre so the warped image still fills the frame
        c = dst.mean(0)
        dst = (dst - c) * (1 + 2.2 * j) + c
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        for b in out_blocks:
            b["polygon"] = [tuple(round(float(v), 1) for v in q) for q in cv2.perspectiveTransform(np.float32(b["polygon"]).reshape(-1, 1, 2), M).reshape(-1, 2)]
            b["line_polygons"] = [[tuple(round(float(v), 1) for v in q) for q in cv2.perspectiveTransform(np.float32(p).reshape(-1, 1, 2), M).reshape(-1, 2)] for p in b["line_polygons"]]
        effects["perspective"] = round(j, 3)
    if rng.random() < 0.3:
        s = rng.uniform(0.3, 1.2)
        img = cv2.GaussianBlur(img, (0, 0), s)
        effects["blur_sigma"] = round(s, 2)
    if rng.random() < 0.4:
        s = rng.uniform(2, 8)
        img = np.clip(img.astype(np.float32) + nrng.normal(0, s, img.shape), 0, 255).astype(np.uint8)
        effects["noise_sigma"] = round(s, 1)
    if rng.random() < 0.25:
        a, bb = rng.uniform(0.85, 1.15), rng.uniform(-18, 18)
        img = np.clip(img.astype(np.float32) * a + bb, 0, 255).astype(np.uint8)
        effects["brightness"] = [round(a, 2), round(bb, 1)]
    pil = Image.fromarray(img)
    if rng.random() < 0.5:
        q = rng.randint(35, 85)
        buf = io.BytesIO()
        pil.save(buf, "JPEG", quality=q)
        pil = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        effects["jpeg_quality"] = q
    for b in out_blocks:
        b["polygon"] = [[min(max(x, 0), W), min(max(y, 0), H)] for x, y in b["polygon"]]
        b["line_polygons"] = [[[min(max(x, 0), W), min(max(y, 0), H)] for x, y in p] for p in b["line_polygons"]]
    caps = captions(scene, bg_desc(bgmeta), out_blocks, W, H)
    leaks = [r for r in ([_FILTER.reason(b["text"]) for b in out_blocks] + [_FILTER.reason(c) for c in caps.values()]) if r]
    meta = {"key": item["key"], "width": W, "height": H, "scene": scene, "seed": seed,
            "background": bgmeta, "panel": panel_meta, "effects": effects,
            "text_blocks": sorted(out_blocks, key=lambda b: b["index"]), "captions": caps,
            "generator": "redrob-image synth/generate.py v1"}
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=95)
    flat = OUT / "flat"
    (flat / f"{item['key']}.jpg").write_bytes(buf.getvalue())
    (flat / f"{item['key']}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return {"key": item["key"], "leaks": leaks}


def _render_star(args):
    try:
        return render(*args)
    except Exception as e:
        return {"key": args[0]["key"], "error": repr(e), "leaks": []}


def write_shards(n: int, per: int):
    sd = OUT / "shards"
    sd.mkdir(parents=True, exist_ok=True)
    flat = OUT / "flat"
    shards = []
    for s in range(0, n, per):
        name = sd / f"synth-{s // per:05d}.tar"
        with tarfile.open(name, "w") as tf:
            for i in range(s, min(n, s + per)):
                key = f"{i:06d}"
                for ext in ("jpg", "json"):
                    tf.add(flat / f"{key}.{ext}", arcname=f"{key}.{ext}")
        shards.append(name.name)
    return shards


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n", type=int, default=20, help="min occurrences per KS X 1001 syllable (stage 1)")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
    ap.add_argument("--n2", type=int, default=2, help="min occurrences per other syllable (stage 2)")
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=OUT, help="output directory (default synth/out)")
    a = ap.parse_args()
    globals()["OUT"] = a.out.expanduser()
    (OUT / "flat").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ts = TextSource()
    plan, info = make_plan(a.count, a.seed, a.n, a.stage, a.n2, ts)
    t_plan = time.time() - t0
    (OUT / "plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    print(f"plan: {len(plan)} images in {t_plan:.1f}s; {info}", flush=True)
    t1 = time.time()
    errors, leaks = [], []
    with Pool(a.workers, initializer=init_worker) as pool:
        for k, r in enumerate(pool.imap_unordered(_render_star, [(it, a.seed) for it in plan], chunksize=8)):
            if r.get("error"):
                errors.append(r)
            leaks += r["leaks"]
            if (k + 1) % 1000 == 0:
                print(f"rendered {k + 1} ({(k + 1) / (time.time() - t1):.1f} img/s)", flush=True)
    t_render = time.time() - t1
    shards = write_shards(a.count, a.shard_size)
    run = {"count": a.count, "seed": a.seed, "workers": a.workers, "n": a.n, "stage": a.stage, "n2": a.n2,
           "plan_seconds": round(t_plan, 1), "render_seconds": round(t_render, 1),
           "images_per_sec": round(a.count / t_render, 2), "errors": errors[:20], "error_count": len(errors),
           "render_time_leaks": len(leaks), "plan_info": info, "shards": shards}
    (OUT / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in run.items() if k != "errors"}, ensure_ascii=False))
    if errors:
        print(errors[:3], file=sys.stderr)
    return 1 if errors or leaks else 0


if __name__ == "__main__":
    sys.exit(main())
