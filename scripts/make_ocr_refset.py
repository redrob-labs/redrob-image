#!/usr/bin/env python3
"""Build the OCR reference set: rendered Korean text crops with known ground truth.

  python scripts/make_ocr_refset.py --fonts data/fonts --out data/ocr_refset --n 600

Texts: 1/3 seen words, 1/3 unseen words, 1/3 strings made of unseen (non-KS X 1001) syllables.
Each crop gets a random OFL font (only fonts whose cmap covers every syllable of the text),
random colours, and a random degradation: clean, blur, noise, rotation, perspective, JPEG,
textured background, or two-line wrap. Writes <out>/img/*.png and <out>/manifest.jsonl.
Fonts are downloaded by --download (OFL, google/fonts).
"""
from __future__ import annotations

import argparse
import io
import json
import random
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
GF = "https://github.com/google/fonts/raw/main/ofl/"
FONTS = {
    "notosanskr": ["NotoSansKR%5Bwght%5D.ttf"],
    "notoserifkr": ["NotoSerifKR%5Bwght%5D.ttf"],
    "nanumgothic": ["NanumGothic-Regular.ttf", "NanumGothic-ExtraBold.ttf"],
    "nanummyeongjo": ["NanumMyeongjo-Regular.ttf"],
    "nanumpenscript": ["NanumPenScript-Regular.ttf"],
    "blackhansans": ["BlackHanSans-Regular.ttf"],
    "dohyeon": ["DoHyeon-Regular.ttf"],
    "jua": ["Jua-Regular.ttf"],
    "gowundodum": ["GowunDodum-Regular.ttf"],
    "gaegu": ["Gaegu-Regular.ttf", "Gaegu-Bold.ttf"],
}
AUGS = ["clean", "blur", "noise", "rotate", "perspective", "jpeg", "texture", "wrap"]


def download(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for fam, files in FONTS.items():
        for f in files + ["OFL.txt"]:
            out = dst / fam / f.replace("%5B", "[").replace("%5D", "]")
            if out.exists():
                continue
            out.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(GF + fam + "/" + f, out)
            print("got", out)


def font_files(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*/*.ttf"))


def cmap(p: Path) -> set[int]:
    from fontTools.ttLib import TTFont

    return set(TTFont(p, lazy=True).getBestCmap())


def load_font(p: Path, size: int, rng: random.Random) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(p), size)
    if "[wght]" in p.name:
        try:
            f.set_variation_by_axes([rng.choice([400, 500, 700, 900])])
        except Exception:
            pass
    return f


def texts(rng: random.Random, n: int) -> list[tuple[str, str]]:
    def lines(f):
        return [l.strip() for l in (ROOT / "eval" / f).read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.startswith("#")]

    seen = [json.loads(l)["target_text"] for l in (ROOT / "eval/prompts.jsonl").open(encoding="utf-8")
            if '"split": "seen"' in l]
    unseen = lines("unseen_words.txt")
    usyl = lines("unseen_syllables.txt")
    k = n // 3
    out = [("seen", rng.choice(seen)) for _ in range(k)]
    out += [("unseen", rng.choice(unseen)) for _ in range(k)]
    for _ in range(n - 2 * k):
        L = rng.choice([1, 2, 2, 3, 3, 4, 5])
        out.append(("unseen_syllable", "".join(rng.choice(usyl) for _ in range(L))))
    return out


def render(text: str, font_path: Path, aug: str, rng: random.Random) -> Image.Image:
    size = rng.choice([28, 36, 48, 64, 80])
    font = load_font(font_path, size, rng)
    lines = [text]
    if aug == "wrap" and " " in text:
        ws = text.split(" ")
        cut = max(1, len(ws) // 2)
        lines = [" ".join(ws[:cut]), " ".join(ws[cut:])]
    bg = tuple(rng.randrange(256) for _ in range(3))
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    fg = tuple(rng.randrange(0, 70) for _ in range(3)) if lum > 128 else tuple(rng.randrange(190, 256) for _ in range(3))
    pad = size // 2
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    boxes = [dummy.textbbox((0, 0), l, font=font) for l in lines]
    w = max(b[2] - b[0] for b in boxes) + 2 * pad
    lh = max(b[3] - b[1] for b in boxes)
    h = int(lh * 1.35 * len(lines)) + 2 * pad
    img = Image.new("RGB", (w, h), bg)
    if aug == "texture":
        a = np.asarray(img, dtype=float)
        noise = np.asarray(Image.fromarray((np.random.default_rng(rng.randrange(1 << 30)).random((h // 8 + 1, w // 8 + 1)) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), dtype=float)
        a = np.clip(a + (noise[..., None] - 128) * 0.35, 0, 255)
        img = Image.fromarray(a.astype(np.uint8))
    d = ImageDraw.Draw(img)
    y = pad
    for l, b in zip(lines, boxes):
        d.text((pad - b[0], y - b[1]), l, font=font, fill=fg)
        y += int(lh * 1.35)
    if aug == "blur":
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(1.0, 2.2)))
    elif aug == "noise":
        a = np.asarray(img, dtype=float) + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 18, (h, w, 3))
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    elif aug == "rotate":
        img = img.rotate(rng.uniform(-8, 8), resample=Image.BICUBIC, expand=True, fillcolor=bg)
    elif aug == "perspective":
        dx, dy = w * 0.08, h * 0.15
        c = [rng.uniform(0, dx), rng.uniform(0, dy), w - rng.uniform(0, dx), rng.uniform(0, dy),
             w - rng.uniform(0, dx), h - rng.uniform(0, dy), rng.uniform(0, dx), h - rng.uniform(0, dy)]
        img = img.transform((w, h), Image.QUAD, c, resample=Image.BICUBIC, fillcolor=bg)
    elif aug == "jpeg":
        buf = io.BytesIO()
        img.resize((max(1, w // 2), max(1, h // 2))).save(buf, "JPEG", quality=rng.randint(15, 35))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB").resize((w, h))
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", default="data/fonts")
    ap.add_argument("--out", default="data/ocr_refset")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    fdir, out = Path(a.fonts), Path(a.out)
    if a.download:
        download(fdir)
    fonts = font_files(fdir)
    cov = {p: cmap(p) for p in fonts}
    rng = random.Random(a.seed)
    (out / "img").mkdir(parents=True, exist_ok=True)
    stats = {p.parent.name + "/" + p.name: dict(hangul=sum(0xAC00 <= c <= 0xD7A3 for c in cov[p])) for p in fonts}
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as mf:
        for i, (kind, t) in enumerate(texts(rng, a.n)):
            need = {ord(c) for c in t if not c.isspace()}
            ok = [p for p in fonts if need <= cov[p]]
            p = rng.choice(ok)
            aug = rng.choice(AUGS)
            img = render(t, p, aug, rng)
            name = f"{i:04d}.png"
            img.save(out / "img" / name)
            mf.write(json.dumps(dict(id=f"ref-{i:04d}", file=f"img/{name}", kind=kind, text=t,
                                     font=p.parent.name + "/" + p.name, aug=aug,
                                     n_fonts_covering=len(ok)), ensure_ascii=False) + "\n")
    (out / "fonts.json").write_text(json.dumps(stats, indent=1))
    print("fonts", json.dumps(stats))


if __name__ == "__main__":
    main()
