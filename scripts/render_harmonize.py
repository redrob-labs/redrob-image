#!/usr/bin/env python3
"""Plan B (docs/render-harmonize-plan.md): the scene comes from the model, the text from a font.

  1. scene     Z-Image Base draws the prompt with the quoted-text clause replaced by a blank surface
  2. place     glyph.layout() default box for the scene
  3. render    target text in Noto Sans KR (Bold when the prompt says 굵은), colour from the prompt
  4. harmonize Z-Image inpaint over a band around the text box; the strokes are outside the mask and are
               pasted back, so they stay exactly as drawn
Every config is written as its own eval folder (<out>/<cfg>/images/<id>_s<seed>.png) so eval_baseline.py
scores it without regenerating (it skips existing images):
  plain              render only, no harmonize
  b<pad>-s<strength> harmonized, band <pad> px around the box

  HF_HOME=~/hf python scripts/render_harmonize.py --prompts eval/prompts.jsonl eval/prompts_unseen_syllables.jsonl \
      --subset 20 --out ~/runs/render-harmonize-trial
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import glyph as G  # noqa: E402
from eval_baseline import load_prompts, stratified  # noqa: E402

FONT_DIR = Path("~/.local/share/fonts").expanduser()
QUOTED_CLAUSE = re.compile(r'[^,]*["“][^"”]*["”][^,]*')
BLANK = "아무 글자도 없는 깨끗한 빈 면"
NEG_TEXT = "글자, 문자, 텍스트, 숫자, 로고, 워터마크"
COLORS = {  # longest first so 검은색 wins over 검은
    "흰색": (245, 245, 245), "흰": (245, 245, 245), "하얀": (245, 245, 245),
    "검은색": (20, 20, 20), "검은": (20, 20, 20), "검정": (20, 20, 20),
    "금색": (212, 175, 55), "은색": (192, 192, 192), "빨간색": (200, 30, 30), "빨간": (200, 30, 30),
    "노란색": (240, 200, 30), "노란": (240, 200, 30), "파란색": (30, 80, 200), "파란": (30, 80, 200),
    "남색": (25, 35, 90), "초록색": (30, 130, 60), "초록": (30, 130, 60), "분홍색": (230, 110, 160),
    "주황색": (240, 130, 30), "갈색": (120, 75, 40), "회색": (128, 128, 128),
}


def blank_prompt(p: str) -> str:
    return QUOTED_CLAUSE.sub(" " + BLANK, p, count=1).strip()


def text_color(p: str, bg):
    """Last colour word before 글씨 (the text colour); else black or white, whichever contrasts the box."""
    head = p.split("글씨")[0] if "글씨" in p else ""
    best, pos = None, -1
    for w, c in COLORS.items():
        i = head.rfind(w)
        if i > pos:
            best, pos = c, i
    if best:
        return best
    return (20, 20, 20) if sum(bg) / 3 > 128 else (245, 245, 245)


def text_alpha(text: str, w: int, h: int, bold: bool):
    """Text as an L tile (255 = ink), centred, filling glyph.FILL of the tighter axis."""
    from PIL import Image, ImageDraw

    path = str(FONT_DIR / ("NotoSansKR-Bold.ttf" if bold else "NotoSansKR-Regular.ttf"))
    ref = G._font(64, path)
    x0, y0, x1, y1 = ref.getbbox(text)
    s = G.FILL * min(w / max(x1 - x0, 1), h / max(y1 - y0, 1))
    f = G._font(max(4, int(64 * s)), path)
    tile = Image.new("L", (w, h), 0)
    ImageDraw.Draw(tile).text((w / 2, h / 2), text, fill=255, font=f, anchor="mm")
    return tile


def render(scene_img, r: dict, prompt: str):
    """Composite (RGB), ink mask (L) and box list for one prompt."""
    from PIL import Image, ImageStat

    W, H = scene_img.size
    blocks = G.layout(r["target_text"], r.get("scene", ""), W, H)
    out, ink = scene_img.copy(), Image.new("L", (W, H), 0)
    bold = "굵은" in prompt or "bold" in prompt
    for b in blocks:
        x0, y0, x1, y1 = (int(round(v)) for v in b["bbox_xyxy"])
        bg = ImageStat.Stat(scene_img.crop((x0, y0, x1, y1))).mean[:3]
        a = text_alpha(b["text"], x1 - x0, y1 - y0, bold)
        out.paste(Image.new("RGB", a.size, text_color(prompt, bg)), (x0, y0), a)
        ink.paste(a, (x0, y0), a)
    return out, ink, [b["bbox_xyxy"] for b in blocks]


def band_mask(size, boxes, ink, pad: int):
    """Repaint mask: boxes grown by pad, minus the strokes grown by 2 px."""
    from PIL import Image, ImageChops, ImageDraw, ImageFilter

    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for x0, y0, x1, y1 in boxes:
        d.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], fill=255)
    keep = ink.point(lambda v: 255 if v > 32 else 0).filter(ImageFilter.MaxFilter(5))
    return ImageChops.subtract(m, keep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Tongyi-MAI/Z-Image")
    ap.add_argument("--prompts", nargs="+", default=["eval/prompts.jsonl"])
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--pads", type=int, nargs="+", default=[16, 48])
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    import torch
    from diffusers import ZImageInpaintPipeline, ZImagePipeline
    from PIL import Image, ImageFilter

    out = a.out.expanduser()
    rows = load_prompts(a.prompts)
    if a.subset:
        rows = stratified(rows, a.subset)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "prompts.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cfgs = ["plain"] + [f"b{p}-s{s}" for p in a.pads for s in a.strengths]
    for c in cfgs + ["scene", "mask"]:
        (out / c / "images").mkdir(parents=True, exist_ok=True)

    pipe = ZImagePipeline.from_pretrained(a.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False).to("cuda")
    inp = ZImageInpaintPipeline.from_pipe(pipe)
    R = a.resolution
    kw = dict(height=R, width=R, num_inference_steps=a.steps, guidance_scale=a.guidance)
    t0, n = time.time(), 0
    for i, r in enumerate(rows):
        iid = f"{r['id']}_s{a.seed}"
        prompt = r["prompt_ko"]
        sp = out / "scene" / "images" / f"{iid}.png"
        if sp.exists():
            scene = Image.open(sp).convert("RGB")
        else:
            scene = pipe(prompt=blank_prompt(prompt), negative_prompt=NEG_TEXT,
                         generator=torch.Generator("cuda").manual_seed(a.seed), **kw).images[0]
            scene.save(sp)
        comp, ink, boxes = render(scene, r, prompt)
        comp.save(out / "plain" / "images" / f"{iid}.png")
        for p in a.pads:
            m = band_mask(comp.size, boxes, ink, p)
            m.save(out / "mask" / "images" / f"{iid}_b{p}.png")
            soft = m.filter(ImageFilter.GaussianBlur(3))
            for s in a.strengths:
                op = out / f"b{p}-s{s}" / "images" / f"{iid}.png"
                if op.exists():
                    continue
                h = inp(prompt=prompt, image=comp, mask_image=m, strength=s, negative_prompt=NEG_TEXT,
                        generator=torch.Generator("cuda").manual_seed(a.seed), **kw).images[0]
                # strokes and everything outside the band come back pixel-exact
                Image.composite(h.resize(comp.size), comp, soft).save(op)
                n += 1
        print(f"{i + 1}/{len(rows)} {r['id']} {time.time() - t0:.0f}s", flush=True)
    (out / "settings.json").write_text(json.dumps(dict(
        model=a.model, steps=a.steps, guidance=a.guidance, resolution=R, seed=a.seed, strengths=a.strengths,
        pads=a.pads, blank=BLANK, negative=NEG_TEXT, n_prompts=len(rows), n_harmonized=n,
        wall_sec=time.time() - t0), indent=1, ensure_ascii=False))
    print("RENDER DONE", " ".join(cfgs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
