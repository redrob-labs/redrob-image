#!/usr/bin/env python3
"""Glyph condition for the glyph pilot (docs/glyph-pilot-plan.md).

A glyph map is the target text drawn white-on-black in one fixed font (Noto Sans KR Bold, all 11,172 Hangul
syllables) at the place it should appear. It goes into Z-Image's omni mode as a clean condition image: the
transformer already takes [condition images..., target] with noise_mask [0, 1], gives every image the same
height/width position ids (only the frame axis differs), and modulates clean tokens with t = 1. So the glyph
tokens line up with the target tokens they describe, and no new parameter is needed.

Pieces used by precompute.py, train_lora.py and eval_baseline.py:
  render(blocks, w, h)        glyph map (PIL RGB) from text blocks with a polygon or bbox_xyxy
  layout(text, scene, w, h)   default eval layout: one block per prompt, box by scene
  segments(tok, te, prompts)  omni caption segments (seg0, [seg1 per prompt], seg2), as ZImageOmniPipeline
  omni(transformer, ...)      one omni forward over a batch: [glyph, x_t] per item, target output only
  patched(transformer, ...)   context manager so ZImagePipeline's standard calls run in omni mode

Overlay check (CPU): python scripts/glyph.py --check <shard.tar> --n 20 --out <dir>
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager
from pathlib import Path

FONT = os.environ.get("REDROB_GLYPH_FONT", str(Path("~/.local/share/fonts/NotoSansKR-Bold.ttf").expanduser()))
FILL = 0.92  # share of the box the text fills along its tighter axis

SEG0 = "<|im_start|>user\n<|vision_start|>"
SEG1 = "<|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n<|vision_start|>"
SEG2 = "<|vision_end|><|im_end|>"

_fonts: dict = {}


def _font(size: int, path: str = FONT):
    from PIL import ImageFont

    k = (path, size)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(path, size)
    return _fonts[k]


def quad(block: dict):
    """(TL, TR, BR, BL) of a block. Polygons run along the top left->right, then the bottom right->left
    (4 points for AI Hub, 12 for synth's curved lines)."""
    p = block.get("polygon")
    if p and len(p) >= 4 and len(p) % 2 == 0:
        n = len(p)
        return [tuple(map(float, p[0])), tuple(map(float, p[n // 2 - 1])), tuple(map(float, p[n // 2])),
                tuple(map(float, p[-1]))]
    x0, y0, x1, y1 = (float(v) for v in block["bbox_xyxy"])
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _blocks_lines(blocks: list[dict]):
    """One (text, quad) per drawn line: synth blocks carry per-line text and polygons."""
    for b in blocks:
        lt, lp = b.get("lines_text"), b.get("line_polygons")
        if lt and lp and len(lt) == len(lp):
            for t, p in zip(lt, lp):
                if t.strip():
                    yield t, quad({"polygon": p})
        elif (b.get("text") or "").strip():
            yield b["text"], quad(b)


def _text_tile(text: str, w: int, h: int, vertical: bool):
    """Text centred on a w x h black tile, aspect kept, filling FILL of the tighter axis."""
    from PIL import Image, ImageDraw

    tile = Image.new("L", (max(w, 1), max(h, 1)), 0)
    if w < 2 or h < 2:
        return tile
    ref = 64
    f = _font(ref)
    chars = [c for c in text if not c.isspace()] if vertical else None
    if vertical:
        cw = max(f.getbbox(c)[2] - f.getbbox(c)[0] for c in chars)
        tw, th = cw, ref * len(chars)
    else:
        x0, y0, x1, y1 = f.getbbox(text)
        tw, th = x1 - x0, y1 - y0
    s = FILL * min(w / max(tw, 1), h / max(th, 1))
    size = max(4, int(ref * s))
    f = _font(size)
    d = ImageDraw.Draw(tile)
    if vertical:
        step = size
        top = (h - step * len(chars)) / 2
        for k, c in enumerate(chars):
            d.text((w / 2, top + step * (k + 0.5)), c, fill=255, font=f, anchor="mm")
    else:
        d.text((w / 2, h / 2), text, fill=255, font=f, anchor="mm")
    return tile


def _perspective_coeffs(dst, src):
    """PIL PERSPECTIVE coefficients mapping output (dst) points to input (src) points."""
    import numpy as np

    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        B += [u, v]
    return np.linalg.solve(np.asarray(A, float), np.asarray(B, float)).tolist()


def render(blocks: list[dict], w: int, h: int):
    """Glyph map, RGB, same size as the image the blocks describe."""
    from PIL import Image, ImageChops

    out = Image.new("L", (w, h), 0)
    for text, q in _blocks_lines(blocks):
        (tl, tr, br, bl) = q
        qw = (math.dist(tl, tr) + math.dist(bl, br)) / 2
        qh = (math.dist(tl, bl) + math.dist(tr, br)) / 2
        if qw < 2 or qh < 2:
            continue
        n = len([c for c in text if not c.isspace()])
        vertical = n >= 2 and qh > 1.3 * qw
        tw, th = int(round(qw)), int(round(qh))
        tile = _text_tile(text, tw, th, vertical)
        src = [(0, 0), (tw, 0), (tw, th), (0, th)]
        coeffs = _perspective_coeffs(q, src)
        warped = tile.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR, fillcolor=0)
        out = ImageChops.lighter(out, warped)
    return out.convert("RGB")


# Eval layout: one text box per scene, in fractions of the image (x0, y0, x1, y1).
SCENE_BOX = {
    "signboard": (0.08, 0.30, 0.92, 0.55),
    "poster": (0.10, 0.10, 0.90, 0.32),
    "book_cover": (0.15, 0.15, 0.85, 0.38),
    "product_label": (0.20, 0.38, 0.80, 0.62),
    "menu": (0.10, 0.08, 0.90, 0.26),
}


def layout(text: str, scene: str, w: int, h: int) -> list[dict]:
    """Default placement for an eval prompt. Text over 12 syllables with a space is split into two lines."""
    x0, y0, x1, y1 = SCENE_BOX.get(scene, (0.10, 0.35, 0.90, 0.60))
    X0, Y0, X1, Y1 = x0 * w, y0 * h, x1 * w, y1 * h
    lines = [text]
    if len(text.replace(" ", "")) > 12 and " " in text:
        mid = len(text) / 2
        cut = min((i for i, c in enumerate(text) if c == " "), key=lambda i: abs(i - mid))
        lines = [text[:cut].strip(), text[cut + 1:].strip()]
    lh = (Y1 - Y0) / len(lines)
    return [{"text": t, "bbox_xyxy": [X0, Y0 + k * lh, X1, Y0 + (k + 1) * lh]} for k, t in enumerate(lines)]


def encode_raw(tok, te, texts: list[str], max_len: int = 512):
    """Text-encoder hidden_states[-2] for raw strings (no chat template), padding dropped, as the omni pipeline."""
    import torch

    enc = tok(texts, padding="longest", max_length=max_len, truncation=True, return_tensors="pt")
    ids, mask = enc.input_ids.to(te.device), enc.attention_mask.to(te.device).bool()
    with torch.inference_mode():
        hs = te(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-2]
    return [hs[i][mask[i]].to(torch.bfloat16).cpu() for i in range(len(texts))]


def segments(tok, te, prompts: list[str]):
    """(seg0, [seg1 for each prompt], seg2) for one condition image."""
    seg0, seg2 = encode_raw(tok, te, [SEG0, SEG2])
    return seg0, encode_raw(tok, te, [SEG1.format(p) for p in prompts]), seg2


def omni(transformer, xt: list, t, seg1: list, glyph: list, seg0, seg2):
    """One forward over a batch. xt / glyph: per item (C, 1, H, W) latents; seg1: per item caption middle
    segment. Returns per item target output (C, 1, H, W)."""
    B = len(xt)
    return transformer(x=[[glyph[i], xt[i]] for i in range(B)], t=t,
                       cap_feats=[[seg0, seg1[i], seg2] for i in range(B)],
                       siglip_feats=[None] * B, image_noise_mask=[[0, 1]] * B, return_dict=False)[0]


@contextmanager
def patched(transformer, glyph_lat, seg0, seg2):
    """Run ZImagePipeline in omni mode: the pipeline passes its own prompt_embeds (give it seg1 for the prompt
    and for the negative), and every item of each call gets glyph_lat (C, 1, H, W) as the clean condition, the
    negative branch too (as ZImageOmniPipeline does)."""
    orig = transformer.forward

    def fwd(x, t, cap_feats, *args, return_dict=True, **kw):
        if isinstance(x[0], list):
            return orig(x, t, cap_feats, *args, return_dict=return_dict, **kw)
        B = len(x)
        g = glyph_lat.to(x[0].device, x[0].dtype)
        return orig(x=[[g, x[i]] for i in range(B)], t=t,
                    cap_feats=[[seg0.to(x[0].device), cap_feats[i], seg2.to(x[0].device)] for i in range(B)],
                    siglip_feats=[None] * B, image_noise_mask=[[0, 1]] * B, return_dict=return_dict)

    transformer.forward = fwd
    try:
        yield
    finally:
        transformer.forward = orig


# ---- channel mode (glyph v2): the glyph latent rides on extra input channels of the noisy latent ----
# The DiT patchifies (C, F, H, W) with C innermost, so a 32-channel input [x_t; glyph] gives every token the glyph
# patch at the same place as the image patch. GlyphEmbed splits the two: the frozen x_embedder sees the image half
# as before, and a new zero-initialised linear adds the glyph half (ControlNet / AnyText style). At step 0 the model
# is exactly the one it wraps, and a zero glyph is exactly "no glyph".
PROJ_FILE = "glyph_proj.safetensors"
EMB_KEY = "2-1"


def _glyph_embed_cls():
    import torch.nn as nn

    class GlyphEmbed(nn.Module):
        def __init__(self, base: nn.Linear, in_ch: int):
            super().__init__()
            self.base, self.in_ch = base, in_ch
            self.proj = nn.Linear(base.in_features, base.out_features, bias=False)
            nn.init.zeros_(self.proj.weight)

        def forward(self, inp):
            if inp.shape[-1] == self.base.in_features:  # 16-channel input: no glyph
                return self.base(inp)
            n = inp.shape[0]
            v = inp.view(n, -1, 2 * self.in_ch)
            img = v[..., : self.in_ch].reshape(n, -1)
            gl = v[..., self.in_ch:].reshape(n, -1)
            return self.base(img) + self.proj(gl.to(self.proj.weight.dtype)).to(img.dtype)

    return GlyphEmbed


def install_channel(transformer, proj_path=None):
    """Wrap the x_embedder; returns the new trainable proj (fp32). Loads proj weights when proj_path exists."""
    import torch

    emb = transformer.all_x_embedder[EMB_KEY]
    if not hasattr(emb, "proj"):
        emb = _glyph_embed_cls()(emb, transformer.in_channels)
        transformer.all_x_embedder[EMB_KEY] = emb
    emb.proj.to(device=emb.base.weight.device, dtype=torch.float32)
    if proj_path and Path(proj_path).exists():
        from safetensors.torch import load_file

        emb.proj.weight.data.copy_(load_file(str(proj_path))["weight"].to(emb.proj.weight.device, torch.float32))
    emb.proj.weight.requires_grad_(True)
    return emb.proj


def save_proj(transformer, path):
    from safetensors.torch import save_file

    w = transformer.all_x_embedder[EMB_KEY].proj.weight
    save_file({"weight": w.detach().float().cpu().contiguous()}, str(path))


def cat_glyph(xt: list, glyph: list):
    """Per item (C, 1, H, W) noisy latent + glyph latent -> (2C, 1, H, W). A None glyph is a zero glyph."""
    import torch

    return [torch.cat([x, torch.zeros_like(x) if g is None else g.to(x.device, x.dtype)], 0) for x, g in zip(xt, glyph)]


@contextmanager
def patched_channel(transformer, glyph_lat):
    """Run ZImagePipeline with glyph_lat (C, 1, H, W) concatenated to every noisy latent, the negative branch too."""
    orig = transformer.forward

    def fwd(x, t, cap_feats, *args, **kw):
        return orig(cat_glyph(x, [glyph_lat] * len(x)), t, cap_feats, *args, **kw)

    transformer.forward = fwd
    try:
        yield
    finally:
        transformer.forward = orig


def encode_glyph(vae, img):
    """VAE latent (C, 1, h, w), shifted and scaled like precompute.py."""
    import numpy as np
    import torch

    x = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
    x = (x / 127.5 - 1.0).to(vae.device, vae.dtype)
    with torch.inference_mode():
        z = vae.encode(x).latent_dist.mode()
    z = (z - vae.config.shift_factor) * vae.config.scaling_factor
    return z[0].unsqueeze(1)


def generate(pipe, prompt: str, text: str, scene: str, res: int, negative: str = "", mode: str = "omni", **kw):
    """One ZImagePipeline image with the glyph condition for `text` at the default layout. Returns
    (image, glyph map). kw goes to the pipeline (num_inference_steps, guidance_scale, generator, ...).
    mode "channel" needs install_channel() on pipe.transformer first."""
    g = render(layout(text, scene, res, res), res, res)
    lat = encode_glyph(pipe.vae, g)
    if mode == "channel":
        with patched_channel(pipe.transformer, lat):
            img = pipe(prompt=prompt, negative_prompt=negative, height=res, width=res, **kw).images[0]
        return img, g
    seg0, (s1, s1n), seg2 = segments(pipe.tokenizer, pipe.text_encoder, [prompt, negative])
    dev = pipe.vae.device
    with patched(pipe.transformer, lat, seg0, seg2):
        img = pipe(prompt_embeds=[s1.to(dev)], negative_prompt_embeds=[s1n.to(dev)], height=res, width=res, **kw).images[0]
    return img, g


def main() -> int:
    import argparse
    import io
    import json
    import tarfile

    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, help="webdataset shard (.tar) with <key>.jpg + <key>.json")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    k = 0
    with tarfile.open(a.check) as tf:
        mem = {m.name: m for m in tf.getmembers()}
        for name in sorted(mem):
            if not name.endswith(".json") or k >= a.n:
                continue
            key = name[:-5]
            if key + ".jpg" not in mem:
                continue
            meta = json.load(tf.extractfile(mem[name]))
            img = Image.open(io.BytesIO(tf.extractfile(mem[key + ".jpg"]).read())).convert("RGB")
            g = render(meta.get("text_blocks") or [], img.width, img.height)
            # glyph in red over the dimmed photo, then the glyph alone, side by side
            over = Image.blend(img, Image.new("RGB", img.size, 0), 0.5)
            red = Image.new("RGB", img.size, (255, 40, 40))
            over.paste(red, mask=g.convert("L"))
            both = Image.new("RGB", (img.width * 2, img.height))
            both.paste(over, (0, 0))
            both.paste(g, (img.width, 0))
            both.thumbnail((1200, 1200))
            both.save(out / f"{k:02d}-{Path(key).name}.jpg", quality=88)
            k += 1
    print(f"{k} overlays in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
