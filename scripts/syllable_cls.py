#!/usr/bin/env python3
"""Hangul syllable classifier for the plan C reward (docs/reward-training-plan.md, stage 0).

Qwen3-VL could not price one wrong jamo in a single syllable cell (stage-0 v5 89.4%, rare and unseen
syllables worst): it reads words it knows. This classifier knows all 11,172 syllables equally, because it
learns them from fonts, so an unseen syllable is just another class.

  render   pre-render cells: every syllable a font covers, with random neighbours cut in at the cell edge,
           cropped the way reward.syllable_cells crops (12% pad). Text = 255 on 0, 64x64, uint8.
  train    ResNet-18 (1 channel, 3x3 stem). Ink, background, blur, low resolution, affine and noise are
           added on the GPU per batch, so the cells stay small on disk.
  check    stage-0 pair ranking with the classifier alone: score = mean log P(target syllable) over the cells.
           Noto Sans KR (the font the pairs are drawn in) is never in training, and no weight is tuned on the
           pairs, so the number is a held-out one.

  python scripts/syllable_cls.py render --fonts ~/.local/share/fonts/ko-train --out ~/data/syl-cls
  python scripts/syllable_cls.py train --data ~/data/syl-cls --out ~/runs/syl-cls
  python scripts/syllable_cls.py check --model ~/runs/syl-cls/model.pt --pairs ~/runs/reward-check
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

N_CLS = 11172
SIZE = 64
HOLDOUT = ("NotoSansKR-Regular.ttf", "NotoSansKR-Bold.ttf")  # val only; the stage-0 pairs use these


def syl(i: int) -> str:
    return chr(0xAC00 + i)


def coverage(path: str) -> list[int]:
    from fontTools.ttLib import TTFont

    cmap = TTFont(path, fontNumber=0, lazy=True).getBestCmap() or {}
    return [i for i in range(N_CLS) if 0xAC00 + i in cmap]


# ---------------------------------------------------------------- render

_FONTS: dict = {}


def _font(path: str, size: int):
    from PIL import ImageFont

    if (path, size) not in _FONTS:
        _FONTS[path, size] = ImageFont.truetype(path, size)
    return _FONTS[path, size]


def cell(ch: str, path: str, rng: random.Random, neighbours: list[str]):
    """One cell as reward.syllable_cells would crop it from a line drawn in this font."""
    from PIL import Image, ImageDraw

    size = rng.randint(48, 72)
    f = _font(path, size)
    left = "".join(rng.choice(neighbours) for _ in range(rng.randint(0, 2))) if rng.random() < 0.75 else ""
    right = "".join(rng.choice(neighbours) for _ in range(rng.randint(0, 2))) if rng.random() < 0.75 else ""
    if rng.random() < 0.1:
        left += " "
    if rng.random() < 0.1:
        right = " " + right
    text = left + ch + right
    W, H = int(f.getlength(text)) + 4 * size, 3 * size
    img = Image.new("L", (W, H), 0)
    cx, cy = W / 2, H / 2
    ImageDraw.Draw(img).text((cx, cy), text, font=f, fill=255, anchor="mm")
    _, t, _, bt = f.getbbox(text, anchor="mm")
    x = cx - f.getlength(text) / 2
    a = x + f.getlength(left)
    e = a + f.getlength(ch)
    pad = rng.uniform(0.06, 0.18)
    px, py = pad * (e - a), pad * (bt - t)
    jx, jy = rng.uniform(-0.06, 0.06) * (e - a), rng.uniform(-0.06, 0.06) * (bt - t)
    box = (int(a - px + jx), int(cy + t - py + jy), int(e + px + jx), int(cy + bt + py + jy))
    return img.crop(box).resize((SIZE, SIZE), Image.BILINEAR)


def _render_font(job):
    import numpy as np

    path, variants, seed = job
    rng = random.Random(seed)
    cov = coverage(path)
    if len(cov) < 2000:
        return path, None, None
    common = [syl(i) for i in cov]
    xs, ys = [], []
    for i in cov:
        for _ in range(variants):
            try:
                xs.append(np.asarray(cell(syl(i), path, rng, common), dtype=np.uint8))
                ys.append(i)
            except Exception:  # a broken glyph in a display font: skip it
                pass
    return path, np.stack(xs), np.asarray(ys, dtype=np.int16)


def render(fonts: Path, out: Path, variants: int, holdout_dir: Path | None) -> None:
    import numpy as np
    from multiprocessing import Pool

    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(str(p) for p in fonts.glob("*.ttf") if p.name not in HOLDOUT)
    jobs = []
    for k, p in enumerate(paths):
        n = len(coverage(p))
        # full-coverage fonts carry the rare syllables, so they get more variants
        jobs.append((p, variants if n >= 11000 else max(1, variants // 2), 1000 + k))
    if holdout_dir is not None:
        for k, name in enumerate(HOLDOUT):
            p = holdout_dir / name
            if p.exists():
                jobs.append((str(p), 1, 9000 + k))
    meta = {}
    with Pool(8) as pool:
        for path, x, y in pool.imap_unordered(_render_font, jobs):
            name = Path(path).name
            if x is None:
                print(f"skip {name} (coverage < 2000)", flush=True)
                continue
            split = "val" if name in HOLDOUT else "train"
            np.savez(out / f"{split}-{Path(path).stem}.npz", x=x, y=y)
            meta[name] = dict(split=split, n=int(len(y)), classes=int(len(set(y.tolist()))))
            print(f"{split} {name} {len(y)}", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def _patches_one(job):
    import numpy as np
    from PIL import Image

    path, k, seed = job
    rng = random.Random(seed)
    try:
        im = Image.open(path).convert("L")
        im.draft("L", (im.width // 2, im.height // 2))
    except Exception:
        return []
    out = []
    for _ in range(k):
        s = rng.randint(32, max(33, min(im.size) // 3))
        x, y = rng.randint(0, im.width - s), rng.randint(0, im.height - s)
        out.append(np.asarray(im.crop((x, y, x + s, y + s)).resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8))
    return out


def patches(src: Path, out: Path, n_img: int, per: int) -> None:
    """Grey 64x64 patches of real photos (signs, walls, shelves): backgrounds and clutter for augment()."""
    import os
    import numpy as np
    from multiprocessing import Pool

    files = []
    for root, _, names in os.walk(src):
        files += [os.path.join(root, f) for f in names if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.Random(0).shuffle(files)
    with Pool(8) as pool:
        xs = [p for ps in pool.imap_unordered(_patches_one, [(f, per, i) for i, f in enumerate(files[:n_img])],
                                               chunksize=16) for p in ps]
    np.save(out / "bg.npy", np.stack(xs))
    print(f"bg patches {len(xs)} from {min(n_img, len(files))} photos", flush=True)


# ---------------------------------------------------------------- model

def model():
    import torch.nn as nn
    import torchvision

    m = torchvision.models.resnet18(num_classes=N_CLS)
    m.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
    return m


def normalise(x):
    """Per-cell standardisation, so ink colour and lighting drop out (train and inference alike)."""
    m = x.mean((1, 2, 3), keepdim=True)
    s = x.std((1, 2, 3), keepdim=True).clamp_min(1e-3)
    return (x - m) / s


def augment(mask, g, bgs=None):
    """mask: uint8 [B,64,64] (255 = ink) on the GPU -> float [B,1,64,64] photo-like cells.

    bgs: uint8 [N,64,64] real photo patches. v1 trained on clean procedural backgrounds only and misread 61%
    of the stage-0 composite cells: those cells carry sign borders, scene texture and clutter at the edges."""
    import torch
    import torch.nn.functional as F

    B, dev = mask.shape[0], mask.device

    def u(*s):
        return torch.rand(*s, device=dev)

    m = mask[:, None].float() / 255
    # geometry: small rotation, shear, scale, shift (cells come from a straight layout, so keep it mild)
    ang = (u(B) - 0.5) * math.radians(10)
    sh = (u(B) - 0.5) * 0.25
    sc = 1 + (u(B, 2) - 0.5) * 0.2
    tr = (u(B, 2) - 0.5) * 0.12
    c, s = torch.cos(ang), torch.sin(ang)
    th = torch.stack([torch.stack([c * sc[:, 0], -s + sh, tr[:, 0]], 1),
                      torch.stack([s, c * sc[:, 1], tr[:, 1]], 1)], 1)
    m = F.grid_sample(m, F.affine_grid(th, m.shape, align_corners=False), align_corners=False)
    # stroke weight: dilate or erode a little
    r = u(B)
    dil = F.max_pool2d(m, 3, 1, 1)
    ero = -F.max_pool2d(-m, 3, 1, 1)
    m = torch.where((r < 0.15)[:, None, None, None], dil, torch.where((r > 0.9)[:, None, None, None], ero, m))
    # background: low-frequency noise + gradient, mixed with a real photo patch most of the time
    lo = F.interpolate(u(B, 1, 4, 4), size=SIZE, mode="bicubic", align_corners=False)
    gx = torch.linspace(-1, 1, SIZE, device=dev)
    grad = (u(B, 1, 1, 1) - 0.5) * gx[None, None, None, :]
    bg = (u(B, 1, 1, 1) * 0.8 + 0.1) + 0.3 * (lo - 0.5) + 0.3 * grad
    if bgs is not None:
        p = bgs[torch.randint(0, len(bgs), (B,), device=dev)][:, None].float() / 255
        flip = u(B) < 0.5
        p = torch.where(flip[:, None, None, None], p.flip(3), p)
        a = u(B, 1, 1, 1) * (u(B, 1, 1, 1) < 0.75)
        bg = (1 - a) * bg + a * p
    # clutter at the edges: sign borders, shelf lines, frames
    ar = torch.arange(SIZE, device=dev)
    for _ in range(3):
        sel = (u(B) < 0.25).float()
        vert = u(B) < 0.5
        pos = torch.where(u(B) < 0.5, torch.randint(0, 10, (B,), device=dev), torch.randint(54, SIZE, (B,), device=dev))
        wid = torch.randint(1, 5, (B,), device=dev)
        band = ((ar[None] >= pos[:, None]) & (ar[None] < pos[:, None] + wid[:, None])).float()
        lm = torch.where(vert[:, None, None], band[:, None, :].expand(B, SIZE, SIZE),
                         band[:, :, None].expand(B, SIZE, SIZE))[:, None] * sel[:, None, None, None]
        bg = bg * (1 - lm) + u(B, 1, 1, 1) * lm
    # ink: brighter or darker than the local background, contrast >= 0.12; sometimes an outline
    bm = (bg * m).sum((2, 3), keepdim=True) / m.sum((2, 3), keepdim=True).clamp_min(1)
    sign = torch.where(u(B, 1, 1, 1) < 0.5, -1.0, 1.0)
    con = u(B, 1, 1, 1) * 0.7 + 0.12
    ink = (bm + sign * con).clamp(0, 1)
    ink = torch.where((ink - bm).abs() < 0.1, bm - sign * con, ink)
    out = (F.max_pool2d(m, 5, 1, 2) - m).clamp(0, 1) * (u(B, 1, 1, 1) < 0.2)
    x = bg * (1 - out) + (bm - sign * con * 0.8).clamp(0, 1) * out
    x = x * (1 - m) + ink * m
    # resolution: model-drawn cells are often 20-40 px; down then up blurs like that
    for lo_res in (16, 24, 32, 48):
        sel = u(B) < 0.15
        if sel.any():
            y = F.interpolate(x[sel], size=lo_res, mode="area")
            x[sel] = F.interpolate(y, size=SIZE, mode="bilinear", align_corners=False)
    x = x + torch.randn_like(x) * u(B, 1, 1, 1) * 0.06
    return normalise(x)


def train(data: Path, out: Path, steps: int, bs: int, lr: float) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F

    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    tr = [np.load(p) for p in sorted(data.glob("train-*.npz"))]
    va = [np.load(p) for p in sorted(data.glob("val-*.npz"))]
    X = torch.from_numpy(np.concatenate([d["x"] for d in tr])).to(dev)
    Y = torch.from_numpy(np.concatenate([d["y"] for d in tr]).astype(np.int64)).to(dev)
    XV = torch.from_numpy(np.concatenate([d["x"] for d in va])).to(dev)
    YV = torch.from_numpy(np.concatenate([d["y"] for d in va]).astype(np.int64)).to(dev)
    print(f"train {len(Y)} cells, {len(set(Y.tolist()))} classes; val {len(YV)}", flush=True)
    bgs = torch.from_numpy(np.load(data / "bg.npy")).to(dev) if (data / "bg.npy").exists() else None
    print(f"bg patches {0 if bgs is None else len(bgs)}", flush=True)
    net = model().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    g = torch.Generator(device=dev).manual_seed(0)
    t0, log = time.time(), (out / "train.jsonl").open("a")

    def val():
        net.eval()
        ok, n, lp = 0, 0, 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(YV), 2048):
                x = normalise(XV[i:i + 2048, None].float() / 255)
                o = net(x.contiguous(memory_format=torch.channels_last)).float()
                ok += (o.argmax(1) == YV[i:i + 2048]).sum().item()
                lp += F.log_softmax(o, 1).gather(1, YV[i:i + 2048, None]).sum().item()
                n += len(o)
        net.train()
        return ok / n, lp / n

    for step in range(1, steps + 1):
        idx = torch.randint(0, len(Y), (bs,), device=dev, generator=g)
        x = augment(X[idx], g, bgs).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(net(x).float(), Y[idx], label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps:
            rec = dict(step=step, loss=round(loss.item(), 4), sec=round(time.time() - t0))
            if step % 2500 == 0 or step == steps:
                rec["val_acc"], rec["val_lp"] = (round(v, 4) for v in val())
                torch.save(net.state_dict(), out / "model.pt")
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
    print("SYL-CLS TRAIN DONE", flush=True)


# ---------------------------------------------------------------- inference

class SylCls:
    def __init__(self, path: Path, device: str = "cuda"):
        import torch

        self.torch, self.dev = torch, device
        self.net = model().to(device)
        self.net.load_state_dict(torch.load(path, map_location=device))
        self.net.eval()

    def logp(self, cells, targets: list[str]) -> list[float]:
        """log P(target syllable) for each PIL cell."""
        import numpy as np
        import torch.nn.functional as F
        from PIL import Image

        torch = self.torch
        if not cells:
            return []
        x = np.stack([np.asarray(c.convert("L").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255
                      for c in cells])
        x = normalise(torch.from_numpy(x)[:, None].to(self.dev))
        y = torch.tensor([ord(t) - 0xAC00 for t in targets], device=self.dev)[:, None]
        with torch.no_grad():
            return F.log_softmax(self.net(x).float(), 1).gather(1, y)[:, 0].tolist()

    def top(self, cell, k: int = 3) -> list[tuple[str, float]]:
        import numpy as np
        from PIL import Image

        torch = self.torch
        x = np.asarray(cell.convert("L").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255
        x = normalise(torch.from_numpy(x)[None, None].to(self.dev))
        with torch.no_grad():
            p = torch.softmax(self.net(x).float()[0], 0)
        v, i = p.topk(k)
        return [(syl(int(a)), round(float(b), 3)) for a, b in zip(i, v)]


def check(model_path: Path, d: Path, tag: str) -> dict:
    from PIL import Image
    from reward import syllable_cells

    cls = SylCls(model_path)
    pairs = [json.loads(l) for l in (d / "pairs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    cache: dict[str, dict] = {}

    def score(key, path, r):
        if key not in cache:
            cells = list(syllable_cells(Image.open(path).convert("RGB"), r))
            lp = cls.logp([c for c, _ in cells], [s for _, s in cells])
            cache[key] = dict(cls=lp, cls_mean=sum(lp) / max(1, len(lp)), cls_min=min(lp) if lp else 0.0,
                              read=[cls.top(c, 1)[0][0] for c, _ in cells])
        return cache[key]

    rows = []
    for p in pairs:
        good = score("good/" + p["id"], d / "images" / "good" / f"{p['id']}.png", p)
        bad = score("bad/" + p["pair_id"], d / "images" / "bad" / f"{p['pair_id']}.png", p)
        rows.append(dict(p, good=good, bad=bad))

    tests = {"cls_mean": lambda x: x["good"]["cls_mean"] > x["bad"]["cls_mean"],
             "cls_min": lambda x: x["good"]["cls_min"] > x["bad"]["cls_min"],
             "good_read_exact": lambda x: "".join(x["good"]["read"]) == "".join(
                 c for c in x["target_text"] if 0xAC00 <= ord(c) <= 0xD7A3)}

    def table(rs):
        return {k: round(sum(f(x) for x in rs) / max(1, len(rs)), 3) for k, f in tests.items()} | {"n": len(rs)}

    res = {"overall": table(rows)}
    for key in ("kind", "split", "has_unseen_syllable"):
        for v in sorted({str(x.get(key)) for x in rows}):
            res[f"{key}={v}"] = table([x for x in rows if str(x.get(key)) == v])
    res["pass"] = res["overall"]["cls_mean"] >= 0.95
    res["score"] = "cls_mean (declared before the run; nothing tuned on these pairs)"
    with (d / f"rewards-{tag}.jsonl").open("w", encoding="utf-8") as f:
        for k, v in cache.items():
            f.write(json.dumps(dict(key=k, **v), ensure_ascii=False) + "\n")
    (d / f"check-{tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    fails = [dict(pair_id=x["pair_id"], kind=x["kind"], good=x["target_text"], bad=x["wrong"],
                  good_read="".join(x["good"]["read"]), bad_read="".join(x["bad"]["read"]),
                  g=round(x["good"]["cls_mean"], 3), b=round(x["bad"]["cls_mean"], 3))
             for x in rows if not tests["cls_mean"](x)]
    (d / f"fails-{tag}.json").write_text(json.dumps(fails, indent=1, ensure_ascii=False))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--fonts", type=Path, required=True)
    r.add_argument("--holdout-dir", type=Path, default=Path("~/.local/share/fonts"))
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--variants", type=int, default=4)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--bs", type=int, default=512)
    t.add_argument("--lr", type=float, default=2e-3)
    c = sub.add_parser("check")
    c.add_argument("--model", type=Path, required=True)
    c.add_argument("--pairs", type=Path, required=True)
    c.add_argument("--tag", default="v6cls")
    b = sub.add_parser("patches")
    b.add_argument("--src", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--n-img", type=int, default=6000)
    b.add_argument("--per", type=int, default=40)
    a = ap.parse_args()
    if a.cmd == "patches":
        patches(a.src.expanduser(), a.out.expanduser(), a.n_img, a.per)
    elif a.cmd == "render":
        render(a.fonts.expanduser(), a.out.expanduser(), a.variants, a.holdout_dir.expanduser())
    elif a.cmd == "train":
        train(a.data.expanduser(), a.out.expanduser(), a.steps, a.bs, a.lr)
    else:
        res = check(a.model.expanduser(), a.pairs.expanduser(), a.tag)
        print(json.dumps(res["overall"]), "PASS" if res["pass"] else "FAIL", flush=True)
        print("SYL-CLS CHECK DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
