#!/usr/bin/env python3
"""Plan C stage 0 (docs/reward-training-plan.md): right/wrong image pairs that differ in ONE jamo.

For each prompt: one blank scene (Z-Image Base), then the target text and N one-jamo corruptions of it are
drawn with the font into the same box and harmonized the plan-B way (band inpaint, strokes pasted back
pixel-exact). So the right image really carries the right strokes and the wrong one really carries the wrong
ones, and nothing else differs except the inpainted band. A reward that cannot rank these pairs cannot steer
training toward correct strokes.

Corruption kinds (one syllable, one jamo):
  vowel    ㅐ<->ㅓ, ㅔ<->ㅕ, ㅏ<->ㅑ ...   similar-looking vowel
  tense    ㄲ->ㄱ, ㄱ->ㄲ ...              double <-> single initial
  initial  ㄱ<->ㅋ, ㄷ<->ㅌ, ㅇ<->ㅎ ...    aspirated / similar initial
  final    받침 swapped, added or dropped

  HF_HOME=~/hf python scripts/make_jamo_pairs.py --prompts eval/prompts.jsonl eval/prompts_unseen_syllables.jsonl \
      --n-prompts 100 --per 5 --out ~/runs/reward-check
writes <out>/pairs.jsonl and <out>/images/{good,bad,scene}/
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_baseline import load_prompts, stratified  # noqa: E402
from render_harmonize import NEG_TEXT, band_mask, blank_prompt, render  # noqa: E402

L = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
V = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
T = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")


def _pairs(*groups):
    m: dict[str, list[str]] = {}
    for g in groups:
        for a in g:
            m.setdefault(a, []).extend(b for b in g if b != a)
    return m


VOWEL = _pairs("ㅐㅓㅔ", "ㅕㅔ", "ㅏㅑ", "ㅗㅛ", "ㅜㅠ", "ㅐㅒ", "ㅔㅖ", "ㅚㅙㅞ", "ㅘㅝ", "ㅡㅜ", "ㅣㅢ")
TENSE = _pairs("ㄱㄲ", "ㄷㄸ", "ㅂㅃ", "ㅅㅆ", "ㅈㅉ")
INITIAL = _pairs("ㄱㅋ", "ㄷㅌ", "ㅂㅍ", "ㅈㅊ", "ㅇㅎ", "ㄴㄹ", "ㅁㅂ")
FINAL = _pairs("ㄱㅋㄲ", "ㄴㄹ", "ㅁㅇ", "ㅂㅍ", "ㅅㅆㅈ", "ㄷㅌ")
KINDS = ["vowel", "tense", "initial", "final"]


def split(c: str):
    o = ord(c) - 0xAC00
    if not 0 <= o < 11172:
        return None
    return o // 588, (o % 588) // 28, o % 28


def join(l: int, v: int, t: int) -> str:
    return chr(0xAC00 + (l * 21 + v) * 28 + t)


def corrupt(c: str, kind: str, rng: random.Random) -> str | None:
    p = split(c)
    if p is None:
        return None
    l, v, t = p
    if kind == "vowel" and V[v] in VOWEL:
        return join(l, V.index(rng.choice(VOWEL[V[v]])), t)
    if kind == "tense" and L[l] in TENSE:
        return join(L.index(rng.choice(TENSE[L[l]])), v, t)
    if kind == "initial" and L[l] in INITIAL:
        return join(L.index(rng.choice(INITIAL[L[l]])), v, t)
    if kind == "final":
        if t and T[t] in FINAL and rng.random() < 0.6:
            return join(l, v, T.index(rng.choice(FINAL[T[t]])))
        if t:
            return join(l, v, 0)  # drop the 받침
        return join(l, v, T.index(rng.choice("ㄱㄴㄹㅁㅇ")))  # add one
    return None


def corruptions(text: str, n: int, rng: random.Random) -> list[dict]:
    """Up to n distinct one-jamo corruptions, kinds spread round-robin."""
    pos = [i for i, c in enumerate(text) if split(c) is not None]
    out, seen = [], {text}
    for k in range(n * 12):
        if len(out) >= n or not pos:
            break
        kind = KINDS[k % len(KINDS)]
        i = rng.choice(pos)
        w = corrupt(text[i], kind, rng)
        if not w:
            continue
        bad = text[:i] + w + text[i + 1:]
        if bad in seen:
            continue
        seen.add(bad)
        out.append(dict(wrong=bad, kind=kind, pos=i, good_char=text[i], bad_char=w))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Tongyi-MAI/Z-Image")
    ap.add_argument("--prompts", nargs="+", default=["eval/prompts.jsonl", "eval/prompts_unseen_syllables.jsonl"])
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--per", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--pad", type=int, default=16)
    ap.add_argument("--strength", type=float, default=0.3)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    import torch
    from diffusers import ZImageInpaintPipeline, ZImagePipeline
    from PIL import Image, ImageFilter

    out = a.out.expanduser()
    for d in ("scene", "good", "bad"):
        (out / "images" / d).mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    rows = stratified(load_prompts(a.prompts), a.n_prompts)
    plan = []
    for r in rows:
        for k, c in enumerate(corruptions(r["target_text"], a.per, rng)):
            plan.append(dict(pair_id=f"{r['id']}_{k}", **{f: r[f] for f in r}, **c))
    with (out / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in plan:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(rows)} prompts, {len(plan)} pairs", flush=True)

    pipe = ZImagePipeline.from_pretrained(a.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False).to("cuda")
    pipe.vae.enable_tiling()
    inp = ZImageInpaintPipeline.from_pipe(pipe)
    R = a.resolution
    kw = dict(height=R, width=R, num_inference_steps=a.steps, guidance_scale=a.guidance)

    def harmonize(scene, r, text, path):
        if path.exists():
            return
        comp, ink, boxes = render(scene, dict(r, target_text=text), r["prompt_ko"])
        m = band_mask(comp.size, boxes, ink, a.pad)
        h = inp(prompt=r["prompt_ko"], image=comp, mask_image=m, strength=a.strength, negative_prompt=NEG_TEXT,
                generator=torch.Generator("cuda").manual_seed(a.seed), **kw).images[0]
        Image.composite(h.resize(comp.size), comp, m.filter(ImageFilter.GaussianBlur(3))).save(path)
        torch.cuda.empty_cache()

    t0 = time.time()
    by_prompt: dict[str, list[dict]] = {}
    for p in plan:
        by_prompt.setdefault(p["id"], []).append(p)
    for i, r in enumerate(rows):
        sp = out / "images" / "scene" / f"{r['id']}.png"
        if sp.exists():
            scene = Image.open(sp).convert("RGB")
        else:
            scene = pipe(prompt=blank_prompt(r["prompt_ko"]), negative_prompt=NEG_TEXT,
                         generator=torch.Generator("cuda").manual_seed(a.seed), **kw).images[0]
            scene.save(sp)
        harmonize(scene, r, r["target_text"], out / "images" / "good" / f"{r['id']}.png")
        for p in by_prompt.get(r["id"], []):
            harmonize(scene, r, p["wrong"], out / "images" / "bad" / f"{p['pair_id']}.png")
        print(f"{i + 1}/{len(rows)} {r['id']} {time.time() - t0:.0f}s", flush=True)
    (out / "settings.json").write_text(json.dumps(dict(
        vars(a), out=str(out), n_prompts=len(rows), n_pairs=len(plan), negative=NEG_TEXT,
        wall_sec=time.time() - t0), default=str, indent=1, ensure_ascii=False))
    print("PAIRS DONE", len(plan), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
