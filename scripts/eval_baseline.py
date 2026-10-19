#!/usr/bin/env python3
"""Korean text-rendering baseline: generate images, OCR them, score, report.

  python scripts/eval_baseline.py --model Tongyi-MAI/Z-Image-Turbo --steps 9 --guidance 0 \
      --prompts eval/prompts.jsonl eval/prompts_unseen_syllables.jsonl --seeds 42 --out out/baseline-turbo

Outputs in --out:
  images/<id>_s<seed>.png   generated images (never committed)
  scores.jsonl              per-image OCR lines + CER/EM
  report.json / report.md   aggregates by split, bucket, scene, unseen-syllable flag
  review.html               20% stratified sample for human checking (relative image paths)
  metadata.json             commit, packages, GPU, sampler settings, seconds per image
Re-running with the same --out resumes: existing images are not regenerated.
"""
from __future__ import annotations

import argparse
import html
import json
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import metrics  # noqa: E402


def load_prompts(files: list[str]) -> list[dict]:
    rows = []
    for f in files:
        rows += [json.loads(l) for l in Path(f).open(encoding="utf-8") if l.strip()]
    return rows


def stratified(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """n prompts spread evenly over (split, bucket) groups, scenes interleaved inside each group."""
    g = defaultdict(list)
    for r in rows:
        g[(r["split"], r["bucket"])].append(r)
    keys = sorted(g)
    out, i = [], 0
    while len(out) < n and any(g.values()):
        k = keys[i % len(keys)]
        if g[k]:
            out.append(g[k].pop(0))
        i += 1
    return out


def git_commit() -> str:
    f = ROOT / "GIT_COMMIT"
    if f.exists():
        return f.read_text().strip()
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    v = {"python": platform.python_version()}
    for p in ("torch", "diffusers", "transformers", "accelerate", "paddleocr", "paddlepaddle-gpu", "easyocr"):
        try:
            v[p] = version(p)
        except PackageNotFoundError:
            pass
    return v


def generate(a, rows, img_dir: Path) -> dict:
    import torch
    from diffusers import ZImagePipeline

    todo = [(r, s) for r in rows for s in a.seeds if not (img_dir / f"{r['id']}_s{s}.png").exists()]
    info = dict(generated=len(todo), sec_per_image=None)
    if not todo:
        return info
    t0 = time.time()
    pipe = ZImagePipeline.from_pretrained(a.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False).to("cuda")
    info["load_sec"] = time.time() - t0
    kw = dict(height=a.resolution, width=a.resolution, num_inference_steps=a.steps, guidance_scale=a.guidance)
    if a.guidance > 0:
        kw.update(negative_prompt=a.negative, cfg_normalization=a.cfg_normalization)
    secs = []
    for i, (r, s) in enumerate(todo):
        prompt = r["prompt_ko"] if a.lang == "ko" else r["prompt_en"]
        torch.cuda.synchronize()
        t = time.time()
        img = pipe(prompt=prompt, generator=torch.Generator("cuda").manual_seed(s), **kw).images[0]
        torch.cuda.synchronize()
        secs.append(time.time() - t)
        img.save(img_dir / f"{r['id']}_s{s}.png")
        if i % 10 == 0:
            print(f"gen {i + 1}/{len(todo)} {secs[-1]:.1f}s", flush=True)
    info.update(sec_per_image=sum(secs[1:] or secs) / len(secs[1:] or secs), first_image_sec=secs[0],
                peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
    del pipe
    torch.cuda.empty_cache()
    return info


def review_html(rows: list[dict], frac: float, title: str) -> str:
    g = defaultdict(list)
    for r in rows:
        g[(r["split"], r["bucket"])].append(r)
    rng = random.Random(0)
    pick = []
    for k in sorted(g):
        rs = g[k]
        pick += rng.sample(rs, max(1, round(len(rs) * frac)))
    trs = []
    for r in pick:
        rid = html.escape(r["image_id"])
        trs.append(
            f"<tr data-id='{rid}'><td><img src='images/{rid}.png' loading='lazy'></td>"
            f"<td>{html.escape(r['split'])}<br>{html.escape(r['bucket'])}<br>{html.escape(r['scene'])}</td>"
            f"<td class=t>{html.escape(r['target_text'])}</td><td class=t>{html.escape(r['ocr_concat'])}</td>"
            f"<td>{r['cer_nospace']:.2f}<br>{'EM' if r['em'] else ''}</td>"
            "<td>" + "".join(f"<label><input type=radio name='{rid}' value='{v}'>{v}</label><br>"
                             for v in ("correct", "minor", "wrong", "no-text")) + "</td></tr>")
    return f"""<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>
<style>body{{font-family:sans-serif}}img{{width:320px}}td{{vertical-align:top;padding:6px;border-bottom:1px solid #ccc}}
.t{{font-size:20px;max-width:260px}}</style>
<h1>{html.escape(title)}</h1><p>{len(pick)} of {len(rows)} images (stratified by split/bucket). Judge the rendered
text against the target, then press Export.</p><button onclick="exp()">Export JSON</button>
<table><tr><th>image</th><th>group</th><th>target</th><th>OCR</th><th>CER/EM</th><th>human</th></tr>
{''.join(trs)}</table>
<script>function exp(){{const o={{}};document.querySelectorAll('tr[data-id]').forEach(t=>{{const c=t.querySelector('input:checked');
if(c)o[t.dataset.id]=c.value}});const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(o,null,1)]));
a.download='human-review.json';a.click()}}</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--guidance", type=float, required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--cfg-normalization", action="store_true")
    ap.add_argument("--prompts", nargs="+", default=["eval/prompts.jsonl"])
    ap.add_argument("--subset", type=int, default=0, help="stratified subset size (0 = all)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--lang", choices=["ko", "en"], default="ko")
    ap.add_argument("--ocr", default="paddleocr")
    ap.add_argument("--review-frac", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = load_prompts(a.prompts)
    if a.subset:
        rows = stratified(rows, a.subset)
    t_all = time.time()
    gen = generate(a, rows, img_dir)

    import ocr_engines
    from PIL import Image

    t0 = time.time()
    eng = ocr_engines.make(a.ocr)
    ocr_load = time.time() - t0
    scored, ocr_secs = [], []
    for r in rows:
        for s in a.seeds:
            iid = f"{r['id']}_s{s}"
            t = time.time()
            lines = eng(Image.open(img_dir / f"{iid}.png"))
            ocr_secs.append(time.time() - t)
            scored.append(dict(r, seed=s, image_id=iid, lines=lines, **metrics.score_one(r["target_text"], lines)))
    with (out / "scores.jsonl").open("w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import torch

    meta = dict(
        model=a.model, steps=a.steps, guidance=a.guidance, negative_prompt=a.negative,
        cfg_normalization=a.cfg_normalization, resolution=a.resolution, seeds=a.seeds, lang=a.lang,
        prompts=a.prompts, subset=a.subset, n_prompts=len(rows), n_images=len(scored), dtype="bfloat16",
        ocr=a.ocr, ocr_sec_per_image=sum(ocr_secs) / len(ocr_secs), ocr_load_sec=ocr_load,
        generation=gen, git_commit=git_commit(), packages=versions(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        host=platform.node(), started=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t_all)),
        wall_sec=time.time() - t_all,
    )
    agg = metrics.aggregate(scored)
    title = f"{Path(a.model).name} steps={a.steps} cfg={a.guidance} {a.resolution}px lang={a.lang} OCR={a.ocr}"
    (out / "metadata.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    (out / "report.json").write_text(json.dumps(dict(meta=meta, metrics=agg), indent=1, ensure_ascii=False))
    (out / "report.md").write_text(metrics.markdown(agg, title) + "\n")
    (out / "review.html").write_text(review_html(scored, a.review_frac, title))
    print(json.dumps(agg["overall"]), json.dumps(agg["split"]))


if __name__ == "__main__":
    main()
