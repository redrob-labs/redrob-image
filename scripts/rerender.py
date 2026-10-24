#!/usr/bin/env python3
"""Plan B, second trial: re-draw the pasted font text so it belongs to the scene.

The first trial (render_harmonize.py) only repainted a band AROUND the strokes and pasted the strokes back,
so the letters stayed flat font pixels and every config looked pasted. Here the strokes themselves are
re-drawn, starting from the font composite (<trial>/plain/images), so the shape is given and the look comes
from the model:

  qwen          Qwen-Image-Edit-2511 (Apache-2.0) edits the whole composite with an instruction that names
                the text and forbids changing it
  zr<pad>-s<s>  Z-Image Base + r64 v2 Korean LoRA inpaints the text box grown by <pad> px, strokes included,
                at strength <s>; outside the soft mask the composite comes back pixel-exact

Every config is its own eval folder (<out>/<cfg>/images/<id>_s<seed>.png), scored by eval_baseline.py and
score_crops.py without regenerating.

  HF_HOME=~/hf python scripts/rerender.py --trial ~/runs/render-harmonize-trial --out ~/runs/rerender-trial \
      --mode zimage --lora ~/runs/pilot-r64v2/final/pytorch_lora_weights.safetensors
  HF_HOME=~/hf python scripts/rerender.py --trial ~/runs/render-harmonize-trial --out ~/runs/rerender-trial --mode qwen
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import glyph as G  # noqa: E402
from render_harmonize import NEG_TEXT  # noqa: E402

QWEN = "Qwen/Qwen-Image-Edit-2511"
QWEN_INSTR = (
    'Make the text "{t}" look like it was really printed or painted on this surface: match the scene\'s '
    "lighting, shadows, material texture, perspective and slight wear. Keep exactly the same characters, "
    "the same position and the same size. Do not change, add or remove any letters, and do not add any "
    "other text. Keep everything else in the image unchanged."
)


def boxes_for(r: dict, size) -> list:
    W, H = size
    return [b["bbox_xyxy"] for b in G.layout(r["target_text"], r.get("scene", ""), W, H)]


def box_mask(size, boxes, pad: int):
    from PIL import Image, ImageDraw

    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for x0, y0, x1, y1 in boxes:
        d.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], fill=255)
    return m


def run_zimage(a, rows, plain, out):
    import torch
    from diffusers import ZImageInpaintPipeline
    from PIL import Image, ImageFilter

    pipe = ZImageInpaintPipeline.from_pretrained(a.zmodel, torch_dtype=torch.bfloat16).to("cuda")
    pipe.vae.enable_tiling()
    if a.lora:
        pipe.load_lora_weights(str(Path(a.lora).expanduser()), adapter_name="ko")
        pipe.set_adapters(["ko"], adapter_weights=[a.lora_scale])
    n = 0
    for i, r in enumerate(rows):
        iid = f"{r['id']}_s{a.seed}"
        comp = Image.open(plain / f"{iid}.png").convert("RGB")
        boxes = boxes_for(r, comp.size)
        for p in a.pads:
            m = box_mask(comp.size, boxes, p)
            soft = m.filter(ImageFilter.GaussianBlur(4))
            for s in a.strengths:
                op = out / f"zr{p}-s{s}" / "images" / f"{iid}.png"
                if op.exists():
                    continue
                op.parent.mkdir(parents=True, exist_ok=True)
                h = pipe(prompt=r["prompt_ko"], image=comp, mask_image=m, strength=s, negative_prompt=NEG_TEXT,
                         height=comp.height, width=comp.width, num_inference_steps=a.steps,
                         guidance_scale=a.guidance, generator=torch.Generator("cuda").manual_seed(a.seed)).images[0]
                Image.composite(h.resize(comp.size), comp, soft).save(op)
                n += 1
                torch.cuda.empty_cache()
        print(f"zimage {i + 1}/{len(rows)} {r['id']}", flush=True)
    return n


def run_qwen(a, rows, plain, out):
    import torch
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
    from PIL import Image

    # 20B transformer in bf16 is 41 GB; the box has a 46 GB card and 61 GB RAM. Load it straight to the GPU and
    # keep the weights in fp8 (compute stays bf16) so it and the 17 GB text encoder fit together.
    tr = QwenImageTransformer2DModel.from_pretrained(QWEN, subfolder="transformer", torch_dtype=torch.bfloat16,
                                                     device_map="cuda")
    tr.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    torch.cuda.empty_cache()
    pipe = QwenImageEditPlusPipeline.from_pretrained(QWEN, transformer=tr, torch_dtype=torch.bfloat16)
    if a.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.vae.enable_tiling()
    od = out / "qwen" / "images"
    od.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, r in enumerate(rows):
        iid = f"{r['id']}_s{a.seed}"
        op = od / f"{iid}.png"
        if op.exists():
            continue
        comp = Image.open(plain / f"{iid}.png").convert("RGB")
        with torch.inference_mode():
            h = pipe(image=[comp], prompt=QWEN_INSTR.format(t=r["target_text"]), negative_prompt=" ",
                     true_cfg_scale=4.0, guidance_scale=1.0, num_inference_steps=a.qsteps,
                     generator=torch.manual_seed(a.seed)).images[0]
        h.resize(comp.size).save(op)
        n += 1
        torch.cuda.empty_cache()
        print(f"qwen {i + 1}/{len(rows)} {r['id']}", flush=True)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True, type=Path, help="render_harmonize.py output (plain/ + prompts.jsonl)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mode", choices=["zimage", "qwen"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zmodel", default="Tongyi-MAI/Z-Image")
    ap.add_argument("--lora", default="")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--pads", type=int, nargs="+", default=[24])
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.35, 0.5, 0.65])
    ap.add_argument("--qsteps", type=int, default=40)
    ap.add_argument("--offload", action="store_true", help="qwen: model CPU offload when both models do not fit")
    a = ap.parse_args()

    trial, out = a.trial.expanduser(), a.out.expanduser()
    rows = [json.loads(l) for l in (trial / "prompts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompts.jsonl").write_text((trial / "prompts.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    t0 = time.time()
    n = (run_zimage if a.mode == "zimage" else run_qwen)(a, rows, trial / "plain" / "images", out)
    (out / f"settings-{a.mode}.json").write_text(json.dumps(dict(
        vars(a), trial=str(trial), out=str(out), n_prompts=len(rows), n_images=n, wall_sec=time.time() - t0,
        qwen_model=QWEN, qwen_instruction=QWEN_INSTR, negative=NEG_TEXT), default=str, indent=1,
        ensure_ascii=False))
    print(f"RERENDER {a.mode} DONE {n} images {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
