"""Plan C stage 1: preference pairs for region DPO from a rollout dir (scripts/run_c1_rollout_box.sh).

Per prompt the rollout has 4 images, each with a classifier reward (rewards_cls.jsonl, scripts/reward_cls.py) and
PaddleOCR line boxes (scores.jsonl). The pair is the best and the worst image of the prompt, kept only when the
reward gap is at least --min-gap. Both images share one latent-grid mask: every cell any detected line of either
image touches, so extra text the worse image drew is inside the region too.

Writes <out>/pairs-NNNN.safetensors with <i>.w, <i>.l (VAE latents, same scaling as precompute.py), <i>.mask
(uint8 latent grid) and <i>.txt (the prompt's caption embedding, the same text the rollout was drawn from), plus
<out>/index.jsonl. One prompt in --val-every goes to the held-out split (split "val").

    HF_HOME=~/hf python scripts/dpo_pairs.py --rollout ~/runs/c1-rollout --out ~/data/dpo-c1
"""
from __future__ import annotations

import argparse
import json
import zlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "scripts"))


def pick(rewards: list[dict], min_gap: float, per_prompt: int) -> list[tuple[dict, dict]]:
    by = defaultdict(list)
    for r in rewards:
        if r.get("reward") is not None:
            by[r["id"]].append(r)
    out = []
    for pid in sorted(by):
        rs = sorted(by[pid], key=lambda r: r["reward"], reverse=True)
        if len(rs) < 2:
            continue
        # best vs worst first, then best vs second worst, ... up to per_prompt pairs
        for lo in rs[::-1][:per_prompt]:
            if lo is rs[0] or rs[0]["reward"] - lo["reward"] < min_gap:
                break
            out.append((rs[0], lo))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Tongyi-MAI/Z-Image")
    ap.add_argument("--min-gap", type=float, default=2.0, help="reward units (sum of log-probs, minus extra cells)")
    ap.add_argument("--per-prompt", type=int, default=1)
    ap.add_argument("--val-every", type=int, default=20, help="1 prompt in N is held out (by id hash)")
    ap.add_argument("--shard", type=int, default=64)
    a = ap.parse_args()

    import torch
    from PIL import Image
    from safetensors.torch import save_file

    from precompute import Encoders, text_mask

    d = a.rollout.expanduser()
    rewards = [json.loads(ln) for ln in open(d / "rewards_cls.jsonl", encoding="utf-8")]
    scores = {}
    for ln in open(d / "scores.jsonl", encoding="utf-8"):
        s = json.loads(ln)
        scores[s["image_id"]] = s
    pairs = pick(rewards, a.min_gap, a.per_prompt)
    n_prompts = len({r["id"] for r in rewards})
    print(f"{len(rewards)} images, {n_prompts} prompts -> {len(pairs)} pairs (min gap {a.min_gap})", flush=True)

    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    enc = Encoders(a.model)
    index = open(out / "index.jsonl", "w", encoding="utf-8")
    for k in range(0, len(pairs), a.shard):
        chunk = pairs[k: k + a.shard]
        name = f"pairs-{k // a.shard:04d}.safetensors"
        tensors = {}
        for i, (w, l) in enumerate(chunk):
            sw, sl = scores[w["image_id"]], scores[l["image_id"]]
            iw = Image.open(d / "images" / f"{w['image_id']}.png").convert("RGB")
            il = Image.open(d / "images" / f"{l['image_id']}.png").convert("RGB")
            if iw.size != il.size or iw.size[0] % 16 or iw.size[1] % 16:
                raise SystemExit(f"pair {w['image_id']} / {l['image_id']}: sizes {iw.size} {il.size}")
            W, H = iw.size
            lat = enc.latents([iw, il])
            boxes = [tuple(x["box"]) for x in (sw.get("lines") or []) + (sl.get("lines") or []) if x.get("box")]
            tensors[f"{i}.w"] = lat[0].contiguous()
            tensors[f"{i}.l"] = lat[1].contiguous()
            tensors[f"{i}.mask"] = text_mask(boxes, W, H).contiguous()
            tensors[f"{i}.txt"] = enc.text([sw["prompt_ko"]])[0].contiguous()
            split = "val" if zlib.crc32(w["id"].encode()) % a.val_every == 0 else "train"
            index.write(json.dumps({"file": name, "i": i, "id": w["id"], "w": w["image_id"], "l": l["image_id"],
                                    "rw": round(w["reward"], 4), "rl": round(l["reward"], 4),
                                    "gap": round(w["reward"] - l["reward"], 4), "target_text": w["target_text"],
                                    "split": split, "size": [W, H]}, ensure_ascii=False) + "\n")
        tmp = out / (name + ".part")
        save_file(tensors, str(tmp))
        tmp.rename(out / name)
        index.flush()
        print(f"{name}: {len(chunk)} pairs", flush=True)
    save_file({"txt": enc.text([""])[0].contiguous()}, str(out / "empty.safetensors"))
    print("DPO PAIRS DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
