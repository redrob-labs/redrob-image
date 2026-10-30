"""Plan C stage 1: region Diffusion-DPO on Z-Image (Base) LoRA, from pairs made by scripts/dpo_pairs.py.

Diffusion-DPO (Wallace et al. 2024) in the flow-matching form used by scripts/train_lora.py: the winner and the
loser get the same noise n and the same sigma s, x_t = (1 - s) x0 + s n, and each one's error is the masked mean of
(v - (n - x0))^2 where the mask weights latent cells inside the pair's text region by loss.text_weight (cells
outside count 1, so the scene is still held in place). With e = error under the trained LoRA ("default") and
e_ref = error under the frozen starting LoRA ("ref", the same r64 v2 weights):

    d    = (e_w - e_ref_w) - (e_l - e_ref_l)
    loss = -log sigmoid(-beta * d)  (+ sft_weight * e_w)

d < 0 means the trained LoRA has moved toward the winner more than toward the loser. beta folds Wallace's
0.5 * T factor; it is a config value, not derived.

Held-out check every val.every steps: on the val pairs, at fixed sigmas and per-pair noise, the mean d and the
share of pairs with d < 0 ("pref_acc"). Both start at 0 (the two LoRAs are equal at step 0).

    HF_HOME=~/hf python scripts/train_dpo.py --config configs/g6e-dpo-c1.yaml --run ~/runs/dpo-c1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_lora as T  # noqa: E402


class Pairs:
    def __init__(self, d: Path, split: str):
        from safetensors import safe_open

        self.dir = d
        self.rows = [r for r in (json.loads(ln) for ln in open(d / "index.jsonl", encoding="utf-8"))
                     if r["split"] == split]
        self.h = {}
        self._open = lambda f: self.h.setdefault(f, safe_open(str(d / f), framework="pt"))

    def load(self, r: dict):
        h = self._open(r["file"])
        i = r["i"]
        return h.get_tensor(f"{i}.w"), h.get_tensor(f"{i}.l"), h.get_tensor(f"{i}.mask"), h.get_tensor(f"{i}.txt")


def load_adapter(transformer, path: Path, name: str):
    from diffusers import ZImagePipeline
    from peft import set_peft_model_state_dict

    sd = ZImagePipeline.lora_state_dict(str(path))
    sd = sd[0] if isinstance(sd, tuple) else sd
    sd = {k.removeprefix("transformer."): v for k, v in sd.items() if k.startswith("transformer.")}
    res = set_peft_model_state_dict(transformer, sd, adapter_name=name)
    missing = [k for k in (res.missing_keys if res else []) if "lora" in k and f".{name}." in k]
    if missing:
        raise RuntimeError(f"adapter {name}: {len(missing)} keys missing, e.g. {missing[:3]}")


def pair_err(transformer, x0, noise, sig, txt, w):
    """Masked flow-matching error per image of one pair (x0: (2, C, H, W))."""
    import torch

    xt = (1 - sig) * x0 + sig * noise
    xs = list(xt.to(torch.bfloat16).unsqueeze(2).unbind(0))
    tt = (1 - sig.view(-1)).to(torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = transformer(xs, tt, [txt, txt], return_dict=False)[0]
    pred = -torch.stack(out, 0).squeeze(2).float()
    err = (pred - (noise - x0)) ** 2
    return (err * w).mean(dim=(1, 2, 3)) / w.mean()


def run_val(transformer, pairs: Pairs, ref_name: str, tw: float, beta: float, levels=(0.2, 0.4, 0.6, 0.8)) -> dict:
    import torch

    ds = []
    was = transformer.training
    transformer.eval()
    with torch.no_grad():
        for j, r in enumerate(pairs.rows):
            xw, xl, m, txt = pairs.load(r)
            x0 = torch.stack([xw, xl]).to("cuda", torch.float32)
            w = 1.0 + (tw - 1.0) * m.to("cuda", torch.float32).view(1, 1, *m.shape)
            txt = txt.to("cuda")
            dd = []
            for k, s in enumerate(levels):
                g = torch.Generator("cuda").manual_seed(1000 * j + k)
                n1 = torch.randn((1, *x0.shape[1:]), generator=g, device="cuda")
                noise = n1.expand_as(x0)
                sig = torch.full((2, 1, 1, 1), s, device="cuda")
                e = pair_err(transformer, x0, noise, sig, txt, w)
                transformer.set_adapter(ref_name)
                er = pair_err(transformer, x0, noise, sig, txt, w)
                transformer.set_adapter("default")
                dd.append(((e[0] - er[0]) - (e[1] - er[1])).item())
            ds.append(sum(dd) / len(dd))
    if was:
        transformer.train()
    n = max(len(ds), 1)
    return {"val_d": round(sum(ds) / n, 8), "pref_acc": round(sum(x < 0 for x in ds) / n, 4), "n": len(ds),
            "val_loss": round(sum(float(-torch.nn.functional.logsigmoid(torch.tensor(-beta * x))) for x in ds) / n, 6)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    run = a.run.expanduser()
    run.mkdir(parents=True, exist_ok=True)
    pdir = Path(os.path.expanduser(cfg["pairs"]))
    s3 = cfg.get("s3", "") or ""
    signal.signal(signal.SIGTERM, T.on_signal)
    signal.signal(signal.SIGINT, T.on_signal)

    import torch
    from diffusers import ZImageTransformer2DModel
    from peft import LoraConfig

    torch.backends.cuda.matmul.allow_tf32 = True
    seed = cfg.get("seed", 0)
    random.seed(seed)
    torch.manual_seed(seed)
    model = cfg["model"]
    init = Path(os.path.expanduser(cfg["init_lora"]))

    transformer = ZImageTransformer2DModel.from_pretrained(model, subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda")
    transformer.requires_grad_(False)
    if cfg.get("gradient_checkpointing", True):
        transformer.enable_gradient_checkpointing()
    lc = cfg["lora"]
    mk = lambda: LoraConfig(r=lc["rank"], lora_alpha=lc.get("alpha", lc["rank"]), lora_dropout=0.0,
                            init_lora_weights="gaussian", target_modules=lc["targets"])
    transformer.add_adapter(mk(), adapter_name="ref")
    load_adapter(transformer, init, "ref")
    transformer.add_adapter(mk(), adapter_name="default")
    transformer.set_adapter("default")

    state = {"epoch": 0, "batch": 0, "step": 0, "config": cfg, "commit": T.git_commit(),
             "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    ck = T.latest_ckpt(run, s3)
    load_adapter(transformer, ck if ck else init, "default")
    params = [p for n, p in transformer.named_parameters() if ".default." in n and "lora_" in n]
    for n, p in transformer.named_parameters():
        p.requires_grad_(".default." in n and "lora_" in n)
    for p in params:
        p.data = p.data.float()
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=cfg["optim"]["lr"], betas=(0.9, 0.999), weight_decay=cfg["optim"].get("weight_decay", 0.0))
    if ck:
        t = torch.load(ck / "trainer.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(t["opt"])
        state.update({k: t["state"][k] for k in ("epoch", "batch", "step") if k in t["state"]})
        random.setstate(t["rng"]["py"])
        torch.set_rng_state(t["rng"]["torch"])
        torch.cuda.set_rng_state(t["rng"]["cuda"])
        print(f"resumed from {ck} at step {state['step']}", flush=True)
    state["lora_params"] = n_params
    (run / "config.json").write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    train, val = Pairs(pdir, "train"), Pairs(pdir, "val")
    print(f"pairs: {len(train.rows)} train, {len(val.rows)} val; LoRA params {n_params}", flush=True)
    beta = float(cfg["dpo"]["beta"])
    sft_w = float(cfg["dpo"].get("sft_weight", 0.0))
    tw = float((cfg.get("loss") or {}).get("text_weight", 1.0))
    ts = cfg.get("timesteps") or {}
    accum = int(cfg["optim"].get("accum", 1))
    warmup = cfg["optim"].get("warmup", 20)
    base_lr = cfg["optim"]["lr"]
    vc = cfg.get("val") or {}
    log = open(run / "log.jsonl", "a", encoding="utf-8")

    def log_val():
        rec = {"step": state["step"]} | run_val(transformer, val, "ref", tw, beta)
        with open(run / "val.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print("val " + json.dumps(rec), flush=True)

    if val.rows and vc.get("every") and state["step"] == 0:
        log_val()
    transformer.train()
    last_ckpt, t0, seen = time.time(), time.time(), 0
    rng = random.Random(seed + state["step"])
    acc_hist, d_hist, l_hist = [], [], []
    for ep in range(state["epoch"], cfg.get("epochs", 1)):
        order = list(range(len(train.rows)))
        random.Random(seed + ep).shuffle(order)
        start = state["batch"] if state["epoch"] == ep else 0
        for bi in range(start, len(order), accum):
            for r in (train.rows[k] for k in order[bi: bi + accum]):
                xw, xl, m, txt = train.load(r)
                x0 = torch.stack([xw, xl]).to("cuda", torch.float32)
                w = 1.0 + (tw - 1.0) * m.to("cuda", torch.float32).view(1, 1, *m.shape)
                txt = txt.to("cuda")
                noise = torch.randn((1, *x0.shape[1:]), device="cuda").expand_as(x0).contiguous()
                if ts.get("dist") == "logit_normal":
                    u = torch.randn(1, device="cuda") * ts.get("std", 1.0) + ts.get("mean", 0.0)
                    s = torch.sigmoid(u).clamp(1e-3, 1 - 1e-3)
                else:
                    s = torch.rand(1, device="cuda").clamp(1e-3, 1 - 1e-3)
                sig = s.view(1, 1, 1, 1).expand(2, 1, 1, 1)
                with torch.no_grad():
                    transformer.set_adapter("ref")
                    er = pair_err(transformer, x0, noise, sig, txt, w)
                    transformer.set_adapter("default")
                for n_, p in transformer.named_parameters():  # set_adapter toggles requires_grad
                    if "lora_" in n_:
                        p.requires_grad_(".default." in n_)
                e = pair_err(transformer, x0, noise, sig, txt, w)
                d = (e[0] - er[0]) - (e[1] - er[1])
                loss = -torch.nn.functional.logsigmoid(-beta * d) + sft_w * e[0]
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {state['step']}")
                (loss / accum).backward()
                acc_hist.append(float(d.item() < 0))
                d_hist.append(d.item())
                l_hist.append(loss.item())
                seen += 1
            lr = base_lr * min(1.0, (state["step"] + 1) / max(warmup, 1))
            for g in opt.param_groups:
                g["lr"] = lr
            gn = torch.nn.utils.clip_grad_norm_(params, cfg["optim"].get("max_grad_norm", 1.0))
            opt.step()
            opt.zero_grad(set_to_none=True)
            state.update(epoch=ep, batch=bi + accum, step=state["step"] + 1)
            if state["step"] % 10 == 0:
                k = 10 * accum
                rec = {"step": state["step"], "epoch": ep, "loss": round(sum(l_hist[-k:]) / len(l_hist[-k:]), 5),
                       "d": round(sum(d_hist[-k:]) / len(d_hist[-k:]), 8),
                       "train_pref_acc": round(sum(acc_hist[-k:]) / len(acc_hist[-k:]), 3),
                       "grad_norm": round(float(gn), 4), "lr": lr, "pairs_per_sec": round(seen / (time.time() - t0), 3),
                       "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
            done = a.max_steps and state["step"] >= a.max_steps
            if T.STOP["flag"] or time.time() - last_ckpt > cfg.get("ckpt_minutes", 30) * 60 or done:
                T.save_ckpt(run, f"step-{state['step']}", transformer, opt, dict(state), s3)
                last_ckpt = time.time()
                if T.STOP["flag"]:
                    print("stopped by signal; re-run the same command to resume", flush=True)
                    return 0
            if val.rows and vc.get("every") and state["step"] % vc["every"] == 0:
                log_val()
            if done:
                break
        if a.max_steps and state["step"] >= a.max_steps:
            break
        state.update(epoch=ep + 1, batch=0)

    if not (run / "ckpt" / f"step-{state['step']}").exists():
        T.save_ckpt(run, f"step-{state['step']}", transformer, opt, dict(state), s3)
    final = run / "final"
    final.mkdir(exist_ok=True)
    for f in (run / "ckpt" / f"step-{state['step']}").iterdir():
        if f.suffix == ".safetensors":
            (final / f.name).write_bytes(f.read_bytes())
    if val.rows:
        log_val()
    el = time.time() - t0
    summary = {"steps": state["step"], "pairs_seen": seen, "seconds": round(el, 1),
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "lora_params": n_params,
               "gpu": torch.cuda.get_device_name(0), "commit": state["commit"]}
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    print("DPO TRAIN DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
