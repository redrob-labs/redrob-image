#!/usr/bin/env python3
"""T8: pilot LoRA on Z-Image (Base) from pre-computed latents and caption embeddings (scripts/precompute.py).

Flow matching exactly as diffusers' train_dreambooth_lora_z_image.py: x_t = (1 - s) x0 + s n, the DiT gets
t = (1000 - timestep) / 1000, its output is negated and regressed onto n - x0. Only LoRA weights train
(fp32 master copy, bf16 autocast); VAE and text encoder are never loaded except for the in-training eval.

Stages run in order (e.g. 512 px for N epochs, then 1024 px for a few steps); each stage reads
<precomp>/<res>/index.jsonl. Batches hold one aspect bucket each.

Robustness: a checkpoint (LoRA + optimizer + position) is written every --ckpt-minutes and on SIGTERM / SIGINT
(spot interruption notice), optionally synced to S3; a re-run with the same --run resumes from the newest one.

In-training eval: every eval.every steps the current LoRA generates the eval subset into
<run>/eval/<step>/images/<id>_s<seed>.png. Score them afterwards (GPU free) with
    python scripts/eval_baseline.py --model Tongyi-MAI/Z-Image --steps 28 --guidance 4 --subset 40 --out <run>/eval/<step>
which skips generation because the images exist.

Usage:
    HF_HOME=~/hf python scripts/train_lora.py --config configs/g6e.yaml --run ~/runs/pilot-r32
    python scripts/train_lora.py --config configs/g6e.yaml --run ~/runs/smoke --max-steps 100 --no-eval
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

STOP = {"flag": False}


def on_signal(signum, _frame):
    STOP["flag"] = True
    print(f"signal {signum}: checkpoint at the next step boundary", flush=True)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


class Data:
    """Batches of one bucket from <precomp>/<res>/index.jsonl, deterministic per (seed, epoch)."""

    def __init__(self, precomp: Path, res: int, mix: dict, batch: int, seed: int, variants: list[str]):
        self.dir = precomp / str(res)
        rows = [json.loads(ln) for ln in open(self.dir / "index.jsonl", encoding="utf-8")]
        rng = random.Random(seed)
        by_src = defaultdict(list)
        for r in rows:
            by_src[r["source"]].append(r)
        self.rows = []
        for src, n in mix.items():
            pool = sorted(by_src.get(src, []), key=lambda r: (r["file"], r["i"]))
            rng.shuffle(pool)
            if n and len(pool) < n:
                print(f"warning: {src} has {len(pool)} < {n} samples at {res}px", flush=True)
            self.rows += pool[: n or None]
        self.batch, self.seed, self.variants = batch, seed, variants
        self.handles: OrderedDict = OrderedDict()

    def epoch_batches(self, epoch: int) -> list[list[dict]]:
        rng = random.Random(self.seed * 1000 + epoch)
        by_b = defaultdict(list)
        for r in self.rows:
            by_b[tuple(r["bucket"])].append(r)
        out = []
        for _, rs in sorted(by_b.items()):
            rng.shuffle(rs)
            out += [rs[i: i + self.batch] for i in range(0, len(rs) - self.batch + 1, self.batch)]
        rng.shuffle(out)
        return out

    def _open(self, name):
        from safetensors import safe_open

        if name not in self.handles:
            if len(self.handles) > 64:
                self.handles.popitem(last=False)
            self.handles[name] = safe_open(str(self.dir / name), framework="pt")
        self.handles.move_to_end(name)
        return self.handles[name]

    def load(self, batch: list[dict], rng: random.Random):
        lats, txts = [], []
        for r in batch:
            h = self._open(r["file"])
            lats.append(h.get_tensor(f"{r['i']}.lat"))
            vs = [v for v in self.variants if v in r["variants"]] or r["variants"]
            txts.append(h.get_tensor(f"{r['i']}.txt.{rng.choice(vs)}"))
        return lats, txts


def lora_targets(cfg: dict) -> list[str]:
    return cfg.get("targets", ["to_q", "to_k", "to_v", "to_out.0"])


class StableVal:
    """Deterministic validation loss on a held-out pre-computed set (SD3, Esser et al. 2024, App. B.3): every
    sample is scored at fixed, equally spaced noise levels t = (k + 0.5) / levels with noise from a per-sample
    seed, so the number moves only when the weights do. Logged overall, per source and per level."""

    def __init__(self, cfg: dict):
        from safetensors import safe_open

        self.cfg = cfg
        d = Path(os.path.expanduser(cfg["precomp"])) / str(cfg.get("res", 512))
        rows = [json.loads(ln) for ln in open(d / "index.jsonl", encoding="utf-8")]
        k = cfg.get("max_per_source", 0)  # a forward pass per sample x level: 512 samples took ~12 min on an L40S
        by = defaultdict(list)
        for r in sorted(rows, key=lambda r: (r["source"], r["file"], r["i"])):
            by[r["source"]].append(r)
        self.rows = [r for s in sorted(by) for r in by[s][: k or None]]
        self.dir, self.h = d, {}
        self.levels = cfg.get("levels", 8)
        self.variant = cfg.get("variant", "ko_long")
        self._open = lambda f: self.h.setdefault(f, safe_open(str(d / f), framework="pt"))

    def __call__(self, transformer) -> dict:
        import torch

        L = self.levels
        sig = torch.tensor([(k + 0.5) / L for k in range(L)], device="cuda").view(-1, 1, 1, 1)
        per_src, per_lvl, tot = defaultdict(list), torch.zeros(L), []
        was_training = transformer.training
        transformer.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for j, r in enumerate(self.rows):
                h = self._open(r["file"])
                x0 = h.get_tensor(f"{r['i']}.lat").to("cuda", torch.float32).unsqueeze(0).expand(L, -1, -1, -1)
                v = self.variant if self.variant in r["variants"] else r["variants"][0]
                txt = h.get_tensor(f"{r['i']}.txt.{v}").to("cuda")
                g = torch.Generator("cuda").manual_seed(self.cfg.get("seed", 1234) + j)
                noise = torch.randn(x0.shape, generator=g, device="cuda")
                xt = (1 - sig) * x0 + sig * noise
                out = transformer(list(xt.to(torch.bfloat16).unsqueeze(2).unbind(0)), (1 - sig.view(-1)).to(torch.bfloat16),
                                  [txt] * L, return_dict=False)[0]
                pred = -torch.stack(out, 0).squeeze(2).float()
                lv = ((pred - (noise - x0)) ** 2).mean(dim=(1, 2, 3)).cpu()
                per_lvl += lv
                per_src[r["source"]].append(lv.mean().item())
                tot.append(lv.mean().item())
        if was_training:
            transformer.train()
        n = max(len(tot), 1)
        return {"val_loss": round(sum(tot) / n, 6), "n": len(tot),
                "by_source": {s: round(sum(v) / len(v), 6) for s, v in sorted(per_src.items())},
                "by_level": [round(x, 6) for x in (per_lvl / n).tolist()]}


def log_val(run: Path, sv: StableVal, transformer, step: int):
    t0 = time.time()
    base_f = run / f"val_base-n{len(sv.rows)}.json"  # one per held-out set size, so a resized set gets its own base
    if not base_f.exists():
        with _adapters_off(transformer):
            base = sv(transformer)
        base_f.write_text(json.dumps(base, indent=2), encoding="utf-8")
        print(f"val base (no LoRA): {base['val_loss']}", flush=True)
    rec = {"step": step} | sv(transformer) | {"seconds": round(time.time() - t0, 1)}
    with open(run / "val.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print("val " + json.dumps(rec), flush=True)


class _adapters_off:
    def __init__(self, m):
        self.m = m

    def __enter__(self):
        self.m.disable_adapters()

    def __exit__(self, *exc):
        self.m.enable_adapters()


def save_ckpt(run: Path, tag: str, transformer, opt, state: dict, s3: str, keep: int = 2):
    import torch
    from diffusers import ZImagePipeline
    from peft.utils import get_peft_model_state_dict

    d = run / "ckpt" / tag
    tmp = d.with_name(d.name + ".part")
    tmp.mkdir(parents=True, exist_ok=True)
    lora = {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in get_peft_model_state_dict(transformer).items()}
    ZImagePipeline.save_lora_weights(str(tmp), transformer_lora_layers=lora)
    torch.save({"opt": opt.state_dict(), "state": state, "rng": {"py": random.getstate(), "torch": torch.get_rng_state(),
                                                               "cuda": torch.cuda.get_rng_state()}}, tmp / "trainer.pt")
    (tmp / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    if d.exists():
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    tmp.rename(d)
    olds = sorted((p for p in (run / "ckpt").iterdir() if p.name.startswith("step-") and not p.name.endswith(".part")),
                  key=lambda p: int(p.name.split("-")[1]))
    for p in olds[:-keep]:
        for f in p.iterdir():
            f.unlink()
        p.rmdir()
    if s3:
        subprocess.run(["aws", "s3", "sync", str(run / "ckpt"), f"{s3.rstrip('/')}/ckpt", "--delete",
                        "--region", "ap-northeast-2", "--only-show-errors"], check=False)
    print(f"checkpoint {d}", flush=True)


def latest_ckpt(run: Path, s3: str) -> Path | None:
    if s3 and not (run / "ckpt").exists():
        subprocess.run(["aws", "s3", "sync", f"{s3.rstrip('/')}/ckpt", str(run / "ckpt"), "--region", "ap-northeast-2",
                        "--only-show-errors"], check=False)
    c = run / "ckpt"
    if not c.exists():
        return None
    steps = [p for p in c.iterdir() if p.name.startswith("step-") and (p / "trainer.pt").exists()]
    return max(steps, key=lambda p: int(p.name.split("-")[1])) if steps else None


def load_lora_into(transformer, path: Path):
    from diffusers import ZImagePipeline
    from peft import set_peft_model_state_dict

    sd = ZImagePipeline.lora_state_dict(str(path))
    sd = sd[0] if isinstance(sd, tuple) else sd
    sd = {k.removeprefix("transformer."): v for k, v in sd.items() if k.startswith("transformer.")}
    res = set_peft_model_state_dict(transformer, sd, adapter_name="default")
    missing = [k for k in (res.missing_keys if res else []) if "lora" in k]
    if missing:
        raise RuntimeError(f"LoRA load missed {len(missing)} keys, e.g. {missing[:3]}")


def run_eval(cfg: dict, transformer, out_dir: Path, model: str):
    import torch
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline
    from transformers import AutoModel, AutoTokenizer

    from eval_baseline import load_prompts, stratified

    rows = load_prompts([str(ROOT / p) for p in cfg.get("prompts", ["eval/prompts.jsonl"])])
    rows = stratified(rows, cfg.get("subset", 40))
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    transformer.eval()
    te = AutoModel.from_pretrained(model, subfolder="text_encoder", torch_dtype=torch.bfloat16).to("cuda")
    pipe = ZImagePipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(model, subfolder="scheduler"),
                          vae=AutoencoderKL.from_pretrained(model, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda"),
                          text_encoder=te, tokenizer=AutoTokenizer.from_pretrained(model, subfolder="tokenizer"),
                          transformer=transformer)
    res = cfg.get("resolution", 1024)
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for r in rows:
            for s in cfg.get("seeds", [42]):
                p = img_dir / f"{r['id']}_s{s}.png"
                if p.exists():
                    continue
                img = pipe(prompt=r["prompt_ko"], height=res, width=res, num_inference_steps=cfg.get("steps", 28),
                           guidance_scale=cfg.get("guidance", 4.0), negative_prompt="",
                           generator=torch.Generator("cuda").manual_seed(s)).images[0]
                img.save(p)
    (out_dir / "eval_meta.json").write_text(json.dumps({"model": model, "lora": "in-training", "resolution": res,
                                                        "steps": cfg.get("steps", 28), "guidance": cfg.get("guidance", 4.0),
                                                        "prompts": len(rows), "seconds": round(time.time() - t0, 1)}), encoding="utf-8")
    del pipe, te
    torch.cuda.empty_cache()
    transformer.train()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps in total (smoke test)")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--rank", type=int, default=0, help="override lora.rank")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    if a.rank:
        cfg["lora"]["rank"] = cfg["lora"]["alpha"] = a.rank
    run = a.run.expanduser()
    run.mkdir(parents=True, exist_ok=True)
    precomp = Path(os.path.expanduser(cfg["precomp"]))
    s3 = cfg.get("s3", "") or ""
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, ZImageTransformer2DModel
    from peft import LoraConfig
    from safetensors.torch import load_file

    torch.backends.cuda.matmul.allow_tf32 = True
    model = cfg["model"]
    seed = cfg.get("seed", 0)
    random.seed(seed)
    torch.manual_seed(seed)

    transformer = ZImageTransformer2DModel.from_pretrained(model, subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda")
    transformer.requires_grad_(False)
    if cfg.get("gradient_checkpointing", True):
        transformer.enable_gradient_checkpointing()
    lc = cfg["lora"]
    transformer.add_adapter(LoraConfig(r=lc["rank"], lora_alpha=lc.get("alpha", lc["rank"]), lora_dropout=0.0,
                                       init_lora_weights="gaussian", target_modules=lora_targets(lc)))
    params = [p for p in transformer.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=cfg["optim"]["lr"], betas=(0.9, 0.999), weight_decay=cfg["optim"].get("weight_decay", 0.0))
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(model, subfolder="scheduler")
    sigmas_all = sched.sigmas.to("cuda", torch.float32)
    n_train = sched.config.num_train_timesteps
    empty = load_file(str(precomp / "empty.safetensors"))["txt"].to("cuda")

    state = {"stage": 0, "epoch": 0, "batch": 0, "step": 0, "config": cfg, "commit": git_commit(),
             "lora_params": n_params, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    ck = latest_ckpt(run, s3)
    if ck:
        load_lora_into(transformer, ck)
        for p in params:
            p.data = p.data.float()
        t = torch.load(ck / "trainer.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(t["opt"])
        state.update({k: t["state"][k] for k in ("stage", "epoch", "batch", "step", "stage_start") if k in t["state"]})
        random.setstate(t["rng"]["py"])
        torch.set_rng_state(t["rng"]["torch"])
        torch.cuda.set_rng_state(t["rng"]["cuda"])
        print(f"resumed from {ck} at step {state['step']}", flush=True)
    elif cfg.get("init_lora"):
        load_lora_into(transformer, Path(os.path.expanduser(cfg["init_lora"])))
        for p in params:
            p.data = p.data.float()
    (run / "config.json").write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    log = open(run / "log.jsonl", "a", encoding="utf-8")
    last_ckpt = time.time()
    t_start, samples_seen, tokens_seen = time.time(), 0, 0
    ev = cfg.get("eval", {})
    vc = cfg.get("val") or {}
    sv = None
    if vc.get("every") and (Path(os.path.expanduser(vc["precomp"])) / str(vc.get("res", 512)) / "index.jsonl").exists():
        sv = StableVal(vc)
        print(f"stable val: {len(sv.rows)} held-out samples x {sv.levels} levels every {vc['every']} steps", flush=True)
        log_val(run, sv, transformer, state["step"])
    transformer.train()
    rng = random.Random(seed + state["step"])
    warmup = cfg["optim"].get("warmup", 100)
    base_lr = cfg["optim"]["lr"]

    for si in range(state["stage"], len(cfg["stages"])):
        st = cfg["stages"][si]
        data = Data(precomp, st["res"], cfg["mix"], st["batch"], seed + si, cfg.get("variants", ["ko_long", "ko_short", "en"]))
        epochs = st.get("epochs", 1)
        stage_steps_cap = st.get("steps", 0)
        if state["stage"] != si or (state["epoch"] == 0 and state["batch"] == 0 and "stage_start" not in state):
            state["stage_start"] = state["step"]
        stage_start_step = state["stage_start"]
        print(f"stage {si}: {st['res']}px, {len(data.rows)} samples, batch {st['batch']}", flush=True)
        for ep in range(state["epoch"] if state["stage"] == si else 0, epochs):
            batches = data.epoch_batches(ep)
            start_b = state["batch"] if (state["stage"] == si and state["epoch"] == ep) else 0
            for bi in range(start_b, len(batches)):
                lats, txts = data.load(batches[bi], rng)
                x0 = torch.stack(lats).to("cuda", torch.float32)
                prompts = [(empty if rng.random() < cfg.get("caption_dropout", 0.1) else t.to("cuda")) for t in txts]
                noise = torch.randn_like(x0)
                idx = torch.randint(0, n_train, (x0.shape[0],))
                sig = sigmas_all[idx.to("cuda")].view(-1, 1, 1, 1)
                xt = (1 - sig) * x0 + sig * noise
                tnorm = (1 - sig.view(-1))  # == (1000 - timestep) / 1000 for this scheduler
                lr = base_lr * min(1.0, (state["step"] + 1) / max(warmup, 1))
                for g in opt.param_groups:
                    g["lr"] = lr
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = transformer(list(xt.to(torch.bfloat16).unsqueeze(2).unbind(0)), tnorm.to(torch.bfloat16),
                                      prompts, return_dict=False)[0]
                pred = -torch.stack(out, 0).squeeze(2).float()
                loss = ((pred - (noise - x0)) ** 2).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {state['step']}")
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, cfg["optim"].get("max_grad_norm", 1.0))
                opt.step()
                opt.zero_grad(set_to_none=True)
                state.update(stage=si, epoch=ep, batch=bi + 1, step=state["step"] + 1)
                samples_seen += x0.shape[0]
                tokens_seen += x0.shape[0] * (x0.shape[2] // 2) * (x0.shape[3] // 2)
                if state["step"] % 10 == 0:
                    el = time.time() - t_start
                    rec = {"step": state["step"], "stage": si, "epoch": ep, "loss": round(loss.item(), 5),
                           "grad_norm": round(float(gn), 4), "lr": lr, "res": st["res"],
                           "samples_per_sec": round(samples_seen / el, 3), "elapsed_sec": round(el, 1),
                           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                           "steps_this_epoch": len(batches)}
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    print(json.dumps(rec), flush=True)
                done_cap = (a.max_steps and state["step"] >= a.max_steps) or \
                           (stage_steps_cap and state["step"] - stage_start_step >= stage_steps_cap)
                if STOP["flag"] or time.time() - last_ckpt > cfg.get("ckpt_minutes", 30) * 60 or done_cap:
                    save_ckpt(run, f"step-{state['step']}", transformer, opt, dict(state), s3)
                    last_ckpt = time.time()
                    if STOP["flag"]:
                        print("stopped by signal; re-run the same command to resume", flush=True)
                        return 0
                if sv and state["step"] % vc["every"] == 0:
                    log_val(run, sv, transformer, state["step"])
                if not a.no_eval and ev.get("every") and state["step"] % ev["every"] == 0:
                    run_eval(ev, transformer, run / "eval" / f"step-{state['step']}", model)
                if done_cap:
                    break
            if (a.max_steps and state["step"] >= a.max_steps) or (stage_steps_cap and state["step"] - stage_start_step >= stage_steps_cap):
                break
            state.update(epoch=ep + 1, batch=0)
        if a.max_steps and state["step"] >= a.max_steps:
            break  # stay inside this stage so a later run with a larger --max-steps continues it
        state.update(stage=si + 1, epoch=0, batch=0)
        state.pop("stage_start", None)


    if not (run / "ckpt" / f"step-{state['step']}").exists():
        save_ckpt(run, f"step-{state['step']}", transformer, opt, dict(state), s3)
    final = run / "final"
    final.mkdir(exist_ok=True)
    src = run / "ckpt" / f"step-{state['step']}"
    for f in src.iterdir():
        if f.suffix == ".safetensors":
            (final / f.name).write_bytes(f.read_bytes())
    el = time.time() - t_start
    summary = {"steps": state["step"], "seconds": round(el, 1), "samples": samples_seen,
               "samples_per_sec": round(samples_seen / max(el, 1e-6), 3), "tokens": tokens_seen,
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "lora_params": n_params,
               "gpu": torch.cuda.get_device_name(0), "commit": state["commit"]}
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    if sv:
        log_val(run, sv, transformer, state["step"])
    if not a.no_eval and ev.get("every"):
        run_eval(ev, transformer, run / "eval" / f"step-{state['step']}", model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
