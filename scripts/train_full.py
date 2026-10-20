#!/usr/bin/env python3
"""Path B: full fine-tuning of the Z-Image DiT with FSDP2 (torch.distributed.fsdp.fully_shard), torchrun launch.

Same data (scripts/precompute.py), loss and stage schedule as scripts/train_lora.py; differences:
  - every DiT parameter trains (fp32 sharded master weights, bf16 compute, fp32 gradient reduce);
  - one fully_shard per transformer block, then the root; activation checkpointing per block;
  - each rank takes every world_size-th batch of the shared deterministic order;
  - checkpoints use torch.distributed.checkpoint (sharded, resumable on the same world size),
    written every --ckpt-minutes and on SIGTERM, then synced to S3 by rank 0;
  - an EMA copy is not kept (memory); export picks the last or best checkpoint.
`--trainable-blocks N` limits training to the last N blocks: only for a single-GPU dry run of this code path.

    torchrun --nproc-per-node 8 scripts/train_full.py --config configs/p5.yaml --run /opt/runs/pathb
    torchrun --nproc-per-node 1 scripts/train_full.py --config configs/smoke.yaml --run ~/runs/full-dry \
        --max-steps 20 --trainable-blocks 2
Export a diffusers transformer folder from a checkpoint:
    python scripts/train_full.py --config configs/p5.yaml --run /opt/runs/pathb --export /opt/runs/pathb/export
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_lora import Data, git_commit  # noqa: E402

STOP = {"flag": False}


def on_signal(signum, _frame):
    STOP["flag"] = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--trainable-blocks", type=int, default=0)
    ap.add_argument("--export", default="", help="write a diffusers transformer folder from the newest checkpoint and exit")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    run = a.run.expanduser()
    precomp = Path(os.path.expanduser(cfg["precomp"]))
    s3 = cfg.get("s3", "") or ""
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from diffusers import FlowMatchEulerDiscreteScheduler, ZImageTransformer2DModel
    from safetensors.torch import load_file
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    main_rank = rank == 0
    if main_rank:
        run.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    def say(*x):
        if main_rank:
            print(*x, flush=True)

    model = cfg["model"]
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed + rank)
    tr = ZImageTransformer2DModel.from_pretrained(model, subfolder="transformer", torch_dtype=torch.float32)
    blocks = None
    for name in ("layers", "transformer_blocks", "blocks"):
        if hasattr(tr, name):
            blocks = getattr(tr, name)
            break
    if blocks is None:
        raise RuntimeError("cannot find the transformer block list")
    tr.requires_grad_(a.trainable_blocks == 0)
    if a.trainable_blocks:
        for b in list(blocks)[-a.trainable_blocks:]:
            b.requires_grad_(True)
    if cfg.get("gradient_checkpointing", True):
        tr.enable_gradient_checkpointing()
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for b in blocks:
        fully_shard(b, mp_policy=mp)
    fully_shard(tr, mp_policy=mp)
    tr.to(dev)
    params = [p for p in tr.parameters() if p.requires_grad]
    o = cfg["optim"]
    opt = torch.optim.AdamW(params, lr=o["lr"], betas=(0.9, o.get("beta2", 0.999)), weight_decay=o.get("weight_decay", 0.0))
    say(f"world {world}, trainable params {sum(p.numel() for p in params) * (world if a.trainable_blocks == 0 else 1):,}")

    ckdir = run / "ckpt"
    if main_rank and s3 and not ckdir.exists():
        subprocess.run(["aws", "s3", "sync", f"{s3.rstrip('/')}/ckpt", str(ckdir), "--region", "ap-northeast-2",
                        "--only-show-errors"], check=False)
    dist.barrier()
    state = {"stage": 0, "epoch": 0, "batch": 0, "step": 0}
    done = sorted((p for p in ckdir.glob("step-*") if (p / "state.json").exists()), key=lambda p: int(p.name.split("-")[1])) \
        if ckdir.exists() else []
    if done:
        msd, osd = get_state_dict(tr, opt)
        sd = {"model": msd, "optim": osd}
        dcp.load(sd, checkpoint_id=str(done[-1]))
        set_state_dict(tr, opt, model_state_dict=sd["model"], optim_state_dict=sd["optim"])
        state.update(json.loads((done[-1] / "state.json").read_text()))
        say(f"resumed {done[-1]} at step {state['step']}")

    if a.export:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        full = get_model_state_dict(tr, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if main_rank:
            out = ZImageTransformer2DModel.from_config(ZImageTransformer2DModel.load_config(model, subfolder="transformer"))
            out.load_state_dict(full)
            out.to(torch.bfloat16).save_pretrained(a.export)
            say(f"exported {a.export}")
        dist.destroy_process_group()
        return 0

    def save(tag):
        d = ckdir / tag
        msd, osd = get_state_dict(tr, opt)
        dcp.save({"model": msd, "optim": osd}, checkpoint_id=str(d))
        dist.barrier()
        if main_rank:
            (d / "state.json").write_text(json.dumps(state), encoding="utf-8")
            olds = sorted((p for p in ckdir.glob("step-*") if (p / "state.json").exists()), key=lambda p: int(p.name.split("-")[1]))
            for p in olds[:-2]:
                subprocess.run(["rm", "-r", str(p)], check=False)
            if s3:
                subprocess.run(["aws", "s3", "sync", str(ckdir), f"{s3.rstrip('/')}/ckpt", "--delete",
                                "--region", "ap-northeast-2", "--only-show-errors"], check=False)
            print(f"checkpoint {d}", flush=True)
        dist.barrier()

    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(model, subfolder="scheduler")
    sig_all = sched.sigmas.to(dev, torch.float32)
    n_train = sched.config.num_train_timesteps
    empty = load_file(str(precomp / "empty.safetensors"))["txt"].to(dev)
    log = open(run / "log.jsonl", "a", encoding="utf-8") if main_rank else None
    rng = random.Random(seed * 100 + rank + state["step"])
    last, t0, seen = time.time(), time.time(), 0
    warm = o.get("warmup", 500)
    tr.train()
    for si in range(state["stage"], len(cfg["stages"])):
        st = cfg["stages"][si]
        data = Data(precomp, st["res"], cfg["mix"], st["batch"], seed + si, cfg.get("variants", ["ko_long", "ko_short", "en"]))
        cap = st.get("steps", 0)
        if state["stage"] != si or "stage_start" not in state:
            state["stage_start"] = state["step"]
        for ep in range(state["epoch"] if state["stage"] == si else 0, st.get("epochs", 1)):
            batches = data.epoch_batches(ep)
            n_local = len(batches) // world
            start = state["batch"] if (state["stage"] == si and state["epoch"] == ep) else 0
            for bi in range(start, n_local):
                lats, txts = data.load(batches[bi * world + rank], rng)
                x0 = torch.stack(lats).to(dev, torch.float32)
                prompts = [(empty if rng.random() < cfg.get("caption_dropout", 0.1) else t.to(dev)) for t in txts]
                noise = torch.randn_like(x0)
                sig = sig_all[torch.randint(0, n_train, (x0.shape[0],), device=dev)].view(-1, 1, 1, 1)
                xt = (1 - sig) * x0 + sig * noise
                lr = o["lr"] * min(1.0, (state["step"] + 1) / max(warm, 1))
                for g in opt.param_groups:
                    g["lr"] = lr
                out = tr(list(xt.to(torch.bfloat16).unsqueeze(2).unbind(0)), (1 - sig.view(-1)).to(torch.bfloat16),
                         prompts, return_dict=False)[0]
                pred = -torch.stack(out, 0).squeeze(2).float()
                loss = ((pred - (noise - x0)) ** 2).mean()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, o.get("max_grad_norm", 1.0))
                opt.step()
                opt.zero_grad(set_to_none=True)
                state.update(stage=si, epoch=ep, batch=bi + 1, step=state["step"] + 1)
                seen += x0.shape[0] * world
                if state["step"] % 10 == 0:
                    lt = loss.detach().clone()
                    dist.all_reduce(lt, op=dist.ReduceOp.AVG)
                    if main_rank:
                        el = time.time() - t0
                        rec = {"step": state["step"], "stage": si, "loss": round(lt.item(), 5), "grad_norm": round(float(gn), 4),
                               "lr": lr, "res": st["res"], "samples_per_sec": round(seen / el, 2),
                               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "steps_per_epoch": n_local}
                        log.write(json.dumps(rec) + "\n")
                        log.flush()
                        print(json.dumps(rec), flush=True)
                flag = torch.tensor([1.0 if STOP["flag"] else 0.0], device=dev)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                hit = (a.max_steps and state["step"] >= a.max_steps) or (cap and state["step"] - state["stage_start"] >= cap)
                if flag.item() or time.time() - last > cfg.get("ckpt_minutes", 30) * 60 or hit:
                    save(f"step-{state['step']}")
                    last = time.time()
                    if flag.item():
                        say("stopped by signal; re-run the same command to resume")
                        dist.destroy_process_group()
                        return 0
                if hit:
                    break
            if (a.max_steps and state["step"] >= a.max_steps) or (cap and state["step"] - state["stage_start"] >= cap):
                break
            state.update(epoch=ep + 1, batch=0)
        if a.max_steps and state["step"] >= a.max_steps:
            break  # stay inside this stage so a later run with a larger --max-steps continues it
        state.update(stage=si + 1, epoch=0, batch=0)
        state.pop("stage_start", None)
    save(f"step-{state['step']}")
    if main_rank:
        (run / "summary.json").write_text(json.dumps({"steps": state["step"], "seconds": round(time.time() - t0, 1),
                                                      "world": world, "samples": seen, "commit": git_commit()}, indent=2))
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
