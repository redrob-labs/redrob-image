# Path B runbook: full fine-tuning on p5.48xlarge, then our own few-step model

Status: prepared, not launched. Launching any p5 instance or buying a Capacity Block needs Janghoon's
explicit approval (docs/plan.md). Everything below runs in ap-northeast-2 (AI Hub data must stay in Korea).

## 0. Go / no-go

| Gate | Evidence | Owner |
|---|---|---|
| Pilot passes | `paper/runs/pilot-*`: unseen-word CER below the Base baseline (0.34), general prompts not worse in side-by-side review | Kiro measures, Janghoon decides |
| AI Hub terms | 105/81 (and 633/71553/71725 if used) allow commercial model release; attribution wording known | Janghoon |
| Capacity | Seoul sells p5.48xlarge Capacity Blocks, or on-demand quota (P instances, 192 vCPU) is granted | Janghoon |
| Credits | Sponsorship credits apply to Capacity Blocks (else on-demand price) | Janghoon |
| Data on S3 | `precomp/512`, `precomp/1024`, `empty.safetensors` in the private bucket, sizes in `paper/data-stats/` | Kiro |
| NVIDIA OCR set | Human sign-off on CC BY 4.0 use (attribution in NOTICE) | Janghoon |

## 1. Data that must exist before the node starts (cheap GPUs, not p5)

| Source | Target | Tool | Where |
|---|---|---|---|
| AI Hub 105, all 28 files | ~0.4-0.5M samples | `infra/aihub_queue.sh`, `scripts/aihub105_parse.py` | gpu-box / g6e |
| AI Hub 81 text-in-the-wild | ~0.1M | parser to add (same output format as 105) | gpu-box / g6e |
| NVIDIA OCR-Synthetic `ko` | 0.3M of 1.8M | `infra/hf_relay.sh`; h5 -> shard converter to add | gpu-box / g6e |
| Our synth | 0.5M | `synth/generate.py --stage 2` (CPU, ~130 img/s on 24 cores) | any CPU host |
| Realism (keeps general quality) | 0.6M | Z-Image Base self-generated images from a general prompt set (Apache-2.0 output) + PD12M subset, recaptioned with `scripts/caption.py` style | g6e fleet |
| Captions | all real images | `scripts/caption.py` (1.1 s/img on one L40S, batch 16) | g6e fleet |
| Pre-compute | 512 for all, 1024 for ~0.3M | `scripts/precompute.py` (~1.5 MiB/sample at 512, mostly text embeddings) | g6e fleet, then `aws s3 sync` |

Storage estimate: ~2.5M samples x 1.5 MiB = ~3.7 TB at 512 plus ~0.5 TB at 1024. p5.48xlarge has 8x 3.84 TB
local NVMe; sync from S3 at the start (~4.2 TB at ~2 GB/s is ~40 min).

## 2. Compute and cost estimate

Measured: pilot LoRA on one L40S, 512 px, batch 8, gradient checkpointing: 1.68 samples/s, 17.4 GB peak.
Assumption (not measured): full fine-tuning on H100 runs ~4 samples/s per GPU at 512 and ~1 at 1024.

| Phase | Samples | Node-hours (8x H100) |
|---|---|---|
| Full FT 512, 1 epoch | 2.3M | ~80 |
| Full FT 1024 | 6000 steps x 32 = 0.19M | ~7 |
| Few-step distillation (section 4) | - | 25-60 |
| Evaluation, export, slack | - | ~10 |
| Total | | ~120-160 |

| Price basis | $/node-hour | Total (USD) | Total (KRW, 1,450/$) |
|---|---|---|---|
| Capacity Block (AWS example ~$31/h; varies) | 31 | 3,700-5,000 | 5.4M-7.2M |
| On-demand (Seoul, Oct 2026) | 75.96 | 9,100-12,200 | 13.2M-17.6M |

Re-measure samples/s in the first 30 minutes on the node and recompute before committing the full block.
If the measured rate is under 3 samples/s/GPU, cut the 512 epoch to the AI Hub + realism mix first.

## 3. Run

1. Reserve: Capacity Block for p5.48xlarge, ap-northeast-2, duration = estimate + 20%. (Janghoon)
2. Launch (AWS Deep Learning AMI, PyTorch 2.11+, EFA not needed for one node). Tag `Project=redrob-image`.
   Set `InstanceInitiatedShutdownBehavior=terminate`.
3. On the node:
   ```bash
   git clone https://github.com/redrob-labs/redrob-image && cd redrob-image
   python -m venv ~/venv && ~/venv/bin/pip install torch==2.11.0 diffusers==0.42.0 transformers==5.20.1 peft==0.21.2 safetensors pyyaml
   aws s3 sync s3://<bucket>/precomp /opt/dlami/nvme/precomp --region ap-northeast-2
   export HF_HOME=/opt/dlami/nvme/hf
   setsid nohup ~/venv/bin/torchrun --nproc-per-node 8 scripts/train_full.py --config configs/p5.yaml \
       --run-dir /opt/dlami/nvme/runs/pathb-v1 > /opt/dlami/nvme/runs/pathb-v1.log 2>&1 &
   setsid nohup ~/venv/bin/python infra/cost_guard.py --pid <torchrun pid> --usd-per-hour 31 \
       --budget-usd <approved budget> --max-hours <block hours> --until <block end, UTC ISO> \
       --log /opt/dlami/nvme/runs/cost_guard.log &
   ```
4. Watch `log.jsonl` (loss, samples/s, memory). Checkpoints every 30 min go to `s3://<bucket>/runs/pathb-v1/ckpt`.
5. Interruption or block end: the trainer checkpoints on SIGTERM; re-run the same command on a new node to resume
   (same world size, 8).
   - EC2 sends an EventBridge event 40 min before the block ends and starts terminating 30 min before it. With
     `--until`, the guard SIGTERMs the trainer at end - 30 - grace - 10 min, so the last checkpoint is in S3 before
     the node goes. At most the work since that checkpoint is lost.
   - Not done yet: extend the block from the console while it runs (payment must clear 35 min before the end), or
     buy the next block, or resume on on-demand p5 / g7e.48xlarge (both 8 GPUs, same world size). The data is
     pre-computed in S3, so a new node is training again within the sync time.
   - Size the block from the samples/s measured in the first 30 min, plus about 20% margin.
6. Export: `torchrun --nproc-per-node 1 scripts/train_full.py --config configs/p5.yaml --run-dir ... --export <dir>`,
   then `aws s3 sync <dir> s3://<bucket>/models/zimage-ko-base-v1`.
7. Evaluate on a g6e (not on p5): `scripts/eval_baseline.py --model <exported pipeline dir> --steps 28 --guidance 4`.
8. Terminate the node (cost guard does it when the trainer exits). Confirm in the console that nothing is left running.

`scripts/train_full.py` was dry-run on one L40S with `--trainable-blocks 2` (same FSDP2, checkpoint and resume code,
world size 1). The 8-GPU run itself is untested until the node exists; budget the first hour for that.

## 4. Few-step model (replaces Turbo)

No official Turbo distillation code exists (docs/model-facts.md). Order of attempts, cheapest first:

1. Delta transfer: apply (our Base - original Base) to Z-Image-Turbo and measure. Free; may already work
   because Turbo shares the architecture. Pilot LoRA on Turbo is the early signal.
2. DiffSynth-Studio trajectory-imitation distillation (Apache-2.0) from our Base to 8 steps.
3. Decoupled-DMD reimplementation (DMD2 + the DMDR demo as reference) if 2 loses text accuracy.

Each candidate goes through the same eval (CER on unseen words, general-quality side-by-side) before release.

## 5. Release checklist

- README.md / README.ko.md limits section updated with measured numbers.
- NOTICE: AI Hub attribution (exact wording from the terms), NVIDIA CC BY 4.0, PD12M, fonts (OFL), Wikipedia (CC BY-SA).
- Model card on HF with the same attributions and the data-use statement AI Hub requires.
- `paper/ai-usage.md` row for every AI-assisted step.
