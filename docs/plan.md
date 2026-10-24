# Korean text rendering plan

Goal: make Redrob Image render Korean (Hangul) text. The previous Z-Image Turbo merge is not kept;
the next release is rebuilt from Z-Image Base (Korean text + realism trained together, then our own
few-step distillation).

## Decisions

| Item | Value |
|---|---|
| Training base | Z-Image Base |
| Trained part | DiT only (text encoder and VAE frozen) |
| Region | ap-northeast-2 (Seoul) for anything touching AI Hub data |
| Repository | redrob-labs/redrob-image |
| Excluded models | HunyuanImage 3.0, Qwen-Image-2.1, Ideogram 4.0, EXAONE (licenses) |

| Path | Method | Instance | Data |
|---|---|---|---|
| Pilot | LoRA on Base | g6e.xlarge / gui-box-gpu (L40S) | 20k real + 20k synthetic |
| Path B | Base full FT (Korean + realism) -> own few-step distillation | p5.48xlarge (human approval) | 1M-1.5M |

Go to Path B only when CER drops on unseen words AND unseen syllables, with no loss in general quality.

## Rules

- AI Hub raw or processed data: never in git, never in a public bucket, never redistributed.
  AI Hub data stays in Seoul until overseas transfer is confirmed.
- This repository is public: no AI Hub paths, account IDs, keys, or raw data samples.
  `scripts/check_public.py` checks staged files.
- Fonts: commercial-training-allowed licenses only (OFL etc.), 20+ families, license files in
  `synth/fonts/LICENSES/`.
- Training writes S3 checkpoints every 30 minutes and resumes.
- p5 is prepared, never launched without human approval.

## Layout

```
configs/  data/ (ignored)  docs/  eval/  infra/  paper/  scripts/  synth/  workflows/
```

## Tasks

| ID | Task | Done when |
|---|---|---|
| T1 | Model facts + dataset sources | `docs/model-facts.md`, `docs/datasets.md` with links |
| T2 | Layout, `.gitignore`, private S3 | check script passes |
| T3 | 200 eval prompts, unseen words/syllables, 2+ OCR engines | baseline reports |
| T4 | AI Hub parser, text crops, unseen filter | sample stats report |
| T5 | Qwen3-VL captions, all text, 2-3 variants | 1k preview |
| T6 | Synthetic generator, 2,350 -> 11,172 syllables | 10k images, coverage |
| T7 | Precompute latents and embeddings | speed and size table |
| T8 | LoRA r32/r64, 512 -> 1024 | 100-step smoke, time estimate |
| T9 | p5 FSDP, self-generated regularization data, runbook, cost guard | `docs/path-b-runbook.md` |
| T10 | README limits, NOTICE, model card | at release |

## Paper data

Every run writes `paper/runs/<run>/` (config, seed, commit, package versions, metrics JSON,
curves). Costs go to `paper/cost.csv`, data statistics (numbers only) to `paper/data-stats/`,
OCR comparison and human review to `paper/ocr/`. Images stay in private storage. Every piece of
work done by an AI agent is logged in `paper/ai-usage.md`.
