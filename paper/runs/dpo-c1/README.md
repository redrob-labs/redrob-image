# Plan C stage 1: region Diffusion-DPO (c1)

Start: r64 v2 LoRA (also the frozen reference). Pairs: 1,746 best/worst
pairs from 2,000 prompts x 4 rollouts, scored by the syllable-classifier
reward. Config `configs/g6e-dpo-c1.yaml`: beta 2000, lr 1e-5, 8 pairs/step,
2 epochs, 418 steps, text-region loss weight 5.

## Held-out preference accuracy (76 pairs)

| step | pref_acc | val_d |
|---:|---:|---:|
| 50 | 0.72 | -1.2e-4 |
| 100 | 0.78 | -2.1e-4 |
| 400 | 0.868 | -8.2e-4 |
| 418 | 0.855 | -8.5e-4 |

## Final eval (220 prompts, seed 42, dual OCR)

| model | CER | seen CER | unseen CER | unseen-syl CER | EM | seen EM | unseen EM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base + r64 v2 | 0.281 | 0.151 | 0.312 | 0.775 | 0.305 | 0.48 | 0.19 |
| Base + DPO c1 | 0.284 | 0.157 | 0.320 | 0.742 | 0.295 | 0.48 | 0.17 |

## Classifier reward on the same 220 images

| model | mean reward | all target cells read | extra text cells |
|---|---:|---:|---:|
| r64 v2 | -7.25 | 0.527 | 5.53 |
| DPO c1 | -7.70 | 0.441 | 5.95 |

## Reading

DPO learned to rank the held-out pairs (0.72 -> 0.86) but the generated
images did not improve: every eval metric is within noise of r64 v2, and the
reward on fresh samples is slightly lower. With beta 2000 and lr 1e-5 the
LoRA stayed close to the reference (step-201 peek: 3-9% pixel RMSE).
Candidate rerun on the same pairs: lr 5e-5, beta 500, 4 epochs.

Box images are not committed (AI Hub-derived training data rules).
