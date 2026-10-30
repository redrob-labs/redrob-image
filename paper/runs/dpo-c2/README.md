# Plan C stage 1: region Diffusion-DPO (c2, stronger settings)

Same 1,746 pairs and reference (r64 v2) as c1. Config
`configs/g6e-dpo-c2.yaml`: beta 500, lr 5e-5, 8 pairs/step, 4 epochs,
836 steps, text-region loss weight 5. Training 13.8 h on one L40S.

## Held-out preference accuracy (76 pairs)

| step | pref_acc |
|---:|---:|
| 150 | 0.88 |
| 400 | 0.91 |
| 700 | 0.95 |
| 800 | 0.895 |
| 836 | 0.882 |

c1 ended at 0.855.

## Final eval (220 prompts, seed 42, dual OCR)

| model | CER | seen CER | unseen CER | unseen-syl CER | EM | seen EM | unseen EM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base + r64 v2 | 0.281 | 0.151 | 0.312 | 0.775 | 0.305 | 0.48 | 0.19 |
| Base + DPO c1 | 0.284 | 0.157 | 0.320 | 0.742 | 0.295 | 0.48 | 0.17 |
| Base + DPO c2 | 0.314 | 0.188 | 0.356 | 0.733 | 0.259 | 0.41 | 0.16 |

## Classifier reward on the same 220 images

| model | mean reward | extra text cells |
|---|---:|---:|
| r64 v2 | -7.25 | 5.53 |
| DPO c1 | -7.70 | 5.95 |
| DPO c2 | -5.25 | 2.41 |

## Reading

c2 moved off the reference: the reward rose (-7.25 -> -5.25), almost all of
it from fewer extra text cells (5.53 -> 2.41). Target-text accuracy got
worse under dual OCR (CER 0.281 -> 0.314, EM 0.305 -> 0.259). The DPO
signal taught the model to draw less stray text, not to draw the right
jamo. Vowel errors in the 8-image peek remain (e.g. 신선한 -> 신션한,
가게 -> 가계, 꼴뚜기 -> 골투기).

The reward's extra-text penalty dominates the pair ranking; the jamo term
is too weak relative to it. Images are not committed.
