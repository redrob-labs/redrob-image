# Base-model comparison: Qwen-Image-2512 (no fine-tuning)

Same 220-prompt eval set, seed 42, 1024px, 50 steps, guidance 4.0, dual OCR
(PaddleOCR and qwen3vl must both agree). Images stay on the box.

| Model | dual CER | seen | unseen | unseen syllable | EM | unseen EM | reward_cls | extra cells |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Z-Image Base | 0.410 | 0.313 | 0.423 | | 18% | 14% | | |
| Z-Image Base + r64 v2 | **0.281** | **0.151** | **0.312** | 0.775 | **30.5%** | **19%** | **-7.25** | 5.53 |
| Qwen-Image-2512 | 0.459 | 0.365 | 0.494 | 0.758 | 12.7% | 10% | -8.18 | 2.21 |

- Qwen-Image-2512 is worse than both Z-Image Base and r64 v2 on Korean target text.
- It draws fewer extra text cells (2.2 vs 5.5), so its low reward comes from wrong syllables.
- Generation: 60.7 s/image, peak 38.1 GB on one L40S.

Files: `report.json`, `scores.jsonl`, `rewards_cls.jsonl`, `metadata.json`;
8-prompt smoke run in `smoke/`.
