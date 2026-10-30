# Pilot r64 v2 — final eval (qwen3vl, 220 prompts)

r64 v2 = LoRA rank 64 on attention + FFN, 5x loss inside text boxes,
logit-normal timesteps, <16px text only at 1024, 3,000 steps at 1024.
Trained 12,216 steps. Config: `configs/g6e-r64v2.yaml`.

CER (lower is better) / exact match.

| Model | All | seen | unseen | unseen syllable | EM all |
|---|---:|---:|---:|---:|---:|
| Base | 0.339 | 0.241 | 0.344 | 0.808 | |
| Base + r32 | 0.295 | 0.192 | 0.302 | 0.775 | |
| Base + r64 v2 | 0.242 | 0.099 | 0.279 | 0.775 | 0.382 |
| Turbo | 0.442 | 0.356 | 0.451 | 0.825 | |
| Turbo + r32 | 0.417 | 0.361 | 0.385 | 0.858 | |
| Turbo + r64 v2 | 0.407 | 0.317 | 0.409 | 0.842 | 0.182 |

Base + r64 v2 exact match: seen 0.56, unseen 0.28.

Notes:
- The four changes were applied together; their effects are not separated.
- unseen-syllable CER is not trustworthy: the OCR cannot read those syllables.
- Quality is still not usable for text (unseen EM 0.28). Next: glyph-conditioned
  pilot (`docs/glyph-pilot-plan.md`).

Reports: `report-r64v2-Z-Image.json`, `report-r64v2-Z-Image-Turbo.json`.
