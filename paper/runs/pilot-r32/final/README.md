# Pilot r32 — final (end of 2 epochs @512 + 1500 steps @1024)

Base LoRA rank 32, attention only, AI Hub 105 real + 20k synth, scored with qwen3vl-8b on all 220 eval prompts.
CER is the share of characters wrong (lower is better).

| Model | all | seen | unseen | unseen syllable |
|---|---:|---:|---:|---:|
| Base (no LoRA) | 0.339 | 0.241 | 0.344 | 0.808 |
| Turbo (no LoRA) | 0.442 | 0.356 | 0.451 | 0.825 |
| **Base + r32 LoRA** | **0.295** | **0.192** | **0.302** | 0.775 |
| **Turbo + r32 LoRA** | 0.417 | 0.361 | 0.385 | 0.858 |

Pilot gate (plan §1.5): unseen-word CER must fall below baseline. Met — Base+LoRA unseen 0.344 → 0.302.

Caveats:
- The unseen-syllable column is not trustworthy: the OCR itself cannot read KS X 1001-outside syllables, so
  0.77–0.86 measures the OCR, not the model.
- Base+LoRA improves every split; Turbo+LoRA helps unseen but slightly hurts seen (Turbo's few steps dilute the
  LoRA), so the merge path still needs re-distillation rather than a direct LoRA graft onto Turbo.
- The remaining error is mostly wrong vowels / missing strokes (깨→꺼, 게→계) that do not change across steps;
  r64 v2 (text-weighted loss, logit-normal timesteps, 1024-only small text) targets exactly this.

Files: report-r32-Z-Image.json (Base+LoRA), report-r32-Z-Image-Turbo.json (Turbo+LoRA), train-summary.json.
Baselines are redrob-image-eval/outputs/baseline-{base,turbo} (committed under paper/runs/baseline-* earlier).
