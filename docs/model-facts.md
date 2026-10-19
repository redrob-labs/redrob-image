# Z-Image model facts

Checked 2026-10-19 against the Hugging Face API, the model repos' config files, the GitHub README,
and the arXiv report. Every link below was fetched (HTTP 200) when this file was written.

## Variants

| Variant | HF repo ID | Status | License | Source |
|---|---|---|---|---|
| Z-Image (the "Base" in our plan) | `Tongyi-MAI/Z-Image` | Released 2026-01-27. Pre-training + SFT, no RL, undistilled, CFG | Apache-2.0 | [HF](https://huggingface.co/Tongyi-MAI/Z-Image), [GitHub news](https://github.com/Tongyi-MAI/Z-Image) |
| Z-Image-Turbo | `Tongyi-MAI/Z-Image-Turbo` | Released 2025-11-26. Pre-training + SFT + RL, distilled to 8 NFE | Apache-2.0 | [HF](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) |
| Z-Image-Omni-Base | none | "To be released" (pre-training only, gen + edit). diffusers already has `ZImageOmniPipeline` | - | [GitHub model zoo](https://github.com/Tongyi-MAI/Z-Image), [diffusers v0.37.0](https://github.com/huggingface/diffusers/releases/tag/v0.37.0) |
| Z-Image-Edit | none | "To be released" | - | [GitHub model zoo](https://github.com/Tongyi-MAI/Z-Image) |

`Tongyi-MAI/Z-Image-Base` and `Tongyi-MAI/Z-Image-Edit` do not exist on HF (API returns 401 /
not found). The org lists only Z-Image and Z-Image-Turbo among image models
([HF org API](https://huggingface.co/api/models?author=Tongyi-MAI)).

## Architecture and inference

| Item | Value | Source |
|---|---|---|
| Architecture | S3-DiT, Scalable Single-Stream DiT: text tokens, (edit-only) SigLIP 2 semantic tokens and VAE tokens are concatenated into one sequence | [Turbo card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), [report](https://arxiv.org/abs/2511.22699) |
| DiT parameters | 6.15B (report Table 2). Base shard total 12.31 GB in bf16 (= ~6.15B); Turbo shards are 24.62 GB (stored fp32) | [report HTML](https://arxiv.org/html/2511.22699), [Base index](https://huggingface.co/Tongyi-MAI/Z-Image/blob/main/transformer/config.json) |
| DiT config | `ZImageTransformer2DModel`: dim 3840, 30 layers, 2 refiner layers, n_heads 30, n_kv_heads 30, qk_norm, patch 2, in_channels 16, cap_feat_dim 2560, 3D RoPE axes (32,48,48), rope_theta 256. Identical for Base and Turbo | [Base transformer config](https://huggingface.co/Tongyi-MAI/Z-Image/blob/main/transformer/config.json) |
| Text encoder | Qwen3-4B (`Qwen3Model` in pipeline, hidden 2560, 36 layers), tokenizer `Qwen2Tokenizer`. Same weights shipped in both repos. Qwen3-4B itself: Apache-2.0, 100+ languages | [model_index.json](https://huggingface.co/Tongyi-MAI/Z-Image), [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |
| VAE | FLUX.1 VAE (`AutoencoderKL`, `_name_or_path: flux-dev`), 16 latent channels, 8x downsample (4 blocks), scaling 0.3611, shift 0.1159. With patch 2 one DiT token covers 16x16 px. BFL states the autoencoder weights are Apache-2.0 | [VAE config](https://huggingface.co/Tongyi-MAI/Z-Image/blob/main/vae/config.json), [flux README](https://github.com/black-forest-labs/flux) |
| Scheduler | `FlowMatchEulerDiscreteScheduler`, 1000 train steps, no dynamic shifting. shift = 6.0 (Base), 3.0 (Turbo) | [Base](https://huggingface.co/Tongyi-MAI/Z-Image/blob/main/scheduler/scheduler_config.json), [Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/scheduler/scheduler_config.json) |
| Base inference | 512x512 to 2048x2048 total area, any aspect ratio; guidance_scale 3.0-5.0; 28-50 steps; negative prompts supported; `cfg_normalization` arg | [Base card](https://huggingface.co/Tongyi-MAI/Z-Image) |
| Turbo inference | `num_inference_steps=9` (= 8 DiT forwards), `guidance_scale=0.0`, example 1024x1024; no CFG, no negative prompt | [Turbo card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) |
| Native text rendering | Claimed: English and Chinese ("bilingual"). Report evaluates CVTG-2K, LongText-Bench (EN/ZH) and shows a Japanese poster as an emerging multilingual example. Korean is not claimed anywhere | [Turbo card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), [report HTML](https://arxiv.org/html/2511.22699) |
| Training curriculum (report) | 256^2 low-res pre-training -> omni pre-training (arbitrary res, T2I + edit) -> SFT -> distillation -> RLHF | [report HTML](https://arxiv.org/html/2511.22699) |

## diffusers

| Item | Value | Source |
|---|---|---|
| Classes | `ZImagePipeline`, `ZImageTransformer2DModel`, plus `ZImageOmniPipeline`, `ZImageControlNetModel` | [docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/z_image) |
| Minimum version | 0.36.0 (2025-12-08) adds Z-Image + LoRA loading. Base (`Tongyi-MAI/Z-Image`) configs were written with 0.37.0.dev0; 0.37.0 (2026-03-05) is the first release after it. Current: 0.41.0 (2026-10-06). Use >= 0.37.0 | [v0.36.0](https://github.com/huggingface/diffusers/releases/tag/v0.36.0), [v0.37.0](https://github.com/huggingface/diffusers/releases/tag/v0.37.0), [PyPI](https://pypi.org/project/diffusers/) |
| Later fixes worth having | 0.38: compile bottlenecks removed; 0.39: deterministic pad tokens, redundant terminal timestep fix, Kohya LoRA conversion; 0.41: LoRA key conversion | [releases](https://github.com/huggingface/diffusers/releases) |
| LoRA training example | `examples/dreambooth/train_dreambooth_lora_z_image.py` (LoRA only) | [script](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_z_image.py) |

## Training code

| Tool | Z-Image support | License | Source |
|---|---|---|---|
| DiffSynth-Studio (ModelScope, listed by Tongyi as community support) | Base and Turbo: full training, LoRA, low-VRAM, `--trainable_models dit`, differential LoRA, experimental trajectory-imitation distillation, DistillPatch LoRA | Apache-2.0 | [doc](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Z-Image.md), [trajectory imitation](https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/z_image/model_training/special/trajectory_imitation) |
| diffusers example | LoRA DreamBooth | Apache-2.0 | [script](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_z_image.py) |
| ostris ai-toolkit | `z_image` model extension (LoRA / FT) | MIT | [code](https://github.com/ostris/ai-toolkit/tree/main/extensions_built_in/diffusion_models/z_image) |
| ostris Turbo adapters | `zimage_turbo_training_adapter` v1/v2 (LoRA that de-distills Turbo during training) and `Z-Image-De-Turbo` (de-distilled DiT). Both Apache-2.0. Obsolete for us since Base exists | Apache-2.0 | [adapter](https://huggingface.co/ostris/zimage_turbo_training_adapter), [De-Turbo](https://huggingface.co/ostris/Z-Image-De-Turbo) |
| musubi-tuner (kohya) | Z-Image LoRA; doc says use Base for training since Jan 2026 | Apache-2.0 (README) | [doc](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/zimage.md) |
| SimpleTuner | Z-Image quickstart (LoRA, turbo + assistant adapter; doc still says the base flavour points to an unavailable checkpoint) | AGPL-3.0 (tool only; does not bind trained weights, but do not vendor its code) | [quickstart](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/ZIMAGE.md) |
| DiffSynth `Z-Image-Turbo-DistillPatch` | LoRA restoring 8-step inference after SFT on Turbo | Apache-2.0 | [HF](https://huggingface.co/DiffSynth-Studio/Z-Image-Turbo-DistillPatch) |

## Turbo distillation method

| Item | Paper | Public code | Source |
|---|---|---|---|
| Decoupled-DMD (CFG augmentation = engine, distribution matching = regularizer) | arXiv 2511.22677 | No official code found (arXiv points to the Z-Image repo, which has no training code). Community single-file reimplementation: `marduk191/z-image-distillation-trainer` (Apache-2.0, 11 stars, unvetted) | [paper](https://arxiv.org/abs/2511.22677), [community](https://github.com/marduk191/z-image-distillation-trainer) |
| DMDR (DMD + RL, ECCV 2026) | arXiv 2511.13649 | `vvvvvjdy/dmdr`, MIT, but only an ImageNet training demo ("We now only have access to open the training demo code of ImageNet"); built on DMD2 | [paper](https://arxiv.org/abs/2511.13649), [code](https://github.com/vvvvvjdy/dmdr) |

## Where the plan was wrong or imprecise

1. "Z-Image Base" is HF `Tongyi-MAI/Z-Image`, not `Z-Image-Base`. It is already SFT'd (only RL
   and distillation are missing). The rawer pre-training checkpoint (Omni-Base) is not released.
2. Our "own few-step distillation" has no official code to start from. Decoupled-DMD has none,
   DMDR has an ImageNet demo only. Realistic starting points: DMD2 + the DMDR demo, DiffSynth's
   trajectory-imitation distillation, or the unvetted community Decoupled-DMD trainer. Budget for
   writing the distillation ourselves.
3. VAE is FLUX.1's (8x, 16 ch) and stays frozen, and one token covers 16x16 px. Small Hangul
   strokes may be lost in the VAE before the DiT sees them: T3 should measure OCR on VAE
   round-trips of the eval set to set the ceiling and minimum glyph size.
4. The report's Table 2 says 32 attention heads; the shipped config has `n_heads: 30`
   (3840 / 30 = 128 head dim). Trust the config.
5. Turbo weights are stored fp32 (24.6 GB) while Base is bf16 (12.3 GB); same parameter count.
6. Korean rendering is not claimed by Tongyi at all (only EN/ZH), consistent with the plan's goal.
   Qwen3-4B's tokenizer does cover Korean, so freezing the text encoder is not a blocker on its
   face, but this is unverified for Hangul glyph fidelity.
