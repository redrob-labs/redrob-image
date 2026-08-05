---
license: apache-2.0
base_model: Tongyi-MAI/Z-Image-Turbo
base_model_relation: merge
tags:
  - text-to-image
  - redrob
  - redrob-image
  - z-image
  - diffusers
  - comfyui
  - diffusion
library_name: diffusers
pipeline_tag: text-to-image
---

# Redrob Image

[한국어](./README.ko.md)

**Redrob Image** is [Redrob](https://redrob.io)'s open-weight diffusion model, built by [Janghoon Lee (이장훈)](https://www.janghoonlee.com).

Redrob's vision is to democratize AI. Our models are free to use and free for commercial use, under [Apache License 2.0](./LICENSE).

## Why this model

- Tuned for a more realistic look than the bare base model: better skin, light, and texture, with less of the plastic, AI-skin tell.
- Strong on photo, portrait, and mood imagery.
- Fast Turbo-style sampling: about 8 DiT steps.
- Runs in plain Python via Diffusers, or as a single merged UNET in ComfyUI.
- Apache-2.0: free to use, free for commercial use, and redistributable.

## Limits

Weak at legible text, including Hangul, Devanagari, and most non-Latin script. Route text-bearing surfaces elsewhere. Also weaker on graphic, print, and typography-heavy work than on photo and portrait.

Turbo-style sampling runs without classifier-free guidance (`guidance_scale=0` / ComfyUI `cfg 1`), so **negative prompts are ignored**. Put avoidances in the positive prompt instead.

## Quick start (Diffusers / Python)

Enterprise and production default. After the Hugging Face upload includes `transformer/`, load that Diffusers transformer and keep the text encoder / VAE from the base pipeline.

```bash
pip install -U torch transformers accelerate safetensors
pip install -U diffusers
```

```python
import torch
from diffusers import ZImagePipeline, ZImageTransformer2DModel

transformer = ZImageTransformer2DModel.from_pretrained(
    "savagemanage/redrob-image",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

prompt = "A documentary portrait in natural window light, shallow depth of field"
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,  # 8 DiT forwards
    guidance_scale=0.0,     # required for Turbo
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("redrob-image.png")
```

Optional: `pipe.enable_model_cpu_offload()` on smaller GPUs.

## Quick start (ComfyUI)

| File                              | Put under                  | Source                                                                    |
| --------------------------------- | -------------------------- | ------------------------------------------------------------------------- |
| `redrob-image.safetensors`        | `models/diffusion_models/` | this repository                                                           |
| `qwen_3_4b_fp8_mixed.safetensors` | `models/text_encoders/`    | [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) |
| `ae.safetensors`                  | `models/vae/`              | same Comfy-Org pack                                                       |

1. `UNETLoader` -> `redrob-image.safetensors`
2. `CLIPLoader` -> `qwen_3_4b_fp8_mixed.safetensors` (`type: lumina2`, ComfyUI loader type for this text encoder)
3. `VAELoader` -> `ae.safetensors`
4. Sampler: **8 steps**, **cfg 1**, `res_multistep` / `sgm_uniform`

Load `workflows/redrob-image-api.json` for a minimal working graph. The graph zeros out negative conditioning (`ConditioningZeroOut`); do not expect a negative text prompt to change the image.

## Files

| Path                              | Role                                  |
| --------------------------------- | ------------------------------------- |
| `redrob-image.safetensors`        | ComfyUI merged UNET (LFS, ~12 GiB)    |
| `transformer/`                    | Diffusers layout (built at HF upload) |
| `workflows/redrob-image-api.json` | Minimal ComfyUI API graph             |
| `README.md` / `README.ko.md`      | Model card (English / Korean)         |
| `LICENSE`                         | Apache License 2.0                    |
| `NOTICE`                          | Attribution                           |

`transformer/` is not in git. On Hugging Face upload, `scripts/push_hf.sh` converts the Comfy UNET into Diffusers format (or copies a prebuilt `TRANSFORMER_DIR`).

Convert a local Comfy UNET yourself:

```bash
python scripts/comfy_to_diffusers_zimage.py \
  --input redrob-image.safetensors \
  --output-dir transformer/
```

## License

[Apache License 2.0](./LICENSE). Copyright [Redrob](https://redrob.io). Built by [Janghoon Lee (이장훈)](https://www.janghoonlee.com). Upstream attribution is in [NOTICE](./NOTICE). Redistributors keep NOTICE with the weights.

## Attribution

- **Copyright:** Redrob ([redrob.io](https://redrob.io))
- **Built by:** [Janghoon Lee (이장훈)](https://www.janghoonlee.com) ([@savagemanage](https://github.com/savagemanage))
- **Base model:** [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) - Apache License 2.0
