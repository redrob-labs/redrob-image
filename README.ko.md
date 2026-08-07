# Redrob Image

[English](./README.md)

**Redrob Image**는 [Redrob](https://redrob.io)의 오픈웨이트 확산 모델이고, [이장훈 (Janghoon Lee)](https://www.janghoonlee.com)가 만들었습니다.

Redrob의 비전은 AI를 민주화하는 것 같은데요. 이 모델은 [Apache License 2.0](./LICENSE) 아래에서 무료로 쓰고, 상업적으로도 쓸 수 있습니다.

## 왜 이 모델인가

- 베이스보다 현실적인 쪽으로 맞췄습니다. 피부, 빛, 질감이 낫고, 플라스틱한 AI 피부 티가 덜합니다.
- 사진, 포트레이트, 분위기 이미지에서 강합니다.
- Turbo 스타일 샘플링이라 DiT 기준 약 8스텝입니다.
- Diffusers로 그냥 Python에서 돌리거나, ComfyUI에서는 머지된 UNET 하나로 씁니다.
- Apache-2.0: 사용, 상업 이용, 재배포 가능합니다.

## 한계

읽히는 텍스트는 약합니다. 한글, 데바나가리, 비라틴 스크립트 대부분도 포함됩니다. 텍스트가 중요한 화면은 다른 경로로 빼는 게 맞다고 봅니다. 그래픽, 인쇄, 타이포 중심 작업도 사진/포트레이트보다 약합니다.

Turbo 샘플링은 classifier-free guidance 없이 돕니다 (`guidance_scale=0` / ComfyUI `cfg 1`). 그래서 **네거티브 프롬프트는 무시됩니다**. 피하고 싶은 건 포지티브 쪽에 넣으세요.

## 빠른 시작 (Diffusers / Python)

엔터프라이즈·프로덕션 기본입니다. Hugging Face 업로드에 `transformer/`가 들어가 있으면, 그 Diffusers transformer를 로드하고 텍스트 인코더 / VAE는 베이스 파이프라인에서 가져옵니다.

```bash
pip install -U torch transformers accelerate safetensors
pip install -U diffusers
```

```python
import torch
from diffusers import ZImagePipeline, ZImageTransformer2DModel

transformer = ZImageTransformer2DModel.from_pretrained(
    "redrob-labs/redrob-image",
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
    num_inference_steps=9,  # DiT forward 8회
    guidance_scale=0.0,     # Turbo에서 필수
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("redrob-image.png")
```

작은 GPU면 `pipe.enable_model_cpu_offload()`를 쓰면 됩니다.

## 빠른 시작 (ComfyUI)

| 파일                              | 위치                       | 출처                                                                      |
| --------------------------------- | -------------------------- | ------------------------------------------------------------------------- |
| `redrob-image.safetensors`        | `models/diffusion_models/` | 이 저장소                                                                 |
| `qwen_3_4b_fp8_mixed.safetensors` | `models/text_encoders/`    | [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) |
| `ae.safetensors`                  | `models/vae/`              | 같은 Comfy-Org 팩                                                         |

1. `UNETLoader` -> `redrob-image.safetensors`
2. `CLIPLoader` -> `qwen_3_4b_fp8_mixed.safetensors` (`type: lumina2`)
3. `VAELoader` -> `ae.safetensors`
4. 샘플러: **8 steps**, **cfg 1**, `res_multistep` / `sgm_uniform`

최소 그래프는 `workflows/redrob-image-api.json`입니다. 네거티브 컨디셔닝은 `ConditioningZeroOut`으로 비웁니다. 네거티브 텍스트가 이미지를 바꿀 거라고 기대하지 마세요.

## 파일

| 경로                              | 역할                                  |
| --------------------------------- | ------------------------------------- |
| `redrob-image.safetensors`        | ComfyUI 머지 UNET (LFS, ~12 GiB)      |
| `transformer/`                    | Diffusers 레이아웃 (HF 업로드 시 생성) |
| `workflows/redrob-image-api.json` | 최소 ComfyUI API 그래프               |
| `README.md` / `README.ko.md`      | 모델 카드 (영어 / 한국어)             |
| `LICENSE`                         | Apache License 2.0                    |
| `NOTICE`                          | 출처 표기                             |

`transformer/`는 git에 없습니다. Hugging Face 업로드 때 `scripts/push_hf.sh`가 Comfy UNET을 Diffusers 포맷으로 변환하거나, 미리 만든 `TRANSFORMER_DIR`을 복사합니다.

로컬에서 Comfy UNET을 직접 변환하려면:

```bash
python scripts/comfy_to_diffusers_zimage.py \
  --input redrob-image.safetensors \
  --output-dir transformer/
```

## 라이선스

[Apache License 2.0](./LICENSE). 저작권은 [Redrob](https://redrob.io). 제작은 [이장훈 (Janghoon Lee)](https://www.janghoonlee.com). 업스트림 표기는 [NOTICE](./NOTICE). 재배포할 때는 NOTICE를 가중치와 함께 유지하세요.

## 저작 / 출처

- **저작권:** Redrob ([redrob.io](https://redrob.io))
- **제작:** [이장훈 (Janghoon Lee)](https://www.janghoonlee.com) ([@savagemanage](https://github.com/savagemanage))
- **저장소:** [redrob-labs/redrob-image](https://huggingface.co/redrob-labs/redrob-image)
- **베이스:** [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) - Apache License 2.0
