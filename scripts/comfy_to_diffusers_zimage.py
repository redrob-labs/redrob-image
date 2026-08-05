#!/usr/bin/env python3
"""Convert a ComfyUI Z-Image UNET safetensors file to Diffusers transformer layout.

Reverse of Comfy-Org/z_image_turbo's z_image_convert_original_to_comfy.py.

Usage:
  python scripts/comfy_to_diffusers_zimage.py \\
    --input redrob-image.safetensors \\
    --output-dir transformer/

Writes:
  transformer/config.json
  transformer/diffusion_pytorch_model.safetensors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import safetensors.torch
import torch

# Default config matches Tongyi-MAI/Z-Image-Turbo transformer/config.json
DEFAULT_CONFIG = {
    "_class_name": "ZImageTransformer2DModel",
    "_diffusers_version": "0.36.0.dev0",
    "all_f_patch_size": [1],
    "all_patch_size": [2],
    "axes_dims": [32, 48, 48],
    "axes_lens": [1536, 512, 512],
    "cap_feat_dim": 2560,
    "dim": 3840,
    "in_channels": 16,
    "n_heads": 30,
    "n_kv_heads": 30,
    "n_layers": 30,
    "n_refiner_layers": 2,
    "norm_eps": 1e-05,
    "qk_norm": True,
    "rope_theta": 256.0,
    "t_scale": 1000.0,
}

PREFIXES = (
    "model.diffusion_model.",
    "diffusion_model.",
)

# Applied after qkv split. Comfy name -> Diffusers name.
REPLACE = (
    ("final_layer.", "all_final_layer.2-1."),
    ("x_embedder.", "all_x_embedder.2-1."),
    (".attention.out.bias", ".attention.to_out.0.bias"),
    (".attention.out.weight", ".attention.to_out.0.weight"),
    (".attention.k_norm.weight", ".attention.norm_k.weight"),
    (".attention.q_norm.weight", ".attention.norm_q.weight"),
)


def strip_prefix(key: str) -> str:
    for prefix in PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def rename_key(key: str) -> str:
    out = key
    for src, dst in REPLACE:
        out = out.replace(src, dst)
    return out


def split_qkv(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.shape[0] % 3 != 0:
        raise ValueError(
            f"qkv first dim not divisible by 3: shape={tuple(weight.shape)}"
        )
    q, k, v = weight.chunk(3, dim=0)
    return q, k, v


def convert_state_dict(comfy_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for raw_key, tensor in comfy_sd.items():
        key = strip_prefix(raw_key)

        if key.endswith(".attention.qkv.weight"):
            q, k, v = split_qkv(tensor)
            base = key[: -len(".attention.qkv.weight")]
            out[rename_key(f"{base}.attention.to_q.weight")] = q.contiguous()
            out[rename_key(f"{base}.attention.to_k.weight")] = k.contiguous()
            out[rename_key(f"{base}.attention.to_v.weight")] = v.contiguous()
            continue

        out[rename_key(key)] = tensor

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ComfyUI Z-Image UNET weights to Diffusers transformer/"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="ComfyUI diffusion_models/*.safetensors",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        type=Path,
        help="Output directory (writes config.json + diffusion_pytorch_model.safetensors)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional Diffusers transformer config.json (defaults to Z-Image-Turbo)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"missing input: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.config is not None:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    else:
        config = dict(DEFAULT_CONFIG)

    print(f"[load] {args.input}")
    comfy_sd = safetensors.torch.load_file(str(args.input), device="cpu")
    print(f"[keys] comfy={len(comfy_sd)}")

    diffusers_sd = convert_state_dict(comfy_sd)
    print(f"[keys] diffusers={len(diffusers_sd)}")

    weight_path = args.output_dir / "diffusion_pytorch_model.safetensors"
    config_path = args.output_dir / "config.json"

    print(f"[write] {weight_path}")
    safetensors.torch.save_file(diffusers_sd, str(weight_path))

    print(f"[write] {config_path}")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print("[done]")


if __name__ == "__main__":
    main()
