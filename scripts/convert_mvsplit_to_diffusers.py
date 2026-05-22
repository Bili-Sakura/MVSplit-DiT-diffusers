#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None

from diffusers.models.transformers import MVSplitDiTTransformer2DModel
from diffusers.schedulers import MVSplitFlowMatchScheduler


def _load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    if checkpoint_path.endswith(".safetensors"):
        if safe_load_file is None:
            raise ImportError("Install safetensors to convert .safetensors checkpoints.")
        state_dict = safe_load_file(checkpoint_path, device="cpu")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict):
            for key in ("state_dict", "model", "module"):
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break
    return _clean_state_dict(state_dict)


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    prefixes = ("model.", "module.", "transformer.", "_orig_mod.")
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def _save_json(output_dir: Path, payload: Dict[str, Any], name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / name, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _print_state_dict_mismatch(missing_keys, unexpected_keys):
    if not missing_keys and not unexpected_keys:
        return
    print("Checkpoint mismatch while loading into MVSplitDiTTransformer2DModel.")
    if missing_keys:
        print("Missing keys:", missing_keys[:10])
    if unexpected_keys:
        print("Unexpected keys:", unexpected_keys[:10])


def _write_model_index(output_dir: Path, include_vae: bool, include_text: bool):
    model_index = {
        "_class_name": "MVSplitDiTPipeline",
        "_diffusers_version": "0.36.0",
        "scheduler": ["diffusers", "MVSplitFlowMatchScheduler"],
        "transformer": ["diffusers", "MVSplitDiTTransformer2DModel"],
    }
    if include_vae:
        model_index["vae"] = ["diffusers", "AutoencoderKL"]
    if include_text:
        model_index["text_encoder"] = ["transformers", "AutoModel"]
        model_index["tokenizer"] = ["transformers", "AutoTokenizer"]
    _save_json(output_dir, model_index, "model_index.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MVSplit-DiT checkpoints into a Diffusers-style pipeline directory."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt/.bin/.safetensors checkpoint.")
    parser.add_argument("--output", required=True, help="Output Diffusers model directory.")
    parser.add_argument("--in-channels", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=1000)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--mlp-hidden-dim", type=int, default=3072)
    parser.add_argument("--context-dim", type=int, default=1024)
    parser.add_argument("--qkv-bias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trainable-rms", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rope-base", type=int, default=10000)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--init-alpha", type=float, default=0.0)
    parser.add_argument("--init-beta", type=float, default=0.03)
    parser.add_argument("--time-shift-alpha", type=float, default=4.0)
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-load", action="store_true", help="Instantiate model and verify state_dict loading.")
    parser.add_argument(
        "--text-encoder",
        default="Qwen/Qwen3-0.6B",
        help="HF model id to save as a loading hint in text_encoder_pretrained_model_name_or_path.txt.",
    )
    parser.add_argument("--copy-vae", default=None, help="Optional local VAE directory to copy into output/vae.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    transformer_dir = output_dir / "transformer"
    scheduler_dir = output_dir / "scheduler"

    transformer_config = {
        "_class_name": "MVSplitDiTTransformer2DModel",
        "in_channels": args.in_channels,
        "patch_size": args.patch_size,
        "hidden_size": args.hidden_size,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "context_dim": args.context_dim,
        "qkv_bias": args.qkv_bias,
        "trainable_rms": args.trainable_rms,
        "use_rope": True,
        "rope_base": args.rope_base,
        "norm_eps": args.norm_eps,
        "init_alpha": args.init_alpha,
        "init_beta": args.init_beta,
    }

    state_dict = _load_state_dict(args.checkpoint)
    model = MVSplitDiTTransformer2DModel(**{k: v for k, v in transformer_config.items() if not k.startswith("_")})
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        _print_state_dict_mismatch(missing_keys, unexpected_keys)
        if args.check_load:
            raise SystemExit(1)
    model.save_pretrained(transformer_dir, safe_serialization=args.safe_serialization)

    scheduler = MVSplitFlowMatchScheduler(
        mode="ode",
        num_train_timesteps=1000,
        time_shift_alpha=args.time_shift_alpha,
    )
    scheduler.save_pretrained(scheduler_dir)

    if args.check_load:
        reloaded_model = MVSplitDiTTransformer2DModel.from_pretrained(transformer_dir)
        missing_keys, unexpected_keys = reloaded_model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            _print_state_dict_mismatch(missing_keys, unexpected_keys)
            raise SystemExit(1)

    if args.copy_vae is not None:
        vae_path = output_dir / "vae"
        if vae_path.exists():
            shutil.rmtree(vae_path)
        shutil.copytree(args.copy_vae, vae_path)
    if args.text_encoder:
        with open(output_dir / "text_encoder_pretrained_model_name_or_path.txt", "w", encoding="utf-8") as f:
            f.write(args.text_encoder + os.linesep)
    _write_model_index(output_dir, include_vae=args.copy_vae is not None, include_text=bool(args.text_encoder))
    print(f"Saved Diffusers-style MVSplit DiT pipeline directory to: {output_dir}")


if __name__ == "__main__":
    main()
