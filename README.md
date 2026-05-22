# MVSplit-DiT Diffusers Refactor

This repository now follows a native Diffusers-compatible layout under:

```text
src/diffusers/
  models/transformers/transformer_mvsplit_dit.py
  schedulers/__init__.py
  pipelines/mvsplit/pipeline_mvsplit_dit.py
```

Legacy standalone source files were removed in favor of this structure.

## What changed

- Replaced script-style model code with a Diffusers-native transformer:
  `MVSplitDiTTransformer2DModel`.
- Uses Diffusers' flow-matching scheduler:
  `FlowMatchEulerDiscreteScheduler`.
- Added a text-conditional pipeline:
  `MVSplitDiTPipeline`.
- Added a conversion script that exports checkpoints in Diffusers directory format:
  `scripts/convert_mvsplit_to_diffusers.py`.

## Convert an existing checkpoint

```bash
python3 scripts/convert_mvsplit_to_diffusers.py \
  --checkpoint /path/to/model.pt \
  --output /path/to/mvsplit-diffusers \
  --depth 1000 \
  --hidden-size 1024 \
  --num-heads 8 \
  --num-kv-heads 8
```

This creates:

```text
/path/to/mvsplit-diffusers/
  model_index.json
  transformer/config.json
  transformer/diffusion_pytorch_model.safetensors (or .bin)
  scheduler/scheduler_config.json
  text_encoder_pretrained_model_name_or_path.txt
```

If you want to bundle a local VAE directory, pass `--copy-vae /path/to/vae`.

## Python usage

```python
from diffusers import MVSplitDiTTransformer2DModel, MVSplitDiTPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import AutoModel, AutoTokenizer

transformer = MVSplitDiTTransformer2DModel(...)
scheduler = FlowMatchEulerDiscreteScheduler(shift=4.0)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
text_encoder = AutoModel.from_pretrained("Qwen/Qwen3-0.6B")

pipe = MVSplitDiTPipeline(
    transformer=transformer,
    scheduler=scheduler,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    vae=None,  # or a compatible VAE module
)

result = pipe(prompt="a red panda climbing a bamboo stalk", output_type="latent")
latents = result.images
```

## Tests

```bash
PYTHONPATH=src python3 -m pytest tests/test_mvsplit_diffusers.py
```

## Citation

Paper: <https://arxiv.org/abs/2605.06169>
