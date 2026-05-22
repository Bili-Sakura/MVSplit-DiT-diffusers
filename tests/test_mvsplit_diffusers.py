import types

import pytest

torch = pytest.importorskip("torch")

from diffusers.models.transformers import MVSplitDiTTransformer2DModel
from diffusers.pipelines.mvsplit import MVSplitDiTPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


def test_transformer_forward_shape():
    model = MVSplitDiTTransformer2DModel(
        in_channels=4,
        patch_size=1,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_kv_heads=4,
        mlp_hidden_dim=64,
        context_dim=16,
    )
    latents = torch.randn(2, 4, 8, 8)
    text = torch.randn(2, 12, 16)
    output = model(latents, encoder_hidden_states=text).sample
    assert output.shape == latents.shape


def test_scheduler_step_matches_time_shifted_euler():
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(2)
    sample = torch.ones(1, 2, 2, 2)
    velocity = torch.full_like(sample, 0.5)
    timestep = scheduler.timesteps[0]
    output = scheduler.step(velocity, timestep, sample).prev_sample
    assert output.shape == sample.shape


class _DummyTokenizer:
    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        del padding, truncation, return_tensors
        batch = len(texts)
        token_ids = torch.ones(batch, max_length, dtype=torch.long)
        mask = torch.ones(batch, max_length, dtype=torch.long)
        return {"input_ids": token_ids, "attention_mask": mask}


class _DummyTextEncoder(torch.nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.embed = torch.nn.Embedding(4, hidden_size)

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        return types.SimpleNamespace(last_hidden_state=self.embed(input_ids % 4))


def test_pipeline_latent_output_smoke():
    transformer = MVSplitDiTTransformer2DModel(
        in_channels=4,
        patch_size=1,
        hidden_size=32,
        depth=1,
        num_heads=4,
        num_kv_heads=4,
        mlp_hidden_dim=64,
        context_dim=16,
    )
    pipe = MVSplitDiTPipeline(
        transformer=transformer,
        vae=None,
        text_encoder=_DummyTextEncoder(hidden_size=16),
        tokenizer=_DummyTokenizer(),
        max_length=8,
    )
    result = pipe(
        prompt=["hello", "world"],
        negative_prompt=["", ""],
        height=16,
        width=16,
        num_inference_steps=3,
        guidance_scale=1.5,
        output_type="latent",
    )
    assert result.images.shape == (2, 4, 16, 16)
