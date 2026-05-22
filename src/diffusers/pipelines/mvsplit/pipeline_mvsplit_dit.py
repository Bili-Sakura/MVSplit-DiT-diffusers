from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

try:
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    from diffusers.utils import BaseOutput
except Exception:
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class DiffusionPipeline:
        def register_modules(self, **kwargs):
            for name, module in kwargs.items():
                setattr(self, name, module)

        @property
        def _execution_device(self):
            return torch.device("cpu")

        def maybe_free_model_hooks(self):
            pass

    class VaeImageProcessor:
        def postprocess(self, image, output_type="pil"):
            return image


@dataclass
class MVSplitDiTPipelineOutput(BaseOutput):
    images: Union[torch.FloatTensor, List]


class MVSplitDiTPipeline(DiffusionPipeline):
    """
    Text-to-latent/image pipeline for MVSplit DiT.

    The pipeline uses the standard Diffusers scheduler interface and expects an
    explicit scheduler module (for example
    `FlowMatchEulerDiscreteScheduler(shift=4.0)`).
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = ["vae", "text_encoder", "tokenizer"]

    def __init__(
        self,
        transformer,
        scheduler,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        max_length: int = 256,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.max_length = max_length
        self.image_processor = VaeImageProcessor()

    def _get_vae_downsample_factor(self) -> int:
        if self.vae is None:
            return 1
        config = getattr(self.vae, "config", None)
        if config is not None and getattr(config, "block_out_channels", None):
            return 2 ** (len(config.block_out_channels) - 1)
        return 16

    def _prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    ) -> torch.Tensor:
        downsample_factor = self._get_vae_downsample_factor()
        if height % downsample_factor != 0 or width % downsample_factor != 0:
            raise ValueError(f"height and width must be divisible by VAE downsample factor {downsample_factor}.")

        latent_height = height // downsample_factor
        latent_width = width // downsample_factor
        latent_shape = (batch_size, self.transformer.config.in_channels, latent_height, latent_width)
        return torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)

    def _encode_text(self, text: Union[str, List[str]], device: torch.device) -> torch.Tensor:
        if self.tokenizer is None or self.text_encoder is None:
            raise ValueError("Both tokenizer and text_encoder must be provided for text-to-image inference.")

        if isinstance(text, str):
            text = [text]

        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device) if "attention_mask" in tokens else None

        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        return outputs

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return latents
        scaling_factor = getattr(getattr(self.vae, "config", None), "scaling_factor", 1.0)
        latents = latents / scaling_factor
        decoded = self.vae.decode(latents)
        return decoded.sample if hasattr(decoded, "sample") else decoded

    @staticmethod
    def _apply_cfg(model_output: torch.Tensor, guidance_scale: float) -> torch.Tensor:
        if guidance_scale <= 1.0:
            return model_output
        model_output_cond, model_output_uncond = model_output.chunk(2)
        return model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 35,
        guidance_scale: float = 2.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[MVSplitDiTPipelineOutput, Tuple]:
        """Run denoising with the configured scheduler and decode the output."""
        device = self._execution_device
        model_dtype = next(self.transformer.parameters()).dtype

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        prompt_embeds = self._encode_text(prompt, device=device).to(dtype=model_dtype)
        do_cfg = guidance_scale > 1.0
        if do_cfg:
            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            elif len(negative_prompt) != batch_size:
                raise ValueError("negative_prompt must have the same batch size as prompt.")

            negative_prompt_embeds = self._encode_text(negative_prompt, device=device).to(dtype=model_dtype)
            prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)

        latents = self._prepare_latents(
            batch_size=batch_size,
            height=height,
            width=width,
            dtype=model_dtype,
            device=device,
            generator=generator,
        )
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        for timestep in timesteps:
            if do_cfg:
                model_input = torch.cat([latents, latents], dim=0)
            else:
                model_input = latents

            timestep_batch = torch.full(
                (model_input.shape[0],),
                float(timestep),
                device=device,
                dtype=model_dtype,
            )
            model_output = self.transformer(
                model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep_batch,
                return_dict=True,
            ).sample
            model_output = self._apply_cfg(model_output, guidance_scale=guidance_scale)
            latents = self.scheduler.step(
                model_output=model_output,
                timestep=timestep,
                sample=latents,
                generator=generator,
                return_dict=True,
            ).prev_sample

        if output_type == "latent":
            image = latents
        else:
            image = self._decode_latents(latents)
            if self.vae is not None:
                image = (image / 2 + 0.5).clamp(0, 1)
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return MVSplitDiTPipelineOutput(images=image)
