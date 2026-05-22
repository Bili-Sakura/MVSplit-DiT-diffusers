from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.schedulers.scheduling_utils import SchedulerMixin
    from diffusers.utils import BaseOutput
except Exception:
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class ConfigMixin:
        config_name = "scheduler_config.json"

    class SchedulerMixin:
        pass

    def register_to_config(init):
        return init


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    config_name = "scheduler_config.json"
    order = 1
    init_noise_sigma = 1.0

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.timesteps = torch.linspace(1.0, 0.0, num_train_timesteps, dtype=torch.float64)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[Union[torch.device, str]] = None,
    ) -> torch.Tensor:
        self.timesteps = torch.linspace(1.0, 0.0, num_inference_steps, dtype=torch.float64, device=device)
        return self.timesteps

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        del timestep
        return sample

    def _time_shift(self, timestep: torch.Tensor) -> torch.Tensor:
        shift = self.shift
        if shift == 1.0:
            return timestep
        return timestep * shift / (1.0 + (shift - 1.0) * timestep)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[torch.Tensor, float],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple[torch.Tensor]]:
        del generator

        sample_dtype = sample.dtype
        sample = sample.to(dtype=torch.float64)
        model_output = model_output.to(dtype=torch.float64)

        timestep = torch.as_tensor(timestep, device=sample.device, dtype=torch.float64).flatten()[0]
        scheduler_timesteps = self.timesteps.to(device=sample.device, dtype=torch.float64)
        step_index = int(torch.argmin(torch.abs(scheduler_timesteps - timestep)).item())

        if step_index + 1 < len(scheduler_timesteps):
            next_timestep = scheduler_timesteps[step_index + 1]
        else:
            next_timestep = torch.tensor(0.0, device=sample.device, dtype=torch.float64)

        delta = self._time_shift(timestep) - self._time_shift(next_timestep)
        prev_sample = (sample + delta * model_output).to(sample_dtype)

        if not return_dict:
            return (prev_sample,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
