from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
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
class MVSplitFlowMatchSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class MVSplitFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(
        self,
        mode: str = "ode",
        num_train_timesteps: int = 1000,
        time_shift_alpha: float = 4.0,
    ):
        if mode != "ode":
            raise ValueError("MVSplitFlowMatchScheduler currently supports only mode='ode'.")
        self.mode = mode
        self.num_train_timesteps = num_train_timesteps
        self.time_shift_alpha = time_shift_alpha
        self.timesteps = torch.from_numpy(np.linspace(1.0, 0.0, num_train_timesteps + 1)).to(dtype=torch.float64)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[Union[torch.device, str]] = None,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        mode = mode or self.mode
        if mode != "ode":
            raise ValueError("MVSplitFlowMatchScheduler currently supports only mode='ode'.")
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float64)
        self.timesteps = timesteps.to(device=device)
        return self.timesteps

    def _time_shift(self, timestep: torch.Tensor) -> torch.Tensor:
        alpha = self.time_shift_alpha
        return timestep * alpha / (1.0 + (alpha - 1.0) * timestep)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        next_timestep: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[MVSplitFlowMatchSchedulerOutput, Tuple[torch.Tensor]]:
        del generator
        sample_dtype = sample.dtype

        sample = sample.to(dtype=torch.float64)
        model_output = model_output.to(dtype=torch.float64)
        timestep = timestep.to(device=sample.device, dtype=torch.float64).flatten()
        next_timestep = next_timestep.to(device=sample.device, dtype=torch.float64).flatten()

        delta = self._time_shift(timestep[0]) - self._time_shift(next_timestep[0])
        prev_sample = sample + delta * model_output
        prev_sample = prev_sample.to(sample_dtype)

        if not return_dict:
            return (prev_sample,)
        return MVSplitFlowMatchSchedulerOutput(prev_sample=prev_sample)
