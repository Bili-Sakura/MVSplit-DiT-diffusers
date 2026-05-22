from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchEulerDiscreteSchedulerOutput,
)


class MVSplitFlowMatchScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(
        self,
        mode: str = "ode",
        num_train_timesteps: int = 1000,
        time_shift_alpha: float = 4.0,
        **kwargs,
    ):
        if mode != "ode":
            raise ValueError("MVSplitFlowMatchScheduler currently supports only mode='ode'.")
        shift = kwargs.pop("shift", time_shift_alpha)
        super().__init__(num_train_timesteps=num_train_timesteps, shift=shift, **kwargs)


MVSplitFlowMatchSchedulerOutput = FlowMatchEulerDiscreteSchedulerOutput
