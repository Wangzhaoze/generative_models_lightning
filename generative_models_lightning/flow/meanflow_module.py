"""MeanFlow Lightning module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .base_flow_module import BaseFlowModule
from .meanflow import MeanFlowEulerSolver, MeanFlowLoss, MeanFlowProbPath, Normalizer


class MeanFlowModule(BaseFlowModule):
    """Train and sample a MeanFlow model under the shared flow base."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        sample_shape: Sequence[int] | None = None,
        normalizer: Sequence[Any] = ("mean_std", 0.0, 1.0),
        flow_ratio: float = 0.5,
        time_dist: Sequence[Any] = ("lognorm", -0.4, 1.0),
        cfg_ratio: float = 0.0,
        cfg_scale: float | None = None,
        cfg_uncond: str = "v",
        jvp_api: str = "autograd",
        gamma: float = 0.5,
        c: float = 1e-3,
        valid_data_min: float | None = None,
        default_sample_steps: int = 5,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        **kwargs: Any,
    ) -> None:
        if default_sample_steps < 1:
            raise ValueError("default_sample_steps must be positive.")

        super().__init__(
            model=model,
            sample_shape=sample_shape,
            lr=lr,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            **kwargs,
        )
        self.normalizer = Normalizer.from_list(normalizer)
        self.path = MeanFlowProbPath(
            flow_ratio=flow_ratio,
            time_dist=time_dist,
        )
        self.objective = MeanFlowLoss(
            num_classes=None,
            cfg_ratio=cfg_ratio,
            cfg_scale=cfg_scale,
            cfg_uncond=cfg_uncond,
            jvp_api=jvp_api,
            gamma=gamma,
            c=c,
        )
        self.valid_data_min = valid_data_min
        self.default_sample_steps = int(default_sample_steps)

    def sample_time_pair(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.path.sample_t_r(batch_size=batch_size, device=device)

    def compute_loss_terms(
        self,
        x: torch.Tensor,
        cond: Any | None,
    ) -> dict[str, torch.Tensor]:
        cond_tensor = cond if isinstance(cond, torch.Tensor) else None
        valid_mask = None
        if self.valid_data_min is not None:
            valid_mask = (x > self.valid_data_min).to(dtype=x.dtype)

        x_data = self.normalizer.norm(x)
        x_noise = self.sample_source(x_data)
        t, r = self.sample_time_pair(batch_size=x.shape[0], device=x.device)
        path_sample = self.path.sample(x_0=x_noise, x_1=x_data, t=t)
        loss, mse = self.objective(
            self.model,
            path_sample,
            r,
            condition=cond_tensor,
            valid_mask=valid_mask,
        )
        return {
            "loss": torch.atleast_1d(loss),
            "mse": torch.atleast_1d(mse),
        }

    @torch.inference_mode()
    def sample(
        self,
        shape: Sequence[int],
        cond: Any | None = None,
        *,
        sample_steps: int | None = None,
        time_grid: torch.Tensor | None = None,
        step_size: float | None = None,
        x_init: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        if sample_steps is None:
            sample_steps = self.default_sample_steps
        if sample_steps < 1:
            raise ValueError("sample_steps must be positive.")

        cond_tensor = cond if isinstance(cond, torch.Tensor) else None
        if cond_tensor is not None:
            cond_tensor = cond_tensor.to(device=self.device)
            if cond_tensor.shape[0] != shape[0]:
                raise ValueError("Condition batch size must match the requested shape.")

        if x_init is None:
            x_init = torch.randn(*shape, device=self.device)
        else:
            x_init = x_init.to(device=self.device)

        if time_grid is None:
            time_grid = torch.linspace(
                1.0,
                0.0,
                sample_steps + 1,
                device=self.device,
            )

        solver = MeanFlowEulerSolver(velocity_model=self.model)
        model_extras = {"y": cond_tensor} if cond_tensor is not None else {}
        result = solver.sample(
            x_init=x_init,
            step_size=step_size,
            time_grid=time_grid,
            return_intermediates=return_intermediates,
            **model_extras,
        )

        if return_intermediates:
            return torch.stack(
                [self.normalizer.unnorm(state) for state in result],
                dim=0,
            )
        return self.normalizer.unnorm(result)
