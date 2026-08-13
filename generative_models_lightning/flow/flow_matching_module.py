"""Continuous Flow Matching Lightning module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from .base_flow_module import BaseFlowModule
from .path import AffineProbPath
from .path.scheduler import CosineScheduler
from .solver import ODESolver
from .utils import ModelWrapper


class _FlowMatchingVelocityModel(ModelWrapper):
    """Adapt a repository backbone to the ODE solver's ``(x, t)`` interface."""

    def __init__(
        self,
        model: torch.nn.Module,
        time_scale: float,
    ) -> None:
        super().__init__(model=model)
        self.time_scale = float(time_scale)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        cond: Tensor | None = None,
        **extras: Any,
    ) -> Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        elif t.ndim == 1 and t.shape[0] == 1:
            t = t.expand(x.shape[0])
        elif t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(
                "FlowMatching solver times must be scalar or shape [batch_size]."
            )

        model_time = t * self.time_scale
        kwargs = dict(extras)
        if cond is not None:
            kwargs["y"] = cond

        try:
            output = self.model(x, model_time, **kwargs)
        except TypeError:
            kwargs.pop("y", None)
            output = self.model(x, model_time, **kwargs)

        if output.shape[0] != x.shape[0] or output.ndim != x.ndim:
            raise ValueError("Velocity model output must match the input batch shape.")
        if output.shape[1] < x.shape[1]:
            raise ValueError(
                "Velocity model output channels must be at least the data channels."
            )
        return output[:, : x.shape[1]]


class FlowMatchingModule(BaseFlowModule):
    """Train and sample a continuous Flow Matching model."""

    def __init__(
        self,
        model: torch.nn.Module,
        path: AffineProbPath | None = None,
        *,
        scheduler: Any | None = None,
        sample_shape: Sequence[int] | None = None,
        drop_rate: float = 0.0,
        time_sampling: str = "uniform",
        lognorm_mean: float = 0.0,
        lognorm_std: float = 1.0,
        time_scale: float = 999.0,
        ode_method: str = "midpoint",
        solver_step_size: float | None = None,
        solver_atol: float = 1e-5,
        solver_rtol: float = 1e-5,
        default_sample_steps: int = 50,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        **kwargs: Any,
    ) -> None:
        if path is not None and scheduler is not None:
            raise ValueError("Provide either path or scheduler, not both.")
        if not 0.0 <= drop_rate <= 1.0:
            raise ValueError("drop_rate must be in [0, 1].")
        if time_sampling not in {"uniform", "lognorm"}:
            raise ValueError("time_sampling must be 'uniform' or 'lognorm'.")
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
        self.path = path or AffineProbPath(scheduler or CosineScheduler())
        self.drop_rate = float(drop_rate)
        self.time_sampling = time_sampling
        self.lognorm_mean = float(lognorm_mean)
        self.lognorm_std = float(lognorm_std)
        self.time_scale = float(time_scale)
        self.ode_method = ode_method
        self.solver_step_size = solver_step_size
        self.solver_atol = float(solver_atol)
        self.solver_rtol = float(solver_rtol)
        self.default_sample_steps = int(default_sample_steps)

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.time_sampling == "uniform":
            return super().sample_time(batch_size=batch_size, device=device)
        samples = torch.randn(batch_size, device=device)
        samples = samples * self.lognorm_std + self.lognorm_mean
        return torch.sigmoid(samples)

    def _predict_velocity(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: Tensor | None,
    ) -> Tensor:
        return _FlowMatchingVelocityModel(
            model=self.model,
            time_scale=self.time_scale,
        )(x=x_t, t=t, cond=cond)

    def compute_loss_terms(
        self,
        x: torch.Tensor,
        cond: Any | None,
    ) -> dict[str, torch.Tensor]:
        cond_tensor = cond if isinstance(cond, torch.Tensor) else None
        cond_tensor = self.apply_condition_dropout(cond_tensor, self.drop_rate)

        x_source = self.sample_source(x)
        t = self.sample_time(batch_size=x.shape[0], device=x.device)
        path_sample = self.path.sample(x_0=x_source, x_1=x, t=t)
        prediction = self._predict_velocity(path_sample.x_t, t, cond_tensor)
        mse = (prediction - path_sample.dx_t).reshape(x.shape[0], -1).pow(2).mean(dim=1)
        return {"loss": mse, "mse": mse}

    @torch.inference_mode()
    def sample(
        self,
        shape: Sequence[int],
        cond: Any | None = None,
        *,
        sample_steps: int | None = None,
        time_grid: torch.Tensor | None = None,
        method: str | None = None,
        step_size: float | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        x_init: torch.Tensor | None = None,
        return_intermediates: bool = False,
        enable_grad: bool = False,
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
                0.0,
                1.0,
                sample_steps + 1,
                device=self.device,
            )

        solver = ODESolver(
            velocity_model=_FlowMatchingVelocityModel(
                model=self.model,
                time_scale=self.time_scale,
            )
        )
        return solver.sample(
            x_init=x_init,
            step_size=self.solver_step_size if step_size is None else step_size,
            method=self.ode_method if method is None else method,
            atol=self.solver_atol if atol is None else atol,
            rtol=self.solver_rtol if rtol is None else rtol,
            time_grid=time_grid,
            return_intermediates=return_intermediates,
            enable_grad=enable_grad,
            cond=cond_tensor,
        )
