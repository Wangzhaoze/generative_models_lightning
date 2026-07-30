# Adapted from https://github.com/haidog-yaqub/MeanFlow
# Commit: 18b67b25a3af86d199005023bb190024d28233e7
# Copyright (c) 2025 Jiarui Hai
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss

from ..path.path_sample import PathSample
from ..utils import expand_tensor_like


def stop_gradient(x: Tensor) -> Tensor:
    """Return ``x`` without a gradient connection to its source."""
    return x.detach()


def stopgrad(x: Tensor) -> Tensor:
    """Backward-compatible alias for :func:`stop_gradient`."""
    return stop_gradient(x)


def _per_example_mean_square(
    error: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    if error.ndim == 0:
        error = error.unsqueeze(0)

    error_sq = error.square()
    if valid_mask is None:
        return error_sq.reshape(error_sq.shape[0], -1).mean(dim=1)

    mask = valid_mask.to(device=error.device, dtype=error.dtype)
    if mask.ndim == 1:
        mask = expand_tensor_like(input_tensor=mask, expand_to=error)
    else:
        while mask.ndim < error.ndim:
            mask = mask.unsqueeze(1)
        try:
            mask = torch.broadcast_to(mask, error.shape)
        except RuntimeError as exc:
            raise ValueError(
                "valid_mask must be broadcastable to the model output; "
                f"got {tuple(valid_mask.shape)} and {tuple(error.shape)}."
            ) from exc

    mask = mask.to(dtype=error_sq.dtype)
    error_sum = (error_sq * mask).reshape(error_sq.shape[0], -1).sum(dim=1)
    valid_count = mask.reshape(mask.shape[0], -1).sum(dim=1).clamp_min(1.0)
    return error_sum / valid_count


def adaptive_l2_loss(
    error: Tensor,
    gamma: float = 0.5,
    c: float = 1e-3,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Compute MeanFlow's adaptive per-example L2 objective."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}.")
    if c <= 0.0:
        raise ValueError(f"c must be positive, got {c}.")

    delta_sq = _per_example_mean_square(error, valid_mask=valid_mask)
    weight = 1.0 / (delta_sq + c).pow(1.0 - gamma)
    return (stop_gradient(weight) * delta_sq).mean()


class MeanFlowLoss(_Loss):
    """MeanFlow objective using a directional derivative of the model."""

    def __init__(
        self,
        num_classes: int | None = None,
        cfg_ratio: float = 0.1,
        cfg_scale: float | None = 2.0,
        cfg_uncond: str = "v",
        jvp_api: str = "autograd",
        gamma: float = 0.5,
        c: float = 1e-3,
    ) -> None:
        super().__init__(reduction="mean")
        if not 0.0 <= cfg_ratio <= 1.0:
            raise ValueError(f"cfg_ratio must be in [0, 1], got {cfg_ratio}.")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}.")
        if c <= 0.0:
            raise ValueError(f"c must be positive, got {c}.")

        jvp_aliases = {
            "autograd": "autograd",
            "funtorch": "torch.func",
            "functorch": "torch.func",
            "torch.func": "torch.func",
        }
        if jvp_api not in jvp_aliases:
            choices = "'autograd', 'funtorch', 'functorch', or 'torch.func'"
            raise ValueError(f"jvp_api must be {choices}; got {jvp_api!r}.")

        self.num_classes = num_classes
        self.cfg_ratio = cfg_ratio
        self.cfg_scale = cfg_scale
        self.cfg_uncond = cfg_uncond
        self.jvp_api = jvp_api
        self._jvp_backend = jvp_aliases[jvp_api]
        self.gamma = gamma
        self.c = c

    @staticmethod
    def _call_model(
        model: nn.Module | Callable[..., Tensor],
        x: Tensor,
        t: Tensor,
        r: Tensor,
        condition: Tensor | None,
    ) -> Tensor:
        if condition is None:
            return model(x, t, r)
        return model(x, t, r, y=condition)

    def _unconditional_condition(self, condition: Tensor) -> Tensor:
        if self.num_classes is not None and condition.ndim == 1:
            return torch.full_like(condition, self.num_classes)
        return torch.zeros_like(condition)

    def _cfg_mask(self, condition: Tensor) -> Tensor:
        batch_size = condition.shape[0]
        if self.cfg_ratio == 0.0:
            return torch.zeros(batch_size, dtype=torch.bool, device=condition.device)
        if self.cfg_ratio == 1.0:
            return torch.ones(batch_size, dtype=torch.bool, device=condition.device)
        return torch.rand(batch_size, device=condition.device) < self.cfg_ratio

    def forward(
        self,
        model: nn.Module | Callable[..., Tensor],
        path_sample: PathSample,
        r: Tensor,
        condition: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        x_t = path_sample.x_t
        velocity = path_sample.dx_t
        t = path_sample.t

        if t.ndim != 1 or r.ndim != 1:
            raise ValueError(
                "MeanFlow times t and r must have shape [batch_size]; "
                f"got {tuple(t.shape)} and {tuple(r.shape)}."
            )
        if t.shape != r.shape or t.shape[0] != x_t.shape[0]:
            raise ValueError(
                "MeanFlow times must have equal shape and match the sample batch."
            )
        if velocity.shape != x_t.shape:
            raise ValueError(
                "path_sample.dx_t must have the same shape as path_sample.x_t."
            )
        if condition is not None and condition.shape[0] != x_t.shape[0]:
            raise ValueError("Condition batch size must match the path sample batch size.")

        condition_for_model = condition
        velocity_target = velocity

        if condition is not None:
            unconditional = self._unconditional_condition(condition)
            cfg_mask = self._cfg_mask(condition)
            condition_mask = expand_tensor_like(input_tensor=cfg_mask, expand_to=condition)
            condition_for_model = torch.where(condition_mask, unconditional, condition)

            if self.cfg_scale is not None:
                with torch.no_grad():
                    unconditional_velocity = self._call_model(
                        model=model,
                        x=x_t,
                        t=t,
                        r=t,
                        condition=unconditional,
                    )
                velocity_target = (
                    self.cfg_scale * velocity
                    + (1.0 - self.cfg_scale) * unconditional_velocity
                )

                if self.cfg_uncond == "v":
                    velocity_mask = expand_tensor_like(
                        input_tensor=cfg_mask,
                        expand_to=velocity_target,
                    )
                    velocity_target = torch.where(
                        velocity_mask,
                        velocity,
                        velocity_target,
                    )

        def model_fn(z: Tensor, upper_t: Tensor, lower_r: Tensor) -> Tensor:
            return self._call_model(
                model=model,
                x=z,
                t=upper_t,
                r=lower_r,
                condition=condition_for_model,
            )

        primals = (x_t, t, r)
        tangents = (
            velocity_target,
            torch.ones_like(t),
            torch.zeros_like(r),
        )
        if self._jvp_backend == "autograd":
            prediction, derivative = torch.autograd.functional.jvp(
                model_fn,
                primals,
                tangents,
                create_graph=True,
            )
        else:
            prediction, derivative = torch.func.jvp(model_fn, primals, tangents)

        delta_time = expand_tensor_like(input_tensor=t - r, expand_to=derivative)
        target = velocity_target - delta_time * derivative
        error = prediction - stop_gradient(target)

        loss = adaptive_l2_loss(
            error,
            gamma=self.gamma,
            c=self.c,
            valid_mask=valid_mask,
        )
        mse = _per_example_mean_square(
            stop_gradient(error),
            valid_mask=valid_mask,
        ).mean()
        return loss, mse
