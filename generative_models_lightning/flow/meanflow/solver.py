# Adapted from https://github.com/haidog-yaqub/MeanFlow
# Commit: 18b67b25a3af86d199005023bb190024d28233e7
# Original license: MIT.

from __future__ import annotations

from math import ceil
from typing import Callable

import torch
from torch import Tensor, nn

from ..solver import Solver


class MeanFlowEulerSolver(Solver):
    """Euler solver for a mean-velocity model parameterized by ``(x, t, r)``."""

    def __init__(self, velocity_model: nn.Module | Callable[..., Tensor]):
        super().__init__()
        self.velocity_model = velocity_model

    @staticmethod
    def _discretize(time_grid: Tensor, step_size: float | None) -> Tensor:
        if time_grid.ndim != 1 or time_grid.numel() < 2:
            raise ValueError("time_grid must be one-dimensional with at least two points.")
        if not torch.all(time_grid[:-1] > time_grid[1:]):
            raise ValueError("MeanFlow requires a strictly descending time_grid.")
        if step_size is None:
            return time_grid
        if step_size <= 0:
            raise ValueError("step_size must be positive when provided.")

        segments = [time_grid[:1]]
        for start, end in zip(time_grid[:-1], time_grid[1:]):
            interval = float((start - end).item())
            num_steps = max(1, ceil(interval / float(step_size)))
            offsets = torch.arange(
                1,
                num_steps,
                device=time_grid.device,
                dtype=time_grid.dtype,
            ) * float(step_size)
            interior = start - offsets
            tolerance = torch.finfo(time_grid.dtype).eps * 8
            interior = interior[interior > end + tolerance]
            segments.append(torch.cat((interior, end.reshape(1))))
        return torch.cat(segments)

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        step_size: float | None = None,
        time_grid: Tensor = torch.tensor([1.0, 0.0]),
        return_intermediates: bool = False,
        **model_extras,
    ) -> Tensor:
        if not time_grid.dtype.is_floating_point:
            time_grid = time_grid.float()
        time_grid = time_grid.to(device=x_init.device)
        times = self._discretize(time_grid, step_size)

        x_t = x_init.clone()
        intermediates = [x_t.clone()] if return_intermediates else None

        for t, r in zip(times[:-1], times[1:]):
            t_batch = t.expand(x_t.shape[0])
            r_batch = r.expand(x_t.shape[0])
            velocity = self.velocity_model(
                x_t,
                t_batch,
                r_batch,
                **model_extras,
            )
            x_t = x_t - (t - r) * velocity
            if intermediates is not None:
                intermediates.append(x_t.clone())

        if intermediates is not None:
            return torch.stack(intermediates, dim=0)
        return x_t
