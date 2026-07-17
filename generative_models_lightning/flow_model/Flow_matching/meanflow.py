# Adapted from https://github.com/haidog-yaqub/MeanFlow
# Commit: 18b67b25a3af86d199005023bb190024d28233e7
# Original license: MIT. See ../meanflow/LICENSE.

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from .loss import MeanFlowLoss
from .path import MeanFlowProbPath
from .solver import MeanFlowEulerSolver
from .utils import Normalizer


class MeanFlow:
    """Compatibility facade composed from MeanFlow path, loss, and solver.

    New code may instantiate the three components independently.  This facade
    preserves the original ``loss`` and ``sample_each_class`` API and adds a
    condition-agnostic ``sample`` method for dense conditioning.
    """

    def __init__(
        self,
        channels: int = 1,
        image_size: int | Sequence[int] = 32,
        num_classes: int | None = 10,
        normalizer=("minmax", None, None),
        flow_ratio: float = 0.50,
        time_dist=("lognorm", -0.4, 1.0),
        cfg_ratio: float = 0.10,
        cfg_scale: float | None = 2.0,
        cfg_uncond: str = "v",
        jvp_api: str = "autograd",
        valid_data_min: float | None = None,
        path: MeanFlowProbPath | None = None,
        objective: MeanFlowLoss | None = None,
    ) -> None:
        if channels < 1:
            raise ValueError("channels must be positive.")

        self.channels = channels
        self.image_size = image_size
        self.image_shape = (
            (int(image_size), int(image_size))
            if isinstance(image_size, int)
            else tuple(int(size) for size in image_size)
        )
        if not self.image_shape or any(size < 1 for size in self.image_shape):
            raise ValueError("image_size must contain positive dimensions.")

        self.num_classes = num_classes
        self.use_cond = num_classes is not None
        self.normer = Normalizer.from_list(normalizer)
        self.normalizer = self.normer
        self.valid_data_min = valid_data_min

        self.path = path or MeanFlowProbPath(
            flow_ratio=flow_ratio,
            time_dist=time_dist,
        )
        self.objective = objective or MeanFlowLoss(
            num_classes=num_classes,
            cfg_ratio=cfg_ratio,
            cfg_scale=cfg_scale,
            cfg_uncond=cfg_uncond,
            jvp_api=jvp_api,
        )

        # Compatibility attributes used by existing configs and scripts.
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        self.w = cfg_scale
        self.cfg_uncond = cfg_uncond
        self.jvp_api = jvp_api

    def sample_t_r(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.path.sample_t_r(batch_size, torch.device(device))

    def loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if valid_mask is None and self.valid_data_min is not None:
            valid_mask = (x > self.valid_data_min).to(dtype=x.dtype)

        x_data = self.normer.norm(x)
        x_noise = torch.randn_like(x_data)
        t, r = self.sample_t_r(x.shape[0], x.device)
        path_sample = self.path.sample(x_0=x_noise, x_1=x_data, t=t)
        return self.objective(
            model,
            path_sample,
            r,
            condition=c,
            valid_mask=valid_mask,
        )

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        cond: torch.Tensor | None = None,
        batch_size: int = 4,
        sample_steps: int = 5,
        device: torch.device | str | None = None,
        *,
        x_init: torch.Tensor | None = None,
        return_intermediates: bool = False,
        **model_extras: Any,
    ) -> torch.Tensor:
        if sample_steps < 1:
            raise ValueError("sample_steps must be positive.")

        if device is None:
            device = next(model.parameters()).device
        device = torch.device(device)

        if cond is not None:
            cond = cond.to(device)
            batch_size = int(cond.shape[0])
        if x_init is None:
            x_init = torch.randn(
                batch_size,
                self.channels,
                *self.image_shape,
                device=device,
            )
        else:
            x_init = x_init.to(device)
            batch_size = int(x_init.shape[0])
            if cond is not None and cond.shape[0] != batch_size:
                raise ValueError("Condition and x_init batch sizes must match.")

        if cond is not None:
            model_extras.setdefault("y", cond)

        solver = MeanFlowEulerSolver(velocity_model=model)
        time_grid = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)
        result = solver.sample(
            x_init=x_init,
            step_size=None,
            time_grid=time_grid,
            return_intermediates=return_intermediates,
            **model_extras,
        )

        if return_intermediates:
            return torch.stack([self.normer.unnorm(state) for state in result])
        return self.normer.unnorm(result)

    @torch.no_grad()
    def sample_each_class(
        self,
        model: nn.Module,
        n_per_class: int,
        classes: Sequence[int] | None = None,
        sample_steps: int = 5,
        device: torch.device | str = "cuda",
    ) -> torch.Tensor:
        if self.num_classes is None:
            raise ValueError("num_classes is required for class-conditional sampling.")
        model.eval()
        if classes is None:
            condition = torch.arange(self.num_classes, device=device).repeat(n_per_class)
        else:
            condition = torch.as_tensor(classes, device=device).repeat(n_per_class)
        return self.sample(
            model=model,
            cond=condition,
            batch_size=int(condition.shape[0]),
            sample_steps=sample_steps,
            device=device,
        )


__all__ = ["MeanFlow"]
