# Adapted for generative_models_lightning Mean Flow training.

from __future__ import annotations

import torch
import torch.nn as nn


class MFUNet(nn.Module):
    """Adapt the repository's SPADE UNet to Mean Flow's `(x, t, r, y)` interface.

    `t` is passed through the UNet timestep embedding as a continuous value, while
    `r` and `dt=t-r` are injected as extra dense conditioning channels so the
    model remains differentiable with respect to both time variables.
    """

    def __init__(
        self,
        unet: nn.Module,
        in_channels: int,
        cond_in_channels: int | None = None,
        cond_proj_channels: int | None = None,
        time_scale: float = 999.0,
    ):
        super().__init__()
        self.unet = unet
        self.in_channels = in_channels
        self.cond_in_channels = cond_in_channels
        self.cond_proj_channels = cond_proj_channels
        self.time_scale = time_scale
        self.cond_proj = None
        if cond_proj_channels is not None:
            if cond_in_channels is None:
                raise ValueError("cond_in_channels must be provided when cond_proj_channels is set.")
            self.cond_proj = nn.Conv2d(cond_in_channels, cond_proj_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y is None:
            raise ValueError("Dense radar condition `y` is required for Mean Flow training.")
        if y.ndim != 4:
            raise ValueError(f"Expected condition shape [B,C,H,W], got {tuple(y.shape)}")
        if y.shape[0] != x.shape[0]:
            raise ValueError("Condition batch size must match input batch size.")
        if self.cond_in_channels is not None and y.shape[1] != self.cond_in_channels:
            raise ValueError(
                f"Expected raw condition with {self.cond_in_channels} channels, got {y.shape[1]}."
            )
        if self.cond_proj is not None:
            y = self.cond_proj(y)

        h, w = x.shape[-2:]
        r_map = r[:, None, None, None].expand(-1, 1, h, w)
        dt_map = (t - r)[:, None, None, None].expand(-1, 1, h, w)
        cond = torch.cat([y, r_map, dt_map], dim=1)

        expected_cond_channels = getattr(self.unet, "num_classes", None)
        if expected_cond_channels is not None and cond.shape[1] != expected_cond_channels:
            raise ValueError(
                f"UNet expects {expected_cond_channels} conditioning channels, "
                f"but Mean Flow built {cond.shape[1]} channels."
            )

        t_scaled = t * self.time_scale
        out = self.unet(x, t_scaled, y=cond)
        return out[:, : self.in_channels]


MeanFlowConditionedUNet = MFUNet

__all__ = ["MFUNet", "MeanFlowConditionedUNet"]
