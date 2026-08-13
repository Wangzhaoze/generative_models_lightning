"""Model adapters for MeanFlow."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MFUNet(nn.Module):
    """Adapt the repository U-Net to MeanFlow's ``(x, t, r, y)`` interface."""

    def __init__(
        self,
        unet: nn.Module,
        in_channels: int,
        cond_in_channels: int | None = None,
        cond_proj_channels: int | None = None,
        time_scale: float = 999.0,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.in_channels = int(in_channels)
        self.cond_in_channels = cond_in_channels
        self.cond_proj_channels = cond_proj_channels
        self.time_scale = float(time_scale)
        self.cond_proj = None
        if cond_proj_channels is not None:
            if cond_in_channels is None:
                raise ValueError("cond_in_channels is required with cond_proj_channels.")
            self.cond_proj = nn.Conv2d(
                cond_in_channels,
                cond_proj_channels,
                kernel_size=1,
            )

    def _normalize_condition(
        self,
        y: torch.Tensor | None,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        if y is None:
            if self.cond_in_channels is None:
                raise ValueError("A condition tensor is required for MeanFlow sampling.")
            y = torch.zeros(
                batch_size,
                self.cond_in_channels,
                1,
                1,
                device=x.device,
                dtype=x.dtype,
            )
        elif y.ndim == 1:
            if self.cond_in_channels is None:
                raise ValueError("cond_in_channels is required for label conditions.")
            if y.shape[0] != batch_size:
                raise ValueError("Condition batch size must match the input batch.")
            if y.is_floating_point():
                raise TypeError("One-dimensional MeanFlow labels must use an integer dtype.")
            y = F.one_hot(y.to(torch.long), num_classes=self.cond_in_channels).to(x.dtype)
        elif y.shape[0] != batch_size:
            raise ValueError("Condition batch size must match the input batch.")

        if y.ndim == 2:
            y = y[:, :, None, None]
        elif y.ndim != 4:
            raise ValueError("Condition must have shape [B], [B, C], or [B, C, H, W].")

        y = y.to(device=x.device, dtype=x.dtype)
        if self.cond_in_channels is not None and y.shape[1] != self.cond_in_channels:
            raise ValueError(
                f"Expected raw condition with {self.cond_in_channels} channels, "
                f"got {y.shape[1]}."
            )
        if y.shape[2:] != x.shape[2:]:
            if all(size == 1 for size in y.shape[2:]):
                y = y.expand(-1, y.shape[1], *x.shape[2:])
            else:
                y = F.interpolate(y, size=x.shape[2:], mode="nearest")

        if self.cond_proj is not None:
            y = self.cond_proj(y)
        return y

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond = self._normalize_condition(y, x)
        h, w = x.shape[-2:]
        r_map = r[:, None, None, None].expand(-1, 1, h, w)
        dt_map = (t - r)[:, None, None, None].expand(-1, 1, h, w)
        cond = torch.cat([cond, r_map, dt_map], dim=1)

        expected_cond_channels = getattr(self.unet, "num_classes", None)
        if expected_cond_channels is not None and cond.shape[1] != expected_cond_channels:
            raise ValueError(
                f"UNet expects {expected_cond_channels} conditioning channels, "
                f"but MeanFlow built {cond.shape[1]} channels."
            )

        t_scaled = t * self.time_scale
        out = self.unet(x, t_scaled, y=cond)
        if out.shape[1] < self.in_channels:
            raise ValueError("UNet output channels must be at least the data channels.")
        return out[:, : self.in_channels]


MeanFlowConditionedUNet = MFUNet

__all__ = ["MFUNet", "MeanFlowConditionedUNet"]
