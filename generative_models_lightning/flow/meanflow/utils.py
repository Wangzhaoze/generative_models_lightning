# Adapted from https://github.com/haidog-yaqub/MeanFlow
# Commit: 18b67b25a3af86d199005023bb190024d28233e7
# Original license: MIT.

from __future__ import annotations

from collections.abc import Sequence

import torch


class Normalizer:
    """Normalize image or latent tensors for MeanFlow training."""

    def __init__(
        self,
        mode: str = "minmax",
        mean: float | Sequence[float] | None = None,
        std: float | Sequence[float] | None = None,
    ) -> None:
        if mode not in {"minmax", "mean_std"}:
            raise ValueError("mode must be 'minmax' or 'mean_std'.")
        if mode == "mean_std" and (mean is None or std is None):
            raise ValueError("mean and std are required for mean_std normalization.")

        self.mode = mode
        self.mean = torch.as_tensor(mean) if mean is not None else None
        self.std = torch.as_tensor(std) if std is not None else None

    @classmethod
    def from_list(cls, config: Sequence[object]) -> "Normalizer":
        mode, mean, std = config
        return cls(mode=mode, mean=mean, std=std)

    def _statistics(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mean is None or self.std is None:
            raise RuntimeError("mean_std statistics are not configured.")
        shape = (1, -1, *([1] * (x.ndim - 2)))
        mean = self.mean.to(device=x.device, dtype=x.dtype).reshape(shape)
        std = self.std.to(device=x.device, dtype=x.dtype).reshape(shape)
        return mean, std

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "minmax":
            return x * 2.0 - 1.0
        mean, std = self._statistics(x)
        return (x - mean) / std

    def unnorm(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "minmax":
            return (x.clamp(-1.0, 1.0) + 1.0) * 0.5
        mean, std = self._statistics(x)
        return x * std + mean

    normalize = norm
    denormalize = unnorm
