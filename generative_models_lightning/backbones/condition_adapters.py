#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-07-29
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : generative_models_lightning/backbones/condition_adapters.py
# @IDE     : vscode

"""Adapters that decouple dataset conditions from backbone representations."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ClassToSpatialCondition(nn.Module):
    """Adapt class labels or class vectors to a spatial condition map."""

    def __init__(self, model: nn.Module, num_classes: int) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be greater than zero")
        self.model = model
        self.num_classes = int(num_classes)

    def _to_spatial(
        self,
        condition: torch.Tensor | None,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        if condition is None:
            condition = torch.zeros(
                batch_size,
                self.num_classes,
                device=x.device,
                dtype=x.dtype,
            )
        elif condition.ndim == 1:
            if condition.shape[0] != batch_size:
                raise ValueError("Class-label batch size must match model input")
            if condition.is_floating_point():
                raise TypeError(
                    "One-dimensional class labels must use an integer dtype"
                )
            if (condition < 0).any() or (condition >= self.num_classes).any():
                raise ValueError("Class labels are outside the valid range")
            condition = F.one_hot(
                condition.to(torch.long),
                num_classes=self.num_classes,
            )

        if condition.shape[0] != batch_size:
            raise ValueError("Condition batch size must match model input")
        if condition.ndim == 2:
            if condition.shape[1] != self.num_classes:
                raise ValueError("Condition width must equal num_classes")
            condition = condition[:, :, None, None]
        elif condition.ndim == 4:
            if condition.shape[1] != self.num_classes:
                raise ValueError("Spatial condition channels must equal num_classes")
        else:
            raise ValueError("Condition must have shape [B], [B, K], or [B, K, H, W]")

        condition = condition.to(
            device=x.device,
            dtype=x.dtype,
        )
        if condition.shape[2:] == x.shape[2:]:
            return condition
        if all(size == 1 for size in condition.shape[2:]):
            return condition.expand(
                batch_size,
                self.num_classes,
                *x.shape[2:],
            )
        return F.interpolate(
            condition,
            size=x.shape[2:],
            mode="nearest",
        )

    def forward(
        self,
        x: torch.Tensor,
        noise_level: torch.Tensor,
        y: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        spatial_condition = self._to_spatial(y, x)
        return self.model(
            x,
            noise_level,
            y=spatial_condition,
            **kwargs,
        )
