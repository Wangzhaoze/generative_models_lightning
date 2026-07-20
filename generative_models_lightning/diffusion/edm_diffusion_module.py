#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-07
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : generative_models_lightning/diffusion/__init__.py
# @IDE     : vscode

"""
Diffusion model implementations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from generative_models_lightning.diffusion.base_diffusion_module import (
    BaseDiffusionModule,
)
from generative_models_lightning.diffusion.process_edm import (
    EDMLoss,
    edm_sampler,
)


class EDMDiffusionModule(BaseDiffusionModule):
    """
    Lightning module for EDM.

    EDM-specific concepts:
    - continuous noise level sigma
    - log-normal sigma distribution
    - EDM weighted denoising loss
    - Euler-Heun reverse sampler
    """

    def __init__(
        self,
        denoiser: nn.Module,
        sample_shape: Sequence[int] | None = None,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_data: float = 1.0,
        num_steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        s_churn: float = 0.0,
        s_min: float = 0.0,
        s_max: float = float("inf"),
        s_noise: float = 1.0,
        condition_key: str = "cond",
        condition_type: str | None = "radar",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_shape=sample_shape,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs,
        )

        if num_steps < 2:
            raise ValueError("num_steps must be at least 2")

        self.denoiser = denoiser
        self.loss_fn = EDMLoss(
            p_mean=p_mean,
            p_std=p_std,
            sigma_data=sigma_data,
        )

        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.s_churn = s_churn
        self.s_min = s_min
        self.s_max = s_max
        self.s_noise = s_noise
        self.condition_key = condition_key
        self.condition_type = condition_type

        self.save_hyperparameters(ignore=["denoiser"])

    def _condition_to_kwargs(
        self,
        cond: Any | None,
    ) -> dict[str, Any]:
        """
        EDM conditions must be explicit mappings.

        Example:
            {
                "label_tokens": radar_tensor,
                "cond_type": "radar",
            }
        """
        if cond is None:
            return {}

        if not isinstance(cond, Mapping):
            raise TypeError(
                "EDM condition must be a mapping, for example "
                "{'label_tokens': ..., 'cond_type': 'radar'}"
            )

        if "label_tokens" in cond:
            return {
                "label_tokens": cond["label_tokens"],
                "cond_type": cond.get("cond_type", self.condition_type),
            }

        if self.condition_key in cond:
            return {
                "label_tokens": cond[self.condition_key],
                "cond_type": self.condition_type,
            }

        raise KeyError(
            "EDM condition mapping must contain either 'label_tokens' or "
            f"{self.condition_key!r}"
        )

    def compute_loss_terms(
        self,
        x: torch.Tensor,
        cond: Any | None,
    ) -> dict[str, torch.Tensor]:
        loss_per_sample = self.loss_fn(
            self.denoiser,
            x,
            model_kwargs=self._condition_to_kwargs(cond),
        )

        return {
            "loss": loss_per_sample,
        }

    @torch.no_grad()
    def sample(
        self,
        shape: Sequence[int],
        cond: Any | None = None,
        *,
        num_steps: int | None = None,
        **sampler_kwargs: Any,
    ) -> torch.Tensor:
        latents = torch.randn(
            tuple(int(value) for value in shape),
            device=self.device,
        )

        settings = {
            "num_steps": self.num_steps if num_steps is None else num_steps,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "rho": self.rho,
            "s_churn": self.s_churn,
            "s_min": self.s_min,
            "s_max": self.s_max,
            "s_noise": self.s_noise,
        }
        settings.update(sampler_kwargs)

        return edm_sampler(
            self.denoiser,
            latents,
            model_kwargs=self._condition_to_kwargs(cond),
            **settings,
        )
