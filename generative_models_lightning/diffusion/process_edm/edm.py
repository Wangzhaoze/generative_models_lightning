#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-07-20
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : /generative_models_lightning/diffusion/process_edm/emd.py
# @IDE     : vscode



"""
EDM training loss and EDM Euler-Heun sampler.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import math
import torch


class EDMLoss:
    """EDM weighted denoising objective."""

    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_data: float = 1.0,
    ) -> None:
        if p_std <= 0:
            raise ValueError("p_std must be greater than zero")
        if sigma_data <= 0:
            raise ValueError("sigma_data must be greater than zero")

        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data

    def __call__(
        self,
        denoiser: torch.nn.Module,
        x: torch.Tensor,
        *,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        """
        Return one loss value per sample: shape [B].
        """
        batch_size = x.shape[0]
        sigma_shape = (batch_size,) + (1,) * (x.ndim - 1)

        sigma = torch.exp(
            torch.randn(
                sigma_shape,
                device=x.device,
                dtype=x.dtype,
            )
            * self.p_std
            + self.p_mean
        )

        weight = (
            sigma.square() + self.sigma_data**2
        ) / (sigma * self.sigma_data).square()

        noisy_x = x + torch.randn_like(x) * sigma
        denoised_x = denoiser(
            noisy_x,
            sigma.flatten(),
            **dict(model_kwargs or {}),
        )

        squared_error = weight * (denoised_x - x).square()
        return squared_error.flatten(start_dim=1).mean(dim=1)


@torch.no_grad()
def edm_sampler(
    denoiser: torch.nn.Module,
    latents: torch.Tensor,
    *,
    model_kwargs: Mapping[str, Any] | None = None,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    s_churn: float = 0.0,
    s_min: float = 0.0,
    s_max: float = math.inf,
    s_noise: float = 1.0,
    randn_like: Callable[[torch.Tensor], torch.Tensor] = torch.randn_like,
) -> torch.Tensor:
    """Algorithm 2 of EDM: Euler sampling with Heun correction."""
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")

    sigma_min = max(sigma_min, float(denoiser.sigma_min))
    sigma_max = min(sigma_max, float(denoiser.sigma_max))

    step_indices = torch.arange(
        num_steps,
        device=latents.device,
        dtype=torch.float32,
    )

    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho

    t_steps = torch.cat(
        [
            denoiser.round_sigma(t_steps),
            torch.zeros_like(t_steps[:1]),
        ]
    )

    kwargs = dict(model_kwargs or {})
    x_next = latents * t_steps[0]

    for index, (t_cur, t_next) in enumerate(
        zip(t_steps[:-1], t_steps[1:])
    ):
        gamma = (
            min(s_churn / num_steps, math.sqrt(2) - 1)
            if s_min <= t_cur <= s_max
            else 0.0
        )

        t_hat = denoiser.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_next + (
            (t_hat.square() - t_cur.square()).sqrt()
            * s_noise
            * randn_like(x_next)
        )

        sigma_hat = t_hat.expand(x_hat.shape[0])
        denoised = denoiser(x_hat, sigma_hat, **kwargs)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if index < num_steps - 1:
            sigma_next = t_next.expand(x_next.shape[0])
            denoised_next = denoiser(
                x_next,
                sigma_next,
                **kwargs,
            )
            d_prime = (x_next - denoised_next) / t_next

            x_next = x_hat + (t_next - t_hat) * (
                0.5 * d_cur + 0.5 * d_prime
            )

    return x_next