#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-07-29
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : generative_models_lightning/diffusion/edm.py
# @IDE     : vscode

"""Elucidated Diffusion Models, from mathematics to Lightning orchestration.

The implementation follows NVLabs EDM commit
008a4e5316c8e3bfe61a62f874bddba254295afb. The upstream implementation is
licensed under CC BY-NC-SA 4.0; see THIRD_PARTY_NOTICES.md.

The file is intentionally self-contained and ordered by execution flow:

1. :class:`EDMPreconditioner` turns a raw noise-conditioned network into an
   EDM denoiser.
2. :class:`EDMLoss` defines the continuous-sigma training objective.
3. :func:`edm_sampler` implements Algorithm 2 (Euler with Heun correction).
4. :class:`EDMDiffusionModule` connects those pieces to Lightning and EMA.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .base_diffusion_module import BaseDiffusionModule

# Preconditioning and training objective.


def _expand_sigma(
    sigma: torch.Tensor | float,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sigma as ``[B]`` and as an x-broadcastable tensor."""
    sigma_tensor = torch.as_tensor(
        sigma,
        device=x.device,
        dtype=torch.float32,
    )
    if sigma_tensor.numel() == 1:
        sigma_flat = sigma_tensor.reshape(1).expand(x.shape[0])
    elif sigma_tensor.numel() == x.shape[0]:
        sigma_flat = sigma_tensor.reshape(x.shape[0])
    else:
        raise ValueError("sigma must be scalar or contain one value per batch element")
    if not torch.isfinite(sigma_flat).all() or (sigma_flat <= 0).any():
        raise ValueError("sigma values must be finite and greater than zero")
    sigma_broadcast = sigma_flat.reshape(
        x.shape[0],
        *([1] * (x.ndim - 1)),
    )
    return sigma_flat, sigma_broadcast


class EDMPreconditioner(nn.Module):
    """Wrap a noise-conditioned network with EDM preconditioning."""

    def __init__(
        self,
        denoiser: nn.Module,
        *,
        sigma_data: float = 0.5,
        sigma_min: float = 0.0,
        sigma_max: float = math.inf,
        condition_key: str = "y",
    ) -> None:
        super().__init__()
        if sigma_data <= 0:
            raise ValueError("sigma_data must be greater than zero")
        if sigma_min < 0:
            raise ValueError("sigma_min must be non-negative")
        if sigma_max <= sigma_min:
            raise ValueError("sigma_max must be greater than sigma_min")
        if not condition_key:
            raise ValueError("condition_key must not be empty")

        self.denoiser = denoiser
        self.sigma_data = float(sigma_data)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.condition_key = condition_key

    def _model_kwargs(
        self,
        cond: Any | None,
        model_kwargs: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        kwargs = dict(model_kwargs or {})
        if cond is None:
            return kwargs
        if isinstance(cond, torch.Tensor):
            if self.condition_key in kwargs:
                raise ValueError(
                    f"Condition supplied twice for key {self.condition_key!r}"
                )
            kwargs[self.condition_key] = cond
            return kwargs
        if isinstance(cond, Mapping):
            return {**dict(cond), **kwargs}
        raise TypeError("EDM condition must be a Tensor, mapping, or None")

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor | float,
        cond: Any | None = None,
        *,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        sigma_flat, sigma_broadcast = _expand_sigma(sigma, x)
        x_float = x.to(torch.float32)
        sigma_data = self.sigma_data

        c_skip = sigma_data**2 / (sigma_broadcast.square() + sigma_data**2)
        c_out = (
            sigma_broadcast
            * sigma_data
            / (sigma_broadcast.square() + sigma_data**2).sqrt()
        )
        c_in = 1.0 / (sigma_broadcast.square() + sigma_data**2).sqrt()
        c_noise = sigma_flat.log() / 4.0

        model_output = self.denoiser(
            c_in * x_float,
            c_noise,
            **self._model_kwargs(cond, model_kwargs),
        )
        if model_output.shape != x.shape:
            raise ValueError(
                "EDM denoiser output must match its input shape: "
                f"{tuple(model_output.shape)} != {tuple(x.shape)}"
            )
        return c_skip * x_float + c_out * model_output.to(torch.float32)

    @staticmethod
    def round_sigma(sigma: torch.Tensor | float) -> torch.Tensor:
        """Return supported sigma levels; continuous EDM needs no rounding."""
        return torch.as_tensor(sigma)


class EDMLoss(nn.Module):
    """Log-normal sigma sampling and EDM weighted denoising objective."""

    def __init__(
        self,
        *,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_data: float = 0.5,
    ) -> None:
        super().__init__()
        if p_std <= 0:
            raise ValueError("p_std must be greater than zero")
        if sigma_data <= 0:
            raise ValueError("sigma_data must be greater than zero")
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.sigma_data = float(sigma_data)

    def forward(
        self,
        denoiser: nn.Module,
        x: torch.Tensor,
        cond: Any | None = None,
        *,
        model_kwargs: Mapping[str, Any] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError("EDM inputs must be floating point tensors")

        sigma_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd_normal = torch.randn(
            sigma_shape,
            device=x.device,
            dtype=torch.float32,
            generator=generator,
        )
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()
        weight = (sigma.square() + self.sigma_data**2) / (
            sigma * self.sigma_data
        ).square()
        noise = torch.randn(
            x.shape,
            device=x.device,
            dtype=torch.float32,
            generator=generator,
        )
        target = x.to(torch.float32)
        denoised = denoiser(
            target + noise * sigma,
            sigma.reshape(x.shape[0]),
            cond=cond,
            model_kwargs=model_kwargs,
        )
        squared_error = weight * (denoised - target).square()
        return squared_error.flatten(start_dim=1).mean(dim=1)


# Euler-Heun sampling and classifier-free guidance.


def _guided_denoise(
    denoiser: nn.Module,
    x: torch.Tensor,
    sigma: torch.Tensor,
    *,
    cond: Any | None,
    uncond: torch.Tensor | None,
    guidance_scale: float,
    model_kwargs: Mapping[str, Any] | None,
) -> torch.Tensor:
    if guidance_scale == 1.0:
        return denoiser(
            x,
            sigma,
            cond=cond,
            model_kwargs=model_kwargs,
        )
    if cond is None:
        raise ValueError("CFG requires a conditional input")
    if not isinstance(cond, torch.Tensor):
        raise TypeError("CFG currently supports Tensor conditions only")
    if cond.shape[0] != x.shape[0]:
        raise ValueError("Condition batch size must match sample batch size")

    null_condition = torch.zeros_like(cond) if uncond is None else uncond
    if null_condition.shape != cond.shape:
        raise ValueError("uncond and cond must have the same shape")
    unconditional = denoiser(
        x,
        sigma,
        cond=null_condition,
        model_kwargs=model_kwargs,
    )
    conditional = denoiser(
        x,
        sigma,
        cond=cond,
        model_kwargs=model_kwargs,
    )
    return unconditional + guidance_scale * (conditional - unconditional)


@torch.inference_mode()
def edm_sampler(
    denoiser: nn.Module,
    latents: torch.Tensor,
    cond: Any | None = None,
    *,
    uncond: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    s_churn: float = 0.0,
    s_min: float = 0.0,
    s_max: float = math.inf,
    s_noise: float = 1.0,
    generator: torch.Generator | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Sample with EDM Algorithm 2 using Euler steps and Heun correction."""
    if not latents.is_floating_point():
        raise TypeError("latents must be floating point tensors")
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if sigma_min <= 0 or sigma_max <= sigma_min:
        raise ValueError("Require 0 < sigma_min < sigma_max for EDM sampling")
    if rho <= 0:
        raise ValueError("rho must be greater than zero")
    if s_churn < 0 or s_noise < 0:
        raise ValueError("s_churn and s_noise must be non-negative")
    if s_min < 0 or s_max < s_min:
        raise ValueError("Require 0 <= s_min <= s_max")
    if not math.isfinite(guidance_scale) or guidance_scale < 0:
        raise ValueError("guidance_scale must be finite and non-negative")
    if isinstance(cond, torch.Tensor) and cond.shape[0] != latents.shape[0]:
        raise ValueError("Condition batch size must match latent batch size")

    network_sigma_min = float(getattr(denoiser, "sigma_min", 0.0))
    network_sigma_max = float(getattr(denoiser, "sigma_max", math.inf))
    sigma_min = max(float(sigma_min), network_sigma_min)
    sigma_max = min(float(sigma_max), network_sigma_max)
    if sigma_max <= sigma_min:
        raise ValueError("Sampler sigma range does not overlap network support")

    step_indices = torch.arange(
        num_steps,
        dtype=torch.float64,
        device=latents.device,
    )
    t_steps = (
        sigma_max ** (1.0 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ).pow(rho)
    round_sigma = getattr(
        denoiser,
        "round_sigma",
        lambda value: torch.as_tensor(value),
    )
    t_steps = round_sigma(t_steps).to(
        device=latents.device,
        dtype=torch.float64,
    )
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    output_dtype = latents.dtype
    x_next = latents.to(torch.float64) * t_steps[0]
    for index, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        current_sigma = float(t_cur.item())
        gamma = (
            min(s_churn / num_steps, math.sqrt(2.0) - 1.0)
            if s_min <= current_sigma <= s_max
            else 0.0
        )
        t_hat = round_sigma(t_cur + gamma * t_cur).to(
            device=latents.device,
            dtype=torch.float64,
        )
        churn_noise = torch.randn(
            x_cur.shape,
            device=x_cur.device,
            dtype=x_cur.dtype,
            generator=generator,
        )
        noise_scale = (t_hat.square() - t_cur.square()).clamp_min(0).sqrt()
        x_hat = x_cur + noise_scale * s_noise * churn_noise

        denoised = _guided_denoise(
            denoiser,
            x_hat,
            t_hat,
            cond=cond,
            uncond=uncond,
            guidance_scale=guidance_scale,
            model_kwargs=model_kwargs,
        ).to(torch.float64)
        derivative = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * derivative

        if index < num_steps - 1:
            denoised_next = _guided_denoise(
                denoiser,
                x_next,
                t_next,
                cond=cond,
                uncond=uncond,
                guidance_scale=guidance_scale,
                model_kwargs=model_kwargs,
            ).to(torch.float64)
            derivative_next = (x_next - denoised_next) / t_next
            x_next = x_hat + (t_next - t_hat) * (
                0.5 * derivative + 0.5 * derivative_next
            )

    return x_next.to(output_dtype)


# Lightning orchestration and EMA.


class EDMDiffusionModule(BaseDiffusionModule):
    """Train conditional EDM and sample from its EMA denoiser."""

    def __init__(
        self,
        denoiser: nn.Module,
        sample_shape: Sequence[int] | None = None,
        *,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_data: float = 0.5,
        num_steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        s_churn: float = 0.0,
        s_min: float = 0.0,
        s_max: float = math.inf,
        s_noise: float = 1.0,
        condition_key: str = "y",
        condition_dropout: float = 0.1,
        guidance_scale: float = 1.0,
        ema_halflife_kimg: float = 0.5,
        ema_rampup_ratio: float | None = 0.05,
        use_ema_for_sampling: bool = True,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_shape=sample_shape,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs,
        )
        if not 0.0 <= condition_dropout <= 1.0:
            raise ValueError("condition_dropout must be in [0, 1]")
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2")
        if sigma_min <= 0 or sigma_max <= sigma_min:
            raise ValueError("Require 0 < sigma_min < sigma_max")
        if rho <= 0:
            raise ValueError("rho must be greater than zero")
        if s_churn < 0 or s_noise < 0:
            raise ValueError("s_churn and s_noise must be non-negative")
        if s_min < 0 or s_max < s_min:
            raise ValueError("Require 0 <= s_min <= s_max")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and non-negative")
        if ema_halflife_kimg <= 0:
            raise ValueError("ema_halflife_kimg must be greater than zero")
        if ema_rampup_ratio is not None and ema_rampup_ratio <= 0:
            raise ValueError("ema_rampup_ratio must be positive or None")

        self.preconditioner = EDMPreconditioner(
            denoiser,
            sigma_data=sigma_data,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            condition_key=condition_key,
        )
        self.loss_fn = EDMLoss(
            p_mean=p_mean,
            p_std=p_std,
            sigma_data=sigma_data,
        )
        self.ema_preconditioner = copy.deepcopy(self.preconditioner)
        self.ema_preconditioner.requires_grad_(False)

        self.num_steps = int(num_steps)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.s_churn = float(s_churn)
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.s_noise = float(s_noise)
        self.condition_dropout = float(condition_dropout)
        self.guidance_scale = float(guidance_scale)
        self.ema_halflife_kimg = float(ema_halflife_kimg)
        self.ema_rampup_ratio = ema_rampup_ratio
        self.use_ema_for_sampling = bool(use_ema_for_sampling)

        self.register_buffer(
            "ema_total_nimg",
            torch.zeros((), dtype=torch.float64),
        )
        self.register_buffer(
            "ema_updates",
            torch.zeros((), dtype=torch.long),
        )
        self._ema_pending_nimg = 0.0
        self.save_hyperparameters(ignore=("denoiser",))

    @property
    def denoiser(self) -> nn.Module:
        """Expose the trainable network without registering it twice."""
        return self.preconditioner.denoiser

    def _drop_condition(
        self,
        cond: Any | None,
        *,
        batch_size: int,
    ) -> Any | None:
        if cond is None or self.condition_dropout == 0.0:
            return cond
        if isinstance(cond, Mapping):
            return cond
        if not isinstance(cond, torch.Tensor):
            raise TypeError(
                "Automatic condition dropout currently supports Tensor "
                "conditions only"
            )
        if cond.shape[0] != batch_size:
            raise ValueError("Condition batch size must match target batch")

        keep = (
            torch.rand(
                batch_size,
                device=cond.device,
            )
            >= self.condition_dropout
        )
        keep = keep.reshape(
            batch_size,
            *([1] * (cond.ndim - 1)),
        )
        return cond * keep.to(cond.dtype)

    def compute_loss_terms(
        self,
        x: torch.Tensor,
        cond: Any | None,
    ) -> dict[str, torch.Tensor]:
        """Apply condition dropout, then evaluate the EDM objective."""
        dropped_cond = self._drop_condition(
            cond,
            batch_size=x.shape[0],
        )
        return {
            "loss": self.loss_fn(
                self.preconditioner,
                x,
                cond=dropped_cond,
            )
        }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Count images across accumulated batches for NVLabs-style EMA."""
        x, _ = self._unpack_batch(batch)
        trainer = getattr(self, "_trainer", None)
        world_size = int(trainer.world_size) if trainer is not None else 1
        self._ema_pending_nimg += x.shape[0] * world_size
        return super().training_step(batch, batch_idx)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Any | None = None,
    ) -> None:
        """Update EMA only after Lightning performs a real optimizer step."""
        super().optimizer_step(
            epoch,
            batch_idx,
            optimizer,
            optimizer_closure,
        )
        self._update_ema(self._ema_pending_nimg)
        self._ema_pending_nimg = 0.0

    @torch.no_grad()
    def _update_ema(self, batch_nimg: float) -> None:
        if batch_nimg <= 0:
            return

        self.ema_total_nimg.add_(batch_nimg)
        halflife_nimg = self.ema_halflife_kimg * 1000.0
        if self.ema_rampup_ratio is not None:
            halflife_nimg = min(
                halflife_nimg,
                float(self.ema_total_nimg) * self.ema_rampup_ratio,
            )
        ema_beta = 0.5 ** (batch_nimg / max(halflife_nimg, 1e-8))

        for ema_param, model_param in zip(
            self.ema_preconditioner.parameters(),
            self.preconditioner.parameters(),
        ):
            ema_param.copy_(model_param.detach().lerp(ema_param, ema_beta))
        for ema_buffer, model_buffer in zip(
            self.ema_preconditioner.buffers(),
            self.preconditioner.buffers(),
        ):
            ema_buffer.copy_(model_buffer)
        self.ema_updates.add_(1)

    @torch.inference_mode()
    def sample(
        self,
        shape: Sequence[int],
        cond: Any | None = None,
        *,
        uncond: torch.Tensor | None = None,
        guidance_scale: float | None = None,
        num_steps: int | None = None,
        use_ema: bool | None = None,
        generator: torch.Generator | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        **sampler_kwargs: Any,
    ) -> torch.Tensor:
        """Create Gaussian latents and run the configured EDM sampler."""
        resolved_shape = tuple(int(value) for value in shape)
        if any(value <= 0 for value in resolved_shape):
            raise ValueError("All sample shape dimensions must be positive")
        if isinstance(cond, torch.Tensor) and cond.shape[0] != resolved_shape[0]:
            raise ValueError("Condition batch size must match sample shape")

        latents = torch.randn(
            resolved_shape,
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )
        resolved_use_ema = (
            self.use_ema_for_sampling if use_ema is None else bool(use_ema)
        )
        sampling_model = (
            self.ema_preconditioner if resolved_use_ema else self.preconditioner
        )
        was_training = sampling_model.training
        sampling_model.eval()
        settings = {
            "num_steps": self.num_steps if num_steps is None else num_steps,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "rho": self.rho,
            "s_churn": self.s_churn,
            "s_min": self.s_min,
            "s_max": self.s_max,
            "s_noise": self.s_noise,
            **sampler_kwargs,
        }
        try:
            return edm_sampler(
                sampling_model,
                latents,
                cond=cond,
                uncond=uncond,
                guidance_scale=(
                    self.guidance_scale
                    if guidance_scale is None
                    else float(guidance_scale)
                ),
                generator=generator,
                model_kwargs=model_kwargs,
                **settings,
            )
        finally:
            sampling_model.train(was_training)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Use the optimizer settings from the original EDM recipe."""
        return torch.optim.Adam(
            self.preconditioner.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay,
        )
