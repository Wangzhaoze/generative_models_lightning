#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-05-09
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : /generative_models_lightning/diffusion/scheduler/base_scheduler.py
# @IDE     : vscode



"""
Describe the purpose of this module.
"""

"""Gaussian diffusion scheduler primitives, beta schedules, and timestep spacing."""

from __future__ import annotations

import enum

import numpy as np
import torch

from .scheduler import BaseDiffusionScheduler
from .beta_schedule import BetaSchedule, LinearBetaSchedule, CosineBetaSchedule, CustomBetaSchedule
from .utils import extract_into_tensor


class PredictionType(enum.Enum):
    EPSILON = "epsilon"
    X_START = "x_start"
    X_PREVIOUS = "x_previous"
    V_PREDICTION = "v_prediction"


class VarianceType(enum.Enum):
    FIXED_SMALL = "fixed_small"
    FIXED_LARGE = "fixed_large"
    LEARNED = "learned"
    LEARNED_RANGE = "learned_range"



class TimestepSpacing:
    """Base class for inference timestep spacing."""

    def timesteps(self, num_train_timesteps: int, num_inference_steps: int, device=None):
        raise NotImplementedError


class UniformTimestepSpacing(TimestepSpacing):
    def timesteps(self, num_train_timesteps: int, num_inference_steps: int, device=None):
        if num_inference_steps > num_train_timesteps:
            raise ValueError("num_inference_steps cannot be larger than num_train_timesteps")

        step_ratio = num_train_timesteps / num_inference_steps
        timesteps = (torch.arange(num_inference_steps, dtype=torch.float64) * step_ratio)
        timesteps = timesteps.round().long()
        timesteps = num_train_timesteps - 1 - timesteps
        timesteps = timesteps.clamp(min=0)
        return timesteps.to(device=device)


class DDIMTimestepSpacing(TimestepSpacing):
    """
    OpenAI 'ddimN' style spacing.

    This only chooses timesteps. It does not implement DDIM update equations.
    """

    def timesteps(self, num_train_timesteps: int, num_inference_steps: int, device=None):
        for stride in range(1, num_train_timesteps):
            selected = list(range(0, num_train_timesteps, stride))
            if len(selected) == num_inference_steps:
                return torch.tensor(
                    sorted(selected, reverse=True),
                    dtype=torch.long,
                    device=device,
                )

        raise ValueError(
            f"Cannot create exactly {num_inference_steps} timesteps "
            f"from {num_train_timesteps}"
        )


class SectionTimestepSpacing(TimestepSpacing):
    """
    OpenAI section-style timestep spacing.

    Example:
        SectionTimestepSpacing([10, 10, 10])
    """

    def __init__(self, section_counts):
        self.section_counts = list(section_counts)

    def timesteps(self, num_train_timesteps: int, num_inference_steps: int = None, device=None):
        size_per = num_train_timesteps // len(self.section_counts)
        extra = num_train_timesteps % len(self.section_counts)
        start_idx = 0
        all_steps = []

        for i, section_count in enumerate(self.section_counts):
            size = size_per + (1 if i < extra else 0)
            if size < section_count:
                raise ValueError(
                    f"Cannot divide section of {size} steps into {section_count}"
                )

            frac_stride = 1 if section_count <= 1 else (size - 1) / (section_count - 1)
            cur_idx = 0.0
            for _ in range(section_count):
                all_steps.append(start_idx + round(cur_idx))
                cur_idx += frac_stride
            start_idx += size

        return torch.tensor(
            sorted(set(all_steps), reverse=True),
            dtype=torch.long,
            device=device,
        )


class GaussianDiffusionScheduler(BaseDiffusionScheduler):
    """
    Common Gaussian diffusion scheduler.

    This class combines what Diffusers usually puts into scheduler:
      - beta schedule
      - alpha coefficients
      - q_sample / add_noise
      - posterior formulas
      - timestep spacing
      - prediction conversion

    It does not implement a concrete reverse sampling step.
    Concrete subclasses:
      - DDPMScheduler
      - DDIMScheduler
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: BetaSchedule = None,
        prediction_type: PredictionType = PredictionType.EPSILON,
        variance_type: VarianceType = VarianceType.FIXED_SMALL,
        timestep_spacing: TimestepSpacing = None,
        rescale_timesteps: bool = False,
        clip_sample: bool = True,
    ):
        super().__init__()

        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule or LinearBetaSchedule()
        self.prediction_type = prediction_type
        self.variance_type = variance_type
        self.timestep_spacing = timestep_spacing or UniformTimestepSpacing()
        self.rescale_timesteps = rescale_timesteps
        self.clip_sample = clip_sample

        self.betas = self.beta_schedule.betas(num_train_timesteps)
        if self.betas.ndim != 1:
            raise ValueError("betas must be 1-D")
        if not ((self.betas > 0).all() and (self.betas <= 1).all()):
            raise ValueError("betas must be in (0, 1]")

        alphas = 1.0 - self.betas
        self.alphas = alphas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1.0)

        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            self.betas
            * np.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.timesteps = torch.arange(
            num_train_timesteps - 1,
            -1,
            -1,
            dtype=torch.long,
        )

    def set_timesteps(self, num_inference_steps: int, device=None):
        self.timesteps = self.timestep_spacing.timesteps(
            num_train_timesteps=self.num_train_timesteps,
            num_inference_steps=num_inference_steps,
            device=device,
        )
        return self.timesteps

    def previous_timestep(self, timestep: int) -> int:
        if self.timesteps is None:
            raise RuntimeError("Call set_timesteps() first.")

        matches = (self.timesteps == int(timestep)).nonzero(as_tuple=False)
        if len(matches) == 0:
            raise ValueError(f"Timestep {timestep} is not in self.timesteps.")

        idx = int(matches[0].item())
        if idx == len(self.timesteps) - 1:
            return -1
        return int(self.timesteps[idx + 1].item())

    def scale_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        if self.rescale_timesteps:
            return timesteps.float() * (1000.0 / self.num_train_timesteps)
        return timesteps

    def add_noise(self, x_start, noise, timesteps):
        return self.q_sample(x_start=x_start, noise=noise, timesteps=timesteps)

    def q_sample(self, x_start, noise, timesteps):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, timesteps):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, timesteps, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, timesteps, x_t.shape)
        posterior_log_variance = extract_into_tensor(
            self.posterior_log_variance_clipped,
            timesteps,
            x_t.shape,
        )
        return posterior_mean, posterior_variance, posterior_log_variance

    def predict_xstart_from_epsilon(self, x_t, timesteps, epsilon):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape)
            * epsilon
        )

    def predict_xstart_from_xprev(self, x_t, timesteps, x_prev):
        return (
            extract_into_tensor(1.0 / self.posterior_mean_coef1, timesteps, x_t.shape) * x_prev
            - extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1,
                timesteps,
                x_t.shape,
            )
            * x_t
        )

    def predict_epsilon_from_xstart(self, x_t, timesteps, x_start):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
            - x_start
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape)

    def get_velocity(self, x_start, noise, timesteps):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
            * x_start
        )

    def model_output_to_xstart(self, model_output, sample, timesteps):
        if self.prediction_type == PredictionType.EPSILON:
            x_start = self.predict_xstart_from_epsilon(
                x_t=sample,
                timesteps=timesteps,
                epsilon=model_output,
            )
        elif self.prediction_type == PredictionType.X_START:
            x_start = model_output
        elif self.prediction_type == PredictionType.V_PREDICTION:
            alpha_bar = extract_into_tensor(self.alphas_cumprod, timesteps, sample.shape)
            x_start = alpha_bar.sqrt() * sample - (1.0 - alpha_bar).sqrt() * model_output
        else:
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")

        if self.clip_sample:
            x_start = x_start.clamp(-1.0, 1.0)
        return x_start

    def p_mean_variance(self, model_output, sample, timesteps):
        if self.variance_type == VarianceType.FIXED_SMALL:
            model_variance = extract_into_tensor(self.posterior_variance, timesteps, sample.shape)
            model_log_variance = extract_into_tensor(
                self.posterior_log_variance_clipped,
                timesteps,
                sample.shape,
            )
        elif self.variance_type == VarianceType.FIXED_LARGE:
            variance = np.append(self.posterior_variance[1], self.betas[1:])
            log_variance = np.log(variance)
            model_variance = extract_into_tensor(variance, timesteps, sample.shape)
            model_log_variance = extract_into_tensor(log_variance, timesteps, sample.shape)
        else:
            raise NotImplementedError(
                f"Variance type {self.variance_type} not implemented here yet."
            )

        if self.prediction_type == PredictionType.X_PREVIOUS:
            pred_xstart = self.predict_xstart_from_xprev(
                x_t=sample,
                timesteps=timesteps,
                x_prev=model_output,
            )
            if self.clip_sample:
                pred_xstart = pred_xstart.clamp(-1.0, 1.0)
            model_mean = model_output
        else:
            pred_xstart = self.model_output_to_xstart(
                model_output=model_output,
                sample=sample,
                timesteps=timesteps,
            )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart,
                x_t=sample,
                timesteps=timesteps,
            )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }



class DDPMScheduler(GaussianDiffusionScheduler):
    """
    DDPM ancestral sampling scheduler.

    Uses GaussianDiffusionScheduler.p_mean_variance(), then samples:
        x_{t-1} = mean + sigma * noise
    """

    def step(self, model_output, timestep, sample, generator=None, **kwargs):
        batch_size = sample.shape[0]
        timesteps = self.expand_timestep(
            timestep,
            batch_size=batch_size,
            device=sample.device,
        )

        out = self.p_mean_variance(
            model_output=model_output,
            sample=sample,
            timesteps=timesteps,
        )

        noise = torch.randn(
            sample.shape,
            device=sample.device,
            dtype=sample.dtype,
            generator=generator,
        )
        nonzero_mask = (timesteps != 0).float().view(
            batch_size,
            *([1] * (sample.ndim - 1)),
        )
        return out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise



class DDIMScheduler(GaussianDiffusionScheduler):
    """
    DDIM scheduler.

    Uses the same Gaussian diffusion coefficients, but applies the DDIM update.
    """

    def __init__(self, *args, eta: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta = eta

    def step(self, model_output, timestep, sample, generator=None, eta=None, **kwargs):
        eta = self.eta if eta is None else eta
        batch_size = sample.shape[0]

        timesteps = self.expand_timestep(
            timestep,
            batch_size=batch_size,
            device=sample.device,
        )
        prev_timestep = self.previous_timestep(int(timesteps[0].item()))

        pred_xstart = self.model_output_to_xstart(
            model_output=model_output,
            sample=sample,
            timesteps=timesteps,
        )
        eps = self.predict_epsilon_from_xstart(
            x_t=sample,
            timesteps=timesteps,
            x_start=pred_xstart,
        )

        alpha_bar = self.alphas_cumprod[int(timesteps[0].item())]
        alpha_bar_prev = 1.0 if prev_timestep < 0 else self.alphas_cumprod[prev_timestep]

        sigma = (
            eta
            * np.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
            * np.sqrt(max(0.0, 1.0 - alpha_bar / alpha_bar_prev))
        )
        mean_pred = (
            pred_xstart * np.sqrt(alpha_bar_prev)
            + np.sqrt(max(0.0, 1.0 - alpha_bar_prev - sigma**2)) * eps
        )

        if eta > 0.0 and prev_timestep >= 0:
            noise = torch.randn(
                sample.shape,
                device=sample.device,
                dtype=sample.dtype,
                generator=generator,
            )
            return mean_pred + sigma * noise

        return mean_pred
