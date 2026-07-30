#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-07
# @Author  : Zhaoze Wang, Chenlin Lang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : generative_models_lightning/diffusion/gaussian_diffusion.py
# @IDE     : vscode

"""Discrete Gaussian diffusion, from schedules to Lightning orchestration.

The numerical process is adapted from OpenAI guided-diffusion. This file is
intentionally self-contained and ordered by execution flow:

1. Public prediction/loss types, tensor helpers, and beta schedules.
2. :class:`GaussianDiffusion`, the DDPM/DDIM forward and reverse process.
3. Timestep spacing and importance samplers.
4. :class:`GaussianDiffusionModule`, the Lightning-facing training API.
"""

from __future__ import annotations

import enum
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from .base_diffusion_module import BaseDiffusionModule

# Public types and numerical helpers.


class DiffusionMeanType(enum.Enum):
    """Prediction represented by the denoiser output."""

    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


class DiffusionVarType(enum.Enum):
    """Variance represented by the denoiser output."""

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class DiffusionLossType(enum.Enum):
    """Objective used to train the discrete process."""

    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self) -> bool:
        return self in {
            DiffusionLossType.KL,
            DiffusionLossType.RESCALED_KL,
        }


def extract_into_tensor(
    values: np.ndarray,
    timesteps: torch.Tensor,
    broadcast_shape: Sequence[int],
) -> torch.Tensor:
    """Extract one schedule value per batch item and broadcast it."""
    result = (
        torch.from_numpy(values)
        .to(
            device=timesteps.device,
        )[timesteps]
        .float()
    )
    while result.ndim < len(broadcast_shape):
        result = result[..., None]
    return result.expand(broadcast_shape)


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """Average over every dimension except the batch dimension."""
    return tensor.mean(dim=tuple(range(1, tensor.ndim)))


def normal_kl(
    mean1: torch.Tensor | float,
    logvar1: torch.Tensor | float,
    mean2: torch.Tensor | float,
    logvar2: torch.Tensor | float,
) -> torch.Tensor:
    """Compute a broadcasted KL divergence between two Gaussians."""
    reference = next(
        (
            value
            for value in (mean1, logvar1, mean2, logvar2)
            if isinstance(value, torch.Tensor)
        ),
        None,
    )
    if reference is None:
        raise TypeError("At least one normal_kl argument must be a Tensor")
    logvar1 = torch.as_tensor(logvar1, device=reference.device)
    logvar2 = torch.as_tensor(logvar2, device=reference.device)
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Fast approximation of the standard normal CDF."""
    return 0.5 * (
        1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
    )


# Beta schedules.


class BetaSchedule(ABC):
    """Array-like base class for a discrete diffusion beta schedule."""

    def __init__(self, num_timesteps: int) -> None:
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")
        self.num_timesteps = int(num_timesteps)
        self._betas = self._build().astype(np.float64)
        if self._betas.shape != (self.num_timesteps,):
            raise ValueError(
                f"Beta schedule must have shape ({self.num_timesteps},), "
                f"got {self._betas.shape}"
            )
        if (
            not np.isfinite(self._betas).all()
            or (self._betas <= 0).any()
            or (self._betas > 1).any()
        ):
            raise ValueError("Beta values must be finite and in (0, 1]")

    @abstractmethod
    def _build(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def betas(self) -> np.ndarray:
        return self._betas

    def __array__(
        self,
        dtype: np.dtype[Any] | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        values = self._betas if dtype is None else self._betas.astype(dtype)
        return values.copy() if copy else values

    def __len__(self) -> int:
        return self.num_timesteps

    def __getitem__(self, index: Any) -> Any:
        return self._betas[index]

    def __iter__(self) -> Iterator[float]:
        return iter(self._betas)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"num_timesteps={self.num_timesteps})"


class LinearBetaSchedule(BetaSchedule):
    """Linear schedule from Ho et al., scaled from the 1000-step recipe."""

    def _build(self) -> np.ndarray:
        scale = 1000.0 / self.num_timesteps
        return np.linspace(
            scale * 0.0001,
            scale * 0.02,
            self.num_timesteps,
            dtype=np.float64,
        )


class AlphaBarBetaSchedule(BetaSchedule):
    """Base for schedules defined by a continuous cumulative alpha."""

    def __init__(
        self,
        num_timesteps: int,
        max_beta: float = 0.999,
    ) -> None:
        if not 0.0 < max_beta <= 1.0:
            raise ValueError("max_beta must be in (0, 1]")
        self.max_beta = float(max_beta)
        super().__init__(num_timesteps)

    @abstractmethod
    def alpha_bar(self, t: float) -> float:
        raise NotImplementedError

    def _build(self) -> np.ndarray:
        betas = []
        for index in range(self.num_timesteps):
            t1 = index / self.num_timesteps
            t2 = (index + 1) / self.num_timesteps
            beta = 1.0 - self.alpha_bar(t2) / self.alpha_bar(t1)
            betas.append(min(beta, self.max_beta))
        return np.asarray(betas, dtype=np.float64)


class CosineBetaSchedule(AlphaBarBetaSchedule):
    """Cosine alpha-bar schedule from Improved DDPM."""

    def alpha_bar(self, t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2.0) ** 2


# DDPM and DDIM process.


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the denoiser outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              denoiser so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas: BetaSchedule | Sequence[float] | np.ndarray,
        model_mean_type: DiffusionMeanType,
        model_var_type: DiffusionVarType,
        loss_type: DiffusionLossType,
        rescale_timesteps: bool = False,
    ) -> None:
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        betas = np.asarray(betas, dtype=np.float64)
        if betas.ndim != 1 or betas.size < 2:
            raise ValueError("betas must be a 1-D sequence with at least 2 steps")
        if not np.isfinite(betas).all() or (betas <= 0).any() or (betas > 1).any():
            raise ValueError("betas must be finite and in (0, 1]")
        self.betas = betas

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    @property
    def num_timesteps(self) -> int:
        return int(self.betas.shape[0])

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        denoiser,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param denoiser: the model, which takes a signal and a batch of timesteps
                          as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        kwargs = self._normalize_model_kwargs(model_kwargs)

        batch_size, channels = x.shape[:2]
        if t.shape != (batch_size,):
            raise ValueError(
                f"timesteps must have shape ({batch_size},), got {t.shape}"
            )
        denoiser_output = denoiser(
            x,
            self._scale_timesteps(t),
            **kwargs,
        )

        if self.model_var_type in {
            DiffusionVarType.LEARNED,
            DiffusionVarType.LEARNED_RANGE,
        }:
            expected_shape = (
                batch_size,
                channels * 2,
                *x.shape[2:],
            )
            if denoiser_output.shape != expected_shape:
                raise ValueError(
                    "Learned variance output must have shape "
                    f"{expected_shape}, got {tuple(denoiser_output.shape)}"
                )
            denoiser_output, model_var_values = torch.split(
                denoiser_output,
                channels,
                dim=1,
            )
            if self.model_var_type == DiffusionVarType.LEARNED:
                pred_log_variance = model_var_values
                pred_variance = torch.exp(pred_log_variance)
            else:
                min_log = extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                pred_log_variance = frac * max_log + (1 - frac) * min_log
                pred_variance = torch.exp(pred_log_variance)
        else:
            pred_variance, pred_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                DiffusionVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                DiffusionVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            pred_variance = extract_into_tensor(pred_variance, t, x.shape)
            pred_log_variance = extract_into_tensor(pred_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                x = x.clamp(-1, 1)
            return x

        if self.model_mean_type == DiffusionMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=denoiser_output)
            )
            pred_mean = denoiser_output
        elif self.model_mean_type in {
            DiffusionMeanType.START_X,
            DiffusionMeanType.EPSILON,
        }:
            if self.model_mean_type == DiffusionMeanType.START_X:
                pred_xstart = process_xstart(denoiser_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=denoiser_output)
                )
            pred_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            pred_mean.shape == pred_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": pred_mean,
            "variance": pred_variance,
            "log_variance": pred_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def condition_mean(
        self,
        cond_fn,
        p_mean_var,
        x,
        t,
        model_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        kwargs = self._normalize_model_kwargs(model_kwargs)
        gradient = cond_fn(x, self._scale_timesteps(t), **kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(
        self,
        cond_fn,
        p_mean_var,
        x,
        t,
        model_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        kwargs = self._normalize_model_kwargs(model_kwargs)
        alpha_bar = extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        denoiser,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param denoiser: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            denoiser,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = (
            out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        )
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        denoiser,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param denoiser: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            denoiser,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        if final is None:
            raise RuntimeError("p_sample_loop_progressive returned no samples.")
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        denoiser,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(denoiser.parameters()).device
        assert isinstance(shape, (tuple, list))
        kwargs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in self._normalize_model_kwargs(model_kwargs).items()
        }
        if noise is not None:
            img = noise.to(device)
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    denoiser,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        denoiser,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            denoiser,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        denoiser,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            denoiser,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        denoiser,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            denoiser,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        if final is None:
            raise RuntimeError("ddim_sample_loop_progressive returned no samples.")
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        denoiser,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(denoiser.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    denoiser,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _discretized_gaussian_log_likelihood(self, x, *, means, log_scales):
        """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image.

        :param x: the target images. It is assumed that this was uint8 values,
                rescaled to the range [-1, 1].
        :param means: the Gaussian mean Tensor.
        :param log_scales: the Gaussian log stddev Tensor.
        :return: a tensor like x of log probabilities (in nats).
        """
        assert x.shape == means.shape == log_scales.shape
        centered_x = x - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(
                x > 0.999,
                log_one_minus_cdf_min,
                torch.log(cdf_delta.clamp(min=1e-12)),
            ),
        )
        assert log_probs.shape == x.shape
        return log_probs

    def _vb_terms_bpd(
        self,
        denoiser,
        x_start,
        x_t,
        t,
        clip_denoised=True,
        model_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            denoiser, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -self._discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(
        self,
        denoiser,
        x_start,
        clip_denoised=True,
        model_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param denoiser: the denoiser model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the denoiser. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = torch.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with torch.no_grad():
                out = self._vb_terms_bpd(
                    denoiser,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = torch.stack(vb, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }

    def _scale_timesteps(self, t):
        if getattr(self, "rescale_timesteps", False):
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    @staticmethod
    def _normalize_model_kwargs(
        model_kwargs: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if model_kwargs is None:
            return {}
        return dict(model_kwargs)


# Timestep spacing and unbiased importance sampling.


def space_timesteps(
    num_timesteps: int,
    section_counts: str | Sequence[int],
) -> set[int]:
    """Select retained timesteps using guided-diffusion spacing rules."""
    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be greater than zero")

    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts.removeprefix("ddim"))
            if desired_count <= 0:
                raise ValueError("DDIM step count must be greater than zero")
            for stride in range(1, num_timesteps + 1):
                steps = range(0, num_timesteps, stride)
                if len(steps) == desired_count:
                    return set(steps)
            raise ValueError(
                f"Cannot create exactly {desired_count} steps " f"from {num_timesteps}"
            )
        section_counts = [int(value) for value in section_counts.split(",")]

    counts = [int(value) for value in section_counts]
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("section_counts must contain positive integers")

    size_per = num_timesteps // len(counts)
    extra = num_timesteps % len(counts)
    start_index = 0
    selected_steps = []
    for section_index, section_count in enumerate(counts):
        size = size_per + (1 if section_index < extra else 0)
        if size < section_count:
            raise ValueError(
                f"Cannot divide section of {size} steps into " f"{section_count}"
            )
        fractional_stride = (
            1.0 if section_count == 1 else (size - 1) / (section_count - 1)
        )
        current = 0.0
        for _ in range(section_count):
            selected_steps.append(start_index + round(current))
            current += fractional_stride
        start_index += size
    return set(selected_steps)


class SpacedDiffusion(GaussianDiffusion):
    """Gaussian process that retains selected steps from a base process."""

    def __init__(
        self,
        num_timesteps: int,
        section_counts: str | Sequence[int],
        **kwargs: Any,
    ) -> None:
        if "betas" not in kwargs:
            raise ValueError("SpacedDiffusion requires a base beta schedule")
        if len(kwargs["betas"]) != num_timesteps:
            raise ValueError("num_timesteps must match the base beta schedule length")

        self.use_timesteps = space_timesteps(
            num_timesteps,
            section_counts,
        )
        self.timestep_map: list[int] = []
        self.original_num_steps = num_timesteps

        base_diffusion = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for index, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if index in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(index)
        kwargs["betas"] = np.asarray(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self,
        denoiser: nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        return super().p_mean_variance(
            self._wrap_model(denoiser),
            *args,
            **kwargs,
        )

    def condition_mean(
        self,
        cond_fn: Callable[..., torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return super().condition_mean(
            self._wrap_model(cond_fn),
            *args,
            **kwargs,
        )

    def condition_score(
        self,
        cond_fn: Callable[..., torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        return super().condition_score(
            self._wrap_model(cond_fn),
            *args,
            **kwargs,
        )

    def _wrap_model(
        self,
        model: Callable[..., torch.Tensor],
    ) -> _WrappedModel:
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model=model,
            timestep_map=self.timestep_map,
            rescale_timesteps=self.rescale_timesteps,
            original_num_steps=self.original_num_steps,
        )

    def _scale_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        return t


class _WrappedModel:
    """Map reduced process indices back to original network timesteps."""

    def __init__(
        self,
        model: Callable[..., torch.Tensor],
        timestep_map: Sequence[int],
        rescale_timesteps: bool,
        original_num_steps: int,
    ) -> None:
        self.model = model
        self.timestep_map = tuple(timestep_map)
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        mapping = torch.tensor(
            self.timestep_map,
            device=timesteps.device,
            dtype=timesteps.dtype,
        )
        mapped_timesteps = mapping[timesteps]
        if self.rescale_timesteps:
            mapped_timesteps = mapped_timesteps.float() * (
                1000.0 / self.original_num_steps
            )
        return self.model(x, mapped_timesteps, **kwargs)


class ScheduleSampler(ABC):
    """Importance-sample discrete training timesteps."""

    def __init__(self, diffusion: GaussianDiffusion) -> None:
        self.diffusion = diffusion

    @abstractmethod
    def weights(self) -> np.ndarray:
        raise NotImplementedError

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        weights = np.asarray(self.weights(), dtype=np.float64)
        if (
            weights.shape != (self.diffusion.num_timesteps,)
            or not np.isfinite(weights).all()
            or (weights <= 0).any()
        ):
            raise ValueError(
                "Schedule sampler weights must be finite, positive, and "
                "match the number of diffusion timesteps"
            )

        probabilities = torch.as_tensor(
            weights / weights.sum(),
            device=device,
            dtype=torch.float64,
        )
        timesteps = torch.multinomial(
            probabilities,
            batch_size,
            replacement=True,
            generator=generator,
        )
        importance = 1.0 / (len(probabilities) * probabilities[timesteps])
        return timesteps.long(), importance.float()


class UniformSampler(ScheduleSampler):
    """Sample every diffusion timestep with equal probability."""

    def __init__(self, diffusion: GaussianDiffusion) -> None:
        super().__init__(diffusion)
        self._weights = np.ones(
            diffusion.num_timesteps,
            dtype=np.float64,
        )

    def weights(self) -> np.ndarray:
        return self._weights


class LossAwareSampler(ScheduleSampler):
    """Synchronize adaptive timestep weights across distributed ranks."""

    def update_with_local_losses(
        self,
        local_timesteps: torch.Tensor,
        local_losses: torch.Tensor,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            self.update_with_all_losses(
                local_timesteps.detach().cpu().tolist(),
                local_losses.detach().cpu().tolist(),
            )
            return

        world_size = dist.get_world_size()
        local_size = torch.tensor(
            [len(local_timesteps)],
            dtype=torch.int64,
            device=local_timesteps.device,
        )
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(gathered_sizes, local_size)
        batch_sizes = [int(value.item()) for value in gathered_sizes]
        max_batch_size = max(batch_sizes)

        padded_timesteps = torch.zeros(
            max_batch_size,
            dtype=local_timesteps.dtype,
            device=local_timesteps.device,
        )
        padded_losses = torch.zeros(
            max_batch_size,
            dtype=local_losses.dtype,
            device=local_losses.device,
        )
        padded_timesteps[: len(local_timesteps)] = local_timesteps
        padded_losses[: len(local_losses)] = local_losses
        timestep_batches = [
            torch.zeros_like(padded_timesteps) for _ in range(world_size)
        ]
        loss_batches = [torch.zeros_like(padded_losses) for _ in range(world_size)]
        dist.all_gather(timestep_batches, padded_timesteps)
        dist.all_gather(loss_batches, padded_losses)

        all_timesteps = [
            int(value.item())
            for values, size in zip(timestep_batches, batch_sizes)
            for value in values[:size]
        ]
        all_losses = [
            float(value.item())
            for values, size in zip(loss_batches, batch_sizes)
            for value in values[:size]
        ]
        self.update_with_all_losses(all_timesteps, all_losses)

    @abstractmethod
    def update_with_all_losses(
        self,
        timesteps: Sequence[int],
        losses: Sequence[float],
    ) -> None:
        raise NotImplementedError


class LossSecondMomentResampler(LossAwareSampler):
    """Sample in proportion to each timestep's recent RMS loss."""

    def __init__(
        self,
        diffusion: GaussianDiffusion,
        history_per_term: int = 10,
        uniform_prob: float = 0.001,
    ) -> None:
        super().__init__(diffusion)
        if history_per_term <= 0:
            raise ValueError("history_per_term must be greater than zero")
        if not 0.0 <= uniform_prob <= 1.0:
            raise ValueError("uniform_prob must be in [0, 1]")
        self.history_per_term = int(history_per_term)
        self.uniform_prob = float(uniform_prob)
        self._loss_history = np.zeros(
            (diffusion.num_timesteps, history_per_term),
            dtype=np.float64,
        )
        self._loss_counts = np.zeros(
            diffusion.num_timesteps,
            dtype=np.int64,
        )

    def weights(self) -> np.ndarray:
        if not self._warmed_up():
            return np.ones(
                self.diffusion.num_timesteps,
                dtype=np.float64,
            )
        weights = np.sqrt(np.mean(np.square(self._loss_history), axis=-1))
        weights /= weights.sum()
        weights *= 1.0 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(
        self,
        timesteps: Sequence[int],
        losses: Sequence[float],
    ) -> None:
        if len(timesteps) != len(losses):
            raise ValueError("timesteps and losses must have equal length")
        for timestep, loss in zip(timesteps, losses):
            if not 0 <= timestep < self.diffusion.num_timesteps:
                raise ValueError(f"Invalid diffusion timestep {timestep}")
            if self._loss_counts[timestep] == self.history_per_term:
                self._loss_history[timestep, :-1] = self._loss_history[timestep, 1:]
                self._loss_history[timestep, -1] = loss
            else:
                index = self._loss_counts[timestep]
                self._loss_history[timestep, index] = loss
                self._loss_counts[timestep] += 1

    def _warmed_up(self) -> bool:
        return bool((self._loss_counts == self.history_per_term).all())


def create_named_schedule_sampler(
    name: str,
    diffusion: GaussianDiffusion,
) -> ScheduleSampler:
    """Construct a built-in Gaussian timestep sampler by name."""
    normalized_name = name.strip().lower()
    if normalized_name == "uniform":
        return UniformSampler(diffusion)
    if normalized_name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    raise ValueError(f"Unknown schedule sampler: {name!r}")


# Lightning orchestration.


class GaussianDiffusionModule(BaseDiffusionModule):
    """Train and sample a denoiser with a discrete DDPM/DDIM process."""

    def __init__(
        self,
        denoiser: nn.Module,
        diffusion_process: GaussianDiffusion | None = None,
        *,
        scheduler: GaussianDiffusion | None = None,
        mean_type: DiffusionMeanType | str | None = None,
        var_type: DiffusionVarType | str | None = None,
        loss_type: DiffusionLossType | str | None = None,
        num_timesteps: int = 1000,
        beta_schedule: str | BetaSchedule = "cosine",
        schedule_sampler: str | ScheduleSampler = "uniform",
        rescale_timesteps: bool = False,
        sampling_method: str = "ddpm",
        condition_key: str = "y",
        sample_shape: Sequence[int] | None = None,
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
        if diffusion_process is not None and scheduler is not None:
            raise ValueError("Provide only one of diffusion_process or scheduler")
        if not condition_key:
            raise ValueError("condition_key must not be empty")
        normalized_sampling_method = sampling_method.strip().lower()
        if normalized_sampling_method not in {"ddpm", "ddim"}:
            raise ValueError("sampling_method must be 'ddpm' or 'ddim'")
        diffusion_process = diffusion_process or scheduler

        resolved_mean = self._resolve_enum(
            mean_type,
            DiffusionMeanType,
            default=(
                diffusion_process.model_mean_type
                if diffusion_process is not None
                else DiffusionMeanType.EPSILON
            ),
        )
        resolved_var = self._resolve_enum(
            var_type,
            DiffusionVarType,
            default=(
                diffusion_process.model_var_type
                if diffusion_process is not None
                else DiffusionVarType.FIXED_SMALL
            ),
        )
        resolved_loss = self._resolve_enum(
            loss_type,
            DiffusionLossType,
            default=(
                diffusion_process.loss_type
                if diffusion_process is not None
                else DiffusionLossType.MSE
            ),
        )

        if diffusion_process is None:
            schedule = self._resolve_beta_schedule(
                beta_schedule,
                num_timesteps=num_timesteps,
            )
            diffusion_process = GaussianDiffusion(
                betas=schedule,
                model_mean_type=resolved_mean,
                model_var_type=resolved_var,
                loss_type=resolved_loss,
                rescale_timesteps=rescale_timesteps,
            )
        else:
            process_types = (
                diffusion_process.model_mean_type,
                diffusion_process.model_var_type,
                diffusion_process.loss_type,
            )
            requested_types = (
                resolved_mean,
                resolved_var,
                resolved_loss,
            )
            if process_types != requested_types:
                raise ValueError(
                    "diffusion_process settings do not match module settings: "
                    f"process={process_types}, module={requested_types}"
                )

        if isinstance(schedule_sampler, str):
            resolved_sampler = create_named_schedule_sampler(
                schedule_sampler,
                diffusion_process,
            )
        elif isinstance(schedule_sampler, ScheduleSampler):
            if (
                schedule_sampler.diffusion.num_timesteps
                != diffusion_process.num_timesteps
            ):
                raise ValueError(
                    "schedule_sampler and diffusion_process must have "
                    "the same number of timesteps"
                )
            resolved_sampler = schedule_sampler
        else:
            raise TypeError("schedule_sampler must be a name or ScheduleSampler")

        self.denoiser = denoiser
        self.diffusion_process = diffusion_process
        self.schedule_sampler = resolved_sampler
        self.sampling_method = normalized_sampling_method
        self.condition_key = condition_key
        self.mean_type = resolved_mean
        self.var_type = resolved_var
        self.loss_type = resolved_loss

        # Kept as a read-only compatibility name for older configurations.
        self.scheduler = diffusion_process
        self.save_hyperparameters(
            ignore=(
                "denoiser",
                "diffusion_process",
                "scheduler",
                "schedule_sampler",
            )
        )

    @staticmethod
    def _resolve_enum(
        value: Any,
        enum_type: type[enum.Enum],
        *,
        default: enum.Enum,
    ) -> enum.Enum:
        if value is None:
            return default
        if isinstance(value, enum_type):
            return value
        if isinstance(value, str):
            key = value.strip().upper()
            try:
                return enum_type[key]
            except KeyError as error:
                choices = ", ".join(item.name.lower() for item in enum_type)
                raise ValueError(
                    f"Unknown {enum_type.__name__} {value!r}; " f"choose {choices}"
                ) from error
        raise TypeError(f"{enum_type.__name__} must be a string or enum value")

    @staticmethod
    def _resolve_beta_schedule(
        schedule: str | BetaSchedule,
        *,
        num_timesteps: int,
    ) -> BetaSchedule:
        if isinstance(schedule, BetaSchedule):
            if len(schedule) != num_timesteps:
                raise ValueError("beta_schedule length must match num_timesteps")
            return schedule
        if not isinstance(schedule, str):
            raise TypeError("beta_schedule must be a name or BetaSchedule")

        name = schedule.strip().lower()
        if name == "linear":
            return LinearBetaSchedule(num_timesteps)
        if name == "cosine":
            return CosineBetaSchedule(num_timesteps)
        raise ValueError("beta_schedule must be 'linear' or 'cosine'")

    def _condition_to_kwargs(
        self,
        cond: Any | None,
    ) -> dict[str, Any]:
        if cond is None:
            return {}
        if isinstance(cond, torch.Tensor):
            return {self.condition_key: cond}
        if isinstance(cond, Mapping):
            return dict(cond)
        raise TypeError("Gaussian condition must be a Tensor, mapping, or None")

    def compute_loss_terms(
        self,
        x: torch.Tensor,
        cond: Any | None,
    ) -> dict[str, torch.Tensor]:
        """Sample timesteps and compute unbiased Gaussian loss terms."""
        timesteps, importance = self.schedule_sampler.sample(
            batch_size=x.shape[0],
            device=x.device,
        )
        terms = self._compute_losses(
            x_start=x,
            timesteps=timesteps,
            model_kwargs=self._condition_to_kwargs(cond),
        )
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                timesteps,
                terms["loss"].detach(),
            )
        terms["loss"] = terms["loss"] * importance
        return terms

    @torch.inference_mode()
    def sample(
        self,
        shape: Sequence[int],
        cond: Any | None = None,
        *,
        sampling_method: str | None = None,
        progress: bool = False,
        eta: float = 0.0,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """Run DDPM or DDIM with condition kwargs passed to the backbone."""
        resolved_shape = tuple(int(value) for value in shape)
        if any(value <= 0 for value in resolved_shape):
            raise ValueError("All sample shape dimensions must be positive")
        if isinstance(cond, torch.Tensor) and cond.shape[0] != resolved_shape[0]:
            raise ValueError("Condition batch size must match sample shape")
        kwargs = {
            **self._condition_to_kwargs(cond),
            **model_kwargs,
        }
        method = (
            self.sampling_method
            if sampling_method is None
            else sampling_method.strip().lower()
        )
        if method == "ddpm":
            return self.diffusion_process.p_sample_loop(
                self.denoiser,
                resolved_shape,
                device=self.device,
                model_kwargs=kwargs,
                progress=progress,
            )
        if method == "ddim":
            return self.diffusion_process.ddim_sample_loop(
                self.denoiser,
                resolved_shape,
                device=self.device,
                model_kwargs=kwargs,
                progress=progress,
                eta=eta,
            )
        raise ValueError("sampling_method must be 'ddpm' or 'ddim'")

    def _compute_losses(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        model_kwargs: Mapping[str, Any] | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        kwargs = self.diffusion_process._normalize_model_kwargs(model_kwargs)
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.diffusion_process.q_sample(
            x_start,
            timesteps,
            noise=noise,
        )
        terms: dict[str, torch.Tensor] = {}

        if self.loss_type in {
            DiffusionLossType.KL,
            DiffusionLossType.RESCALED_KL,
        }:
            terms["loss"] = self.diffusion_process._vb_terms_bpd(
                denoiser=self.denoiser,
                x_start=x_start,
                x_t=x_t,
                t=timesteps,
                clip_denoised=False,
                model_kwargs=kwargs,
            )["output"]
            if self.loss_type == DiffusionLossType.RESCALED_KL:
                terms["loss"] *= self.diffusion_process.num_timesteps
            return terms

        if self.loss_type not in {
            DiffusionLossType.MSE,
            DiffusionLossType.RESCALED_MSE,
        }:
            raise NotImplementedError(self.loss_type)

        model_output = self.denoiser(
            x_t,
            self.diffusion_process._scale_timesteps(timesteps),
            **kwargs,
        )
        if self.var_type in {
            DiffusionVarType.LEARNED,
            DiffusionVarType.LEARNED_RANGE,
        }:
            batch_size, channels = x_t.shape[:2]
            expected_shape = (
                batch_size,
                channels * 2,
                *x_t.shape[2:],
            )
            if model_output.shape != expected_shape:
                raise ValueError(
                    "Learned variance output must have shape " f"{expected_shape}"
                )
            model_output, model_var_values = torch.split(
                model_output,
                channels,
                dim=1,
            )
            frozen_output = torch.cat(
                [model_output.detach(), model_var_values],
                dim=1,
            )

            def frozen_denoiser(
                *args: Any,
                **unused: Any,
            ) -> torch.Tensor:
                del args, unused
                return frozen_output

            terms["vb"] = self.diffusion_process._vb_terms_bpd(
                denoiser=frozen_denoiser,
                x_start=x_start,
                x_t=x_t,
                t=timesteps,
                clip_denoised=False,
            )["output"]
            if self.loss_type == DiffusionLossType.RESCALED_MSE:
                terms["vb"] *= self.diffusion_process.num_timesteps / 1000.0

        if self.mean_type == DiffusionMeanType.PREVIOUS_X:
            target = self.diffusion_process.q_posterior_mean_variance(
                x_start=x_start,
                x_t=x_t,
                t=timesteps,
            )[0]
        elif self.mean_type == DiffusionMeanType.START_X:
            target = x_start
        elif self.mean_type == DiffusionMeanType.EPSILON:
            target = noise
        else:
            raise NotImplementedError(self.mean_type)
        if model_output.shape != target.shape:
            raise ValueError(
                "Denoiser output and Gaussian target shapes differ: "
                f"{tuple(model_output.shape)} != {tuple(target.shape)}"
            )

        terms["mse"] = mean_flat((target - model_output) ** 2)
        terms["loss"] = terms["mse"] + terms["vb"] if "vb" in terms else terms["mse"]
        return terms
