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

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Optional, cast

import torch
import torch.nn.functional as F

from generative_models_lightning.diffusion.base_diffusion_module import (
    BaseDiffusionModule,
)
import torch.nn as nn

from generative_models_lightning.backbones.vae import AutoencoderKL

from generative_models_lightning.diffusion.process.gaussian_diffusion import (
    DiffusionLossType,
    DiffusionMeanType,
    DiffusionVarType,
    GaussianDiffusion,
    mean_flat,
)
from generative_models_lightning.diffusion.process.utils import (
    get_named_beta_schedule,
)


class GaussianDiffusionModule(BaseDiffusionModule):
    """
    Lightning module for discrete Gaussian diffusion models such as DDPM.

    This module owns DDPM-specific concepts:
    - discrete timesteps t
    - beta schedule
    - GaussianDiffusion
    - epsilon/x0/variance objectives
    - DDPM reverse sampling
    """

    def __init__(
        self,
        denoiser: nn.Module,
        vae: AutoencoderKL | None = None,
        vae_checkpoint_path: str | None = None,
        vae_checkpoint_prefix: str = "vae.",
        strict_vae_checkpoint: bool = True,
        freeze_vae: bool = True,
        diffusion_process: GaussianDiffusion | None = None,
        mean_type: DiffusionMeanType | str | None = None,
        var_type: DiffusionVarType | str | None = None,
        loss_type: DiffusionLossType | str | None = None,
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        rescale_timesteps: bool = False,
        sample_shape: Sequence[int] | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        **kwargs,
    ):
        super().__init__(
        sample_shape=sample_shape,
        lr=lr,
        weight_decay=weight_decay,
        **kwargs,
        )

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
                else DiffusionVarType.LEARNED_RANGE
            ),
        )
        resolved_loss = self._resolve_enum(
            loss_type,
            DiffusionLossType,
            default=(
                diffusion_process.loss_type
                if diffusion_process is not None
                else DiffusionLossType.RESCALED_MSE
            ),
        )
        if diffusion_process is None:
            if num_timesteps <= 0:
                raise ValueError("num_timesteps must be greater than zero")
            diffusion_process = GaussianDiffusion(
                betas=get_named_beta_schedule(beta_schedule, num_timesteps),
                model_mean_type=resolved_mean,
                model_var_type=resolved_var,
                loss_type=resolved_loss,
                rescale_timesteps=rescale_timesteps,
            )
        else:
            expected = (resolved_mean, resolved_var, resolved_loss)
            actual = (
                diffusion_process.model_mean_type,
                diffusion_process.model_var_type,
                diffusion_process.loss_type,
            )
            if actual != expected:
                raise ValueError(
                    "diffusion_process types do not match mean_type/var_type/loss_type: "
                    f"process={actual}, module={expected}"
                )

        self.diffusion_process = diffusion_process
        self.denoiser = denoiser
        self.vae = vae
        self.freeze_vae = bool(freeze_vae)
        if self.vae is not None and vae_checkpoint_path is not None:
            self._load_vae_checkpoint(
                vae_checkpoint_path,
                prefix=vae_checkpoint_prefix,
                strict=strict_vae_checkpoint,
            )
        if self.vae is not None and self.freeze_vae:
            self.vae.freeze()
        self.mean_type = resolved_mean
        self.var_type = resolved_var
        self.loss_type = resolved_loss
        self.sample_shape = (
            tuple(int(value) for value in sample_shape)
            if sample_shape is not None
            else None
        )
        if self.sample_shape is not None and any(value <= 0 for value in self.sample_shape):
            raise ValueError("sample_shape dimensions must be greater than zero")

        self.save_hyperparameters(ignore=("denoiser", "diffusion_process", "vae"))

    def _load_vae_checkpoint(
        self,
        checkpoint_path: str,
        *,
        prefix: str,
        strict: bool,
    ) -> None:
        """Load a separately trained AutoencoderKL before LDM training."""
        if self.vae is None:
            raise RuntimeError("Cannot load a VAE checkpoint without an AutoencoderKL")
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"KL-VAE checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint
            else checkpoint
        )
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Checkpoint {path} does not contain a state dictionary")
        if prefix:
            state_dict = {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if not state_dict:
                raise KeyError(
                    f"Checkpoint {path} has no parameters with prefix {prefix!r}"
                )
        self.vae.load_state_dict(dict(state_dict), strict=strict)

    @staticmethod
    def _resolve_enum(value, enum_type, *, default):
        if value is None:
            return default
        if isinstance(value, enum_type):
            return value
        if isinstance(value, str):
            key = value.strip().upper()
            try:
                return enum_type[key]
            except KeyError as error:
                available = ", ".join(item.name.lower() for item in enum_type)
                raise ValueError(
                    f"Unknown {enum_type.__name__} value {value!r}; "
                    f"choose one of: {available}"
                ) from error
        raise TypeError(
            f"{enum_type.__name__} must be a string or enum value, "
            f"got {type(value).__name__}"
        )

    def compute_loss_terms(
        self,
        x: torch.Tensor,
        cond: Any | None,
    ) -> dict[str, torch.Tensor]:
        """
        DDPM-specific training loss.

        Sample a discrete timestep t, add noise through GaussianDiffusion,
        then compute epsilon/x0/VLB-related loss terms.
        """
        vae_kl = None
        if self.vae is not None:
            x, vae_kl = self._encode_vae(x, sample_posterior=True)
            vae_kl = vae_kl.detach()

        model_kwargs = self._condition_to_model_kwargs(cond)
        model_kwargs = self._resize_spatial_condition(model_kwargs, x.shape[-2:])

        t = torch.randint(
            low=0,
            high=self.diffusion_process.num_timesteps,
            size=(x.shape[0],),
            device=x.device,
        )

        terms = self._compute_losses(
            x_start=x,
            t=t,
            model_kwargs=model_kwargs,
        )
        if vae_kl is not None:
            terms["vae_kl"] = vae_kl
        return terms


    @torch.no_grad()
    def sample(
        self,
        shape: Sequence[int],
        cond: Any | None = None,
        *,
        progress: bool = False,
        decode: bool = True,
        clip_denoised: bool | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """DDPM-specific reverse sampling."""
        model_kwargs = self._condition_to_model_kwargs(cond)
        model_kwargs.update(kwargs)
        model_kwargs = self._resize_spatial_condition(model_kwargs, shape[-2:])
        if clip_denoised is None:
            clip_denoised = self.vae is None

        samples = self.diffusion_process.p_sample_loop(
            self.denoiser,
            tuple(int(value) for value in shape),
            device=self.device,
            model_kwargs=model_kwargs,
            clip_denoised=clip_denoised,
            progress=progress,
        )
        if self.vae is not None and decode:
            return self._decode_vae(samples)
        return samples

    @torch.no_grad()
    def generate(
        self,
        batch: Any | None = None,
        *,
        batch_size: int | None = None,
        shape: Sequence[int] | None = None,
        cond: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate data-space samples, inferring latent shape when possible."""
        if shape is None and batch is not None and self.vae is not None:
            batch_x, _ = self._unpack_batch(batch)
            latents, _ = self._encode_vae(
                batch_x,
                sample_posterior=False,
            )
            shape = latents.shape

        return super().generate(
            batch=batch,
            batch_size=batch_size,
            shape=shape,
            cond=cond,
            **kwargs,
        )

    def _encode_vae(
        self,
        x: torch.Tensor,
        *,
        sample_posterior: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode image-space data and apply the KL-VAE latent scaling."""
        if self.vae is None:
            raise RuntimeError("Cannot encode without an AutoencoderKL")
        gradient_enabled = torch.is_grad_enabled() and not self.freeze_vae
        with torch.set_grad_enabled(gradient_enabled):
            posterior = self.vae.encode(x)
            latents = posterior.sample() if sample_posterior else posterior.mode()
            kl = posterior.kl()
        scaling_factor = float(self.vae.config.scaling_factor)
        shift_factor = float(self.vae.config.shift_factor or 0.0)
        latents = (latents - shift_factor) * scaling_factor
        return latents, kl

    def _decode_vae(self, latents: torch.Tensor) -> torch.Tensor:
        """Undo latent scaling and decode into the original data space."""
        if self.vae is None:
            raise RuntimeError("Cannot decode without an AutoencoderKL")
        scaling_factor = float(self.vae.config.scaling_factor)
        shift_factor = float(self.vae.config.shift_factor or 0.0)
        return self.vae.decode(latents / scaling_factor + shift_factor)

    @staticmethod
    def _condition_to_model_kwargs(
        cond: Any | None,
    ) -> dict[str, Any]:
        if cond is None:
            return {}
        if isinstance(cond, torch.Tensor):
            return {"y": cond}
        if isinstance(cond, Mapping):
            model_kwargs = {
                key: value for key, value in cond.items() if key in {"y", "s"}
            }
            if "y" not in model_kwargs and "cond" in cond:
                model_kwargs["y"] = cond["cond"]
            return model_kwargs
        raise TypeError(
            "GaussianDiffusionModule cond must be a Tensor, mapping, or None"
        )

    @staticmethod
    def _resize_spatial_condition(
        model_kwargs: Mapping[str, Any],
        spatial_shape: Sequence[int],
    ) -> dict[str, Any]:
        resized = dict(model_kwargs)
        y = resized.get("y")
        target_shape = tuple(int(value) for value in spatial_shape)
        if (
            isinstance(y, torch.Tensor)
            and y.ndim == 4
            and tuple(y.shape[-2:]) != target_shape
        ):
            resized["y"] = F.interpolate(
                y,
                size=target_shape,
                mode="bilinear",
                align_corners=False,
            )
        return resized

    def _compute_losses(
        self,
        x_start,
        t,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        noise=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        kwargs = self.diffusion_process._normalize_model_kwargs(model_kwargs)
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.diffusion_process.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == DiffusionLossType.KL or self.loss_type == DiffusionLossType.RESCALED_KL:
            terms["loss"] = self.diffusion_process._vb_terms_bpd(
                denoiser=self.denoiser,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=kwargs,
            )["output"]
            if self.loss_type == DiffusionLossType.RESCALED_KL:
                terms["loss"] *= self.diffusion_process.num_timesteps
        elif self.loss_type == DiffusionLossType.MSE or self.loss_type == DiffusionLossType.RESCALED_MSE:
            y = cast(Optional[torch.Tensor], kwargs.get("y"))
            if y is not None:
                model_output = self.denoiser(
                    x_t,
                    self.diffusion_process._scale_timesteps(t),
                    y=y,
                )
            else:
                model_output = self.denoiser(
                    x_t,
                    self.diffusion_process._scale_timesteps(t),
                    **kwargs,
                )

            if self.var_type in [
                DiffusionVarType.LEARNED,
                DiffusionVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self.diffusion_process._vb_terms_bpd(
                    denoiser=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == DiffusionLossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.diffusion_process.num_timesteps / 1000.0

            target = {
                DiffusionMeanType.PREVIOUS_X: self.diffusion_process.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                DiffusionMeanType.START_X: x_start,
                DiffusionMeanType.EPSILON: noise,
            }[self.mean_type]
            
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms
