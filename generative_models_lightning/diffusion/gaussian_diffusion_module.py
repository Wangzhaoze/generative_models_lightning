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
from typing import Any, Dict, Optional, cast

import torch

from generative_models_lightning import BaseGenerativeModule, Batch
import torch.nn as nn

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


class BaseDiffusionModule(BaseGenerativeModule):
    """
    Shared LightningModule scaffolding for diffusion models.

    Diffusion modules own batch interpretation and optimization-time orchestration.
    The underlying diffusion process stays responsible for process-specific
    forward/reverse diffusion math and sampling routines.
    """

    def __init__(
        self,
        denoiser: nn.Module,
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
        super().__init__(lr=lr, weight_decay=weight_decay, **kwargs)

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

        self.save_hyperparameters(ignore=("denoiser", "diffusion_process"))

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

    @staticmethod
    def _unpack_batch(batch) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(batch, Mapping):
            if "x" not in batch:
                raise KeyError("Diffusion batch mapping must contain 'x'")
            x = batch["x"]
            cond = batch.get("cond")
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, condition = batch
            cond = condition.get("cond") if isinstance(condition, Mapping) else condition
        elif isinstance(batch, torch.Tensor):
            x, cond = batch, None
        else:
            raise TypeError(
                "Diffusion batch must be a mapping, tensor, or (x, condition) pair"
            )
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"batch input must be a Tensor, got {type(x).__name__}")
        if cond is not None and not isinstance(cond, torch.Tensor):
            raise TypeError(f"batch condition must be a Tensor, got {type(cond).__name__}")
        return x, cond

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        x_start, cond = self._unpack_batch(batch)
        t = torch.randint(
            low=0,
            high=self.diffusion_process.num_timesteps,
            size=(x_start.shape[0],),
            device=x_start.device,
        )
        losses = self._compute_losses(
            x_start=x_start,
            t=t,
            model_kwargs={"y": cond} if cond is not None else None,
        )
        if "loss" not in losses:
            raise ValueError("Expected 'loss' key in losses dict")
        loss = losses["loss"].mean()
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=True,
        )
        for name, value in losses.items():
            if name != "loss":
                self.log(
                    f"{stage}/{name}",
                    value.mean(),
                    on_step=stage == "train",
                    on_epoch=True,
                    sync_dist=True,
                )
        return loss

    def training_step(self, batch: Batch, batch_idx: int):
        _ = batch_idx
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Batch, batch_idx: int):
        _ = batch_idx
        return self._shared_step(batch, "val")


    def generate(
        self,
        batch=None,
        *,
        batch_size: int | None = None,
        shape: Sequence[int] | None = None,
        cond: torch.Tensor | Mapping[str, Any] | None = None,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        progress: bool = False,
    ):
        x = None
        batch_cond = None
        if batch is not None:
            x, batch_cond = self._unpack_batch(batch)
        if cond is None:
            cond = batch_cond
        if isinstance(cond, Mapping):
            cond = cond.get("cond")

        if shape is None:
            if x is not None:
                shape = tuple(x.shape)
            elif self.sample_shape is not None:
                shape = (int(batch_size or 1), *self.sample_shape)
            else:
                raise ValueError(
                    "Generation requires batch, shape, or configured sample_shape"
                )
        shape = tuple(int(value) for value in shape)
        kwargs = dict(model_kwargs or {})
        if cond is not None:
            kwargs["y"] = cond
        with torch.no_grad():
            generated = self.diffusion_process.p_sample_loop(
                self.denoiser,
                shape,
                device=self.device,
                model_kwargs=kwargs,
                progress=progress,
            )
        return generated


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
