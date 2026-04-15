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

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, cast, Mapping

import torch

from generative_models_lightning import BaseGenerativeModule, Batch
import torch.nn as nn

from generative_models_lightning.diffusion.process.gaussian_diffusion import GaussianDiffusion, mean_flat, DiffusionLossType, DiffusionMeanType, DiffusionVarType


class BaseDiffusionModule(BaseGenerativeModule):
    """
    Shared LightningModule scaffolding for diffusion models.

    Diffusion modules own batch interpretation and optimization-time orchestration.
    The underlying diffusion process stays responsible for process-specific
    forward/reverse diffusion math and sampling routines.
    """

    def __init__(
        self,
        diffusion_process: GaussianDiffusion,
        denoiser: nn.Module,
        mean_type: DiffusionMeanType,
        var_type: DiffusionVarType,
        loss_type: DiffusionLossType,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.diffusion_process = diffusion_process
        self.denoiser = denoiser
        self.mean_type = mean_type
        self.var_type = var_type
        self.loss_type = loss_type

    def training_step(self, batch: Batch, batch_idx: int):
        t=torch.randint(
            low=0, 
            high=self.diffusion_process.num_timesteps, 
            size=(batch["x"].shape[0],), 
            device=batch["x"].device
            )
        losses = self._compute_losses(
            x_start=batch["x"],
            t=t,
            model_kwargs={"y": batch["cond"]} if batch.get("cond") is not None else None,
        )

        # Extract main loss
        if isinstance(losses, dict) and "loss" in losses:
            loss = losses["loss"].mean()
        else:
            raise ValueError("Expected 'loss' key in losses dict.")
        # Log loss
        self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        # Log other loss components (if any)
        if isinstance(losses, dict):
            for k, v in losses.items():
                if k == "loss":
                    continue
                try:
                    val = v.mean()
                    self.log(f"val/{k}", val, on_step=True, on_epoch=True, sync_dist=True)
                except Exception:
                    pass
        
        return loss
    
    def validation_step(self, batch: Batch, batch_idx: int):
        t=torch.randint(
            low=0, 
            high=self.diffusion_process.num_timesteps, 
            size=(batch["x"].shape[0],), 
            device=batch["x"].device
            )
        losses = self._compute_losses(
            x_start=batch["x"],
            t=t,
            model_kwargs={"y": batch["cond"]} if batch.get("cond") is not None else None,
        )

        # Extract main loss
        if isinstance(losses, dict) and "loss" in losses:
            loss = losses["loss"].mean()
        else:
            raise ValueError("Expected 'loss' key in losses dict.")
        # Log loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        # Log other loss components (if any)
        if isinstance(losses, dict):
            for k, v in losses.items():
                if k == "loss":
                    continue
                try:
                    val = v.mean()
                    self.log(f"train/{k}", val, on_step=True, on_epoch=True, sync_dist=True)
                except Exception:
                    pass
        
        return loss

    
    def generate(self, batch: Batch, *args, **kwargs):
        with torch.no_grad():
            generated = self.diffusion_process.p_sample_loop(
                self.denoiser,
                batch["x"],
                device=self.device,
                model_kwargs=kwargs
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
