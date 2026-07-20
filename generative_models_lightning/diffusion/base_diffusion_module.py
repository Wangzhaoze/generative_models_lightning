#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-07
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : generative_models_lightning/diffusion/__init__.py
# @IDE     : vscode

"""
Algorithm-agnostic Lightning scaffolding for denoising diffusion models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from generative_models_lightning import BaseGenerativeModule


class BaseDiffusionModule(BaseGenerativeModule):
    """
    Common Lightning orchestration for denoising diffusion models.

    Subclasses implement the diffusion mathematics:
    - GaussianDiffusionModule: discrete DDPM process.
    - EDMDiffusionModule: continuous EDM sigma process.
    """

    def __init__(
        self,
        sample_shape: Sequence[int] | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            **kwargs,
        )

        self.sample_shape = (
            tuple(int(value) for value in sample_shape)
            if sample_shape is not None
            else None
        )

        if self.sample_shape is not None and any(
            value <= 0 for value in self.sample_shape
        ):
            raise ValueError(
                "sample_shape dimensions must be greater than zero"
            )

    @staticmethod
    def _unpack_batch(
        batch: Any,
    ) -> tuple[torch.Tensor, Any | None]:
        """
        Normalize batches to (x, cond).

        Recommended batch format:

            {
                "x": target_tensor,
                "cond": {
                    "radar": radar_tensor,
                    "cond_type": "radar",
                },
            }

        The base class deliberately does not interpret ``cond``.
        DDPM and EDM may need different condition arguments.
        """
        if isinstance(batch, Mapping):
            if "x" not in batch:
                raise KeyError(
                    "Diffusion batch mapping must contain key 'x'"
                )
            x = batch["x"]
            cond = batch.get("cond")

        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, cond = batch

        elif isinstance(batch, torch.Tensor):
            x, cond = batch, None

        else:
            raise TypeError(
                "Batch must be a tensor, a mapping, or an (x, cond) pair"
            )

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"Batch input x must be a Tensor, got {type(x).__name__}"
            )

        return x, cond

    def _shared_step(
        self,
        batch: Any,
        stage: str,
    ) -> torch.Tensor:
        x, cond = self._unpack_batch(batch)

        loss_terms = self.compute_loss_terms(
            x=x,
            cond=cond,
        )

        if "loss" not in loss_terms:
            raise ValueError(
                "compute_loss_terms() must return a dictionary with key 'loss'"
            )

        loss = loss_terms["loss"].mean()

        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=True,
            batch_size=x.shape[0],
        )

        for name, value in loss_terms.items():
            if name != "loss":
                self.log(
                    f"{stage}/{name}",
                    value.mean(),
                    on_step=stage == "train",
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=x.shape[0],
                )

        return loss

    def training_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> torch.Tensor:
        _ = batch_idx
        return self._shared_step(batch, stage="train")

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> torch.Tensor:
        _ = batch_idx
        return self._shared_step(batch, stage="val")

    def compute_loss_terms(
        self,
        x: torch.Tensor,
        cond: Any | None,
    ) -> dict[str, torch.Tensor]:
        """Compute algorithm-specific loss terms."""
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        shape: Sequence[int],
        cond: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate samples using algorithm-specific reverse diffusion."""
        raise NotImplementedError

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
        """
        Public generation interface shared by DDPM and EDM.
        """
        batch_x = None
        batch_cond = None

        if batch is not None:
            batch_x, batch_cond = self._unpack_batch(batch)

        if cond is None:
            cond = batch_cond

        if shape is None:
            if batch_x is not None:
                shape = tuple(batch_x.shape)
            elif self.sample_shape is not None:
                shape = (int(batch_size or 1), *self.sample_shape)
            else:
                raise ValueError(
                    "Provide batch, shape, or configure sample_shape"
                )

        return self.sample(
            shape=tuple(int(value) for value in shape),
            cond=cond,
            **kwargs,
        )