#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-07
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : generative_models_lightning/__init__.py
# @IDE     : vscode

"""
Public package exports for generative models.
"""

__all__ = ["BaseGenerativeModule"]


from typing import Any, Optional, Union, TypedDict

import lightning as pl
import torch
import numpy as np
from dataclasses import dataclass, field

class Batch(TypedDict):
    x: torch.Tensor
    y_embed: Optional[torch.Tensor]
    cond: Optional[torch.Tensor]


class BaseGenerativeModule(pl.LightningModule):
    """Base class placeholder shared by diffusion, GAN, flow, and VAE models."""

    def __init__(self, lr: float = 1e-4, weight_decay: float = 0.01, **kwargs):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay

    def training_step(self, batch: Batch, batch_idx: int):
        """Define the shared training-step interface for child modules."""
        raise NotImplementedError("Subclasses must implement training_step.")
    
    def validation_step(self, batch: Batch, batch_idx: int):
        """Define the shared validation-step interface for child modules."""
        raise NotImplementedError("Subclasses must implement validation_step.")
    
    def predict_step(self, batch: Batch, batch_idx: int) -> Any:
        return super().predict_step(batch, batch_idx)
    
    @torch.inference_mode()
    def generate(self, batch: Batch, batch_idx: int):
        """Define the shared generation interface for child modules."""
        raise NotImplementedError("Subclasses must implement generate.")
    
    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        return optimizer

