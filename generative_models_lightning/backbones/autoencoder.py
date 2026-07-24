#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-07
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : generative_models_lightning/backbones/autoencoder.py
# @IDE     : vscode

"""Common backbone contract for deterministic and variational autoencoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import nn


class AutoEncoderBackbone(nn.Module, ABC):
    """Shared encode/decode and trainability interface for all autoencoders.

    Concrete AE and VAE backbones decide the exact latent/output contract. This
    class only defines the operations that every autoencoder must provide.
    """

    def freeze(self) -> "AutoEncoderBackbone":
        self.requires_grad_(False)
        self.eval()
        return self

    def unfreeze(self) -> "AutoEncoderBackbone":
        self.requires_grad_(True)
        self.train()
        return self

    @abstractmethod
    def encode(self, x, **kwargs: Any):
        """Map an input tensor to the concrete autoencoder's latent contract."""

    @abstractmethod
    def decode(self, latents, **kwargs: Any):
        """Map latent values back to the input data space."""


# Alternative spelling for callers that prefer the PyTorch class style.
AutoencoderBackbone = AutoEncoderBackbone


__all__ = ["AutoEncoderBackbone", "AutoencoderBackbone"]
