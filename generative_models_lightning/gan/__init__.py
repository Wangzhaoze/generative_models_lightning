#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-07
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : generative_models_lightning/gan/__init__.py
# @IDE     : vscode

"""GAN modules, networks, and shared training scaffolding."""

from .base_gan_module import BaseGANModule
from .base_latent_gan_module import BaseLatentGANModule
from .base_translation_gan_module import BaseTranslationGANModule
from .cyclegan import CycleGANModule
from .dcgan import DCGANModule
from .pix2pix import Pix2PixModule
from .wgan_gp import WGANGPModule

__all__ = [
    "BaseGANModule",
    "BaseLatentGANModule",
    "BaseTranslationGANModule",
    "CycleGANModule",
    "DCGANModule",
    "Pix2PixModule",
    "WGANGPModule",
]
