#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-05-09
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : /generative_models_lightning/diffusion/scheduler/__init__.py
# @IDE     : vscode



"""
Describe the purpose of this module.
"""

"""Diffusion schedulers."""

from .gaussian_diffusion import GaussianDiffusion
from .spaced_diffusion import SpacedDiffusion, space_timesteps
from .utils import extract_into_tensor, mean_flat, normal_kl
from .beta_schedule import get_named_beta_schedule
from .resample import ScheduleSampler, create_named_schedule_sampler
from .ddpm import DDPMScheduler
from .ddim import DDIMScheduler

__all__ = [
    "GaussianDiffusion",
    "SpacedDiffusion",
    "space_timesteps",
    "extract_into_tensor",
    "mean_flat",
    "normal_kl",
    "get_named_beta_schedule",
    "ScheduleSampler",
    "create_named_schedule_sampler",
    "DDPMScheduler",
    "DDIMScheduler",
]
