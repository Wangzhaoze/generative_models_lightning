"""Parallel diffusion implementations sharing one Lightning base."""

from .base_diffusion_module import BaseDiffusionModule
from .edm import EDMDiffusionModule, EDMLoss, EDMPreconditioner, edm_sampler
from .gaussian_diffusion import (
    AlphaBarBetaSchedule,
    BetaSchedule,
    CosineBetaSchedule,
    DiffusionLossType,
    DiffusionMeanType,
    DiffusionVarType,
    GaussianDiffusion,
    GaussianDiffusionModule,
    LinearBetaSchedule,
    LossAwareSampler,
    LossSecondMomentResampler,
    ScheduleSampler,
    SpacedDiffusion,
    UniformSampler,
    approx_standard_normal_cdf,
    create_named_schedule_sampler,
    extract_into_tensor,
    mean_flat,
    normal_kl,
    space_timesteps,
)

# Backwards-compatible package-level name for the old Gaussian-specific class.
BaseDiffusionLitModule = GaussianDiffusionModule

__all__ = [
    "AlphaBarBetaSchedule",
    "BaseDiffusionLitModule",
    "BaseDiffusionModule",
    "BetaSchedule",
    "CosineBetaSchedule",
    "EDMDiffusionModule",
    "EDMLoss",
    "EDMPreconditioner",
    "DiffusionLossType",
    "DiffusionMeanType",
    "DiffusionVarType",
    "GaussianDiffusion",
    "GaussianDiffusionModule",
    "LinearBetaSchedule",
    "LossAwareSampler",
    "LossSecondMomentResampler",
    "ScheduleSampler",
    "SpacedDiffusion",
    "UniformSampler",
    "approx_standard_normal_cdf",
    "create_named_schedule_sampler",
    "edm_sampler",
    "extract_into_tensor",
    "mean_flat",
    "normal_kl",
    "space_timesteps",
]
