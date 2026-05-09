#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-05-09
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : /generative_models_lightning/diffusion/scheduler/beta_schedule.py
# @IDE     : vscode



"""
Describe the purpose of this module.
"""

import math
from abc import ABC, abstractmethod
from typing import Callable, Iterator

import numpy as np


class BetaSchedule(ABC):
    """
    Base class for beta schedules.

    A BetaSchedule object behaves like a numpy array when passed into
    np.asarray(...), len(...), indexing, or iteration.
    """

    def __init__(self, num_timesteps: int):
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive.")

        self.num_timesteps = int(num_timesteps)
        self._betas = self._build().astype(np.float64)

        if self._betas.shape != (self.num_timesteps,):
            raise ValueError(
                f"Beta schedule must have shape ({self.num_timesteps},), "
                f"but got {self._betas.shape}."
            )

    @abstractmethod
    def _build(self) -> np.ndarray:
        """
        Build beta values as a numpy array of shape [num_timesteps].
        """
        raise NotImplementedError

    @property
    def betas(self) -> np.ndarray:
        """
        Return beta values as a numpy array.
        """
        return self._betas

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Allows np.asarray(schedule) to work.
        """
        if dtype is None:
            return self._betas
        return self._betas.astype(dtype)

    def __len__(self) -> int:
        return self.num_timesteps

    def __getitem__(self, index):
        return self._betas[index]

    def __iter__(self) -> Iterator[float]:
        return iter(self._betas)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_timesteps={self.num_timesteps})"
        )


class LinearBetaSchedule(BetaSchedule):
    """
    Linear beta schedule from Ho et al.,
    extended to work for any number of diffusion steps.

    Equivalent to:

        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        np.linspace(beta_start, beta_end, num_timesteps)
    """

    def _build(self) -> np.ndarray:
        scale = 1000.0 / self.num_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02

        return np.linspace(
            beta_start,
            beta_end,
            self.num_timesteps,
            dtype=np.float64,
        )


class AlphaBarBetaSchedule(BetaSchedule):
    """
    Base class for schedules defined through alpha_bar(t).

    alpha_bar(t) defines the cumulative product of (1 - beta)
    over continuous time t in [0, 1].
    """

    def __init__(
        self,
        num_timesteps: int,
        max_beta: float = 0.999,
    ):
        if not 0.0 < max_beta <= 1.0:
            raise ValueError("max_beta must be in (0, 1].")

        self.max_beta = float(max_beta)
        super().__init__(num_timesteps)

    @abstractmethod
    def alpha_bar(self, t: float) -> float:
        raise NotImplementedError

    def _build(self) -> np.ndarray:
        betas = []

        for i in range(self.num_timesteps):
            t1 = i / self.num_timesteps
            t2 = (i + 1) / self.num_timesteps

            beta = 1.0 - self.alpha_bar(t2) / self.alpha_bar(t1)
            beta = min(beta, self.max_beta)

            betas.append(beta)

        return np.array(betas, dtype=np.float64)


class CosineBetaSchedule(AlphaBarBetaSchedule):
    """
    Cosine beta schedule.

    Equivalent to:

        betas_for_alpha_bar(
            num_timesteps,
            lambda t: cos((t + 0.008) / 1.008 * pi / 2) ** 2
        )
    """

    def alpha_bar(self, t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2.0) ** 2

