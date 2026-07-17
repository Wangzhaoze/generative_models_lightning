"""Backward-compatible imports for the decomposed MeanFlow implementation.

MeanFlow now lives under ``flow_model.Flow_matching`` as independent path,
loss, and solver components.  This module preserves the original import path.
"""

from ..Flow_matching.loss import adaptive_l2_loss, stopgrad, stop_gradient
from ..Flow_matching.meanflow import MeanFlow
from ..Flow_matching.utils import Normalizer

__all__ = [
    "MeanFlow",
    "Normalizer",
    "adaptive_l2_loss",
    "stopgrad",
    "stop_gradient",
]
