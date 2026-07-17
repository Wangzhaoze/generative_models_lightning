"""Backward-compatible MeanFlow imports and model backbones."""

from .meanflow import MeanFlow, Normalizer, adaptive_l2_loss, stopgrad, stop_gradient

__all__ = [
    "MeanFlow",
    "Normalizer",
    "adaptive_l2_loss",
    "stopgrad",
    "stop_gradient",
    "MFDiT",
    "MFUNet",
    "MeanFlowConditionedUNet",
]


def __getattr__(name):
    if name == "MFDiT":
        from .models import MFDiT

        return MFDiT
    if name == "MFUNet":
        from .models import MFUNet

        return MFUNet
    if name == "MeanFlowConditionedUNet":
        from .models import MeanFlowConditionedUNet

        return MeanFlowConditionedUNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
