"""Vendored MeanFlow components."""

from .meanflow import MeanFlow, Normalizer

__all__ = ["MeanFlow", "Normalizer", "MFDiT", "MFUNet", "MeanFlowConditionedUNet"]


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
