"""Vendored MeanFlow components."""

from .meanflow import MeanFlow, Normalizer

__all__ = ["MeanFlow", "Normalizer", "MFDiT"]


def __getattr__(name):
    if name == "MFDiT":
        from .models import MFDiT

        return MFDiT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
