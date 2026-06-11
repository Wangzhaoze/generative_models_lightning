#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared output locations for generated artifacts."""

from __future__ import annotations

from pathlib import Path


GENERATE_RESULT_ROOT = Path("/home/local/Desktop/code/Datasets/processing/Generate_result")
OUTPUTS_ROOT = GENERATE_RESULT_ROOT
CHECKPOINTS_ROOT = GENERATE_RESULT_ROOT / "checkpoints"
SAMPLES_ROOT = GENERATE_RESULT_ROOT / "samples"
VIDEOS_ROOT = GENERATE_RESULT_ROOT / "videos"
DEMO_ROOT = GENERATE_RESULT_ROOT / "demo"


__all__ = [
    "GENERATE_RESULT_ROOT",
    "OUTPUTS_ROOT",
    "CHECKPOINTS_ROOT",
    "SAMPLES_ROOT",
    "VIDEOS_ROOT",
    "DEMO_ROOT",
]
