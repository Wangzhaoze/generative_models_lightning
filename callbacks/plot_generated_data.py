#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-07
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : callbacks/plot_generated_data.py
# @IDE     : vscode

"""
Placeholder callback for visualizing generated samples during training.
"""


class PlotGeneratedDataCallback:
    """Minimal callback placeholder for generated data visualization."""

    def __init__(self, num_samples: int = 8) -> None:
        self.num_samples = num_samples
