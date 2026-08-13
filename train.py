#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-07
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : train.py
# @IDE     : vscode

"""
Entry point for model training experiments.
"""


import os

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

path = os.getcwd()

@hydra.main(config_path='configs', config_name='mnist_config.yaml')
def main(cfg: DictConfig) -> None:
    """
    Main function to execute the training of a segmentation model.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    os.chdir(path)

    torch.set_float32_matmul_precision('medium')  # Alternatively, use 'high'

    data_module = instantiate(cfg.data_module)  # Instantiate the data module

    model = instantiate(cfg.model)  # Instantiate the segmentation model

    trainer = instantiate(cfg.trainer)

    trainer.fit(model=model, datamodule=data_module)


if __name__ == '__main__':
    main()  # Execute the main function if this script is run directly