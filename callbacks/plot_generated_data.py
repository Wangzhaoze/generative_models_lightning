#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-07
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : callbacks/plot_generated_data.py
# @IDE     : vscode

import torch
import wandb
import lightning as pl
from torchvision.utils import make_grid


class PlotGeneratedDataCallback(pl.Callback):
    def __init__(self, dataset, num_samples: int = 4, log_every_n_epochs: int = 1,
                 cond_is_image: bool = False):
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        indices = list(range(num_samples))
        self.cond = torch.stack([dataset[i][1]["cond"] for i in indices])
        gt = torch.stack([dataset[i][0] for i in indices]).clamp(-1, 1)
        self.ground_truth = (gt + 1) / 2  # [0, 1]
        # expand 1-channel targets (e.g. radar) to 3 channels for grid display
        if self.ground_truth.shape[1] == 1:
            self.ground_truth = self.ground_truth.expand(-1, 3, -1, -1)
        if cond_is_image:
            # cond is already an image in [-1, 1]
            cond_vis = (self.cond.clamp(-1, 1) + 1) / 2  # [N,C,H,W]
            if cond_vis.shape[1] == 1:
                cond_vis = cond_vis.expand(-1, 3, -1, -1)
            self.mask = cond_vis
        else:
            # argmax 还原 mask id → 归一化到 [0,1]，扩展到3通道
            self.mask = self.cond.argmax(dim=1, keepdim=True).float() / 18.0
            self.mask = self.mask.expand(-1, 3, -1, -1)  # [N,3,H,W]

    def _psnr(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 10 * torch.log10(1 / mse)

    def sample_images(self, pl_module: pl.LightningModule):
        cond = {"cond": self.cond.to(pl_module.device)}
        with torch.no_grad():
            img: torch.Tensor = pl_module.generate(cond=cond, batch_size=self.num_samples)  # type: ignore[call-arg]
        return ((img.clamp(-1, 1) + 1) / 2).float()

    def on_train_epoch_end(self, trainer, pl_module: pl.LightningModule):
        if not trainer.is_global_zero:
            return

        if (pl_module.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        img = self.sample_images(pl_module)
        # expand 1-channel outputs to 3 channels for grid display
        if img.shape[1] == 1:
            img = img.expand(-1, 3, -1, -1)
        gt   = self.ground_truth.to(img.device)
        mask = self.mask.to(img.device)
        psnr = self._psnr(img, gt)

        # 三行拼接：mask / ground_truth / generated，每行 num_samples 张
        combined = torch.cat([mask, gt, img], dim=0)  # [3N, 3, H, W]
        grid = make_grid(combined, nrow=self.num_samples)

        for logger in trainer.loggers:
            exp = getattr(logger, "experiment", None)
            if exp is None:
                continue
            if hasattr(exp, 'add_image'):
                exp.add_image("comparison", grid, pl_module.current_epoch)
                exp.add_scalar("psnr", psnr, pl_module.current_epoch)
            elif hasattr(exp, 'log'):
                exp.log({
                    "comparison": wandb.Image(grid),
                    "psnr": psnr.item(),
                }, step=pl_module.global_step)
