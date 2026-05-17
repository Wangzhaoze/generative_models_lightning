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

import torch
import wandb
import numpy as np
import matplotlib
import lightning as pl
from torchvision.utils import make_grid


class PlotGeneratedDataCallback(pl.Callback):
    LIDAR_CMAP = "viridis"
    RADAR_CMAP = LIDAR_CMAP

    def __init__(self, dataset, num_samples: int = 4, log_every_n_epochs: int = 1,
                 cond_is_image: bool = False, sample_steps: int | None = 50):
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.sample_steps = sample_steps
        indices = list(range(num_samples))
        self.cond = torch.stack([dataset[i][1]["cond"] for i in indices])
        gt = torch.stack([dataset[i][0] for i in indices]).clamp(-1, 1)
        self.ground_truth = self._plot_lidar_batch(gt)
        if cond_is_image:
            self.cond_vis = self._plot_radar_batch(self.cond)
        else:
            # fallback: treat as single-channel image
            cond_vis = (self.cond.clamp(-1, 1) + 1) / 2
            if cond_vis.shape[1] != 1:
                cond_vis = cond_vis.max(dim=1, keepdim=True)[0]
            self.cond_vis = cond_vis.expand(-1, 3, -1, -1)  # [N,3,H,W]

    @staticmethod
    def _collapse_range_channels(tensor_chw: torch.Tensor) -> torch.Tensor:
        """Collapse multi-channel radar range image to one display channel."""
        if tensor_chw.ndim != 3:
            raise ValueError("Range image should have shape (C,H,W)")
        if tensor_chw.shape[0] == 1:
            return tensor_chw
        return tensor_chw.amax(dim=0, keepdim=True)

    @classmethod
    def _plot_single_colored_range_image(
        cls,
        tensor_chw: torch.Tensor,
        cmap_name: str,
    ) -> torch.Tensor:
        """Render one range image with the same valid-mask normalization as demo1.py."""
        tensor_chw = cls._collapse_range_channels(tensor_chw)
        img = tensor_chw.squeeze(0).detach().cpu().numpy()
        valid = img > -0.99
        img_01 = np.clip((img + 1.0) * 0.5, 0.0, 1.0)
        vmax = img_01[valid].max() if valid.any() else 1.0
        if vmax > 1e-6:
            img_01 = np.clip(img_01 / vmax, 0.0, 1.0)
        colored = matplotlib.colormaps[cmap_name](img_01)[..., :3].astype(np.float32)
        colored[~valid] = 0.0
        return torch.from_numpy(colored.transpose(2, 0, 1)).to(tensor_chw.device)

    @classmethod
    def _plot_lidar_batch(cls, batch_nchw: torch.Tensor) -> torch.Tensor:
        """LiDAR display: demo1.py-style viridis color range image."""
        return torch.stack([
            cls._plot_single_colored_range_image(img, cls.LIDAR_CMAP)
            for img in batch_nchw
        ])

    @classmethod
    def _plot_radar_batch(cls, batch_nchw: torch.Tensor) -> torch.Tensor:
        """Radar display: compress channels, then use the same color map as LiDAR."""
        return torch.stack([
            cls._plot_single_colored_range_image(img, cls.RADAR_CMAP)
            for img in batch_nchw
        ])

    def _psnr(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 10 * torch.log10(1 / mse)

    def sample_images(self, pl_module: pl.LightningModule):
        cond = {"cond": self.cond.to(pl_module.device)}
        with torch.no_grad():
            img: torch.Tensor = pl_module.generate(  # type: ignore[call-arg]
                cond=cond,
                batch_size=self.num_samples,
                sample_steps=self.sample_steps,
            )
        return img.clamp(-1, 1).float()

    def on_train_epoch_end(self, trainer, pl_module: pl.LightningModule):
        if not trainer.is_global_zero:
            return

        if (pl_module.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        img = self.sample_images(pl_module)
        img = self._plot_lidar_batch(img)
        gt   = self.ground_truth.to(img.device)
        mask = self.cond_vis.to(img.device)
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
                trainer_step = int(getattr(trainer, "global_step", pl_module.global_step))
                wandb_step = getattr(exp, "step", None)
                if isinstance(wandb_step, int):
                    safe_step = max(trainer_step, wandb_step)
                else:
                    safe_step = trainer_step
                exp.log({
                    "comparison": wandb.Image(grid),
                    "psnr": psnr.item(),
                }, step=safe_step)
