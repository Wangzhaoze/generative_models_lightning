#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Chenlin Zhang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : /train_flow.py
# @IDE     : vscode



"""
Describe the purpose of this module.
"""

from typing import Union

import os
import torch
import torch.nn.functional as F

from generative_models_lightning.flow.path import AffineProbPath
from flow_matching.path.scheduler.scheduler import CosineScheduler
from generative_models_lightning.flow.solver import ODESolver
from generative_models_lightning.flow.utils import ModelWrapper

from generative_models_lightning import BaseGenerativeModule


class FlowMatchingModel(BaseGenerativeModule):
    """LightningModule wrapping ConditionalFlowMatcher for training and Euler ODE sampling.

    Args:
        model:          UNet backbone (velocity field predictor).
        sigma:          CFM sigma parameter (unused, kept for config compat).
        num_classes:    Number of conditioning classes.
        image_size:     Spatial resolution of generated images.
        drop_rate:      Probability of zeroing out conditioning (CFG training).
        lr:             Peak learning rate for AdamW.
        weight_decay:   Weight decay for AdamW.
        warmup_steps:   Linear warmup steps before cosine decay.
        lognorm_mean:   Mean of logit-normal time distribution (0 = symmetric).
        lognorm_std:    Std of logit-normal time distribution (1 = moderate focus).
    """

    def __init__(
        self,
        model,
        sigma: float = 0.0,
        num_classes: int = 19,
        in_channels: int = 3,
        image_size: int = 64,
        drop_rate: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        warmup_steps: int = 2000,
        lognorm_mean: float = 0.0,
        lognorm_std: float = 1.0,
    ):
        super().__init__()
        self.model = model
        # CosineScheduler: alpha_t=sin(π/2·t), sigma_t=cos(π/2·t)
        # Smoother path than linear CondOT; preserves more mid-frequency face details
        self.fm = AffineProbPath(CosineScheduler())
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.image_size = image_size
        self.drop_rate = drop_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.lognorm_mean = lognorm_mean
        self.lognorm_std = lognorm_std

    def _sample_t(self, batch_size: int, device) -> torch.Tensor:
        """Logit-normal time sampling: concentrates steps near t≈0.5 (hard region).

        Face structure (eyes, nose, mouth layout) is formed at intermediate t,
        so sampling more densely there improves training efficiency.
        """
        eps = torch.randn(batch_size, device=device) * self.lognorm_std + self.lognorm_mean
        return torch.sigmoid(eps)

    def training_step(self, batch: tuple[torch.Tensor, dict], batch_idx: int):
        x1, cond_dict = batch
        cond = cond_dict.get("cond", None)  # [B, num_classes, H, W]

        x0 = torch.randn_like(x1)
        t = self._sample_t(x1.shape[0], x1.device)
        result = self.fm.sample(x0, x1, t)
        xt = result.x_t
        ut = result.dx_t

        # Classifier-free guidance: randomly zero out conditioning
        assert cond is not None, "cond must be provided (set drop_rate=0 to disable CFG)"
        if self.drop_rate > 0:
            mask = (torch.rand(x1.shape[0], device=x1.device) < self.drop_rate)
            cond = cond.clone()
            cond[mask] = 0.0

        # Scale t from [0,1] to [0,999] to match UNet timestep embedding range
        t_scaled = (t * 999).long()

        # 始终走 SPADE 路径
        vt = self.model(xt, t_scaled, y=cond)
        vt = vt[:, :self.in_channels]

        loss = F.mse_loss(vt, ut)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, dict], _batch_idx: int):
        x1, cond_dict = batch
        assert isinstance(cond_dict, dict)
        cond = cond_dict.get("cond", None)

        x0 = torch.randn_like(x1)
        t = self._sample_t(x1.shape[0], x1.device)
        result = self.fm.sample(x0, x1, t)
        xt, ut = result.x_t, result.dx_t

        assert cond is not None
        t_scaled = (t * 999).long()
        vt = self.model(xt, t_scaled, y=cond)
        vt = vt[:, :self.in_channels]

        loss = F.mse_loss(vt, ut)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Warmup + cosine decay: stabilizes early training and smoothly anneals lr
        total_steps = self.trainer.estimated_stepping_batches
        warmup = self.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    @torch.no_grad()
    def generate(self, cond=None, batch_size=4, num_steps=100, **kwargs):
        """ODESolver: integrate velocity field from t=0 (noise) to t=1 (data)."""
        device = next(self.model.parameters()).device
        x = torch.randn(batch_size, self.in_channels, self.image_size, self.image_size, device=device)

        cond_tensor = cond["cond"] if isinstance(cond, dict) else cond
        if cond_tensor is not None:
            cond_tensor = cond_tensor.to(device)

        # 包装 UNet，把 cond 通过 extras 传入
        unet = self.model
        in_channels = self.in_channels

        class CondModelWrapper(ModelWrapper):
            def forward(self, x, t, **extras):
                # ODESolver 传入的 t 是标量 tensor，扩展为 [B] 并缩放到 UNet 范围
                t_batch = t.expand(x.shape[0])
                t_scaled = (t_batch * 999).long()
                c = extras["cond"]
                vt = unet(x, t_scaled, y=c)
                return vt[:, :in_channels]

        solver = ODESolver(velocity_model=CondModelWrapper(unet))
        time_grid = torch.linspace(0, 1, num_steps + 1, device=device)

        result: torch.Tensor = solver.sample(  # type: ignore[assignment]
            x_init=x,
            step_size=None,
            method="midpoint",
            time_grid=time_grid,
            cond=cond_tensor,
        )

        return result.clamp(-1, 1)


if __name__ == "__main__":
    import lightning as pl
    from lightning.pytorch.loggers import WandbLogger
    from torch.utils.data import DataLoader, random_split
    from generative_models_lightning.backbones.cond_unet import UNetModel
    from data_module.ColoRadar.dataset import ColoRadarDataset
    from callbacks.plot_generated_data import PlotGeneratedDataCallback

    # ──parameters────────────────────────────────────────────────────────────
    DATA_ROOT       = "/home/local/Desktop/code/Datasets/ColoRadar/COLO_RPD_Dataset"
    SEQUENCES       = [f"{DATA_ROOT}/c4c_garage_run0"]
    IMAGE_SIZE      = 64
    BATCH_SIZE      = 8
    NUM_WORKERS     = 4
    MAX_EPOCHS      = 100
    LR              = 1e-4
    WEIGHT_DECAY    = 0.0
    WARMUP_STEPS    = 2000
    DROP_RATE       = 0.1
    IN_CHANNELS     = 1    # LiDAR BEV 单通道
    COND_CHANNELS   = 1    # 雷达 range-azimuth 单通道
    NUM_CLASSES     = COND_CHANNELS  # SPADE 条件通道数
    VAL_SPLIT       = 0.1
    SEED            = 42
    CKPT_DIR        = "checkpoints/flow"
    # ─────────────────────────────────────────────────────────────────────────

    pl.seed_everything(SEED)

    # 数据集
    dataset = ColoRadarDataset(image_size=IMAGE_SIZE, sequence=SEQUENCES)
    n_val   = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, persistent_workers=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, persistent_workers=True)

    # UNet backbone
    unet = UNetModel(
        image_size          = IMAGE_SIZE,
        in_channels         = IN_CHANNELS,
        model_channels      = 64,
        out_channels        = IN_CHANNELS * 2,
        num_res_blocks      = 2,
        attention_resolutions = (8, 16),
        num_classes         = NUM_CLASSES,
        dropout             = 0.0,
        channel_mult        = (1, 2, 4, 8),
    )

    model = FlowMatchingModel(
        model         = unet,
        in_channels   = IN_CHANNELS,
        image_size    = IMAGE_SIZE,
        drop_rate     = DROP_RATE,
        lr            = LR,
        weight_decay  = WEIGHT_DECAY,
        warmup_steps  = WARMUP_STEPS,
    )

    os.makedirs(CKPT_DIR, exist_ok=True)

    wandb_logger = WandbLogger(project="generative_models", entity="ELAB", save_dir=CKPT_DIR)
    sample_cb    = PlotGeneratedDataCallback(dataset, num_samples=20, log_every_n_epochs=5, cond_is_image=True)

    trainer = pl.Trainer(
        max_epochs          = MAX_EPOCHS,
        accelerator         = "auto",
        devices             = "auto",
        precision           = "32-true",
        default_root_dir    = CKPT_DIR,
        log_every_n_steps   = 10,
        logger              = wandb_logger,
        callbacks           = [sample_cb],
    )

    trainer.fit(model, train_loader, val_loader)

