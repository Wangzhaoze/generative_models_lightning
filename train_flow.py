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
        image_shape: Union[int, tuple[int, int]] = 64,
        drop_rate: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        warmup_steps: int = 2000,
        lognorm_mean: float = 0.0,
        lognorm_std: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.fm = AffineProbPath(CosineScheduler())
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.image_shape = (image_shape, image_shape) if isinstance(image_shape, int) else tuple(image_shape)
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
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
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
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
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
    def generate(self, cond=None, batch_size=4, num_steps=100, sample_steps=None, **kwargs):
        """ODESolver: integrate velocity field from t=0 (noise) to t=1 (data)."""
        if sample_steps is not None:
            num_steps = sample_steps
        device = next(self.model.parameters()).device
        x = torch.randn(batch_size, self.in_channels, *self.image_shape, device=device)

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
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader, Subset
    from generative_models_lightning.backbones.cond_unet import UNetModel
    from data_module.ColoRadar.dataset import ColoRadarDataset3D
    from callbacks.plot_generated_data import PlotGeneratedDataCallback

    # ── parameters ───────────────────────────────────────────────────────────
    DATA_ROOT       = "/home/local/Desktop/code/Datasets/coloradar_plus/kitti"
    RUN_ROOTS       = [
        f"{DATA_ROOT}/ec_courtyard_run0",
        #f"{DATA_ROOT}/ec_courtyard_run1",
        #f"{DATA_ROOT}/ec_courtyard_run2",
        #f"{DATA_ROOT}/ec_courtyard_run3",
    ]
    SEQUENCES       = [
        f"{run_root}/seq" for run_root in RUN_ROOTS
    ]
    # Training data projection mode:
    # - "direct_frame": no accumulation, project each frame's own point cloud to range image
    # - "accumulated_scene": build/use merged scene point clouds, then project by pose
    TRAIN_PROJECTION_MODE = "direct_frame"
    # Cache filename inside each run's cache dir. Leave None to follow the dataset default:
    # - direct_frame -> direct_frame_range_images.pt
    # - accumulated_scene -> range_images.pt
    TRAIN_RANGE_IMAGE_CACHE_NAME = None
    DATA_CACHE_DIR  = "per_run"
    IMAGE_SHAPE     = (60, 128)   # LiDAR range image 实际空间尺寸 [H, W]
    BATCH_SIZE      = 16
    NUM_WORKERS     = 4
    MAX_EPOCHS      = 500
    LR              = 1e-4
    WEIGHT_DECAY    = 0.0
    WARMUP_STEPS    = 2000
    DROP_RATE       = 0.1
    IN_CHANNELS     = 1    # LiDAR range image 单通道
    COND_CHANNELS   = 16   # 雷达 range image 16 通道
    NUM_CLASSES     = COND_CHANNELS
    TRAIN_SPLIT     = 0.8
    VAL_SPLIT       = 0.1
    TEST_SPLIT      = 0.1
    SEED            = 42
    CKPT_DIR        = "checkpoints/flow"
    RESUME_CKPT     = os.path.join(CKPT_DIR, "last.ckpt")
    GPU_ID          = 1
    # ─────────────────────────────────────────────────────────────────────────

    pl.seed_everything(SEED)

    # 数据集（legacy 模式：直接加载预处理好的 .npy 点云）
    dataset = ColoRadarDataset3D(
        image_size=IMAGE_SHAPE[0],
        sequence=SEQUENCES,
        projection_mode=TRAIN_PROJECTION_MODE,
        range_image_cache_name=TRAIN_RANGE_IMAGE_CACHE_NAME,
        cache_dir=DATA_CACHE_DIR,
    )
    print(
        f"Training dataset mode: projection_mode={TRAIN_PROJECTION_MODE}, "
        f"range_image_cache_name={TRAIN_RANGE_IMAGE_CACHE_NAME}"
    )

    def _matched_frame_count(seq: str) -> int:
        lidar_dir = os.path.join(seq, "lidar_pcl")
        radar_dir = os.path.join(seq, "pcl_npy")
        lidar_ids = {
            int(os.path.splitext(name)[0])
            for name in os.listdir(lidar_dir)
            if name.endswith(".npy")
        }
        radar_ids = {
            int(os.path.splitext(name)[0])
            for name in os.listdir(radar_dir)
            if name.endswith(".npy")
        }
        return len(lidar_ids & radar_ids)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    offset = 0
    for seq in SEQUENCES:
        n = _matched_frame_count(seq)
        n_train = int(n * TRAIN_SPLIT)
        n_val = int(n * VAL_SPLIT)
        train_indices.extend(range(offset, offset + n_train))
        val_indices.extend(range(offset + n_train, offset + n_train + n_val))
        test_indices.extend(range(offset + n_train + n_val, offset + n))
        offset += n

    assert offset == len(dataset), (
        f"Split counts ({offset}) do not match dataset length ({len(dataset)})"
    )
    print(
        f"Dataset split: train={len(train_indices)}, "
        f"val={len(val_indices)}, test={len(test_indices)}"
    )

    train_set = Subset(dataset, train_indices)
    val_set   = Subset(dataset, val_indices)
    test_set  = Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, persistent_workers=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, persistent_workers=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, persistent_workers=True)

    # UNet backbone（out_channels=IN_CHANNELS*2 仅为兼容接口，flow 只用前 IN_CHANNELS 通道）
    unet = UNetModel(
        image_size            = IMAGE_SHAPE,
        in_channels           = IN_CHANNELS,
        model_channels        = 64,
        out_channels          = IN_CHANNELS * 2,
        num_res_blocks        = 2,
        attention_resolutions = (2, 4),
        num_classes           = NUM_CLASSES,
        dropout               = 0.0,
        channel_mult          = (1, 2, 4),
    )

    model = FlowMatchingModel(
        model         = unet,
        in_channels   = IN_CHANNELS,
        image_shape   = IMAGE_SHAPE,
        drop_rate     = DROP_RATE,
        lr            = LR,
        weight_decay  = WEIGHT_DECAY,
        warmup_steps  = WARMUP_STEPS,
    )

    os.makedirs(CKPT_DIR, exist_ok=True)

    wandb_logger = WandbLogger(project="generative_models", entity="ELAB", save_dir=CKPT_DIR,
                               name="radar_flow")
    sample_cb    = PlotGeneratedDataCallback(
        dataset,
        num_samples=4,
        log_every_n_epochs=1,
        cond_is_image=True,
        sample_steps=50,
    )

    class SaveLatestCheckpoint(pl.Callback):
        """Overwrite last.ckpt every epoch, independent of validation-loss ranking."""

        def __init__(self, dirpath: str, filename: str = "last.ckpt"):
            self.path = os.path.join(dirpath, filename)

        def _save(self, trainer: pl.Trainer) -> None:
            if trainer.sanity_checking or not trainer.is_global_zero:
                return
            trainer.save_checkpoint(self.path)

        def on_validation_end(self, trainer: pl.Trainer, _pl_module: pl.LightningModule) -> None:
            self._save(trainer)

        def on_exception(
            self,
            trainer: pl.Trainer,
            _pl_module: pl.LightningModule,
            _exception: BaseException,
        ) -> None:
            self._save(trainer)

    latest_ckpt_cb = SaveLatestCheckpoint(CKPT_DIR)
    ckpt_cb = ModelCheckpoint(
        dirpath            = os.path.join(CKPT_DIR, "periodic"),
        filename           = "flow2-{epoch:03d}",
        every_n_epochs     = 10,
        save_top_k         = -1,
        save_on_exception  = True,
    )

    trainer = pl.Trainer(
        max_epochs          = MAX_EPOCHS,
        accelerator         = "gpu",
        devices             = [GPU_ID],
        precision           = "32-true",
        default_root_dir    = CKPT_DIR,
        log_every_n_steps   = 1,
        logger              = wandb_logger,
        callbacks           = [sample_cb, latest_ckpt_cb, ckpt_cb],
    )

    trainer.fit(
        model,
        train_loader,
        val_loader,
        #ckpt_path=RESUME_CKPT if os.path.exists(RESUME_CKPT) else None,
    )
