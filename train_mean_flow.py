#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-06-03
# @Author  : Zhaoze Wang, Chenlin Lang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : /train_mean_flow.py
# @IDE     : vscode

"""
Mean Flow training entry based on train_flow.py, adapted for dense radar-image conditioning.
"""

from __future__ import annotations

from functools import partial
from typing import Union

import os
import torch
import torch.nn as nn

from einops import rearrange

from generative_models_lightning import BaseGenerativeModule
from generative_models_lightning.flow.meanflow.meanflow import (
    Normalizer,
    stopgrad,
)
from generative_models_lightning.flow.meanflow.models import MFUNet
from output_paths import CHECKPOINTS_ROOT


class DenseConditionalMeanFlow:
    """
    Dense Conditional Mean Flow implementation.
    """

    def __init__(
        self,
        channels: int = 1,
        image_shape: Union[int, tuple[int, int]] = 64,
        normalizer: list = ["mean_std", 0.0, 1.0],
        flow_ratio: float = 0.50,
        time_dist: list = ["lognorm", -0.4, 1.0],
        cfg_ratio: float = 0.10,
        cfg_scale: float | None = 2.0,
        cfg_uncond: str = "v",
        jvp_api: str = "autograd",
    ):
        self.channels = channels
        self.image_shape = (image_shape, image_shape) if isinstance(image_shape, int) else tuple(image_shape)
        self.normer = Normalizer.from_list(normalizer)
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        self.w = cfg_scale
        self.cfg_uncond = cfg_uncond
        self.jvp_api = jvp_api

        if jvp_api == "funtorch":
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == "autograd":
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True
        else:
            raise ValueError("jvp_api must be 'funtorch' or 'autograd'")

    def sample_t_r(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        dist_name = self.time_dist[0]
        if dist_name == "uniform":
            samples = torch.rand(batch_size, 2, device=device)
        elif dist_name == "lognorm":
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            samples = torch.randn(batch_size, 2, device=device) * sigma + mu
            samples = torch.sigmoid(samples)
        else:
            raise ValueError(f"unsupported time distribution: {dist_name}")

        t = torch.maximum(samples[:, 0], samples[:, 1])
        r = torch.minimum(samples[:, 0], samples[:, 1]).clone()

        num_selected = int(self.flow_ratio * batch_size)
        if num_selected > 0:
            indices = torch.randperm(batch_size, device=device)[:num_selected]
            r[indices] = t[indices]

        return t, r

    def loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        device = x.device
        valid_mask = (x > -0.99).to(dtype=x.dtype)

        t, r = self.sample_t_r(batch_size, device)
        t_ = rearrange(t, "b -> b 1 1 1")
        r_ = rearrange(r, "b -> b 1 1 1")

        e = torch.randn_like(x)
        x = self.normer.norm(x)
        z = (1.0 - t_) * x + t_ * e
        v = e - x

        cond_for_model = cond
        cfg_mask = None
        v_hat = v

        if cond is not None:
            uncond = torch.zeros_like(cond)
            if self.cfg_ratio > 0:
                cfg_mask = (torch.rand(batch_size, device=device) < self.cfg_ratio).view(-1, 1, 1, 1)
                cond_for_model = torch.where(cfg_mask, uncond, cond)
            if self.w is not None:
                with torch.no_grad():
                    u_t = model(z, t, t, uncond)
                v_hat = self.w * v + (1.0 - self.w) * u_t
                if self.cfg_uncond == "v" and cfg_mask is not None:
                    v_hat = torch.where(cfg_mask, v, v_hat)

        model_partial = partial(model, y=cond_for_model)
        jvp_args = (
            lambda z_cur, t_cur, r_cur: model_partial(z_cur, t_cur, r_cur),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )

        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        u_tgt = v_hat - (t_ - r_) * dudt
        error = u - stopgrad(u_tgt)
        error_sq = error.square() * valid_mask
        valid_counts = valid_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        delta_sq = error_sq.sum(dim=(1, 2, 3)) / valid_counts
        weights = 1.0 / (delta_sq + 1e-3).pow(0.5)
        loss = (stopgrad(weights) * delta_sq).mean()
        mse_val = ((stopgrad(error).square() * valid_mask).sum(dim=(1, 2, 3)) / valid_counts).mean()
        return loss, mse_val

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        cond: torch.Tensor | None,
        batch_size: int,
        sample_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        if cond is not None:
            batch_size = cond.shape[0]

        h, w = self.image_shape
        z = torch.randn(batch_size, self.channels, h, w, device=device)
        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)

        for idx in range(sample_steps):
            t = torch.full((batch_size,), t_vals[idx], device=device)
            r = torch.full((batch_size,), t_vals[idx + 1], device=device)
            step = rearrange(t - r, "b -> b 1 1 1")
            v = model(z, t, r, cond)
            z = z - step * v

        return self.normer.unnorm(z)


class MeanFlowModel(BaseGenerativeModule):
    """LightningModule that trains a dense-conditional Mean Flow model."""

    def __init__(
        self,
        model: nn.Module,
        in_channels: int = 1,
        image_shape: Union[int, tuple[int, int]] = 64,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        warmup_steps: int = 2000,
        flow_ratio: float = 0.50,
        time_dist: list = ["lognorm", -0.4, 1.0],
        cfg_ratio: float = 0.10,
        cfg_scale: float | None = 2.0,
        cfg_uncond: str = "v",
        jvp_api: str = "autograd",
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.model = model
        self.in_channels = in_channels
        self.image_shape = (image_shape, image_shape) if isinstance(image_shape, int) else tuple(image_shape)
        self.warmup_steps = warmup_steps
        self.meanflow = DenseConditionalMeanFlow(
            channels=in_channels,
            image_shape=self.image_shape,
            normalizer=["mean_std", 0.0, 1.0],
            flow_ratio=flow_ratio,
            time_dist=time_dist,
            cfg_ratio=cfg_ratio,
            cfg_scale=cfg_scale,
            cfg_uncond=cfg_uncond,
            jvp_api=jvp_api,
        )

    def training_step(self, batch: tuple[torch.Tensor, dict], _batch_idx: int):
        x, cond_dict = batch
        cond = cond_dict.get("cond", None)
        loss, mse_val = self.meanflow.loss(self.model, x, cond)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at global_step={int(self.global_step)}, "
                f"epoch={int(self.current_epoch)}: loss={float(loss.detach().item())}"
            )
        if not torch.isfinite(mse_val):
            raise RuntimeError(
                f"Non-finite training mse at global_step={int(self.global_step)}, "
                f"epoch={int(self.current_epoch)}: mse={float(mse_val.detach().item())}"
            )
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, dict], _batch_idx: int):
        x, cond_dict = batch
        cond = cond_dict.get("cond", None)
        loss, mse_val = self.meanflow.loss(self.model, x, cond)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite validation loss at global_step={int(self.global_step)}, "
                f"epoch={int(self.current_epoch)}: loss={float(loss.detach().item())}"
            )
        if not torch.isfinite(mse_val):
            raise RuntimeError(
                f"Non-finite validation mse at global_step={int(self.global_step)}, "
                f"epoch={int(self.current_epoch)}: mse={float(mse_val.detach().item())}"
            )
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=int(x.shape[0]),
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

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

    @torch.inference_mode()
    def generate(
        self,
        cond: dict | torch.Tensor | None = None,
        batch_size: int = 4,
        num_steps: int = 10,
        sample_steps: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if sample_steps is not None:
            num_steps = sample_steps

        device = next(self.model.parameters()).device
        cond_tensor = cond["cond"] if isinstance(cond, dict) else cond
        if cond_tensor is not None:
            cond_tensor = cond_tensor.to(device)

        # Keep inference precision consistent with mixed-precision training so
        # the epoch-end WandB image callback does not hit Half-vs-float module
        # mismatches during sampling.
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            samples = self.meanflow.sample(
                model=self.model,
                cond=cond_tensor,
                batch_size=batch_size,
                sample_steps=num_steps,
                device=device,
            )
        return samples.clamp(-1, 1)


if __name__ == "__main__":
    import lightning as pl
    from lightning.fabric.plugins.environments import LightningEnvironment
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
    from torch.utils.data import DataLoader, Subset

    from callbacks.plot_generated_data import PlotGeneratedDataCallback
    from data_module.RadarPromptLiDAR.dataset import RadarPromptLiDARDataset
    from generative_models_lightning.backbones.cond_unet import UNetModel

    # Fixed training config.
    ALIGNED_ROOT = "/home/local/Desktop/code/Datasets/processing/aligned"
    TRAIN_RUNS = (
        "ec_courtyard_run0",
        "ec_courtyard_run1",
        "ec_courtyard_run2",
    )
    COND_DIR_NAME = "radar"
    TARGET_DIR_NAME = "camera"
    # Full 800x1280 inputs OOM because the UNet's middle attention block would
    # still run on a very large feature map. Start from a smaller fixed size.
    IMAGE_SHAPE = (160, 256)

    BATCH_SIZE = 2
    NUM_WORKERS = 2
    MAX_EPOCHS = 500
    LR = 5e-5
    WEIGHT_DECAY = 0.0
    WARMUP_STEPS = 2000
    IN_CHANNELS = 1

    FLOW_RATIO = 0.50
    TIME_DIST_MU = -0.4
    TIME_DIST_SIGMA = 1.0
    TIME_DIST = ["lognorm", TIME_DIST_MU, TIME_DIST_SIGMA]
    # The aligned radar condition is a sparse single-channel metric depth image,
    # not a learned unconditional token like the original class-label MeanFlow.
    # Disabling CFG distillation avoids injecting an unstable dense-uncond branch.
    CFG_RATIO = 0.0
    CFG_SCALE = None
    CFG_UNCOND = "v"

    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    SEED = 42
    CKPT_DIR = str(CHECKPOINTS_ROOT / "mean_flow")
    RESUME_CKPT = os.path.join(CKPT_DIR, "last.ckpt")

    # Batch-lift config: keep 64 base channels, but compress dense radar
    # conditioning and remove the extra deepest stage to recover memory.
    MODEL_CHANNELS = 64
    NUM_RES_BLOCKS = 1
    ATTENTION_RESOLUTIONS = ()
    CHANNEL_MULT = (1, 2, 4)
    # This UNet's custom checkpoint autograd.Function is not compatible with
    # the funtorch JVP path used by Mean Flow here.
    USE_GRAD_CHECKPOINTING = False

    USE_WANDB = True
    RUN_NAME = "radar_mean_flow"
    ENABLE_SAMPLE_CALLBACK = True
    SAMPLE_NUM_SAMPLES = 1
    SAMPLE_LOG_EVERY_N_EPOCHS = 5
    SAMPLE_STEPS = 4
    FAIL_FAST_ON_NONFINITE = True
    SAVE_EVERY_N_EPOCHS = 10
    AUTO_RESUME = False
    TRAINER_PRECISION = "16-mixed"
    MATMUL_PRECISION = "high"

    USE_GPU = True
    GPU_INDICES = [0, 1]
    CUDA_ALLOC_CONF = "expandable_segments:True"

    if USE_GPU:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", CUDA_ALLOC_CONF)

    accelerator = "gpu" if USE_GPU and torch.cuda.is_available() else "cpu"
    devices: int | list[int] = GPU_INDICES if accelerator == "gpu" else 1
    TRAINER_STRATEGY = (
        "ddp_find_unused_parameters_false"
        if accelerator == "gpu" and isinstance(devices, list) and len(devices) > 1
        else "auto"
    )

    LIMIT_TRAIN_BATCHES = 0.5
    LIMIT_VAL_BATCHES = 0.5
    FAST_DEV_RUN = False

    if accelerator == "gpu":
        torch.set_float32_matmul_precision(MATMUL_PRECISION)

    resolved_precision = TRAINER_PRECISION if accelerator == "gpu" else "32-true"

    pl.seed_everything(SEED)

    dataset = RadarPromptLiDARDataset(
        aligned_root=ALIGNED_ROOT,
        run_names=TRAIN_RUNS,
        cond_dir_name=COND_DIR_NAME,
        target_dir_name=TARGET_DIR_NAME,
        spatial_size=IMAGE_SHAPE,
        cond_normalize_to_minus1_1=False,
        target_normalize_to_minus1_1=True,
    )

    sample_x, sample_extra = dataset[0]
    sample_cond = sample_extra["cond"]
    COND_CHANNELS = int(sample_cond.shape[0])
    EXTRA_TIME_COND_CHANNELS = 2
    REDUCED_COND_CHANNELS = 14
    MODEL_COND_CHANNELS = REDUCED_COND_CHANNELS + EXTRA_TIME_COND_CHANNELS

    print(
        f"Mean Flow dataset ready: samples={len(dataset)}, "
        f"target_shape={tuple(sample_x.shape)}, cond_shape={tuple(sample_cond.shape)}, "
        f"cond_proj={COND_CHANNELS}->{REDUCED_COND_CHANNELS}+{EXTRA_TIME_COND_CHANNELS}, "
        f"runs={TRAIN_RUNS}, accelerator={accelerator}, devices={devices}"
    )

    num_samples = len(dataset)
    n_train = int(num_samples * TRAIN_SPLIT)
    n_val = int(num_samples * VAL_SPLIT)
    n_test = min(num_samples - n_train - n_val, int(num_samples * TEST_SPLIT))
    if n_train + n_val + n_test < num_samples:
        n_train += num_samples - (n_train + n_val + n_test)

    shuffled_indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(SEED)).tolist()
    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train:n_train + n_val]
    test_indices = shuffled_indices[n_train + n_val:n_train + n_val + n_test]

    print(
        f"Dataset split: train={len(train_indices)}, "
        f"val={len(val_indices)}, test={len(test_indices)}"
    )

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices) if val_indices else Subset(dataset, train_indices[: min(len(train_indices), 8)])
    test_set = Subset(dataset, test_indices)

    persistent_workers = NUM_WORKERS > 0
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=persistent_workers,
    )

    unet = UNetModel(
        image_size=IMAGE_SHAPE,
        in_channels=IN_CHANNELS,
        model_channels=MODEL_CHANNELS,
        out_channels=IN_CHANNELS,
        num_res_blocks=NUM_RES_BLOCKS,
        attention_resolutions=ATTENTION_RESOLUTIONS,
        num_classes=MODEL_COND_CHANNELS,
        dropout=0.0,
        channel_mult=CHANNEL_MULT,
        use_checkpoint=USE_GRAD_CHECKPOINTING,
        use_fp16=accelerator == "gpu" and resolved_precision.startswith("16"),
    )
    backbone = MFUNet(
        unet,
        in_channels=IN_CHANNELS,
        cond_in_channels=COND_CHANNELS,
        cond_proj_channels=REDUCED_COND_CHANNELS,
    )

    model = MeanFlowModel(
        model=backbone,
        in_channels=IN_CHANNELS,
        image_shape=IMAGE_SHAPE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        flow_ratio=FLOW_RATIO,
        time_dist=TIME_DIST,
        cfg_ratio=CFG_RATIO,
        cfg_scale=CFG_SCALE,
        cfg_uncond=CFG_UNCOND,
        jvp_api="funtorch",
    )

    os.makedirs(CKPT_DIR, exist_ok=True)

    if USE_WANDB:
        logger = WandbLogger(
            project="generative_models",
            entity="ELAB",
            save_dir=CKPT_DIR,
            name=RUN_NAME,
        )
    else:
        logger = CSVLogger(save_dir=CKPT_DIR, name=RUN_NAME)
    callbacks: list[Callback] = []
    if ENABLE_SAMPLE_CALLBACK:
        sample_cb = PlotGeneratedDataCallback(
            dataset,
            num_samples=SAMPLE_NUM_SAMPLES,
            log_every_n_epochs=SAMPLE_LOG_EVERY_N_EPOCHS,
            cond_is_image=True,
            sample_steps=SAMPLE_STEPS,
            stop_on_nonfinite=FAIL_FAST_ON_NONFINITE,
        )
        callbacks.append(sample_cb)

    latest_ckpt_cb = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename="meanflow-latest-{epoch:03d}",
        monitor="epoch",
        mode="max",
        every_n_epochs=1,
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        save_on_exception=True,
    )
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(CKPT_DIR, "periodic"),
        filename="meanflow-{epoch:03d}",
        every_n_epochs=SAVE_EVERY_N_EPOCHS,
        save_top_k=-1,
        save_on_exception=True,
    )
    best_ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(CKPT_DIR, "best"),
        filename="meanflow-best-{epoch:03d}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
        save_on_exception=True,
    )
    callbacks.extend([latest_ckpt_cb, ckpt_cb, best_ckpt_cb])

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=accelerator,
        strategy=TRAINER_STRATEGY,
        devices=devices,
        precision=resolved_precision,
        default_root_dir=CKPT_DIR,
        log_every_n_steps=1,
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=FAST_DEV_RUN,
        limit_train_batches=LIMIT_TRAIN_BATCHES,
        limit_val_batches=LIMIT_VAL_BATCHES,
        plugins=LightningEnvironment(),
    )

    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=RESUME_CKPT if AUTO_RESUME and os.path.exists(RESUME_CKPT) else None,
    )
