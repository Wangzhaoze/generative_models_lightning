#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Small debug entrypoint for dense-conditional Mean Flow training."""

from __future__ import annotations

import os

import lightning as pl
import torch
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Subset

from data_module.RadarPromptLiDAR.dataset import RadarPromptLiDARDataset
from generative_models_lightning.backbones.cond_unet import UNetModel
from generative_models_lightning.flow.meanflow.models import MFUNet
from output_paths import CHECKPOINTS_ROOT
from train_mean_flow import MeanFlowModel


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")# keep clean out put

    ALIGNED_ROOT = "/home/local/Desktop/code/Datasets/processing/aligned"
    TRAIN_RUNS = (
        "ec_courtyard_run0",
        "ec_courtyard_run1",
        "ec_courtyard_run2",
    )
    COND_DIR_NAME = "radar"
    TARGET_DIR_NAME = "camera"
    IMAGE_SHAPE = (160, 256)

    BATCH_SIZE = 1
    NUM_WORKERS = 0
    MAX_EPOCHS = 3
    LR = 2e-5
    WEIGHT_DECAY = 0.0
    WARMUP_STEPS = 8
    IN_CHANNELS = 1

    FLOW_RATIO = 0.50
    TIME_DIST = ["lognorm", -0.4, 1.0]
    CFG_RATIO = 0.0
    CFG_SCALE = None
    CFG_UNCOND = "v"

    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    SEED = 42
    CKPT_DIR = str(CHECKPOINTS_ROOT / "mean_flow_debug")

    MODEL_CHANNELS = 32
    NUM_RES_BLOCKS = 1
    ATTENTION_RESOLUTIONS = ()
    CHANNEL_MULT = (1, 2, 4, 4)
    USE_GRAD_CHECKPOINTING = False

    USE_GPU = True
    GPU_INDICES = [0]
    TRAINER_PRECISION = "32-true"
    MATMUL_PRECISION = "high"
    CUDA_ALLOC_CONF = "expandable_segments:True"

    DEBUG_TRAIN_SAMPLES = 8
    DEBUG_VAL_SAMPLES = 4
    LIMIT_TRAIN_BATCHES = 2
    LIMIT_VAL_BATCHES = 1
    FAST_DEV_RUN = False
    DETECT_ANOMALY = True
    GRADIENT_CLIP_VAL = 1.0

    if USE_GPU:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", CUDA_ALLOC_CONF)

    accelerator = "gpu" if USE_GPU and torch.cuda.is_available() else "cpu"
    devices: int | list[int] = GPU_INDICES if accelerator == "gpu" else 1
    trainer_strategy = (
        "ddp_find_unused_parameters_false"
        if accelerator == "gpu" and isinstance(devices, list) and len(devices) > 1
        else "auto"
    )

    if accelerator == "gpu":
        torch.set_float32_matmul_precision(MATMUL_PRECISION)

    pl.seed_everything(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)

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
    cond_channels = int(sample_cond.shape[0])
    model_cond_channels = cond_channels + 2

    num_samples = len(dataset)
    n_train = int(num_samples * TRAIN_SPLIT)
    n_val = int(num_samples * VAL_SPLIT)
    n_test = min(num_samples - n_train - n_val, int(num_samples * TEST_SPLIT))
    if n_train + n_val + n_test < num_samples:
        n_train += num_samples - (n_train + n_val + n_test)

    shuffled_indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(SEED)).tolist()
    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train:n_train + n_val]
    if DEBUG_TRAIN_SAMPLES is not None:
        train_indices = train_indices[: min(len(train_indices), DEBUG_TRAIN_SAMPLES)]
    if DEBUG_VAL_SAMPLES is not None:
        val_indices = val_indices[: min(len(val_indices), DEBUG_VAL_SAMPLES)]

    if not val_indices:
        val_indices = train_indices[: min(len(train_indices), 2)]

    print(
        f"Debug Mean Flow dataset: samples={len(dataset)}, "
        f"target_shape={tuple(sample_x.shape)}, cond_shape={tuple(sample_cond.shape)}, "
        f"train={len(train_indices)}, val={len(val_indices)}, "
        f"accelerator={accelerator}, devices={devices}"
    )

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=False,
    )

    unet = UNetModel(
        image_size=IMAGE_SHAPE,
        in_channels=IN_CHANNELS,
        model_channels=MODEL_CHANNELS,
        out_channels=IN_CHANNELS * 2,
        num_res_blocks=NUM_RES_BLOCKS,
        attention_resolutions=ATTENTION_RESOLUTIONS,
        num_classes=model_cond_channels,
        dropout=0.0,
        channel_mult=CHANNEL_MULT,
        use_checkpoint=USE_GRAD_CHECKPOINTING,
        use_fp16=False,
    )
    backbone = MFUNet(unet, in_channels=IN_CHANNELS)

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

    logger = CSVLogger(save_dir=CKPT_DIR, name="radar_mean_flow_debug_small")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=accelerator,
        strategy=trainer_strategy,
        devices=devices,
        precision=TRAINER_PRECISION,
        default_root_dir=CKPT_DIR,
        log_every_n_steps=1,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        fast_dev_run=FAST_DEV_RUN,
        limit_train_batches=LIMIT_TRAIN_BATCHES,
        limit_val_batches=LIMIT_VAL_BATCHES,
        num_sanity_val_steps=0,
        detect_anomaly=DETECT_ANOMALY,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        plugins=LightningEnvironment(),
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
