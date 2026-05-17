#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared builders for the radar-LiDAR MVP pipeline."""

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data_module.ColoRadar.dataset import (
    Aligner,
    LidarProcessor,
    MultiSensorPipeline,
    RadarProcessor,
)
from data_module.ColoRadarPlus.dataset import ColoRadarDataset
from data_module.RaDelft.dataset import RaDelftDataset


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def resolve_runtime_cfg(cfg):
    """Resolve active configs from the single Hydra YAML file."""

    container = OmegaConf.to_container(cfg, resolve=True)

    dataset_key = container["dataset"]
    radar_key = container["radar"]
    lidar_key = container["lidar"]
    align_key = container["align"]
    dataloader_key = container["dataloader"]

    runtime_cfg = {
        "selection": {
            "dataset": dataset_key,
            "radar": radar_key,
            "lidar": lidar_key,
            "align": align_key,
            "dataloader": dataloader_key,
        },
        "dataset": container["datasets"][dataset_key],
        "radar": container["radars"][radar_key],
        "lidar": container["lidars"][lidar_key],
        "align": container["aligns"][align_key],
        "dataloader": container["dataloaders"][dataloader_key],
    }
    return _to_namespace(runtime_cfg)


def build_dataset(cfg):
    runtime_cfg = resolve_runtime_cfg(cfg)

    pipeline = MultiSensorPipeline(
        aligner=Aligner(runtime_cfg.align),
        radar_processor=RadarProcessor(runtime_cfg.radar),
        lidar_processor=LidarProcessor(runtime_cfg.lidar),
    )

    dataset_name = str(runtime_cfg.dataset.name).lower()
    if dataset_name in {"coloradar", "coloradar_plus"}:
        return ColoRadarDataset(cfg=runtime_cfg, pipeline=pipeline)

    if dataset_name == "radeft":
        # TODO: Add the real RaDEFT raw-data adapter here in a later step.
        raise NotImplementedError("RaDEFT is intentionally not implemented in this MVP.")

    raise ValueError(f"Unsupported dataset: {runtime_cfg.dataset.name}")


def multisensor_collate(batch):
    """Stack radar tensors and keep variable-size LiDAR clouds as a list."""

    radar_tensors = [sample["radar"] for sample in batch]
    lidar_tensors = [sample["lidar"] for sample in batch]
    meta = [sample["meta"] for sample in batch]
    return {
        "radar": torch.stack(radar_tensors, dim=0),
        "lidar": lidar_tensors,
        "meta": meta,
    }


def build_dataloader(cfg, dataset=None):
    runtime_cfg = resolve_runtime_cfg(cfg)
    if dataset is None:
        dataset = build_dataset(cfg)

    return DataLoader(
        dataset,
        batch_size=int(runtime_cfg.dataloader.batch_size),
        num_workers=int(runtime_cfg.dataloader.num_workers),
        shuffle=bool(runtime_cfg.dataloader.shuffle),
        pin_memory=bool(getattr(runtime_cfg.dataloader, "pin_memory", False)),
        drop_last=bool(getattr(runtime_cfg.dataloader, "drop_last", False)),
        collate_fn=multisensor_collate,
    )


class ColoRadarDataModule:
    """Small utility wrapper around the new dataset/dataloader builders."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = None

    def setup(self, stage: str | None = None) -> None:
        _ = stage
        self.dataset = build_dataset(self.cfg)

    def train_dataloader(self):
        if self.dataset is None:
            self.setup()
        return build_dataloader(self.cfg, dataset=self.dataset)
