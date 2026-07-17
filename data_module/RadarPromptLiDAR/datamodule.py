#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convenience dataloader builders for aligned radar -> PromptDA depth training."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import lightning as pl
from torch.utils.data import DataLoader

from .dataset import DEFAULT_ALIGNED_ROOT, DEFAULT_RUN_NAMES, RadarPromptLiDARDataset


def build_radar_prompt_lidar_dataloader(
    aligned_root: str | Path = DEFAULT_ALIGNED_ROOT,
    run_names: Iterable[str] = DEFAULT_RUN_NAMES,
    cond_dir_name: str = "radar",
    target_dir_name: str = "camera",
    spatial_size: int | tuple[int, int] | None = None,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    persistent_workers: bool | None = None,
    match_target_to_condition: bool = True,
):
    dataset = RadarPromptLiDARDataset(
        aligned_root=aligned_root,
        run_names=run_names,
        cond_dir_name=cond_dir_name,
        target_dir_name=target_dir_name,
        spatial_size=spatial_size,
        match_target_to_condition=match_target_to_condition,
    )

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )


class RadarPromptLiDARDataModule(pl.LightningDataModule):
    """Minimal wrapper mirroring the repo's lightweight datamodule style."""

    def __init__(
        self,
        aligned_root: str | Path = DEFAULT_ALIGNED_ROOT,
        run_names: Iterable[str] = DEFAULT_RUN_NAMES,
        cond_dir_name: str = "radar",
        target_dir_name: str = "camera",
        spatial_size: int | tuple[int, int] | None = None,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        match_target_to_condition: bool = True,
    ):
        super().__init__()
        self.aligned_root = aligned_root
        self.run_names = tuple(str(name) for name in run_names)
        self.cond_dir_name = cond_dir_name
        self.target_dir_name = target_dir_name
        self.spatial_size = spatial_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.match_target_to_condition = match_target_to_condition
        self.dataset = None

    def setup(self, stage: str | None = None) -> None:
        _ = stage
        self.dataset = RadarPromptLiDARDataset(
            aligned_root=self.aligned_root,
            run_names=self.run_names,
            cond_dir_name=self.cond_dir_name,
            target_dir_name=self.target_dir_name,
            spatial_size=self.spatial_size,
            match_target_to_condition=self.match_target_to_condition,
        )

    def train_dataloader(self):
        if self.dataset is None:
            self.setup()

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.num_workers > 0,
        )
