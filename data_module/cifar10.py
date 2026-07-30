#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-07-29
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : data_module/cifar10.py
# @IDE     : vscode

"""Conditional CIFAR-10 Lightning data module."""

from __future__ import annotations

from pathlib import Path

import lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class _ClassConditionalDataset(Dataset):
    """Return the repository's ``{"x", "cond"}`` batch contract."""

    def __init__(self, dataset: Dataset, num_classes: int = 10) -> None:
        self.dataset = dataset
        self.num_classes = int(num_classes)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image, label = self.dataset[index]
        label_tensor = torch.as_tensor(label, dtype=torch.long)
        condition = F.one_hot(
            label_tensor,
            num_classes=self.num_classes,
        ).to(torch.float32)
        return {"x": image, "cond": condition}


class CIFAR10DataModule(pl.LightningDataModule):
    """Load CIFAR-10 or an offline FakeData-compatible smoke dataset."""

    num_classes = 10
    image_shape = (3, 32, 32)

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        val_size: int = 5000,
        seed: int = 42,
        download: bool = True,
        pin_memory: bool = True,
        use_fake_data: bool = False,
        fake_train_size: int = 256,
        fake_val_size: int = 64,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if val_size <= 0 or val_size >= 50000:
            raise ValueError("val_size must be between 1 and 49999")
        if fake_train_size <= 0 or fake_val_size <= 0:
            raise ValueError("FakeData split sizes must be positive")

        self.data_dir = str(Path(data_dir).expanduser())
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.val_size = int(val_size)
        self.seed = int(seed)
        self.download = bool(download)
        self.pin_memory = bool(pin_memory)
        self.use_fake_data = bool(use_fake_data)
        self.fake_train_size = int(fake_train_size)
        self.fake_val_size = int(fake_val_size)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def prepare_data(self) -> None:
        if self.use_fake_data:
            return
        datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=self.download,
        )
        datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=self.download,
        )

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.use_fake_data:
            total_size = self.fake_train_size + self.fake_val_size
            full_dataset: Dataset = datasets.FakeData(
                size=total_size,
                image_size=self.image_shape,
                num_classes=self.num_classes,
                transform=self.transform,
                random_offset=self.seed,
            )
            test_dataset: Dataset = datasets.FakeData(
                size=self.fake_val_size,
                image_size=self.image_shape,
                num_classes=self.num_classes,
                transform=self.transform,
                random_offset=self.seed + total_size,
            )
            split_sizes = [self.fake_train_size, self.fake_val_size]
        else:
            full_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                transform=self.transform,
                download=False,
            )
            test_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                transform=self.transform,
                download=False,
            )
            split_sizes = [len(full_dataset) - self.val_size, self.val_size]

        train_dataset, val_dataset = random_split(
            full_dataset,
            split_sizes,
            generator=torch.Generator().manual_seed(self.seed),
        )
        self.train_dataset = _ClassConditionalDataset(
            train_dataset,
            self.num_classes,
        )
        self.val_dataset = _ClassConditionalDataset(
            val_dataset,
            self.num_classes,
        )
        self.test_dataset = _ClassConditionalDataset(
            test_dataset,
            self.num_classes,
        )

    def _loader(
        self,
        dataset: Dataset | None,
        *,
        shuffle: bool,
    ) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Call setup() before requesting a dataloader")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
