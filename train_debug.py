#!/usr/bin/env python3

"""Debug entrypoint for the modular radar-LiDAR MVP pipeline."""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import hydra
import torch
import torch.nn as nn
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from data_module.ColoRadar.datamodule import build_dataloader, build_dataset, resolve_runtime_cfg


class DummyModel(nn.Module):
    def forward(self, radar):
        print("Radar input shape:", tuple(radar.shape))
        return radar


def describe_sample(sample):
    print("Sample keys:", list(sample.keys()))
    print("Sample radar shape:", tuple(sample["radar"].shape), "dtype:", sample["radar"].dtype)
    print("Sample lidar shape:", tuple(sample["lidar"].shape), "dtype:", sample["lidar"].dtype)
    print("Sample meta:", sample["meta"])


def describe_batch(batch):
    print("Batch radar shape:", tuple(batch["radar"].shape), "dtype:", batch["radar"].dtype)
    print("LiDAR point cloud sizes:", [tuple(points.shape) for points in batch["lidar"]])
    print("Batch meta count:", len(batch["meta"]))


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    os.chdir(get_original_cwd())
    runtime_cfg = resolve_runtime_cfg(cfg)

    print(
        "Selections:",
        f"dataset={runtime_cfg.selection.dataset}",
        f"radar={runtime_cfg.selection.radar}",
        f"lidar={runtime_cfg.selection.lidar}",
        f"align={runtime_cfg.selection.align}",
        f"dataloader={runtime_cfg.selection.dataloader}",
    )
    print("Dataset root:", runtime_cfg.dataset.root)

    dataset = build_dataset(cfg)
    print("Dataset length:", len(dataset))

    sample = dataset[0]
    describe_sample(sample)

    dataloader = build_dataloader(cfg, dataset=dataset)
    batch = next(iter(dataloader))
    describe_batch(batch)

    model = DummyModel().eval()
    with torch.no_grad():
        _ = model(batch["radar"])

    print("Dummy forward pass: success")


if __name__ == "__main__":
    main()
