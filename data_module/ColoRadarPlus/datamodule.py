#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ColoRadar+ datamodule wrapper for the MVP pipeline."""

from data_module.ColoRadar.datamodule import build_dataloader, build_dataset


class ColoRadarPlusDataModule:
    """Thin wrapper so the new MVP stays compatible with existing entrypoints."""

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
