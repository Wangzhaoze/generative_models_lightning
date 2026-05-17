#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""RaDEFT interface placeholder for future raw radar-LiDAR support."""

from data_module.ColoRadar.dataset import BaseMultiSensorDataset


class RaDelftDataset(BaseMultiSensorDataset):
    """Placeholder dataset adapter.

    TODO: Implement RaDEFT raw file indexing, loading, calibration, and sync
    using the same BaseMultiSensorDataset + MultiSensorPipeline interface.
    """

    def build_index(self):
        raise NotImplementedError("RaDEFT support is not part of the first MVP.")

    def load_raw(self, meta):
        _ = meta
        raise NotImplementedError("RaDEFT support is not part of the first MVP.")
