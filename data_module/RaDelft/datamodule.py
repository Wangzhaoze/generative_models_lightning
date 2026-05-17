#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""RaDEFT datamodule placeholder."""


class RaDelftDataModule:
    """Keeps the extension point explicit without implementing it yet."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    def setup(self, stage: str | None = None) -> None:
        _ = stage
        raise NotImplementedError("RaDEFT is intentionally not implemented in this MVP.")
