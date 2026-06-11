#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dataset for aligned radar tensors paired with PromptDA depth PNGs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_ALIGNED_ROOT = Path("/home/local/Desktop/code/Datasets/processing/aligned")
DEFAULT_RUN_NAMES = (
    "ec_courtyard_run0",
    "ec_courtyard_run1",
    "ec_courtyard_run2",
)


@dataclass(frozen=True)
class RadarPromptLiDARSample:
    run_name: str
    pair_index: int
    cond_path: Path
    target_path: Path


class RadarPromptLiDARDataset(Dataset):
    """Load aligned radar/camera depth pairs from the preprocessing output."""

    def __init__(
        self,
        aligned_root: str | Path = DEFAULT_ALIGNED_ROOT,
        run_names: Iterable[str] = DEFAULT_RUN_NAMES,
        cond_dir_name: str = "radar",
        target_dir_name: str = "camera",
        spatial_size: int | tuple[int, int] | None = None,
        # Keep sparse radar metric-depth conditions in [0, 1] so empty pixels
        # stay at 0 instead of collapsing the whole condition map near -1.
        cond_normalize_to_minus1_1: bool = False,
        target_normalize_to_minus1_1: bool = True,
        cond_value_scale: float | None = 16000.0,
        target_value_scale: float | None = 16000.0,
        match_target_to_condition: bool = True,
    ):
        self.aligned_root = Path(aligned_root)
        self.run_names = tuple(str(name) for name in run_names)
        self.cond_dir_name = cond_dir_name
        self.target_dir_name = target_dir_name
        self.spatial_size = self._normalize_spatial_size(spatial_size)
        self.cond_normalize_to_minus1_1 = cond_normalize_to_minus1_1
        self.target_normalize_to_minus1_1 = target_normalize_to_minus1_1
        self.cond_value_scale = cond_value_scale
        self.target_value_scale = target_value_scale
        self.match_target_to_condition = match_target_to_condition

        if not self.aligned_root.exists():
            raise FileNotFoundError(f"Aligned dataset root not found: {self.aligned_root}")
        if not self.run_names:
            raise ValueError("run_names must not be empty")

        self.samples = self._build_index()
        if not self.samples:
            raise RuntimeError(
                "No aligned radar/camera samples were found in the requested runs."
            )

    @staticmethod
    def _normalize_spatial_size(
        spatial_size: int | tuple[int, int] | None,
    ) -> tuple[int, int] | None:
        if spatial_size is None:
            return None
        if isinstance(spatial_size, int):
            return (spatial_size, spatial_size)
        return tuple(int(value) for value in spatial_size)

    @staticmethod
    def _collect_numbered_files(directory: Path) -> dict[int, Path]:
        files: dict[int, Path] = {}
        supported_suffixes = {".png", ".npy"}
        for path in sorted(directory.iterdir(), key=lambda item: (item.suffix, item.stem)):
            if path.suffix.lower() not in supported_suffixes:
                continue
            if path.stem.isdigit():
                files[int(path.stem)] = path
        return files

    def _build_index(self) -> list[RadarPromptLiDARSample]:
        samples: list[RadarPromptLiDARSample] = []
        for run_name in self.run_names:
            run_root = self.aligned_root / run_name
            cond_dir = run_root / self.cond_dir_name
            target_dir = run_root / self.target_dir_name

            if not cond_dir.exists():
                raise FileNotFoundError(f"Condition directory not found: {cond_dir}")
            if not target_dir.exists():
                raise FileNotFoundError(f"Target directory not found: {target_dir}")

            cond_map = self._collect_numbered_files(cond_dir)
            target_map = self._collect_numbered_files(target_dir)
            common_indices = sorted(set(cond_map) & set(target_map))

            for pair_index in common_indices:
                samples.append(
                    RadarPromptLiDARSample(
                        run_name=run_name,
                        pair_index=pair_index,
                        cond_path=cond_map[pair_index],
                        target_path=target_map[pair_index],
                    )
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(
        self,
        path: Path,
        spatial_size: tuple[int, int] | None,
        normalize_to_minus1_1: bool,
        value_scale: float | None,
    ) -> torch.Tensor:
        if path.suffix.lower() == ".npy":
            array = np.load(path)
        else:
            array = np.array(Image.open(path))

        if array.ndim == 2:
            tensor = torch.from_numpy(array.astype(np.float32)).unsqueeze(0)
        elif array.ndim == 3:
            if path.suffix.lower() == ".npy":
                tensor = torch.from_numpy(array.astype(np.float32))
            else:
                tensor = torch.from_numpy(array.astype(np.float32)).permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported image shape: {array.shape}")

        if value_scale is not None:
            is_unit_float_npy = (
                path.suffix.lower() == ".npy"
                and np.issubdtype(array.dtype, np.floating)
                and tensor.numel() > 0
                and float(tensor.min()) >= -1e-6
                and float(tensor.max()) <= 1.0 + 1e-6
            )
            if not is_unit_float_npy:
                tensor = tensor / max(float(value_scale), 1.0)
        elif np.issubdtype(array.dtype, np.integer):
            max_value = float(np.iinfo(array.dtype).max)
            tensor = tensor / max(max_value, 1.0)
        else:
            max_value = float(tensor.max()) if tensor.numel() > 0 else 1.0
            if max_value > 1.0:
                tensor = tensor / max_value

        tensor = tensor.clamp(0.0, 1.0)

        if normalize_to_minus1_1:
            tensor = tensor.mul(2.0).sub(1.0)

        if spatial_size is not None:
            tensor = self._resize_tensor(tensor, spatial_size)

        return tensor.contiguous()

    @staticmethod
    def _resize_tensor(tensor: torch.Tensor, spatial_size: tuple[int, int]) -> torch.Tensor:
        resized = F.interpolate(
            tensor.unsqueeze(0),
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0)

    def __getitem__(self, index: int) -> Any:
        sample = self.samples[index]
        cond = self._load_image(
            sample.cond_path,
            spatial_size=self.spatial_size,
            normalize_to_minus1_1=self.cond_normalize_to_minus1_1,
            value_scale=self.cond_value_scale,
        )
        target_spatial_size = self.spatial_size
        if target_spatial_size is None and self.match_target_to_condition:
            target_spatial_size = tuple(cond.shape[-2:])
        target = self._load_image(
            sample.target_path,
            spatial_size=target_spatial_size,
            normalize_to_minus1_1=self.target_normalize_to_minus1_1,
            value_scale=self.target_value_scale,
        )

        return target, {
            "cond": cond,
            "meta": {
                "run_name": sample.run_name,
                "pair_index": sample.pair_index,
                "cond_path": str(sample.cond_path),
                "target_path": str(sample.target_path),
            },
        }
