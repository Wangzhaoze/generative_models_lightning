#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-07
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : data_module/ColoRadar/dataset.py
# @IDE     : vscode

"""
Dataset placeholder for the ColoRadar dataset.
"""
import os
from torch.utils.data import Dataset
from glob import glob
import numpy as np
from PIL import Image
import torch
from typing import Any

class ColoRadarDataset(Dataset):
    """ColoRadar dataset: radar range-azimuth heatmap → LiDAR BEV image.

    Returns (lidar_bev [1,H,W], {"cond": radar_ra [1,H,W]}), both in [-1, 1].
    """
    def __init__(self, image_size: int = 64, sequence: list[str] = []):
        self.image_size = image_size
        self.radar_files: list[str] = []
        self.lidar_files: list[str] = []
        for seq in sequence:
            rah   = sorted(glob(f"{seq}/range_azimuth_heatmap/*.png"))
            bev  = sorted(glob(f"{seq}/lidar_pcl_bev_img/*.png"))
            rah_ids  = {int(os.path.splitext(os.path.basename(f))[0]): f for f in rah}
            bev_ids = {int(os.path.splitext(os.path.basename(f))[0]): f for f in bev}
            common  = sorted(set(rah_ids) & set(bev_ids))
            assert len(common) > 0, f"{seq}: no matched radar/lidar pairs"
            self.radar_files += [rah_ids[i]  for i in common]
            self.lidar_files += [bev_ids[i] for i in common]

    def __len__(self) -> int:
        return len(self.radar_files)

    def __getitem__(self, index) -> Any:
        rah  = Image.open(self.radar_files[index]).convert("L").resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR)
        bev = Image.open(self.lidar_files[index]).convert("L").resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR)
        # [H, W] → [1, H, W]，归一化到 [-1, 1]
        ra_t  = torch.from_numpy(np.array(rah).astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
        bev_t = torch.from_numpy(np.array(bev).astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
        return bev_t, {"cond": ra_t}
        # return {
        #     "condition": torch.tensor(ra),
        #     "target":    torch.tensor(pcl),
        # }



# class ColorRadarDataset(Dataset):
#     """Paired RGB image + radar Cartesian BEV — returns (bev, {"cond": camera_rgb}).

#     The cascade heatmap (polar: range × azimuth) is converted to a Cartesian
#     bird's-eye-view image so the spatial layout matches LiDAR BEV conventions.
#     """

#     def __init__(self, data_dir: str, image_size: int = 64, random_flip: bool = True,
#                  max_range: float = 16.0, fov_deg: float = 120.0):
#         self.image_size = image_size
#         self.random_flip = random_flip
#         self.max_range = max_range
#         self.fov_deg = fov_deg

#         self.samples = []
#         for run in sorted(os.listdir(data_dir)):
#             run_dir = os.path.join(data_dir, run)
#             rgb_dir = os.path.join(run_dir, "camera", "images", "rgb")
#             radar_dir = os.path.join(run_dir, "cascade", "heatmaps", "data")
#             if not os.path.isdir(rgb_dir) or not os.path.isdir(radar_dir):
#                 continue

#             rgb_files = {
#                 int(f.split("_")[-1].split(".")[0]): os.path.join(rgb_dir, f)
#                 for f in os.listdir(rgb_dir) if f.endswith(".png")
#             }
#             radar_files = {
#                 int(f.split("_")[-1].split(".")[0]): os.path.join(radar_dir, f)
#                 for f in os.listdir(radar_dir) if f.endswith(".bin")
#             }

#             for i in sorted(set(rgb_files) & set(radar_files)):
#                 self.samples.append((rgb_files[i], radar_files[i]))

#         print(f"[ColorRadarDataset] size={len(self.samples)}")

#     def __len__(self):
#         return len(self.samples)

#     # Cascade heatmap: complex64, stored as (elevation=32, range=64, azimuth=256)
#     _RADAR_SHAPE = (32, 64, 256)
#     # 99th-percentile dB ceiling measured from dataset
#     _DB_MAX = 20.0

#     @staticmethod
#     def _polar_to_cartesian(heatmap_polar: np.ndarray,
#                             image_size: int,
#                             max_range: float,
#                             fov_deg: float) -> np.ndarray:
#         """Convert polar (range × azimuth) heatmap to square Cartesian BEV.

#         Output layout (matches LiDAR BEV convention):
#           - top    = far  (y = max_range)
#           - bottom = near (y = 0)
#           - left   = left, right = right
#         """
#         from scipy.ndimage import map_coordinates

#         R, A = heatmap_polar.shape
#         fov_rad = np.deg2rad(fov_deg / 2)
#         half = max_range * np.sin(fov_rad)          # lateral extent (m)

#         # Cartesian grid in metres
#         xs = np.linspace(-half, half, image_size)   # left → right
#         ys = np.linspace(max_range, 0.0, image_size) # far  → near (top→bottom)
#         xg, yg = np.meshgrid(xs, ys)

#         # Cartesian (x, y) → polar (range_idx, azimuth_idx)
#         r = np.sqrt(xg ** 2 + yg ** 2)
#         az = np.arctan2(xg, yg)                     # [-fov_rad, fov_rad]

#         r_idx  = r / max_range * (R - 1)
#         az_idx = (az / fov_rad + 1.0) / 2.0 * (A - 1)

#         coords = np.array([r_idx.ravel(), az_idx.ravel()])
#         bev = map_coordinates(heatmap_polar, coords, order=1,
#                               mode="constant", cval=0.0)
#         bev = bev.reshape(image_size, image_size).astype(np.float32)

#         # Zero out pixels outside radar FOV / range
#         bev[(r > max_range) | (np.abs(az) > fov_rad)] = 0.0
#         return bev

#     def __getitem__(self, idx):
#         rgb_path, radar_path = self.samples[idx]

#         image = Image.open(rgb_path).convert("RGB").resize(
#             (self.image_size, self.image_size), Image.Resampling.BICUBIC
#         )

#         # ── Load cascade heatmap ─────────────────────────────────────────────
#         # Stored as complex64 (elevation=32, range=64, azimuth=256)
#         raw = np.fromfile(radar_path, dtype=np.complex64).reshape(self._RADAR_SHAPE)
#         mag = np.abs(raw)

#         # Sum over elevation → polar range-azimuth map (64 × 256)
#         heatmap_polar = mag.sum(axis=0).astype(np.float32)

#         # Noise normalisation: divide by 30th-percentile noise floor
#         noise = np.percentile(heatmap_polar, 30)
#         heatmap_polar = heatmap_polar / (noise + 1e-8)

#         # dB conversion + clip
#         heatmap_db = 10.0 * np.log10(heatmap_polar + 1.0)
#         heatmap_db = np.clip(heatmap_db, 0.0, self._DB_MAX)

#         # Polar → Cartesian BEV (LiDAR-style layout)
#         bev = self._polar_to_cartesian(
#             heatmap_db, self.image_size, self.max_range, self.fov_deg
#         )

#         # Normalise to [-1, 1]
#         radar = torch.from_numpy(bev / self._DB_MAX * 2.0 - 1.0).unsqueeze(0)

#         if self.random_flip and random.random() < 0.5:
#             image = image.transpose(Image.FLIP_LEFT_RIGHT)
#             radar = radar.flip(-1)

#         # Camera RGB as SPADE condition [3, H, W] in [-1, 1]
#         cond = torch.from_numpy(np.array(image).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)

#         return radar, {"cond": cond}