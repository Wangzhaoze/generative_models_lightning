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
from typing import Any, Optional
from .range_image_process.radgs_utils import (
    project_scene_to_range_image,
    quat_to_matrix,
    radar_adc_to_pointcloud,
    range_image_to_points,
    transform_pcd,
    xyz2aer,
)

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



class ColoRadarDataset3D(Dataset):
    """
    ColoRadar dataset: 3D point cloud → 2D range image.

    Two modes:
    - Legacy mode  (sequence param):  load pre-processed .npy point clouds
    - New mode     (rec_ids param):   use ColoRadarPlus raw data; accumulate all
                                       frames into a global point cloud at init,
                                       then generate range images per-frame via
                                       compute_range_image_with_visibility().

    Projection modes:
    - accumulated_scene: build/use a merged scene point cloud, then project it
      from each frame pose to obtain dense range images.
    - direct_frame: project each frame's own point cloud directly to a range
      image without accumulating points across time.
    """
    def __init__(self,
                 # ── Legacy params (unchanged) ───────────────────────────────
                 image_size: int = 64,
                 sequence: list[str] = [],
                 dataset_root: str = "/home/local/Desktop/code/Datasets/ColoRadar",
                 # ── New ColoRadar+ params ────────────────────────────────────
                 data_dir: str = "/home/local/Desktop/code/Datasets/ColoRadar+",
                 rec_ids: list[str] = [],
                 # Range image FOV / resolution
                 lidar_az_fov: float = 64.0,
                 lidar_el_fov: float = 30.0,
                 lidar_max_range: float = 50.0,
                 lidar_az_res: float = 1.0,
                 lidar_el_res: float = 1.0,
                 radar_az_fov: float = 64.0,
                 radar_el_fov: float = 30.0,
                 radar_max_range: float = 14.0,
                 radar_az_res: float = 1.0,
                 radar_el_res: float = 1.0,
                 radar_channels: int = 16,
                 radar_power_threshold_db: float = 20.0,
                 projection_mode: str = "accumulated_scene",
                 range_image_cache_name: Optional[str] = None,
                 # Cache dir to avoid re-computing global point clouds
                 cache_dir: Optional[str] = None):

        assert not (sequence and rec_ids), \
            "Provide either 'sequence' (legacy) or 'rec_ids' (new mode), not both."
        if projection_mode not in {"accumulated_scene", "direct_frame"}:
            raise ValueError(
                f"projection_mode must be 'accumulated_scene' or 'direct_frame', got {projection_mode!r}"
            )

        self.projection_mode = projection_mode
        self.range_image_cache_name = (
            range_image_cache_name
            if range_image_cache_name is not None
            else (
                "range_images.pt"
                if projection_mode == "accumulated_scene"
                else "direct_frame_range_images.pt"
            )
        )

        if rec_ids:
            # ── NEW MODE ────────────────────────────────────────────────────
            self._mode = "new"
            self.lidar_az_fov   = lidar_az_fov
            self.lidar_el_fov   = lidar_el_fov
            self.lidar_max_range = lidar_max_range
            self.lidar_az_res   = lidar_az_res
            self.lidar_el_res   = lidar_el_res
            self.radar_az_fov   = radar_az_fov
            self.radar_el_fov   = radar_el_fov
            self.radar_max_range = radar_max_range
            self.radar_az_res   = radar_az_res
            self.radar_el_res   = radar_el_res
            self.radar_channels = radar_channels
            self.radar_power_threshold_db = radar_power_threshold_db

            if self.projection_mode != "accumulated_scene":
                raise NotImplementedError(
                    "projection_mode='direct_frame' is currently supported in legacy sequence mode only"
                )

            from .range_image_process.radgs_dataloader import ColoRadarPlusDataset
            from tqdm import trange

            self._lidar_poses: list[np.ndarray] = []
            self._radar_poses: list[np.ndarray] = []
            lidar_world_parts: list[np.ndarray] = []
            radar_world_parts: list[np.ndarray] = []

            for rec_id in rec_ids:
                ds = ColoRadarPlusDataset(data_dir=data_dir, rec_id=rec_id)

                # ── Try loading from cache ───────────────────────────────────
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                    lidar_cache = os.path.join(cache_dir, f"{rec_id}_lidar_scene.npy")
                    radar_cache = os.path.join(cache_dir, f"{rec_id}_radar_scene.npy")
                    poses_cache = os.path.join(cache_dir, f"{rec_id}_poses.npz")
                    if (os.path.exists(lidar_cache) and
                            os.path.exists(radar_cache) and
                            os.path.exists(poses_cache)):
                        print(f"[ColoRadarDataset3D] Loading cached pcd for {rec_id}")
                        lidar_world_parts.append(np.load(lidar_cache))
                        radar_world_parts.append(np.load(radar_cache))
                        cached = np.load(poses_cache, allow_pickle=False)
                        self._lidar_poses.extend(list(cached['lidar_poses']))
                        self._radar_poses.extend(list(cached['radar_poses']))
                        continue

                # ── Interpolate poses ────────────────────────────────────────
                ds.lidar.interpolate_poses(ds.ego)
                ds.radar.interpolate_poses(ds.ego)
                n_lidar = min(len(ds.lidar), len(ds.lidar.lidar_calib_poses))
                n_radar = min(len(ds.radar), len(ds.radar.frame_calib_poses))

                # ── Build global LiDAR point cloud ───────────────────────────
                lidar_parts: list[np.ndarray] = []
                for i in trange(n_lidar, desc=f"LiDAR global pcd [{rec_id}]"):
                    pcd_local = ds.lidar[i][:, :3].astype(np.float32)
                    dist = np.linalg.norm(pcd_local, axis=1)
                    pcd_local = pcd_local[(dist > 0.05) & (dist < 100.0)]
                    if len(pcd_local) == 0:
                        continue
                    pose = np.array(ds.lidar.lidar_calib_poses[i], dtype=np.float32)
                    lidar_parts.append(transform_pcd(pcd_local, pose))
                lidar_pcd_rec = (np.concatenate(lidar_parts)
                                 if lidar_parts else np.zeros((0, 3), dtype=np.float32))
                lidar_world_parts.append(lidar_pcd_rec)

                # ── Build global radar point cloud ───────────────────────────
                radar_parts: list[np.ndarray] = []
                for i in trange(n_radar, desc=f"Radar global pcd  [{rec_id}]"):
                    adc = ds.radar[i]
                    pcd_local = radar_adc_to_pointcloud(
                        adc, ds.radar, radar_power_threshold_db)
                    if pcd_local.shape[0] == 0:
                        continue
                    pose = np.array(ds.radar.frame_calib_poses[i], dtype=np.float32)
                    radar_parts.append(transform_pcd(pcd_local, pose))
                radar_pcd_rec = (np.concatenate(radar_parts)
                                 if radar_parts else np.zeros((0, 3), dtype=np.float32))
                radar_world_parts.append(radar_pcd_rec)

                # ── Build frame index: sync radar frames to lidar poses ──────
                rec_lidar_poses: list[np.ndarray] = []
                rec_radar_poses: list[np.ndarray] = []
                lidar_ts = np.array(ds.lidar.time_stamps)
                radar_ts = np.array(ds.radar.time_stamps)
                for r_idx in range(n_radar):
                    l_idx = int(np.argmin(np.abs(lidar_ts - radar_ts[r_idx])))
                    if abs(float(lidar_ts[l_idx]) - float(radar_ts[r_idx])) < 0.2:
                        rec_lidar_poses.append(
                            np.array(ds.lidar.lidar_calib_poses[l_idx], dtype=np.float32))
                        rec_radar_poses.append(
                            np.array(ds.radar.frame_calib_poses[r_idx], dtype=np.float32))
                self._lidar_poses.extend(rec_lidar_poses)
                self._radar_poses.extend(rec_radar_poses)

                # ── Save cache ───────────────────────────────────────────────
                if cache_dir:
                    np.save(lidar_cache, lidar_pcd_rec)
                    np.save(radar_cache, radar_pcd_rec)
                    if rec_lidar_poses:
                        np.savez(poses_cache,
                                 lidar_poses=np.array(rec_lidar_poses, dtype=np.float32),
                                 radar_poses=np.array(rec_radar_poses, dtype=np.float32))

            self.lidar_scene_pcd = (np.concatenate(lidar_world_parts)
                                    if lidar_world_parts else np.zeros((0, 3), dtype=np.float32))
            self.radar_scene_pcd = (np.concatenate(radar_world_parts)
                                    if radar_world_parts else np.zeros((0, 3), dtype=np.float32))
            print(f"[ColoRadarDataset3D] New mode ready: {len(self._lidar_poses)} samples | "
                  f"LiDAR pts: {len(self.lidar_scene_pcd):,} | "
                  f"Radar pts: {len(self.radar_scene_pcd):,}")
            self._cache_dir = cache_dir
            self._precomputed = self._precompute_range_images()

        else:
            # ── LEGACY MODE ─────────────────────────────────────────────────
            self._mode = "legacy"
            import json
            self.image_size = image_size
            self.radar_files: list[str] = []
            self.lidar_files: list[str] = []
            self.lidar_poses: list[np.ndarray] = []

            # Store FOV/resolution params (same interface as new mode)
            self.lidar_az_fov    = lidar_az_fov
            self.lidar_el_fov    = lidar_el_fov
            self.lidar_max_range = lidar_max_range
            self.lidar_az_res    = lidar_az_res
            self.lidar_el_res    = lidar_el_res
            self.radar_az_fov    = radar_az_fov
            self.radar_el_fov    = radar_el_fov
            self.radar_max_range = radar_max_range
            self.radar_az_res    = radar_az_res
            self.radar_el_res    = radar_el_res
            self.radar_channels  = radar_channels

            self._precomputed = []
            with open(os.path.join(dataset_root, "dataset.json")) as f:
                meta = json.load(f)
            codename_to_run = {item["codename"]: item["path"]
                               for item in meta["datastore"]["folders"]}

            for seq in sequence:
                seq_path = seq.rstrip(os.sep)
                run_root = os.path.dirname(seq_path) if os.path.basename(seq_path) == "seq" else seq_path
                run_name = os.path.basename(run_root)
                run_cache_dir = (
                    os.path.join(run_root, "cache")
                    if cache_dir is not None
                    else None
                )

                lidar_3d = sorted(glob(f"{seq}/lidar_pcl/*.npy"))
                radar_3d = sorted(glob(f"{seq}/pcl_npy/*.npy"))
                lidar_ids = {int(os.path.splitext(os.path.basename(f))[0]): f for f in lidar_3d}
                radar_ids = {int(os.path.splitext(os.path.basename(f))[0]): f for f in radar_3d}
                common = sorted(set(lidar_ids) & set(radar_ids))
                assert len(common) > 0, f"{seq}: no matched radar/lidar pairs"

                codename = next((p for p in seq.split(os.sep) if p in codename_to_run), None)
                if codename is not None:
                    run_dir = os.path.join(dataset_root, codename_to_run[codename])
                else:
                    run_candidates = [run_root]
                    if "_run" in run_name:
                        site_name = run_name.rsplit("_run", 1)[0]
                        run_candidates.append(
                            os.path.join(os.path.dirname(run_root), site_name, run_name)
                        )
                    run_dir = next(
                        (
                            path for path in run_candidates
                            if os.path.exists(os.path.join(path, "lidar/timestamps.txt"))
                            and os.path.exists(os.path.join(path, "groundtruth/timestamps.txt"))
                            and os.path.exists(os.path.join(path, "groundtruth/groundtruth_poses.txt"))
                        ),
                        None,
                    )
                    assert run_dir is not None, (
                        f"Could not find pose/timestamp files for sequence path: {seq}"
                    )
                lidar_ts = np.loadtxt(os.path.join(run_dir, "lidar/timestamps.txt"))
                gt_ts    = np.loadtxt(os.path.join(run_dir, "groundtruth/timestamps.txt"))
                gt_raw   = np.loadtxt(os.path.join(run_dir, "groundtruth/groundtruth_poses.txt"))

                run_lidar_files = [lidar_ids[i] for i in common]
                run_radar_files = [radar_ids[i] for i in common]
                run_lidar_poses: list[np.ndarray] = []
                for i in common:
                    nearest = int(np.argmin(np.abs(gt_ts - lidar_ts[i])))
                    p = gt_raw[nearest]
                    T = np.eye(4, dtype=np.float32)
                    T[:3, :3] = quat_to_matrix(*p[3:])
                    T[:3, 3]  = p[:3]
                    run_lidar_poses.append(T)

                self.lidar_files.extend(run_lidar_files)
                self.radar_files.extend(run_radar_files)
                self.lidar_poses.extend(run_lidar_poses)
                if self.projection_mode == "direct_frame":
                    self._precomputed.extend(
                        self._build_legacy_run_direct_precomputed(
                            run_name=run_name,
                            lidar_files=run_lidar_files,
                            radar_files=run_radar_files,
                            poses=run_lidar_poses,
                            cache_dir=run_cache_dir,
                        )
                    )
                else:
                    self._precomputed.extend(
                        self._build_legacy_run_precomputed(
                            run_name=run_name,
                            lidar_files=run_lidar_files,
                            radar_files=run_radar_files,
                            poses=run_lidar_poses,
                            cache_dir=run_cache_dir,
                        )
                    )

            self._lidar_poses = self.lidar_poses
            self._radar_poses = self.lidar_poses
            print(
                f"[ColoRadarDataset3D] Legacy mode ready: {len(self._precomputed)} samples | "
                f"projection_mode={self.projection_mode}"
            )

    def _precompute_range_images(self) -> list:
        """Precompute all (lidar_rm, radar_rm) pairs at init time.

        Avoids O(N_scene) numpy work inside __getitem__, which blocks DataLoader
        workers and stalls training steps.
        """
        from tqdm import trange
        if self._cache_dir:
            os.makedirs(self._cache_dir, exist_ok=True)
        ri_cache = (os.path.join(self._cache_dir, self.range_image_cache_name)
                    if self._cache_dir else None)
        if ri_cache and os.path.exists(ri_cache):
            print("[ColoRadarDataset3D] Loading cached range images")
            return torch.load(ri_cache, weights_only=False)

        n = len(self._lidar_poses)
        items: list = []
        for i in trange(n, desc="Precomputing range images"):
            lidar_pose = self._lidar_poses[i]
            radar_pose = self._radar_poses[i]
            li_rm = project_scene_to_range_image(
                self.lidar_scene_pcd, lidar_pose,
                self.lidar_az_fov, self.lidar_el_fov,
                self.lidar_max_range, self.lidar_az_res, self.lidar_el_res,
                coord_order=[1, 2, 0],
            )
            ra_rm = self._radar_to_range(
                self.radar_scene_pcd,
                antenna_pose=radar_pose,
            )
            items.append((li_rm, ra_rm))

        if ri_cache:
            torch.save(items, ri_cache)
            print(f"[ColoRadarDataset3D] Cached range images → {ri_cache}")
        return items

    def _build_legacy_run_precomputed(
        self,
        run_name: str,
        lidar_files: list[str],
        radar_files: list[str],
        poses: list[np.ndarray],
        cache_dir: Optional[str],
    ) -> list:
        """Build/cache range images using only one run's merged point cloud."""
        from tqdm import trange
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            ri_cache = os.path.join(cache_dir, self.range_image_cache_name)
            if os.path.exists(ri_cache):
                print(f"[ColoRadarDataset3D] Loading cached range images for {run_name}")
                return torch.load(ri_cache, weights_only=False)
        else:
            ri_cache = None

        lidar_parts: list[np.ndarray] = []
        radar_parts: list[np.ndarray] = []
        for i in trange(len(lidar_files), desc=f"Building global pcd [{run_name}]"):
            pose = poses[i]
            lpts = np.load(lidar_files[i])[:, :3].astype(np.float32)
            dist = np.linalg.norm(lpts, axis=1)
            lpts = lpts[(dist > 0.05) & (dist < 100.0)]
            if len(lpts):
                lidar_parts.append(transform_pcd(lpts, pose))

            rpts = np.load(radar_files[i])[:, :3].astype(np.float32)
            rdist = np.linalg.norm(rpts, axis=1)
            rpts = rpts[(rdist > 0.05) & (rdist < self.radar_max_range * 1.2)]
            if len(rpts):
                radar_parts.append(transform_pcd(rpts, pose))

        lidar_scene_pcd = (np.concatenate(lidar_parts)
                           if lidar_parts else np.zeros((0, 3), dtype=np.float32))
        radar_scene_pcd = (np.concatenate(radar_parts)
                           if radar_parts else np.zeros((0, 3), dtype=np.float32))
        print(f"[ColoRadarDataset3D] {run_name}: {len(poses)} samples | "
              f"LiDAR pts: {len(lidar_scene_pcd):,} | "
              f"Radar pts: {len(radar_scene_pcd):,}")

        items: list = []
        for i in trange(len(poses), desc=f"Precomputing range images [{run_name}]"):
            pose = poses[i]
            li_rm = project_scene_to_range_image(
                lidar_scene_pcd, pose,
                self.lidar_az_fov, self.lidar_el_fov,
                self.lidar_max_range, self.lidar_az_res, self.lidar_el_res,
                coord_order=[1, 2, 0],
            )
            ra_rm = self._radar_to_range(
                radar_scene_pcd,
                antenna_pose=pose,
            )
            items.append((li_rm, ra_rm))

        if ri_cache:
            torch.save(items, ri_cache)
            print(f"[ColoRadarDataset3D] Cached {run_name} range images → {ri_cache}")
        return items

    def _build_legacy_run_direct_precomputed(
        self,
        run_name: str,
        lidar_files: list[str],
        radar_files: list[str],
        poses: list[np.ndarray],
        cache_dir: Optional[str],
    ) -> list:
        """Build/cache per-frame range images without scene accumulation."""
        from tqdm import trange
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            ri_cache = os.path.join(cache_dir, self.range_image_cache_name)
            if os.path.exists(ri_cache):
                print(f"[ColoRadarDataset3D] Loading cached direct-frame range images for {run_name}")
                return torch.load(ri_cache, weights_only=False)
        else:
            ri_cache = None

        items: list = []
        for i in trange(len(poses), desc=f"Precomputing direct-frame range images [{run_name}]"):
            pose = poses[i]

            lidar_local = np.load(lidar_files[i])[:, :3].astype(np.float32)
            lidar_dist = np.linalg.norm(lidar_local, axis=1)
            lidar_local = lidar_local[(lidar_dist > 0.05) & (lidar_dist < max(100.0, self.lidar_max_range * 1.5))]
            lidar_world = (
                transform_pcd(lidar_local, pose)
                if len(lidar_local)
                else np.zeros((0, 3), dtype=np.float32)
            )
            li_rm = project_scene_to_range_image(
                lidar_world,
                pose,
                self.lidar_az_fov,
                self.lidar_el_fov,
                self.lidar_max_range,
                self.lidar_az_res,
                self.lidar_el_res,
                coord_order=[1, 2, 0],
            )

            radar_local = np.load(radar_files[i])[:, :3].astype(np.float32)
            radar_dist = np.linalg.norm(radar_local, axis=1)
            radar_local = radar_local[
                (radar_dist > 0.05) & (radar_dist < max(self.radar_max_range * 1.2, self.radar_max_range + 1e-3))
            ]
            radar_world = (
                transform_pcd(radar_local, pose)
                if len(radar_local)
                else np.zeros((0, 3), dtype=np.float32)
            )
            ra_rm = self._radar_to_range(
                radar_world,
                antenna_pose=pose,
            )
            items.append((li_rm, ra_rm))

        if ri_cache:
            torch.save(items, ri_cache)
            print(f"[ColoRadarDataset3D] Cached direct-frame {run_name} range images → {ri_cache}")
        return items

    def __len__(self) -> int:
        return len(self._precomputed)

    # def __len__(self) -> int:
    #     if self._mode == "new":
    #         return len(self._lidar_poses)
    #     return len(self.lidar_files)
    
    def xyz2aer(self, points: np.ndarray, as_degrees: bool = True) -> np.ndarray:
        """
        Convert 3D points from Cartesian coordinates (x, y, z) to spherical coordinates (azimuth, elevation, range).

        Args:
            points (np.ndarray): Nx3 array of points in Cartesian coordinates.

        Returns:
            np.ndarray: Nx3 array of points in spherical coordinates (azimuth, elevation, range).
        """
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        range = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.arctan2(-y, z)
        elevation = np.arcsin(x / range)

        if as_degrees:
            azimuth = np.degrees(azimuth)
            elevation = np.degrees(elevation)
        return np.column_stack((azimuth, elevation, range))

    def _lidar_to_range(self,
                        path,
                        antenna_pose: np.ndarray,
                        az_fov=180, 
                        el_fov=22, 
                        max_range=13,
                        az_res: float = 1, 
                        el_res: float = 1, # Resolution in degrees
                        point_radius: float = 0): # Optional radius in degrees 
        """
        convert LiDAR point cloud to range image (H, W)
        input: points (N, 3) in lidar coordinates
        output: range image (n_channels, H, W) 
        """
        features = None
        points = np.load(path).astype(np.float32)  # (N, 3)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        points_h = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1), np.ones((points.shape[0], 1))))
        # coloradar+的时候变回去
        
        points_local = (np.linalg.inv(antenna_pose) @ points_h.T).T[:, :3]

        visibility_mask = np.ones(points_local.shape[0], dtype=bool)
        azimuth, elevation, range = xyz2aer(points_local, as_degrees=True).T

        # filter points based on azimuth and elevation FOV
        azimuth_mask = (azimuth >= -az_fov) & (azimuth <= az_fov)
        elevation_mask = (elevation >= -el_fov) & (elevation <= el_fov)
        range_mask = (range > 0) & (range <= max_range)
        visibility_mask &= azimuth_mask & elevation_mask & range_mask 

        azimuth = azimuth[visibility_mask]
        elevation = elevation[visibility_mask]
        range = range[visibility_mask]
        if features is not None:
            features = features[visibility_mask]

        # sorted azimuth, elevation, and range from range max to min
        sorted_indices = np.argsort(range)[::-1]
        azimuth = azimuth[sorted_indices]
        elevation = elevation[sorted_indices]
        range = range[sorted_indices]

        # Convert azimuth and elevation to pixel coordinates.
        az_bins = int(az_fov * 2 / az_res)
        el_bins = int(el_fov * 2 / el_res)
        azimuth_pixel = np.floor((azimuth + az_fov) / az_res).astype(int)
        elevation_pixel = np.floor((elevation + el_fov) / el_res).astype(int)
        azimuth_pixel = np.clip(azimuth_pixel, 0, az_bins - 1)
        elevation_pixel = np.clip(elevation_pixel, 0, el_bins - 1)

        # Image shape is [elevation, azimuth].
        range_image = np.full((el_bins, az_bins), 0, dtype=np.float32)

        if features is not None:
            features = features[sorted_indices]
            feature_image = np.full(
                (el_bins, az_bins, features.shape[1]),
                0.0,
                dtype=features.dtype
            )
            feature_image[elevation_pixel, azimuth_pixel] = features
        else:
            feature_image = None

        range_image[elevation_pixel, azimuth_pixel] = range

        # 归一化到 [-1, 1]：0（无返回）→ -1，max_range → 1
        range_image = range_image / max_range * 2.0 - 1.0
        return torch.from_numpy(range_image).unsqueeze(0)  # (1, H, W)

        # points = np.load(path).astype(np.float32)  # (N, 3)
        # x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # r = np.sqrt(x**2 + y**2 + z**2)
        # theta = np.arctan2(y, x)
        # alpha = np.arcsin(z / (r + 1e-8))

        # """
        # new way of map spherical coordinate
        # """
       
        # range_image = np.zeros((H, W), dtype=np.float32)

        # # map spherical coords to image pixels
        # fov_up_rad = np.deg2rad(fov_up)
        # fov_down_rad = np.deg2rad(fov_down)
        # fov_range = fov_up_rad - fov_down_rad

        # # u: [0,1] from theta in [-pi, pi]
        # u = (theta + np.pi) / (2 * np.pi)
        # # v: [0,1] from alpha between fov_down_rad..fov_up_rad
        # v = 1.0 - (alpha - fov_down_rad) / (fov_range + 1e-8)

        # col = np.floor(u * W).astype(np.int32).clip(0, W - 1)
        # row = np.floor(v * H).astype(np.int32).clip(0, H - 1)

        # # fill pixels: keep farthest point when collisions occur
        # order = np.argsort(r)[::-1]
        # range_image[row[order], col[order]] = r[order]

        # # normalize to [-1, 1]
        # r_max = range_image.max()
        # if r_max > 0:
        #     range_image = range_image / r_max * 2.0 - 1.0

        # """
        # end new
        # """
        # # col = np.floor(u * W).astype(np.int32).clip(0, W - 1)
        # # row = np.floor(v * H).astype(np.int32).clip(0, H - 1)

        # # range_image = np.zeros((H, W), dtype=np.float32)
        # # order = np.argsort(r)[::-1]
        # # range_image[row[order], col[order]] = r[order]
        # # r_max = range_image.max()
        # # if r_max > 0:
        # #     range_image = range_image / r_max * 2.0 - 1.0

        # return torch.from_numpy(range_image).unsqueeze(0)  # (1, H, W)
    
    def _radar_to_range(
        self,
        points_or_path,
        antenna_pose: Optional[np.ndarray] = None,
        r_max: Optional[float] = None,
        n_channels: Optional[int] = None,
        az_fov: Optional[float] = None,
        el_fov: Optional[float] = None,
        az_res: Optional[float] = None,
        el_res: Optional[float] = None,
        coord_order: Optional[list[int]] = None,
        min_range: float = 1e-3,
    ) -> torch.Tensor:
        """Convert radar points to a 16-channel normalized range image.

        Expected local point convention after any `coord_order` remap is
        x-forward, y-left/right, z-up. Empty pixels stay at -1.
        """
        r_max = self.radar_max_range if r_max is None else r_max
        n_channels = self.radar_channels if n_channels is None else n_channels
        az_fov = self.radar_az_fov if az_fov is None else az_fov
        el_fov = self.radar_el_fov if el_fov is None else el_fov
        az_res = self.radar_az_res if az_res is None else az_res
        el_res = self.radar_el_res if el_res is None else el_res

        if r_max <= 0 or n_channels <= 0 or az_res <= 0 or el_res <= 0:
            raise ValueError("Radar range/FOV parameters must be positive")

        az_bins = max(1, int(round(az_fov * 2.0 / az_res)))
        el_bins = max(1, int(round(el_fov * 2.0 / el_res)))
        empty = torch.full((n_channels, el_bins, az_bins), -1.0, dtype=torch.float32)

        if isinstance(points_or_path, (str, os.PathLike)):
            points = np.load(points_or_path).astype(np.float32)
        else:
            points = np.asarray(points_or_path, dtype=np.float32)

        if points.size == 0:
            return empty
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("Radar point cloud should have shape (N,3+) or be empty")

        points = points[:, :3]
        finite_points = np.isfinite(points).all(axis=1)
        points = points[finite_points]
        if len(points) == 0:
            return empty

        if antenna_pose is not None:
            sensor_pos = antenna_pose[:3, 3]
            nearby = np.linalg.norm(points - sensor_pos, axis=1) < r_max * 1.5
            points = points[nearby]
            if len(points) == 0:
                return empty

            points_h = np.hstack([points, np.ones((len(points), 1), dtype=np.float32)])
            points = (np.linalg.inv(antenna_pose) @ points_h.T).T[:, :3]

        if coord_order is not None:
            points = points[:, coord_order]

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        finite = np.isfinite(r) & (r > min_range) & (r <= r_max)
        if not finite.any():
            return empty

        x, y, z, r = x[finite], y[finite], z[finite], r[finite]
        azimuth = np.degrees(np.arctan2(y, x))
        elevation = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))

        visible = (
            (azimuth >= -az_fov) & (azimuth <= az_fov) &
            (elevation >= -el_fov) & (elevation <= el_fov)
        )
        if not visible.any():
            return empty

        azimuth = azimuth[visible]
        elevation = elevation[visible]
        r = r[visible]

        col = np.floor((azimuth + az_fov) / az_res).astype(np.int32)
        row = np.floor((el_fov - elevation) / el_res).astype(np.int32)
        col = np.clip(col, 0, az_bins - 1)
        row = np.clip(row, 0, el_bins - 1)

        dr = r_max / float(n_channels)
        channel = np.minimum((r / dr).astype(np.int32), n_channels - 1)

        raw = np.zeros((n_channels, el_bins, az_bins), dtype=np.float32)
        order = np.argsort(r)[::-1]
        raw[channel[order], row[order], col[order]] = r[order]

        normalized = raw / r_max * 2.0 - 1.0
        return torch.from_numpy(normalized.astype(np.float32))
    
    # def _radar_to_range(self,
    #                     path,
    #                     r_max=14.0,
    #                     n_channels=16,
    #                     H=32, W=32,
    #                     fov_up=10.0, fov_down=-35.0,
    #                     fov_left=172.0, fov_right=8.0):
    #     """
    #     convert radar point cloud to range image (n_channels, H, W)
    #     input: points (N, 3) in radar coordinates
    #     output: range image (n_channels, H, W) 
    #     """
    #     points = np.load(path).astype(np.float32)
    #     x, y, z = points[:, 0], points[:, 1], points[:, 2]
    #     r = np.sqrt(x**2 + y**2 + z**2)
    #     theta = np.arctan2(y, x)
    #     alpha = np.arcsin(z / (r + 1e-8))

    #     fov_up_rad    = np.deg2rad(fov_up)
    #     fov_down_rad  = np.deg2rad(fov_down)
    #     fov_left_rad  = np.deg2rad(fov_left)
    #     fov_right_rad = np.deg2rad(fov_right)
    #     fov_v = fov_up_rad - fov_down_rad
    #     fov_h = fov_left_rad - fov_right_rad

    #     u = 1.0 - (theta - fov_right_rad) / fov_h
    #     v = 1.0 - (alpha - fov_down_rad)  / fov_v
    #     col = np.floor(u * W).astype(np.int32).clip(0, W - 1)
    #     row = np.floor(v * H).astype(np.int32).clip(0, H - 1)

    #     dr = r_max / n_channels
    #     channels = []

    #     for i in range(n_channels):
    #         r_min_i = i * dr
    #         r_max_i = (i + 1) * dr
    #         mask = (r >= r_min_i) & (r < r_max_i)

    #         channel = np.zeros((H, W), dtype=np.float32)
    #         if mask.sum() > 0:
    #             r_i   = r[mask]
    #             col_i = col[mask]
    #             row_i = row[mask]
    #             order = np.argsort(r_i)[::-1]
    #             channel[row_i[order], col_i[order]] = r_i[order]

    #         channels.append(channel)

    #     stacked = np.stack(channels, axis=0)  # (n_channels, H, W)

    #     if stacked.max() > 0:
    #         stacked = stacked / stacked.max() * 2.0 - 1.0

    #     return torch.from_numpy(stacked)  # (n_channels, H, W)

    def __getitem__(self, index) -> Any:
        li_rm, ra_rm = self._precomputed[index]
        return li_rm, {"cond": ra_rm}

    # def __getitem__(self, index) -> Any:
    #     if self._mode == "new":
    #         lidar_pose = self._lidar_poses[index]
    #         radar_pose = self._radar_poses[index]
    #         li_rm = project_scene_to_range_image(
    #             self.lidar_scene_pcd, lidar_pose,
    #             self.lidar_az_fov, self.lidar_el_fov,
    #             self.lidar_max_range, self.lidar_az_res, self.lidar_el_res,
    #             coord_order=[1, 2, 0],
    #         )
    #         ra_rm = self._radar_to_range(
    #             self.radar_scene_pcd,
    #             antenna_pose=radar_pose,
    #         )
    #         return li_rm, {"cond": ra_rm}
    #     # Legacy mode: project accumulated global pcd via compute_range_image_with_visibility.
    #     # coord_order=[1,2,0] converts x-forward → compatible with the internal x/y-swap.
    #     pose = self._lidar_poses[index]
    #     li_rm = project_scene_to_range_image(
    #         self.lidar_scene_pcd, pose,
    #         self.lidar_az_fov, self.lidar_el_fov,
    #         self.lidar_max_range, self.lidar_az_res, self.lidar_el_res,
    #         coord_order=[1, 2, 0],
    #     )
    #     ra_rm = self._radar_to_range(
    #         self.radar_scene_pcd,
    #         antenna_pose=pose,
    #     )
    #     return li_rm, {"cond": ra_rm}

    def _lidar_frame_max_range(self, index: int) -> float:
        """Return the maximum radial range (metres) of the raw LiDAR frame.

        Used to invert the [-1, 1] normalization when reconstructing a point
        cloud from a range image via `lidar_range_to_points`.
        """
        points = np.load(self.lidar_files[index]).astype(np.float32)
        if points.size == 0:
            return 0.0
        r = np.sqrt(np.sum(points[:, :3] ** 2, axis=1))
        return float(r.max()) if r.size > 0 else 0.0

    def lidar_range_to_points(
        self,
        range_image,
        range_max: float,
        fov_up: float = 20.0,
        fov_down: float = -35.0,
        min_range: float = 1e-3,
        is_normalized: bool = True,
    ) -> np.ndarray:
        """Reconstruct a 3-D point cloud from a LiDAR range image.

        Inverts the spherical projection used in `_lidar_to_range`.

        Args:
            range_image: Tensor or ndarray, shape (1,H,W) or (H,W).
            range_max: maximum range used during normalization (metres).
            fov_up / fov_down: vertical FOV limits in degrees.
            min_range: discard pixels with range below this threshold.
            is_normalized: True if the image is in [-1, 1]; False for raw metres.

        Returns:
            points: ndarray of shape (N, 3), columns are x, y, z.
        """
        return range_image_to_points(
            range_image=range_image,
            range_max=range_max,
            fov_up=fov_up,
            fov_down=fov_down,
            min_range=min_range,
            is_normalized=is_normalized,
        )



# /home/local/Desktop/code/Radar-Diffusion/COLO_RPD_Dataset/2_24_2021_aspen_run7/lidar_pcl/0.npy


if __name__ == "__main__":
    pass
    # DATA_ROOT       = "/home/local/Desktop/code/Radar-Diffusion/COLO_RPD_Dataset"
    # SEQUENCES       = [f"{DATA_ROOT}/2_24_2021_aspen_run7",
    #                    f"{DATA_ROOT}/2_24_2021_aspen_run8"
    #                    ]
    # dataset = ColoRadarDataset3D(sequence=SEQUENCES, image_size=64)

    # print(f"radarfile: {dataset.radar_files[0]}")
    # print(f"lidarfile: {dataset.lidar_files[0]}")

    # sample = dataset[0]        
    # print(type(sample))
    # print(sample[0].shape)
    # print(sample[1]["cond"].shape)
    # import glob
    # files = glob.glob("/home/local/Desktop/code/Radar-Diffusion/COLO_RPD_Dataset/**/lidar_pcl/*.npy", recursive=True)

    # fov_up_global = -90
    # fov_down_global = 90
    # theta_min_global = 180
    # theta_max_global = -180
    # r_max_global = 0

    # for f in files:  # ??50?
    #     points = np.load(f).astype(np.float32)
    #     if points.ndim != 2 or points.shape[1] < 3:
    #         continue
    #     x, y, z = points[:,0], points[:,1], points[:,2]
    #     r = np.sqrt(x**2 + y**2 + z**2)
    #     alpha = np.rad2deg(np.arcsin(z / (r + 1e-8)))
    #     theta = np.rad2deg(np.arctan2(y, x))

    #     r_max_global = max(r_max_global, r.max())
    #     fov_up_global = max(fov_up_global, alpha.max())
    #     fov_down_global = min(fov_down_global, alpha.min())
    #     theta_min_global = min(theta_min_global, theta.min())
    #     theta_max_global = max(theta_max_global, theta.max())

    # print("r_max:", r_max_global)
    # print("fov_up:", fov_up_global)
    # print("fov_down:", fov_down_global)
    # print("theta:", theta_min_global, theta_max_global)

    # arr = np.load("/home/local/Desktop/code/Radar-Diffusion/COLO_RPD_Dataset/2_24_2021_aspen_run7/lidar_pcl/0.npy")
    # print(arr.shape)
    # print(arr.dtype)
    # print(arr[:5])  # Print the first 5 elements
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


class BaseMultiSensorDataset(Dataset):
    """Thin base dataset that delegates processing to a pipeline."""

    def __init__(self, cfg, pipeline):
        self.cfg = cfg
        self.pipeline = pipeline
        self.samples = self.build_index()

    def build_index(self):
        raise NotImplementedError

    def load_raw(self, meta):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = dict(self.samples[idx])
        raw = self.load_raw(meta)
        return self.pipeline(raw, meta)


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform an [N, 3] point cloud with a 4x4 rigid transform."""

    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points.reshape(0, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {points.shape}")

    T = np.asarray(T, dtype=np.float32)
    if T.shape != (4, 4):
        raise ValueError(f"Expected transform with shape [4, 4], got {T.shape}")

    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate([points, ones], axis=1)
    transformed = (T @ points_h.T).T
    return transformed[:, :3].astype(np.float32)


class Aligner:
    """Transform LiDAR points into the target frame."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, lidar_points: np.ndarray, calib: dict) -> np.ndarray:
        if str(self.cfg.target_frame).lower() != "radar":
            raise NotImplementedError("The MVP only supports target_frame='radar'.")

        if not bool(self.cfg.use_extrinsic):
            return np.asarray(lidar_points, dtype=np.float32)

        return transform_points(lidar_points, calib["T_radar_lidar"])


class RadarProcessor:
    """Convert raw radar ADC into a network-ready tensor."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, radar_adc: np.ndarray) -> torch.Tensor:
        if str(self.cfg.output_type).lower() != "spectrum":
            raise NotImplementedError(
                "The MVP only supports radar output_type='spectrum'."
            )
        spectrum = self.adc_to_spectrum(radar_adc)
        return torch.from_numpy(spectrum.astype(np.float32))

    def adc_to_spectrum(self, radar_adc: np.ndarray) -> np.ndarray:
        """Placeholder FFT pipeline for raw ADC.

        TODO: Replace the FFT axes, windowing, calibration, and tensor layout
        here with the exact ColoRadar+ raw ADC signal chain once the final
        dataset-specific convention is fixed for training.
        """

        adc = np.asarray(radar_adc)
        if adc.size == 0:
            return np.zeros((0,), dtype=np.float32)

        if adc.shape[-1] == 2 and not np.iscomplexobj(adc):
            adc_complex = adc[..., 0] + 1j * adc[..., 1]
        elif np.iscomplexobj(adc):
            adc_complex = adc
        else:
            adc_complex = adc.astype(np.complex64)

        fft_cfg = getattr(self.cfg, "fft", None)
        fft_enabled = True if fft_cfg is None else bool(getattr(fft_cfg, "enabled", True))
        fft_axes = "all" if fft_cfg is None else getattr(fft_cfg, "axes", "all")
        fft_shift = False if fft_cfg is None else bool(getattr(fft_cfg, "shift", False))

        if fft_enabled:
            axes = tuple(range(adc_complex.ndim)) if fft_axes == "all" else tuple(int(axis) for axis in fft_axes)
            spectrum = np.fft.fftn(adc_complex, axes=axes)
            if fft_shift:
                spectrum = np.fft.fftshift(spectrum, axes=axes)
        else:
            spectrum = adc_complex

        magnitude = np.abs(spectrum).astype(np.float32)
        if bool(getattr(self.cfg, "normalize", True)) and magnitude.size > 0:
            max_value = float(magnitude.max())
            if max_value > 0.0:
                magnitude /= max_value
        return magnitude


class LidarProcessor:
    """Convert aligned LiDAR points into a training representation."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, lidar_points: np.ndarray) -> torch.Tensor:
        if str(self.cfg.output_type).lower() != "point_cloud":
            raise NotImplementedError(
                "The MVP only supports lidar output_type='point_cloud'."
            )
        return self.to_point_cloud(lidar_points)

    def to_point_cloud(self, lidar_points: np.ndarray) -> torch.Tensor:
        """Return aligned xyz points.

        TODO: Add range-image, voxel, and frustum occupancy outputs here as
        additional LiDAR representations are introduced.
        """

        points = np.asarray(lidar_points, dtype=np.float32)
        if points.size == 0:
            points = points.reshape(0, 3)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected LiDAR points with shape [N, 3], got {points.shape}")
        return torch.from_numpy(points.astype(np.float32))


class MultiSensorPipeline:
    """Connect raw loading, alignment, and representation conversion."""

    def __init__(self, aligner, radar_processor, lidar_processor):
        self.aligner = aligner
        self.radar_processor = radar_processor
        self.lidar_processor = lidar_processor

    def __call__(self, raw: dict, meta: dict) -> dict:
        # TODO: Add explicit timestamp synchronization validation here once the
        # real ColoRadar+ sync metadata is wired into the dataset adapter.
        aligned_lidar_points = self.aligner(raw["lidar_points"], raw["calib"])
        radar_tensor = self.radar_processor(raw["radar_adc"])
        lidar_tensor = self.lidar_processor(aligned_lidar_points)

        sample_meta = dict(meta)
        sample_meta["timestamp"] = raw.get("timestamp")
        sample_meta["calibration_source"] = raw.get("calib", {}).get("source")
        sample_meta["radar_adc_shape"] = tuple(np.asarray(raw["radar_adc"]).shape)
        sample_meta["lidar_num_points"] = int(np.asarray(aligned_lidar_points).shape[0])

        return {
            "radar": radar_tensor,
            "lidar": lidar_tensor,
            "meta": sample_meta,
        }
