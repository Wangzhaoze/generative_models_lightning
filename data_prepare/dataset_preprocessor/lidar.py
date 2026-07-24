"""Align and preprocess ColoRadar LiDAR for single-chip or cascade radar."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from easydict import EasyDict
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "data_prepare"))

from dataset_preprocessor.constants import (  # noqa: E402
    EXCLUDE_DIR_NAMES,
    NUMBER_RECORDING_ATTRIBUTES,
    T_RADAR_TO_LIDAR,
)


def arg_parser():
    parser = argparse.ArgumentParser(description="Preprocess ColoRadar LiDAR")
    parser.add_argument(
        "--config",
        type=str,
        default="data_prepare/dataset_preprocessor/config/coloradar_config.yaml",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sc",
        choices=["sc", "cc"],
        help="sc: single-chip radar, cc: cascade radar",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_lidar_data(lidar_path: Path, return_xyz: bool = True) -> np.ndarray:
    raw = np.fromfile(lidar_path, dtype=np.float32)
    if raw.size % NUMBER_RECORDING_ATTRIBUTES != 0:
        raise ValueError(
            f"{lidar_path} contains {raw.size} float32 values, which is not "
            f"divisible by {NUMBER_RECORDING_ATTRIBUTES}"
        )
    points = raw.reshape(-1, NUMBER_RECORDING_ATTRIBUTES)
    return points[:, :3] if return_xyz else points


def transform_lidar_data(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected LiDAR points [N,3], got {points.shape}")
    homogeneous = np.hstack((points, np.ones((len(points), 1))))
    return (homogeneous @ T_RADAR_TO_LIDAR.T)[:, :3]


def cartesian2polar(points: np.ndarray) -> np.ndarray:
    x, y, z = points.T
    radius = np.linalg.norm(points, axis=1)
    azimuth = -np.rad2deg(np.arctan2(y, x))
    elevation = np.rad2deg(np.arcsin(np.clip(z / radius, -1.0, 1.0)))
    return np.stack((radius, azimuth, elevation), axis=1)


def polar2cartesian(points: np.ndarray) -> np.ndarray:
    radius = points[:, 0]
    azimuth = -np.deg2rad(points[:, 1])
    elevation = np.deg2rad(points[:, 2])
    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)
    return np.stack((x, y, z), axis=1)


def save_lidar_data(points: np.ndarray, save_path: Path) -> None:
    temporary_path = save_path.with_suffix(save_path.suffix + ".tmp")
    np.asarray(points, dtype=np.float32).tofile(temporary_path)
    os.replace(temporary_path, save_path)


def get_overlap_index(
    radar_time_stamps: list[str] | np.ndarray,
    lidar_time_stamps: list[str] | np.ndarray,
) -> tuple[list[int], list[int]]:
    """Match each radar timestamp to its nearest LiDAR timestamp."""
    radar_ts = np.asarray(radar_time_stamps, dtype=np.float64).reshape(-1)
    lidar_ts = np.asarray(lidar_time_stamps, dtype=np.float64).reshape(-1)
    if radar_ts.size == 0 or lidar_ts.size == 0:
        raise ValueError("Radar and LiDAR timestamp arrays must not be empty")
    if np.any(np.diff(lidar_ts) < 0):
        raise ValueError("LiDAR timestamps must be sorted in ascending order")

    right = np.searchsorted(lidar_ts, radar_ts, side="left")
    right = np.clip(right, 0, lidar_ts.size - 1)
    left = np.clip(right - 1, 0, lidar_ts.size - 1)
    # Strict comparison keeps RaLD's tie behaviour: when two LiDAR frames are
    # equally close, choose the later (right-hand) frame.
    use_left = np.abs(radar_ts - lidar_ts[left]) < np.abs(
        lidar_ts[right] - radar_ts
    )
    lidar_indices = np.where(use_left, left, right).astype(int)
    radar_indices = np.arange(radar_ts.size, dtype=int)
    return radar_indices.tolist(), lidar_indices.tolist()


def filter_points_polar(points: np.ndarray, limits: list[list[float]]) -> np.ndarray:
    mask = np.logical_and.reduce(
        [
            points[:, 0] >= limits[0][0],
            points[:, 0] <= limits[0][1],
            points[:, 1] >= limits[1][0],
            points[:, 1] <= limits[1][1],
            points[:, 2] >= limits[2][0],
            points[:, 2] <= limits[2][1],
        ]
    )
    return points[mask]


def remove_empty_points(points: np.ndarray) -> np.ndarray:
    return points[np.linalg.norm(points[:, :3], axis=1) > 0]


def _frame_id(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _load_lidar_indices(sequence_dir: Path, mode: str) -> list[int]:
    lidar_dir = sequence_dir / "lidar"
    sensor_name = "single_chip" if mode == "sc" else "cascade"
    candidates = (
        [
            lidar_dir / "lidar_index_sequence_single_chip.txt",
            lidar_dir / "lidar_index_sequence.txt",
        ]
        if mode == "sc"
        else [lidar_dir / "lidar_index_sequence_cascade.txt"]
    )
    for index_path in candidates:
        if index_path.is_file():
            with open(index_path, "r", encoding="utf-8") as file:
                return [int(line.strip()) for line in file if line.strip()]

    # Make lidar.py usable on new sequences even before index files are saved.
    # gen_index_file_coloradar.py writes the same matches for later reuse.
    radar_ts_path = sequence_dir / sensor_name / "adc_samples" / "timestamps.txt"
    lidar_ts_path = lidar_dir / "timestamps.txt"
    if not radar_ts_path.is_file() or not lidar_ts_path.is_file():
        raise FileNotFoundError(
            f"Missing alignment indices and/or timestamps in {sequence_dir}"
        )
    radar_ts = np.loadtxt(radar_ts_path, dtype=np.float64)
    lidar_ts = np.loadtxt(lidar_ts_path, dtype=np.float64)
    _, lidar_indices = get_overlap_index(radar_ts, lidar_ts)
    print(
        f"{sequence_dir.name}: no saved {mode} index; using nearest-timestamp "
        "alignment in memory"
    )
    return lidar_indices


def main() -> None:
    args = arg_parser()
    with open(resolve_path(args.config), "r", encoding="utf-8") as config_file:
        config = EasyDict(yaml.load(config_file, Loader=yaml.FullLoader))

    dataset_dir = resolve_path(config.root_dir)
    output_base_dir = resolve_path(config.output_dir)
    mode_key = "single_chip_mode" if args.mode == "sc" else "cascade_mode"
    lidar_settings = config[mode_key].lidar
    sensor_name = "single_chip" if args.mode == "sc" else "cascade"
    output_name = "lidar_sc" if args.mode == "sc" else "lidar_cc"

    sequence_dirs = [
        path
        for path in sorted(dataset_dir.iterdir())
        if path.is_dir()
        and path.name not in EXCLUDE_DIR_NAMES
        and (path / "lidar" / "pointclouds").is_dir()
        and (path / sensor_name / "adc_samples").is_dir()
    ]
    if not sequence_dirs:
        raise FileNotFoundError(f"No {args.mode} ColoRadar sequences in {dataset_dir}")

    fov = lidar_settings.FOV
    limits = [
        [0.0, float(fov.max_range)],
        [float(fov.az_range[0]), float(fov.az_range[1])],
        [float(fov.el_range[0]), float(fov.el_range[1])],
    ]

    for sequence_dir in tqdm(sequence_dirs, desc=f"LiDAR {args.mode}"):
        pointcloud_dir = sequence_dir / "lidar" / "pointclouds"
        files = sorted(pointcloud_dir.glob("*.bin"), key=_frame_id)
        files_by_id = {_frame_id(path): path for path in files}
        lidar_indices = _load_lidar_indices(sequence_dir, args.mode)
        missing = [index for index in lidar_indices if index not in files_by_id]
        if missing:
            raise IndexError(
                f"{sequence_dir.name} references missing LiDAR frames: {missing[:10]}"
            )

        output_dir = output_base_dir / sequence_dir.name / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for output_index, lidar_index in enumerate(
            tqdm(lidar_indices, desc=sequence_dir.name, leave=False)
        ):
            points = remove_empty_points(load_lidar_data(files_by_id[lidar_index]))
            points = transform_lidar_data(points)
            polar_points = cartesian2polar(points)
            filtered = filter_points_polar(polar_points, limits)
            save_lidar_data(
                polar2cartesian(filtered),
                output_dir / f"{output_index:04d}.bin",
            )


if __name__ == "__main__":
    main()
