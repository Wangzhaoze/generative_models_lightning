"""Generate SC/CC radar-to-LiDAR alignment files for raw ColoRadar data.

The nearest-timestamp matching rule is adapted from RaLD's
``dataset_preprocessor/depth/gen_index_file_coloradar.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def arg_parser():
    parser = argparse.ArgumentParser(description="Generate ColoRadar indices")
    parser.add_argument(
        "--config",
        type=str,
        default="data_prepare/dataset_preprocessor/config/coloradar_config.yaml",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_timestamps(path: Path) -> np.ndarray:
    timestamps = np.atleast_1d(np.loadtxt(path, dtype=np.float64))
    if timestamps.size == 0:
        raise ValueError(f"No timestamps found in {path}")
    if np.any(np.diff(timestamps) < 0):
        raise ValueError(f"Timestamps must be sorted: {path}")
    return timestamps


def match_reference_to_lidar(
    reference_timestamps: np.ndarray,
    lidar_timestamps: np.ndarray,
) -> list[int]:
    right = np.searchsorted(lidar_timestamps, reference_timestamps, side="left")
    right = np.clip(right, 0, lidar_timestamps.size - 1)
    left = np.clip(right - 1, 0, lidar_timestamps.size - 1)
    # Match RaLD exactly: an equal-distance tie selects the later timestamp.
    use_left = np.abs(reference_timestamps - lidar_timestamps[left]) < np.abs(
        lidar_timestamps[right] - reference_timestamps
    )
    return np.where(use_left, left, right).astype(int).tolist()


def write_indices(indices: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.writelines(f"{index}\n" for index in indices)


def save_sensor_alignment(
    sequence_dir: Path,
    sensor_name: str,
    lidar_index_name: str,
) -> None:
    sensor_dir = sequence_dir / sensor_name / "adc_samples"
    sensor_timestamp_path = sensor_dir / "timestamps.txt"
    if not sensor_timestamp_path.is_file():
        return

    lidar_timestamps = load_timestamps(sequence_dir / "lidar" / "timestamps.txt")
    sensor_timestamps = load_timestamps(sensor_timestamp_path)
    lidar_indices = match_reference_to_lidar(sensor_timestamps, lidar_timestamps)
    sensor_indices = list(range(sensor_timestamps.size))

    lidar_index_path = sequence_dir / "lidar" / lidar_index_name
    sensor_index_path = sensor_dir / "radar_index_sequence.txt"
    write_indices(lidar_indices, lidar_index_path)
    write_indices(sensor_indices, sensor_index_path)

    if sensor_name == "single_chip":
        write_indices(
            lidar_indices,
            sequence_dir / "lidar" / "lidar_index_sequence.txt",
        )

    dt = np.abs(sensor_timestamps - lidar_timestamps[np.asarray(lidar_indices)])
    print(
        f"[{sequence_dir.name}] {sensor_name}: {len(sensor_indices)} pairs, "
        f"|dt| mean={dt.mean():.6f}s max={dt.max():.6f}s"
    )


def main() -> None:
    args = arg_parser()
    with open(resolve_path(args.config), "r", encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    dataset_dir = resolve_path(config["root_dir"])

    sequence_dirs = sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_dir() and path.name != "calib"
    )
    for sequence_dir in sequence_dirs:
        lidar_timestamps = sequence_dir / "lidar" / "timestamps.txt"
        if not lidar_timestamps.is_file():
            print(f"[{sequence_dir.name}] skipped: no LiDAR timestamps")
            continue
        save_sensor_alignment(
            sequence_dir,
            "single_chip",
            "lidar_index_sequence_single_chip.txt",
        )
        save_sensor_alignment(
            sequence_dir,
            "cascade",
            "lidar_index_sequence_cascade.txt",
        )


if __name__ == "__main__":
    main()
