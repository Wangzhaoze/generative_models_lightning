"""Preprocess single-chip (sc) or cascade (cc) raw radar ADC frames."""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch as th
import yaml
from easydict import EasyDict
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "data_prepare"))

from dataset_preprocessor.constants import EXCLUDE_DIR_NAMES
from dataset_preprocessor.utils.radar_preprocessing import RAEIVVmap


def arg_parser():
    parser = argparse.ArgumentParser(description="Preprocess raw radar ADC data")
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
    doa_group = parser.add_mutually_exclusive_group()
    doa_group.add_argument("--doa", dest="use_doa", action="store_true")
    doa_group.add_argument("--no-doa", dest="use_doa", action="store_false")
    parser.set_defaults(use_doa=None)
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def antenna_array(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    rx_layout = []
    tx_layout = []
    with open(resolve_path(file_path), "r", encoding="utf-8") as file:
        for line in file:
            chunks = line.strip().split()
            if not chunks or chunks[0].startswith("#"):
                continue
            if chunks[0] == "rx":
                rx_layout.append([int(value) for value in chunks[1:]])
            elif chunks[0] == "tx":
                tx_layout.append([int(value) for value in chunks[1:]])
    if not tx_layout or not rx_layout:
        raise ValueError(f"No TX/RX antenna layout found in {file_path}")
    return np.asarray(tx_layout), np.asarray(rx_layout)


def save_radarcube(radarcube: np.ndarray, save_path: Path) -> None:
    if isinstance(radarcube, th.Tensor):
        radarcube = radarcube.detach().cpu().numpy()
    radarcube = np.asarray(radarcube, dtype=np.float32)
    temporary_path = save_path.with_suffix(save_path.suffix + ".tmp")
    radarcube.tofile(temporary_path)
    os.replace(temporary_path, save_path)


def load_radar_data(radar_config, radar_path: Path) -> np.ndarray:
    raw = np.fromfile(radar_path, dtype=np.int16)
    shape = (
        int(radar_config.numTxChan),
        int(radar_config.numRxChan),
        int(radar_config.numChirpsPerFrame),
        int(radar_config.numAdcSamples),
        2,
    )
    expected_values = int(np.prod(shape))
    if raw.size != expected_values:
        raise ValueError(
            f"{radar_path} contains {raw.size} int16 values; "
            f"expected {expected_values} for shape {shape}"
        )
    raw = raw.reshape(shape)
    adc = raw[..., 0] + 1j * raw[..., 1]
    return adc - np.mean(adc)


def _frame_id(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _selected_adc_files(adc_dir: Path, index_file: Path) -> list[Path]:
    files = sorted(adc_dir.glob("*.bin"), key=_frame_id)
    file_by_id = {_frame_id(path): path for path in files}
    if not index_file.is_file():
        raise FileNotFoundError(
            f"Radar/LiDAR index file not found: {index_file}. "
            "Create temporal alignment before radar preprocessing."
        )
    with open(index_file, "r", encoding="utf-8") as file:
        indices = [int(line.strip()) for line in file if line.strip()]
    missing = [index for index in indices if index not in file_by_id]
    if missing:
        raise IndexError(
            f"{index_file} references missing radar frames: {missing[:10]}"
        )
    return [file_by_id[index] for index in indices]


def _load_radar_config(path_value: str | Path) -> EasyDict:
    with open(resolve_path(path_value), "r", encoding="utf-8") as file:
        radar_config = EasyDict(yaml.load(file, Loader=yaml.FullLoader))
    radar_config.chirpRampTime = (
        radar_config.SamplePerChripUp / radar_config.Fs
    )
    radar_config.chirpBandwidth = radar_config.Kr * radar_config.chirpRampTime
    radar_config.max_range = (
        3.0e8 * radar_config.chirpRampTime * radar_config.Fs
    ) / (2.0 * radar_config.chirpBandwidth)
    return radar_config


def _subproc_process_radar(params) -> int:
    expected_shape = (
        (
            int(params.radar_config.range_fftsize),
            int(params.radar_config.ANGLE_fftsize),
            int(params.radar_config.ELEVATION_fftsize),
            3,
        )
        if params.use_doa
        else (
            int(params.radar_config.range_fftsize),
            int(params.radar_config.numRxChan),
            int(params.radar_config.numTxChan),
            3,
        )
    )
    expected_bytes = int(np.prod(expected_shape)) * np.dtype(np.float32).itemsize

    processed = 0
    for output_index, adc_file in enumerate(params.adc_files):
        output_path = params.out_dir / f"{output_index:04d}.bin"
        if output_path.is_file() and output_path.stat().st_size == expected_bytes:
            continue
        adc_data = load_radar_data(params.radar_config, adc_file)
        radar_cube = RAEIVVmap(
            adc_data,
            params.radar_config,
            params.tx_array,
            params.rx_array,
            use_doa=params.use_doa,
            allow_aperture_truncation=params.allow_aperture_truncation,
        )
        save_radarcube(radar_cube, output_path)
        processed += 1
    return processed


def _run(params_list: list[EasyDict], num_workers: int) -> None:
    if num_workers <= 1:
        for params in tqdm(params_list, desc="Processing radar sequences"):
            _subproc_process_radar(params)
        return
    with Pool(processes=num_workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(_subproc_process_radar, params_list),
                total=len(params_list),
                desc="Processing radar sequences",
            )
        )


def main():
    args = arg_parser()
    with open(resolve_path(args.config), "r", encoding="utf-8") as config_file:
        config = EasyDict(yaml.load(config_file, yaml.FullLoader))

    sensor_name = "single_chip" if args.mode == "sc" else "cascade"
    mode_key = "single_chip_mode" if args.mode == "sc" else "cascade_mode"
    if mode_key not in config:
        raise KeyError(
            f"Config {args.config} must define '{mode_key}.radar' for mode {args.mode}"
        )
    radar_settings = config[mode_key].radar
    use_doa = (
        bool(radar_settings.get("use_DOA", False))
        if args.use_doa is None
        else bool(args.use_doa)
    )

    dataset_dir = resolve_path(config.root_dir)
    output_base_dir = resolve_path(config.output_dir)
    sequence_dirs = [
        path
        for path in sorted(dataset_dir.iterdir())
        if path.is_dir()
        and path.name not in EXCLUDE_DIR_NAMES
        and (path / sensor_name / "adc_samples" / "data").is_dir()
    ]
    if not sequence_dirs:
        raise FileNotFoundError(
            f"No {sensor_name} radar sequences found under {dataset_dir}"
        )

    radar_config = _load_radar_config(radar_settings.config)
    tx_array, rx_array = antenna_array(radar_settings.antenna_file_path)
    params_list = []
    for sequence_dir in sequence_dirs:
        sensor_dir = sequence_dir / sensor_name / "adc_samples"
        selected_files = _selected_adc_files(
            sensor_dir / "data",
            sensor_dir / "radar_index_sequence.txt",
        )
        output_dir = output_base_dir / sequence_dir.name / sensor_name / "radarcube_raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"{sequence_dir.name}: {len(selected_files)} aligned {sensor_name} "
            f"frames, use_doa={use_doa}"
        )

        params_list.append(
            EasyDict(
                adc_files=selected_files,
                out_dir=output_dir,
                radar_config=radar_config,
                tx_array=tx_array,
                rx_array=rx_array,
                use_doa=use_doa,
                allow_aperture_truncation=bool(
                    radar_settings.get("allow_doa_aperture_truncation", False)
                ),
            )
        )

    _run(params_list, int(config.get("num_workers", 1)))


if __name__ == "__main__":
    main()
