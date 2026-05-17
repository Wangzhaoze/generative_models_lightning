#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Minimal ColoRadar+ raw radar-LiDAR dataset adapter."""

from pathlib import Path
import io
import json

import numpy as np

from data_module.ColoRadar.dataset import BaseMultiSensorDataset


class ColoRadarDataset(BaseMultiSensorDataset):
    """MVP dataset adapter for raw radar ADC and raw LiDAR files.

    The current implementation keeps the interfaces realistic while staying
    intentionally simple:
    - sample indexing is filename based
    - synchronization is a TODO for later
    - calibration loading supports both direct extrinsics and base-frame files
    """

    def build_index(self):
        root_dir = self._resolve_root(self.cfg.dataset.root)
        radar_paths = self._collect_files(root_dir, self.cfg.dataset.radar_glob)
        lidar_paths = self._collect_files(root_dir, self.cfg.dataset.lidar_glob)

        if not lidar_paths and getattr(self.cfg.dataset, "lidar_fallback_glob", None):
            lidar_paths = self._collect_files(root_dir, self.cfg.dataset.lidar_fallback_glob)

        radar_index = {self._pair_key(root_dir, path): path for path in radar_paths}
        lidar_index = {self._pair_key(root_dir, path): path for path in lidar_paths}
        common_keys = sorted(set(radar_index) & set(lidar_index))

        samples = []
        for sample_key in common_keys:
            radar_path = radar_index[sample_key]
            lidar_path = lidar_index[sample_key]
            sequence_dir = self._sequence_dir(root_dir, radar_path)
            frame_id = radar_path.stem
            samples.append(
                {
                    "sample_id": sample_key,
                    "sequence_id": sequence_dir.name if sequence_dir != root_dir else ".",
                    "frame_id": frame_id,
                    "timestamp": self._parse_timestamp(frame_id),
                    "radar_path": str(radar_path),
                    "lidar_path": str(lidar_path),
                    "sequence_dir": str(sequence_dir),
                    "is_dummy": False,
                }
            )

        if samples:
            return samples

        if bool(self.cfg.dataset.allow_dummy_data):
            return self._build_dummy_index()

        raise RuntimeError(
            "No matched ColoRadar+ radar/LiDAR pairs were found. "
            f"root={root_dir}, radar_glob={self.cfg.dataset.radar_glob}, "
            f"lidar_glob={self.cfg.dataset.lidar_glob}"
        )

    def load_raw(self, meta):
        if meta.get("is_dummy", False):
            calib = self._identity_calibration()
            return {
                "radar_adc": self._build_dummy_radar_adc(self.cfg.dataset.dummy_radar_shape),
                "lidar_points": self._build_dummy_lidar_points(self.cfg.dataset.dummy_num_lidar_points),
                "calib": calib,
                "timestamp": meta.get("timestamp"),
                "meta": meta,
            }

        radar_adc = self._load_radar_adc(
            meta["radar_path"],
            adc_shape=self.cfg.dataset.radar_adc_shape,
            dtype=str(self.cfg.dataset.radar_adc_dtype),
        )
        lidar_points = self._load_lidar_points(
            meta["lidar_path"],
            lidar_bin_num_features=int(self.cfg.dataset.lidar_bin_num_features),
        )
        calib = self._load_calibration(meta)

        return {
            "radar_adc": radar_adc,
            "lidar_points": lidar_points,
            "calib": calib,
            "timestamp": meta.get("timestamp"),
            "meta": meta,
        }

    def _collect_files(self, root_dir: Path, pattern: str):
        if not root_dir.exists():
            return []
        return sorted(root_dir.glob(pattern))

    def _pair_key(self, root_dir: Path, path: Path) -> str:
        relative = path.relative_to(root_dir)
        sequence_depth = max(int(self.cfg.dataset.sequence_depth), 1)
        sequence_parts = relative.parts[:sequence_depth]
        sequence_id = "/".join(sequence_parts)
        return f"{sequence_id}:{path.stem}"

    def _sequence_dir(self, root_dir: Path, path: Path) -> Path:
        relative = path.relative_to(root_dir)
        sequence_depth = max(int(self.cfg.dataset.sequence_depth), 1)
        sequence_parts = relative.parts[:sequence_depth]
        return root_dir.joinpath(*sequence_parts)

    def _build_dummy_index(self):
        return [
            {
                "sample_id": f"dummy_sample_{index:04d}",
                "sequence_id": "dummy_sequence",
                "frame_id": f"{index:06d}",
                "timestamp": float(index),
                "radar_path": None,
                "lidar_path": None,
                "sequence_dir": None,
                "is_dummy": True,
            }
            for index in range(int(self.cfg.dataset.dummy_num_samples))
        ]

    def _parse_timestamp(self, frame_id: str):
        try:
            return float(frame_id)
        except ValueError:
            # TODO: Replace this filename fallback with the real sensor
            # synchronization metadata from ColoRadar+.
            return None

    def _resolve_root(self, root: str) -> Path:
        root_path = Path(root).expanduser()
        if root_path.is_absolute():
            return root_path
        return (Path.cwd() / root_path).resolve()

    def _load_radar_adc(self, path, adc_shape=None, dtype="float32"):
        """Load raw ADC data.

        TODO: Adapt this loader to the exact ColoRadar+ raw ADC layout if the
        final training pipeline requires a different axis order or dtype.
        """

        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".npy":
            return np.load(path)
        if suffix == ".npz":
            payload = np.load(path)
            return payload["adc"] if "adc" in payload else payload[payload.files[0]]
        if suffix in {".bin", ".adc"}:
            if adc_shape is None:
                raise ValueError(
                    "Binary radar ADC loading requires datasets.<name>.radar_adc_shape "
                    "to be set in configs/config.yaml."
                )
            return np.fromfile(path, dtype=np.dtype(dtype)).reshape(tuple(adc_shape))
        if suffix in {".txt", ".csv"}:
            delimiter = "," if suffix == ".csv" else None
            return np.loadtxt(path, delimiter=delimiter)

        raise ValueError(f"Unsupported radar ADC file format: {path}")

    def _load_lidar_points(self, path, lidar_bin_num_features=4):
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".pcd":
            return self._load_pcd_points(path)
        if suffix == ".bin":
            raw = np.fromfile(path, dtype=np.float32)
            if raw.size == 0:
                return np.zeros((0, 3), dtype=np.float32)
            points = raw.reshape((-1, lidar_bin_num_features))
            return points[:, :3].astype(np.float32)
        if suffix == ".npy":
            points = np.load(path).astype(np.float32)
            return points[:, :3] if points.ndim == 2 else points
        if suffix == ".npz":
            payload = np.load(path)
            points = payload["points"] if "points" in payload else payload[payload.files[0]]
            points = np.asarray(points, dtype=np.float32)
            return points[:, :3] if points.ndim == 2 else points
        if suffix in {".txt", ".csv"}:
            delimiter = "," if suffix == ".csv" else None
            points = np.loadtxt(path, delimiter=delimiter).astype(np.float32)
            return points[:, :3] if points.ndim == 2 else points

        raise ValueError(f"Unsupported LiDAR file format: {path}")

    def _load_pcd_points(self, path: Path) -> np.ndarray:
        with path.open("rb") as handle:
            header_lines = []
            while True:
                line = handle.readline()
                if not line:
                    raise ValueError(f"Invalid PCD file, missing DATA header: {path}")
                decoded = line.decode("utf-8").strip()
                header_lines.append(decoded)
                if decoded.startswith("DATA"):
                    break
            payload = handle.read()

        header = self._parse_pcd_header(header_lines)
        data_type = header["DATA"].lower()

        if data_type == "ascii":
            points = np.loadtxt(io.StringIO(payload.decode("utf-8")))
            if points.ndim == 1:
                points = points[None, :]
            field_lookup = self._pcd_field_lookup(header)
            return np.stack(
                [
                    points[:, field_lookup["x"]].astype(np.float32),
                    points[:, field_lookup["y"]].astype(np.float32),
                    points[:, field_lookup["z"]].astype(np.float32),
                ],
                axis=1,
            )

        if data_type == "binary":
            structured = np.frombuffer(
                payload,
                dtype=self._build_pcd_dtype(header),
                count=int(header["POINTS"]),
            )
            return np.stack(
                [
                    structured["x"].astype(np.float32),
                    structured["y"].astype(np.float32),
                    structured["z"].astype(np.float32),
                ],
                axis=1,
            )

        raise NotImplementedError(
            f"PCD DATA type '{header['DATA']}' is not supported in the MVP."
        )

    def _parse_pcd_header(self, lines):
        header = {}
        for line in lines:
            if not line or line.startswith("#"):
                continue
            key, *values = line.split()
            header[key.upper()] = values if len(values) > 1 else values[0]

        if isinstance(header.get("FIELDS"), str):
            header["FIELDS"] = [header["FIELDS"]]
        if isinstance(header.get("SIZE"), str):
            header["SIZE"] = [header["SIZE"]]
        if isinstance(header.get("TYPE"), str):
            header["TYPE"] = [header["TYPE"]]
        if isinstance(header.get("COUNT"), str):
            header["COUNT"] = [header["COUNT"]]
        if "COUNT" not in header:
            header["COUNT"] = ["1"] * len(header["FIELDS"])
        return header

    def _pcd_field_lookup(self, header):
        lookup = {}
        cursor = 0
        for field_name, count in zip(header["FIELDS"], header["COUNT"]):
            lookup[field_name] = cursor
            cursor += int(count)
        missing = {"x", "y", "z"} - set(lookup)
        if missing:
            raise ValueError(f"PCD file is missing xyz fields: {sorted(missing)}")
        return lookup

    def _build_pcd_dtype(self, header):
        dtype_fields = []
        for name, size, field_type, count in zip(
            header["FIELDS"],
            [int(item) for item in header["SIZE"]],
            header["TYPE"],
            [int(item) for item in header["COUNT"]],
        ):
            base_dtype = self._pcd_scalar_dtype(size=size, field_type=field_type)
            if count == 1:
                dtype_fields.append((name, base_dtype))
            else:
                dtype_fields.append((name, base_dtype, (count,)))
        return np.dtype(dtype_fields)

    def _pcd_scalar_dtype(self, size, field_type):
        mapping = {
            ("F", 4): np.float32,
            ("F", 8): np.float64,
            ("I", 1): np.int8,
            ("I", 2): np.int16,
            ("I", 4): np.int32,
            ("U", 1): np.uint8,
            ("U", 2): np.uint16,
            ("U", 4): np.uint32,
        }
        key = (field_type.upper(), int(size))
        if key not in mapping:
            raise ValueError(f"Unsupported PCD dtype mapping: {key}")
        return np.dtype(mapping[key])

    def _load_calibration(self, meta):
        sequence_dir = Path(meta["sequence_dir"]) if meta.get("sequence_dir") else None
        root_dir = self._resolve_root(self.cfg.dataset.root)

        for candidate in self.cfg.dataset.direct_extrinsic_candidates:
            candidate_path = self._resolve_candidate_path(candidate, sequence_dir, root_dir)
            if candidate_path is not None and candidate_path.exists():
                matrix = self._load_matrix_file(candidate_path)
                return {
                    "T_radar_lidar": np.asarray(matrix, dtype=np.float32).reshape(4, 4),
                    "source": str(candidate_path),
                }

        base_to_lidar = self._resolve_candidate_path(
            self.cfg.dataset.base_to_lidar_path,
            sequence_dir,
            root_dir,
        )
        base_to_radar = self._resolve_candidate_path(
            self.cfg.dataset.base_to_radar_path,
            sequence_dir,
            root_dir,
        )

        if base_to_lidar is not None and base_to_radar is not None:
            T_base_lidar = self._load_pose_txt(base_to_lidar)
            T_base_radar = self._load_pose_txt(base_to_radar)
            T_radar_lidar = np.linalg.inv(T_base_radar) @ T_base_lidar
            return {
                "T_radar_lidar": T_radar_lidar.astype(np.float32),
                "source": f"{base_to_radar} + {base_to_lidar}",
            }

        if bool(self.cfg.dataset.fallback_to_identity_calibration):
            return self._identity_calibration()

        raise FileNotFoundError("Could not resolve ColoRadar+ calibration files.")

    def _resolve_candidate_path(self, relative_path, sequence_dir, root_dir):
        if relative_path is None:
            return None

        path = Path(relative_path)
        if path.is_absolute():
            return path

        search_roots = []
        if sequence_dir is not None:
            search_roots.append(sequence_dir)
        search_roots.append(root_dir)

        for base in search_roots:
            candidate = base / path
            if candidate.exists():
                return candidate
        return None

    def _load_matrix_file(self, path: Path):
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.load(path)
        if suffix == ".npz":
            payload = np.load(path)
            return payload["T_radar_lidar"] if "T_radar_lidar" in payload else payload[payload.files[0]]
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                if "T_radar_lidar" in payload:
                    return payload["T_radar_lidar"]
                if "matrix" in payload:
                    return payload["matrix"]
            return payload
        return np.loadtxt(path)

    def _load_pose_txt(self, path: Path) -> np.ndarray:
        with path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle.readlines() if line.strip()]

        if len(lines) < 2:
            raise ValueError(f"Invalid pose calibration file: {path}")

        translation = np.array([float(item) for item in lines[0].split()], dtype=np.float32)
        quaternion = np.array([float(item) for item in lines[1].split()], dtype=np.float32)

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = self._quat_to_matrix(quaternion)
        T[:3, 3] = translation
        return T

    def _quat_to_matrix(self, quaternion):
        qx, qy, qz, qw = quaternion.astype(np.float32)
        norm = np.linalg.norm([qx, qy, qz, qw])
        if norm == 0.0:
            return np.eye(3, dtype=np.float32)
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

        return np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=np.float32,
        )

    def _identity_calibration(self):
        return {
            "T_radar_lidar": np.eye(4, dtype=np.float32),
            "source": "identity_fallback",
        }

    def _build_dummy_radar_adc(self, shape):
        total_values = int(np.prod(shape))
        adc = np.linspace(0.0, 1.0, total_values, dtype=np.float32).reshape(tuple(shape))
        if adc.ndim > 0 and adc.shape[-1] == 2:
            adc[..., 1] *= 0.5
        return adc

    def _build_dummy_lidar_points(self, num_points):
        angles = np.linspace(0.0, 2.0 * np.pi, int(num_points), endpoint=False, dtype=np.float32)
        radius = np.linspace(2.0, 8.0, int(num_points), dtype=np.float32)
        z = np.linspace(-1.0, 1.0, int(num_points), dtype=np.float32)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        return np.stack([x, y, z], axis=1).astype(np.float32)
