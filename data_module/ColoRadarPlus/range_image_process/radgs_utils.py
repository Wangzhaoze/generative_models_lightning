import os
import open3d as o3d
from typing import Optional, Union
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

import numpy as np
import cv2
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

def fill_depth_image(depth_img, invalid_val=1e-3):
    # Step 1: 去除异常值（中值滤波）
    cleaned = median_filter(depth_img, size=3)

    # # Step 2: 创建掩码（无效值为0或NaN）
    # mask = (cleaned <= invalid_val) | np.isnan(cleaned)

    # # Step 3: 均值插值（使用邻域均值）
    # filled = cleaned.copy()
    # filled[mask] = 0  # 先置零

    # kernel = np.ones((5, 5), np.float32)
    # avg = cv2.filter2D(filled, -1, kernel) / cv2.filter2D((~mask).astype(np.uint8), -1, kernel)
    # filled[mask] = avg[mask]

    # Step 4: 使用双边滤波保留边缘
    filled = cv2.bilateralFilter(cleaned.astype(np.float32), d=5, sigmaColor=75, sigmaSpace=75)

    return filled


def load_point_cloud(pcd_path: str) -> o3d.geometry.PointCloud:
    """
    Load a point cloud from a file.

    Args:
        pcd_path (str): Path to the point cloud file.

    Returns:
        o3d.geometry.PointCloud: Loaded point cloud.
    """
    return o3d.io.read_point_cloud(pcd_path)


def merge_point_clouds(point_clouds: list) -> o3d.geometry.PointCloud:
    """
    Merge multiple point clouds into a single point cloud.

    Args:
        point_clouds (list): List of point clouds to merge.

    Returns:
        o3d.geometry.PointCloud: Merged point cloud.
    """
    # Concatenate point coordinates and colors
    merged_points = np.concatenate(
        [np.asarray(pcd.points) for pcd in point_clouds], axis=0
    )
    merged_colors = np.concatenate(
        [np.asarray(pcd.colors) for pcd in point_clouds], axis=0
    )

    # Create a new point cloud

    return numpy_to_open3d_pointcloud(points=merged_points, colors=merged_colors)


def numpy_to_open3d_pointcloud(
    points: np.ndarray, colors: Optional[np.ndarray] = None
) -> o3d.geometry.PointCloud:
    """
    Converts a NumPy array to Open3D point cloud data.

    Args:
        numpy_array (numpy.ndarray): NumPy array representing the point cloud with shape (N, 3).

    Returns:
        open3d.geometry.PointCloud: Open3D point cloud data.

    Raises:
        ValueError: If the input object is not a NumPy array.
    """

    if isinstance(points, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        raise ValueError('Input object should be a numpy.ndarray object')

    if colors is not None and len(colors) != 0:
        if np.max(colors) > 1:
            colors = (colors / np.max(colors)).astype(np.float32)
        # make sure numpy version < 2.0.0 (not fixed until 2025-02-27)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def save_point_cloud(pcd: o3d.geometry.PointCloud, save_path: str) -> None:
    """
    Save a point cloud to a file.

    Args:
        pcd (o3d.geometry.PointCloud): Point cloud to be saved.
        save_path (str): Path to save the point cloud.
    """
    o3d.io.write_point_cloud(save_path, pcd)


def visualize_point_cloud(
    pcd: Union[o3d.geometry.PointCloud, list], title='point cloud'
):
    """
    Visualize one or more point clouds.

    Args:
        pcd (Union[o3d.geometry.PointCloud, list]): Point cloud or a list of point clouds.
        title (str): Title for the visualization window.
    """
    # Visualize point cloud
    if isinstance(pcd, list):
        pass  # Placeholder for handling multiple point clouds
    else:
        pcd = [pcd]

    o3d.visualization.draw_geometries(pcd, window_name=title)

def down_sample_point_cloud(
    input_pcd: o3d.geometry.PointCloud, total_num_sample: int
) -> np.ndarray:
    """
    Sample the input Open3D point cloud by dividing it into a 3D grid and uniformly sampling a specified number of points.

    Parameters:
    - input_pcd: Input Open3D point cloud object.
    - total_num_sample: Total number of points to be sampled.

    Returns:
    - sampled_indices: Sampled Open3D point cloud object.
    """

    # Calculate the grid resolution based on the total number of points
    total_num_points = len(input_pcd.points)
    sample_grid_resolution = (total_num_points / total_num_sample) ** (1 / 3)

    # Calculate the grid index for each point
    grid_indices = np.floor(
        np.asarray(input_pcd.points) / sample_grid_resolution
    ).astype(int)

    # Use a dictionary to store the indices of points in each grid voxel
    grid_points_dict = {}
    for i, grid_index in enumerate(grid_indices):
        grid_index_tuple = tuple(grid_index)
        if grid_index_tuple not in grid_points_dict:
            grid_points_dict[grid_index_tuple] = []
        grid_points_dict[grid_index_tuple].append(i)

    # Calculate the number of points to be sampled in each grid voxel
    total_num_grid = len(grid_points_dict)
    sample_count_per_grid = int(total_num_sample / total_num_grid)

    # Uniformly sample points within each grid voxel
    sampled_indices = []
    for grid_index_tuple, point_indices in grid_points_dict.items():
        if len(point_indices) >= sample_count_per_grid:
            sampled_indices.extend(
                np.random.choice(point_indices, sample_count_per_grid, replace=False)
            )
        else:
            sampled_indices.extend(point_indices)

    return np.asarray(sampled_indices)


def sample_point_cloud(
    input_pcd: o3d.geometry.PointCloud,
    sample_mask: Optional[np.ndarray] = None,
    sample_indices: Union[list, np.ndarray] = None,
) -> o3d.geometry.PointCloud:
    """
    Sample a point cloud based on a mask or index and return a new point cloud with colors.

    Parameters:
    - input_pc: Input Open3D point cloud.
    - sample_mask: A boolean mask indicating which points to sample.
    - sample_indices: A list of indices indicating which points to sample.

    Returns:
    - sampled_pc: Sampled point cloud with colors.
    """
    if sample_mask is not None and sample_indices is not None:
        raise ValueError('Please provide either sample_mask or sample_indices')

    points = np.asarray(input_pcd.points)
    colors = np.asarray(input_pcd.colors)

    if sample_mask is not None:
        if len(sample_mask) != len(points):
            raise ValueError(
                'Length of sample_mask must be the same as the number of points in the input point cloud.'
            )
        sampled_indices = np.where(sample_mask)[0]
    elif sample_indices is not None:
        sampled_indices = sample_indices
    else:
        raise ValueError('Please provide either sample_mask or sample_indices.')

    sampled_points = points[sampled_indices]
    if len(colors) != 0:
        sampled_colors = colors[sampled_indices]

        return numpy_to_open3d_pointcloud(points=sampled_points, colors=sampled_colors)
    else:
        return numpy_to_open3d_pointcloud(points=sampled_points)
    

def interpolate_quaternion_poses(src_poses, src_stamps, tgt_stamps):

  src_start_idx = 0
  tgt_start_idx = 0
  src_end_idx = len(src_stamps) - 1
  tgt_end_idx = len(tgt_stamps) - 1

  # ensure first source timestamp is immediately before first target timestamp
  while tgt_start_idx < tgt_end_idx and tgt_stamps[tgt_start_idx] < src_stamps[src_start_idx]:
    tgt_start_idx += 1

  # ensure last source timestamp is immediately after last target timestamp
  while tgt_end_idx > tgt_start_idx and tgt_stamps[tgt_end_idx] > src_stamps[src_end_idx]:
    tgt_end_idx -= 1

  # iterate through target timestamps, 
  # interpolating a pose for each as a 4x4 transformation matrix
  tgt_idx = tgt_start_idx
  src_idx = src_start_idx
  tgt_poses = []
  while tgt_idx <= tgt_end_idx and src_idx <= src_end_idx:
    # find source timestamps bracketing target timestamp
    while src_idx + 1 <= src_end_idx and src_stamps[src_idx + 1] < tgt_stamps[tgt_idx]:
      src_idx += 1

    # get interpolation coefficient, clamped to [0,1] for boundary safety
    c = ((tgt_stamps[tgt_idx] - src_stamps[src_idx])
          / (src_stamps[src_idx+1] - src_stamps[src_idx]))
    c = float(np.clip(c, 0.0, 1.0))

    # interpolate position
    pose = np.eye(4)
    pose[:3,3] = ((1.0 - c) * src_poses[src_idx][0:3]
                        + c * src_poses[src_idx+1][0:3])

    # interpolate orientation
    r_src = Rotation.from_quat([src_poses[src_idx][3:7],
                            src_poses[src_idx+1][3:7]])
    slerp = Slerp([0,1],r_src)
    pose[:3,:3] = slerp([c])[0].as_matrix()

    tgt_poses.append(pose)

    # advance target index
    tgt_idx += 1

  return tgt_poses


def xyz2aer(points: np.ndarray, as_degrees: bool = True) -> np.ndarray:
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


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a quaternion in xyzw order to a 3x3 rotation matrix."""
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
    ], dtype=np.float32)


def transform_pcd(pcd: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Apply a 4x4 SE(3) pose to an (N,3) point cloud."""
    pcd_h = np.hstack([pcd, np.ones((len(pcd), 1), dtype=np.float32)])
    return (pose @ pcd_h.T).T[:, :3].astype(np.float32)


def radar_adc_to_pointcloud(
    adc: np.ndarray,
    radar_obj,
    threshold_db: float = 20.0,
) -> np.ndarray:
    """Convert raw radar ADC samples to a radar-local point cloud.

    Pipeline: angle FFT -> collapse Doppler -> robust dB threshold -> angle/range
    bin coordinates to xyz. This keeps the dataset focused on sample indexing,
    while radar preprocessing stays in range_image_process.
    """
    power_4d = radar_obj.angle_fft(adc)          # (Ne=32, Na=128, Nc=16, Ns=256)
    power_3d = power_4d.max(axis=2)              # collapse Doppler -> (32,128,256)

    noise = np.percentile(power_3d, 30)
    power_db = 10.0 * np.log10(power_3d / (noise + 1e-8) + 1e-8)
    el_idx, az_idx, r_idx = np.where(power_db > threshold_db)

    if len(r_idx) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    Ne, Na = 32, 128
    B = (
        radar_obj.waveform.chirpSlope
        * radar_obj.sampler.numSamplesPerChirp
        / radar_obj.sampler.adcSampleRate
    )
    range_res = 3e8 / (2.0 * B)

    r_m = r_idx.astype(np.float32) * range_res
    sin_az = np.clip((az_idx.astype(np.float32) - Na / 2.0) / (Na / 2.0), -1.0, 1.0)
    sin_el = np.clip((el_idx.astype(np.float32) - Ne / 2.0) / (Ne / 2.0), -1.0, 1.0)
    az_r = np.arcsin(sin_az)
    el_r = np.arcsin(sin_el)
    cos_el = np.cos(el_r)
    x = r_m * cos_el * np.cos(az_r)
    y = r_m * cos_el * np.sin(az_r)
    z = r_m * np.sin(el_r)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def project_scene_to_range_image(
    scene_pcd: np.ndarray,
    sensor_pose: np.ndarray,
    az_fov: float,
    el_fov: float,
    max_range: float,
    az_res: float,
    el_res: float,
    coord_order: Optional[list] = None,
) -> torch.Tensor:
    """Project a world-coordinate scene point cloud into a normalized range image."""
    sensor_pos = sensor_pose[:3, 3]
    dist_to_sensor = np.linalg.norm(scene_pcd - sensor_pos, axis=1)
    nearby = scene_pcd[dist_to_sensor < max_range * 1.5]

    az_bins = int(az_fov * 2 / az_res)
    el_bins = int(el_fov * 2 / el_res)
    if len(nearby) == 0:
        return torch.full((1, el_bins, az_bins), -1.0, dtype=torch.float32)

    nearby_h = np.hstack([nearby, np.ones((len(nearby), 1), dtype=np.float32)])
    pts_local = (np.linalg.inv(sensor_pose) @ nearby_h.T).T[:, :3]

    if coord_order is not None:
        pts_local = pts_local[:, coord_order]

    range_image, _, _ = compute_range_image_with_visibility(
        xyz_abs=pts_local,
        antenna_pose=np.eye(4, dtype=np.float32),
        az_fov=az_fov,
        el_fov=el_fov,
        max_range=max_range,
        az_res=az_res,
        el_res=el_res,
    )
    range_image = range_image / max_range * 2.0 - 1.0
    return torch.from_numpy(range_image).unsqueeze(0).float()


def _coerce_loaded_points(loaded_obj) -> np.ndarray:
    if torch.is_tensor(loaded_obj):
        points = loaded_obj.detach().cpu().numpy()
    elif isinstance(loaded_obj, np.ndarray):
        points = loaded_obj
    elif isinstance(loaded_obj, dict):
        preferred_keys = (
            "points",
            "point_cloud",
            "pcd",
            "xyz",
            "scene_pcd",
            "lidar_points",
        )
        for key in preferred_keys:
            if key in loaded_obj:
                return _coerce_loaded_points(loaded_obj[key])
        for value in loaded_obj.values():
            try:
                return _coerce_loaded_points(value)
            except ValueError:
                continue
        raise ValueError("Could not find point cloud data inside the loaded dict object")
    elif isinstance(loaded_obj, (list, tuple)):
        for value in loaded_obj:
            try:
                return _coerce_loaded_points(value)
            except ValueError:
                continue
        raise ValueError("Could not find point cloud data inside the loaded list/tuple object")
    elif hasattr(loaded_obj, "points"):
        points = np.asarray(loaded_obj.points)
    else:
        raise ValueError(f"Unsupported point cloud container type: {type(loaded_obj)!r}")

    return np.asarray(points, dtype=np.float32)


def _load_xyz_points(points_or_path: Union[str, os.PathLike, np.ndarray]) -> np.ndarray:
    if isinstance(points_or_path, (str, os.PathLike)):
        path = os.fspath(points_or_path)
        suffix = os.path.splitext(path)[1].lower()
        if suffix in {".pt", ".pth"}:
            try:
                loaded_obj = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                loaded_obj = torch.load(path, map_location="cpu")
            points = _coerce_loaded_points(loaded_obj)
        else:
            points = np.load(path).astype(np.float32)
    else:
        points = _coerce_loaded_points(points_or_path)

    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Point cloud should have shape (N,3+) or be empty, got {points.shape}")

    points = points[:, :3]
    return points[np.isfinite(points).all(axis=1)].astype(np.float32)


def load_xyz_points(points_or_path: Union[str, os.PathLike, np.ndarray]) -> np.ndarray:
    """Load xyz points from ndarray, .npy, .pt/.pth, or simple point-cloud containers."""
    return _load_xyz_points(points_or_path)


def lidar_raw_to_range_process_coords(points_or_path: Union[str, os.PathLike, np.ndarray]) -> np.ndarray:
    """Map raw LiDAR xyz to the coordinate convention used by range_image_process.

    Raw seq/lidar_pcl points follow a different axis convention than the
    spherical projection helpers in this module. This converts:
      raw (x, y, z) -> process (x, y, z) = (-z, y, x)
    """
    points = _load_xyz_points(points_or_path)
    if len(points) == 0:
        return points

    converted = np.empty_like(points)
    converted[:, 0] = -points[:, 2]
    converted[:, 1] = points[:, 1]
    converted[:, 2] = points[:, 0]
    return converted


def lidar_range_process_to_raw_coords(points_or_path: Union[str, os.PathLike, np.ndarray]) -> np.ndarray:
    """Map range_image_process LiDAR xyz back to the raw seq/lidar_pcl convention.

    Inverse of `lidar_raw_to_range_process_coords`:
      process (x, y, z) -> raw (x, y, z) = (z, y, -x)
    """
    points = _load_xyz_points(points_or_path)
    if len(points) == 0:
        return points

    converted = np.empty_like(points)
    converted[:, 0] = points[:, 2]
    converted[:, 1] = points[:, 1]
    converted[:, 2] = -points[:, 0]
    return converted


def lidar_points_to_range_image(
    points_or_path: Union[str, os.PathLike, np.ndarray],
    antenna_pose: Optional[np.ndarray] = None,
    az_fov: float = 180.0,
    el_fov: float = 22.0,
    max_range: float = 13.0,
    az_res: float = 1.0,
    el_res: float = 1.0,
) -> torch.Tensor:
    """Project a LiDAR point cloud to a normalized single-channel range image."""
    if az_res <= 0 or el_res <= 0 or max_range <= 0:
        raise ValueError("LiDAR range image parameters must be positive") 

    az_bins = max(1, int(round(az_fov * 2.0 / az_res)))
    el_bins = max(1, int(round(el_fov * 2.0 / el_res)))
    empty = torch.full((1, el_bins, az_bins), -1.0, dtype=torch.float32)

    points = _load_xyz_points(points_or_path)
    if len(points) == 0:
        return empty

    points_h = np.hstack([points, np.ones((len(points), 1), dtype=np.float32)])
    if antenna_pose is None:
        points_local = points_h[:, :3]
    else:
        points_local = (np.linalg.inv(antenna_pose) @ points_h.T).T[:, :3]

    azimuth, elevation, point_range = xyz2aer(points_local, as_degrees=True).T
    visible = (
        (azimuth >= -az_fov) & (azimuth <= az_fov)
        & (elevation >= -el_fov) & (elevation <= el_fov)
        & (point_range > 0.0) & (point_range <= max_range)
    )
    if not visible.any():
        return empty

    azimuth = azimuth[visible]
    elevation = elevation[visible]
    point_range = point_range[visible]

    order = np.argsort(point_range)[::-1]
    azimuth = azimuth[order]
    elevation = elevation[order]
    point_range = point_range[order]

    col = np.floor((azimuth + az_fov) / az_res).astype(np.int32)
    row = np.floor((elevation + el_fov) / el_res).astype(np.int32)
    col = np.clip(col, 0, az_bins - 1)
    row = np.clip(row, 0, el_bins - 1)

    range_image = np.zeros((el_bins, az_bins), dtype=np.float32)
    range_image[row, col] = point_range
    normalized = range_image / max_range * 2.0 - 1.0
    return torch.from_numpy(normalized).unsqueeze(0).float()


def lidar_range_image_to_point_cloud(
    range_image: Union[torch.Tensor, np.ndarray],
    max_range: float = 13.0,
    az_fov: float = 180.0,
    el_fov: float = 22.0,
    az_res: float = 1.0,
    el_res: float = 1.0,
    min_range: float = 1e-3,
    is_normalized: bool = True,
) -> np.ndarray:
    """Reconstruct a LiDAR point cloud from the normalized range image above."""
    if torch.is_tensor(range_image):
        ri = range_image.detach().cpu().numpy()
    else:
        ri = np.asarray(range_image, dtype=np.float32)

    if ri.ndim == 3:
        if ri.shape[0] != 1:
            raise ValueError(f"LiDAR range image should have shape (1,H,W) or (H,W), got {ri.shape}")
        ri = ri[0]
    elif ri.ndim != 2:
        raise ValueError(f"LiDAR range image should have shape (1,H,W) or (H,W), got {ri.shape}")

    if is_normalized:
        point_range = (ri + 1.0) * 0.5 * max_range
    else:
        point_range = ri.astype(np.float32)

    rows, cols = np.indices(ri.shape, dtype=np.float32)
    azimuth = np.deg2rad(-az_fov + (cols + 0.5) * az_res)
    elevation = np.deg2rad(-el_fov + (rows + 0.5) * el_res)

    x = point_range * np.sin(elevation)
    yz = point_range * np.cos(elevation)
    y = -yz * np.sin(azimuth)
    z = yz * np.cos(azimuth)

    valid = np.isfinite(point_range) & (point_range > min_range)
    return np.stack([x[valid], y[valid], z[valid]], axis=1).astype(np.float32)


def radar_points_to_range_image(
    points_or_path: Union[str, os.PathLike, np.ndarray],
    r_max: float = 14.0,
    n_channels: int = 16,
    az_fov: float = 64.0,
    el_fov: float = 15.0,
    az_res: Optional[float] = None,
    el_res: Optional[float] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    min_range: float = 1e-3,
    coord_order: Optional[list[int]] = None,
) -> torch.Tensor:
    """Project a radar point cloud to a normalized multi-channel range image."""
    if r_max <= 0 or n_channels <= 0:
        raise ValueError("Radar range image parameters must be positive")

    if width is None:
        if az_res is None or az_res <= 0:
            raise ValueError("Provide either a positive az_res or width for radar projection")
        width = max(1, int(round(az_fov * 2.0 / az_res)))
    else:
        az_res = (az_fov * 2.0) / float(width)

    if height is None:
        if el_res is None or el_res <= 0:
            raise ValueError("Provide either a positive el_res or height for radar projection")
        height = max(1, int(round(el_fov * 2.0 / el_res)))
    else:
        el_res = (el_fov * 2.0) / float(height)

    empty = torch.full((n_channels, height, width), -1.0, dtype=torch.float32)
    points = _load_xyz_points(points_or_path)
    if len(points) == 0:
        return empty

    if coord_order is not None:
        points = points[:, coord_order]

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    point_range = np.sqrt(x**2 + y**2 + z**2)
    finite = np.isfinite(point_range) & (point_range > min_range) & (point_range <= r_max)
    if not finite.any():
        return empty

    x, y, z, point_range = x[finite], y[finite], z[finite], point_range[finite]
    azimuth = np.degrees(np.arctan2(y, x))
    elevation = np.degrees(np.arcsin(np.clip(z / point_range, -1.0, 1.0)))

    visible = (
        (azimuth >= -az_fov) & (azimuth <= az_fov)
        & (elevation >= -el_fov) & (elevation <= el_fov)
    )
    if not visible.any():
        return empty

    azimuth = azimuth[visible]
    elevation = elevation[visible]
    point_range = point_range[visible]

    col = np.floor((azimuth + az_fov) / az_res).astype(np.int32)
    row = np.floor((el_fov - elevation) / el_res).astype(np.int32)
    col = np.clip(col, 0, width - 1)
    row = np.clip(row, 0, height - 1)

    dr = r_max / float(n_channels)
    channel = np.minimum((point_range / dr).astype(np.int32), n_channels - 1)

    raw = np.zeros((n_channels, height, width), dtype=np.float32)
    order = np.argsort(point_range)[::-1]
    raw[channel[order], row[order], col[order]] = point_range[order]
    normalized = raw / r_max * 2.0 - 1.0
    return torch.from_numpy(normalized.astype(np.float32))


def radar_range_image_to_point_cloud(
    range_image: Union[torch.Tensor, np.ndarray],
    r_max: float = 14.0,
    az_fov: float = 64.0,
    el_fov: float = 15.0,
    az_res: Optional[float] = None,
    el_res: Optional[float] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    min_range: float = 1e-3,
    is_normalized: bool = True,
) -> np.ndarray:
    """Reconstruct a radar point cloud from the normalized range image above."""
    if torch.is_tensor(range_image):
        ri = range_image.detach().cpu().numpy()
    else:
        ri = np.asarray(range_image, dtype=np.float32)

    if ri.ndim != 3:
        raise ValueError(f"Radar range image should have shape (C,H,W), got {ri.shape}")

    channels, image_height, image_width = ri.shape
    if width is None:
        width = image_width
    if height is None:
        height = image_height
    if image_height != height or image_width != width:
        raise ValueError("Provided radar height/width do not match the input range image shape")

    if az_res is None:
        az_res = (az_fov * 2.0) / float(width)
    if el_res is None:
        el_res = (el_fov * 2.0) / float(height)

    if is_normalized:
        point_range = (ri + 1.0) * 0.5 * r_max
    else:
        point_range = ri.astype(np.float32)

    valid = np.isfinite(point_range) & (point_range > min_range)
    if is_normalized:
        valid &= ri > -0.99
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float32)

    _, row, col = np.nonzero(valid)
    valid_range = point_range[valid]
    azimuth = np.deg2rad(-az_fov + (col.astype(np.float32) + 0.5) * az_res)
    elevation = np.deg2rad(el_fov - (row.astype(np.float32) + 0.5) * el_res)

    xy = valid_range * np.cos(elevation)
    x = xy * np.cos(azimuth)
    y = xy * np.sin(azimuth)
    z = valid_range * np.sin(elevation)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def range_image_to_points(
    range_image,
    range_max: float,
    fov_up: float = 20.0,
    fov_down: float = -35.0,
    min_range: float = 1e-3,
    is_normalized: bool = True,
) -> np.ndarray:
    """Reconstruct a 3D point cloud from a single-channel LiDAR range image."""
    if torch.is_tensor(range_image):
        ri = range_image.detach().cpu().numpy()
    else:
        ri = np.asarray(range_image)

    if ri.ndim == 3:
        if ri.shape[0] != 1:
            raise ValueError("LiDAR range image should have shape (1,H,W) or (H,W)")
        ri = ri[0]
    elif ri.ndim != 2:
        raise ValueError("LiDAR range image should have shape (1,H,W) or (H,W)")

    H, W = ri.shape

    if is_normalized:
        if range_max <= 0:
            raise ValueError("range_max must be > 0 when is_normalized=True")
        r = (ri + 1.0) * 0.5 * range_max
    else:
        r = ri.astype(np.float32)

    rows, cols = np.indices((H, W), dtype=np.float32)
    u = (cols + 0.5) / float(W)
    v = (rows + 0.5) / float(H)

    theta = np.pi * (1.0 - 2.0 * u)
    fov_up_rad = np.deg2rad(fov_up)
    fov_dn_rad = np.deg2rad(fov_down)
    alpha = fov_dn_rad + (1.0 - v) * (fov_up_rad - fov_dn_rad)

    xy = r * np.cos(alpha)
    x = xy * np.cos(theta)
    y = xy * np.sin(theta)
    z = r * np.sin(alpha)

    valid = r > min_range
    return np.stack([x[valid], y[valid], z[valid]], axis=1).astype(np.float32)

def compute_range_image_with_visibility(
    xyz_abs: np.ndarray,
    antenna_pose: np.ndarray, # [4, 4] antenna pose (world to antenna)
    features: Optional[np.ndarray] = None,  # [N, 3] world coords                  
    az_fov: float=90, 
    el_fov: float=90,
    max_range: float = 50,   # Field of view in degrees
    az_res: float = 1, 
    el_res: float = 1,     # Resolution in degrees
    # point_radius: float = 0          # Optional radius in degrees
):
    # points_h = np.hstack((xyz_abs, np.ones((xyz_abs.shape[0], 1))))
    points_h = np.hstack((xyz_abs[:, 1:2],xyz_abs[:, 0:1],xyz_abs[:, 2:3], np.ones((xyz_abs.shape[0], 1))))
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

    return range_image, visibility_mask, feature_image


# def visualize_3d_cube_pyvista(data_cube, cmap='jet', opacity=0.1, threshold=None):
#     """
#     使用PyVista进行GPU加速的3D可视化
    
#     参数:
#         data_cube: 3D numpy数组 (x, y, z)
#         cmap: 使用的颜色图
#         opacity: 透明度
#         threshold: 值阈值
#     """
#     try:
#         # 尝试使用UniformGrid (旧版本)
#         grid = pv.UniformGrid()
#     except AttributeError:
#         # 使用ImageData (新版本)
#         grid = pv.ImageData()
    
#     # 设置网格维度
#     grid.dimensions = np.array(data_cube.shape) + 1
#     grid.origin = (0, 0, 0)  # 设置原点
#     grid.spacing = (1, 1, 1)  # 设置间距
    
#     # 添加数据 (注意Fortran顺序)
#     grid.cell_data["values"] = data_cube.flatten(order="F")
    
#     # 应用阈值
#     if threshold is not None:
#         threshed = grid.threshold([threshold, np.nanmax(data_cube)])
#     else:
#         threshed = grid
    
#     # 创建绘图器
#     plotter = pv.Plotter()
    
#     # 添加体积渲染
#     plotter.add_volume(
#         threshed,
#         cmap=cmap,
#         opacity=opacity,
#         show_scalar_bar=True
#     )
    
#     # 设置背景和标题
#     plotter.set_background('white')
#     plotter.add_text(
#         "GPU加速的3D数据立方体可视化\n(使用PyVista)",
#         position='upper_edge',
#         font_size=18
#     )
    
#     # 显示交互式窗口
#     plotter.show()




# def visualize_3d_cube_gpu(data_cube, cmap='viridis', opacity=0.5, threshold=None):
#     """
#     GPU加速的3D数据立方体可视化
    
#     参数:
#         data_cube: 3D numpy数组 (x, y, z)
#         cmap: 使用的颜色图 (默认'viridis')
#         opacity: 透明度 (0-1)
#         threshold: 只显示高于此阈值的体素
#     """
#     # 将数据转移到GPU (如果可用)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # 转换为PyTorch张量
#     if not isinstance(data_cube, torch.Tensor):
#         data_cube = torch.tensor(data_cube, device=device)
#     else:
#         data_cube = data_cube.to(device)
    
#     # 应用阈值
#     if threshold is not None:
#         mask = data_cube > threshold
#         x_idx, y_idx, z_idx = torch.where(mask)
#         values = data_cube[mask]
#     else:
#         x_idx, y_idx, z_idx = torch.meshgrid(
#             torch.arange(data_cube.shape[0], device=device),
#             torch.arange(data_cube.shape[1], device=device),
#             torch.arange(data_cube.shape[2], device=device),
#             indexing='ij'
#         )
#         values = data_cube.flatten()
#         x_idx, y_idx, z_idx = x_idx.flatten(), y_idx.flatten(), z_idx.flatten()
    
#     # 获取颜色映射
#     norm = Normalize(vmin=values.min().item(), vmax=values.max().item())
#     cmap_func = plt.cm.get_cmap(cmap)
#     colors = cmap_func(norm(values.cpu().numpy())) * 255
    
#     # 创建Plotly图形
#     fig = go.Figure(data=go.Scatter3d(
#         x=x_idx.cpu().numpy(),
#         y=y_idx.cpu().numpy(),
#         z=z_idx.cpu().numpy(),
#         mode='markers',
#         marker=dict(
#             size=3,
#             color=values.cpu().numpy(),
#             colorscale=cmap,
#             opacity=opacity,
#             showscale=True
#         )
#     ))
    
#     fig.update_layout(
#         scene=dict(
#             xaxis_title='X轴',
#             yaxis_title='Y轴',
#             zaxis_title='Z轴'
#         ),
#         title='GPU加速的3D数据立方体可视化 (可旋转缩放)',
#         width=1000,
#         height=800
#     )
    
#     fig.show()
