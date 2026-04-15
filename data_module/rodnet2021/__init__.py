from copy import deepcopy
from torch.utils.data import Dataset, Subset, DataLoader
import lightning as pl
import numpy as np
import torch
from torch.utils.data import Dataset
from data_module.rodnet2021.configs import RadarConfig, ConfMapConfig, BBXMaskConfig
from typing import Dict, List, Literal, Iterable, Tuple, TypedDict, Any
import pandas as pd
from scipy.ndimage import gaussian_filter
from generative_models_lightning import BaseGenerativeModule, Batch

class FrameDataDict(TypedDict):
    timestamp: int
    label_map: np.ndarray
    ra_path: str
    rgb_path: str
    category: List[str]


class BaseDataset(Dataset):
    def __init__(
            self, 
            data_dir: str, 
            rec_key: str,
            spectrum_format: Literal['RI', 'AP'] = 'RI',
            radar_cfg: RadarConfig = RadarConfig(),
            label_map_cfg= ConfMapConfig(),
            ):
        super().__init__()
        self.data_dir = data_dir
        self.rec_key = rec_key
        self.spectrum_format = spectrum_format
        self.radar_cfg = radar_cfg
        self.label_map_cfg = label_map_cfg

        self.data_table = self._generate_data_table()

    def split(self, k_val_per_rec: int = 3) -> Tuple['BaseDataset', 'BaseDataset']:
        """
        Split dataset by recordings — each recording contributes k_val_per_rec frames to validation set.

        Args:
            k_val_per_rec (int): Number of frames per recording to assign to validation set.

        Returns:
            Tuple[BaseDataset, BaseDataset]: train_dataset, val_dataset
        """
        # 确保 data_table 中包含 recording 标识列
        assert "rec" in self.data_table.columns, \
            "data_table 必须包含 'rec' 列来标识 recording"

        val_indices = []
        train_indices = []

        # 按 recording 分组
        for rec_name, group in self.data_table.groupby("rec"):
            all_indices = group.index.tolist()
            np.random.shuffle(all_indices)

            # 每个 recording 取 k 个 frame 做验证（若帧数不足，则全取）
            k = min(k_val_per_rec, len(all_indices))
            val_indices.extend(all_indices[:k])
            train_indices.extend(all_indices[k:])

        # 切分 DataFrame
        train_data_table = self.data_table.loc[train_indices].reset_index(drop=True)
        val_data_table = self.data_table.loc[val_indices].reset_index(drop=True)

        # ✅ 深拷贝对象（保持其他参数一致）
        train_dataset = deepcopy(self)
        val_dataset = deepcopy(self)

        # 替换各自的数据表
        train_dataset.data_table = train_data_table
        val_dataset.data_table = val_data_table

        return train_dataset, val_dataset
        
    def __len__(self):
        return len(self.data_table)
    
    def __getitem__(self, index) -> Batch:
        frame_data_dict = self.get_frame_data(index)
        spectrum = frame_data_dict['spectrum']
        label_map = frame_data_dict['label_map']

        return Batch(x=spectrum, y_embed=torch.Tensor(), cond=label_map)

    def _generate_data_table(self) -> pd.DataFrame:
        return pd.DataFrame()  # Placeholder, to be implemented in subclass
    
    def get_frame_data(self, index) -> dict:
        return dict()
    
    def _get_label_map(self, frame_data: dict):
        if isinstance(self.label_map_cfg, BBXMaskConfig):
            return self._generate_bbx_mask(frame_data)
        elif isinstance(self.label_map_cfg, ConfMapConfig):
            return self._generate_confmap(frame_data)
        else:
            raise ValueError("Unsupported label map configuration.")

    def _generate_bbx_mask(self, frame_label: dict) -> np.ndarray:
        """
        Generate physically consistent bounding-box masks for radar range-angle domain.
        - Range extent: fixed (based on config)
        - Azimuth extent: decreases with distance (angle width ≈ arctan(object_width / range))

        Args:
            frame_label: dict with keys ['range', 'azimuth', 'category']

        Returns:
            bbx_mask: np.ndarray of shape [n_class, numRangeBins, numAngleBins]
        """
        assert isinstance(self.label_map_cfg, BBXMaskConfig), \
            "label_map_cfg must be BBXMaskConfig when using _generate_bbx_mask"

        bbx_mask = np.zeros(
            (self.label_map_cfg.n_class,
            self.radar_cfg.numRangeBins,
            self.radar_cfg.numAngleBins),
            dtype=float
        )

        # 坐标网格
        range_grid = np.linspace(self.radar_cfg.minRange, self.radar_cfg.maxRange, self.radar_cfg.numRangeBins)
        angle_grid = np.linspace(-self.radar_cfg.maxAngle, self.radar_cfg.maxAngle, self.radar_cfg.numAngleBins)
        angle_res = angle_grid[1] - angle_grid[0]
        range_res = range_grid[1] - range_grid[0]

        ranges = frame_label['range']
        azimuths = frame_label['azimuth']
        classes = frame_label['category']

        for range_val, azimuth_val, class_name in zip(ranges, azimuths, classes):
            if class_name not in self.label_map_cfg.classes:
                print(f"⚠️ Warning: Unrecognized class: {class_name}")
                continue

            class_id = self.label_map_cfg.classes.index(class_name)
            range_size, phys_width = self.label_map_cfg.box_size[class_name]  # [m, m]

            # --- 1️⃣ 中心点定位 ---
            range_idx = (np.abs(range_grid - range_val)).argmin()
            angle_idx = (np.abs(angle_grid - np.degrees(azimuth_val))).argmin()

            # --- 2️⃣ Range方向固定像素大小 ---
            range_extent = int((range_size / 2) / range_res)

            # --- 3️⃣ Azimuth方向自适应（近大远小） ---
            # 对于给定物理宽度 W 和距离 R，角宽 α ≈ arctan(W / R)
            angle_span = np.degrees(np.arctan(phys_width / range_val))  # 单边角度
            angle_extent = int((angle_span / angle_res))

            # --- 4️⃣ 计算边界 ---
            r1 = max(0, range_idx - range_extent)
            r2 = min(self.radar_cfg.numRangeBins, range_idx + range_extent)
            a1 = max(0, angle_idx - angle_extent)
            a2 = min(self.radar_cfg.numAngleBins, angle_idx + angle_extent)

            # --- 5️⃣ 填充矩形区域 ---
            bbx_mask[class_id, r1:r2, a1:a2] = 1.0

            # --- 6️⃣ 可选soft边缘 ---
            if self.label_map_cfg.soft_edge_sigma > 0:
                bbx_mask[class_id] = gaussian_filter(bbx_mask[class_id], sigma=self.label_map_cfg.soft_edge_sigma)

        return bbx_mask
    
    def _generate_confmap(self, frame_label: dict) -> np.ndarray:
        """
        Generate confidence map for a single frame.
        
        :param frame_label: DataFrame row containing frame label information with:
            - 'range': list of range values
            - 'azimuth': list of azimuth values  
            - 'category': list of class names
        :param radar_cfg: Radar configuration
        :param confmap_cfg: Confidence map configuration
        :return: Confidence map with shape [n_class, numRangeBins, numAngleBins]
        """
        # Initialize confidence map
        confmap = np.zeros((self.label_map_cfg.n_class, self.radar_cfg.numRangeBins, self.radar_cfg.numAngleBins), dtype=float)

        # Create range and angle grids
        range_grid = np.linspace(self.radar_cfg.minRange, self.radar_cfg.maxRange, self.radar_cfg.numRangeBins)
        angle_grid = np.linspace(-self.radar_cfg.maxAngle, self.radar_cfg.maxAngle, self.radar_cfg.numAngleBins)

        # Process each object in the frame
        ranges = frame_label['range']
        azimuths = frame_label['azimuth']
        classes = frame_label['category']
        
        for range_val, azimuth_val, class_name in zip(ranges, azimuths, classes):
            if class_name not in self.label_map_cfg.classes:
                print(f"Warning: Unrecognized class: {class_name}")
                continue
            class_id = self.label_map_cfg.classes.index(class_name)
            # Convert physical coordinates to pixel indices
            # range_idx = int(np.clip(range_val / self.radar_cfg.maxRange * self.radar_cfg.numRangeBins, 0, self.radar_cfg.numRangeBins - 1))
            # angle_idx = int(np.clip((azimuth_val + self.radar_cfg.maxAngle) / (2 * self.radar_cfg.maxAngle) * self.radar_cfg.numAngleBins, 
            #                     0, self.radar_cfg.numAngleBins - 1))
            range_idx = (np.abs(range_grid - range_val)).argmin()
            angle_idx = (np.abs(angle_grid - azimuth_val * 180 / np.pi)).argmin()

            # Calculate adaptive sigma value based on range
            sigma = 2 * np.arctan(self.label_map_cfg.confmap_length[class_name] / (2 * range_val)) * self.label_map_cfg.confmap_sigmas[class_name]

            # Apply sigma range limits
            sigma_min, sigma_max = self.label_map_cfg.confmap_sigmas_interval[class_name]
            sigma = np.clip(sigma, sigma_min, sigma_max)
            
            # Generate Gaussian distribution
            for i in range(self.radar_cfg.numRangeBins):
                for j in range(self.radar_cfg.numAngleBins):
                    # Calculate distance (range dimension multiplied by 2 to balance scales)
                    distance = (((range_idx - i) * 2) ** 2 + (angle_idx - j) ** 2) / sigma ** 2

                    if distance < self.label_map_cfg.gaussian_thres:
                        value = np.exp(-distance / 2) / (2 * np.pi)
                        # Keep maximum value
                        if value > confmap[class_id, i, j]:
                            confmap[class_id, i, j] = value 
        return confmap
    
    # def _generate_confmap(
    #     self,
    #     frame_label: dict,
    #     occlusion: bool = True,
    #     antenna_gain: bool = True,
    # ) -> np.ndarray:
    #     """
    #     Generate confidence map for a single radar frame with optional occlusion and antenna-gain modeling.

    #     Args:
    #         frame_label (dict): A dict containing
    #             - 'range': list of object ranges (m)
    #             - 'azimuth': list of object azimuths (rad)
    #             - 'category': list of class names
    #         occlusion (bool): Whether to apply occlusion-aware attenuation.
    #         antenna_gain (bool): Whether to apply antenna gain weighting.

    #     Returns:
    #         np.ndarray: Confidence map with shape [n_class, numRangeBins, numAngleBins]
    #     """


    #     # --- 1. Initialize ConfMap ---
    #     confmap = np.zeros(
    #         (self.label_map_cfg.n_class, self.radar_cfg.numRangeBins, self.radar_cfg.numAngleBins),
    #         dtype=float
    #     )

    #     # --- 2. Range & Angle Grids ---
    #     range_grid = np.linspace(self.radar_cfg.minRange, self.radar_cfg.maxRange, self.radar_cfg.numRangeBins)
    #     angle_grid = np.linspace(-self.radar_cfg.maxAngle, self.radar_cfg.maxAngle, self.radar_cfg.numAngleBins)

    #     # --- 3. Prepare object lists ---
    #     ranges = np.array(frame_label['range'])
    #     azimuths = np.array(frame_label['azimuth'])
    #     classes = frame_label['category']

    #     # Sort objects by range (nearest first)
    #     sort_idx = np.argsort(ranges)
    #     ranges = ranges[sort_idx]
    #     azimuths = azimuths[sort_idx]
    #     classes = [classes[i] for i in sort_idx]

    #     # --- 4. Optional: Define antenna gain pattern ---
    #     # TI mmWave antennas have typically a cosine-shaped gain pattern in azimuth
    #     if antenna_gain:
    #         # Convert azimuth (deg) to index and compute gain
    #         # Example: G(θ) = cos^n(θ), with n controlling main lobe width
    #         n_gain = 6
    #         antenna_pattern = np.cos(np.deg2rad(angle_grid)) ** n_gain
    #         antenna_pattern = np.clip(antenna_pattern, 0.0, 1.0)
    #     else:
    #         antenna_pattern = np.ones_like(angle_grid)

    #     # --- 5. Process each object ---
    #     for i, (range_val, azimuth_val, class_name) in enumerate(zip(ranges, azimuths, classes)):
    #         if class_name not in self.label_map_cfg.classes:
    #             print(f"Warning: Unrecognized class: {class_name}")
    #             continue
    #         class_id = self.label_map_cfg.classes.index(class_name)

    #         # Find corresponding grid indices
    #         range_idx = (np.abs(range_grid - range_val)).argmin()
    #         angle_idx = (np.abs(angle_grid - np.degrees(azimuth_val))).argmin()

    #         # --- 6. Base Gaussian sigma (spread) ---
    #         sigma = 2 * np.arctan(self.label_map_cfg.confmap_length[class_name] / (2 * range_val)) \
    #                 * self.label_map_cfg.confmap_sigmas[class_name]
    #         sigma_min, sigma_max = self.label_map_cfg.confmap_sigmas_interval[class_name]
    #         sigma = np.clip(sigma, sigma_min, sigma_max)

    #         # --- 7. Compute base peak intensity (without attenuation) ---
    #         # Use simplified radar equation: received power ∝ 1 / range^4
    #         base_peak = 1.0 / (range_val ** 4 + 1e-6)

    #         # --- 8. Apply occlusion attenuation ---
    #         if occlusion:
    #             # If this target is behind another closer object along similar azimuth
    #             for j in range(i):  # check closer targets only
    #                 delta_angle = np.abs(azimuths[j] - azimuth_val)
    #                 if delta_angle < np.deg2rad(3.0):  # occlusion angular threshold
    #                     # Attenuation based on occluder's type (e.g., car > pedestrian)
    #                     occ_class = classes[j]
    #                     occ_factor = self.label_map_cfg.occlusion_factor.get(occ_class, 0.6)
    #                     base_peak *= occ_factor
    #                     break  # Only apply once for the nearest occluder

    #         # --- 9. Apply antenna gain attenuation ---
    #         gain = antenna_pattern[angle_idx] if antenna_gain else 1.0
    #         peak_value = base_peak * gain

    #         # --- 10. Fill ConfMap with Gaussian distribution ---
    #         for r_i in range(self.radar_cfg.numRangeBins):
    #             for a_i in range(self.radar_cfg.numAngleBins):
    #                 distance = (((range_idx - r_i) * 2) ** 2 + (angle_idx - a_i) ** 2) / (sigma ** 2)
    #                 if distance < self.label_map_cfg.gaussian_thres:
    #                     value = peak_value * np.exp(-distance / 2) / (2 * np.pi)
    #                     if value > confmap[class_id, r_i, a_i]:
    #                         confmap[class_id, r_i, a_i] = value

    #     return confmap



class BaseDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            dataset: BaseDataset,
            batch_size: int,
            num_workers: int,
            prepare_data_flag: bool,
            ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prepare_data_flag = prepare_data_flag

    def prepare_data(self):
        if self.prepare_data_flag:
            # Download, tokenize, etc.
            pass
        else:
            pass

    def setup(self, stage=None):
        self.train_dataset, self.val_dataset = self.dataset.split()
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            persistent_workers=True
        )
    