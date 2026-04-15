
from typing import List, Dict, Literal, Optional, Tuple
import pandas as pd
import numpy as np
import glob 
import os

from data_module.rodnet2021 import BaseDataset
from data_module.rodnet2021.configs import RadarConfig, ConfMapConfig, BBXMaskConfig, RODNet2021_RECORDINGS, CRUW_SCENARIOS



class RODNet2021Dataset(BaseDataset):
    def __init__(
            self, 
            data_dir: str = 'D:/Datasets/RODNet', 
            rec_key: Literal['All', 'debug'] = 'debug',
            radar_cfg: RadarConfig = RadarConfig(),
            label_map_cfg: ConfMapConfig = ConfMapConfig(),
            ):
        
        self.recordings = RODNet2021_RECORDINGS.get(rec_key, [])
        super().__init__(data_dir=data_dir, rec_key=rec_key, radar_cfg=radar_cfg, label_map_cfg=label_map_cfg)

    def _generate_data_table(self) -> pd.DataFrame:

        data_table_list = []

        for rec in self.recordings:
            # Collect ADC (radar) and RGB (image) file paths
            adc_paths = glob.glob(os.path.join(self.data_dir, 'sequences', 'train', rec, 'RADAR_RA_H', "*_0000.npy"))
            rgb_paths = glob.glob(os.path.join(self.data_dir, 'sequences', 'train', rec, 'IMAGES_0', "*.jpg"))
            
            # Load annotation table
            rec_data_table = pd.read_csv(
                os.path.join(self.data_dir, 'annotations', 'train', f'{rec}.txt'),
                sep=r"\s+",   # split on whitespace
                header=None,  # no header in the file
                names=["timestamp", "range", "azimuth", 'category']
            )
            
            # Group rows by timestamp; aggregate multiple objects into lists
            rec_data_table = rec_data_table.groupby('timestamp').agg({
                'range': list,
                'azimuth': list,
                'category': list
            }).reset_index()
            
            # Handle mismatch in number of timestamps vs. file paths
            if len(rec_data_table) != len(adc_paths) or len(rec_data_table) != len(rgb_paths):
                # Extract timestamps from file paths
                adc_timestamps = [int(os.path.basename(p).split('_')[0]) for p in adc_paths]
                rgb_timestamps = [int(os.path.basename(p).split('.')[0]) for p in rgb_paths]

                # Find common timestamps across annotations, ADC, and RGB
                common_adc_ts = set(rec_data_table['timestamp']) & set(adc_timestamps)
                common_rgb_ts = set(rec_data_table['timestamp']) & set(rgb_timestamps)
                common_ts = sorted(common_adc_ts & common_rgb_ts)
                
                # Filter tables and paths to only keep common timestamps
                rec_data_table = rec_data_table[rec_data_table['timestamp'].isin(common_ts)].copy()
                adc_dict = {ts: p for ts, p in zip(adc_timestamps, adc_paths)}
                rgb_dict = {ts: p for ts, p in zip(rgb_timestamps, rgb_paths)}
                
                # Map file paths back into the table
                rec_data_table['ra_path'] = rec_data_table['timestamp'].map(adc_dict)
                rec_data_table['rgb_path'] = rec_data_table['timestamp'].map(rgb_dict)
            else:
                rec_data_table['ra_path'] = adc_paths
                rec_data_table['rgb_path'] = rgb_paths

            rec_data_table['rec'] = rec
            rec_data_table['scenario'] = self._get_rec_scenario(rec)
            data_table_list.append(rec_data_table)

        return pd.concat(data_table_list, ignore_index=True)

    def _get_rec_scenario(self, rec: str) -> Optional[str]:
        """
        Get the scenario type for a given recording ID.

        Args:
            rec (str): Recording ID.

        Returns:
            str: Scenario name if found, else None.
        """
        for scenario in CRUW_SCENARIOS:
            recordings = RODNet2021_RECORDINGS.get(scenario, [])
            if rec in recordings:
                return scenario
    
    def get_frame_data(self, index: int) -> dict:

        frame_data: pd.Series = self.data_table.iloc[index]
        if pd.isna(frame_data['ra_path']) or pd.isna(frame_data['rgb_path']):
            raise ValueError(f"Missing file paths for index {index}: ra_path={frame_data['ra_path']}, rgb_path={frame_data['rgb_path']}")
        else:
            frame_data_dict = frame_data.to_dict()
            ra_map = self._load_ra_maps(frame_data_dict)  # Shape: [4*2, numRangeBins, numAngleBins]
            label = self._get_label_map(frame_data_dict)  # Shape: [n_class, numRangeBins, numAngleBins]

            frame_data_dict.update(
                {
                    'spectrum': ra_map,  # (8, 128, 128)
                    'label_map': label,  # (num_classes, 128, 128)
                    }
                )
        
        return frame_data_dict
    
    def _load_ra_maps(self, frame_data: dict) -> np.ndarray:
        ra_path_0: str = frame_data['ra_path']
        ra_path_64 = ra_path_0.replace('_0000.npy', '_0064.npy')
        ra_path_128 = ra_path_0.replace('_0000.npy', '_0128.npy')
        ra_path_192 = ra_path_0.replace('_0000.npy', '_0192.npy')
        ra_map_0: np.ndarray = np.load(ra_path_0)  # Shape: [numRangeBins, numAngleBins, 2]
        ra_map_64: np.ndarray = np.load(ra_path_64)  # Shape: [numRangeBins, numAngleBins, 2]
        ra_map_128: np.ndarray = np.load(ra_path_128)  # Shape: [numRangeBins, numAngleBins, 2]
        ra_map_192: np.ndarray = np.load(ra_path_192)  # Shape: [numRangeBins, numAngleBins, 2]

        ra_map = np.concatenate([ra_map_0, ra_map_64, ra_map_128, ra_map_192], axis=2)  # Shape: [numRangeBins, numAngleBins, 8]
        ra_map = ra_map.transpose((2, 0, 1))  # Shape: [8, numRangeBins, numAngleBins]
        ra_map = ra_map.astype(np.float32)
        return ra_map
    