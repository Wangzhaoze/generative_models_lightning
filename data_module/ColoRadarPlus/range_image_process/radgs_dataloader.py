import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from torch.utils.data import Dataset
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional
from abc import ABC, abstractmethod
import glob
from functools import cached_property
import math
import json
import cv2
import matplotlib.pyplot as plt
import torch
import struct
from typing import Tuple
import open3d as o3d
from tqdm import trange
from data_module.ColoRadarPlus.range_image_process.radgs_utils import interpolate_quaternion_poses, numpy_to_open3d_pointcloud, merge_point_clouds, down_sample_point_cloud, sample_point_cloud, save_point_cloud, visualize_point_cloud
from data_module.ColoRadarPlus.range_image_process.radgs_utils import xyz2aer, compute_range_image_with_visibility, fill_depth_image

try:
    from pyradar.base import FMCW, Sampler, Transceivers, Radar
    from pyradar.rsp.fft import range_fft, doppler_fft
    from pyradar.rsp.doa import compute_spatial_covariance, compute_steering_vector, doa_bartlett, doa_capon, doa_music
    PYRADAR_AVAILABLE = True
except ModuleNotFoundError:
    FMCW = Sampler = Transceivers = None
    range_fft = doppler_fft = None
    compute_spatial_covariance = compute_steering_vector = None
    doa_bartlett = doa_capon = doa_music = None
    PYRADAR_AVAILABLE = False

    class Radar:
        @classmethod
        def load(cls, *_args, **_kwargs):
            raise ModuleNotFoundError(
                "pyradar is required for radar processing in "
                "data_module/ColoRadarPlus/range_image_process/radgs_dataloader.py"
            )

@dataclass
class Record(ABC):
    def __init__(
            self, 
            files: list,
            time_stamps: List[float],
            calib_path: Optional[str],
            drop_first_n_frames: Optional[int] = None,
            drop_last_n_frames: Optional[int] = None,
        ):

        self.files = files
        self.time_stamps = time_stamps
        self.calib_path = calib_path
        if calib_path:
            self.calibration = self.load_calibration(calib_path)
        else:
            self.calibration = np.eye(4, dtype=np.float32)


        if drop_first_n_frames:
            self.files = self.files[drop_first_n_frames:]
            self.time_stamps = self.time_stamps[drop_first_n_frames:]

        if drop_last_n_frames:
            self.files = self.files[:-drop_last_n_frames]
            self.time_stamps = self.time_stamps[:-drop_last_n_frames]


    @abstractmethod
    def __len__(self) -> int:
        return NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> np.ndarray:
        return NotImplementedError

    def _load_data(self, filename: str):
        return None
    
    def load_calibration(self, filename: str) -> np.ndarray:
        """
        Load a 4x4 transformation matrix from a calibration file.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found")

        with open(filename, mode='r') as file:
            lines = file.readlines()

        translation = [float(s) for s in lines[0].split()]
        rot = [float(s) for s in lines[1].split()]

        translation = np.array(translation, dtype=np.float32).reshape(3, 1)
        rot = np.array(rot, dtype=np.float32)

        rotation_matrix = Rotation.from_quat(rot).as_matrix()

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation.flatten()

        return T
    
class EgoTrajectory(Record):
    def __init__(self, files: list,
        time_stamps: List[float],
        calib_path: Optional[str],
        ):
        super().__init__(files, time_stamps, calib_path)

    def __len__(self) -> int:
        return len(self.time_stamps)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.files[idx]
    
class Lidar(Record):
    def __init__(self, files: List[str],
        time_stamps: List[float],
        calib_path: Optional[str],
        ):
        super().__init__(files, time_stamps, calib_path)

    def __len__(self) -> int:
        return len(self.time_stamps)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.load_point_cloud(self.files[idx])

    def load_point_cloud(self, filename: str) -> np.ndarray:
        if not os.path.exists(filename):
            print('File ' + filename + ' not found')
            return None

        with open(filename, mode='rb') as file:
            cloud_bytes = file.read()

        cloud_vals = struct.unpack(str(len(cloud_bytes) // 4)+'f', cloud_bytes)
        pcd_array = np.array(cloud_vals).reshape((-1, 4))

        return pcd_array
    
    def interpolate_poses(self, traj: EgoTrajectory):
        """
        Interpolate poses for each chirp.
        """
        self.raw_poses = interpolate_quaternion_poses(
            src_poses=traj.files,
            src_stamps=traj.time_stamps,
            tgt_stamps=self.time_stamps,
        )

        # self.lidar_poses = self.calibration[np.newaxis, ...] @ interpolated_poses 
        self.lidar_calib_poses = self.raw_poses @ self.calibration[np.newaxis, ...]

        return self.lidar_calib_poses


class RGB(Record):
    def __init__(self, files: list,
        time_stamps: List[float],
        calib_path: Optional[str],
        ):
        super().__init__(files, time_stamps, calib_path)

    def __len__(self) -> int:
        return len(self.time_stamps)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.files[idx]
    
    def _load_data(self, filename):
        rgb = plt.imread(filename)
        return rgb

    def interpolate_poses(self, traj: EgoTrajectory):
        """
        Interpolate poses for each chirp.
        """
        interpolated_poses = interpolate_quaternion_poses(
            src_poses=traj.files,
            src_stamps=traj.time_stamps,
            tgt_stamps=self.time_stamps,
        )

        self.camera_calib_poses = interpolated_poses @ self.calibration[np.newaxis, ...]

        return self.camera_calib_poses


class Depth(Record):
    def __init__(self, files: list,
        time_stamps: List[float],
        calib_path: Optional[str],
        ):
        super().__init__(files, time_stamps, calib_path)

    def _load_data(self, filename):
        depth = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if depth.max() > 50:  # 粗略判断：大于50说明可能是毫米
            depth /= 1000.0  # 转成米
        return depth

    def __len__(self) -> int:
        return len(self.time_stamps)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._load_data(self.files[idx])
    
    def interpolate_poses(self, traj: EgoTrajectory):
        """
        Interpolate poses for each chirp.
        """
        interpolated_poses = interpolate_quaternion_poses(
            src_poses=traj.files,
            src_stamps=traj.time_stamps,
            tgt_stamps=self.time_stamps,
        )

        self.depth_calib_poses = interpolated_poses @ self.calibration[np.newaxis, ...]

        return self.depth_calib_poses

class CascadeRadar(Record, Radar):
    def __init__(self, files: list,
        time_stamps: List[float],
        calib_path: Optional[str],
        radar_config_path: Optional[str],
        drop_first_n_frames: Optional[int] = None,
        drop_last_n_frames: Optional[int] = None,
        ):
        super().__init__(files, time_stamps, calib_path)

        self.calib_path = calib_path

        if radar_config_path:
            radar = Radar.load(radar_config_path)

            for attr in dir(radar):
                if not attr.startswith("__"):
                    value = getattr(radar, attr)
                    if not callable(value) and not isinstance(type(radar).__dict__.get(attr, None), property):
                        setattr(self, attr, value)

        if drop_first_n_frames:
            self.files = self.files[drop_first_n_frames:]
            self.time_stamps = self.time_stamps[drop_first_n_frames:]

        if drop_last_n_frames:
            self.files = self.files[:-drop_last_n_frames]
            self.time_stamps = self.time_stamps[:-drop_last_n_frames]
  
    @cached_property
    def num_frames(self) -> int:
        return len(self.files)
    
    @cached_property
    def frameShape(self) -> tuple:
        return (12, 16, 16, 256, 2)

    def __len__(self) -> int:
        return len(self.time_stamps)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._load_data(self.files[idx])
    
    def _load_data(self, filename) -> np.ndarray:
        adc_samples = np.fromfile(filename, dtype=np.int16).reshape(self.frameShape)
        
        I = np.float32(adc_samples[:, :, :, :, 0])
        Q = np.float32(adc_samples[:, :, :, :, 1])
        adc_samples = I + 1j * Q
        # adc_samples = np.concatenate((I[:, :, :, :, np.newaxis], Q[:, :, :, :, np.newaxis]), axis=-1)

        return adc_samples
    
    @cached_property
    def _phase_freq_calib(self) -> Dict[str, Any]:
        phase_freq_calib_path = os.path.join(self.calib_path, '../../cascade/phase_frequency_calib.txt')
        with open(phase_freq_calib_path, 'r') as f:
            phase_freq_calib: dict = json.load(f)
        return phase_freq_calib
    
    @cached_property
    def _phase_calibration(self) -> np.array:
        """Return the phase calibration array."""
        # Phase calibration matrix
        pm = np.array(self._phase_freq_calib['antennaCalib']['phaseCalibrationMatrix'])
        pm = pm[::2] + 1j * pm[1::2]
        pm = pm[0] / pm
        return pm.reshape(
            self._phase_freq_calib['antennaCalib']['numTx'],
            self._phase_freq_calib['antennaCalib']['numRx'],
            1,
            1
        )
    
    @cached_property
    def _frequency_calibration(self) -> np.array:
        """Return the frequency calibration array."""
        num_tx: int = self._phase_freq_calib['antennaCalib']['numTx']
        num_rx: int = self._phase_freq_calib['antennaCalib']['numRx']

        # Calibration frequency slope
        fcal_slope: float = self._phase_freq_calib['antennaCalib']['frequencySlope']
        # Calibration sampling rate
        cal_srate: int = self._phase_freq_calib['antennaCalib']['samplingRate']

        fcal_matrix = np.array(self._phase_freq_calib['antennaCalib']['frequencyCalibrationMatrix'])

        fslope: float = self.waveform.chirpSlope
        srate: int = self.sampler.adcSampleRate

        # Delta P
        # <frequencyCalibrationMatrix> - <frequencyCalibrationMatrix>[0]
        dp = fcal_matrix - fcal_matrix[0]

        cal_matrix = 2 * np.pi * dp * (fslope / fcal_slope) * (cal_srate / srate)
        cal_matrix /= self.sampler.numSamplesPerChirp
        cal_matrix.reshape(num_tx, num_rx)
        cal_matrix = np.expand_dims(cal_matrix, -1) * np.arange(
            self.sampler.numSamplesPerChirp
        )
        return np.exp(-1j * cal_matrix).reshape(
            num_tx,
            num_rx,
            1,
            self.sampler.numSamplesPerChirp
        )
    
    def _get_velocity_compensation(self, ntx, nc) -> None:
        """Generate the compensation matrix for velocity-induced phase-shift.

        Meant to compensate the velocity-induced phase shift created by TDM MIMO
        configuration.

        Arguments:
            ntx: Number of transmission antenna
            nc: Number of chirps per frame

        Return:
            Phase shift correction matrix
        """
        tl = np.arange(0, ntx)
        cl = np.arange(-nc//2, nc//2)
        tcl = np.kron(tl, cl) / (ntx * nc)
        # Velocity compensation
        vcomp = np.exp(-2j * np.pi * tcl)
        vcomp = vcomp.reshape(ntx, 1, nc, 1)
        return vcomp

    @cached_property
    def _coupling_matrix(self) -> Dict[str, Any]:
        coupling_calib_path = os.path.join(self.calib_path, '../../cascade/coupling_calib.txt')
        coupling_calib = dict()
        with open(coupling_calib_path, "r") as fh:
            for line in fh:
                if line.startswith("# "):
                    continue
                else:
                    name, value = line.split(":")
                    value = value.strip().split(",")
                    if len(value) > 1:
                        coupling_calib[name.strip().lower()] = [float(x) for x in value]
                    else:
                        coupling_calib[name.strip().lower()] = int(value[0])
        
        coupling_matrix = np.array(coupling_calib['data']).reshape(
            coupling_calib['num_tx'],
            coupling_calib['num_rx'],
            1,
            self.sampler.numSamplesPerChirp,
        )
        return coupling_matrix

    @cached_property
    def chirp_timestamps(self) -> np.ndarray:
        total_num_chirps = self.num_frames * self.sampler.numChirpsPerFrame
        chirp_timestamps = np.zeros(total_num_chirps, dtype=np.float64)
        chirp_indices = np.arange(self.sampler.numChirpsPerFrame, dtype=np.int32)
        for idxFrameTimeStamp in range(self.num_frames):
            start = idxFrameTimeStamp * self.sampler.numChirpsPerFrame
            stop = start + self.sampler.numChirpsPerFrame
            chirp_timestamps[start:stop] = self.time_stamps[idxFrameTimeStamp] \
                + chirp_indices * (self.waveform.chirpDuration + self.waveform.chirpIdleTime) \
                + self.waveform.adcStartTime
        return chirp_timestamps
    
    def interpolate_poses(self, traj: EgoTrajectory):
        """
        Interpolate poses for each chirp.
        """
        interpolated_frame_poses = interpolate_quaternion_poses(
            src_poses=traj.files,
            src_stamps=traj.time_stamps,
            tgt_stamps=self.time_stamps,
        )

        interpolated_chirp_poses = interpolate_quaternion_poses(
            src_poses=traj.files,
            src_stamps=traj.time_stamps,
            tgt_stamps=self.chirp_timestamps,
        )

        self.frame_calib_poses = interpolated_frame_poses @ self.calibration[np.newaxis, ...]
        self.chirp_calib_poses = interpolated_chirp_poses @ self.calibration[np.newaxis, ...]

        return self.chirp_calib_poses
    
    def adc2rdmap(self, adc_samples: np.ndarray) -> np.ndarray:
        """
        Convert ADC samples to range-Doppler map.
        """
        # Remove DC bias
        adc_samples -= np.mean(adc_samples)

        adc_samples *= self._frequency_calibration
        adc_samples *= self._phase_calibration

        rfft = range_fft(
            signal=adc_samples,
            IdxSamples=-1,
            window_type='Blackman'
        )


        rfft -= self._coupling_matrix

        dfft = doppler_fft(
            signal=rfft,
            IdxChirps=-2,
            window_type='Blackman'
        )

        snapshot = dfft[0, 0]
        plt.imsave("snapshot_112.png", np.abs(snapshot), cmap='jet')

        # vcomp = self._get_velocity_compensation(self.transceivers.numTX, self.sampler.numChirpsPerFrame)
        # dfft *= vcomp
        
        return dfft
    
    def angle_fft(self, adc_samples: np.ndarray) -> np.ndarray:
        # Remove DC bias
        adc_samples -= np.mean(adc_samples)

        adc_samples *= self._frequency_calibration
        adc_samples *= self._phase_calibration

        rfft = range_fft(
            signal=adc_samples,
            IdxSamples=-1,
            window_type='Blackman'
        )


        rfft -= self._coupling_matrix

        dfft = doppler_fft(
            signal=rfft,
            IdxChirps=-2,
            window_type='Blackman'
        )

        # vcomp = self._get_velocity_compensation(self.transceivers.numTX, self.sampler.numChirpsPerFrame)
        # dfft *= vcomp

        tx_x = self.transceivers.TX[:, 0] - np.min(self.transceivers.TX[:, 0])
        tx_y = self.transceivers.TX[:, 1] - np.min(self.transceivers.TX[:, 1])
        rx_x = self.transceivers.RX[:, 0] - np.min(self.transceivers.RX[:, 0])
        rx_y = self.transceivers.RX[:, 1] - np.min(self.transceivers.RX[:, 1])

        # Shape of the virtual antenna array
        va_shape: tuple = (
            # Length of the elevation axis
            # the "+1" is to count for the 0-indexing used
            np.max(tx_y) + np.max(rx_y) + 1,

            # Length of the azimuth axis
            # the "+1" is to count for the 0-indexing used
            np.max(tx_x) + np.max(rx_x) + 1,

            # Number of chirps per frame
            self.sampler.numChirpsPerFrame,

            # Number of samples per chirp
            self.sampler.numSamplesPerChirp,
        )

        # Virtual antenna array
        virtual_array = np.zeros(va_shape, dtype=np.complex128)

        # *idx: index of the antenna element
        # *az: azimuth of the antenna element
        # *el: elevation of the antenna element
        for tidx,( taz, tel, _) in enumerate(self.transceivers.dTX):
            for ridx,(raz, rel, _) in enumerate(self.transceivers.dRX):
                # When a given azimuth and elevation position is already
                # populated, the new value is added to the previous to have
                # a strong signal feedback
                virtual_array[tel+rel, taz+raz, :, :] += dfft[tidx, ridx, :, :]

        # va_nel: Number of elevations in the virtual array
        # va_naz: Number of azimuth in the virtual array
        # va_nc: Number of chirp per antenna in the virtual array
        # va_ns: Number of samples per chirp
        va_nel, va_naz, va_nc, va_ns = virtual_array.shape

        Ne, Na, Nc, Ns = 32, 128, 16, 256
        virtual_array = np.pad(
            virtual_array,
            (
                (0, Ne - va_nel), (0, Na - va_naz),
                (0, Nc - va_nc), (0, Ns - va_ns)
            ),
            "constant",
            constant_values=((0, 0), (0, 0), (0, 0), (0, 0))
        )

        # Ne: Number of elevations in the virtual array
        # Na: Number of azimuth in the virtual array
        # Nc: Number of chirp per antenna in the virtual array
        # Ns: Number of samples per chirp

        # Azimuth estimation
        afft = np.fft.fft(virtual_array, Na, 1)
        afft = np.fft.fftshift(afft, 1)

        # Elevation esitamtion
        efft = np.fft.fft(afft, Ne, 0)
        efft = np.fft.fftshift(efft, 0)

        # Return the signal power
        return np.abs(efft)
    

    
    def _get_doa_spectrum(self, adc_samples):
        # Remove DC bias
        # adc_samples -= np.mean(adc_samples)

        # adc_samples *= self._frequency_calibration
        # adc_samples *= self._phase_calibration

        # adc_samples = adc_samples[::-1, ::-1]

        adc_samples = adc_samples[2:6, 8:, :, :] # [numTX, numRX, numChirps, numSamples]

        # adc_samples = np.repeat(adc_samples[0, 0, 0].reshape((1, -1)), 16, axis=0)

        # adc_samples = range_fft(
        #     signal=adc_samples,
        #     IdxSamples=1,
        #     window_type='Blackman'
        # )

        # adc_samples = doppler_fft(
        #     signal=adc_samples,
        #     IdxChirps=0,
        #     window_type='Blackman'
        # )

        # adc_samples = range_fft(
        #     signal=adc_samples,
        #     IdxSamples=-1,
        #     window_type='Blackman'
        # )

        # adc_samples = doppler_fft(
        #     signal=adc_samples,
        #     IdxChirps=-2,
        #     window_type='Blackman'
        # )

        # vcomp = self._get_velocity_compensation(self.transceivers.numTX, self.sampler.numChirpsPerFrame)
        # adc_samples *= vcomp

        # rd_map = np.abs(adc_samples)
        # rd_map[:, :20] = 0
        # plt.imshow(rd_map, cmap='jet')
        # plt.show()
        # plt.imsave('output/rd_map_332.png', rd_map, cmap='jet')

        adc_samples = adc_samples[:, :, 0, :]
        # vcomp = self._get_velocity_compensation(self.transceivers.numTX, self.sampler.numChirpsPerFrame)
        # adc_samples *= vcomp

        # adc_samples = np.sum(adc_samples, axis=-2)

        az_grid = np.linspace(np.pi/2, -np.pi/2, 180) # azimuth angles
        el_grid = np.linspace(np.pi/2, -np.pi/2, 180) # elevation angles
        # el_grid = 90 * np.ones(90) * np.pi / 180  # degrees

        steering_vec = compute_steering_vector(
            self.transceivers,
            azimuth=az_grid,
            elevation=el_grid
        )

        # Combine TX and RX to form virtual array (num_antennas, num_samples)
        signal = adc_samples.reshape(
            self.transceivers.numRX * self.transceivers.numTX,
            self.sampler.numSamplesPerChirp
        )

        spectrum, weight = doa_capon(signal, steering_vec)

        spectrum = spectrum.reshape(len(az_grid), len(el_grid))

        return spectrum

    
class ColoRadarPlusDataset(Dataset):
    """
    Dataset class for ColoRadar dataset.
    """
    def __init__(self,
                 data_dir: str = '/home/local/Desktop/code/Datasets/ColoRadar+',
                 rec_id: str = "ec_courtyard_run0",
                 load_radar: bool = True):

        self.data_dir = data_dir
        self.rec_id = rec_id
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        

        self.ego = EgoTrajectory(
            files=np.loadtxt(os.path.join(self.data_dir, self.rec_id, 'groundtruth', 'groundtruth_poses.txt')),
            time_stamps=self.load_time_stamps(os.path.join(self.data_dir, self.rec_id, 'groundtruth', 'timestamps.txt')),
            calib_path=None,
        )

        self.rgb = RGB(
            files=sorted(
                glob.glob(os.path.join(self.data_dir, self.rec_id, 'camera', 'images', 'rgb', '*.png')),
                key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))))),
            time_stamps=self.load_time_stamps(os.path.join(self.data_dir, self.rec_id, 'camera', 'rgb_timestamps.txt')),
            calib_path=os.path.join(self.data_dir, 'calib', 'transforms', 'base_to_camera.txt')
        )
        self.rgb.interpolate_poses(self.ego)

        self.depth = Depth(
            files = sorted(
                glob.glob(os.path.join(self.data_dir, self.rec_id, 'camera', 'images', 'depth', '*.png')),
                key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))))),
            time_stamps=self.load_time_stamps(os.path.join(self.data_dir, self.rec_id, 'camera', 'depth_timestamps.txt')),
            calib_path=os.path.join(self.data_dir, 'calib', 'transforms', 'base_to_camera.txt')
        )
        self.depth.interpolate_poses(self.ego)


        self.lidar = Lidar(
            files = sorted(
                    glob.glob(os.path.join(self.data_dir, self.rec_id, 'lidar', 'pointclouds', '*.bin')), 
                    key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))))),
            time_stamps=self.load_time_stamps(os.path.join(self.data_dir, self.rec_id, 'lidar', 'timestamps.txt')),
            calib_path=os.path.join(self.data_dir, 'calib', 'transforms', 'base_to_lidar.txt'),
            )
        self.lidar.interpolate_poses(self.ego)      

        self.radar = None
        if load_radar:
            self.radar = CascadeRadar(
                files = sorted(
                    glob.glob(os.path.join(self.data_dir, self.rec_id, 'cascade', 'adc_samples', 'data', '*.bin')),
                    key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))))),
                time_stamps=self.load_time_stamps(os.path.join(self.data_dir, self.rec_id, 'cascade', 'adc_samples', 'timestamps.txt')),
                calib_path=os.path.join(self.data_dir, 'calib', 'transforms', 'base_to_radar.txt'),
                radar_config_path = 'pyradar/configs/TI-MMWCAS-RF-EVM.yaml',
                drop_first_n_frames=100,
                drop_last_n_frames=100,
            )

            self.radar.interpolate_poses(self.ego)

    @cached_property
    def transform_z_forward(self) -> np.ndarray:
        theta = np.pi / 2  # -90°
        return np.array([
            [np.cos(theta), 0, np.sin(theta), 0],
            [0,             1, 0,             0],
            [-np.sin(theta),0, np.cos(theta), 0],
            [0,             0, 0,             1]
        ])
    
    @cached_property
    def idx_shape(self) -> tuple:
        return (
            100, 
            # self.radar.radar_config.numChirpsPerFrame, 
            # self.radar.radar_config.numTX,
            # self.radar.radar_config.numRX,
            )

    def __len__(self):
        return math.prod(self.idx_shape)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return None


    def load_time_stamps(self, filename: str) -> np.ndarray:
        """
        Load timestamps from a text file.
        """
        return np.loadtxt(filename, dtype=np.float64)
    



if __name__ == "__main__":
    dataset = ColoRadarPlusDataset(rec_id="ec_courtyard_run0")

    lidar_pcd = dataset.generate_scene_pcd_for_radar()

    # lidar_pcd = dataset.lidar[72]
    pcd = numpy_to_open3d_pointcloud(lidar_pcd[:, :3])

    pcd = pcd.transform(dataset.radar.calibration)
    pcd_array = np.asarray(pcd.points)
    aligned_pcd = np.copy(pcd_array)
    aligned_pcd[:, 0] = -pcd_array[:, 1]
    aligned_pcd[:, 1] = -pcd_array[:, 2]
    aligned_pcd[:, 2] = pcd_array[:, 0]

    np.save('lidar2radar_pcd.npy', aligned_pcd)

    # pcd = numpy_to_open3d_pointcloud(aligned_pcd)

    # xyz_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()

    # visualize_point_cloud([o3d.geometry.TriangleMesh.create_coordinate_frame(), pcd], 'Radar Coordinate Frame')

    for i in range(500):
        i = 200
        adc_samples = dataset.radar[i]
        doa_spectrum = dataset.radar._get_doa_spectrum(adc_samples)
        plt.imsave(f'output/doa_spectrum_{i}.png', doa_spectrum, cmap='jet')

