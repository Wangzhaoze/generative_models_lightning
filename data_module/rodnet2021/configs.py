#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-10
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : /data_module/ronet2021/configs.py
# @IDE     : vscode



"""
Describe the purpose of this module.
"""


from typing import Dict, List, Tuple, Any, Literal, Optional
from dataclasses import dataclass, field

@dataclass
class ConfMapConfig:
    classes: List[str] = field(default_factory=lambda: ['pedestrian', 'cyclist', 'car'])
    confmap_sigmas: Dict[str, float] = field(default_factory=lambda: {
        'pedestrian': 15, 
        'cyclist': 20, 
        'car': 30, 
        'van': 40, 
        'truck': 50
    })
    confmap_sigmas_interval: Dict[str, List[float]] = field(default_factory=lambda: {
        'pedestrian': [5, 15],
        'cyclist': [8, 20],
        'car': [10, 30],
        'van': [15, 40],
        'truck': [20, 50],
    })
    confmap_length: Dict[str, int] = field(default_factory=lambda: {
        'pedestrian': 1, 
        'cyclist': 2, 
        'car': 3, 
        'van': 4, 
        'truck': 5
    })
    gaussian_thres: float = 36

    occlusion_factor: dict[str, float] = field(default_factory=lambda: {
            'car': 0.5,
            'pedestrian': 0.8,
            'cyclist': 0.7,
        })

    @property
    def n_class(self) -> int:
        return len(self.classes)

@dataclass
class BBXMaskConfig:
    """
    Configuration for generating bounding-box masks in radar range-angle space.
    box_size: [range_length (m), physical_width (m)]
    """
    classes: List[str] = field(default_factory=lambda: ['pedestrian', 'cyclist', 'car'])
    box_size: Dict[str, List[float]] = field(default_factory=lambda: {
        'pedestrian': [1.0, 0.5],  # range_depth=1m, width=0.5m
        'cyclist': [1.5, 0.8],
        'car': [3.0, 1.8],
        'van': [4.0, 2.2],
        'truck': [5.0, 2.5],
    })
    soft_edge_sigma: float = 0.0  # optional blur for smoothness

    occlusion_factor: dict[str, float] = field(default_factory=lambda: {
            'car': 0.5,
            'pedestrian': 0.8,
            'cyclist': 0.7,
        })

    @property
    def n_class(self) -> int:
        return len(self.classes)

from typing import Dict, List, Tuple, Any, Literal, Optional
from dataclasses import dataclass, field

LABELMAP: Dict[int, str] = {
    0: 'person',
    2: 'car', 
    3: 'motorbike',
    5: 'bus',
    7: 'truck',
    80: 'cyclist'
}

CRUW_SCENARIOS: List[str] = ["Parking_Lot", "Campus_Road", "City_Street", "Highway"]

RODNet2021_RECORDINGS: Dict[str, List[str]] = {
    "debug":[
        '2019_04_30_MLMS001'
        ],
    "2019_04_09": [
        '2019_04_09_BMS1000',
        '2019_04_09_BMS1001',
        '2019_04_09_BMS1002',
        '2019_04_09_CMS1002',
        '2019_04_09_PMS1000',
        '2019_04_09_PMS1001',
        '2019_04_09_PMS2000',
        '2019_04_09_PMS3000'
    ],
    "2019_04_30": [
        '2019_04_30_MLMS000',
        '2019_04_30_MLMS001',
        '2019_04_30_MLMS002',
        '2019_04_30_PBMS002',
        '2019_04_30_PBMS003',
        '2019_04_30_PCMS001',
        '2019_04_30_PM2S003',
        '2019_04_30_PM2S004'
    ],
    "2019_05_09": [
        '2019_05_09_BM1S008',
        '2019_05_09_CM1S004',
        '2019_05_09_MLMS003',
        '2019_05_09_PBMS004',
        '2019_05_09_PCMS002'
    ],
    "2019_05_23": [
        '2019_05_23_PM1S012',
        '2019_05_23_PM1S013',
        '2019_05_23_PM1S014',
        '2019_05_23_PM1S015',
        '2019_05_23_PM2S011'
    ],
    "2019_05_29": [
        '2019_05_29_BCMS000',
        '2019_05_29_BM1S016',
        '2019_05_29_BM1S017',
        # '2019_05_29_MLMS006',
        # '2019_05_29_PBMS007',
        '2019_05_29_PCMS005',
        '2019_05_29_PM2S015',
        '2019_05_29_PM3S000'
    ],
    "2019_09_29": [
        # '2019_09_29_ONRD001',
        '2019_09_29_ONRD002',
        '2019_09_29_ONRD005',
        # '2019_09_29_ONRD006',
        '2019_09_29_ONRD011',
        '2019_09_29_ONRD013'
    ],
    "mix": [
        '2019_09_29_ONRD002',  
        '2019_09_29_ONRD011',
        '2019_09_29_ONRD013',
        '2019_04_09_BMS1002',
        '2019_04_30_MLMS001',
        '2019_04_30_MLMS002',
        '2019_05_09_MLMS003',
        '2019_04_30_PBMS003',
        '2019_05_09_PBMS004',
        '2019_05_09_PCMS002',
        '2019_05_29_BCMS000',
        '2019_09_29_ONRD005',
    ],
    "Parking_Lot": [
        '2019_04_09_BMS1000',
        '2019_04_09_BMS1001',
        '2019_04_09_BMS1002',
        '2019_04_09_CMS1002',
        '2019_04_09_PMS1000',
        '2019_04_09_PMS1001',
        '2019_04_09_PMS2000',
        '2019_04_09_PMS3000',
        '2019_04_30_PBMS002',
        '2019_04_30_PBMS003',
        '2019_04_30_PM2S003',
        '2019_04_30_PM2S004',
        '2019_05_23_PM1S012',
        '2019_05_23_PM1S013',
        '2019_05_23_PM1S014',
        '2019_05_23_PM1S015',
        '2019_05_23_PM2S011',
        '2019_05_29_BM1S016',
        '2019_05_29_BM1S017',
        # '2019_05_29_PBMS007',
        '2019_05_29_PM2S015',
        '2019_05_29_PM3S000',
    ],
    "Campus_Road": [
        '2019_04_30_MLMS000',
        '2019_04_30_MLMS001',
        '2019_04_30_MLMS002',
        '2019_04_30_PCMS001',
        '2019_05_09_BM1S008',
        '2019_05_09_CM1S004',
        '2019_05_09_MLMS003',
        '2019_05_09_PBMS004',
        '2019_05_09_PCMS002',
        '2019_05_29_BCMS000',
        # '2019_05_29_MLMS006',
        '2019_05_29_PCMS005',
    ],
    "City_Street": [
        # '2019_09_29_ONRD001',
        '2019_09_29_ONRD002',    
    ],
    "Highway": [
        '2019_09_29_ONRD005',
        # '2019_09_29_ONRD006',
        '2019_09_29_ONRD011',
        '2019_09_29_ONRD013'
    ],
    "All":[
        '2019_04_09_BMS1000',
        '2019_04_09_BMS1001',
        '2019_04_09_BMS1002',
        '2019_04_09_CMS1002',
        '2019_04_09_PMS1000',
        '2019_04_09_PMS1001',
        '2019_04_09_PMS2000',
        '2019_04_09_PMS3000',
        '2019_04_30_PBMS002',
        '2019_04_30_PBMS003',
        '2019_04_30_PM2S003',
        '2019_04_30_PM2S004',
        '2019_05_23_PM1S012',
        '2019_05_23_PM1S013',
        '2019_05_23_PM1S014',
        '2019_05_23_PM1S015',
        '2019_05_23_PM2S011',
        '2019_05_29_BM1S016',
        '2019_05_29_BM1S017',
        # '2019_05_29_PBMS007',
        '2019_05_29_PM2S015',
        '2019_05_29_PM3S000',
        '2019_04_30_MLMS000',
        '2019_04_30_MLMS001',
        '2019_04_30_MLMS002',
        '2019_04_30_PCMS001',
        '2019_05_09_BM1S008',
        '2019_05_09_CM1S004',
        '2019_05_09_MLMS003',
        '2019_05_09_PBMS004',
        '2019_05_09_PCMS002',
        '2019_05_29_BCMS000',
        # '2019_05_29_MLMS006',
        '2019_05_29_PCMS005',
        # '2019_09_29_ONRD001',
        '2019_09_29_ONRD002', 
        '2019_09_29_ONRD005',
        # '2019_09_29_ONRD006',
        '2019_09_29_ONRD011',
        '2019_09_29_ONRD013'
    ]
}

from dataclasses import dataclass, field
import numpy as np


@dataclass
class RadarConfig:
    numRangeBins: int = 128
    numDopplerBins: int = 128
    numAngleBins: int = 128

    minRange: float = 0.63916459
    maxRange: float = 27.69713208
    maxVelocity: float = 5.0
    maxAngle: float = 60.0

    frameRate: float = 30.0
    crop_range_bins: int = 3

    chirpSlope: float = 21.0017e12
    startFrequency: float = 76.999999488e9 # center
    adcStartTime: float = 0.0
    chirpIdleTime: float = 5.0e-4

    numChirpsPerFrame: int = 255
    numSamplesPerChirp: int = 128
    adcSampleRate: int = 400000
    numVirtualAntennas: int = 8


    range_grid = np.linspace(0.63916459, 27.69713208, 128) 
    angle_grid = np.linspace(-90.0, 90.0, 128)
    