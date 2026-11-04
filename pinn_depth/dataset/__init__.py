"""
Datasets, procedural scene generators, and synthetic haze augmentations.
"""

from .synthetic_haze import SyntheticHazeAugmentation
from .sample_generator import generate_synthetic_outdoor_scene, generate_benchmark_dataset
from .kitti_dataset import OutdoorDepthDataset

__all__ = [
    "SyntheticHazeAugmentation",
    "generate_synthetic_outdoor_scene",
    "generate_benchmark_dataset",
    "OutdoorDepthDataset",
]

