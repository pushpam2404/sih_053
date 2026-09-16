"""DRDO ID26053 — Semantic Segmentation Subpackage
Provides 3D sparse convolutional network definitions (MinkUNet18),
dataset loaders for RELLIS-3D, and point cloud augmentation utilities.
"""

from .remap import remap_rellis, RELLIS_TO_DRDO
from .dataset import RELLISDataset
from .model import MinkUNet18Scaffold, build_minkunet
from .augmentation import augment_pointcloud

__all__ = [
    "remap_rellis",
    "RELLIS_TO_DRDO",
    "RELLISDataset",
    "MinkUNet18Scaffold",
    "build_minkunet",
    "augment_pointcloud",
]
