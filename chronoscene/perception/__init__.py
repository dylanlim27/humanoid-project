"""
Perception module: monocular depth estimation, cross-video feature matching, and 3D segmentation.
"""

from .depth_estimator import DepthEstimator, MockDepthEstimator
from .feature_matcher import CrossScanAligner, BackgroundSE3Registrar
from .segmentation import Cluster3D, ObjectInstance3D, EuclideanClusterer3D

__all__ = [
    "DepthEstimator",
    "MockDepthEstimator",
    "CrossScanAligner",
    "BackgroundSE3Registrar",
    "Cluster3D",
    "ObjectInstance3D",
    "EuclideanClusterer3D"
]
