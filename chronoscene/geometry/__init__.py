"""
Geometry module for 3D coordinate transformations, camera projections, and point cloud math.
"""

from .transforms import (
    CameraIntrinsics,
    CameraPose,
    se3_transform_points,
    se3_inverse,
    compute_relative_pose,
    project_points_to_camera,
    OrientedBoundingBox3D
)
from .pointcloud import PointCloud, unproject_depth_map, voxel_downsample
from .frustum import CameraFrustum, points_in_frustums

__all__ = [
    "CameraIntrinsics",
    "CameraPose",
    "se3_transform_points",
    "se3_inverse",
    "compute_relative_pose",
    "project_points_to_camera",
    "OrientedBoundingBox3D",
    "PointCloud",
    "unproject_depth_map",
    "voxel_downsample",
    "CameraFrustum",
    "points_in_frustums"
]
