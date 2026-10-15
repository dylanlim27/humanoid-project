"""
Unit tests for 3D change detection and proximity classification.
"""

import numpy as np
import pytest

from chronoscene.geometry.transforms import CameraIntrinsics, CameraPose
from chronoscene.geometry.pointcloud import PointCloud
from chronoscene.reasoning.change_detector import ChangeDetector3D
from chronoscene.reasoning.occupancy_grid import VoxelState


def test_static_scene_change_detector():
    intrinsics = CameraIntrinsics.from_fov(fov_deg=60.0, width=100, height=100)
    pose = CameraPose(T_world_camera=np.eye(4))

    # Identical point clouds in A and B (static scene)
    pts = np.random.uniform(-0.5, 0.5, (100, 3))
    pts[:, 2] += 2.0  # 2m in front of camera
    cols = np.ones((100, 3)) * 0.5

    pcd_a = PointCloud(points=pts, colors=cols, scan_id="A")
    pcd_b = PointCloud(points=pts.copy(), colors=cols.copy(), scan_id="B")

    depth_map = np.full((100, 100), 2.0, dtype=np.float32)

    detector = ChangeDetector3D(static_distance_threshold=0.05)
    result = detector.detect_changes(
        pcd_a=pcd_a,
        pcd_b=pcd_b,
        poses_a=[pose],
        depth_maps_a=[depth_map],
        poses_b=[pose],
        depth_maps_b=[depth_map],
        intrinsics=intrinsics
    )

    # 100% of points should be static
    assert result.metrics["static_percentage"] == 100.0
    assert result.metrics["removed_points"] == 0
    assert result.metrics["added_points"] == 0
