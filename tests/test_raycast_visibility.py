"""
Unit tests for line-of-sight raycasting, multi-view visibility, and occlusion checking.
"""

import numpy as np
import pytest

from chronoscene.geometry.transforms import CameraIntrinsics, CameraPose
from chronoscene.reasoning.occupancy_grid import (
    MultiViewRaycastEngine,
    ObservationStatus,
    VoxelState
)


def test_raycast_single_view_free_space_and_surface():
    intrinsics = CameraIntrinsics.from_fov(fov_deg=60.0, width=100, height=100)
    pose = CameraPose(T_world_camera=np.eye(4))
    engine = MultiViewRaycastEngine(surface_thickness_epsilon=0.04)

    # Flat depth map at 2.0 meters across the entire sensor
    depth_map = np.full((100, 100), 2.0, dtype=np.float32)

    query_points = np.array([
        [0.0, 0.0, 1.0],   # In front of depth map at 1.0m -> FREE_SPACE
        [0.0, 0.0, 2.0],   # Exactly on depth surface at 2.0m -> OCCUPIED
        [0.0, 0.0, 3.0],   # Behind depth surface at 3.0m -> OCCLUDED
        [5.0, 0.0, 1.0],   # Far outside image frustum -> OUT_OF_FRUSTUM
    ])

    statuses = engine.test_points_single_view(query_points, pose, intrinsics, depth_map)

    assert statuses[0] == ObservationStatus.FREE_SPACE
    assert statuses[1] == ObservationStatus.OCCUPIED
    assert statuses[2] == ObservationStatus.OCCLUDED
    assert statuses[3] == ObservationStatus.OUT_OF_FRUSTUM


def test_multi_view_aggregation():
    intrinsics = CameraIntrinsics.from_fov(fov_deg=60.0, width=100, height=100)
    engine = MultiViewRaycastEngine(surface_thickness_epsilon=0.04)

    pose1 = CameraPose(T_world_camera=np.eye(4))
    depth1 = np.full((100, 100), 2.0, dtype=np.float32)

    poses = [pose1]
    depths = [depth1]

    # Query point at 1.0m (free space)
    pts = np.array([[0.0, 0.0, 1.0]])
    agg = engine.aggregate_scan_visibility(pts, poses, intrinsics, depths)

    assert agg["num_free"][0] == 1
    assert agg["is_verified_empty"][0] == True
    assert agg["is_observed_surface"][0] == False
    assert agg["is_unobserved_or_occluded"][0] == False
