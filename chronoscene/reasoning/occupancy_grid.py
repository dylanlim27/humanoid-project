"""
Line-of-sight raycasting, multi-view visibility reasoning, and 3D occupancy grid.
"""

from __future__ import annotations
from enum import Enum, auto
from typing import List, Tuple, Dict, Optional
import numpy as np

from ..geometry.transforms import CameraIntrinsics, CameraPose, project_points_to_camera


class ObservationStatus(Enum):
    """Observation status of a 3D point from a single camera viewpoint."""
    OUT_OF_FRUSTUM = auto()  # Outside the camera's field of view or behind camera
    FREE_SPACE = auto()      # Camera ray passed through this point to hit a background surface
    OCCUPIED = auto()        # Measured surface lies at this point
    OCCLUDED = auto()        # Blocked by a foreground surface closer to the camera


class VoxelState(Enum):
    """Disambiguated 3D spatial state across dual scans."""
    STATIC = "static"                    # Present in both scans in identical location
    MOVED = "moved"                      # Same object moved to a new 3D location
    ADDED = "added"                      # Observed occupied in Scan B, verified free space in Scan A
    REMOVED = "removed"                  # Observed occupied in Scan A, verified free space in Scan B
    OCCLUDED = "occluded"                # In field of view but occluded by foreground obstacles
    OUT_OF_VIEW = "out_of_view"          # Never covered by camera trajectory (incomplete sweep)


class MultiViewRaycastEngine:
    """
    Evaluates multi-view line-of-sight visibility across a sequence of camera frames.
    Distinguishes true scene disappearance/appearance from occlusions and incomplete trajectories.
    """

    def __init__(
        self,
        surface_thickness_epsilon: float = 0.04,  # +/- 4cm surface tolerance
        min_views_free_space: int = 1,            # At least 1 clear free-space ray needed to confirm empty
    ):
        self.surface_epsilon = surface_thickness_epsilon
        self.min_views_free_space = min_views_free_space

    def test_points_single_view(
        self,
        points_world: np.ndarray,
        pose: CameraPose,
        intrinsics: CameraIntrinsics,
        depth_map: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate observation status for an array of 3D points from one viewpoint.

        Args:
            points_world: (N, 3) 3D coordinates in world frame.
            pose: CameraPose of this frame.
            intrinsics: CameraIntrinsics of this frame.
            depth_map: (H, W) float depth map in meters.

        Returns:
            statuses: (N,) array of ObservationStatus enum values.
        """
        N = len(points_world)
        if N == 0:
            return np.array([], dtype=object)

        uv, z_c, valid = project_points_to_camera(points_world, pose, intrinsics)
        H, W = depth_map.shape[:2]

        in_image = (
            valid &
            (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) &
            (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
        )

        statuses = np.full(N, ObservationStatus.OUT_OF_FRUSTUM, dtype=object)
        if not np.any(in_image):
            return statuses

        valid_idx = np.where(in_image)[0]
        u_int = np.clip(np.round(uv[valid_idx, 0]).astype(np.int64), 0, W - 1)
        v_int = np.clip(np.round(uv[valid_idx, 1]).astype(np.int64), 0, H - 1)

        d_meas = depth_map[v_int, u_int]
        z_query = z_c[valid_idx]

        # Valid depth measurements
        has_depth = np.isfinite(d_meas) & (d_meas > 0.1)

        # 1. Free Space: query point is significantly closer to camera than measured surface
        is_free = has_depth & (z_query < (d_meas - self.surface_epsilon))

        # 2. Surface Hit: query point is within epsilon of measured depth
        is_surface = has_depth & (np.abs(z_query - d_meas) <= self.surface_epsilon)

        # 3. Occluded: query point is deeper than measured surface (hidden behind it)
        is_occluded = has_depth & (z_query > (d_meas + self.surface_epsilon))

        statuses[valid_idx[is_free]] = ObservationStatus.FREE_SPACE
        statuses[valid_idx[is_surface]] = ObservationStatus.OCCUPIED
        statuses[valid_idx[is_occluded]] = ObservationStatus.OCCLUDED

        return statuses

    def aggregate_scan_visibility(
        self,
        points_world: np.ndarray,
        poses: List[CameraPose],
        intrinsics: CameraIntrinsics,
        depth_maps: List[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate multi-view observation statistics for points across an entire video scan.

        Returns:
            Dictionary containing:
                'num_free': (N,) int - count of views seeing free space
                'num_occupied': (N,) int - count of views seeing surface hit
                'num_occluded': (N,) int - count of views seeing occlusion
                'num_in_frustum': (N,) int - total views where point was in frustum
                'is_verified_empty': (N,) bool - point is conclusively observed as free space
                'is_observed_surface': (N,) bool - point is observed as physical surface
                'is_unobserved_or_occluded': (N,) bool - point was not adequately observed
        """
        N = len(points_world)
        num_free = np.zeros(N, dtype=np.int32)
        num_occupied = np.zeros(N, dtype=np.int32)
        num_occluded = np.zeros(N, dtype=np.int32)
        num_in_frustum = np.zeros(N, dtype=np.int32)

        for pose, depth_map in zip(poses, depth_maps):
            statuses = self.test_points_single_view(points_world, pose, intrinsics, depth_map)
            num_free += (statuses == ObservationStatus.FREE_SPACE)
            num_occupied += (statuses == ObservationStatus.OCCUPIED)
            num_occluded += (statuses == ObservationStatus.OCCLUDED)
            num_in_frustum += (statuses != ObservationStatus.OUT_OF_FRUSTUM)

        is_observed_surface = num_occupied >= 1
        is_verified_empty = (num_free >= self.min_views_free_space) & (~is_observed_surface)
        is_unobserved_or_occluded = (num_in_frustum == 0) | ((num_occluded > 0) & (num_occupied == 0) & (num_free == 0))

        return {
            "num_free": num_free,
            "num_occupied": num_occupied,
            "num_occluded": num_occluded,
            "num_in_frustum": num_in_frustum,
            "is_verified_empty": is_verified_empty,
            "is_observed_surface": is_observed_surface,
            "is_unobserved_or_occluded": is_unobserved_or_occluded
        }
