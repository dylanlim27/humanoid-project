"""
Camera frustum modeling and 3D frustum visibility checks.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .transforms import CameraIntrinsics, CameraPose, project_points_to_camera


@dataclass
class CameraFrustum:
    """
    Representation of a 3D camera viewing frustum defined by camera pose and intrinsics.
    """
    pose: CameraPose
    intrinsics: CameraIntrinsics
    z_near: float = 0.15
    z_far: float = 4.0

    def get_corners(self) -> np.ndarray:
        """
        Compute the 8 corners of the frustum in world coordinates.
        Returns:
            (8, 3) array: [4 near plane corners, 4 far plane corners]
        """
        K_inv = self.intrinsics.K_inv
        w, h = self.intrinsics.width, self.intrinsics.height

        # 4 corners in 2D pixel homogeneous coordinates: [u, v, 1]
        uv1 = np.array([
            [0, 0, 1.0],
            [w, 0, 1.0],
            [w, h, 1.0],
            [0, h, 1.0]
        ], dtype=np.float64)

        # Rays in camera frame: K_inv * uv1
        rays_cam = (K_inv @ uv1.T).T  # (4, 3)

        near_cam = rays_cam * self.z_near
        far_cam = rays_cam * self.z_far
        corners_cam = np.vstack([near_cam, far_cam])  # (8, 3)

        # Transform to world
        corners_world = (corners_cam @ self.pose.R.T) + self.pose.t
        return corners_world

    def contains_points(self, points_world: np.ndarray) -> np.ndarray:
        """
        Test whether 3D world points fall inside the camera frustum pyramid.

        Args:
            points_world: (N, 3) array of 3D world coordinates.

        Returns:
            (N,) boolean mask: True if point is inside the frustum.
        """
        if len(points_world) == 0:
            return np.zeros((0,), dtype=bool)

        uv, z_c, valid = project_points_to_camera(points_world, self.pose, self.intrinsics)

        in_depth = (z_c >= self.z_near) & (z_c <= self.z_far)
        in_image = (
            (uv[:, 0] >= 0) & (uv[:, 0] < self.intrinsics.width) &
            (uv[:, 1] >= 0) & (uv[:, 1] < self.intrinsics.height)
        )

        return valid & in_depth & in_image


def points_in_frustums(
    points_world: np.ndarray,
    poses: List[CameraPose],
    intrinsics: CameraIntrinsics,
    z_near: float = 0.15,
    z_far: float = 4.0
) -> np.ndarray:
    """
    Test whether each 3D point is inside AT LEAST ONE camera frustum in a trajectory.

    Args:
        points_world: (N, 3) array of 3D points in world coordinates.
        poses: List of CameraPose instances.
        intrinsics: CameraIntrinsics.

    Returns:
        (N,) boolean mask: True if the point is within the field of view of any camera.
    """
    if len(points_world) == 0 or len(poses) == 0:
        return np.zeros((len(points_world),), dtype=bool)

    covered = np.zeros(len(points_world), dtype=bool)
    for pose in poses:
        frustum = CameraFrustum(pose=pose, intrinsics=intrinsics, z_near=z_near, z_far=z_far)
        covered |= frustum.contains_points(points_world)
        if np.all(covered):
            break

    return covered
