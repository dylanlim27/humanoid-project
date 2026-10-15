"""
SE(3) rigid transformations, camera intrinsics/extrinsics, and 3D projections.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics model."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        """3x3 Camera calibration matrix."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    @property
    def K_inv(self) -> np.ndarray:
        """Inverse calibration matrix."""
        return np.linalg.inv(self.K)

    @classmethod
    def from_fov(cls, fov_deg: float, width: int, height: int) -> CameraIntrinsics:
        """Construct intrinsics from diagonal or horizontal FOV."""
        fov_rad = np.radians(fov_deg)
        fx = (width / 2.0) / np.tan(fov_rad / 2.0)
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


@dataclass
class CameraPose:
    """
    Camera pose representing Camera-to-World transformation T_w_c in SE(3).
    A point in camera frame p_c transforms to world frame p_w = R * p_c + t.
    """
    T_world_camera: np.ndarray  # 4x4 matrix

    def __post_init__(self):
        self.T_world_camera = np.asarray(self.T_world_camera, dtype=np.float64)
        if self.T_world_camera.shape != (4, 4):
            raise ValueError(f"T_world_camera must be 4x4, got {self.T_world_camera.shape}")

    @property
    def R(self) -> np.ndarray:
        """3x3 Rotation matrix (Camera to World)."""
        return self.T_world_camera[:3, :3]

    @property
    def t(self) -> np.ndarray:
        """3D translation vector (Camera optical center in World frame)."""
        return self.T_world_camera[:3, 3]

    @property
    def T_camera_world(self) -> np.ndarray:
        """World-to-Camera transformation matrix."""
        return se3_inverse(self.T_world_camera)

    @property
    def forward_vector(self) -> np.ndarray:
        """Camera optical z-axis in world coordinates (viewing direction)."""
        return self.R[:, 2]

    @property
    def up_vector(self) -> np.ndarray:
        """Camera negative y-axis in world coordinates (up direction in standard CV convention)."""
        return -self.R[:, 1]

    @classmethod
    def look_at(cls, eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> CameraPose:
        """
        Construct a camera pose looking from 'eye' at 'target' with up-vector 'up' (OpenCV convention: +Z forward, +X right, +Y down).
        """
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)

        forward = target - eye
        norm_f = np.linalg.norm(forward)
        if norm_f < 1e-8:
            forward = np.array([0.0, 0.0, 1.0])
        else:
            forward = forward / norm_f

        right = np.cross(forward, up)
        norm_r = np.linalg.norm(right)
        if norm_r < 1e-8:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / norm_r

        # OpenCV convention: Y points down
        down = np.cross(forward, right)

        R = np.column_stack([right, down, forward])
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = eye
        return cls(T_world_camera=T)


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Compute exact inverse of a 4x4 SE(3) rigid matrix."""
    T_inv = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def se3_transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Apply 4x4 rigid transformation to an (N, 3) array of points.
    p' = R @ p + t
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    R = T[:3, :3]
    t = T[:3, 3]
    return (points @ R.T) + t


def compute_relative_pose(T_w_c1: np.ndarray, T_w_c2: np.ndarray) -> np.ndarray:
    """
    Compute relative transformation T_c2_c1 from camera 1 to camera 2:
    p_c2 = T_c2_c1 @ p_c1 = T_c2_w @ T_w_c1 @ p_c1
    """
    T_c2_w = se3_inverse(T_w_c2)
    return T_c2_w @ T_w_c1


def project_points_to_camera(
    points_world: np.ndarray,
    pose: CameraPose,
    intrinsics: CameraIntrinsics
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D world points into 2D camera coordinates.

    Args:
        points_world: (N, 3) array of 3D points in world coordinates
        pose: CameraPose (T_world_camera)
        intrinsics: CameraIntrinsics

    Returns:
        uv: (N, 2) pixel coordinates (u, v)
        z_c: (N,) depth in camera coordinate frame (+Z forward)
        valid_mask: (N,) boolean mask indicating points in front of camera (z_c > 0)
    """
    points_world = np.asarray(points_world, dtype=np.float64)
    if points_world.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,), dtype=bool)

    # Transform to camera frame: p_c = T_c_w @ p_w
    points_cam = se3_transform_points(points_world, pose.T_camera_world)
    z_c = points_cam[:, 2]

    valid_mask = z_c > 1e-4

    # Safe projection
    z_safe = np.where(valid_mask, z_c, 1.0)
    u = (intrinsics.fx * points_cam[:, 0] / z_safe) + intrinsics.cx
    v = (intrinsics.fy * points_cam[:, 1] / z_safe) + intrinsics.cy

    uv = np.column_stack([u, v])
    return uv, z_c, valid_mask


@dataclass
class OrientedBoundingBox3D:
    """3D Oriented Bounding Box (OBB) representing object extents and pose."""
    center: np.ndarray          # (3,) centroid [x, y, z]
    extents: np.ndarray         # (3,) dimensions [dx, dy, dz] (width, depth, height)
    rotation: np.ndarray        # (3, 3) rotation matrix
    label: str = "object"
    confidence: float = 1.0

    @property
    def volume(self) -> float:
        return float(np.prod(self.extents))

    def get_corners(self) -> np.ndarray:
        """Compute the 8 corner vertices in world space (8, 3)."""
        dx, dy, dz = self.extents / 2.0
        local_corners = np.array([
            [-dx, -dy, -dz],
            [ dx, -dy, -dz],
            [ dx,  dy, -dz],
            [-dx,  dy, -dz],
            [-dx, -dy,  dz],
            [ dx, -dy,  dz],
            [ dx,  dy,  dz],
            [-dx,  dy,  dz],
        ])
        return (local_corners @ self.rotation.T) + self.center

    @classmethod
    def from_points(cls, points: np.ndarray, label: str = "object") -> OrientedBoundingBox3D:
        """Fit an oriented bounding box to 3D points using Principal Component Analysis (PCA)."""
        points = np.asarray(points, dtype=np.float64)
        if len(points) < 4:
            # Fallback to axis-aligned box
            min_pt = np.min(points, axis=0) if len(points) > 0 else np.zeros(3)
            max_pt = np.max(points, axis=0) if len(points) > 0 else np.ones(3) * 0.1
            center = (min_pt + max_pt) / 2.0
            extents = np.maximum(max_pt - min_pt, 0.02)
            return cls(center=center, extents=extents, rotation=np.eye(3), label=label)

        center = np.mean(points, axis=0)
        centered_pts = points - center

        # Covariance matrix
        cov = np.cov(centered_pts, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort eigenvectors by descending eigenvalue
        sort_idx = np.argsort(eigenvalues)[::-1]
        R = eigenvectors[:, sort_idx]

        # Ensure right-handed coordinate system (det(R) == 1)
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1.0

        # Project points onto principal axes
        projected = centered_pts @ R
        min_proj = np.min(projected, axis=0)
        max_proj = np.max(projected, axis=0)

        # Center in world coordinates
        center_offset = (min_proj + max_proj) / 2.0
        center = center + (R @ center_offset)
        extents = np.maximum(max_proj - min_proj, 0.02)

        return cls(center=center, extents=extents, rotation=R, label=label)
