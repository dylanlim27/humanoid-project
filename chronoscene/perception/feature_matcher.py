"""
Cross-scan feature matching and SE(3) background co-registration.
Aligns two casual phone recordings into a single shared metric coordinate frame.
"""

from __future__ import annotations
from typing import Tuple, Optional, List
import numpy as np
from scipy.spatial import KDTree

from ..geometry.transforms import CameraPose, se3_inverse, se3_transform_points
from ..geometry.pointcloud import PointCloud


def estimate_rigid_transform_svd(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Compute optimal 4x4 SE(3) rigid transform [R | t] aligning src_pts to dst_pts
    using Kabsch/Umeyama SVD algorithm:
    min sum || dst_i - (R * src_i + t) ||^2
    """
    assert len(src_pts) == len(dst_pts) and len(src_pts) >= 3

    centroid_src = np.mean(src_pts, axis=0)
    centroid_dst = np.mean(dst_pts, axis=0)

    src_centered = src_pts - centroid_src
    dst_centered = dst_pts - centroid_dst

    # Covariance matrix H
    H = src_centered.T @ dst_centered

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure right-handed rotation (det(R) > 0)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = centroid_dst - (R @ centroid_src)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


class BackgroundSE3Registrar:
    """
    Registers Scan B into Scan A's 3D coordinate frame using robust RANSAC + ICP
    on invariant background structures (e.g. walls, floor, stationary tables).
    """

    def __init__(
        self,
        max_correspondence_distance: float = 0.08,
        max_iterations: int = 50,
        tolerance: float = 1e-5
    ):
        self.max_correspondence_distance = max_correspondence_distance
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def register(
        self,
        pcd_a: PointCloud,
        pcd_b: PointCloud,
        initial_guess: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Aligns pcd_b to pcd_a using robust Point-to-Point ICP.

        Args:
            pcd_a: Target point cloud (Reference Frame A).
            pcd_b: Source point cloud (Scan B to be aligned).
            initial_guess: Optional 4x4 initial SE(3) transformation matrix.

        Returns:
            T_a_b: Optimal 4x4 transformation matrix mapping points in B to A: p_a = T_a_b * p_b
            inlier_rmse: Root Mean Squared Error of aligned inlier points.
            inlier_mask: Boolean mask indicating which points in pcd_b are inliers.
        """
        if initial_guess is None:
            T_current = np.eye(4, dtype=np.float64)
        else:
            T_current = np.array(initial_guess, dtype=np.float64)

        if len(pcd_a.points) == 0 or len(pcd_b.points) == 0:
            return T_current, 0.0, np.zeros(len(pcd_b.points), dtype=bool)

        # Downsample for fast ICP convergence
        pcd_a_sub = pcd_a.downsample(voxel_size=0.03)
        pcd_b_sub = pcd_b.downsample(voxel_size=0.03)

        tree_a = KDTree(pcd_a_sub.points)
        prev_error = float("inf")

        for iteration in range(self.max_iterations):
            # Transform source points by current estimate
            pts_b_trans = se3_transform_points(pcd_b_sub.points, T_current)

            # Find nearest neighbors in target
            distances, indices = tree_a.query(pts_b_trans, distance_upper_bound=self.max_correspondence_distance)
            valid = distances < self.max_correspondence_distance

            if np.sum(valid) < 6:
                # Not enough correspondences; break
                break

            matched_b = pcd_b_sub.points[valid]
            matched_a = pcd_a_sub.points[indices[valid]]

            # Estimate incremental delta transformation
            T_delta = estimate_rigid_transform_svd(matched_b, matched_a)
            T_current = T_delta

            mean_error = np.mean(distances[valid])
            if abs(prev_error - mean_error) < self.tolerance:
                break
            prev_error = mean_error

        # Compute final inlier statistics on full point cloud
        full_tree = KDTree(pcd_a.points)
        full_pts_b_trans = se3_transform_points(pcd_b.points, T_current)
        distances, _ = full_tree.query(full_pts_b_trans, distance_upper_bound=self.max_correspondence_distance)
        inlier_mask = distances < self.max_correspondence_distance
        rmse = float(np.sqrt(np.mean(distances[inlier_mask] ** 2))) if np.any(inlier_mask) else 0.0

        return T_current, rmse, inlier_mask


class CrossScanAligner:
    """
    High-level manager for establishing a single shared coordinate system between Scan A and Scan B.
    """

    def __init__(self, registrar: Optional[BackgroundSE3Registrar] = None):
        self.registrar = registrar or BackgroundSE3Registrar()

    def align_scans(
        self,
        pcd_a: PointCloud,
        pcd_b: PointCloud,
        poses_b: List[CameraPose]
    ) -> Tuple[PointCloud, List[CameraPose], np.ndarray]:
        """
        Transforms Scan B's point cloud and camera poses into Scan A's global reference frame.

        Returns:
            pcd_b_aligned: PointCloud B expressed in Frame A.
            poses_b_aligned: Camera poses of Scan B expressed in Frame A.
            T_a_b: 4x4 SE(3) transformation matrix applied.
        """
        T_a_b, rmse, inliers = self.registrar.register(pcd_a, pcd_b)

        # Transform point cloud B
        pcd_b_aligned = PointCloud(
            points=se3_transform_points(pcd_b.points, T_a_b),
            colors=pcd_b.colors.copy(),
            normals=pcd_b.normals.copy() if pcd_b.normals is not None else None,
            labels=pcd_b.labels.copy() if pcd_b.labels is not None else None,
            scan_id="B"
        )

        # Transform camera poses: T_wA_cB = T_wA_wB @ T_wB_cB
        poses_b_aligned = []
        for pose in poses_b:
            T_aligned = T_a_b @ pose.T_world_camera
            poses_b_aligned.append(CameraPose(T_world_camera=T_aligned))

        return pcd_b_aligned, poses_b_aligned, T_a_b
