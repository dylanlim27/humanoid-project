"""
Core 3D change detection engine combining spatial proximity with line-of-sight raycasting.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.spatial import KDTree

from ..geometry.transforms import CameraIntrinsics, CameraPose
from ..geometry.pointcloud import PointCloud
from .occupancy_grid import MultiViewRaycastEngine, VoxelState, ObservationStatus


@dataclass
class ChangeClassificationResult:
    """Detailed results of 3D change classification across dual scans."""
    pcd_a: PointCloud
    pcd_b: PointCloud
    labels_a: np.ndarray             # (N_a,) VoxelState for each point in Scan A
    labels_b: np.ndarray             # (N_b,) VoxelState for each point in Scan B
    metrics: Dict[str, Any]          # Summary counts and percentage distributions
    static_pcd: PointCloud           # Merged static background
    added_pcd: PointCloud            # Verified newly added points in Scan B
    removed_pcd: PointCloud          # Verified removed points from Scan A
    occluded_pcd: PointCloud         # Unobserved / occluded points


class ChangeDetector3D:
    """
    Identifies meaningful physical 3D changes while rigorously rejecting false positives
    from viewpoint differences, occlusions, and incomplete observations.
    """

    def __init__(
        self,
        static_distance_threshold: float = 0.04,  # 4 cm tolerance for static coincidence
        raycast_engine: Optional[MultiViewRaycastEngine] = None
    ):
        self.static_threshold = static_distance_threshold
        self.raycast_engine = raycast_engine or MultiViewRaycastEngine(surface_thickness_epsilon=0.04)

    def detect_changes(
        self,
        pcd_a: PointCloud,
        pcd_b: PointCloud,
        poses_a: List[CameraPose],
        depth_maps_a: List[np.ndarray],
        poses_b: List[CameraPose],
        depth_maps_b: List[np.ndarray],
        intrinsics: CameraIntrinsics
    ) -> ChangeClassificationResult:
        """
        Execute full change classification on aligned scans A and B.
        """
        Na = len(pcd_a.points)
        Nb = len(pcd_b.points)

        labels_a = np.full(Na, VoxelState.STATIC, dtype=object)
        labels_b = np.full(Nb, VoxelState.STATIC, dtype=object)

        if Na == 0 or Nb == 0:
            empty = PointCloud(points=np.zeros((0, 3)), colors=np.zeros((0, 3)))
            return ChangeClassificationResult(
                pcd_a=pcd_a, pcd_b=pcd_b,
                labels_a=labels_a, labels_b=labels_b,
                metrics={}, static_pcd=empty, added_pcd=empty,
                removed_pcd=empty, occluded_pcd=empty
            )

        # 1. Proximity matching via KDTree
        tree_b = KDTree(pcd_b.points)
        dist_a_to_b, _ = tree_a_to_b = tree_b.query(pcd_a.points)

        tree_a = KDTree(pcd_a.points)
        dist_b_to_a, _ = tree_a.query(pcd_b.points)

        # Points close to opposite scan are STATIC
        static_mask_a = dist_a_to_b <= self.static_threshold
        static_mask_b = dist_b_to_a <= self.static_threshold

        labels_a[static_mask_a] = VoxelState.STATIC
        labels_b[static_mask_b] = VoxelState.STATIC

        # 2. Candidate changed points
        cand_removed_idx = np.where(~static_mask_a)[0]
        cand_added_idx = np.where(~static_mask_b)[0]

        # 3. Raycast Disambiguation on Scan A candidates tested in Scan B
        if len(cand_removed_idx) > 0:
            query_pts_a = pcd_a.points[cand_removed_idx]
            vis_in_b = self.raycast_engine.aggregate_scan_visibility(
                query_pts_a, poses_b, intrinsics, depth_maps_b
            )

            for i, orig_idx in enumerate(cand_removed_idx):
                if vis_in_b["is_verified_empty"][i]:
                    # Scan B explicitly looked here and verified free space -> REMOVED
                    labels_a[orig_idx] = VoxelState.REMOVED
                elif vis_in_b["is_unobserved_or_occluded"][i]:
                    # Scan B didn't see this point due to occlusion or trajectory cut-off -> NOT REMOVED
                    labels_a[orig_idx] = VoxelState.OCCLUDED
                else:
                    labels_a[orig_idx] = VoxelState.OUT_OF_VIEW

        # 4. Raycast Disambiguation on Scan B candidates tested in Scan A
        if len(cand_added_idx) > 0:
            query_pts_b = pcd_b.points[cand_added_idx]
            vis_in_a = self.raycast_engine.aggregate_scan_visibility(
                query_pts_b, poses_a, intrinsics, depth_maps_a
            )

            for i, orig_idx in enumerate(cand_added_idx):
                if vis_in_a["is_verified_empty"][i]:
                    # Scan A explicitly verified free space here -> ADDED
                    labels_b[orig_idx] = VoxelState.ADDED
                elif vis_in_a["is_unobserved_or_occluded"][i]:
                    # Scan A didn't sweep here or it was occluded in A -> NOT ADDED
                    labels_b[orig_idx] = VoxelState.OCCLUDED
                else:
                    labels_b[orig_idx] = VoxelState.OUT_OF_VIEW

        # Extract segmented sub-clouds
        rem_mask = labels_a == VoxelState.REMOVED
        add_mask = labels_b == VoxelState.ADDED
        stat_a_mask = labels_a == VoxelState.STATIC
        stat_b_mask = labels_b == VoxelState.STATIC
        occ_mask_a = (labels_a == VoxelState.OCCLUDED) | (labels_a == VoxelState.OUT_OF_VIEW)
        occ_mask_b = (labels_b == VoxelState.OCCLUDED) | (labels_b == VoxelState.OUT_OF_VIEW)

        removed_pcd = PointCloud(
            points=pcd_a.points[rem_mask],
            colors=pcd_a.colors[rem_mask].copy() if np.sum(rem_mask) > 0 else np.zeros((0, 3)),
            scan_id="A"
        )
        added_pcd = PointCloud(
            points=pcd_b.points[add_mask],
            colors=pcd_b.colors[add_mask].copy() if np.sum(add_mask) > 0 else np.zeros((0, 3)),
            scan_id="B"
        )

        static_pts = np.vstack([pcd_a.points[stat_a_mask], pcd_b.points[stat_b_mask]]) if (np.sum(stat_a_mask) + np.sum(stat_b_mask)) > 0 else np.zeros((0, 3))
        static_cols = np.vstack([pcd_a.colors[stat_a_mask], pcd_b.colors[stat_b_mask]]) if (np.sum(stat_a_mask) + np.sum(stat_b_mask)) > 0 else np.zeros((0, 3))
        static_pcd = PointCloud(points=static_pts, colors=static_cols, scan_id="merged")

        occ_pts = np.vstack([pcd_a.points[occ_mask_a], pcd_b.points[occ_mask_b]]) if (np.sum(occ_mask_a) + np.sum(occ_mask_b)) > 0 else np.zeros((0, 3))
        occ_cols = np.tile(np.array([0.2, 0.6, 0.9]), (len(occ_pts), 1))  # Light Blue
        occluded_pcd = PointCloud(points=occ_pts, colors=occ_cols, scan_id="merged")

        total_points = Na + Nb
        metrics = {
            "total_points_scan_a": Na,
            "total_points_scan_b": Nb,
            "static_points": int(np.sum(stat_a_mask) + np.sum(stat_b_mask)),
            "static_percentage": round(100.0 * (np.sum(stat_a_mask) + np.sum(stat_b_mask)) / total_points, 1),
            "removed_points": int(np.sum(rem_mask)),
            "added_points": int(np.sum(add_mask)),
            "occluded_or_unobserved_points": int(len(occ_pts)),
            "occluded_percentage": round(100.0 * len(occ_pts) / total_points, 1)
        }

        return ChangeClassificationResult(
            pcd_a=pcd_a,
            pcd_b=pcd_b,
            labels_a=labels_a,
            labels_b=labels_b,
            metrics=metrics,
            static_pcd=static_pcd,
            added_pcd=added_pcd,
            removed_pcd=removed_pcd,
            occluded_pcd=occluded_pcd
        )
