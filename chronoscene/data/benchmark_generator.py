"""
Procedural photorealistic 3D indoor benchmark generator with ground truth scene changes.
Simulates casual phone camera sweeps over a tabletop robotics workspace with known moved,
added, removed, and occluded objects.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from ..geometry.transforms import CameraIntrinsics, CameraPose, se3_transform_points
from ..geometry.pointcloud import PointCloud
from ..reasoning.occupancy_grid import VoxelState
from .video_loader import VideoScanData


@dataclass
class GroundTruthSceneDelta:
    """Ground truth 3D change definition for benchmark evaluation."""
    object_name: str
    true_state: VoxelState
    centroid_a: Optional[np.ndarray] = None
    centroid_b: Optional[np.ndarray] = None
    displacement_m: float = 0.0


class IndoorBenchmarkGenerator:
    """
    Synthesizes dual-scan indoor benchmarks with exact analytical 3D geometry.
    Guarantees reproducible quantitative verification without needing external data downloads.
    """

    def __init__(
        self,
        width: int = 480,
        height: int = 360,
        fov_deg: float = 65.0
    ):
        self.width = width
        self.height = height
        self.intrinsics = CameraIntrinsics.from_fov(fov_deg, width, height)

    def _render_scene_geometry(
        self,
        objects: List[Dict[str, Any]],
        pose: CameraPose
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analytical raycasting to generate dense depth map and RGB image.
        """
        H, W = self.height, self.width
        depth_map = np.full((H, W), np.inf, dtype=np.float32)
        rgb_image = np.zeros((H, W, 3), dtype=np.uint8)

        # Background ambient color (dark room wall/floor)
        rgb_image[:, :] = [35, 38, 46]

        # Generate camera rays
        v_grid, u_grid = np.mgrid[0:H, 0:W]
        x_c = (u_grid - self.intrinsics.cx) / self.intrinsics.fx
        y_c = (v_grid - self.intrinsics.cy) / self.intrinsics.fy
        z_c = np.ones_like(x_c)

        rays_cam = np.stack([x_c, y_c, z_c], axis=-1)
        ray_norms = np.linalg.norm(rays_cam, axis=-1, keepdims=True)
        dirs_cam = rays_cam / ray_norms  # (H, W, 3)

        # Transform rays to world: dir_w = R * dir_c, orig_w = pose.t
        R = pose.R
        orig_w = pose.t
        dirs_w = dirs_cam @ R.T  # (H, W, 3)

        # 1. Render Tabletop Plane: z = 0.75, x in [-0.65, 0.65], y in [-0.45, 0.45]
        # Ray equation: orig_w + t * dirs_w. z = 0.75 => t = (0.75 - orig_w[2]) / dirs_w[..., 2]
        dz = dirs_w[..., 2]
        valid_z = np.abs(dz) > 1e-4
        t_plane = (0.75 - orig_w[2]) / np.where(valid_z, dz, 1.0)
        valid_t = valid_z & (t_plane > 0.1)

        hit_x = orig_w[0] + (t_plane * dirs_w[..., 0])
        hit_y = orig_w[1] + (t_plane * dirs_w[..., 1])
        on_table = valid_t & (hit_x >= -0.65) & (hit_x <= 0.65) & (hit_y >= -0.45) & (hit_y <= 0.45)

        # Depth in camera z-axis: z_hit = t_plane * dirs_cam[..., 2] = t_plane / ray_norms[..., 0]
        z_plane_cam = t_plane * dirs_cam[..., 2]
        table_mask = on_table & (z_plane_cam < depth_map)

        depth_map[table_mask] = z_plane_cam[table_mask]
        rgb_image[table_mask] = [170, 140, 110]  # Oak wood table

        # 2. Render 3D Box Objects (Tabletop items)
        for obj in objects:
            center = obj["center"]
            half = obj["extents"] / 2.0
            color = obj["color"]

            # Bounding box slab method
            min_b = center - half
            max_b = center + half

            # Intersect all rays with AABB
            inv_d = 1.0 / np.where(np.abs(dirs_w) > 1e-6, dirs_w, 1e-6)
            t0 = (min_b - orig_w) * inv_d
            t1 = (max_b - orig_w) * inv_d

            t_min = np.maximum(np.maximum(np.minimum(t0[..., 0], t1[..., 0]),
                                          np.minimum(t0[..., 1], t1[..., 1])),
                               np.minimum(t0[..., 2], t1[..., 2]))
            t_max = np.minimum(np.minimum(np.maximum(t0[..., 0], t1[..., 0]),
                                          np.maximum(t0[..., 1], t1[..., 1])),
                               np.maximum(t0[..., 2], t1[..., 2]))

            hit = (t_max >= t_min) & (t_max > 0.1)
            t_box = np.where(t_min > 0.1, t_min, t_max)
            z_box_cam = t_box * dirs_cam[..., 2]

            box_mask = hit & (z_box_cam < depth_map)
            depth_map[box_mask] = z_box_cam[box_mask]
            # Simple directional shading
            shading = np.clip(0.7 + 0.3 * (dirs_w[..., 2][box_mask] + 0.5), 0.4, 1.0)
            shaded_color = (np.array(color) * shading[:, None]).astype(np.uint8)
            rgb_image[box_mask] = shaded_color

        # Replace inf with 0.0
        depth_map[~np.isfinite(depth_map)] = 0.0

        return depth_map, rgb_image

    def generate_benchmark_pair(
        self,
        num_frames: int = 12
    ) -> Tuple[VideoScanData, VideoScanData, List[GroundTruthSceneDelta]]:
        """
        Synthesize Scan A and Scan B video sequences with ground truth changes.
        """
        # Ground Truth Objects
        gt_deltas = [
            # 1. MOVED: Ceramic mug shifted 45cm across table
            GroundTruthSceneDelta(
                object_name="ceramic_mug",
                true_state=VoxelState.MOVED,
                centroid_a=np.array([-0.30, -0.12, 0.81]),
                centroid_b=np.array([+0.15, -0.15, 0.81]),
                displacement_m=0.45
            ),
            # 2. ADDED: Yellow cordless drill placed on table
            GroundTruthSceneDelta(
                object_name="electric_drill",
                true_state=VoxelState.ADDED,
                centroid_b=np.array([+0.35, 0.10, 0.83])
            ),
            # 3. REMOVED: Blue notebook removed
            GroundTruthSceneDelta(
                object_name="blue_notebook",
                true_state=VoxelState.REMOVED,
                centroid_a=np.array([0.05, -0.22, 0.77])
            ),
            # 4. UNOBSERVED: Tool rack on the far left
            GroundTruthSceneDelta(
                object_name="far_tool_rack",
                true_state=VoxelState.OUT_OF_VIEW,
                centroid_a=np.array([-0.85, 0.15, 0.85])
            )
        ]

        # Scan A Objects
        objects_a = [
            # Static laptop
            {"center": np.array([-0.10, 0.12, 0.83]), "extents": np.array([0.32, 0.24, 0.16]), "color": [60, 64, 72], "name": "laptop"},
            # Static desktop monitor
            {"center": np.array([0.0, 0.40, 0.98]), "extents": np.array([0.50, 0.08, 0.36]), "color": [40, 42, 48], "name": "monitor"},
            # Mug in initial position
            {"center": np.array([-0.30, -0.12, 0.81]), "extents": np.array([0.11, 0.11, 0.12]), "color": [240, 240, 245], "name": "ceramic_mug"},
            # Notebook present
            {"center": np.array([0.05, -0.22, 0.77]), "extents": np.array([0.22, 0.28, 0.04]), "color": [45, 105, 220], "name": "blue_notebook"},
            # Far left tool rack (only visible in Scan A trajectory)
            {"center": np.array([-0.85, 0.15, 0.85]), "extents": np.array([0.16, 0.16, 0.20]), "color": [180, 80, 60], "name": "far_tool_rack"}
        ]

        # Scan B Objects
        objects_b = [
            # Static laptop (same)
            {"center": np.array([-0.10, 0.12, 0.83]), "extents": np.array([0.32, 0.24, 0.16]), "color": [60, 64, 72], "name": "laptop"},
            # Static monitor (same)
            {"center": np.array([0.0, 0.40, 0.98]), "extents": np.array([0.50, 0.08, 0.36]), "color": [40, 42, 48], "name": "monitor"},
            # Mug in NEW position
            {"center": np.array([+0.15, -0.15, 0.81]), "extents": np.array([0.11, 0.11, 0.12]), "color": [240, 240, 245], "name": "ceramic_mug"},
            # Drill newly added
            {"center": np.array([+0.35, 0.10, 0.83]), "extents": np.array([0.14, 0.22, 0.15]), "color": [245, 190, 20], "name": "electric_drill"},
            # Notebook is REMOVED (absent)
            # Far left tool rack is still present in reality, but Scan B trajectory does not sweep left!
            {"center": np.array([-0.85, 0.15, 0.85]), "extents": np.array([0.16, 0.16, 0.20]), "color": [180, 80, 60], "name": "far_tool_rack"}
        ]

        # Generate Camera Trajectories (Simulating casual handheld phone sweeping arc)
        target = np.array([0.0, 0.0, 0.75])  # Tabletop center

        # Scan A: Sweeping from left to center (azimuth -45 to 0 deg)
        angles_a = np.linspace(-np.radians(45.0), np.radians(5.0), num_frames)
        radius_a = 1.35
        poses_a = []
        depths_a = []
        rgbs_a = []

        for ang in angles_a:
            eye = np.array([
                radius_a * np.sin(ang),
                -radius_a * np.cos(ang),
                1.30  # Standing eye height
            ])
            pose = CameraPose.look_at(eye=eye, target=target)
            poses_a.append(pose)
            d, rgb = self._render_scene_geometry(objects_a, pose)
            depths_a.append(d)
            rgbs_a.append(rgb)

        # Scan B: Sweeping from center to right (azimuth 0 to +50 deg)
        angles_b = np.linspace(-np.radians(5.0), np.radians(50.0), num_frames)
        radius_b = 1.30
        poses_b = []
        depths_b = []
        rgbs_b = []

        for ang in angles_b:
            eye = np.array([
                radius_b * np.sin(ang),
                -radius_b * np.cos(ang),
                1.28
            ])
            pose = CameraPose.look_at(eye=eye, target=target)
            poses_b.append(pose)
            d, rgb = self._render_scene_geometry(objects_b, pose)
            depths_b.append(d)
            rgbs_b.append(rgb)

        scan_a = VideoScanData(
            rgb_frames=rgbs_a,
            frame_indices=list(range(num_frames)),
            intrinsics=self.intrinsics,
            fps=30.0,
            duration_sec=float(num_frames / 30.0),
            scan_id="A",
            depth_maps=depths_a,
            poses=poses_a
        )

        scan_b = VideoScanData(
            rgb_frames=rgbs_b,
            frame_indices=list(range(num_frames)),
            intrinsics=self.intrinsics,
            fps=30.0,
            duration_sec=float(num_frames / 30.0),
            scan_id="B",
            depth_maps=depths_b,
            poses=poses_b
        )

        return scan_a, scan_b, gt_deltas
