"""
Point cloud representation, multi-view unprojection, filtering, and serialization.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from .transforms import CameraIntrinsics, CameraPose, se3_transform_points


@dataclass
class PointCloud:
    """
    3D Point Cloud container with color, labels, and geometry helpers.
    """
    points: np.ndarray             # (N, 3) float64
    colors: np.ndarray             # (N, 3) float64 in range [0.0, 1.0]
    normals: Optional[np.ndarray] = None  # (N, 3)
    labels: Optional[np.ndarray] = None   # (N,) int32 or string
    scan_id: str = "A"             # "A" or "B" or "merged"

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 3) if len(self.points) > 0 else np.zeros((0, 3), dtype=np.float64)
        n = len(self.points)
        if self.colors is None or len(self.colors) == 0:
            self.colors = np.ones((n, 3), dtype=np.float64) * 0.7
        else:
            self.colors = np.asarray(self.colors, dtype=np.float64).reshape(-1, 3)
            # Ensure 0.0 - 1.0 range
            if self.colors.max() > 1.0:
                self.colors = self.colors / 255.0

        if self.normals is not None and len(self.normals) > 0:
            self.normals = np.asarray(self.normals, dtype=np.float64).reshape(-1, 3)
        if self.labels is not None and len(self.labels) > 0:
            self.labels = np.asarray(self.labels)

    def __len__(self) -> int:
        return len(self.points)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box (min_pt, max_pt)."""
        if len(self.points) == 0:
            return np.zeros(3), np.zeros(3)
        return np.min(self.points, axis=0), np.max(self.points, axis=0)

    @property
    def centroid(self) -> np.ndarray:
        """3D centroid."""
        if len(self.points) == 0:
            return np.zeros(3)
        return np.mean(self.points, axis=0)

    def transform(self, T: np.ndarray) -> PointCloud:
        """Apply 4x4 SE(3) transformation in-place and return self."""
        if len(self.points) > 0:
            self.points = se3_transform_points(self.points, T)
            if self.normals is not None:
                R = T[:3, :3]
                self.normals = self.normals @ R.T
        return self

    def filter_bounds(self, min_bound: np.ndarray, max_bound: np.ndarray) -> PointCloud:
        """Filter points strictly inside [min_bound, max_bound]."""
        if len(self.points) == 0:
            return self
        mask = np.all((self.points >= min_bound) & (self.points <= max_bound), axis=1)
        return PointCloud(
            points=self.points[mask],
            colors=self.colors[mask],
            normals=self.normals[mask] if self.normals is not None else None,
            labels=self.labels[mask] if self.labels is not None else None,
            scan_id=self.scan_id
        )

    def downsample(self, voxel_size: float = 0.02) -> PointCloud:
        """Voxel grid downsampling for spatial uniformity and fast processing."""
        return voxel_downsample(self, voxel_size=voxel_size)

    @classmethod
    def merge(cls, pcds: List[PointCloud], scan_id: str = "merged") -> PointCloud:
        """Merge a list of PointCloud objects into a single cloud."""
        if not pcds:
            return cls(points=np.zeros((0, 3)), colors=np.zeros((0, 3)), scan_id=scan_id)

        all_pts = []
        all_cols = []
        all_norms = []
        all_labels = []

        has_normals = any(p.normals is not None for p in pcds)
        has_labels = any(p.labels is not None for p in pcds)

        for p in pcds:
            if len(p.points) == 0:
                continue
            all_pts.append(p.points)
            all_cols.append(p.colors)
            if has_normals:
                if p.normals is not None:
                    all_norms.append(p.normals)
                else:
                    all_norms.append(np.zeros((len(p.points), 3)))
            if has_labels:
                if p.labels is not None:
                    all_labels.append(p.labels)
                else:
                    all_labels.append(np.full((len(p.points),), -1))

        if not all_pts:
            return cls(points=np.zeros((0, 3)), colors=np.zeros((0, 3)), scan_id=scan_id)

        pts_cat = np.vstack(all_pts)
        cols_cat = np.vstack(all_cols)
        norms_cat = np.vstack(all_norms) if has_normals else None
        labels_cat = np.concatenate(all_labels) if has_labels else None

        return cls(points=pts_cat, colors=cols_cat, normals=norms_cat, labels=labels_cat, scan_id=scan_id)

    def to_ply(self, filepath: str) -> None:
        """Export point cloud to standard binary PLY format."""
        num_points = len(self.points)
        colors_uint8 = np.clip(self.colors * 255.0, 0, 255).astype(np.uint8)

        header = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {num_points}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header\n"
        ]
        header_bytes = "\n".join(header).encode("ascii")

        # Pack vertex data
        data = np.zeros(num_points, dtype=[
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        data['x'] = self.points[:, 0].astype(np.float32)
        data['y'] = self.points[:, 1].astype(np.float32)
        data['z'] = self.points[:, 2].astype(np.float32)
        data['red'] = colors_uint8[:, 0]
        data['green'] = colors_uint8[:, 1]
        data['blue'] = colors_uint8[:, 2]

        with open(filepath, "wb") as f:
            f.write(header_bytes)
            f.write(data.tobytes())


def unproject_depth_map(
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    rgb_image: Optional[np.ndarray] = None,
    depth_min: float = 0.15,
    depth_max: float = 4.0,
    stride: int = 1
) -> PointCloud:
    """
    Unproject 2D depth map and corresponding RGB image into 3D world coordinate frame.

    Args:
        depth_map: (H, W) float depth in meters
        intrinsics: CameraIntrinsics
        pose: CameraPose (T_world_camera)
        rgb_image: Optional (H, W, 3) RGB uint8 or float image
        depth_min: Minimum valid depth in meters
        depth_max: Maximum valid depth in meters
        stride: Pixel subsampling stride for performance

    Returns:
        PointCloud in global world frame.
    """
    H, W = depth_map.shape[:2]
    # Pixel coordinate grid
    v_grid, u_grid = np.mgrid[0:H:stride, 0:W:stride]
    u = u_grid.flatten()
    v = v_grid.flatten()
    z = depth_map[v, u].flatten()

    valid = (z >= depth_min) & (z <= depth_max) & np.isfinite(z)
    u_val = u[valid]
    v_val = v[valid]
    z_val = z[valid]

    if len(z_val) == 0:
        return PointCloud(points=np.zeros((0, 3)), colors=np.zeros((0, 3)))

    # Unproject to camera frame
    x_c = (u_val - intrinsics.cx) * z_val / intrinsics.fx
    y_c = (v_val - intrinsics.cy) * z_val / intrinsics.fy
    pts_cam = np.column_stack([x_c, y_c, z_val])

    # Transform to world coordinates: p_w = R * p_c + t
    pts_world = se3_transform_points(pts_cam, pose.T_world_camera)

    if rgb_image is not None:
        rgb_val = rgb_image[v_val, u_val].astype(np.float64)
        if rgb_val.max() > 1.0:
            rgb_val = rgb_val / 255.0
    else:
        rgb_val = np.ones((len(pts_world), 3)) * 0.7

    return PointCloud(points=pts_world, colors=rgb_val)


def voxel_downsample(pcd: PointCloud, voxel_size: float = 0.02) -> PointCloud:
    """
    Downsample point cloud by hashing into regular cubic voxels and computing centroid.
    Fast vectorized implementation in pure NumPy.
    """
    if len(pcd.points) <= 1:
        return pcd

    coords = np.floor(pcd.points / voxel_size).astype(np.int64)

    # Use unique inverse index to group points per voxel
    _, unique_indices, inverse_indices = np.unique(coords, axis=0, return_index=True, return_inverse=True)
    num_voxels = len(unique_indices)

    # Compute mean per voxel using bincount
    downsampled_points = np.zeros((num_voxels, 3), dtype=np.float64)
    downsampled_colors = np.zeros((num_voxels, 3), dtype=np.float64)
    counts = np.bincount(inverse_indices, minlength=num_voxels).astype(np.float64)

    for i in range(3):
        downsampled_points[:, i] = np.bincount(inverse_indices, weights=pcd.points[:, i], minlength=num_voxels) / counts
        downsampled_colors[:, i] = np.bincount(inverse_indices, weights=pcd.colors[:, i], minlength=num_voxels) / counts

    labels = None
    if pcd.labels is not None:
        labels = pcd.labels[unique_indices]

    normals = None
    if pcd.normals is not None:
        downsampled_normals = np.zeros((num_voxels, 3), dtype=np.float64)
        for i in range(3):
            downsampled_normals[:, i] = np.bincount(inverse_indices, weights=pcd.normals[:, i], minlength=num_voxels) / counts
        norm_lens = np.linalg.norm(downsampled_normals, axis=1, keepdims=True)
        norm_lens[norm_lens < 1e-6] = 1.0
        normals = downsampled_normals / norm_lens

    return PointCloud(
        points=downsampled_points,
        colors=downsampled_colors,
        normals=normals,
        labels=labels,
        scan_id=pcd.scan_id
    )
