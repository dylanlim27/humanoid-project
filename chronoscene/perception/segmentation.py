"""
3D instance segmentation, Euclidean point clustering, and object feature extraction.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy.spatial import KDTree

from ..geometry.transforms import OrientedBoundingBox3D
from ..geometry.pointcloud import PointCloud


@dataclass
class Cluster3D:
    """A single spatial cluster of 3D points."""
    points: np.ndarray        # (M, 3)
    colors: np.ndarray        # (M, 3)
    indices: np.ndarray       # (M,) indices in parent point cloud

    @property
    def centroid(self) -> np.ndarray:
        return np.mean(self.points, axis=0) if len(self.points) > 0 else np.zeros(3)

    @property
    def mean_color(self) -> np.ndarray:
        return np.mean(self.colors, axis=0) if len(self.colors) > 0 else np.ones(3) * 0.5


@dataclass
class ObjectInstance3D:
    """
    Physical object entity identified in the 3D scene.
    """
    id: str
    label: str
    centroid: np.ndarray
    obb: OrientedBoundingBox3D
    points: np.ndarray
    mean_color: np.ndarray
    scan_id: str = "A"
    confidence: float = 1.0

    @property
    def volume(self) -> float:
        return self.obb.volume

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "scan_id": self.scan_id,
            "centroid": [round(float(c), 3) for c in self.centroid],
            "extents": [round(float(e), 3) for e in self.obb.extents],
            "volume_m3": round(float(self.volume), 4),
            "mean_rgb": [round(float(c), 2) for c in self.mean_color],
            "point_count": len(self.points)
        }


class EuclideanClusterer3D:
    """
    Clusters 3D points into discrete objects using spatial connectivity (DBSCAN / Euclidean).
    """

    def __init__(
        self,
        cluster_tolerance: float = 0.05,  # 5 cm neighbor radius
        min_cluster_size: int = 40,        # Minimum points to qualify as object
        max_cluster_size: int = 50000      # Maximum points per single object
    ):
        self.cluster_tolerance = cluster_tolerance
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

    def cluster(self, pcd: PointCloud) -> List[Cluster3D]:
        """
        Segment points into spatial clusters.
        """
        pts = pcd.points
        n = len(pts)
        if n < self.min_cluster_size:
            return []

        tree = KDTree(pts)
        visited = np.zeros(n, dtype=bool)
        clusters = []

        for i in range(n):
            if visited[i]:
                continue

            # Breadth-first search / region growing
            cluster_indices = []
            queue = [i]
            visited[i] = True

            while queue:
                curr = queue.pop(0)
                cluster_indices.append(curr)

                neighbors = tree.query_ball_point(pts[curr], r=self.cluster_tolerance)
                for nb in neighbors:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)

            if self.min_cluster_size <= len(cluster_indices) <= self.max_cluster_size:
                idx_arr = np.array(cluster_indices, dtype=np.int64)
                clusters.append(Cluster3D(
                    points=pts[idx_arr],
                    colors=pcd.colors[idx_arr],
                    indices=idx_arr
                ))

        return clusters

    def extract_objects(
        self,
        pcd: PointCloud,
        scan_id: str = "A",
        semantic_labels: Optional[List[str]] = None
    ) -> List[ObjectInstance3D]:
        """
        Segment point cloud into distinct 3D object instances with fitted bounding boxes.
        """
        clusters = self.cluster(pcd)
        objects = []

        for idx, cluster in enumerate(clusters):
            obb = OrientedBoundingBox3D.from_points(cluster.points)
            label = "object"
            if semantic_labels and idx < len(semantic_labels):
                label = semantic_labels[idx]
            else:
                # Infer simple shape heuristic (e.g. cylinder, box, compact)
                ext = obb.extents
                aspect = max(ext) / (min(ext) + 1e-4)
                if aspect > 2.5:
                    label = "elongated_object"
                else:
                    label = "compact_object"

            obb.label = label
            obj = ObjectInstance3D(
                id=f"obj_{scan_id}_{idx+1:02d}",
                label=label,
                centroid=cluster.centroid,
                obb=obb,
                points=cluster.points,
                mean_color=cluster.mean_color,
                scan_id=scan_id
            )
            objects.append(obj)

        return objects
