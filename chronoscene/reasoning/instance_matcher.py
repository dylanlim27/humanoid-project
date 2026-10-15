"""
Instance-level object association, Hungarian bipartite matching, and robotic 6-DoF delta estimation.
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..perception.segmentation import ObjectInstance3D
from .occupancy_grid import VoxelState


class ObjectChangeType(Enum):
    STATIC = "static"
    MOVED = "moved"
    ADDED = "added"
    REMOVED = "removed"
    UNOBSERVED = "unobserved"


@dataclass
class ObjectChangeDelta:
    """
    Detailed physical change delta for a specific 3D object instance.
    Includes robotic action attributes for downstream pick-and-place / navigation.
    """
    change_type: ObjectChangeType
    object_a: Optional[ObjectInstance3D] = None
    object_b: Optional[ObjectInstance3D] = None
    displacement_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [dx, dy, dz] in meters
    displacement_distance: float = 0.0             # meters
    volume_ratio: float = 1.0
    label: str = "object"
    confidence: float = 1.0

    @property
    def robotic_action_primitive(self) -> Dict[str, Any]:
        """
        Embodied robotics action directive format for robot arms or mobile manipulators.
        """
        if self.change_type == ObjectChangeType.MOVED:
            return {
                "action": "relocate_object",
                "object_label": self.label,
                "pickup_pose": [round(float(c), 3) for c in self.object_b.centroid],
                "return_pose": [round(float(c), 3) for c in self.object_a.centroid],
                "displacement_m": round(float(self.displacement_distance), 3),
                "delta_vector": [round(float(v), 3) for v in self.displacement_vector],
                "dimensions": [round(float(e), 3) for e in self.object_b.obb.extents]
            }
        elif self.change_type == ObjectChangeType.ADDED:
            return {
                "action": "inspect_novel_object",
                "object_label": self.label,
                "current_pose": [round(float(c), 3) for c in self.object_b.centroid],
                "dimensions": [round(float(e), 3) for e in self.object_b.obb.extents]
            }
        elif self.change_type == ObjectChangeType.REMOVED:
            return {
                "action": "replenish_or_verify_missing",
                "object_label": self.label,
                "previous_pose": [round(float(c), 3) for c in self.object_a.centroid],
                "dimensions": [round(float(e), 3) for e in self.object_a.obb.extents]
            }
        return {"action": "none", "object_label": self.label}


class InstanceMatcher3D:
    """
    Performs optimal bipartite Hungarian matching between 3D object instances across Scan A and Scan B.
    """

    def __init__(
        self,
        motion_threshold: float = 0.05,       # Objects displaced > 5cm are classified as MOVED
        max_association_distance: float = 1.2, # Maximum plausible movement distance across scans
        weight_dist: float = 1.0,
        weight_vol: float = 0.5,
        weight_color: float = 0.5
    ):
        self.motion_threshold = motion_threshold
        self.max_association_dist = max_association_distance
        self.w_dist = weight_dist
        self.w_vol = weight_vol
        self.w_color = weight_color

    def match_instances(
        self,
        objects_a: List[ObjectInstance3D],
        objects_b: List[ObjectInstance3D],
        verified_empty_in_b_fn = None,
        verified_empty_in_a_fn = None
    ) -> List[ObjectChangeDelta]:
        """
        Associate objects and classify into STATIC, MOVED, ADDED, REMOVED, or UNOBSERVED.
        """
        deltas: List[ObjectChangeDelta] = []
        Na = len(objects_a)
        Nb = len(objects_b)

        if Na == 0 and Nb == 0:
            return deltas

        if Na == 0:
            # All objects in B are potentially ADDED
            for obj_b in objects_b:
                deltas.append(ObjectChangeDelta(
                    change_type=ObjectChangeType.ADDED,
                    object_b=obj_b,
                    label=obj_b.label
                ))
            return deltas

        if Nb == 0:
            # All objects in A are potentially REMOVED
            for obj_a in objects_a:
                deltas.append(ObjectChangeDelta(
                    change_type=ObjectChangeType.REMOVED,
                    object_a=obj_a,
                    label=obj_a.label
                ))
            return deltas

        # Build pairwise cost matrix
        cost_matrix = np.full((Na, Nb), 1e6, dtype=np.float64)

        for i, oa in enumerate(objects_a):
            for j, ob in enumerate(objects_b):
                dist = np.linalg.norm(oa.centroid - ob.centroid)
                if dist > self.max_association_dist:
                    continue

                # Volume ratio penalty
                vol_ratio = max(oa.volume, 1e-4) / max(ob.volume, 1e-4)
                vol_cost = abs(1.0 - (vol_ratio if vol_ratio <= 1.0 else 1.0 / vol_ratio))

                # Color similarity
                color_dist = np.linalg.norm(oa.mean_color - ob.mean_color)

                cost = (self.w_dist * dist) + (self.w_vol * vol_cost) + (self.w_color * color_dist)
                cost_matrix[i, j] = cost

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_a = set()
        matched_b = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e5:  # Valid match
                oa = objects_a[r]
                ob = objects_b[c]
                matched_a.add(r)
                matched_b.add(c)

                disp_vec = ob.centroid - oa.centroid
                disp_dist = float(np.linalg.norm(disp_vec))

                if disp_dist > self.motion_threshold:
                    change_type = ObjectChangeType.MOVED
                else:
                    change_type = ObjectChangeType.STATIC

                label = ob.label if ob.label != "object" else oa.label
                deltas.append(ObjectChangeDelta(
                    change_type=change_type,
                    object_a=oa,
                    object_b=ob,
                    displacement_vector=disp_vec,
                    displacement_distance=disp_dist,
                    volume_ratio=float(oa.volume / max(ob.volume, 1e-4)),
                    label=label
                ))

        # Unmatched in A -> REMOVED or UNOBSERVED
        for i, oa in enumerate(objects_a):
            if i not in matched_a:
                change_type = ObjectChangeType.REMOVED
                if verified_empty_in_b_fn is not None:
                    if not verified_empty_in_b_fn(oa.centroid):
                        change_type = ObjectChangeType.UNOBSERVED

                deltas.append(ObjectChangeDelta(
                    change_type=change_type,
                    object_a=oa,
                    label=oa.label
                ))

        # Unmatched in B -> ADDED or UNOBSERVED
        for j, ob in enumerate(objects_b):
            if j not in matched_b:
                change_type = ObjectChangeType.ADDED
                if verified_empty_in_a_fn is not None:
                    if not verified_empty_in_a_fn(ob.centroid):
                        change_type = ObjectChangeType.UNOBSERVED

                deltas.append(ObjectChangeDelta(
                    change_type=change_type,
                    object_b=ob,
                    label=ob.label
                ))

        return deltas
