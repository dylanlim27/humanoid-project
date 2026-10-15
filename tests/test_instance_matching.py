"""
Unit tests for Hungarian matching and 6-DoF object displacement estimation.
"""

import numpy as np
import pytest

from chronoscene.geometry.transforms import OrientedBoundingBox3D
from chronoscene.perception.segmentation import ObjectInstance3D
from chronoscene.reasoning.instance_matcher import (
    InstanceMatcher3D,
    ObjectChangeType
)


def test_moved_object_matching():
    # Object in A at [0, 0, 0]
    obb_a = OrientedBoundingBox3D(
        center=np.array([0.0, 0.0, 0.0]),
        extents=np.array([0.1, 0.1, 0.1]),
        rotation=np.eye(3),
        label="mug"
    )
    obj_a = ObjectInstance3D(
        id="obj_A_01",
        label="mug",
        centroid=np.array([0.0, 0.0, 0.0]),
        obb=obb_a,
        points=np.zeros((10, 3)),
        mean_color=np.array([0.9, 0.9, 0.9]),
        scan_id="A"
    )

    # Same object moved to [0.3, 0.0, 0.0] in B
    obb_b = OrientedBoundingBox3D(
        center=np.array([0.3, 0.0, 0.0]),
        extents=np.array([0.1, 0.1, 0.1]),
        rotation=np.eye(3),
        label="mug"
    )
    obj_b = ObjectInstance3D(
        id="obj_B_01",
        label="mug",
        centroid=np.array([0.3, 0.0, 0.0]),
        obb=obb_b,
        points=np.zeros((10, 3)),
        mean_color=np.array([0.9, 0.9, 0.9]),
        scan_id="B"
    )

    matcher = InstanceMatcher3D(motion_threshold=0.05)
    deltas = matcher.match_instances([obj_a], [obj_b])

    assert len(deltas) == 1
    assert deltas[0].change_type == ObjectChangeType.MOVED
    assert np.isclose(deltas[0].displacement_distance, 0.3, atol=1e-3)
    assert np.allclose(deltas[0].displacement_vector, [0.3, 0.0, 0.0], atol=1e-3)


def test_added_and_removed_objects():
    # Object only in A (removed)
    obb_a = OrientedBoundingBox3D(
        center=np.array([-0.5, 0.0, 0.0]),
        extents=np.array([0.1, 0.1, 0.1]),
        rotation=np.eye(3),
        label="notebook"
    )
    obj_a = ObjectInstance3D(
        id="obj_A_01",
        label="notebook",
        centroid=np.array([-0.5, 0.0, 0.0]),
        obb=obb_a,
        points=np.zeros((10, 3)),
        mean_color=np.array([0.1, 0.3, 0.9]),
        scan_id="A"
    )

    # Object only in B (added, distant location, different color)
    obb_b = OrientedBoundingBox3D(
        center=np.array([0.8, 0.5, 0.0]),
        extents=np.array([0.2, 0.2, 0.2]),
        rotation=np.eye(3),
        label="drill"
    )
    obj_b = ObjectInstance3D(
        id="obj_B_01",
        label="drill",
        centroid=np.array([0.8, 0.5, 0.0]),
        obb=obb_b,
        points=np.zeros((10, 3)),
        mean_color=np.array([0.9, 0.8, 0.1]),
        scan_id="B"
    )

    matcher = InstanceMatcher3D(motion_threshold=0.05, max_association_distance=0.5)
    deltas = matcher.match_instances([obj_a], [obj_b])

    types = {d.change_type for d in deltas}
    assert ObjectChangeType.REMOVED in types
    assert ObjectChangeType.ADDED in types
