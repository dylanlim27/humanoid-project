"""
Unit tests for 3D geometry, SE(3) transformations, camera projections, and frustum tests.
"""

import numpy as np
import pytest

from chronoscene.geometry.transforms import (
    CameraIntrinsics,
    CameraPose,
    se3_inverse,
    se3_transform_points,
    project_points_to_camera,
    OrientedBoundingBox3D
)
from chronoscene.geometry.pointcloud import PointCloud, voxel_downsample
from chronoscene.geometry.frustum import CameraFrustum, points_in_frustums


def test_camera_intrinsics():
    intrinsics = CameraIntrinsics.from_fov(fov_deg=60.0, width=640, height=480)
    assert intrinsics.width == 640
    assert intrinsics.height == 480
    assert intrinsics.fx > 0
    assert intrinsics.fy > 0
    assert np.allclose(intrinsics.K @ intrinsics.K_inv, np.eye(3))


def test_se3_inverse():
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
    ])
    t = np.array([1.5, -2.0, 0.5])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    T_inv = se3_inverse(T)
    assert np.allclose(T @ T_inv, np.eye(4))
    assert np.allclose(T_inv @ T, np.eye(4))


def test_camera_projection():
    intrinsics = CameraIntrinsics.from_fov(fov_deg=60.0, width=640, height=480)
    # Camera at origin looking along +Z
    pose = CameraPose(T_world_camera=np.eye(4))

    # A point straight ahead at [0, 0, 2.0] should project to principal point (cx, cy)
    pts = np.array([[0.0, 0.0, 2.0]])
    uv, z_c, valid = project_points_to_camera(pts, pose, intrinsics)

    assert valid[0]
    assert np.isclose(uv[0, 0], intrinsics.cx, atol=1e-3)
    assert np.isclose(uv[0, 1], intrinsics.cy, atol=1e-3)
    assert np.isclose(z_c[0], 2.0, atol=1e-3)


def test_frustum_visibility():
    intrinsics = CameraIntrinsics.from_fov(fov_deg=60.0, width=640, height=480)
    pose = CameraPose(T_world_camera=np.eye(4))
    frustum = CameraFrustum(pose=pose, intrinsics=intrinsics, z_near=0.1, z_far=5.0)

    # Point in front and centered -> inside
    pt_inside = np.array([[0.0, 0.0, 2.0]])
    assert frustum.contains_points(pt_inside)[0]

    # Point behind camera -> outside
    pt_behind = np.array([[0.0, 0.0, -1.0]])
    assert not frustum.contains_points(pt_behind)[0]

    # Point way too far -> outside
    pt_far = np.array([[0.0, 0.0, 10.0]])
    assert not frustum.contains_points(pt_far)[0]


def test_voxel_downsampling():
    pts = np.array([
        [0.001, 0.001, 0.001],
        [0.002, 0.002, 0.002],
        [0.003, 0.003, 0.003],
        [1.000, 1.000, 1.000]
    ])
    cols = np.ones((4, 3))
    pcd = PointCloud(points=pts, colors=cols)
    down = voxel_downsample(pcd, voxel_size=0.05)
    # The first 3 points collapse into 1 voxel; the 4th is in another voxel
    assert len(down.points) == 2


def test_oriented_bounding_box():
    # Box from [-0.5, -0.2, -0.1] to [0.5, 0.2, 0.1]
    pts = np.array([
        [-0.5, -0.2, -0.1],
        [ 0.5, -0.2, -0.1],
        [ 0.5,  0.2, -0.1],
        [-0.5,  0.2, -0.1],
        [-0.5, -0.2,  0.1],
        [ 0.5, -0.2,  0.1],
        [ 0.5,  0.2,  0.1],
        [-0.5,  0.2,  0.1],
    ])
    obb = OrientedBoundingBox3D.from_points(pts)
    assert np.allclose(obb.center, [0.0, 0.0, 0.0], atol=1e-2)
    assert np.isclose(obb.extents[0], 1.0, atol=0.05)
    assert np.isclose(obb.extents[1], 0.4, atol=0.05)
    assert np.isclose(obb.extents[2], 0.2, atol=0.05)
