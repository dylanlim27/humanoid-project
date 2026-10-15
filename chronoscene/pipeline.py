"""
ChronoScenePipeline: End-to-end pipeline orchestrator for 3D scene differencing and spatial reasoning.
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from .geometry.transforms import CameraIntrinsics, CameraPose, OrientedBoundingBox3D
from .geometry.pointcloud import PointCloud, unproject_depth_map
from .perception.depth_estimator import DepthEstimator, MockDepthEstimator
from .perception.feature_matcher import CrossScanAligner
from .perception.segmentation import EuclideanClusterer3D, ObjectInstance3D
from .reasoning.occupancy_grid import MultiViewRaycastEngine, VoxelState
from .reasoning.change_detector import ChangeDetector3D, ChangeClassificationResult
from .reasoning.instance_matcher import InstanceMatcher3D, ObjectChangeDelta, ObjectChangeType
from .reasoning.vlm_narrator import VLMNarrator, SceneDifferenceReport
from .visualization.threejs_viewer import save_interactive_3d_viewer
from .data.video_loader import VideoScanData, PhoneVideoLoader


@dataclass
class PipelineOutput:
    """Consolidated artifacts produced by a complete pipeline run."""
    classification: ChangeClassificationResult
    object_deltas: List[ObjectChangeDelta]
    report: SceneDifferenceReport
    viewer_html_path: str
    report_md_path: str
    action_json_path: str
    aligned_poses_a: List[CameraPose]
    aligned_poses_b: List[CameraPose]
    intrinsics: CameraIntrinsics


class ChronoScenePipeline:
    """
    Main entry point for comparing two phone scans and generating actionable 3D changes.
    """

    def __init__(
        self,
        output_dir: str = "output",
        voxel_size: float = 0.02,
        static_threshold: float = 0.04,
        motion_threshold: float = 0.05,
        depth_estimator: Optional[DepthEstimator] = None,
        aligner: Optional[CrossScanAligner] = None,
        narrator: Optional[VLMNarrator] = None
    ):
        self.output_dir = output_dir
        self.voxel_size = voxel_size
        self.static_threshold = static_threshold
        self.motion_threshold = motion_threshold

        self.depth_estimator = depth_estimator or MockDepthEstimator()
        self.aligner = aligner or CrossScanAligner()
        self.raycast_engine = MultiViewRaycastEngine(surface_thickness_epsilon=0.04)
        self.change_detector = ChangeDetector3D(
            static_distance_threshold=static_threshold,
            raycast_engine=self.raycast_engine
        )
        self.clusterer = EuclideanClusterer3D(cluster_tolerance=0.06, min_cluster_size=30)
        self.matcher = InstanceMatcher3D(motion_threshold=motion_threshold)
        self.narrator = narrator or VLMNarrator()

        os.makedirs(self.output_dir, exist_ok=True)

    def _lift_scan_to_pointcloud(
        self,
        scan: VideoScanData,
        sample_stride: int = 1
    ) -> PointCloud:
        """Unproject multi-view depth maps into a combined point cloud in the scan's world frame."""
        clouds = []
        for i in range(0, len(scan.rgb_frames), sample_stride):
            pose = scan.poses[i]
            depth = scan.depth_maps[i]
            rgb = scan.rgb_frames[i]
            pcd = unproject_depth_map(
                depth_map=depth,
                intrinsics=scan.intrinsics,
                pose=pose,
                rgb_image=rgb,
                stride=3  # Pixel subsampling stride
            )
            clouds.append(pcd)

        merged = PointCloud.merge(clouds, scan_id=scan.scan_id)
        return merged.downsample(voxel_size=self.voxel_size)

    def _infer_semantic_label(
        self,
        centroid: np.ndarray,
        color: np.ndarray,
        extents: np.ndarray,
        candidate_labels: Optional[List[str]] = None
    ) -> str:
        """Infer semantic label using color, shape, and prior labels."""
        r, g, b = color[0], color[1], color[2]

        if candidate_labels:
            for l in candidate_labels:
                l_lower = l.lower()
                if "mug" in l_lower and (r > 0.6 and g > 0.6 and b > 0.6):
                    return l
                if ("drill" in l_lower or "tool" in l_lower) and (r > 0.6 and g > 0.5 and b < 0.4):
                    return l
                if "notebook" in l_lower and (b > 0.5 and r < 0.4):
                    return l
                if "rack" in l_lower and (centroid[0] < -0.5):
                    return l

        # Physical attribute heuristics
        if r > 0.65 and g > 0.50 and b < 0.35:
            return "cordless_drill"
        elif b > 0.45 and r < 0.35:
            return "blue_notebook"
        elif r > 0.65 and g > 0.65 and b > 0.65:
            return "ceramic_mug"
        elif max(extents) > 0.35:
            return "large_apparatus"
        return "tabletop_object"

    def _extract_and_match_changes(
        self,
        classification: ChangeClassificationResult,
        semantic_labels_a: Optional[List[str]] = None,
        semantic_labels_b: Optional[List[str]] = None
    ) -> List[ObjectChangeDelta]:
        """
        Cluster change subclouds and associate instances via Hungarian optimization.
        """
        change_clusterer = EuclideanClusterer3D(cluster_tolerance=0.08, min_cluster_size=20)

        rem_clusters = change_clusterer.cluster(classification.removed_pcd)
        add_clusters = change_clusterer.cluster(classification.added_pcd)
        occ_clusters = change_clusterer.cluster(classification.occluded_pcd)

        rem_objects: List[ObjectInstance3D] = []
        for i, c in enumerate(rem_clusters):
            obb = OrientedBoundingBox3D.from_points(c.points)
            label = self._infer_semantic_label(c.centroid, c.mean_color, obb.extents, semantic_labels_a)
            obb.label = label
            rem_objects.append(ObjectInstance3D(
                id=f"rem_{i+1:02d}",
                label=label,
                centroid=c.centroid,
                obb=obb,
                points=c.points,
                mean_color=c.mean_color,
                scan_id="A"
            ))

        add_objects: List[ObjectInstance3D] = []
        for j, c in enumerate(add_clusters):
            obb = OrientedBoundingBox3D.from_points(c.points)
            label = self._infer_semantic_label(c.centroid, c.mean_color, obb.extents, semantic_labels_b)
            obb.label = label
            add_objects.append(ObjectInstance3D(
                id=f"add_{j+1:02d}",
                label=label,
                centroid=c.centroid,
                obb=obb,
                points=c.points,
                mean_color=c.mean_color,
                scan_id="B"
            ))

        # Perform Hungarian association between candidate removed and added objects
        deltas: List[ObjectChangeDelta] = []
        matched_rem = set()
        matched_add = set()

        if len(rem_objects) > 0 and len(add_objects) > 0:
            # Build pairwise similarity cost matrix
            from scipy.optimize import linear_sum_assignment
            cost_matrix = np.full((len(rem_objects), len(add_objects)), 1e6, dtype=np.float64)

            for i, ro in enumerate(rem_objects):
                for j, ao in enumerate(add_objects):
                    dist = float(np.linalg.norm(ro.centroid - ao.centroid))
                    if dist < self.motion_threshold or dist > 1.2:
                        continue  # Too close or too far to be a plausible move

                    # Volume ratio cost
                    vol_r = max(ro.volume, 1e-4) / max(ao.volume, 1e-4)
                    vol_cost = abs(1.0 - (vol_r if vol_r <= 1.0 else 1.0 / vol_r))

                    # Color similarity check
                    col_cost = float(np.linalg.norm(ro.mean_color - ao.mean_color))
                    if col_cost > 0.45 and (ro.label != ao.label or ro.label == "tabletop_object"):
                        continue  # Distinct physical appearances cannot be the same moved object

                    # Label match bonus
                    label_bonus = -0.5 if ro.label == ao.label and ro.label != "tabletop_object" else 0.0

                    cost_matrix[i, j] = dist + (0.5 * vol_cost) + (0.8 * col_cost) + label_bonus

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 2.0:  # Valid matched move
                    ro = rem_objects[r]
                    ao = add_objects[c]
                    matched_rem.add(r)
                    matched_add.add(c)
                    disp_vec = ao.centroid - ro.centroid
                    disp_dist = float(np.linalg.norm(disp_vec))

                    label = ao.label if ao.label != "tabletop_object" else ro.label
                    deltas.append(ObjectChangeDelta(
                        change_type=ObjectChangeType.MOVED,
                        object_a=ro,
                        object_b=ao,
                        displacement_vector=disp_vec,
                        displacement_distance=disp_dist,
                        volume_ratio=float(ro.volume / max(ao.volume, 1e-4)),
                        label=label
                    ))

        # Unmatched in rem -> Truly REMOVED
        for i, ro in enumerate(rem_objects):
            if i not in matched_rem:
                deltas.append(ObjectChangeDelta(
                    change_type=ObjectChangeType.REMOVED,
                    object_a=ro,
                    label=ro.label
                ))

        # Unmatched in add -> Truly ADDED
        for j, ao in enumerate(add_objects):
            if j not in matched_add:
                deltas.append(ObjectChangeDelta(
                    change_type=ObjectChangeType.ADDED,
                    object_b=ao,
                    label=ao.label
                ))

        # Large occluded/unobserved clusters -> UNOBSERVED
        for k, oc in enumerate(occ_clusters):
            if len(oc.points) >= 40:
                obb = OrientedBoundingBox3D.from_points(oc.points)
                label = "unobserved_shelf_region"
                if oc.centroid[0] < -0.5:
                    label = "far_tool_rack"
                deltas.append(ObjectChangeDelta(
                    change_type=ObjectChangeType.UNOBSERVED,
                    object_a=ObjectInstance3D(
                        id=f"unobs_{k+1:02d}",
                        label=label,
                        centroid=oc.centroid,
                        obb=obb,
                        points=oc.points,
                        mean_color=oc.mean_color,
                        scan_id="A"
                    ),
                    label=label
                ))

        return deltas

    def run(
        self,
        scan_a: VideoScanData,
        scan_b: VideoScanData,
        semantic_labels_a: Optional[List[str]] = None,
        semantic_labels_b: Optional[List[str]] = None
    ) -> PipelineOutput:
        """
        Execute full 3D spatial reasoning and change detection.
        """
        print("[ChronoScene-3D] 1/6: Ensuring depth maps and poses...")
        if scan_a.depth_maps is None:
            scan_a.depth_maps = [self.depth_estimator.estimate_depth(f) for f in scan_a.rgb_frames]
        if scan_b.depth_maps is None:
            scan_b.depth_maps = [self.depth_estimator.estimate_depth(f) for f in scan_b.rgb_frames]

        print("[ChronoScene-3D] 2/6: Lifting 2D frames to 3D point clouds...")
        pcd_a = self._lift_scan_to_pointcloud(scan_a)
        pcd_b = self._lift_scan_to_pointcloud(scan_b)
        print(f"  -> Scan A Point Cloud: {len(pcd_a)} points")
        print(f"  -> Scan B Point Cloud: {len(pcd_b)} points")

        print("[ChronoScene-3D] 3/6: Co-registering Scan B into Scan A coordinate frame...")
        pcd_b_aligned, poses_b_aligned, T_a_b = self.aligner.align_scans(
            pcd_a=pcd_a,
            pcd_b=pcd_b,
            poses_b=scan_b.poses
        )

        print("[ChronoScene-3D] 4/6: Executing line-of-sight raycast & visibility disambiguation...")
        classification = self.change_detector.detect_changes(
            pcd_a=pcd_a,
            pcd_b=pcd_b_aligned,
            poses_a=scan_a.poses,
            depth_maps_a=scan_a.depth_maps,
            poses_b=poses_b_aligned,
            depth_maps_b=scan_b.depth_maps,
            intrinsics=scan_a.intrinsics
        )
        print(f"  -> Static: {classification.metrics['static_percentage']}%")
        print(f"  -> Added points: {classification.metrics['added_points']}")
        print(f"  -> Removed points: {classification.metrics['removed_points']}")
        print(f"  -> Occluded/Unobserved: {classification.metrics['occluded_percentage']}%")

        print("[ChronoScene-3D] 5/6: Extracting 3D object instances & Hungarian matching...")
        object_deltas = self._extract_and_match_changes(
            classification=classification,
            semantic_labels_a=semantic_labels_a,
            semantic_labels_b=semantic_labels_b
        )

        print(f"  -> Identified {len(object_deltas)} object change deltas.")
        for d in object_deltas:
            if d.change_type == ObjectChangeType.MOVED:
                print(f"     [MOVED] {d.label}: displacement {d.displacement_distance*100:.1f} cm")
            elif d.change_type == ObjectChangeType.ADDED:
                print(f"     [ADDED] {d.label}")
            elif d.change_type == ObjectChangeType.REMOVED:
                print(f"     [REMOVED] {d.label}")
            elif d.change_type == ObjectChangeType.UNOBSERVED:
                print(f"     [UNOBSERVED / OCCLUDED] {d.label}")

        print("[ChronoScene-3D] 6/6: Synthesizing multimodal narrative and export deliverables...")
        report = self.narrator.generate_report(object_deltas, classification)

        # Save deliverables
        viewer_path = os.path.join(self.output_dir, "viewer_3d.html")
        report_md_path = os.path.join(self.output_dir, "change_report.md")
        action_json_path = os.path.join(self.output_dir, "robotic_action_plan.json")
        ply_path = os.path.join(self.output_dir, "scene_classified.ply")

        save_interactive_3d_viewer(
            classification=classification,
            deltas=object_deltas,
            report=report,
            poses_a=scan_a.poses,
            poses_b=poses_b_aligned,
            intrinsics=scan_a.intrinsics,
            output_html_path=viewer_path
        )

        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())

        with open(action_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": report.executive_summary,
                "metrics": classification.metrics,
                "manipulation_plan": report.robotic_manipulation_plan,
                "object_deltas": [
                    {
                        "label": d.label,
                        "type": d.change_type.value,
                        "displacement_m": round(float(d.displacement_distance), 4),
                        "delta_vector": [round(float(v), 4) for v in d.displacement_vector]
                    }
                    for d in object_deltas
                ]
            }, f, indent=2)

        # Merge colored point clouds for PLY export
        all_vis_pcd = PointCloud.merge([
            classification.static_pcd,
            classification.added_pcd,
            classification.removed_pcd,
            classification.occluded_pcd
        ])
        all_vis_pcd.to_ply(ply_path)

        print(f"\n[ChronoScene-3D] Execution Complete! Artifacts written to '{self.output_dir}/':")
        print(f"  - Interactive 3D Viewer: {viewer_path}")
        print(f"  - Spatial Narrative:     {report_md_path}")
        print(f"  - Robotic Action Plan:   {action_json_path}")
        print(f"  - Color-coded PLY Cloud: {ply_path}")

        return PipelineOutput(
            classification=classification,
            object_deltas=object_deltas,
            report=report,
            viewer_html_path=viewer_path,
            report_md_path=report_md_path,
            action_json_path=action_json_path,
            aligned_poses_a=scan_a.poses,
            aligned_poses_b=poses_b_aligned,
            intrinsics=scan_a.intrinsics
        )
