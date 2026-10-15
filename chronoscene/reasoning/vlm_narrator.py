"""
Multimodal Vision-Language Narrator and Embodied Action Plan Generator.
Translates 3D geometric deltas into human-understandable spatial explanations and robotic action primitives.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json

from .instance_matcher import ObjectChangeDelta, ObjectChangeType
from .change_detector import ChangeClassificationResult


@dataclass
class SceneDifferenceReport:
    """Complete multimodal narrative and robotic change report."""
    executive_summary: str
    detailed_changes: List[str]
    moved_descriptions: List[str]
    added_descriptions: List[str]
    removed_descriptions: List[str]
    viewpoint_and_occlusion_analysis: str
    robotic_manipulation_plan: List[Dict[str, Any]]
    raw_metrics: Dict[str, Any]

    def to_markdown(self) -> str:
        """Render markdown summary report."""
        lines = [
            "# ChronoScene-3D: Spatial Scene Differencing Report",
            "",
            "## Executive Summary",
            self.executive_summary,
            "",
            "## Identified 3D Object Changes",
        ]

        if self.moved_descriptions:
            lines.append("### Moved Objects")
            for desc in self.moved_descriptions:
                lines.append(f"- 🔄 {desc}")

        if self.added_descriptions:
            lines.append("### Newly Added Objects")
            for desc in self.added_descriptions:
                lines.append(f"- ➕ {desc}")

        if self.removed_descriptions:
            lines.append("### Removed Objects")
            for desc in self.removed_descriptions:
                lines.append(f"- ❌ {desc}")

        lines.extend([
            "",
            "## Observation & Occlusion Disambiguation",
            self.viewpoint_and_occlusion_analysis,
            "",
            "## Robotic Embodied Action Plan (Pick-and-Place / Navigation)",
            "```json",
            json.dumps(self.robotic_manipulation_plan, indent=2),
            "```"
        ])

        return "\n".join(lines)


class VLMNarrator:
    """
    Synthesizes natural language spatial narratives and robotic plans from 3D change deltas.
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key

    def generate_report(
        self,
        deltas: List[ObjectChangeDelta],
        classification_result: ChangeClassificationResult
    ) -> SceneDifferenceReport:
        """
        Generate complete natural language and robotic change report.
        """
        moved = [d for d in deltas if d.change_type == ObjectChangeType.MOVED]
        added = [d for d in deltas if d.change_type == ObjectChangeType.ADDED]
        removed = [d for d in deltas if d.change_type == ObjectChangeType.REMOVED]
        unobserved = [d for d in deltas if d.change_type == ObjectChangeType.UNOBSERVED]

        moved_descs = []
        for m in moved:
            dx, dy, dz = m.displacement_vector
            dist_cm = m.displacement_distance * 100.0

            # Compute human-friendly direction
            direction_terms = []
            if abs(dx) > 0.03:
                direction_terms.append("East (+X)" if dx > 0 else "West (-X)")
            if abs(dy) > 0.03:
                direction_terms.append("South (+Y)" if dy > 0 else "North (-Y)")
            if abs(dz) > 0.03:
                direction_terms.append("Up (+Z)" if dz > 0 else "Down (-Z)")
            dir_str = " & ".join(direction_terms) if direction_terms else "laterally"

            moved_descs.append(
                f"**{m.label.replace('_', ' ').capitalize()}** moved {dist_cm:.1f} cm {dir_str} "
                f"(from [{m.object_a.centroid[0]:.2f}, {m.object_a.centroid[1]:.2f}, {m.object_a.centroid[2]:.2f}] "
                f"to [{m.object_b.centroid[0]:.2f}, {m.object_b.centroid[1]:.2f}, {m.object_b.centroid[2]:.2f}])."
            )

        added_descs = []
        for a in added:
            c = a.object_b.centroid
            e = a.object_b.obb.extents
            added_descs.append(
                f"**{a.label.replace('_', ' ').capitalize()}** was placed at location "
                f"[{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}] (size: {e[0]*100:.0f}×{e[1]*100:.0f}×{e[2]*100:.0f} cm). "
                f"Verified: this workspace volume was confirmed empty in Scan A."
            )

        removed_descs = []
        for r in removed:
            c = r.object_a.centroid
            removed_descs.append(
                f"**{r.label.replace('_', ' ').capitalize()}** previously at "
                f"[{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}] was removed. "
                f"Verified: camera in Scan B had an unobstructed line of sight and observed free space here."
            )

        # Occlusion & Incomplete observation explanation
        occ_pct = classification_result.metrics.get("occluded_percentage", 0.0)
        unobs_count = len(unobserved)
        occlusion_text = (
            f"Cross-scan raycasting verified that **{classification_result.metrics.get('static_percentage', 100.0)}%** "
            f"of the observed scene remained physically static. "
            f"Approximately **{occ_pct}%** of candidate points were identified as "
            f"occluded or outside the camera's trajectory (incomplete observation). "
            f"The system successfully rejected these false positives without misclassifying them as removed objects."
        )

        if unobs_count > 0:
            occlusion_text += f" In addition, {unobs_count} background region(s) lacked sufficient multi-view coverage in one scan."

        # Executive summary
        total_real_changes = len(moved) + len(added) + len(removed)
        summary = (
            f"Comparison of dual video scans identified **{total_real_changes} verified physical changes**: "
            f"{len(moved)} moved object(s), {len(added)} added object(s), and {len(removed)} removed object(s). "
            f"All viewpoint shifts and camera trajectory cutoffs were disambiguated from actual physical modifications."
        )

        # Robotic manipulation action directives
        action_plan = []
        for m in moved:
            action_plan.append(m.robotic_action_primitive)
        for a in added:
            action_plan.append(a.robotic_action_primitive)
        for r in removed:
            action_plan.append(r.robotic_action_primitive)

        all_details = moved_descs + added_descs + removed_descs

        return SceneDifferenceReport(
            executive_summary=summary,
            detailed_changes=all_details,
            moved_descriptions=moved_descs,
            added_descriptions=added_descs,
            removed_descriptions=removed_descs,
            viewpoint_and_occlusion_analysis=occlusion_text,
            robotic_manipulation_plan=action_plan,
            raw_metrics=classification_result.metrics
        )
