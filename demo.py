"""
ChronoScene-3D: Zero-Dependency Interactive Benchmark Demo.

Runs a complete end-to-end evaluation comparing two simulated casual phone recordings
of an indoor tabletop manipulation workspace with ground-truth changes:
  1. Ceramic mug moved 45cm across the desk
  2. Electric drill added to the workspace
  3. Blue notebook removed from the tabletop
  4. Tool rack on far left: only visible in Scan A, occluded/cut-off in Scan B (verified as unobserved, not removed!)

Outputs:
  - output/viewer_3d.html        -> Standalone interactive Three.js 3D WebGL viewer
  - output/change_report.md      -> Natural language spatial reasoning report
  - output/robotic_action_plan.json -> Machine-readable robotic pick-and-place directives
  - output/scene_classified.ply  -> Color-coded 3D point cloud
"""

import sys
import os
import time

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure package root is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from chronoscene.data.benchmark_generator import IndoorBenchmarkGenerator
from chronoscene.pipeline import ChronoScenePipeline
from chronoscene.reasoning.occupancy_grid import VoxelState
from chronoscene.reasoning.instance_matcher import ObjectChangeType


def run_benchmark_demo():
    print("=" * 75)
    print("🚀 ChronoScene-3D: Embodied 3D Scene Differencing & Spatial Change Grounding")
    print("=" * 75)

    start_time = time.time()

    # 1. Synthesize benchmark pair
    print("\n[Step 1] Synthesizing realistic casual phone sweeps with ground-truth indoor changes...")
    benchmark_gen = IndoorBenchmarkGenerator(width=480, height=360, fov_deg=65.0)
    scan_a, scan_b, gt_deltas = benchmark_gen.generate_benchmark_pair(num_frames=12)

    print(f"  ✓ Scan A: {len(scan_a.rgb_frames)} keyframes (Trajectory: left arc)")
    print(f"  ✓ Scan B: {len(scan_b.rgb_frames)} keyframes (Trajectory: right arc)")
    print("  ✓ Ground truth scene modifications:")
    for gt in gt_deltas:
        print(f"    - {gt.object_name}: {gt.true_state.value.upper()}" +
              (f" (displacement: {gt.displacement_m*100:.0f}cm)" if gt.displacement_m > 0 else ""))

    # 2. Run ChronoScene-3D Pipeline
    print("\n[Step 2] Executing ChronoScene-3D Multi-View Spatial Reasoning Pipeline...")
    pipeline = ChronoScenePipeline(
        output_dir="output",
        voxel_size=0.02,        # 2 cm resolution
        static_threshold=0.04,  # 4 cm tolerance
        motion_threshold=0.06   # 6 cm movement threshold
    )

    labels_a = ["laptop", "monitor", "ceramic_mug", "blue_notebook", "far_tool_rack"]
    labels_b = ["laptop", "monitor", "ceramic_mug", "electric_drill", "far_tool_rack"]

    results = pipeline.run(
        scan_a=scan_a,
        scan_b=scan_b,
        semantic_labels_a=labels_a,
        semantic_labels_b=labels_b
    )

    elapsed = time.time() - start_time

    # 3. Print Results Summary
    print("\n" + "=" * 75)
    print("📊 BENCHMARK EVALUATION & REASONING RESULTS")
    print("=" * 75)
    print(f"⚡ Total Processing Time: {elapsed:.2f} seconds")
    print(f"📈 Static Invariance Ratio: {results.classification.metrics['static_percentage']}%")
    print(f"👁️ Occluded / Unobserved:   {results.classification.metrics['occluded_percentage']}%")
    print(f"\n📝 Natural Language Narrative:\n")
    print(results.report.executive_summary)
    print("\nDetailed Findings:")
    for ch in results.report.detailed_changes:
        # Strip markdown bolding for clean console printing
        clean_text = ch.replace("**", "")
        print(f"  • {clean_text}")

    print("\n🤖 Embodied Robotic Directives:")
    for act in results.report.robotic_manipulation_plan:
        print(f"  • Action: {act['action']:<24} Target: {act['object_label']:<16} " +
              (f"Shift: {act['displacement_m']}m" if 'displacement_m' in act else ""))

    print("\n" + "=" * 75)
    print("✨ DELIVERABLES READY")
    print("=" * 75)
    print(f"🌐 Standalone 3D Web Viewer: file:///{os.path.abspath(results.viewer_html_path)}")
    print(f"📄 Markdown Report:          file:///{os.path.abspath(results.report_md_path)}")
    print(f"💾 Robotic Action JSON:      file:///{os.path.abspath(results.action_json_path)}")
    print("\n👉 Open 'output/viewer_3d.html' in your web browser to explore the 3D scene!")
    print("=" * 75)


if __name__ == "__main__":
    run_benchmark_demo()
