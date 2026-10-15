"""
ChronoScene-3D: Command Line Interface for Custom Phone Video Recordings.

Usage:
    python run_pipeline.py --video_a path/to/scan1.mp4 --video_b path/to/scan2.mp4 --output_dir output
"""

import argparse
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from chronoscene.data.video_loader import PhoneVideoLoader
from chronoscene.perception.depth_estimator import PyTorchDepthEstimator, MockDepthEstimator
from chronoscene.pipeline import ChronoScenePipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="ChronoScene-3D: Embodied 3D Scene Differencing from Dual Phone Videos"
    )
    parser.add_argument(
        "--video_a",
        type=str,
        required=True,
        help="Path to initial phone recording (Time t1, Reference Scan A)"
    )
    parser.add_argument(
        "--video_b",
        type=str,
        required=True,
        help="Path to subsequent phone recording (Time t2, Query Scan B)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save 3D viewer HTML, reports, and JSON action plans (default: output)"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=20,
        help="Maximum keyframes to extract per video scan (default: 20)"
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=68.0,
        help="Estimated horizontal camera Field of View in degrees (default: 68.0)"
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.02,
        help="Voxel downsampling resolution in meters (default: 0.02m = 2cm)"
    )
    parser.add_argument(
        "--static_threshold",
        type=float,
        default=0.04,
        help="Distance tolerance for static background alignment in meters (default: 0.04m)"
    )
    parser.add_argument(
        "--motion_threshold",
        type=float,
        default=0.05,
        help="Minimum displacement to classify an object as MOVED vs static in meters (default: 0.05m)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("🎥 ChronoScene-3D Video Processing CLI")
    print("=" * 70)
    print(f"• Video A (Scan 1): {args.video_a}")
    print(f"• Video B (Scan 2): {args.video_b}")
    print(f"• Output Directory: {args.output_dir}")

    loader = PhoneVideoLoader(target_width=640, fov_degrees=args.fov)

    print("\n[Step 1/3] Extracting keyframes and filtering motion blur from Video A...")
    scan_a = loader.load_video(args.video_a, max_keyframes=args.max_frames, scan_id="A")
    print(f"  ✓ Extracted {len(scan_a.rgb_frames)} keyframes (duration: {scan_a.duration_sec:.1f}s)")

    print("\n[Step 2/3] Extracting keyframes and filtering motion blur from Video B...")
    scan_b = loader.load_video(args.video_b, max_keyframes=args.max_frames, scan_id="B")
    print(f"  ✓ Extracted {len(scan_b.rgb_frames)} keyframes (duration: {scan_b.duration_sec:.1f}s)")

    print("\n[Step 3/3] Initializing Depth Estimator & ChronoScene-3D Pipeline...")
    depth_estimator = PyTorchDepthEstimator(model_type="MiDaS_small")

    pipeline = ChronoScenePipeline(
        output_dir=args.output_dir,
        voxel_size=args.voxel_size,
        static_threshold=args.static_threshold,
        motion_threshold=args.motion_threshold,
        depth_estimator=depth_estimator
    )

    start_time = time.time()
    results = pipeline.run(scan_a, scan_b)
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"✅ Finished in {elapsed:.2f} seconds!")
    print(f"🌐 Open 3D Viewer: file:///{os.path.abspath(results.viewer_html_path)}")
    print(f"📄 View Report:   file:///{os.path.abspath(results.report_md_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
