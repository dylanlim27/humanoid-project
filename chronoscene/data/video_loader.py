"""
Phone video loader with motion blur rejection and keyframe extraction.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import os
import cv2
import numpy as np

from ..geometry.transforms import CameraIntrinsics, CameraPose


@dataclass
class VideoScanData:
    """Extracted keyframes and camera metadata for a casual phone scan."""
    rgb_frames: List[np.ndarray]      # List of (H, W, 3) RGB images
    frame_indices: List[int]          # Original frame indices
    intrinsics: CameraIntrinsics      # Camera calibration
    fps: float
    duration_sec: float
    scan_id: str = "A"
    depth_maps: Optional[List[np.ndarray]] = None
    poses: Optional[List[CameraPose]] = None


class PhoneVideoLoader:
    """
    Ingests raw phone video (.mp4/.mov), filters motion blur, and samples high-quality keyframes.
    """

    def __init__(
        self,
        target_width: int = 640,
        fov_degrees: float = 68.0,
        min_blur_score: float = 40.0
    ):
        self.target_width = target_width
        self.fov_degrees = fov_degrees
        self.min_blur_score = min_blur_score

    def compute_sharpness(self, bgr_image: np.ndarray) -> float:
        """Compute image sharpness using variance of Laplacian."""
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def load_video(
        self,
        video_path: str,
        max_keyframes: int = 24,
        scan_id: str = "A"
    ) -> VideoScanData:
        """
        Extract keyframes from a video file, rejecting motion blur and downsampling for speed.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        duration_sec = total_frames / fps if total_frames > 0 else 0.0

        raw_frames = []
        frame_sharpness = []

        # Read frames with uniform stride
        stride = max(1, total_frames // (max_keyframes * 3)) if total_frames > 0 else 1
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                # Resize keeping aspect ratio
                h, w = frame.shape[:2]
                target_height = int(h * (self.target_width / w))
                resized = cv2.resize(frame, (self.target_width, target_height), interpolation=cv2.INTER_AREA)

                sharpness = self.compute_sharpness(resized)
                raw_frames.append((frame_idx, resized, sharpness))

            frame_idx += 1

        cap.release()

        if not raw_frames:
            raise ValueError(f"No frames could be extracted from: {video_path}")

        # Filter out severely blurred frames
        valid_frames = [f for f in raw_frames if f[2] >= self.min_blur_score]
        if len(valid_frames) < max_keyframes // 2:
            # If camera moved fast everywhere, sort by sharpness and pick the best
            valid_frames = sorted(raw_frames, key=lambda x: x[2], reverse=True)

        # Select evenly spaced subset from valid frames
        total_valid = len(valid_frames)
        sample_indices = np.linspace(0, total_valid - 1, min(max_keyframes, total_valid), dtype=int)
        selected = [valid_frames[i] for i in sample_indices]

        # Convert BGR to RGB
        rgb_frames = [cv2.cvtColor(f[1], cv2.COLOR_BGR2RGB) for f in selected]
        indices = [f[0] for f in selected]

        h, w = rgb_frames[0].shape[:2]
        intrinsics = CameraIntrinsics.from_fov(fov_deg=self.fov_degrees, width=w, height=h)

        return VideoScanData(
            rgb_frames=rgb_frames,
            frame_indices=indices,
            intrinsics=intrinsics,
            fps=fps,
            duration_sec=duration_sec,
            scan_id=scan_id
        )
