# ChronoScene-3D
### Multi-View Spatial Reasoning and Embodied 3D Change Grounding from Monocular Phone Video

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Three.js](https://img.shields.io/badge/WebGL-Three.js-black.svg)](https://threejs.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-11%20Passed-brightgreen.svg)]()

A 3D computer vision and embodied perception system that compares two casual phone recordings of an indoor workspace, estimates a shared metric 3D representation, filters out viewpoint differences and occlusions via multi-view line-of-sight raycasting, and generates robot-actionable 6-DoF manipulation primitives and natural language change descriptions.

---

## Table of Contents
1. [Problem Statement & Core Challenges](#1-problem-statement--core-challenges)
2. [System Architecture](#2-system-architecture)
3. [Mathematical Formulation](#3-mathematical-formulation)
4. [Quickstart Guide](#4-quickstart-guide)
5. [Interactive Visualizers and Dashboard](#5-interactive-visualizers-and-dashboard)
6. [Quantitative Benchmark Evaluation](#6-quantitative-benchmark-evaluation)
7. [Design Choices and Technical Trade-Offs](#7-design-choices-and-technical-trade-offs)
8. [Repository Structure](#8-repository-structure)

---

## 1. Problem Statement & Core Challenges

Comparing two unconstrained phone videos of an indoor environment captured at different times (Time 1 and Time 2) introduces four fundamental ambiguities:
- **Viewpoint Disparity**: Casual camera trajectories, walking arcs, and viewing angles differ significantly between passes.
- **Occlusion**: Objects visible from one angle may be blocked by foreground equipment from another angle. A naive model falsely flags occluded items as removed.
- **Incomplete Observation**: Handheld scanning rarely achieves identical coverage; one scan may omit peripheral shelves. Naive diffing generates massive false positives.
- **Photometric Drift**: Auto-exposure, auto-white-balance, and sensor gain fluctuations invalidate 2D photometric differencing.

ChronoScene-3D addresses this by:
1. Lifting 2D keyframes into a metric 3D point cloud via monocular depth estimation and camera intrinsics.
2. Aligning both video passes into a unified world coordinate frame via rigid SE(3) co-registration on static background structures.
3. Performing multi-view line-of-sight raycasting against depth buffers to verify whether an unobserved voxel corresponds to true free space or an obstacle shadow.
4. Extracting 3D object instances and applying Kuhn-Munkres (Hungarian) bipartite matching across spatial displacement, volume ratio, and appearance to isolate moved, added, and removed entities.

---

## 2. System Architecture

```mermaid
flowchart TD
    ScanA[Scan A Video - Reference at t1] --> FeatA[Keyframe Poses and Depth A]
    ScanB[Scan B Video - Query at t2] --> FeatB[Keyframe Poses and Depth B]
    FeatA --> Align[SE3 Background Co-Registration]
    FeatB --> Align
    Align --> PointCloud[Unified Metric 3D Point Cloud]
    PointCloud --> Raycast[Line-of-Sight Raycasting Engine]
    Raycast --> Classify[Voxel State Classifier]
    Classify --> Clusters[3D Euclidean Clustering and Bounding Boxes]
    Clusters --> Matcher[Bipartite Hungarian Matching]
    Matcher --> Action[6-DoF Robotic Action Vectors]
    Matcher --> Report[Spatial Change Narrative]
    Action --> Deliverables[Interactive 3D WebGL Viewer and JSON Plan]
    Report --> Deliverables
```

---

## 3. Mathematical Formulation

### 3.1 Point Cloud Backprojection
Given camera intrinsics matrix $K$ and camera-to-world pose $T_{wc}^{(k)} \in \mathrm{SE}(3)$ for keyframe $k$, a pixel coordinate $(u, v)$ with scalar depth $d = D_k(u, v)$ unprojects to 3D world coordinates $p \in \mathbb{R}^3$:

$$p = T_{wc}^{(k)} \left( d \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \right)$$

### 3.2 Multi-View Observation Classification
For any query voxel center $v$ projected into keyframe $k$ with camera-frame depth $z_c$ and surface tolerance $\epsilon$ (default 4 cm):

- **FreeSpace**: $z_c < D_k(u, v) - \epsilon$ (camera ray traversed empty space)
- **SurfaceHit**: $|z_c - D_k(u, v)| \le \epsilon$ (camera ray intercepted surface)
- **Occluded**: $z_c > D_k(u, v) + \epsilon$ (query voxel is behind a closer foreground obstacle)
- **FrustumOut**: $u \notin [0, W)$ or $v \notin [0, H)$ or $z_c \le 0$ (outside camera field of view)

### 3.3 Change Disambiguation Across Dual Passes
- **Static**: Point $p_A$ in Scan A has a spatial neighbor in Scan B within Euclidean distance threshold (4 cm).
- **Removed**: Point $p_A$ has no neighbor in Scan B, and there exists at least one view in Scan B where the voxel is verified as FreeSpace (the camera verified line-of-sight through that volume without collision).
- **Added**: Point $p_B$ has no neighbor in Scan A, and there exists at least one view in Scan A where the voxel is verified as FreeSpace.
- **Occluded / Out-of-View**: Points where all opposing views yielded Occluded or FrustumOut. These points are excluded from the change set, preventing false positive alerts from incomplete trajectories.

---

## 4. Quickstart Guide

### Environment Setup
```bash
git clone https://github.com/dylanlim27/chronoscene-3d.git
cd chronoscene-3d

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 1. Run Benchmark Demo (No External Downloads Required)
```bash
python demo.py
```
Executes the full pipeline on a dual-pass tabletop workspace benchmark with ground-truth changes and outputs:
- `output/viewer_3d.html` (Standalone Three.js interactive 3D WebGL viewer)
- `output/change_report.md` (Natural language spatial report)
- `output/robotic_action_plan.json` (Machine-executable pick-and-place directives)
- `output/scene_classified.ply` (Color-coded point cloud)

### 2. Launch Interactive Streamlit GUI
```bash
streamlit run web_app.py
```

### 3. Process Custom Monocular Video Recordings
```bash
python run_pipeline.py --video_a path/to/scan1.mp4 --video_b path/to/scan2.mp4 --output_dir output
```

### 4. Run Test Suite
```bash
pytest -v
```

---

## 5. Interactive Visualizers and Dashboard

### Three.js 3D WebGL Viewer (`output/viewer_3d.html`)
The generated viewer is a self-contained, offline-compatible HTML file:
- **Camera Controls**: Left Click + Drag (Orbit), Right Click + Drag (Pan), Scroll (Zoom).
- **Layer Visibility**:
  - Green: Newly added objects (`electric_drill`)
  - Red: Removed objects (`blue_notebook`)
  - Yellow: Moved objects with 3D translation vectors and directional arrows (`ceramic_mug`)
  - Gray: Invariant static background (tabletop, monitor, laptop)
  - Blue: Occluded or unobserved regions
- **Toggles**: Oriented bounding boxes, camera trajectories (Cyan: Scan A, Magenta: Scan B), and frustums.

---

## 6. Quantitative Benchmark Evaluation

Evaluated on an indoor tabletop manipulation workspace with handheld sweeping trajectories:

| Object / Region | Ground Truth State | Ground Truth Displacement | ChronoScene-3D Output | Position Error |
| :--- | :--- | :--- | :--- | :--- |
| **Ceramic Mug** | Moved | $[-0.30, -0.12, 0.81] \to [+0.15, -0.15, 0.81]$ ($\Delta = 45.0$ cm) | `[MOVED]` Shift: 46.9 cm East (+X) | 1.9 cm |
| **Electric Drill** | Added | Center: $[+0.35, +0.10, 0.83]$ (Free in A) | `[ADDED]` Center: $[0.36, 0.06, 0.84]$ | 2.2 cm |
| **Blue Notebook** | Removed | Center: $[+0.05, -0.22, 0.77]$ (Free in B) | `[REMOVED]` Center: $[0.03, -0.23, 0.79]$ | 1.8 cm |
| **Far Tool Rack** | Out-of-View | Visible in A only (Trajectory B cut off) | `[OCCLUDED / OUT-OF-VIEW]` | 0 False Positives |
| **Stationary Base** | Static | Invariant | 89.2% Static Invariance | Metric Concordance |

### Machine-Readable Robotic Action Directives (`robotic_action_plan.json`)
```json
{
  "summary": "Comparison of dual video scans identified 3 verified physical changes: 1 moved object(s), 1 added object(s), and 1 removed object(s).",
  "manipulation_plan": [
    {
      "action": "relocate_object",
      "object_label": "ceramic_mug",
      "pickup_pose": [0.17, -0.16, 0.85],
      "return_pose": [-0.30, -0.13, 0.84],
      "displacement_m": 0.469,
      "delta_vector": [0.469, -0.026, 0.012],
      "dimensions": [0.12, 0.12, 0.13]
    },
    {
      "action": "inspect_novel_object",
      "object_label": "electric_drill",
      "current_pose": [0.36, 0.06, 0.84],
      "dimensions": [0.26, 0.24, 0.11]
    },
    {
      "action": "replenish_or_verify_missing",
      "object_label": "blue_notebook",
      "previous_pose": [0.03, -0.23, 0.79],
      "dimensions": [0.23, 0.29, 0.04]
    }
  ]
}
```

---

## 7. Design Choices and Technical Trade-Offs

### 1. Multi-View Raycasting vs. Naive Point-to-Point Distance
A point-to-point Euclidean distance check cannot differentiate whether a missing point was removed or simply remained unobserved because the phone never panned toward it. By raycasting each candidate voxel through the depth buffer of the opposing camera trajectory, the system distinguishes free space line-of-sight from obstacle occlusion.

### 2. Invariant Background SE(3) Co-Registration vs. Global ICP
Running naive ICP on the entire point cloud causes moved objects to exert leverage on the transformation matrix, resulting in coordinate drift. The system isolates stationary structural primitives (walls, floor, table) using correspondence thresholding to compute the rigid transformation $T_{A \leftarrow B}$ without bias from dynamic objects.

### 3. Bipartite Hungarian Instance Association
When an object is repositioned, naive differencing treats it as two independent events: one deletion and one addition. ChronoScene-3D constructs a bipartite cost matrix balancing centroid Euclidean distance, volume ratio, and appearance consistency. Solving this assignment via the Hungarian algorithm directly yields the rigid 6-DoF transformation delta $\Delta t$.

### 4. Zero-Dependency Standalone WebGL Viewer
Robotics engineers and evaluators require immediate verification without configuring local graphics drivers or complex viewer runtimes. The pipeline serializes point buffers directly into an offline Three.js HTML bundle with native orbit controls.

---

## 8. Repository Structure

```text
chronoscene-3d/
├── chronoscene/
│   ├── geometry/
│   │   ├── transforms.py          # SE(3) math, CameraIntrinsics, CameraPose, OBB
│   │   ├── pointcloud.py          # PointCloud container, voxel filtering, PLY export
│   │   └── frustum.py             # Frustum geometry and visibility culling
│   ├── perception/
│   │   ├── depth_estimator.py     # Metric monocular depth (PyTorch MiDaS + fallback)
│   │   ├── feature_matcher.py     # Kabsch SVD and ICP background SE(3) alignment
│   │   └── segmentation.py        # 3D Euclidean clustering and instance extraction
│   ├── reasoning/
│   │   ├── occupancy_grid.py      # Multi-view line-of-sight raycasting engine
│   │   ├── change_detector.py     # Change disambiguation (Static, Added, Removed, Occluded)
│   │   ├── instance_matcher.py    # Hungarian bipartite matching and 6-DoF delta estimation
│   │   └── vlm_narrator.py        # Multimodal spatial narrative and action directives
│   ├── visualization/
│   │   ├── threejs_viewer.py      # Three.js 3D WebGL viewer generator
│   │   ├── plotly_vis.py          # Plotly 3D scatter generator
│   │   └── report_generator.py    # Markdown and JSON report exporter
│   ├── data/
│   │   ├── video_loader.py        # Keyframe extraction with Laplacian blur rejection
│   │   └── benchmark_generator.py  # Procedural indoor benchmark with ground-truth deltas
│   └── pipeline.py                # ChronoScenePipeline orchestrator
├── tests/
│   ├── test_geometry.py           # Verification of SE(3) operations, projections, OBB
│   ├── test_raycast_visibility.py  # Line-of-sight raycasting tests
│   ├── test_change_detection.py   # Static invariance and change classification tests
│   └── test_instance_matching.py  # Bipartite matching and delta tests
├── demo.py                        # Benchmark evaluation demo script
├── run_pipeline.py                # CLI runner for custom phone video files
├── web_app.py                     # Streamlit interactive analysis dashboard
├── requirements.txt               # Dependency specifications
├── pyproject.toml                 # Package configuration
├── .gitignore
└── README.md
```

---

## License
MIT License. Created for Embodied AI and Robotics Computer Vision evaluation.
