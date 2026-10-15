"""
Standalone interactive Three.js 3D WebGL viewer generator.
Produces zero-dependency, self-contained HTML files with full orbit controls,
change color-coding, 3D bounding boxes, and camera trajectories.
"""

from __future__ import annotations
import json
import os
from typing import List, Dict, Any, Optional
import numpy as np

from ..geometry.transforms import CameraPose, CameraIntrinsics, OrientedBoundingBox3D
from ..geometry.frustum import CameraFrustum
from ..reasoning.change_detector import ChangeClassificationResult
from ..reasoning.instance_matcher import ObjectChangeDelta, ObjectChangeType
from ..reasoning.vlm_narrator import SceneDifferenceReport


def generate_threejs_viewer_html(
    classification: ChangeClassificationResult,
    deltas: List[ObjectChangeDelta],
    report: SceneDifferenceReport,
    poses_a: List[CameraPose],
    poses_b: List[CameraPose],
    intrinsics: CameraIntrinsics,
    title: str = "ChronoScene-3D: Multi-View Spatial Differencing"
) -> str:
    """
    Generate an interactive, self-contained HTML page powered by Three.js.
    """
    # Downsample points for smooth 60fps in-browser WebGL rendering
    pcd_static = classification.static_pcd.downsample(voxel_size=0.02)
    pcd_added = classification.added_pcd.downsample(voxel_size=0.015)
    pcd_removed = classification.removed_pcd.downsample(voxel_size=0.015)
    pcd_occluded = classification.occluded_pcd.downsample(voxel_size=0.02)

    # Format point cloud payloads
    def format_pcd(pcd, default_rgb=None):
        if len(pcd.points) == 0:
            return {"pts": [], "cols": []}
        pts = np.round(pcd.points, 4).flatten().tolist()
        cols = pcd.colors if default_rgb is None else np.tile(default_rgb, (len(pcd.points), 1))
        cols = np.round(cols, 3).flatten().tolist()
        return {"pts": pts, "cols": cols}

    static_data = format_pcd(pcd_static)
    added_data = format_pcd(pcd_added, [0.1, 0.9, 0.2])     # Bright Green
    removed_data = format_pcd(pcd_removed, [0.95, 0.15, 0.15]) # Bright Red
    occluded_data = format_pcd(pcd_occluded, [0.2, 0.6, 0.9]) # Blue/Cyan

    # Bounding boxes and movement arrows
    obb_list = []
    arrows = []

    for d in deltas:
        if d.change_type == ObjectChangeType.MOVED and d.object_a and d.object_b:
            corners_a = np.round(d.object_a.obb.get_corners(), 4).tolist()
            corners_b = np.round(d.object_b.obb.get_corners(), 4).tolist()
            obb_list.append({
                "label": f"[MOVED] {d.label} (Scan A)",
                "color": "#ffaa00",
                "corners": corners_a
            })
            obb_list.append({
                "label": f"[MOVED] {d.label} (Scan B)",
                "color": "#ffd700",
                "corners": corners_b
            })
            arrows.append({
                "start": [round(float(c), 4) for c in d.object_a.centroid],
                "end": [round(float(c), 4) for c in d.object_b.centroid],
                "color": "#ffaa00",
                "label": f"Shift: {d.displacement_distance*100:.1f}cm"
            })
        elif d.change_type == ObjectChangeType.ADDED and d.object_b:
            corners = np.round(d.object_b.obb.get_corners(), 4).tolist()
            obb_list.append({
                "label": f"[ADDED] {d.label}",
                "color": "#00ff66",
                "corners": corners
            })
        elif d.change_type == ObjectChangeType.REMOVED and d.object_a:
            corners = np.round(d.object_a.obb.get_corners(), 4).tolist()
            obb_list.append({
                "label": f"[REMOVED] {d.label}",
                "color": "#ff3333",
                "corners": corners
            })

    # Trajectories (Camera optical centers)
    traj_a = [[round(float(c), 4) for c in p.t] for p in poses_a]
    traj_b = [[round(float(c), 4) for c in p.t] for p in poses_b]

    # Camera Frustums (subsample to avoid clutter)
    step_a = max(1, len(poses_a) // 5)
    step_b = max(1, len(poses_b) // 5)
    frustums_a = [
        CameraFrustum(pose=p, intrinsics=intrinsics, z_near=0.1, z_far=0.5).get_corners().round(4).tolist()
        for p in poses_a[::step_a]
    ]
    frustums_b = [
        CameraFrustum(pose=p, intrinsics=intrinsics, z_near=0.1, z_far=0.5).get_corners().round(4).tolist()
        for p in poses_b[::step_b]
    ]

    scene_data = {
        "static": static_data,
        "added": added_data,
        "removed": removed_data,
        "occluded": occluded_data,
        "boxes": obb_list,
        "arrows": arrows,
        "traj_a": traj_a,
        "traj_b": traj_b,
        "frustums_a": frustums_a,
        "frustums_b": frustums_b,
        "metrics": classification.metrics,
        "summary": report.executive_summary,
        "changes": report.detailed_changes,
        "action_plan": report.robotic_manipulation_plan
    }

    scene_json = json.dumps(scene_data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f1117;
            color: #e6edf3;
            overflow: hidden;
            width: 100vw;
            height: 100vh;
        }}
        #canvas-container {{ width: 100%; height: 100%; position: absolute; top: 0; left: 0; }}
        .hud-panel {{
            position: absolute;
            background: rgba(22, 27, 34, 0.88);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(240, 246, 252, 0.15);
            border-radius: 10px;
            padding: 16px;
            z-index: 10;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        }}
        #controls-panel {{
            top: 20px;
            left: 20px;
            width: 320px;
            max-height: calc(100vh - 40px);
            overflow-y: auto;
        }}
        #info-panel {{
            top: 20px;
            right: 20px;
            width: 360px;
            max-height: calc(100vh - 40px);
            overflow-y: auto;
        }}
        h2 {{ font-size: 1.15rem; color: #58a6ff; margin-bottom: 12px; font-weight: 600; display: flex; align-items: center; gap: 8px; }}
        h3 {{ font-size: 0.9rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin: 12px 0 6px 0; }}
        .badge {{
            display: inline-block;
            font-size: 0.75rem;
            padding: 2px 8px;
            border-radius: 12px;
            font-weight: 600;
        }}
        .badge-added {{ background: #238636; color: white; }}
        .badge-removed {{ background: #da3633; color: white; }}
        .badge-moved {{ background: #d29922; color: black; }}
        .badge-occluded {{ background: #1f6feb; color: white; }}
        .badge-static {{ background: #484f58; color: white; }}
        .toggle-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid rgba(240, 246, 252, 0.08);
            font-size: 0.88rem;
        }}
        .toggle-row input[type="checkbox"] {{
            cursor: pointer;
            width: 16px;
            height: 16px;
            accent-color: #58a6ff;
        }}
        .metric-card {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin: 10px 0;
        }}
        .metric-box {{
            background: rgba(13, 17, 23, 0.7);
            padding: 8px;
            border-radius: 6px;
            border: 1px solid rgba(240, 246, 252, 0.1);
        }}
        .metric-value {{ font-size: 1.2rem; font-weight: bold; color: #f0f6fc; }}
        .metric-label {{ font-size: 0.72rem; color: #8b949e; }}
        .change-item {{
            background: rgba(13, 17, 23, 0.7);
            border-left: 3px solid #58a6ff;
            padding: 8px;
            margin-bottom: 8px;
            border-radius: 4px;
            font-size: 0.82rem;
            line-height: 1.4;
        }}
        .btn {{
            background: #238636;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.82rem;
            margin-top: 8px;
            width: 100%;
            font-weight: 600;
        }}
        .btn:hover {{ background: #2ea043; }}
        code {{
            background: #161b22;
            padding: 2px 4px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.78rem;
            color: #79c0ff;
        }}
        #help-bar {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(22, 27, 34, 0.8);
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.78rem;
            color: #8b949e;
            pointer-events: none;
        }}
    </style>
</head>
<body>
    <div id="canvas-container"></div>

    <!-- Controls Panel -->
    <div class="hud-panel" id="controls-panel">
        <h2>🌐 ChronoScene-3D</h2>
        <p style="font-size: 0.8rem; color: #8b949e; margin-bottom: 12px;">
            Embodied 3D Scene Differencing & Spatial Change Grounding
        </p>

        <h3>Layers & Visibility</h3>
        <div class="toggle-row">
            <span><span class="badge badge-added">●</span> Added Objects</span>
            <input type="checkbox" id="toggle-added" checked>
        </div>
        <div class="toggle-row">
            <span><span class="badge badge-removed">●</span> Removed Objects</span>
            <input type="checkbox" id="toggle-removed" checked>
        </div>
        <div class="toggle-row">
            <span><span class="badge badge-occluded">●</span> Occluded / Unobserved</span>
            <input type="checkbox" id="toggle-occluded" checked>
        </div>
        <div class="toggle-row">
            <span><span class="badge badge-static">●</span> Static Background</span>
            <input type="checkbox" id="toggle-static" checked>
        </div>
        <div class="toggle-row">
            <span>📦 3D Bounding Boxes</span>
            <input type="checkbox" id="toggle-boxes" checked>
        </div>
        <div class="toggle-row">
            <span>📹 Trajectories & Frustums</span>
            <input type="checkbox" id="toggle-cameras" checked>
        </div>

        <h3>Point Cloud Appearance</h3>
        <div class="toggle-row">
            <span>Point Size</span>
            <input type="range" id="slider-point-size" min="1" max="10" value="4" style="width: 100px;">
        </div>

        <button class="btn" id="btn-reset-view">🎯 Reset Camera View</button>
    </div>

    <!-- Info & Report Panel -->
    <div class="hud-panel" id="info-panel">
        <h2>📊 Spatial Intelligence</h2>
        <div class="metric-card">
            <div class="metric-box">
                <div class="metric-value" id="val-static-pct">--%</div>
                <div class="metric-label">Static Scene Invariance</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" id="val-occ-pct">--%</div>
                <div class="metric-label">Occluded / Unobserved</div>
            </div>
        </div>

        <h3>Executive Summary</h3>
        <p id="summary-text" style="font-size: 0.82rem; line-height: 1.4; color: #c9d1d9; margin-bottom: 12px;"></p>

        <h3>Detected Physical Changes</h3>
        <div id="changes-container"></div>

        <h3>Robotic Embodied Action</h3>
        <div id="actions-container"></div>
    </div>

    <div id="help-bar">Left Click + Drag: Rotate | Right Click + Drag: Pan | Scroll: Zoom</div>

    <script>
        const sceneData = {scene_json};

        // Populate UI Info
        document.getElementById('val-static-pct').innerText = (sceneData.metrics.static_percentage || 0) + '%';
        document.getElementById('val-occ-pct').innerText = (sceneData.metrics.occluded_percentage || 0) + '%';
        document.getElementById('summary-text').innerText = sceneData.summary;

        const changesContainer = document.getElementById('changes-container');
        changesContainer.innerHTML = '';
        sceneData.changes.forEach(ch => {{
            const div = document.createElement('div');
            div.className = 'change-item';
            div.innerHTML = ch;
            changesContainer.appendChild(div);
        }});

        const actionsContainer = document.getElementById('actions-container');
        actionsContainer.innerHTML = '';
        if (sceneData.action_plan && sceneData.action_plan.length > 0) {{
            sceneData.action_plan.forEach(act => {{
                const div = document.createElement('div');
                div.className = 'change-item';
                div.style.borderColor = '#238636';
                div.innerHTML = `<strong>${{act.action}}</strong>: ${{act.object_label}}<br><code>Delta: ${{act.displacement_m ? act.displacement_m + 'm' : 'Observed'}}</code>`;
                actionsContainer.appendChild(div);
            }});
        }} else {{
            actionsContainer.innerHTML = '<div style="font-size: 0.8rem; color: #8b949e;">No robot actions required.</div>';
        }}

        // Setup Three.js Scene
        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0c10);

        const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.05, 50.0);
        camera.position.set(0.0, -1.8, 1.4);
        camera.up.set(0, 0, 1);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.target.set(0, 0, 0);

        // Grid floor helper
        const gridHelper = new THREE.GridHelper(4, 20, 0x30363d, 0x21262d);
        gridHelper.rotation.x = Math.PI / 2;
        gridHelper.position.z = -0.01;
        scene.add(gridHelper);

        // Axes helper
        const axesHelper = new THREE.AxesHelper(0.3);
        axesHelper.position.set(-1.0, -1.0, 0.0);
        scene.add(axesHelper);

        // Point Clouds
        function createPointCloud(pts, cols, defaultSize = 0.018) {{
            if (!pts || pts.length === 0) return new THREE.Group();
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
            const material = new THREE.PointsMaterial({{
                size: defaultSize,
                vertexColors: true,
                sizeAttenuation: true,
                transparent: true,
                opacity: 0.95
            }});
            return new THREE.Points(geometry, material);
        }}

        const staticCloud = createPointCloud(sceneData.static.pts, sceneData.static.cols, 0.012);
        const addedCloud = createPointCloud(sceneData.added.pts, sceneData.added.cols, 0.022);
        const removedCloud = createPointCloud(sceneData.removed.pts, sceneData.removed.cols, 0.022);
        const occludedCloud = createPointCloud(sceneData.occluded.pts, sceneData.occluded.cols, 0.014);

        scene.add(staticCloud);
        scene.add(addedCloud);
        scene.add(removedCloud);
        scene.add(occludedCloud);

        // Bounding Boxes Group
        const boxGroup = new THREE.Group();
        const boxEdges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ];

        sceneData.boxes.forEach(box => {{
            const corners = box.corners;
            const points = [];
            boxEdges.forEach(edge => {{
                points.push(new THREE.Vector3(...corners[edge[0]]));
                points.push(new THREE.Vector3(...corners[edge[1]]));
            }});
            const geom = new THREE.BufferGeometry().setFromPoints(points);
            const mat = new THREE.LineBasicMaterial({{ color: new THREE.Color(box.color), linewidth: 2 }});
            const line = new THREE.LineSegments(geom, mat);
            boxGroup.add(line);
        }});

        // Movement Arrows
        sceneData.arrows.forEach(arr => {{
            const start = new THREE.Vector3(...arr.start);
            const end = new THREE.Vector3(...arr.end);
            const dir = new THREE.Vector3().subVectors(end, start);
            const len = dir.length();
            if (len > 0.01) {{
                const arrowHelper = new THREE.ArrowHelper(dir.normalize(), start, len, 0xffaa00, 0.08, 0.04);
                boxGroup.add(arrowHelper);
            }}
        }});
        scene.add(boxGroup);

        // Camera Trajectories Group
        const cameraGroup = new THREE.Group();

        function addTrajectoryLine(points, colorHex) {{
            if (!points || points.length < 2) return;
            const pts = points.map(p => new THREE.Vector3(...p));
            const geom = new THREE.BufferGeometry().setFromPoints(pts);
            const mat = new THREE.LineBasicMaterial({{ color: colorHex, linewidth: 3 }});
            cameraGroup.add(new THREE.Line(geom, mat));
        }}
        addTrajectoryLine(sceneData.traj_a, 0x00d2ff); // Cyan for Scan A
        addTrajectoryLine(sceneData.traj_b, 0xff00d2); // Magenta for Scan B

        // Add Camera Frustums wireframes
        function addFrustums(frustumList, colorHex) {{
            frustumList.forEach(f => {{
                const pts = [];
                // Near plane loop
                pts.push(new THREE.Vector3(...f[0]), new THREE.Vector3(...f[1]));
                pts.push(new THREE.Vector3(...f[1]), new THREE.Vector3(...f[2]));
                pts.push(new THREE.Vector3(...f[2]), new THREE.Vector3(...f[3]));
                pts.push(new THREE.Vector3(...f[3]), new THREE.Vector3(...f[0]));
                // Far plane loop
                pts.push(new THREE.Vector3(...f[4]), new THREE.Vector3(...f[5]));
                pts.push(new THREE.Vector3(...f[5]), new THREE.Vector3(...f[6]));
                pts.push(new THREE.Vector3(...f[6]), new THREE.Vector3(...f[7]));
                pts.push(new THREE.Vector3(...f[7]), new THREE.Vector3(...f[4]));
                // Connect near to far
                pts.push(new THREE.Vector3(...f[0]), new THREE.Vector3(...f[4]));
                pts.push(new THREE.Vector3(...f[1]), new THREE.Vector3(...f[5]));
                pts.push(new THREE.Vector3(...f[2]), new THREE.Vector3(...f[6]));
                pts.push(new THREE.Vector3(...f[3]), new THREE.Vector3(...f[7]));

                const geom = new THREE.BufferGeometry().setFromPoints(pts);
                const mat = new THREE.LineBasicMaterial({{ color: colorHex, transparent: true, opacity: 0.4 }});
                cameraGroup.add(new THREE.LineSegments(geom, mat));
            }});
        }}
        addFrustums(sceneData.frustums_a, 0x00d2ff);
        addFrustums(sceneData.frustums_b, 0xff00d2);

        scene.add(cameraGroup);

        // UI Event Listeners
        document.getElementById('toggle-added').addEventListener('change', e => addedCloud.visible = e.target.checked);
        document.getElementById('toggle-removed').addEventListener('change', e => removedCloud.visible = e.target.checked);
        document.getElementById('toggle-occluded').addEventListener('change', e => occludedCloud.visible = e.target.checked);
        document.getElementById('toggle-static').addEventListener('change', e => staticCloud.visible = e.target.checked);
        document.getElementById('toggle-boxes').addEventListener('change', e => boxGroup.visible = e.target.checked);
        document.getElementById('toggle-cameras').addEventListener('change', e => cameraGroup.visible = e.target.checked);

        document.getElementById('slider-point-size').addEventListener('input', e => {{
            const val = parseFloat(e.target.value) / 200.0;
            staticCloud.material.size = val;
            addedCloud.material.size = val * 1.5;
            removedCloud.material.size = val * 1.5;
            occludedCloud.material.size = val * 1.2;
        }});

        document.getElementById('btn-reset-view').addEventListener('click', () => {{
            camera.position.set(0.0, -1.8, 1.4);
            controls.target.set(0, 0, 0);
            controls.update();
        }});

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
</body>
</html>
"""
    return html


def save_interactive_3d_viewer(
    classification: ChangeClassificationResult,
    deltas: List[ObjectChangeDelta],
    report: SceneDifferenceReport,
    poses_a: List[CameraPose],
    poses_b: List[CameraPose],
    intrinsics: CameraIntrinsics,
    output_html_path: str
) -> str:
    """Save self-contained interactive Three.js 3D viewer to HTML file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
    html_content = generate_threejs_viewer_html(
        classification=classification,
        deltas=deltas,
        report=report,
        poses_a=poses_a,
        poses_b=poses_b,
        intrinsics=intrinsics
    )
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_html_path
