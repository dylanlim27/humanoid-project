"""
ChronoScene-3D: Interactive Streamlit Web Application.
Provides interactive visual comparison of dual phone scans, 3D point cloud inspection,
real-time threshold adjustments, and robotic action plan downloads.

Run:
    streamlit run web_app.py
"""

import os
import sys
import json
import streamlit as st
import streamlit.components.v1 as components
import numpy as np

# Ensure package root is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from chronoscene.data.benchmark_generator import IndoorBenchmarkGenerator
from chronoscene.data.video_loader import PhoneVideoLoader
from chronoscene.pipeline import ChronoScenePipeline
from chronoscene.visualization.plotly_vis import create_plotly_3d_figure
from chronoscene.reasoning.instance_matcher import ObjectChangeType


st.set_page_config(
    page_title="ChronoScene-3D | 3D Spatial Differencing",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean research laboratory styling
st.markdown("""
<style>
    .main-header {
        font-size: 1.9rem;
        font-weight: 600;
        color: #e6edf3;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #8b949e;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 12px 16px;
        text-align: left;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 600;
        color: #58a6ff;
    }
    .metric-lbl {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">ChronoScene-3D</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-View 3D Scene Differencing & Spatial Change Grounding from Monocular Phone Video</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Configuration & Data Source")
mode = st.sidebar.radio(
    "Source Mode",
    ["Indoor Tabletop Benchmark", "Upload Custom Videos (MP4/MOV)"]
)

st.sidebar.subheader("Geometric Parameters")
voxel_size = st.sidebar.slider("Voxel Resolution (m)", 0.01, 0.05, 0.02, 0.005)
static_thresh = st.sidebar.slider("Static Distance Tolerance (m)", 0.02, 0.08, 0.04, 0.005)
motion_thresh = st.sidebar.slider("Motion Displacement Threshold (m)", 0.03, 0.15, 0.06, 0.01)

st.sidebar.subheader("Line-of-Sight Filter")
surface_eps = st.sidebar.slider("Surface Thickness Epsilon (m)", 0.02, 0.08, 0.04, 0.005)

@st.cache_resource
def get_pipeline(v_size, s_thresh, m_thresh, s_eps):
    p = ChronoScenePipeline(
        output_dir="output",
        voxel_size=v_size,
        static_threshold=s_thresh,
        motion_threshold=m_thresh
    )
    p.raycast_engine.surface_epsilon = s_eps
    return p

pipeline = get_pipeline(voxel_size, static_thresh, motion_thresh, surface_eps)

# Execution
if mode == "Indoor Tabletop Benchmark":
    st.info("Benchmark Scenario: Evaluates two camera trajectories over a robotics manipulation workspace with known moved, added, removed, and out-of-view objects.")

    if st.button("Run Spatial Differencing Pipeline", type="primary"):
        with st.spinner("Processing multi-view scans and computing line-of-sight visibility..."):
            gen = IndoorBenchmarkGenerator(width=480, height=360, fov_deg=65.0)
            scan_a, scan_b, gt_deltas = gen.generate_benchmark_pair(num_frames=12)

            labels_a = ["laptop", "monitor", "ceramic_mug", "blue_notebook", "far_tool_rack"]
            labels_b = ["laptop", "monitor", "ceramic_mug", "electric_drill", "far_tool_rack"]

            results = pipeline.run(
                scan_a=scan_a,
                scan_b=scan_b,
                semantic_labels_a=labels_a,
                semantic_labels_b=labels_b
            )
            st.session_state["results"] = results
            st.session_state["scan_a"] = scan_a
            st.session_state["scan_b"] = scan_b

else:
    st.subheader("Upload Two Phone Video Recordings")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        vid_a = st.file_uploader("Scan A (Reference Time 1)", type=["mp4", "mov", "avi"])
    with col_u2:
        vid_b = st.file_uploader("Scan B (Query Time 2)", type=["mp4", "mov", "avi"])

    if vid_a and vid_b:
        if st.button("Process Phone Recordings", type="primary"):
            with st.spinner("Extracting keyframes and executing pipeline..."):
                os.makedirs("data/uploads", exist_ok=True)
                path_a = os.path.join("data/uploads", vid_a.name)
                path_b = os.path.join("data/uploads", vid_b.name)
                with open(path_a, "wb") as f:
                    f.write(vid_a.getbuffer())
                with open(path_b, "wb") as f:
                    f.write(vid_b.getbuffer())

                loader = PhoneVideoLoader(target_width=640)
                scan_a = loader.load_video(path_a, max_keyframes=16, scan_id="A")
                scan_b = loader.load_video(path_b, max_keyframes=16, scan_id="B")

                results = pipeline.run(scan_a, scan_b)
                st.session_state["results"] = results
                st.session_state["scan_a"] = scan_a
                st.session_state["scan_b"] = scan_b

# Render Results
if "results" in st.session_state:
    results = st.session_state["results"]
    scan_a = st.session_state["scan_a"]
    scan_b = st.session_state["scan_b"]

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{results.classification.metrics.get('static_percentage', 0)}%</div>
            <div class="metric-lbl">Static Invariance Ratio</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        moved_count = len([d for d in results.object_deltas if d.change_type == ObjectChangeType.MOVED])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #d29922;">{moved_count}</div>
            <div class="metric-lbl">Moved Objects</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        added_count = len([d for d in results.object_deltas if d.change_type == ObjectChangeType.ADDED])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #238636;">{added_count}</div>
            <div class="metric-lbl">Added Objects</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        removed_count = len([d for d in results.object_deltas if d.change_type == ObjectChangeType.REMOVED])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #da3633;">{removed_count}</div>
            <div class="metric-lbl">Removed Objects</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 3D Spatial Differencing Results")
    tab3d, tab_keyframes, tab_report, tab_robot = st.tabs([
        "Interactive 3D WebGL Viewer", "Keyframe Inspection", "Spatial Intelligence Report", "Robotic Action Directives"
    ])

    with tab3d:
        viewer_path = results.viewer_html_path
        if os.path.exists(viewer_path):
            with open(viewer_path, "r", encoding="utf-8") as f:
                html_data = f.read()
            components.html(html_data, height=650, scrolling=False)
        else:
            fig = create_plotly_3d_figure(
                results.classification,
                results.object_deltas,
                poses_a=results.aligned_poses_a,
                poses_b=results.aligned_poses_b
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_keyframes:
        st.subheader("Keyframe Trajectory Comparison")
        col_ka, col_kb = st.columns(2)
        with col_ka:
            st.markdown("**Scan A Keyframes (Time 1)**")
            if scan_a and scan_a.rgb_frames:
                idx_a = st.slider("Scan A Frame", 0, len(scan_a.rgb_frames)-1, 0)
                st.image(scan_a.rgb_frames[idx_a], use_container_width=True)
        with col_kb:
            st.markdown("**Scan B Keyframes (Time 2)**")
            if scan_b and scan_b.rgb_frames:
                idx_b = st.slider("Scan B Frame", 0, len(scan_b.rgb_frames)-1, 0)
                st.image(scan_b.rgb_frames[idx_b], use_container_width=True)

    with tab_report:
        st.markdown(results.report.to_markdown())

    with tab_robot:
        st.subheader("Robotic Manipulation Directives (6-DoF Pick-and-Place)")
        st.write("Structured machine-executable instructions derived from 3D object deltas:")
        st.json(results.report.robotic_manipulation_plan)

    st.markdown("---")
    st.subheader("Export Deliverables")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        if os.path.exists(results.viewer_html_path):
            with open(results.viewer_html_path, "r", encoding="utf-8") as f:
                st.download_button(
                    "Download 3D Viewer (HTML)",
                    f.read(),
                    file_name="viewer_3d.html",
                    mime="text/html"
                )
    with col_d2:
        if os.path.exists(results.action_json_path):
            with open(results.action_json_path, "r", encoding="utf-8") as f:
                st.download_button(
                    "Download Action Plan (JSON)",
                    f.read(),
                    file_name="robotic_action_plan.json",
                    mime="application/json"
                )
    with col_d3:
        if os.path.exists(results.report_md_path):
            with open(results.report_md_path, "r", encoding="utf-8") as f:
                st.download_button(
                    "Download Change Report (MD)",
                    f.read(),
                    file_name="change_report.md",
                    mime="text/markdown"
                )
