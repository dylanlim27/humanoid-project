"""
Plotly 3D visualizer for interactive browser and Streamlit dashboard integration.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
import plotly.graph_objects as go

from ..reasoning.change_detector import ChangeClassificationResult
from ..reasoning.instance_matcher import ObjectChangeDelta, ObjectChangeType
from ..geometry.transforms import CameraPose


def create_plotly_3d_figure(
    classification: ChangeClassificationResult,
    deltas: List[ObjectChangeDelta],
    poses_a: Optional[List[CameraPose]] = None,
    poses_b: Optional[List[CameraPose]] = None,
    max_points: int = 8000
) -> go.Figure:
    """
    Generate an interactive Plotly 3D figure showing the point clouds and changes.
    """
    fig = go.Figure()

    # Subsample points for smooth rendering
    def subsample(pts, cols, limit):
        if len(pts) == 0:
            return pts, cols
        if len(pts) > limit:
            idx = np.random.choice(len(pts), limit, replace=False)
            return pts[idx], cols[idx]
        return pts, cols

    # 1. Static points
    s_pts, s_cols = subsample(classification.static_pcd.points, classification.static_pcd.colors, max_points)
    if len(s_pts) > 0:
        fig.add_trace(go.Scatter3d(
            x=s_pts[:, 0], y=s_pts[:, 1], z=s_pts[:, 2],
            mode='markers',
            marker=dict(size=1.5, color='gray', opacity=0.4),
            name='Static Scene'
        ))

    # 2. Added points (Green)
    a_pts, _ = subsample(classification.added_pcd.points, classification.added_pcd.colors, 4000)
    if len(a_pts) > 0:
        fig.add_trace(go.Scatter3d(
            x=a_pts[:, 0], y=a_pts[:, 1], z=a_pts[:, 2],
            mode='markers',
            marker=dict(size=3.0, color='rgb(40, 220, 80)', opacity=0.9),
            name='Added (+)'
        ))

    # 3. Removed points (Red)
    r_pts, _ = subsample(classification.removed_pcd.points, classification.removed_pcd.colors, 4000)
    if len(r_pts) > 0:
        fig.add_trace(go.Scatter3d(
            x=r_pts[:, 0], y=r_pts[:, 1], z=r_pts[:, 2],
            mode='markers',
            marker=dict(size=3.0, color='rgb(235, 45, 45)', opacity=0.9),
            name='Removed (-)'
        ))

    # 4. Occluded / Unobserved points (Blue)
    o_pts, _ = subsample(classification.occluded_pcd.points, classification.occluded_pcd.colors, 3000)
    if len(o_pts) > 0:
        fig.add_trace(go.Scatter3d(
            x=o_pts[:, 0], y=o_pts[:, 1], z=o_pts[:, 2],
            mode='markers',
            marker=dict(size=2.0, color='rgb(60, 160, 240)', opacity=0.5),
            name='Occluded / Out-of-View'
        ))

    # 5. Trajectories
    if poses_a:
        traj_a = np.array([p.t for p in poses_a])
        fig.add_trace(go.Scatter3d(
            x=traj_a[:, 0], y=traj_a[:, 1], z=traj_a[:, 2],
            mode='lines+markers',
            line=dict(color='cyan', width=4),
            marker=dict(size=3, color='cyan'),
            name='Camera Trajectory Scan A'
        ))

    if poses_b:
        traj_b = np.array([p.t for p in poses_b])
        fig.add_trace(go.Scatter3d(
            x=traj_b[:, 0], y=traj_b[:, 1], z=traj_b[:, 2],
            mode='lines+markers',
            line=dict(color='magenta', width=4),
            marker=dict(size=3, color='magenta'),
            name='Camera Trajectory Scan B'
        ))

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(title="X (m)", showbackground=False),
            yaxis=dict(title="Y (m)", showbackground=False),
            zaxis=dict(title="Z (m)", showbackground=False),
            aspectmode='data'
        ),
        margin=dict(r=10, l=10, b=10, t=10),
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05)
    )

    return fig
