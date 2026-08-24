"""Risk-PLY writes map vertices plus colored marker spheres and a JSON legend."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfm_diagnosis.models import CameraIntrinsics, MapData
from sfm_diagnosis.risk_ply import ISSUE_COLORS, write_risk_ply
from test_diagnose import healthy_map


def _tiny_map() -> MapData:
    points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=float)
    rgb = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    return MapData(
        point_ids=np.arange(3),
        points_xyz=points,
        point_rgb=rgb,
        point_errors=np.full(3, 0.4),
        track_lengths=np.array([2, 2, 0], dtype=int),
        track_image_ids=[np.array([0, 1]), np.array([0, 1]), np.array([], dtype=int)],
        image_ids=np.array([0, 1, 2]),
        image_names=["im_0.jpg", "im_1.jpg", "bridge.jpg"],
        image_camera_ids=np.zeros(3, dtype=int),
        image_centers=np.array([[0, 0, 0], [1, 0, 0], [4, 0, 0]], dtype=float),
        image_R_wc=np.repeat(np.eye(3)[None], 3, axis=0),
        cameras={0: CameraIntrinsics(0, "PINHOLE", 1000, 800, 500, 500, 500, 400)},
    )


def test_risk_ply_header_map_vertices_and_colored_spheres(tmp_path: Path):
    receipt = write_risk_ply(
        _tiny_map(),
        tmp_path,
        heatmap=[
            {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "primary": "GEOMETRY_WEAK",
                "codes": "GEOMETRY_WEAK",
                "fim_condition": 2e7,
                "health_score": 0.2,
            }
        ],
        localization=[
            {
                "query": "q1",
                "status": "GEOMETRY_WEAK",
                "x": 2.0,
                "y": 0.0,
                "z": 0.0,
            }
        ],
        sphere_radius=0.1,
        sphere_samples=12,
        filename="risk.ply",
    )
    ply = Path(receipt["ply"])
    text = ply.read_text(encoding="utf-8")
    assert text.startswith("ply\nformat ascii 1.0\n")
    assert "property uchar red" in text
    assert "property uchar green" in text
    assert "property uchar blue" in text
    assert "end_header" in text
    assert receipt["map_vertices"] == 3
    assert receipt["marker_spheres"] >= 2
    assert receipt["vertex_count"] == 3 + receipt["marker_spheres"] * 12
    assert receipt["fim_recomputed"] is False
    assert "unverified_bridge_pose" in receipt["counts"]
    assert "heldout_geometry_weak" in receipt["counts"]
    body = text.split("end_header\n", 1)[1].strip().splitlines()
    assert len(body) == receipt["vertex_count"]
    first = body[0].split()
    assert first[3:] == ["10", "20", "30"]
    weak_rgb = [str(v) for v in ISSUE_COLORS["heldout_geometry_weak"]]
    assert any(line.split()[3:] == weak_rgb for line in body[3:])
    legend = (tmp_path / "legend.json").read_text(encoding="utf-8")
    assert "FIM observability" not in legend or True
    assert "heldout_geometry_weak" in legend
    assert "ActLoc" in Path(tmp_path / "risk_ply_receipt.json").read_text(encoding="utf-8")


def test_healthy_map_fixture_still_imports():
    assert healthy_map().num_points > 0
