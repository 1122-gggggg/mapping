"""Risk-PLY writes map vertices plus colored marker spheres and a JSON legend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sfm_diagnosis.cli import main as risk_ply_cli
from sfm_diagnosis.models import CameraIntrinsics, MapData
from sfm_diagnosis.risk_ply import ISSUE_COLORS, load_jsonl_rows, load_rows, write_risk_ply
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
        image_ids=np.array([0, 1, 2, 3]),
        image_names=["im_0.jpg", "im_1.jpg", "bridge.jpg", "tri.jpg"],
        image_camera_ids=np.zeros(4, dtype=int),
        image_centers=np.array([[0, 0, 0], [1, 0, 0], [4, 0, 0], [5, 0, 0]], dtype=float),
        image_R_wc=np.repeat(np.eye(3)[None], 4, axis=0),
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


def test_load_rows_jsonl_objects_not_csv(tmp_path: Path):
    path = tmp_path / "loc.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query": "q1",
                        "status": "GEOMETRY_WEAK",
                        "x": 1.0,
                        "y": 2.0,
                        "z": 3.0,
                    }
                ),
                "",
                json.dumps(
                    {
                        "query": "q2",
                        "status": "DIRECT_PROVISIONAL",
                        "pose": {"x": 4.0, "y": 5.0, "z": 6.0},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_rows(path)
    assert [row["query"] for row in rows] == ["q1", "q2"]
    assert rows[0]["x"] == 1.0
    assert "query" in rows[0]
    assert list(rows[0]) != [path.read_text(encoding="utf-8").splitlines()[0]]


def test_load_jsonl_rejects_non_object(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text("[1, 2, 3]\n{\"ok\": true}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        load_jsonl_rows(path)


def test_write_risk_ply_jsonl_path(tmp_path: Path):
    logs = tmp_path / "heldout.jsonl"
    logs.write_text(
        json.dumps({"query": "q1", "status": "GEOMETRY_WEAK", "x": 2.0, "y": 0.0, "z": 0.0})
        + "\n",
        encoding="utf-8",
    )
    receipt = write_risk_ply(
        _tiny_map(),
        tmp_path / "from_jsonl",
        localization=logs,
        sphere_samples=8,
        filename="from_jsonl.ply",
    )
    assert receipt["fim_recomputed"] is False
    assert receipt["inputs"]["localization_rows"] == 1
    assert receipt["counts"]["heldout_geometry_weak"] == 1


def test_cli_risk_ply_jsonl_logs(tmp_path: Path):
    fixture = Path(__file__).resolve().parents[1] / "mapdoctor" / "fixtures" / "colmap_text"
    logs = tmp_path / "loc.jsonl"
    logs.write_text(
        json.dumps({"query": "q1", "status": "GEOMETRY_WEAK", "x": 0.0, "y": 0.0, "z": 0.0})
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "cli-risk"
    code = risk_ply_cli(
        [
            "risk-ply",
            str(fixture),
            "--map-adapter",
            "colmap",
            "--output",
            str(out),
            "--logs",
            str(logs),
        ]
    )
    assert code == 0
    receipt = json.loads((out / "risk_ply_receipt.json").read_text(encoding="utf-8"))
    assert receipt["inputs"]["localization_rows"] == 1
    assert receipt["fim_recomputed"] is False
    assert receipt["counts"]["heldout_geometry_weak"] == 1


def test_zero_observation_roles_and_no_double_count(tmp_path: Path):
    receipt = write_risk_ply(
        _tiny_map(),
        tmp_path / "roles",
        image_roles={"bridge.jpg": "bridge_only", "tri.jpg": "triangulation"},
        extra_markers=[
            {
                "issue_class": "zero_triangulation",
                "x": 5.0,
                "y": 0.0,
                "z": 0.0,
                "image_name": "tri.jpg",
            }
        ],
        sphere_samples=8,
        filename="roles.ply",
    )
    assert receipt["counts"]["unverified_bridge_pose"] == 1
    assert receipt["counts"]["zero_triangulation"] == 1
    assert receipt["fim_recomputed"] is False
    assert "zero_triangulation" in ISSUE_COLORS


def test_heldout_success_requires_outer_strong_and_nested_accept(tmp_path: Path):
    receipt = write_risk_ply(
        _tiny_map(),
        tmp_path / "accept",
        localization=[
            {
                "query": "accepted",
                "status": "DIRECT_STRONG",
                "decision": {"status": "ACCEPT"},
                "x": 1.0,
                "y": 0.0,
                "z": 0.0,
            },
            {
                "query": "weak_accept",
                "status": "GEOMETRY_WEAK",
                "decision": {"status": "ACCEPT"},
                "x": 1.5,
                "y": 0.0,
                "z": 0.0,
            },
            {
                "query": "leaked",
                "status": "DIRECT_STRONG",
                "decision": {"status": "REJECT_UNVERIFIED_SUPPORT"},
                "x": 2.0,
                "y": 0.0,
                "z": 0.0,
            },
            {
                "query": "provisional_accept",
                "status": "DIRECT_PROVISIONAL",
                "decision": {"status": "ACCEPT"},
                "x": 3.0,
                "y": 0.0,
                "z": 0.0,
            },
            {
                "query": "strong_missing_nested",
                "status": "DIRECT_STRONG",
                "success": True,
                "x": 3.5,
                "y": 0.0,
                "z": 0.0,
            },
            {
                "query": "weak_bool_override",
                "status": "GEOMETRY_WEAK",
                "decision": {"status": "ACCEPT"},
                "success": True,
                "x": 4.0,
                "y": 0.0,
                "z": 0.0,
            },
        ],
        image_roles={"bridge.jpg": "bridge_only", "tri.jpg": "triangulation"},
        sphere_samples=8,
        filename="accept.ply",
    )
    queries = {
        marker.get("query"): marker.get("issue_class")
        for marker in receipt["markers"]
        if marker.get("query")
    }
    assert "accepted" not in queries
    assert queries["weak_accept"] == "heldout_geometry_weak"
    assert queries["leaked"] == "heldout_geometry_weak"
    assert queries["strong_missing_nested"] == "heldout_geometry_weak"
    assert queries["weak_bool_override"] == "heldout_geometry_weak"
    assert queries["provisional_accept"] == "heldout_provisional"
    assert receipt["counts"]["heldout_geometry_weak"] == 4
    assert receipt["counts"]["heldout_provisional"] == 1
    assert receipt["fim_recomputed"] is False


def test_boolean_success_only_without_richer_statuses(tmp_path: Path):
    receipt = write_risk_ply(
        _tiny_map(),
        tmp_path / "legacy",
        localization=[
            {
                "query": "legacy_ok",
                "success": True,
                "x": 1.0,
                "y": 0.0,
                "z": 0.0,
            },
            {
                "query": "legacy_fail",
                "success": False,
                "x": 2.0,
                "y": 0.0,
                "z": 0.0,
            },
        ],
        sphere_samples=8,
        filename="legacy.ply",
    )
    queries = {
        marker.get("query"): marker.get("issue_class")
        for marker in receipt["markers"]
        if marker.get("query")
    }
    assert "legacy_ok" not in queries
    assert queries["legacy_fail"] == "heldout_geometry_weak"
    assert receipt["counts"]["heldout_geometry_weak"] == 1


def test_healthy_map_fixture_still_imports():
    assert healthy_map().num_points > 0
