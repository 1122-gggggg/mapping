"""Risk-PLY writes map vertices plus colored marker spheres and a JSON legend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sfm_diagnosis.cli import main as risk_ply_cli
from sfm_diagnosis.models import CameraIntrinsics, MapData
from sfm_diagnosis.risk_ply import (
    ISSUE_COLORS,
    MARKER_RADIUS_ABS_FLOOR,
    MARKER_RADIUS_DIAG_FRACTION,
    VISIBLE_MAP_RGB,
    camera_nearest_spacing,
    load_jsonl_rows,
    load_rows,
    robust_spatial_clip,
    sphere_mesh,
    sphere_points,
    write_risk_ply,
)
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


def _parse_ply(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    sep = b"end_header\n"
    idx = raw.index(sep)
    header = raw[: idx + len(sep)].decode("ascii")
    payload = raw[idx + len(sep) :]
    meta: dict[str, object] = {"header": header, "payload": payload}
    for line in header.splitlines():
        if line.startswith("format "):
            meta["format"] = line.split(" ", 1)[1]
        elif line.startswith("element vertex"):
            meta["vertex"] = int(line.split()[-1])
        elif line.startswith("element face"):
            meta["face"] = int(line.split()[-1])
        elif line.startswith("property "):
            meta.setdefault("properties", []).append(line)
    return meta


def _read_vertices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    meta = _parse_ply(path)
    payload = meta["payload"]
    n = int(meta["vertex"])
    rec = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
    )
    assert rec.itemsize == 15
    vertex_bytes = n * 15
    verts = np.frombuffer(payload[:vertex_bytes], dtype=rec)
    xyz = np.column_stack([verts["x"], verts["y"], verts["z"]]).astype(float)
    rgb = np.column_stack([verts["r"], verts["g"], verts["b"]]).astype(np.uint8)
    return xyz, rgb


def _read_faces(path: Path) -> np.ndarray:
    meta = _parse_ply(path)
    n_vertices = int(meta["vertex"])
    n_faces = int(meta.get("face") or 0)
    rec = np.dtype([("n", "u1"), ("i0", "<i4"), ("i1", "<i4"), ("i2", "<i4")])
    faces = np.frombuffer(meta["payload"][n_vertices * 15 :], dtype=rec)
    assert rec.itemsize == 13
    assert len(faces) == n_faces
    return np.column_stack([faces["i0"], faces["i1"], faces["i2"]])



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
    meta = _parse_ply(ply)
    assert meta["format"] == "binary_little_endian 1.0"
    assert "property uchar red" in meta["header"]
    assert "property uchar green" in meta["header"]
    assert "property uchar blue" in meta["header"]
    assert "end_header" in meta["header"]
    assert int(meta["vertex"]) == receipt["vertex_count"]
    assert len(meta["payload"]) == receipt["vertex_count"] * 15
    assert receipt["map_vertices"] == 3
    assert receipt["marker_spheres"] >= 2
    assert receipt["vertex_count"] == 3 + receipt["marker_spheres"] * 12
    assert receipt["fim_recomputed"] is False
    assert "unverified_bridge_pose" in receipt["counts"]
    assert "heldout_geometry_weak" in receipt["counts"]
    xyz, rgb = _read_vertices(ply)
    assert len(xyz) == receipt["vertex_count"]
    assert rgb[0].tolist() == [10, 20, 30]
    weak_rgb = list(ISSUE_COLORS["heldout_geometry_weak"])
    assert any(row.tolist() == weak_rgb for row in rgb[3:])
    legend = (tmp_path / "legend.json").read_text(encoding="utf-8")
    assert "heldout_geometry_weak" in legend
    assert "ActLoc" in Path(tmp_path / "risk_ply_receipt.json").read_text(encoding="utf-8")
    assert Path(receipt["ply_full"]).is_file()
    assert Path(receipt["clipping_receipt"]).is_file()



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


def _outlier_map() -> MapData:
    core = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.5, 1.2],
            [0.2, 0.1, 0.9],
        ],
        dtype=float,
    )
    outliers = np.array([[1.0e6, 0.0, 0.0], [0.0, -8.0e5, 3.0e5]], dtype=float)
    points = np.vstack([core, outliers])
    rgb = np.vstack(
        [
            np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90], [11, 22, 33], [44, 55, 66]], dtype=np.uint8),
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
        ]
    )
    n = len(points)
    return MapData(
        point_ids=np.arange(n),
        points_xyz=points,
        point_rgb=rgb,
        point_errors=np.full(n, 0.4),
        track_lengths=np.full(n, 2, dtype=int),
        track_image_ids=[np.array([0, 1]) for _ in range(n)],
        image_ids=np.array([0, 1, 2, 3]),
        image_names=["im_0.jpg", "im_1.jpg", "bridge.jpg", "tri.jpg"],
        image_camera_ids=np.zeros(4, dtype=int),
        image_centers=np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.2, 0.1], [0.4, 0.1, 0.2]], dtype=float),
        image_R_wc=np.repeat(np.eye(3)[None], 4, axis=0),
        cameras={0: CameraIntrinsics(0, "PINHOLE", 1000, 800, 500, 500, 500, 400)},
    )


def test_extreme_outlier_clip_retains_core_rgb_and_full_archive(tmp_path: Path):
    receipt = write_risk_ply(
        _outlier_map(),
        tmp_path / "clip",
        localization=[{"query": "q1", "status": "GEOMETRY_WEAK", "x": 0.4, "y": 0.1, "z": 0.2}],
        sphere_samples=8,
        filename="clip.ply",
    )
    clip = receipt["clip"]
    assert receipt["map_vertices"] == 7
    assert receipt["map_vertices_retained"] == 5
    assert receipt["map_vertices_excluded"] == 2
    assert receipt["vertex_count_full"] > receipt["vertex_count"]
    assert set(clip["excluded_point_ids"]) == {5, 6}
    assert np.isfinite(clip["robust_diagonal"])
    assert all(abs(v) < 100 for v in clip["robust_bounds"]["min"])
    assert all(abs(v) < 100 for v in clip["robust_bounds"]["max"])
    assert clip["robust_diagonal"] < clip["full_diagonal"] / 20.0
    cc_xyz, cc_rgb = _read_vertices(Path(receipt["ply"]))
    assert np.isfinite(cc_xyz).all()
    assert cc_xyz.max() < 100
    assert [10, 20, 30] in cc_rgb.tolist()
    assert [40, 50, 60] in cc_rgb.tolist()
    full_xyz, full_rgb = _read_vertices(Path(receipt["ply_full"]))
    assert full_xyz.max() > 1.0e5
    assert [1, 2, 3] in full_rgb.tolist()
    assert receipt["marker_spheres_cloudcompare"] == receipt["marker_spheres"]
    assert receipt["sphere_radius"] >= MARKER_RADIUS_ABS_FLOOR
    assert receipt["sphere_radius"] >= MARKER_RADIUS_DIAG_FRACTION * clip["robust_diagonal"]
    meta = _parse_ply(Path(receipt["ply"]))
    assert int(meta["vertex"]) == receipt["vertex_count"]
    assert len(meta["payload"]) == receipt["vertex_count"] * 15
    clipping = json.loads(Path(receipt["clipping_receipt"]).read_text(encoding="utf-8"))
    assert clipping["excluded_count"] == 2


def test_black_base_rgb_fallback_keeps_marker_colors(tmp_path: Path):
    base = _tiny_map()
    black = MapData(
        point_ids=base.point_ids,
        points_xyz=base.points_xyz,
        point_rgb=np.zeros((3, 3), dtype=np.uint8),
        point_errors=base.point_errors,
        track_lengths=base.track_lengths,
        track_image_ids=base.track_image_ids,
        image_ids=base.image_ids,
        image_names=base.image_names,
        image_camera_ids=base.image_camera_ids,
        image_centers=base.image_centers,
        image_R_wc=base.image_R_wc,
        cameras=base.cameras,
    )
    receipt = write_risk_ply(
        black,
        tmp_path / "gray",
        localization=[{"query": "q1", "status": "GEOMETRY_WEAK", "x": 0.2, "y": 0.0, "z": 0.1}],
        sphere_radius=0.1,
        sphere_samples=8,
        filename="gray.ply",
    )
    visible = receipt["cloudcompare"]["visible_rgb"]
    assert visible["applied"] is True
    assert visible["override_count"] == 3
    assert visible["fallback_rgb"] == list(VISIBLE_MAP_RGB)
    cc_xyz, cc_rgb = _read_vertices(Path(receipt["ply"]))
    assert any(row.tolist() == list(VISIBLE_MAP_RGB) for row in cc_rgb[:3])
    weak = list(ISSUE_COLORS["heldout_geometry_weak"])
    assert any(row.tolist() == weak for row in cc_rgb[3:])
    _full_xyz, full_rgb = _read_vertices(Path(receipt["ply_full"]))
    assert full_rgb[:3].max() == 0
    legend = Path(tmp_path / "gray" / "LEGEND.md").read_text(encoding="utf-8")
    assert "background" in legend.lower()
    assert "point size" in legend.lower()


def test_marker_radius_floor_without_override(tmp_path: Path):
    receipt = write_risk_ply(_tiny_map(), tmp_path / "floor", filename="floor.ply")
    assert receipt["sphere_radius"] >= MARKER_RADIUS_ABS_FLOOR
    assert receipt["sphere_radius"] >= MARKER_RADIUS_DIAG_FRACTION * receipt["clip"]["robust_diagonal"]
    assert receipt["sphere_samples"] >= 96


def test_bad_finite_camera_does_not_dominate_clip(tmp_path: Path):
    base = _tiny_map()
    points = np.vstack([base.points_xyz, np.array([[1.0e6, 0.0, 0.0]])])
    rgb = np.vstack([base.point_rgb, np.array([[9, 9, 9]], dtype=np.uint8)])
    cameras = np.vstack([base.image_centers, np.array([[5.0e5, 5.0e5, 5.0e5]])])
    names = list(base.image_names) + ["bad.jpg"]
    n = len(points)
    spoiled = MapData(
        point_ids=np.arange(n),
        points_xyz=points,
        point_rgb=rgb,
        point_errors=np.full(n, 0.4),
        track_lengths=np.full(n, 2, dtype=int),
        track_image_ids=[np.array([0, 1]) for _ in range(n)],
        image_ids=np.arange(len(cameras)),
        image_names=names,
        image_camera_ids=np.zeros(len(cameras), dtype=int),
        image_centers=cameras,
        image_R_wc=np.repeat(np.eye(3)[None], len(cameras), axis=0),
        cameras=base.cameras,
    )
    receipt = write_risk_ply(spoiled, tmp_path / "badcam", sphere_samples=8, filename="badcam.ply")
    clip = receipt["clip"]
    assert clip["camera_diagonal_full"] > 1.0e5
    assert clip["camera_diagonal"] < 100.0
    assert clip["robust_diagonal"] < 100.0
    assert 3 in set(clip["excluded_point_ids"])
    cc_xyz, _cc_rgb = _read_vertices(Path(receipt["ply"]))
    assert cc_xyz.max() < 100.0
    assert receipt["marker_spheres_excluded"] >= 1
    assert receipt["marker_spheres_cloudcompare"] < receipt["marker_spheres"]
    assert clip["excluded_marker_count"] >= 1
    assert 4 in set(clip["excluded_marker_ids"])
    assert clip["excluded_marker_classes"].get("unverified_bridge_pose", 0) >= 1
    assert any(not row["in_cloudcompare_clip"] and row.get("image_name") == "bad.jpg" for row in receipt["markers"])
    full_xyz, _full_rgb = _read_vertices(Path(receipt["ply_full"]))
    assert full_xyz.max() > 1.0e5
    mesh_xyz, _mesh_rgb = _read_vertices(Path(receipt["ply_mesh"]))
    assert mesh_xyz.max() > 1.0e5


def test_extreme_zero_obs_marker_excluded_from_cloudcompare(tmp_path: Path):
    base = _tiny_map()
    cameras = np.vstack([base.image_centers, np.array([[5.0e5, 5.0e5, 5.0e5]])])
    spoiled = MapData(
        point_ids=base.point_ids,
        points_xyz=base.points_xyz,
        point_rgb=base.point_rgb,
        point_errors=base.point_errors,
        track_lengths=base.track_lengths,
        track_image_ids=base.track_image_ids,
        image_ids=np.arange(len(cameras)),
        image_names=list(base.image_names) + ["bad.jpg"],
        image_camera_ids=np.zeros(len(cameras), dtype=int),
        image_centers=cameras,
        image_R_wc=np.repeat(np.eye(3)[None], len(cameras), axis=0),
        cameras=base.cameras,
    )
    receipt = write_risk_ply(spoiled, tmp_path / "badmark", sphere_samples=8, filename="badmark.ply")
    clip = receipt["clip"]
    cc_xyz, _cc_rgb = _read_vertices(Path(receipt["ply"]))
    assert cc_xyz.max() < 100.0
    assert receipt["marker_spheres_cloudcompare"] == receipt["marker_spheres"] - 1
    assert clip["excluded_marker_count"] == 1
    assert clip["excluded_marker_ids"] == [4]
    assert clip["excluded_marker_classes"] == {"unverified_bridge_pose": 1}
    assert clip["excluded_markers"][0]["image_name"] == "bad.jpg"
    full_xyz, _full_rgb = _read_vertices(Path(receipt["ply_full"]))
    assert full_xyz.max() > 1.0e5
    mesh_xyz, _mesh_rgb = _read_vertices(Path(receipt["ply_mesh"]))
    assert mesh_xyz.max() > 1.0e5


def test_camera_nearest_spacing_ignores_input_order():
    rng = np.random.default_rng(0)
    path = np.cumsum(np.full((1205, 3), 0.25), axis=0)
    path += rng.normal(0.0, 0.01, size=path.shape)
    shuffled = path[rng.permutation(len(path))]
    spaced = camera_nearest_spacing(shuffled)
    ordered = camera_nearest_spacing(path)
    assert spaced == pytest.approx(ordered, rel=1e-6, abs=1e-9)
    sequential = float(np.median(np.linalg.norm(np.diff(shuffled, axis=0), axis=1)))
    assert sequential > 2.0 * spaced
    compact = robust_spatial_clip(
        path[:10],
        camera_xyz=np.vstack([path[:20], [[1.0e6, 0.0, 0.0]]]),
    )
    assert compact["camera_diagonal"] < 50.0
    assert compact["camera_diagonal_full"] > 1.0e5
    assert compact["pad"] < 50.0
    baseline = robust_spatial_clip(path[:10], camera_xyz=path)
    spoiled = robust_spatial_clip(path[:10], camera_xyz=np.vstack([path, [[1.0e6, 0.0, 0.0]]]))
    assert spoiled["camera_diagonal"] == pytest.approx(baseline["camera_diagonal"], rel=0.05)
    assert spoiled["camera_diagonal_full"] > 1.0e5
    assert spoiled["camera_method"] in {"distance_quantile", "distance_mad", "distance_mad_guarded"}



def test_write_risk_ply_instances_shared_sphere_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    extras = [
        {"issue_class": "coverage_hole", "x": 0.4, "y": 0.2, "z": 0.9},
        {"issue_class": "fim_weak", "x": 1.1, "y": -0.3, "z": 1.2},
        {"issue_class": "weak_region", "x": 250.0, "y": -80.0, "z": 40.0},
    ]
    samples = 24
    radius = 0.25
    unit_shell = sphere_points((0.0, 0.0, 0.0), 1.0, samples)
    unit_mesh, unit_faces = sphere_mesh((0.0, 0.0, 0.0), 1.0)
    calls = {"points": 0, "mesh": 0}
    real_points = sphere_points
    real_mesh = sphere_mesh

    def counted_points(*args, **kwargs):
        calls["points"] += 1
        return real_points(*args, **kwargs)

    def counted_mesh(*args, **kwargs):
        calls["mesh"] += 1
        return real_mesh(*args, **kwargs)

    monkeypatch.setattr("sfm_diagnosis.risk_ply.sphere_points", counted_points)
    monkeypatch.setattr("sfm_diagnosis.risk_ply.sphere_mesh", counted_mesh)

    receipt = write_risk_ply(
        _tiny_map(),
        tmp_path / "templates",
        extra_markers=extras,
        sphere_radius=radius,
        sphere_samples=samples,
        filename="templates.ply",
    )
    assert calls["points"] == 1
    assert calls["mesh"] == 1

    markers = receipt["markers"]
    n_markers = len(markers)
    assert n_markers >= len(extras)
    n_mesh = len(unit_mesh)
    n_faces = len(unit_faces)
    assert receipt["sphere_samples"] == samples
    assert receipt["marker_spheres"] == n_markers
    assert receipt["vertex_count_full"] == receipt["map_vertices"] + n_markers * samples
    assert receipt["mesh_vertex_count"] == n_markers * n_mesh
    assert receipt["mesh_face_count"] == n_markers * n_faces

    full_xyz, _ = _read_vertices(Path(receipt["ply_full"]))
    mesh_xyz, _ = _read_vertices(Path(receipt["ply_mesh"]))
    mesh_faces = _read_faces(Path(receipt["ply_mesh"]))
    marker_xyz = full_xyz[int(receipt["map_vertices"]) :]
    assert len(marker_xyz) == n_markers * samples
    assert len(mesh_xyz) == n_markers * n_mesh
    assert len(mesh_faces) == n_markers * n_faces

    written_shells: list[np.ndarray] = []
    for index, row in enumerate(markers):
        center = np.asarray((row["x"], row["y"], row["z"]), dtype=float)
        assert row["sphere_index"] == index
        shell = marker_xyz[index * samples : (index + 1) * samples]
        mesh = mesh_xyz[index * n_mesh : (index + 1) * n_mesh]
        faces = mesh_faces[index * n_faces : (index + 1) * n_faces]
        np.testing.assert_allclose(shell, unit_shell * radius + center, rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(mesh, unit_mesh * radius + center, rtol=0.0, atol=1e-5)
        np.testing.assert_array_equal(faces, unit_faces + index * n_mesh)
        written_shells.append(shell)

    extra_rows = markers[-len(extras) :]
    extra_centers = [
        np.asarray((row["x"], row["y"], row["z"]), dtype=float) for row in extra_rows
    ]
    for expected, row, center in zip(extras, extra_rows, extra_centers):
        assert row["issue_class"] == expected["issue_class"]
        np.testing.assert_allclose(center, (expected["x"], expected["y"], expected["z"]))
    assert not np.allclose(extra_centers[0], extra_centers[1])
    assert not np.allclose(extra_centers[1], extra_centers[2])
    first = written_shells[n_markers - 3]
    second = written_shells[n_markers - 2]
    third = written_shells[n_markers - 1]
    assert not np.allclose(first, second)
    assert not np.allclose(second, third)
    assert not np.allclose(first, third)

