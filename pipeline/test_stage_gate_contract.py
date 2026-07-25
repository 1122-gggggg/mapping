from __future__ import annotations

import json
import sqlite3
from argparse import Namespace
from pathlib import Path

from build_pipeline import (
    auto_package_verify_enabled,
    auto_system_verify_enabled,
    build_final_gate_command,
    build_holdout_localization_command,
    build_package_verify_command,
    build_production_replay_command,
    build_system_verify_command,
    field_validation_preflight_reasons,
    prepare_validation_videos_dir,
    split_build_validation_videos,
    final_gate_required,
    latest_link_for_handoff_profile,
    summary_overall_ok,
)
from stage_gate_contract import evaluate_build_run, write_summary_files


BUNDLE_METRICS = {
    "stage": "triangulate",
    "exists": True,
    "refs": 262,
    "unique_refs": 262,
    "ref_global_shape": [262, 8448],
    "meta": {
        "bundle_vpr": "megaloc",
        "tracking_metadata": True,
        "total_3d_anchored_kp": 325061,
        "mean_3d_anchored_per_ref": 1240.69,
    },
    "total_3d_anchored_kp": 325061,
    "mean_3d_anchored_per_ref": 1240.69,
}

VIEW_GRAPH_METRICS = {
    "stage": "pairs",
    "total_frames": 262,
    "pairs": 2046,
    "connected_components": 1,
    "largest_component": 262,
    "largest_component_ratio": 1.0,
    "isolated_images": 0,
    "relations": {"same_video": 1750, "cross_video": 296},
    "pair_kinds": {"temporal": 1750, "cross_topk": 295, "cross_grid": 1},
}

FRAME_MOTION_METRICS = {
    "stage": "extract",
    "frames": 262,
    "motion_gate": {
        "P1270127": {
            "total_before": 207,
            "kept": 107,
            "rejected": 100,
            "motion_classes": {"seed": 1, "pure_rotation": 95, "parallax": 87, "hover": 24},
        },
        "P1280128": {
            "total_before": 120,
            "kept": 51,
            "rejected": 69,
            "motion_classes": {"seed": 1, "parallax": 35, "pure_rotation": 77, "hover": 7},
        },
        "P1290129": {
            "total_before": 186,
            "kept": 104,
            "rejected": 82,
            "motion_classes": {"seed": 1, "pure_rotation": 96, "parallax": 84, "hover": 5},
        },
    },
}

PREFLIGHT_METRICS = {
    "stage": "preflight",
    "videos": 3,
    "target_resolution": {"width": 1920, "height": 1080},
    "disk_free_gb": 200.0,
    "tool_exists": {
        "ffmpeg": True,
        "ffprobe": True,
        "glomap": True,
        "python_sfm": True,
        "python_sfmdb": True,
        "python_mvroma": True,
        "template_repo": True,
    },
    "gpu": {"required": True, "ok": True},
    "camera": {
        "model": "FULL_OPENCV",
        "params": [
            1440.7279649640707,
            1437.2942620721813,
            1006.2251477118223,
            538.0787720175211,
            -0.016359355216362784,
            0.256336300878371,
            -0.006099082030819077,
            0.019509803298460405,
            -0.1198628127364991,
            0.0,
            0.0,
            0.0,
        ],
    },
}

MANIFEST_METRICS = {
    "stage": "manifest",
    "total_frames": 262,
    "image_count": 262,
    "camera_resolution_key": "1920x1080",
    "camera_model": "FULL_OPENCV",
}

SPARSE_MODEL_METRICS = {
    "stage": "glomap",
    "exists": True,
    "required": {"cameras.bin": True, "images.bin": True, "points3D.bin": True},
    "registered_images": 262,
    "points3D": 659024,
    "mean_reprojection_error": 0.879,
    "registered_ratio": 1.0,
    "points_per_registered_image": 2515.3,
}


POINT_CLOUD_VERTICES = 220000


def write_binary_rgb_ply(path: Path, vertex_count: int = POINT_CLOUD_VERTICES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        f.truncate(len(header) + vertex_count * 15)


def write_preflight_report(run_dir: Path) -> None:
    report = {
        "strict_gates": True,
        "strict_profile": "field_default",
        "videos": [
            {
                "path": str(run_dir / "inputs" / f"P12{idx}.MP4"),
                "exists": True,
                "codec_name": "h264",
                "width": 2688,
                "height": 1512,
                "fps": 23.976,
                "duration": 60.0 + idx,
                "nb_frames": 1440 + idx,
                "target_width": 1920,
                "target_height": 1080,
                "sanitized_stem": f"P12{idx}",
                "expected_extracted_frames": 80 + idx,
            }
            for idx in range(7, 10)
        ],
        "fps": 2.0,
        "target_resolution": {"width": 1920, "height": 1080},
        "camera": PREFLIGHT_METRICS["camera"],
        "tool_exists": PREFLIGHT_METRICS["tool_exists"],
        "disk": {"path": str(run_dir), "free_gb": 200.0, "free_bytes": 200 * 1024**3},
        "gpu": PREFLIGHT_METRICS["gpu"],
    }
    (run_dir / "preflight_report.json").write_text(json.dumps(report), encoding="utf-8")


def write_report_package(run_dir: Path) -> None:
    (run_dir / "build_config.json").write_text(json.dumps({
        "site_name": "test_site",
        "stages": "all",
        "handoff_profile": "field",
    }), encoding="utf-8")
    (run_dir / "stage_times.json").write_text(json.dumps({
        "total_seconds": 123.4,
        "stages": [
            {"stage": "preflight", "status": "success", "duration_seconds": 1.0},
            {"stage": "extract", "status": "success", "duration_seconds": 2.0},
            {"stage": "report", "status": "success", "duration_seconds": 0.5},
        ],
    }), encoding="utf-8")
    (run_dir / "build_report.json").write_text(json.dumps({
        "site_name": "test_site",
        "created_at": "2026-07-10 00:00:00",
        "outputs": {
            "glomap_model": str(run_dir / "glomap" / "0"),
            "rgb_point_cloud": str(run_dir / "deploy" / "map_rgb.ply"),
            "triangulated_bundle": str(run_dir / "deploy" / "reloc_map_xfeat_tri.pt"),
            "frame_manifest": str(run_dir / "frame_manifest.json"),
            "intrinsics": str(run_dir / "map_intrinsics.json"),
            "config": str(run_dir / "build_config.json"),
            "stage_times": str(run_dir / "stage_times.json"),
        },
        "parameters": {
            "site_name": "test_site",
            "backend": "glomap",
        },
        "stage_times": [
            {"stage": "preflight", "status": "success", "duration_seconds": 1.0},
            {"stage": "report", "status": "success", "duration_seconds": 0.5},
        ],
    }), encoding="utf-8")
    (run_dir / "BUILD_LOCALIZABLE_MAP_REPORT.md").write_text(
        "# Localizable Map Build Report: test_site\n\n"
        "## Outputs\n\n"
        "- `glomap_model`: glomap/0\n"
        "- `rgb_point_cloud`: deploy/map_rgb.ply\n"
        "- `triangulated_bundle`: deploy/reloc_map_xfeat_tri.pt\n",
        encoding="utf-8",
    )


def write_colmap_database(
    db_path: Path,
    *,
    image_count: int = 262,
    pair_count: int = 1100,
    keypoints_per_image: int = 1000,
    geometry_inliers: int = 200,
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute(
        "CREATE TABLE cameras "
        "(camera_id INTEGER PRIMARY KEY, model INTEGER, width INTEGER, height INTEGER, params BLOB, prior_focal_length INTEGER)"
    )
    cur.execute(
        "CREATE TABLE images "
        "(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
    )
    cur.execute(
        "CREATE TABLE keypoints "
        "(image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)"
    )
    cur.execute(
        "CREATE TABLE matches "
        "(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)"
    )
    cur.execute(
        "CREATE TABLE two_view_geometries "
        "(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB, config INTEGER, "
        "F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB)"
    )
    cur.execute("INSERT INTO cameras VALUES (1, 6, 1920, 1080, ?, 0)", (b"params",))
    for idx in range(1, image_count + 1):
        cur.execute("INSERT INTO images VALUES (?, ?, 1)", (idx, f"P1270127/{idx:06d}.jpg"))
        cur.execute("INSERT INTO keypoints VALUES (?, ?, 2, ?)", (idx, keypoints_per_image, b""))
    for idx in range(1, pair_count + 1):
        cur.execute("INSERT INTO matches VALUES (?, ?, 2, ?)", (idx, geometry_inliers, b""))
        cur.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, 2, ?, 7, NULL, NULL, NULL, NULL, NULL)",
            (idx, geometry_inliers, b""),
        )
    con.commit()
    con.close()


def write_gate(run_dir: Path, name: str, ok: bool = True, *, legacy: bool = False,
               metrics: dict | None = None) -> None:
    gates = run_dir / "gates"
    gates.mkdir(parents=True, exist_ok=True)
    payload = (
        {"gate": name, "passed": ok, "hard": True, "metrics": metrics or {}}
        if legacy
        else {"stage": name, "ok": ok, "reasons": [] if ok else ["failed for test"], "metrics": metrics or {}}
    )
    (gates / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def complete_current_style_run(run_dir: Path) -> None:
    for name in (
        "preflight",
        "extract",
        "manifest",
        "pairs",
        "mvroma",
        "aggregate",
        "db",
        "glomap",
        "color",
        "report",
    ):
        if name == "preflight":
            metrics = PREFLIGHT_METRICS
        elif name == "extract":
            metrics = FRAME_MOTION_METRICS
        elif name == "pairs":
            metrics = VIEW_GRAPH_METRICS
        elif name == "glomap":
            metrics = SPARSE_MODEL_METRICS
        elif name == "color":
            metrics = {
                "stage": "color",
                "rgb_ply_bytes": 1024 * 1024,
                "ply_vertices": POINT_CLOUD_VERTICES,
                "ply_vertex_ratio_vs_points3D": POINT_CLOUD_VERTICES / SPARSE_MODEL_METRICS["points3D"],
            }
        elif name == "manifest":
            metrics = MANIFEST_METRICS
        else:
            metrics = {"stage": name}
        write_gate(run_dir, name, metrics=metrics)
    write_preflight_report(run_dir)
    write_gate(run_dir, "triangulate", metrics=BUNDLE_METRICS)
    (run_dir / "frame_manifest.json").write_text(json.dumps({
        "total_frames": 262,
        "frames": [
            {"name": f"P1270127/{idx:06d}.jpg", "width": 1920, "height": 1080}
            for idx in range(262)
        ],
    }), encoding="utf-8")
    (run_dir / "map_intrinsics.json").write_text(json.dumps({
        "camera_mode": "PER_FOLDER",
        "intrinsics_by_resolution": {
            "1920x1080": {
                "model": "FULL_OPENCV",
                "params": [
                    1440.7279649640707,
                    1437.2942620721813,
                    1006.2251477118223,
                    538.0787720175211,
                    -0.016359355216362784,
                    0.256336300878371,
                    -0.006099082030819077,
                    0.019509803298460405,
                    -0.1198628127364991,
                    0.0,
                    0.0,
                    0.0,
                ],
            },
        },
    }), encoding="utf-8")
    deploy = run_dir / "deploy"
    deploy.mkdir()
    (deploy / "reloc_map_xfeat_tri.pt").write_bytes(b"bundle")
    write_binary_rgb_ply(deploy / "map_rgb.ply")
    write_colmap_database(run_dir / "glomap" / "0" / "database.db")
    (run_dir / "glomap" / "0").mkdir(parents=True, exist_ok=True)
    write_report_package(run_dir)


def write_production_preflight(replay_json: Path, *, ok: bool = True, max_frame_gap: int = 1) -> Path:
    preflight_json = replay_json.with_suffix(".preflight.json")
    preflight_json.write_text(json.dumps({
        "kind": "production_stream_preflight",
        "ok": ok,
        "reasons": [] if ok else ["frame index gap too large"],
        "metrics": {
            "direct_jpg_count": 40,
            "numeric_frame_count": 40,
            "max_frame_gap": max_frame_gap,
        },
    }), encoding="utf-8")
    return preflight_json


def test_current_style_complete_run_passes_one_click_contract(tmp_path: Path):
    complete_current_style_run(tmp_path)

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is True
    assert report["next_blocked_stage"] is None
    assert report["stages"]["preflight_quality"]["ok"] is True
    assert report["stages"]["frame_motion_quality"]["ok"] is True
    assert report["stages"]["intrinsics_manifest_quality"]["ok"] is True
    assert report["stages"]["view_graph_quality"]["ok"] is True
    assert report["stages"]["database_quality"]["ok"] is True
    db_metrics = report["stages"]["database_quality"]["evidence"][0]["metrics"]["database_quality"]
    assert db_metrics["images"] == 262
    assert db_metrics["two_view_geometries"] == 1100
    assert db_metrics["nonzero_two_view_ratio"] == 1.0
    assert report["stages"]["sfm_reconstruction_quality"]["ok"] is True
    assert report["stages"]["point_cloud_quality"]["ok"] is True
    ply_metrics = report["stages"]["point_cloud_quality"]["evidence"][0]["metrics"]["point_cloud_quality"]
    assert ply_metrics["vertices"] == POINT_CLOUD_VERTICES
    assert ply_metrics["has_rgb"] is True
    assert report["stages"]["localization_bundle"]["ok"] is True
    assert report["stages"]["localization_bundle"]["evidence"][0]["name"] == "triangulate"
    assert report["stages"]["report_package_quality"]["ok"] is True


def test_preflight_quality_requires_machine_readable_video_preflight(tmp_path: Path):
    complete_current_style_run(tmp_path)
    (tmp_path / "preflight_report.json").unlink()

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "preflight_quality"
    reasons = "; ".join(report["stages"]["preflight_quality"]["reasons"])
    assert "missing preflight_report.json" in reasons


def test_preflight_quality_rejects_missing_core_tool_and_weak_video(tmp_path: Path):
    complete_current_style_run(tmp_path)
    path = tmp_path / "preflight_report.json"
    data = json.loads(path.read_text())
    data["tool_exists"]["ffmpeg"] = False
    data["videos"][0]["exists"] = False
    data["videos"][1]["expected_extracted_frames"] = 3
    path.write_text(json.dumps(data), encoding="utf-8")

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "preflight_quality"
    reasons = "; ".join(report["stages"]["preflight_quality"]["reasons"])
    assert "preflight tool ffmpeg missing" in reasons
    assert "preflight video[0] exists=false" in reasons
    assert "preflight video[1] expected_extracted_frames=3 < 15" in reasons


def test_failed_optional_gate_blocks_promotion(tmp_path: Path):
    complete_current_style_run(tmp_path)
    write_gate(tmp_path, "doppelgangers", ok=False)

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "pair_filtering_optional"
    assert "failed for test" in report["stages"]["pair_filtering_optional"]["reasons"]


def test_legacy_gluemap_run_without_bundle_fails_localizable_contract(tmp_path: Path):
    for name in (
        "preflight",
        "selection_motion_quality",
        "intrinsics",
        "gluemap_pair_graph",
        "gluemap_database",
        "gluemap_model",
        "rgb_ply",
    ):
        if name == "gluemap_model":
            metrics = {**SPARSE_MODEL_METRICS, "stage": "gluemap_model"}
        elif name == "gluemap_pair_graph":
            metrics = {
                **VIEW_GRAPH_METRICS,
                "stage": "gluemap_pair_graph",
                "cross_sequence_pairs": 296,
            }
        elif name == "selection_motion_quality":
            metrics = {
                "total": 140,
                "selected": 125,
                "selected_ratio": 0.893,
                "groups": {
                    "P1090109_002": 16,
                    "P1100110_005": 18,
                    "P1110111": 19,
                    "P1120112": 26,
                    "P1130113": 26,
                    "P1140114": 20,
                },
                "motion_class_counts": {"seed": 6, "parallax": 99, "pure_rotation": 20},
                "motion_role_counts": {"triangulation": 105, "bridge_only": 20},
                "parallax_or_seed_ratio": 0.84,
                "hover_ratio": 0.0,
                "bridge_frames": 20,
            }
        elif name == "intrinsics":
            metrics = {"stage": "intrinsics"}
        else:
            metrics = None
        write_gate(tmp_path, name, legacy=True, metrics=metrics)
    write_preflight_report(tmp_path)
    (tmp_path / "frame_manifest.json").write_text(json.dumps({
        "total_frames": 125,
        "frames": [
            {"name": f"P1090109_002/{idx:06d}.jpg", "width": 1920, "height": 1080}
            for idx in range(125)
        ],
    }), encoding="utf-8")
    (tmp_path / "map_intrinsics.json").write_text(json.dumps({
        "camera_mode": "SHARED",
        "camera_model": "PINHOLE",
        "image_width": 1920,
        "image_height": 1080,
        "params": [1396.8086675255472, 1396.8086675255472, 960.0, 540.0],
        "variant": "no_undistort_official69",
        "source_model": "PINHOLE_OFFICIAL_HFOV",
        "source_width": 1280,
        "source_height": 720,
        "target_width": 1920,
        "target_height": 1080,
        "scale_x": 1.5,
        "scale_y": 1.5,
        "undistort_applied": False,
        "official_video_hfov_deg": 69.0,
    }), encoding="utf-8")
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    write_binary_rgb_ply(deploy / "map_rgb.ply")
    write_colmap_database(tmp_path / "glomap" / "0" / "database.db", image_count=125, pair_count=600)
    (tmp_path / "BUILD_GLUEMAP_REPORT.md").write_text("# legacy report\n", encoding="utf-8")

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "localization_bundle"
    assert "missing deploy/reloc_map_xfeat_tri.pt" in report["stages"]["localization_bundle"]["reasons"]
    assert report["stages"]["report_package"]["ok"] is True


def test_intrinsics_manifest_quality_blocks_bad_camera_params(tmp_path: Path):
    complete_current_style_run(tmp_path)
    (tmp_path / "map_intrinsics.json").write_text(json.dumps({
        "camera_mode": "SHARED",
        "camera_model": "PINHOLE",
        "image_width": 1920,
        "image_height": 1080,
        "params": [0.0, 1396.8, 2500.0, 540.0],
    }), encoding="utf-8")

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "intrinsics_manifest_quality"
    reasons = "; ".join(report["stages"]["intrinsics_manifest_quality"]["reasons"])
    assert "PINHOLE fx=0.0 must be positive" in reasons
    assert "PINHOLE cx=2500.0 outside image width 1920" in reasons


def test_intrinsics_manifest_quality_requires_files_not_only_gate_success(tmp_path: Path):
    complete_current_style_run(tmp_path)
    (tmp_path / "frame_manifest.json").unlink()
    (tmp_path / "map_intrinsics.json").unlink()

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "intrinsics_manifest_quality"
    reasons = "; ".join(report["stages"]["intrinsics_manifest_quality"]["reasons"])
    assert "missing frame_manifest.json" in reasons
    assert "missing map_intrinsics.json" in reasons


def test_frame_motion_quality_blocks_low_parallax_or_hover_dominated_selection(tmp_path: Path):
    complete_current_style_run(tmp_path)
    write_gate(tmp_path, "selection_motion_quality", legacy=True, metrics={
        "total": 200,
        "selected": 70,
        "selected_ratio": 0.35,
        "groups": {"A": 35, "B": 35},
        "parallax_or_seed_ratio": 0.40,
        "hover_ratio": 0.20,
        "bridge_frames": 0,
    })

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "frame_motion_quality"
    reasons = "; ".join(report["stages"]["frame_motion_quality"]["reasons"])
    assert "selected_ratio=0.350 < 0.650" in reasons
    assert "parallax_or_seed_ratio=0.400 < 0.650" in reasons
    assert "hover_ratio=0.200 > 0.050" in reasons
    assert "bridge_frames=0 for multi-group selection" in reasons


def test_frame_motion_quality_requires_metrics_not_only_extract_success(tmp_path: Path):
    complete_current_style_run(tmp_path)
    write_gate(tmp_path, "extract", metrics={"stage": "extract"})

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "frame_motion_quality"
    reasons = "; ".join(report["stages"]["frame_motion_quality"]["reasons"])
    assert "extract frames missing" in reasons
    assert "extract motion_gate missing" in reasons


def test_view_graph_quality_blocks_disconnected_or_unbridged_graph(tmp_path: Path):
    complete_current_style_run(tmp_path)
    write_gate(tmp_path, "pairs", metrics={
        "stage": "pairs",
        "total_frames": 50,
        "pairs": 80,
        "connected_components": 3,
        "largest_component_ratio": 0.55,
        "isolated_images": 5,
        "cross_sequence_pairs": 0,
    })

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "view_graph_quality"
    reasons = "; ".join(report["stages"]["view_graph_quality"]["reasons"])
    assert "pairs=80 < 500" in reasons
    assert "pairs_per_frame=1.60 < 4.00" in reasons
    assert "connected_components=3 > 1" in reasons
    assert "largest_component_ratio=0.550 < 0.900" in reasons
    assert "isolated_images=5 > 0" in reasons
    assert "cross_video_or_sequence_pairs=0" in reasons


def test_view_graph_quality_requires_bridge_and_connectivity_metrics(tmp_path: Path):
    complete_current_style_run(tmp_path)
    write_gate(tmp_path, "pairs", metrics={
        "stage": "pairs",
        "pairs": 8995,
    })

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "view_graph_quality"
    reasons = "; ".join(report["stages"]["view_graph_quality"]["reasons"])
    assert "largest_component_ratio missing" in reasons
    assert "cross_video_or_sequence_pairs missing" in reasons


def test_sfm_reconstruction_quality_blocks_sparse_or_noisy_models(tmp_path: Path):
    complete_current_style_run(tmp_path)
    write_gate(tmp_path, "glomap", metrics={
        "stage": "glomap",
        "exists": True,
        "registered_images": 55,
        "registered_ratio": 0.72,
        "points3D": 12000,
        "mean_reprojection_error": 2.8,
    })

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "sfm_reconstruction_quality"
    reasons = "; ".join(report["stages"]["sfm_reconstruction_quality"]["reasons"])
    assert "registered_images=55 < 60" in reasons
    assert "registered_ratio=0.720 < 0.800" in reasons
    assert "points3D=12000 < 30000" in reasons
    assert "points_per_registered=218.2 < 500.0" in reasons
    assert "mean_reprojection_error=2.800 > 2.000" in reasons


def test_database_quality_blocks_empty_sqlite_even_when_db_gate_passes(tmp_path: Path):
    complete_current_style_run(tmp_path)
    db_path = tmp_path / "glomap" / "0" / "database.db"
    db_path.unlink()
    con = sqlite3.connect(str(db_path))
    con.execute("CREATE TABLE unrelated (id INTEGER)")
    con.commit()
    con.close()

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "database_quality"
    reasons = "; ".join(report["stages"]["database_quality"]["reasons"])
    assert "missing required database tables" in reasons


def test_database_quality_rejects_missing_keypoints_and_weak_geometry(tmp_path: Path):
    complete_current_style_run(tmp_path)
    db_path = tmp_path / "glomap" / "0" / "database.db"
    db_path.unlink()
    write_colmap_database(
        db_path,
        image_count=262,
        pair_count=100,
        keypoints_per_image=0,
        geometry_inliers=0,
    )

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "database_quality"
    reasons = "; ".join(report["stages"]["database_quality"]["reasons"])
    assert "images_with_keypoints=0 < images=262" in reasons
    assert "matches pairs=100 < 1048" in reasons
    assert "two_view_geometries nonzero_ratio=0.000 < 0.900" in reasons


def test_sfm_reconstruction_quality_requires_metrics_not_only_gate_success(tmp_path: Path):
    complete_current_style_run(tmp_path)
    write_gate(tmp_path, "glomap", metrics={"stage": "glomap", "exists": True})

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "sfm_reconstruction_quality"
    reasons = "; ".join(report["stages"]["sfm_reconstruction_quality"]["reasons"])
    assert "glomap registered_images missing" in reasons
    assert "glomap points3D missing" in reasons
    assert "glomap mean_reprojection_error missing" in reasons


def test_point_cloud_quality_blocks_tiny_or_non_rgb_ply(tmp_path: Path):
    complete_current_style_run(tmp_path)
    (tmp_path / "deploy" / "map_rgb.ply").write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 10\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n",
        encoding="ascii",
    )

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "point_cloud_quality"
    reasons = "; ".join(report["stages"]["point_cloud_quality"]["reasons"])
    assert "PLY vertices=10 < 10000" in reasons
    assert "PLY missing RGB vertex properties" in reasons


def test_point_cloud_quality_requires_vertices_to_cover_sparse_model(tmp_path: Path):
    complete_current_style_run(tmp_path)
    write_binary_rgb_ply(tmp_path / "deploy" / "map_rgb.ply", vertex_count=1000)

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "point_cloud_quality"
    reasons = "; ".join(report["stages"]["point_cloud_quality"]["reasons"])
    assert "PLY vertex_ratio_vs_points3D=0.002 < 0.300" in reasons


def test_localization_bundle_requires_metadata_not_only_file(tmp_path: Path):
    complete_current_style_run(tmp_path)
    write_gate(tmp_path, "triangulate", metrics={"stage": "triangulate"})

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "localization_bundle"
    reasons = "; ".join(report["stages"]["localization_bundle"]["reasons"])
    assert "triangulate refs=0" in reasons
    assert "triangulate ref_global_shape missing" in reasons


def test_localization_bundle_requires_tracking_metadata_and_anchors(tmp_path: Path):
    complete_current_style_run(tmp_path)
    write_gate(tmp_path, "triangulate", metrics={
        "stage": "triangulate",
        "exists": True,
        "refs": 10,
        "unique_refs": 10,
        "ref_global_shape": [10, 8448],
        "meta": {"bundle_vpr": "megaloc"},
        "total_3d_anchored_kp": 0,
        "mean_3d_anchored_per_ref": 0.0,
    })

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "localization_bundle"
    reasons = "; ".join(report["stages"]["localization_bundle"]["reasons"])
    assert "tracking_metadata missing" in reasons
    assert "total_3d_anchored_kp=0" in reasons
    assert "mean_3d_anchored_per_ref=0.0" in reasons


def test_report_package_quality_requires_machine_readable_provenance(tmp_path: Path):
    complete_current_style_run(tmp_path)
    (tmp_path / "build_report.json").unlink()
    (tmp_path / "stage_times.json").unlink()

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "report_package_quality"
    reasons = "; ".join(report["stages"]["report_package_quality"]["reasons"])
    assert "missing build_report.json" in reasons
    assert "missing stage_times.json" in reasons


def test_report_package_quality_rejects_failed_stage_timing(tmp_path: Path):
    complete_current_style_run(tmp_path)
    (tmp_path / "stage_times.json").write_text(json.dumps({
        "total_seconds": 3.0,
        "stages": [
            {"stage": "preflight", "status": "success", "duration_seconds": 1.0},
            {"stage": "report", "status": "failed", "duration_seconds": 2.0, "error": "report crashed"},
        ],
    }), encoding="utf-8")

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "report_package_quality"
    reasons = "; ".join(report["stages"]["report_package_quality"]["reasons"])
    assert "stage_times latest report status=failed" in reasons


def test_missing_required_gate_reports_first_blocker(tmp_path: Path):
    write_gate(tmp_path, "preflight")
    write_preflight_report(tmp_path)
    write_gate(tmp_path, "extract", metrics=FRAME_MOTION_METRICS)
    write_gate(tmp_path, "manifest")
    (tmp_path / "frame_manifest.json").write_text(json.dumps({
        "total_frames": 262,
        "frames": [
            {"name": f"P1270127/{idx:06d}.jpg", "width": 1920, "height": 1080}
            for idx in range(262)
        ],
    }), encoding="utf-8")
    (tmp_path / "map_intrinsics.json").write_text(json.dumps({
        "camera_mode": "PER_FOLDER",
        "intrinsics_by_resolution": {
            "1920x1080": {
                "model": "FULL_OPENCV",
                "params": [
                    1440.7279649640707,
                    1437.2942620721813,
                    1006.2251477118223,
                    538.0787720175211,
                    -0.016359355216362784,
                    0.256336300878371,
                    -0.006099082030819077,
                    0.019509803298460405,
                ],
            },
        },
    }), encoding="utf-8")

    report = evaluate_build_run(tmp_path)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "pair_graph"
    assert "missing required gate: pairs or gluemap_pair_graph" in report["stages"]["pair_graph"]["reasons"]


def test_summary_writer_creates_machine_and_operator_reports(tmp_path: Path):
    complete_current_style_run(tmp_path)
    report = evaluate_build_run(tmp_path)

    json_out, md_out = write_summary_files(report, tmp_path / "BUILD_GATE_SUMMARY.json", tmp_path / "BUILD_GATE_SUMMARY.md")

    saved = json.loads(json_out.read_text(encoding="utf-8"))
    text = md_out.read_text(encoding="utf-8")
    assert saved["overall_ok"] is True
    assert "# One-Click Map Build Gate Summary" in text
    assert "| localization_bundle | PASS |" in text


def test_build_pipeline_runs_final_gate_only_for_complete_builds():
    assert final_gate_required("all") is True
    assert final_gate_required("extract,pairs,report") is True
    assert final_gate_required("extract,pairs") is False


def test_build_pipeline_forwards_external_final_gate_inputs(tmp_path: Path):
    args = Namespace(
        python="/usr/bin/python3.12",
        handoff_profile="field",
        final_gate_localization_json=["holdout_a.json", "holdout_b.json"],
        final_gate_production_json="production.json",
        final_gate_package_verify_json="package_verify.json",
        final_gate_system_verify_json="system_verify.json",
        require_final_localization=True,
        require_final_production=True,
        require_final_package_verify=True,
        require_final_system_verify=True,
        final_gate_min_success=0.91,
        final_gate_max_ok_to_fail=0,
        final_gate_max_final_fail_run=25,
        final_gate_min_production_success=0.92,
        final_gate_max_production_wall_p90=120.0,
        final_gate_min_production_frames=50,
        final_gate_max_production_fail_run=12,
        final_gate_min_production_inliers_p5=35.0,
        allow_final_gate_fail=False,
    )

    cmd = build_final_gate_command(args, tmp_path)

    assert "--localization-json" in cmd
    assert cmd.count("--localization-json") == 2
    assert "--require-localization" in cmd
    assert "--require-production" in cmd
    assert "--package-verify-json" in cmd
    assert "package_verify.json" in cmd
    assert "--system-verify-json" in cmd
    assert "system_verify.json" in cmd
    assert "--max-production-wall-p90" in cmd
    assert "120.0" in cmd
    assert "--min-production-frames" in cmd
    assert "50" in cmd
    assert "--max-production-fail-run" in cmd
    assert "12" in cmd
    assert "--min-production-inliers-p5" in cmd
    assert "35.0" in cmd


def test_build_pipeline_field_handoff_requires_external_evidence_by_default(tmp_path: Path):
    args = Namespace(
        python="/usr/bin/python3.12",
        handoff_profile="field",
        final_gate_localization_json=[],
        final_gate_production_json="",
        final_gate_package_verify_json="",
        final_gate_system_verify_json="",
        require_final_localization=False,
        require_final_production=False,
        require_final_package_verify=False,
        require_final_system_verify=False,
        final_gate_min_success=0.90,
        final_gate_max_ok_to_fail=0,
        final_gate_max_final_fail_run=30,
        final_gate_min_production_success=0.90,
        final_gate_max_production_wall_p90=0.0,
        final_gate_min_production_frames=30,
        final_gate_max_production_fail_run=30,
        final_gate_min_production_inliers_p5=30.0,
        allow_final_gate_fail=False,
    )

    cmd = build_final_gate_command(args, tmp_path)

    assert "--require-localization" in cmd
    assert "--require-production" in cmd
    assert "--require-package-verify" in cmd
    assert "--require-system-verify" in cmd
    assert "--min-production-frames" in cmd
    assert cmd[cmd.index("--min-production-frames") + 1] == "30"
    assert "--min-production-inliers-p5" in cmd
    assert cmd[cmd.index("--min-production-inliers-p5") + 1] == "30.0"


def test_build_pipeline_candidate_handoff_keeps_external_evidence_optional(tmp_path: Path):
    args = Namespace(
        python="/usr/bin/python3.12",
        handoff_profile="candidate",
        final_gate_localization_json=[],
        final_gate_production_json="",
        final_gate_package_verify_json="",
        final_gate_system_verify_json="",
        require_final_localization=False,
        require_final_production=False,
        require_final_package_verify=False,
        require_final_system_verify=False,
        final_gate_min_success=0.90,
        final_gate_max_ok_to_fail=0,
        final_gate_max_final_fail_run=30,
        final_gate_min_production_success=0.90,
        final_gate_max_production_wall_p90=0.0,
        final_gate_min_production_frames=30,
        final_gate_max_production_fail_run=30,
        final_gate_min_production_inliers_p5=30.0,
        allow_final_gate_fail=False,
    )

    cmd = build_final_gate_command(args, tmp_path)

    assert "--require-localization" not in cmd
    assert "--require-production" not in cmd
    assert "--require-package-verify" not in cmd
    assert "--require-system-verify" not in cmd


def test_build_pipeline_field_handoff_auto_generates_package_and_system_verify(tmp_path: Path):
    args = Namespace(
        python="/usr/bin/python3.12",
        handoff_profile="field",
        final_gate_localization_json=["holdout.json"],
        final_gate_production_json="production.json",
        final_gate_package_verify_json="",
        final_gate_system_verify_json="",
        require_final_localization=False,
        require_final_production=False,
        require_final_package_verify=False,
        require_final_system_verify=False,
        final_gate_min_success=0.90,
        final_gate_max_ok_to_fail=0,
        final_gate_max_final_fail_run=30,
        final_gate_min_production_success=0.90,
        final_gate_max_production_wall_p90=0.0,
        final_gate_min_production_frames=30,
        final_gate_max_production_fail_run=30,
        final_gate_min_production_inliers_p5=30.0,
        allow_final_gate_fail=False,
    )

    package_json = tmp_path / "package_verify.json"
    pre_summary = tmp_path / "BUILD_GATE_SUMMARY.pre_system_verify.json"
    system_json = tmp_path / "system_verify.json"

    assert auto_package_verify_enabled(args) is True
    assert auto_system_verify_enabled(args) is True

    package_cmd = build_package_verify_command(args, package_json)
    assert str(package_json) in package_cmd
    assert "verify_package.py" in package_cmd[1]

    pre_cmd = build_final_gate_command(
        args,
        tmp_path,
        out_json=pre_summary,
        package_verify_json=package_json,
        system_verify_json="",
        require_system_verify=False,
    )
    assert "--out-json" in pre_cmd
    assert str(pre_summary) in pre_cmd
    assert "--package-verify-json" in pre_cmd
    assert str(package_json) in pre_cmd
    assert "--require-system-verify" not in pre_cmd
    assert "--system-verify-json" not in pre_cmd

    system_cmd = build_system_verify_command(args, pre_summary, system_json)
    assert "--build-summary-json" in system_cmd
    assert str(pre_summary) in system_cmd
    assert "--json-out" in system_cmd
    assert str(system_json) in system_cmd
    assert "--skip-runtime" in system_cmd
    assert "--allow-blocked" in system_cmd

    final_cmd = build_final_gate_command(
        args,
        tmp_path,
        package_verify_json=package_json,
        system_verify_json=system_json,
    )
    assert "--require-system-verify" in final_cmd
    assert "--system-verify-json" in final_cmd
    assert str(system_json) in final_cmd


def test_build_pipeline_explicit_system_verify_disables_auto_generation():
    args = Namespace(
        handoff_profile="field",
        final_gate_package_verify_json="package_verify.json",
        final_gate_system_verify_json="system_verify.json",
    )

    assert auto_package_verify_enabled(args) is False
    assert auto_system_verify_enabled(args) is False


def test_build_pipeline_reserves_last_video_for_auto_field_validation(tmp_path: Path):
    args = Namespace(
        handoff_profile="field",
        videos=["A.MP4", "B.MP4", "C.MP4"],
        validation_videos=[],
        auto_validation_holdout_count=1,
        skip_auto_validation_evidence=False,
        final_gate_localization_json=[],
        final_gate_production_json="",
    )

    build_videos, validation_videos = split_build_validation_videos(args)

    assert build_videos == ["A.MP4", "B.MP4"]
    assert validation_videos == ["C.MP4"]


def test_build_pipeline_does_not_reserve_when_validation_evidence_supplied():
    args = Namespace(
        handoff_profile="field",
        videos=["A.MP4", "B.MP4"],
        validation_videos=[],
        auto_validation_holdout_count=1,
        skip_auto_validation_evidence=False,
        final_gate_localization_json=["holdout.json"],
        final_gate_production_json="production.json",
    )

    build_videos, validation_videos = split_build_validation_videos(args)

    assert build_videos == ["A.MP4", "B.MP4"]
    assert validation_videos == []


def test_build_pipeline_builds_auto_holdout_and_production_commands(tmp_path: Path):
    args = Namespace(
        python="/usr/bin/python3.12",
        final_gate_min_success=0.91,
        final_gate_max_ok_to_fail=0,
        final_gate_max_final_fail_run=20,
        final_gate_min_production_success=0.92,
        final_gate_max_production_wall_p90=120.0,
        final_gate_min_production_frames=30,
        final_gate_max_production_fail_run=10,
        final_gate_min_production_inliers_p5=35.0,
        auto_validation_stride=8,
        auto_validation_resize="1280x720",
        auto_production_video_stride=8,
        auto_production_video_limit=300,
        auto_production_resize_width=1280,
    )
    validation_dir = tmp_path / "validation_videos"
    holdout_json = tmp_path / "holdout_localization.json"
    production_json = tmp_path / "production_replay.json"
    bundle = tmp_path / "deploy" / "reloc_map_xfeat_tri.pt"
    model = tmp_path / "glomap" / "0"
    video = validation_dir / "C.MP4"

    holdout_cmd = build_holdout_localization_command(
        args,
        bundle=bundle,
        validation_dir=validation_dir,
        out_json=holdout_json,
    )
    assert "--mode" in holdout_cmd and "compare" in holdout_cmd
    assert "--base" in holdout_cmd and str(bundle) in holdout_cmd
    assert "--final" in holdout_cmd and holdout_cmd.count(str(bundle)) == 2
    assert "--base-megaloc-cache" in holdout_cmd
    assert holdout_cmd[holdout_cmd.index("--base-megaloc-cache") + 1] == ""
    assert "--test-dir" in holdout_cmd and str(validation_dir) in holdout_cmd
    assert "--out-json" in holdout_cmd and str(holdout_json) in holdout_cmd

    production_cmd = build_production_replay_command(
        args,
        bundle=bundle,
        model=model,
        query_video=video,
        out_json=production_json,
    )
    assert "--mode" in production_cmd and "production-stream" in production_cmd
    assert "--bundle" in production_cmd and str(bundle) in production_cmd
    assert "--model" in production_cmd and str(model) in production_cmd
    assert "--query-video" in production_cmd and str(video) in production_cmd
    assert "--query-video-stride" in production_cmd and "8" in production_cmd
    assert "--query-video-limit" in production_cmd and "300" in production_cmd
    assert "--min-production-inliers-p5" in production_cmd and "35.0" in production_cmd


def test_build_pipeline_field_validation_preflight_blocks_single_video_without_evidence():
    args = Namespace(
        handoff_profile="field",
        image_root="",
        videos=["A.MP4"],
        final_gate_localization_json=[],
        final_gate_production_json="",
        skip_auto_validation_evidence=False,
    )

    reasons = field_validation_preflight_reasons(args, build_videos=["A.MP4"], validation_videos=[])

    assert "field handoff needs validation video for auto localization evidence" in reasons
    assert "field handoff needs validation video for auto production replay evidence" in reasons


def test_build_pipeline_field_validation_preflight_passes_with_reserved_video(tmp_path: Path):
    build_video = tmp_path / "build.MP4"
    validation_video = tmp_path / "validation.MP4"
    build_video.write_bytes(b"build")
    validation_video.write_bytes(b"validation")
    args = Namespace(
        handoff_profile="field",
        image_root="",
        videos=[str(build_video), str(validation_video)],
        final_gate_localization_json=[],
        final_gate_production_json="",
        skip_auto_validation_evidence=False,
    )

    reasons = field_validation_preflight_reasons(
        args,
        build_videos=[str(build_video)],
        validation_videos=[str(validation_video)],
    )

    assert reasons == []


def test_build_pipeline_field_validation_preflight_allows_explicit_evidence_without_videos():
    args = Namespace(
        handoff_profile="field",
        image_root="frames",
        videos=[],
        final_gate_localization_json=["holdout.json"],
        final_gate_production_json="production.json",
        skip_auto_validation_evidence=False,
    )

    reasons = field_validation_preflight_reasons(args, build_videos=[], validation_videos=[])

    assert reasons == []


def test_build_pipeline_field_validation_preflight_rejects_missing_validation_file(tmp_path: Path):
    missing_video = tmp_path / "missing.MP4"
    args = Namespace(
        handoff_profile="field",
        image_root="",
        videos=["A.MP4", str(missing_video)],
        final_gate_localization_json=[],
        final_gate_production_json="",
        skip_auto_validation_evidence=False,
    )

    reasons = field_validation_preflight_reasons(
        args,
        build_videos=["A.MP4"],
        validation_videos=[str(missing_video)],
    )

    assert f"validation video missing or empty: {missing_video}" in reasons


def test_build_pipeline_validation_symlinks_are_visible_to_eval_stream_core(tmp_path: Path):
    source = tmp_path / "flight_lowercase.mp4"
    source.write_bytes(b"video")
    validation_dir = tmp_path / "validation_videos"

    prepare_validation_videos_dir([str(source)], validation_dir)

    assert sorted(path.name for path in validation_dir.glob("*.MP4")) == ["validation_00.MP4"]
    assert (validation_dir / "validation_00.MP4").resolve() == source


def test_build_pipeline_summary_overall_ok_requires_true_json(tmp_path: Path):
    summary = tmp_path / "BUILD_GATE_SUMMARY.json"

    summary.write_text(json.dumps({"overall_ok": False}), encoding="utf-8")
    assert summary_overall_ok(summary) is False

    summary.write_text(json.dumps({"overall_ok": True}), encoding="utf-8")
    assert summary_overall_ok(summary) is True

    summary.write_text("{not json", encoding="utf-8")
    assert summary_overall_ok(summary) is False


def test_build_pipeline_latest_link_is_profile_specific():
    field_args = Namespace(handoff_profile="field")
    candidate_args = Namespace(handoff_profile="candidate")

    assert latest_link_for_handoff_profile(field_args).name == "latest_build"
    assert latest_link_for_handoff_profile(candidate_args).name == "latest_candidate_build"


def test_holdout_localization_artifact_is_required_when_requested(tmp_path: Path):
    complete_current_style_run(tmp_path)

    report = evaluate_build_run(tmp_path, require_localization=True)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "holdout_localization"
    assert "missing localization validation JSON" in report["stages"]["holdout_localization"]["reasons"]


def test_holdout_localization_passes_baseline_improved_case(tmp_path: Path):
    complete_current_style_run(tmp_path)
    eval_json = tmp_path / "holdout.json"
    eval_json.write_text(json.dumps({
        "rows": [
            {
                "set": "P1230123",
                "n": 100,
                "base_success": 0.88,
                "final_success": 0.895,
                "ok_to_fail": 0,
                "base_max_fail_run": 31,
                "final_max_fail_run": 28,
            }
        ]
    }), encoding="utf-8")

    report = evaluate_build_run(tmp_path, localization_jsons=[eval_json], require_localization=True)

    assert report["overall_ok"] is True
    assert report["stages"]["holdout_localization"]["ok"] is True


def test_holdout_localization_fails_empty_or_regressed_rows(tmp_path: Path):
    complete_current_style_run(tmp_path)
    eval_json = tmp_path / "bad_holdout.json"
    eval_json.write_text(json.dumps({
        "rows": [
            {
                "set": "P1260126",
                "n": 50,
                "base_success": 1.0,
                "final_success": 0.8,
                "ok_to_fail": 3,
                "base_max_fail_run": 0,
                "final_max_fail_run": 35,
            }
        ]
    }), encoding="utf-8")

    report = evaluate_build_run(tmp_path, localization_jsons=[eval_json], require_localization=True)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "holdout_localization"
    assert any("P1260126" in reason for reason in report["stages"]["holdout_localization"]["reasons"])


def test_production_replay_quality_gate_checks_success_and_latency(tmp_path: Path):
    complete_current_style_run(tmp_path)
    replay_json = tmp_path / "production.json"
    write_production_preflight(replay_json)
    replay_json.write_text(json.dumps({
        "summary": {
            "n": 10,
            "success": 9,
            "success_rate": 0.9,
            "wall_ms": {"p90": 80.0},
        },
        "rows": [{"success": True, "inliers": 100} for _ in range(10)],
    }), encoding="utf-8")

    report = evaluate_build_run(
        tmp_path,
        production_json=replay_json,
        require_production=True,
        max_production_wall_p90=100.0,
    )

    assert report["overall_ok"] is True
    assert report["stages"]["production_replay"]["ok"] is True


def test_production_replay_requires_paired_preflight_json(tmp_path: Path):
    complete_current_style_run(tmp_path)
    replay_json = tmp_path / "production.json"
    replay_json.write_text(json.dumps({
        "summary": {
            "n": 10,
            "success": 10,
            "success_rate": 1.0,
            "wall_ms": {"p90": 80.0},
        },
        "rows": [{"success": True, "inliers": 100} for _ in range(10)],
    }), encoding="utf-8")

    report = evaluate_build_run(tmp_path, production_json=replay_json, require_production=True)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "production_replay"
    assert any(
        "missing paired production preflight JSON" in reason
        for reason in report["stages"]["production_replay"]["reasons"]
    )


def test_production_replay_rejects_failed_paired_preflight_json(tmp_path: Path):
    complete_current_style_run(tmp_path)
    replay_json = tmp_path / "production.json"
    write_production_preflight(replay_json, ok=False, max_frame_gap=1944)
    replay_json.write_text(json.dumps({
        "summary": {
            "n": 10,
            "success": 10,
            "success_rate": 1.0,
            "wall_ms": {"p90": 80.0},
        },
        "rows": [{"success": True, "inliers": 100} for _ in range(10)],
    }), encoding="utf-8")

    report = evaluate_build_run(tmp_path, production_json=replay_json, require_production=True)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "production_replay"
    reasons = "; ".join(report["stages"]["production_replay"]["reasons"])
    assert "paired preflight ok=false" in reasons
    assert "frame index gap too large" in reasons


def test_production_replay_blocks_short_or_fragile_streams(tmp_path: Path):
    complete_current_style_run(tmp_path)
    replay_json = tmp_path / "production_fragile.json"
    write_production_preflight(replay_json)
    replay_json.write_text(json.dumps({
        "summary": {
            "n": 5,
            "success": 5,
            "success_rate": 1.0,
            "wall_ms": {"p90": 80.0},
        },
        "rows": [
            {"success": True, "inliers": 10},
            {"success": False, "inliers": 0},
            {"success": False, "inliers": 0},
            {"success": False, "inliers": 0},
            {"success": True, "inliers": 90},
        ],
    }), encoding="utf-8")

    report = evaluate_build_run(
        tmp_path,
        production_json=replay_json,
        require_production=True,
        min_production_frames=30,
        max_production_fail_run=2,
        min_production_inliers_p5=30.0,
    )

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "production_replay"
    reasons = "; ".join(report["stages"]["production_replay"]["reasons"])
    assert "n=5 < min_frames=30" in reasons
    assert "max_failure_run=3 > 2" in reasons
    assert "inliers_p5=" in reasons


def test_production_replay_reports_preflight_failures(tmp_path: Path):
    complete_current_style_run(tmp_path)
    preflight_json = tmp_path / "production_preflight.json"
    preflight_json.write_text(json.dumps({
        "kind": "production_stream_preflight",
        "ok": False,
        "reasons": [
            "MegaLoc weights missing: no snapshots/*/model.safetensors under cache",
            "MegaLoc weights incomplete: blobs/model.incomplete",
        ],
        "metrics": {"direct_jpg_count": 3},
    }), encoding="utf-8")

    report = evaluate_build_run(tmp_path, production_json=preflight_json, require_production=True)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "production_replay"
    reasons = "; ".join(report["stages"]["production_replay"]["reasons"])
    assert "production replay did not run; preflight-only artifact" in reasons
    assert "MegaLoc weights missing" in reasons


def test_production_replay_preflight_success_is_not_a_replay_pass(tmp_path: Path):
    complete_current_style_run(tmp_path)
    preflight_json = tmp_path / "production_preflight_ok.json"
    preflight_json.write_text(json.dumps({
        "kind": "production_stream_preflight",
        "ok": True,
        "reasons": [],
        "metrics": {"direct_jpg_count": 3},
    }), encoding="utf-8")

    report = evaluate_build_run(tmp_path, production_json=preflight_json, require_production=True)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "production_replay"
    assert "production replay did not run; preflight-only artifact" in report["stages"]["production_replay"]["reasons"]


def test_system_verify_optional_blocker_does_not_fail_required_gate(tmp_path: Path):
    complete_current_style_run(tmp_path)
    system_json = tmp_path / "system_verify.json"
    system_json.write_text(json.dumps({
        "status": "BLOCKED",
        "checks": [
            {"name": "package_verify_json", "status": "PASS", "required": True},
            {"name": "sphinx_anafi_firmware_launch", "status": "BLOCKED", "required": False},
        ],
    }), encoding="utf-8")

    report = evaluate_build_run(tmp_path, system_verify_json=system_json, require_system_verify=True)

    assert report["overall_ok"] is True
    assert report["stages"]["system_verify"]["status"] == "PASS"


def test_system_verify_required_failure_blocks_promotion(tmp_path: Path):
    complete_current_style_run(tmp_path)
    system_json = tmp_path / "system_verify_bad.json"
    system_json.write_text(json.dumps({
        "status": "FAIL",
        "checks": [
            {"name": "base_localization_bundle", "status": "FAIL", "required": True},
        ],
    }), encoding="utf-8")

    report = evaluate_build_run(tmp_path, system_verify_json=system_json, require_system_verify=True)

    assert report["overall_ok"] is False
    assert report["next_blocked_stage"] == "system_verify"
    assert "base_localization_bundle" in "; ".join(report["stages"]["system_verify"]["reasons"])
