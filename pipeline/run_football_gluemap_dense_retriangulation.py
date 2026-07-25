#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any

import pycolmap


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log(message: str) -> None:
    print(f"[football_gluemap_dense] {message}", flush=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def database_stats(path: Path) -> dict[str, Any]:
    con = sqlite3.connect(str(path))
    cur = con.cursor()
    stats: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "user_version": cur.execute("pragma user_version").fetchone()[0],
    }
    for table in ("cameras", "images", "keypoints", "matches", "two_view_geometries"):
        stats[table] = int(cur.execute(f"select count(*) from {table}").fetchone()[0])
    row = cur.execute("select min(rows), avg(rows), max(rows) from two_view_geometries").fetchone()
    stats["two_view_geometry_rows"] = {
        "min": int(row[0] or 0),
        "mean": float(row[1] or 0.0),
        "max": int(row[2] or 0),
    }
    con.close()
    return stats


def configure_options(args: argparse.Namespace) -> pycolmap.IncrementalPipelineOptions:
    opts = pycolmap.IncrementalPipelineOptions()
    opts.min_num_matches = int(args.min_num_matches)
    opts.ba_refine_focal_length = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    opts.ba_global_max_num_iterations = int(args.ba_global_max_num_iterations)
    opts.ba_local_max_num_iterations = int(args.ba_local_max_num_iterations)
    opts.fix_existing_frames = bool(args.fix_existing_frames)
    opts.mapper.fix_existing_frames = bool(args.fix_existing_frames)
    opts.mapper.filter_max_reproj_error = float(args.filter_max_reproj_error)
    opts.mapper.filter_min_tri_angle = float(args.min_tri_angle)
    opts.mapper.ba_local_min_tri_angle = float(args.min_tri_angle)
    opts.mapper.ba_local_num_images = int(args.ba_local_num_images)
    opts.triangulation.ignore_two_view_tracks = not bool(args.use_two_view_tracks)
    opts.triangulation.min_angle = float(args.min_tri_angle)
    opts.triangulation.create_max_angle_error = float(args.create_max_angle_error)
    opts.triangulation.continue_max_angle_error = float(args.continue_max_angle_error)
    opts.triangulation.merge_max_reproj_error = float(args.merge_max_reproj_error)
    opts.triangulation.complete_max_reproj_error = float(args.complete_max_reproj_error)
    opts.triangulation.complete_max_transitivity = int(args.complete_max_transitivity)
    opts.triangulation.max_transitivity = int(args.max_transitivity)
    opts.triangulation.re_max_angle_error = float(args.re_max_angle_error)
    return opts


def ensure_seed_intrinsics(rec: pycolmap.Reconstruction, seed_params: list[float], width: int, height: int) -> None:
    for camera in rec.cameras.values():
        if "PINHOLE" not in str(camera.model):
            raise RuntimeError(f"expected PINHOLE camera, got {camera.model}")
        camera.width = int(width)
        camera.height = int(height)
        camera.params = seed_params


def run_retriangulation(args: argparse.Namespace, seed_params: list[float]) -> dict[str, Any]:
    output_model = args.output_model
    if args.resume and all((output_model / f).exists() for f in ("cameras.bin", "images.bin", "points3D.bin")):
        rec = pycolmap.Reconstruction(str(output_model))
        log(f"reuse dense retriangulated model: {output_model}")
        return {
            "status": "reused",
            "duration_seconds": 0.0,
            "output_model": str(output_model),
            "registered_images": int(rec.num_reg_images()),
            "points3D": int(rec.num_points3D()),
        }

    if output_model.exists():
        shutil.rmtree(output_model)
    output_model.mkdir(parents=True, exist_ok=True)

    rec = pycolmap.Reconstruction(str(args.input_model))
    ensure_seed_intrinsics(rec, seed_params, args.width, args.height)
    opts = configure_options(args)
    started = time.time()
    log(f"run pycolmap.triangulate_points -> {output_model}")
    rec2 = pycolmap.triangulate_points(
        rec,
        str(args.database),
        str(args.images),
        str(output_model),
        True,
        opts,
        False,
    )
    duration = time.time() - started
    return {
        "status": "success",
        "duration_seconds": round(duration, 3),
        "output_model": str(output_model),
        "registered_images": int(rec2.num_reg_images()),
        "points3D": int(rec2.num_points3D()),
        "pycolmap_mean_reprojection_error": float(rec2.compute_mean_reprojection_error()) if rec2.num_points3D() else None,
    }


def validate_dense_database(base: Any, run_dir: Path, stats: dict[str, Any], selected_images: int) -> None:
    passed = (
        stats["images"] == selected_images
        and stats["cameras"] == 1
        and stats["keypoints"] == selected_images
        and stats["two_view_geometries"] >= 3000
        and stats["two_view_geometry_rows"]["mean"] >= 50.0
    )
    base.strict_gate(run_dir, "dense_database", passed, stats)


def validate_dense_model(
    base: Any,
    run_dir: Path,
    summary: dict[str, Any],
    selected_images: int,
    source_points: int,
    args: argparse.Namespace,
) -> None:
    points = int(summary["points3D"])
    registered = int(summary["registered_images"])
    gain = points / max(source_points, 1)
    stats = summary["reprojection_stats"]
    drift = summary["intrinsics_drift"]
    metrics = {
        **summary,
        "selected_images": selected_images,
        "source_points3D": source_points,
        "density_gain": gain,
        "registered_ratio": registered / max(selected_images, 1),
        "points_per_registered": points / max(registered, 1),
        "max_allowed_mean_reprojection_px": args.max_mean_reprojection_px,
        "min_required_density_gain": args.min_density_gain,
        "min_required_points3D": args.min_points,
    }
    passed = (
        registered == selected_images
        and points >= int(args.min_points)
        and gain >= float(args.min_density_gain)
        and metrics["points_per_registered"] >= float(args.min_points_per_registered)
        and float(summary["mean_reprojection_error"]) <= float(args.max_mean_reprojection_px)
        and float(stats["invalid_projection_count"]) <= max(100, stats["observation_count"] * 0.001)
        and float(drift["max_focal_relative_drift"]) <= 0.005
        and float(drift["max_principal_point_pixel_drift"]) <= 0.5
    )
    base.strict_gate(run_dir, "dense_retriangulated_model", passed, metrics)


def validate_dense_ply(base: Any, run_dir: Path, summary: dict[str, Any], ply_stats: dict[str, Any]) -> None:
    points = max(int(summary["points3D"]), 1)
    rgb = int(ply_stats["rgb_points"])
    metrics = {
        **ply_stats,
        "model_points3D": int(summary["points3D"]),
        "rgb_ratio": rgb / points,
    }
    passed = rgb >= int(points * 0.95) and int(ply_stats["ply_bytes"]) > 5_000_000
    base.strict_gate(run_dir, "dense_rgb_ply", passed, metrics)


def write_report(
    run_dir: Path,
    args: argparse.Namespace,
    db_stats: dict[str, Any],
    stage: dict[str, Any],
    source_summary: dict[str, Any],
    dense_summary: dict[str, Any],
    ply_stats: dict[str, Any],
) -> None:
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "GLUEMAP fixed-intrinsics model + pycolmap.triangulate_points on GLUEMAP merged database",
        "database": db_stats,
        "stage": stage,
        "source_model": source_summary,
        "dense_model": dense_summary,
        "ply_stats": ply_stats,
        "outputs": {
            "run_dir": str(run_dir),
            "dense_model": str(args.output_model),
            "dense_rgb_point_cloud": str(args.output_ply),
            "report_json": str(run_dir / "build_report_dense_retriangulated.json"),
        },
    }
    write_json(run_dir / "build_report_dense_retriangulated.json", report)
    lines = [
        "# Football Field GLUEMAP Dense Retriangulation",
        "",
        "## Method",
        "",
        "- Input model: fixed-intrinsics GLUEMAP BA",
        "- Database: GLUEMAP merged matches",
        "- Retriangulation: pycolmap.triangulate_points",
        "- Intrinsics: fixed shared PINHOLE seed",
        "",
        "## Result",
        "",
        f"- source points3D: `{source_summary['points3D']}`",
        f"- dense points3D: `{dense_summary['points3D']}`",
        f"- density gain: `{dense_summary['points3D'] / max(source_summary['points3D'], 1):.3f}x`",
        f"- registered images: `{dense_summary['registered_images']}`",
        f"- mean reprojection error: `{dense_summary['mean_reprojection_error']}`",
        f"- median reprojection error: `{dense_summary['reprojection_stats']['median_px']}`",
        f"- RGB points: `{ply_stats['rgb_points']}`",
        f"- RGB ratio: `{ply_stats['rgb_points'] / max(dense_summary['points3D'], 1):.6f}`",
        "",
        "## Outputs",
        "",
        f"- dense model: `{args.output_model}`",
        f"- RGB PLY: `{args.output_ply}`",
    ]
    (run_dir / "BUILD_GLUEMAP_DENSE_RETRIANGULATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    default_run = Path("/media/cihcilab/新增磁碟區/sfm_system/建圖/runs/football_field_gluemap_pi3_motion262_1920_20260707")
    ap = argparse.ArgumentParser(description="Densify the football-field GLUEMAP model with post-GLUEMAP retriangulation.")
    ap.add_argument("--run-dir", type=Path, default=default_run)
    ap.add_argument("--base-pipeline", type=Path, default=Path("/media/cihcilab/新增磁碟區/sfm_system/建圖/pipeline/run_football_gluemap_from_motion_manifest.py"))
    ap.add_argument("--river-helper", type=Path, default=Path("/media/cihcilab/新增磁碟區/河濱場域/gluemap_build/tools/run_highres_optimized_build.py"))
    ap.add_argument("--input-model", type=Path, default=default_run / "gluemap" / "gluemap_fixed_intrinsics_ba")
    ap.add_argument("--database", type=Path, default=default_run / "gluemap" / "database_merged.db")
    ap.add_argument("--images", type=Path, default=default_run / "images")
    ap.add_argument("--output-model", type=Path, default=default_run / "gluemap" / "gluemap_retriangulated_dense_balanced")
    ap.add_argument("--output-ply", type=Path, default=default_run / "deploy" / "map_rgb_dense_retriangulated.ply")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min-num-matches", type=int, default=8)
    ap.add_argument("--use-two-view-tracks", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fix-existing-frames", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--min-tri-angle", type=float, default=0.5)
    ap.add_argument("--filter-max-reproj-error", type=float, default=8.0)
    ap.add_argument("--create-max-angle-error", type=float, default=4.0)
    ap.add_argument("--continue-max-angle-error", type=float, default=4.0)
    ap.add_argument("--merge-max-reproj-error", type=float, default=6.0)
    ap.add_argument("--complete-max-reproj-error", type=float, default=6.0)
    ap.add_argument("--complete-max-transitivity", type=int, default=10)
    ap.add_argument("--max-transitivity", type=int, default=2)
    ap.add_argument("--re-max-angle-error", type=float, default=8.0)
    ap.add_argument("--ba-local-num-images", type=int, default=8)
    ap.add_argument("--ba-local-max-num-iterations", type=int, default=25)
    ap.add_argument("--ba-global-max-num-iterations", type=int, default=50)
    ap.add_argument("--min-points", type=int, default=350000)
    ap.add_argument("--min-density-gain", type=float, default=1.8)
    ap.add_argument("--min-points-per-registered", type=float, default=1500.0)
    ap.add_argument("--max-mean-reprojection-px", type=float, default=3.5)
    args = ap.parse_args()

    base = import_module(args.base_pipeline, "football_gluemap_base")
    river = import_module(args.river_helper, "river_gluemap_helper")
    map_intrinsics = read_json(args.run_dir / "map_intrinsics.json")
    seed_params = [float(x) for x in map_intrinsics["params"]]
    selection = read_json(args.run_dir / "gates" / "selection_motion_quality.json")["metrics"]
    selected_images = int(selection["selected"])

    db_stats = database_stats(args.database)
    validate_dense_database(base, args.run_dir, db_stats, selected_images)

    source_summary = base.strict_model_summary(args.input_model, seed_params)
    stage_started = time.time()
    stage = run_retriangulation(args, seed_params)
    write_json(args.run_dir / "stage_records" / "dense_retriangulation.json", {
        **stage,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stage_started)),
        "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "database": str(args.database),
        "input_model": str(args.input_model),
        "options": {
            "min_num_matches": args.min_num_matches,
            "use_two_view_tracks": args.use_two_view_tracks,
            "min_tri_angle": args.min_tri_angle,
            "filter_max_reproj_error": args.filter_max_reproj_error,
            "complete_max_transitivity": args.complete_max_transitivity,
        },
    })

    dense_summary = base.strict_model_summary(args.output_model, seed_params)
    validate_dense_model(base, args.run_dir, dense_summary, selected_images, int(source_summary["points3D"]), args)

    ply_stats = river.export_rgb_ply(args.output_model, args.images, args.output_ply, 8.0)
    validate_dense_ply(base, args.run_dir, dense_summary, ply_stats)
    write_report(args.run_dir, args, db_stats, stage, source_summary, dense_summary, ply_stats)
    log(f"done: {args.run_dir}")
    log(f"report: {args.run_dir / 'BUILD_GLUEMAP_DENSE_RETRIANGULATION_REPORT.md'}")
    log(f"ply: {args.output_ply}")


if __name__ == "__main__":
    main()
