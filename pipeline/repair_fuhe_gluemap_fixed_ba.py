#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap


PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import run_football_gluemap_from_motion_manifest as base  # noqa: E402


DEFAULT_RUN_DIR = Path(
    "/media/cihcilab/新增磁碟區/sfm_system/建圖/runs/"
    "fuhe_bridge_gluemap_pi3_1fps_1920_20260707"
)


def log(message: str) -> None:
    print(f"[fuhe_repair] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def enforce_seed_intrinsics(
    rec: pycolmap.Reconstruction,
    params: list[float],
    width: int,
    height: int,
) -> None:
    seed = np.asarray(params, dtype=np.float64)
    for camera in rec.cameras.values():
        if "PINHOLE" not in str(camera.model):
            raise RuntimeError(f"expected PINHOLE camera, got {camera.model}")
        camera.width = int(width)
        camera.height = int(height)
        camera.params = seed.copy()


def run_fixed_ba(
    rec: pycolmap.Reconstruction,
    iterations: int,
    num_threads: int,
    function_tolerance: float,
    gradient_tolerance: float,
    parameter_tolerance: float,
) -> float:
    started = time.time()
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.refine_points3D = True
    options.print_summary = True
    options.ceres.solver_options.max_num_iterations = int(iterations)
    options.ceres.solver_options.num_threads = int(num_threads)
    options.ceres.solver_options.function_tolerance = float(function_tolerance)
    options.ceres.solver_options.gradient_tolerance = float(gradient_tolerance)
    options.ceres.solver_options.parameter_tolerance = float(parameter_tolerance)
    pycolmap.bundle_adjustment(rec, options)
    return time.time() - started


def filter_model(
    rec: pycolmap.Reconstruction,
    max_reproj_error: float,
    min_tri_angle: float,
    min_track_length: int,
) -> dict[str, int | float]:
    points_before = int(rec.num_points3D())
    obs_before = int(rec.compute_num_observations())
    manager = pycolmap.ObservationManager(rec)
    negative_removed = int(manager.filter_observations_with_negative_depth())
    filtered_removed = int(manager.filter_all_points3D(float(max_reproj_error), float(min_tri_angle)))
    short_removed = int(manager.filter_points3D_with_short_tracks(int(min_track_length)))
    return {
        "max_reproj_error": float(max_reproj_error),
        "min_tri_angle": float(min_tri_angle),
        "min_track_length": int(min_track_length),
        "points_before": points_before,
        "points_after": int(rec.num_points3D()),
        "observations_before": obs_before,
        "observations_after": int(rec.compute_num_observations()),
        "negative_depth_observations_removed": negative_removed,
        "filter_observations_removed": filtered_removed,
        "short_track_observations_removed": short_removed,
    }


def passes_strict(summary: dict[str, Any], selected_images: int, max_mean: float, max_p95: float) -> bool:
    stats = summary["reprojection_stats"]
    drift = summary["intrinsics_drift"]
    invalid_limit = max(100, int(stats["observation_count"] * 0.001))
    return (
        int(summary["registered_images"]) == int(selected_images)
        and int(summary["points3D"]) >= int(selected_images) * 500
        and float(summary["mean_reprojection_error"]) <= float(max_mean)
        and float(stats["p95_px"]) < float(max_p95)
        and int(stats["invalid_projection_count"]) <= invalid_limit
        and float(drift["max_focal_relative_drift"]) <= 1e-9
        and float(drift["max_principal_point_pixel_drift"]) <= 1e-9
    )


def parse_thresholds(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one threshold is required")
    return values


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Repair Fuhe GLUEMAP fixed-intrinsics BA outliers.")
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--input-model", type=Path)
    ap.add_argument("--output-model", type=Path)
    ap.add_argument("--intrinsics", type=Path)
    ap.add_argument("--selected-images", type=int, default=594)
    ap.add_argument("--filter-thresholds", type=parse_thresholds, default=parse_thresholds("8,6"))
    ap.add_argument("--min-tri-angle", type=float, default=0.1)
    ap.add_argument("--min-track-length", type=int, default=3)
    ap.add_argument("--ba-iterations", type=int, default=200)
    ap.add_argument("--num-threads", type=int, default=24)
    ap.add_argument("--function-tolerance", type=float, default=1e-6)
    ap.add_argument("--gradient-tolerance", type=float, default=1e-4)
    ap.add_argument("--parameter-tolerance", type=float, default=1e-8)
    ap.add_argument("--max-mean-reprojection-px", type=float, default=3.0)
    ap.add_argument("--max-p95-reprojection-px", type=float, default=8.0)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    input_model = args.input_model or (run_dir / "gluemap" / "gluemap_fixed_intrinsics_ba")
    output_model = args.output_model or (run_dir / "gluemap" / "gluemap_fixed_intrinsics_ba_repaired")
    intrinsics_path = args.intrinsics or (run_dir / "map_intrinsics.json")
    log_path = run_dir / "logs" / "fixed_intrinsics_ba_repair.log"

    if args.resume and all((output_model / name).exists() for name in ("cameras.bin", "images.bin", "points3D.bin")):
        summary = base.strict_model_summary(output_model, read_json(intrinsics_path)["params"])
        log(f"reuse repaired model: {output_model}")
        write_json(run_dir / "build_report_fixed_ba_repair.json", {"status": "reused", "summary": summary})
        return

    if output_model.exists():
        shutil.rmtree(output_model)
    output_model.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    intrinsics = read_json(intrinsics_path)
    seed_params = [float(x) for x in intrinsics["params"]]
    width = int(intrinsics["image_width"])
    height = int(intrinsics["image_height"])

    rec = pycolmap.Reconstruction(str(input_model))
    enforce_seed_intrinsics(rec, seed_params, width, height)
    pre_summary = {
        "registered_images": int(rec.num_reg_images()),
        "points3D": int(rec.num_points3D()),
        "reprojection_stats": base.recompute_reprojection_stats(rec, update_point_errors=False),
        "intrinsics_drift": base.camera_intrinsics_drift(rec, seed_params),
    }
    log(f"input model: {input_model}")
    log(
        "pre: images={images} points={points} mean={mean:.4f} p95={p95:.4f} invalid={invalid}".format(
            images=pre_summary["registered_images"],
            points=pre_summary["points3D"],
            mean=pre_summary["reprojection_stats"]["mean_px"],
            p95=pre_summary["reprojection_stats"]["p95_px"],
            invalid=pre_summary["reprojection_stats"]["invalid_projection_count"],
        )
    )

    rounds: list[dict[str, Any]] = []
    final_summary: dict[str, Any] | None = None
    started_all = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        for round_idx, threshold in enumerate(args.filter_thresholds, start=1):
            round_started = time.time()
            filt = filter_model(rec, threshold, args.min_tri_angle, args.min_track_length)
            log(f"round {round_idx}: filter {filt}")
            ba_seconds = run_fixed_ba(
                rec,
                args.ba_iterations,
                args.num_threads,
                args.function_tolerance,
                args.gradient_tolerance,
                args.parameter_tolerance,
            )
            tmp_dir = output_model.parent / f"{output_model.name}_round{round_idx}"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            rec.write(str(tmp_dir))
            summary = base.strict_model_summary(tmp_dir, seed_params)
            final_summary = summary
            round_record = {
                "round": round_idx,
                "filter": filt,
                "ba_seconds": round(ba_seconds, 3),
                "duration_seconds": round(time.time() - round_started, 3),
                "model_dir": str(tmp_dir),
                "summary": summary,
                "passes_strict": passes_strict(
                    summary,
                    args.selected_images,
                    args.max_mean_reprojection_px,
                    args.max_p95_reprojection_px,
                ),
            }
            rounds.append(round_record)
            log_file.write(json.dumps(round_record, ensure_ascii=False) + "\n")
            log_file.flush()
            stats = summary["reprojection_stats"]
            log(
                "round {idx}: mean={mean:.4f} median={median:.4f} p95={p95:.4f} "
                "invalid={invalid} points={points}".format(
                    idx=round_idx,
                    mean=summary["mean_reprojection_error"],
                    median=stats["median_px"],
                    p95=stats["p95_px"],
                    invalid=stats["invalid_projection_count"],
                    points=summary["points3D"],
                )
            )
            if round_record["passes_strict"]:
                break

    if final_summary is None:
        raise RuntimeError("repair did not produce a model")

    final_tmp = Path(rounds[-1]["model_dir"])
    if output_model.exists():
        shutil.rmtree(output_model)
    shutil.copytree(final_tmp, output_model)
    final_summary = base.strict_model_summary(output_model, seed_params)
    report = {
        "status": "success" if passes_strict(
            final_summary,
            args.selected_images,
            args.max_mean_reprojection_px,
            args.max_p95_reprojection_px,
        ) else "failed",
        "duration_seconds": round(time.time() - started_all, 3),
        "input_model": str(input_model),
        "output_model": str(output_model),
        "intrinsics": str(intrinsics_path),
        "selected_images": int(args.selected_images),
        "filter_thresholds": [float(x) for x in args.filter_thresholds],
        "min_tri_angle": float(args.min_tri_angle),
        "min_track_length": int(args.min_track_length),
        "ba_iterations": int(args.ba_iterations),
        "num_threads": int(args.num_threads),
        "function_tolerance": float(args.function_tolerance),
        "gradient_tolerance": float(args.gradient_tolerance),
        "parameter_tolerance": float(args.parameter_tolerance),
        "pre_summary": pre_summary,
        "rounds": rounds,
        "summary": final_summary,
    }
    write_json(run_dir / "build_report_fixed_ba_repair.json", report)
    log(f"final status: {report['status']}")
    if report["status"] != "success":
        raise SystemExit(f"repair failed strict thresholds: {final_summary}")


if __name__ == "__main__":
    main()
