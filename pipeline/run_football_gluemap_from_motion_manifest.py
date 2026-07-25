#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycolmap
import yaml


def log(message: str) -> None:
    print(f"[football_gluemap] {message}", flush=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def import_river_helpers(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("river_gluemap_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import river helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_full_opencv_intrinsics(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = read_json(path)
    key = f"{width}x{height}"
    if key in data:
        entry = data[key]
        source_w, source_h = width, height
    elif "1920x1080" in data:
        entry = data["1920x1080"]
        source_w, source_h = 1920, 1080
    elif "K" in data:
        source_w, source_h = int(data["image_width"]), int(data["image_height"])
        sx = width / source_w
        sy = height / source_h
        k = np.asarray(data["K"], dtype=np.float64).copy()
        k[0, 0] *= sx
        k[1, 1] *= sy
        k[0, 2] *= sx
        k[1, 2] *= sy
        dist = np.asarray(data.get("dist", []), dtype=np.float64).reshape(-1)
        return k, dist, {
            "source_format": "K_dist",
            "source_width": source_w,
            "source_height": source_h,
            "target_width": width,
            "target_height": height,
            "scale_x": sx,
            "scale_y": sy,
            "scaled_K": k.tolist(),
            "dist": dist.tolist(),
        }
    else:
        raise RuntimeError(f"unsupported intrinsics JSON: {path}")

    params = [float(x) for x in entry["params"]]
    model = entry.get("model", "FULL_OPENCV")
    if model != "FULL_OPENCV":
        raise RuntimeError(f"expected FULL_OPENCV intrinsics, got {model}")
    if len(params) < 12:
        raise RuntimeError(f"FULL_OPENCV requires 12 params, got {len(params)}")

    sx = width / source_w
    sy = height / source_h
    fx, fy, cx, cy = params[:4]
    k = np.array(
        [
            [fx * sx, 0.0, cx * sx],
            [0.0, fy * sy, cy * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    # OpenCV accepts the rational 8-coefficient layout: k1,k2,p1,p2,k3,k4,k5,k6.
    k1, k2, p1, p2, k3, k4, k5, k6 = params[4:12]
    dist = np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64)
    meta = {
        "source_format": "FULL_OPENCV",
        "source_model": model,
        "source_width": source_w,
        "source_height": source_h,
        "target_width": width,
        "target_height": height,
        "scale_x": sx,
        "scale_y": sy,
        "scaled_K": k.tolist(),
        "dist": dist.tolist(),
        "note": "Frames are resized to target resolution and undistorted with FULL_OPENCV coefficients; GLUEMAP receives shared PINHOLE images.",
    }
    return k, dist, meta


def video_by_sequence(video_dir: Path) -> dict[str, Path]:
    videos = {}
    for suffix in ("*.MP4", "*.mp4", "*.MOV", "*.mov", "*.mkv", "*.avi"):
        for path in video_dir.glob(suffix):
            videos[path.stem] = path
    if not videos:
        raise RuntimeError(f"no videos found under {video_dir}")
    return videos


def grouped_manifest_frames(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for frame in manifest.get("frames", []):
        name = frame["name"]
        seq, filename = name.split("/", 1)
        grouped.setdefault(seq, []).append({**frame, "sequence": seq, "filename": filename})
    for rows in grouped.values():
        rows.sort(key=lambda r: r["filename"])
    return grouped


def seek_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"failed to seek/read video frame {frame_index}")
    return frame


def extract_motion_manifest_frames(
    video_dir: Path,
    manifest_path: Path,
    build_config_path: Path,
    images_all: Path,
    width: int,
    height: int,
    k: np.ndarray,
    dist: np.ndarray,
    jpeg_quality: int,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    manifest = read_json(manifest_path)
    build_config = read_json(build_config_path)
    sample_fps = float(build_config.get("fps") or 2.0)
    videos = video_by_sequence(video_dir)
    grouped = grouped_manifest_frames(manifest)
    if not grouped:
        raise RuntimeError(f"manifest has no frames: {manifest_path}")

    if images_all.exists() and not resume:
        shutil.rmtree(images_all)
    images_all.mkdir(parents=True, exist_ok=True)

    motion_roles: dict[str, dict[str, str]] = {}
    per_sequence: dict[str, Any] = {}
    total = 0
    for seq, rows in grouped.items():
        if seq not in videos:
            raise RuntimeError(f"no source video for sequence {seq}; available={sorted(videos)}")
        out_dir = images_all / seq
        existing = sorted(out_dir.glob("*.jpg"))
        if len(existing) == len(rows) and resume:
            log(f"reuse undistorted motion frames {seq}: {len(existing)}")
            for row in rows:
                motion_roles[row["name"]] = {
                    "motion_class": row.get("motion_class", "unknown"),
                    "motion_role": row.get("motion_role", "unknown"),
                }
            per_sequence[seq] = {"selected_from_manifest": len(rows), "saved": len(existing), "reused": True}
            total += len(existing)
            continue
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(videos[seq]))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {videos[seq]}")
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        log(
            f"extract selected {seq}: {len(rows)} frames from {src_w}x{src_h}@{src_fps:.3f}, "
            f"manifest_sample_fps={sample_fps:g}, target={width}x{height}"
        )
        saved = 0
        for row in rows:
            saved_index = int(Path(row["filename"]).stem)
            time_s = (saved_index - 1) / sample_fps
            source_frame_index = int(round(time_s * src_fps))
            source_frame_index = min(max(source_frame_index, 0), max(frame_count - 1, 0))
            frame = seek_frame(cap, source_frame_index)
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            undistorted = cv2.undistort(frame, k, dist, None, k)
            out = out_dir / row["filename"]
            cv2.imwrite(str(out), undistorted, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            saved += 1
            motion_roles[row["name"]] = {
                "motion_class": row.get("motion_class", "unknown"),
                "motion_role": row.get("motion_role", "unknown"),
            }
        cap.release()
        total += saved
        per_sequence[seq] = {"selected_from_manifest": len(rows), "saved": saved, "reused": False}
        log(f"  saved {saved} undistorted frames -> {out_dir}")

    summary = {
        "source_manifest": str(manifest_path),
        "source_build_config": str(build_config_path),
        "source_sampling_fps": sample_fps,
        "total_images": total,
        "per_sequence": per_sequence,
    }
    return summary, motion_roles


def write_manifest_with_motion(
    run_dir: Path,
    images_dir: Path,
    site_name: str,
    motion_roles: dict[str, dict[str, str]],
    intrinsics_meta: dict[str, Any],
) -> dict[str, Any]:
    names = sorted(str(p.relative_to(images_dir)) for p in images_dir.glob("*/*.jpg"))
    groups: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    frames = []
    for name in names:
        img = cv2.imread(str(images_dir / name), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(images_dir / name)
        h, w = img.shape[:2]
        seq = name.split("/", 1)[0]
        groups[seq] = groups.get(seq, 0) + 1
        role = motion_roles.get(name, {})
        motion_class = role.get("motion_class", "unknown")
        motion_role = role.get("motion_role", "unknown")
        class_counts[motion_class] = class_counts.get(motion_class, 0) + 1
        role_counts[motion_role] = role_counts.get(motion_role, 0) + 1
        frames.append(
            {
                "name": name,
                "width": int(w),
                "height": int(h),
                "motion_class": motion_class,
                "motion_role": motion_role,
            }
        )

    k = intrinsics_meta["scaled_K"]
    manifest = {
        "site_name": site_name,
        "image_root": str(images_dir),
        "total_frames": len(frames),
        "groups": groups,
        "motion_class_counts": class_counts,
        "motion_role_counts": role_counts,
        "frames": frames,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    intrinsics = {
        "camera_mode": "SHARED",
        "camera_model": "PINHOLE",
        "image_width": int(intrinsics_meta["target_width"]),
        "image_height": int(intrinsics_meta["target_height"]),
        "params": [k[0][0], k[1][1], k[0][2], k[1][2]],
        **intrinsics_meta,
    }
    write_json(run_dir / "frame_manifest.json", manifest)
    write_json(run_dir / "map_intrinsics.json", intrinsics)
    return manifest


def strict_gate(path: Path, name: str, passed: bool, metrics: dict[str, Any], hard: bool = True) -> None:
    record = {
        "gate": name,
        "passed": bool(passed),
        "hard": bool(hard),
        "metrics": metrics,
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(path / "gates" / f"{name}.json", record)
    status = "PASS" if passed else "FAIL"
    log(f"gate {name}: {status} {metrics}")
    if hard and not passed:
        raise SystemExit(f"strict gate failed: {name} {metrics}")


def run_logged_with_env(run_dir: Path, stage: str, cmd: list[str], cwd: Path, gluemap_env: Path) -> None:
    log_path = run_dir / "logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = f"{gluemap_env / 'bin'}:" + env.get("PATH", "")
    env["CONDA_PREFIX"] = str(gluemap_env)
    env["LD_LIBRARY_PATH"] = f"{gluemap_env / 'lib'}:" + env.get("LD_LIBRARY_PATH", "")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    started = time.time()
    log(f"run {stage}: {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    ended = time.time()
    write_json(
        run_dir / "stage_records" / f"{stage}.json",
        {
            "stage": stage,
            "status": "success" if proc.returncode == 0 else "failed",
            "duration_seconds": round(ended - started, 3),
            "log": str(log_path),
            "cmd": cmd,
        },
    )
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-120:])
        raise SystemExit(f"{stage} failed with exit {proc.returncode}; log tail:\n{tail}")


def recompute_reprojection_stats(rec: pycolmap.Reconstruction, update_point_errors: bool = False) -> dict[str, Any]:
    errors: list[float] = []
    invalid = 0
    for point in rec.points3D.values():
        point_errors: list[float] = []
        for elem in point.track.elements:
            image = rec.images[elem.image_id]
            camera = rec.cameras[image.camera_id]
            point2d = image.points2D[elem.point2D_idx].xy
            try:
                projected = image.project_point(point.xyz)
            except TypeError:
                projected = image.project_point(point.xyz, camera)
            if projected is None:
                invalid += 1
                continue
            error = float(np.linalg.norm(np.asarray(projected) - np.asarray(point2d)))
            errors.append(error)
            point_errors.append(error)
        if update_point_errors and point_errors:
            point.error = float(np.mean(point_errors))
    if not errors:
        return {
            "observation_count": 0,
            "invalid_projection_count": invalid,
            "mean_px": None,
            "median_px": None,
            "p75_px": None,
            "p90_px": None,
            "p95_px": None,
            "p99_px": None,
            "max_px": None,
            "within_4_5px_ratio": 0.0,
        }
    arr = np.asarray(errors, dtype=np.float64)
    return {
        "observation_count": int(arr.size),
        "invalid_projection_count": invalid,
        "mean_px": float(arr.mean()),
        "median_px": float(np.median(arr)),
        "p75_px": float(np.percentile(arr, 75)),
        "p90_px": float(np.percentile(arr, 90)),
        "p95_px": float(np.percentile(arr, 95)),
        "p99_px": float(np.percentile(arr, 99)),
        "max_px": float(arr.max()),
        "within_4_5px_ratio": float((arr <= 4.5).mean()),
    }


def camera_intrinsics_drift(rec: pycolmap.Reconstruction, seed_params: list[float]) -> dict[str, Any]:
    seed = np.asarray(seed_params, dtype=np.float64)
    cameras = []
    max_focal_rel_drift = 0.0
    max_principal_point_px_drift = 0.0
    for camera_id, camera in rec.cameras.items():
        params = np.asarray(camera.params, dtype=np.float64)
        focal_rel = np.abs(params[:2] - seed[:2]) / np.maximum(np.abs(seed[:2]), 1e-9)
        pp_abs = np.abs(params[2:4] - seed[2:4])
        max_focal_rel_drift = max(max_focal_rel_drift, float(focal_rel.max()))
        max_principal_point_px_drift = max(max_principal_point_px_drift, float(pp_abs.max()))
        cameras.append(
            {
                "camera_id": int(camera_id),
                "model": str(camera.model),
                "width": int(camera.width),
                "height": int(camera.height),
                "params": [float(x) for x in params.tolist()],
                "seed_params": [float(x) for x in seed.tolist()],
                "focal_relative_drift": [float(x) for x in focal_rel.tolist()],
                "principal_point_pixel_drift": [float(x) for x in pp_abs.tolist()],
            }
        )
    return {
        "camera_count": len(cameras),
        "max_focal_relative_drift": max_focal_rel_drift,
        "max_principal_point_pixel_drift": max_principal_point_px_drift,
        "cameras": cameras,
    }


def strict_model_summary(model_dir: Path, seed_params: list[float]) -> dict[str, Any]:
    rec = pycolmap.Reconstruction(str(model_dir))
    recomputed = recompute_reprojection_stats(rec, update_point_errors=True)
    rec.write(str(model_dir))
    return {
        "model_dir": str(model_dir),
        "registered_images": int(rec.num_reg_images()),
        "points3D": int(rec.num_points3D()),
        "mean_reprojection_error": recomputed["mean_px"],
        "reprojection_stats": recomputed,
        "intrinsics_drift": camera_intrinsics_drift(rec, seed_params),
    }


def run_fixed_intrinsics_ba(
    run_dir: Path,
    input_model_dir: Path,
    seed_params: list[float],
    width: int,
    height: int,
    resume: bool,
) -> Path:
    seeded_dir = run_dir / "gluemap" / "gluemap_seed_intrinsics"
    fixed_dir = run_dir / "gluemap" / "gluemap_fixed_intrinsics_ba"
    if resume and all((fixed_dir / f).exists() for f in ("cameras.bin", "images.bin", "points3D.bin")):
        log(f"reuse fixed-intrinsics BA model: {fixed_dir}")
        return fixed_dir

    for path in (seeded_dir, fixed_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    rec = pycolmap.Reconstruction(str(input_model_dir))
    seed = np.asarray(seed_params, dtype=np.float64)
    for camera in rec.cameras.values():
        if "PINHOLE" not in str(camera.model):
            raise RuntimeError(f"fixed intrinsics BA expects PINHOLE cameras, got {camera.model}")
        camera.width = int(width)
        camera.height = int(height)
        camera.params = seed.copy()
    rec.write(str(seeded_dir))

    started = time.time()
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.refine_points3D = True
    options.ceres.solver_options.max_num_iterations = 100
    options.print_summary = True
    pycolmap.bundle_adjustment(rec, options)
    rec.write(str(fixed_dir))
    helpers_record = {
        "stage": "fixed_intrinsics_ba",
        "status": "success",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(time.time() - started, 3),
        "input_model": str(input_model_dir),
        "seeded_model": str(seeded_dir),
        "output_model": str(fixed_dir),
        "refine_focal_length": False,
        "refine_principal_point": False,
        "refine_extra_params": False,
        "refine_points3D": True,
    }
    write_json(run_dir / "stage_records" / "fixed_intrinsics_ba.json", helpers_record)
    return fixed_dir


def validate_selection(run_dir: Path, selection: dict[str, Any], manifest: dict[str, Any]) -> None:
    selected = int(selection["selected_images"])
    total = int(selection["total_images"])
    dropped = total - selected
    groups = manifest["groups"]
    class_counts = manifest.get("motion_class_counts", {})
    role_counts = manifest.get("motion_role_counts", {})
    parallax = int(class_counts.get("parallax", 0) + class_counts.get("seed", 0))
    pure_rotation = int(class_counts.get("pure_rotation", 0))
    hover = int(class_counts.get("hover", 0))
    bridge = int(role_counts.get("bridge_only", 0))
    metrics = {
        "total": total,
        "selected": selected,
        "dropped": dropped,
        "selected_ratio": selected / max(total, 1),
        "groups": groups,
        "motion_class_counts": class_counts,
        "motion_role_counts": role_counts,
        "parallax_or_seed_ratio": parallax / max(selected, 1),
        "pure_rotation_ratio": pure_rotation / max(selected, 1),
        "hover_ratio": hover / max(selected, 1),
        "bridge_frames": bridge,
    }
    passed = (
        selected >= 120
        and all(count >= 20 for count in groups.values())
        and metrics["selected_ratio"] >= 0.75
        and metrics["parallax_or_seed_ratio"] >= 0.65
        and metrics["hover_ratio"] <= 0.02
        and bridge >= 20
    )
    strict_gate(run_dir, "selection_motion_quality", passed, metrics)


def validate_model(run_dir: Path, selection: dict[str, Any], summary: dict[str, Any], manifest: dict[str, Any]) -> None:
    selected = int(selection["selected_images"])
    registered = int(summary["registered_images"])
    points = int(summary["points3D"])
    reproj = summary.get("mean_reprojection_error")
    intrinsics = summary.get("intrinsics_drift", {})
    metrics = {
        **summary,
        "selected_images": selected,
        "registered_ratio": registered / max(selected, 1),
        "points_per_registered": points / max(registered, 1),
        "required_sequences": sorted(manifest["groups"]),
        "max_allowed_mean_reprojection_px": 4.5,
        "max_allowed_focal_relative_drift": 0.005,
        "max_allowed_principal_point_pixel_drift": 0.5,
    }
    passed = (
        registered >= max(2, int(selected * 0.85))
        and points >= 100000
        and metrics["points_per_registered"] >= 500
        and reproj is not None
        and float(reproj) <= 4.5
        and float(intrinsics.get("max_focal_relative_drift", 1.0)) <= 0.005
        and float(intrinsics.get("max_principal_point_pixel_drift", 9999.0)) <= 0.5
    )
    strict_gate(run_dir, "gluemap_model", passed, metrics)


def validate_ply(run_dir: Path, summary: dict[str, Any], ply_stats: dict[str, Any]) -> None:
    points = max(int(summary["points3D"]), 1)
    rgb = int(ply_stats["rgb_points"])
    metrics = {
        **ply_stats,
        "model_points3D": int(summary["points3D"]),
        "rgb_ratio": rgb / points,
    }
    passed = rgb >= int(points * 0.90) and int(ply_stats["ply_bytes"]) > 1_000_000
    strict_gate(run_dir, "rgb_ply", passed, metrics)


def write_config(
    run_dir: Path,
    repo_dir: Path,
    images_dir: Path,
    seed_dir: Path,
    args: argparse.Namespace,
) -> Path:
    config = {
        "chosen_model": "pi3",
        "path_feedforward": str(repo_dir / "checkpoints" / "pi3.safetensors"),
        "path_retrieval": str(repo_dir / "checkpoints" / "dino_salad.ckpt"),
        "path_tracker": str(repo_dir / "checkpoints" / "vggsfm_v2_0_0_track_predictor.bin"),
        "path_dg": str(repo_dir / "checkpoints" / "checkpoint-dg+visym.pth"),
        "images_path": str(images_dir),
        "write_path": str(run_dir / "gluemap"),
        "temp_path": str(run_dir / "tmp"),
        "chosen_output": "gluemap_aba",
        "num_track_per_img": int(args.num_track_per_img),
        "max_num_tracks": None,
        "camera_model": "PINHOLE",
        "intrinsics_mode": "SHARED",
        "num_neighbors": int(args.num_neighbors),
        "num_neighbors_sequential": int(args.num_neighbors_sequential),
        "batch_size": int(args.batch_size),
        "retrieval_batch_size": int(args.retrieval_batch_size),
        "num_workers": int(args.num_workers),
        "valid_pose_threshold": 0.05,
        "valid_dg_threshold": 0.8,
        "force_load": True,
        "rerun_from": None,
        "coarse_only": False,
        "use_dummy_tracks": False,
        "skip_doppelgangers": bool(args.skip_doppelgangers),
        "use_gt_intrinsics": True,
        "gt_intrinsics_path": str(seed_dir),
        "is_sequential": True,
        "sample_frequency": 1,
        "is_multi_sequence": True,
        "subfolder_regex": "^P[0-9]+",
    }
    path = run_dir / "gluemap_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(run_dir / "build_config.json", config)
    return path


def write_report(
    run_dir: Path,
    args: argparse.Namespace,
    extraction: dict[str, Any],
    selection: dict[str, Any],
    manifest: dict[str, Any],
    model_summary: dict[str, Any],
    ply_stats: dict[str, Any],
) -> None:
    baseline = {
        "run": "/media/cihcilab/新增磁碟區/sfm_system/建圖/runs/football_field_20260707",
        "registered_images": 262,
        "points3D": 659024,
        "mean_reprojection_error": 0.879,
        "ply": "/media/cihcilab/新增磁碟區/sfm_system/建圖/runs/football_field_20260707/deploy/map_rgb.ply",
    }
    report = {
        "site_name": args.site_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "extraction": extraction,
        "selection": selection,
        "manifest_summary": {
            "groups": manifest["groups"],
            "motion_class_counts": manifest.get("motion_class_counts", {}),
            "motion_role_counts": manifest.get("motion_role_counts", {}),
        },
        "model_summary": model_summary,
        "ply_stats": ply_stats,
        "baseline_glo_map_mvroma": baseline,
        "outputs": {
            "run_dir": str(run_dir),
            "images": str(run_dir / "images"),
            "gluemap_model": model_summary["model_dir"],
            "rgb_point_cloud": str(run_dir / "deploy" / "map_rgb.ply"),
            "report_json": str(run_dir / "build_report.json"),
            "gluemap_log": str(run_dir / "logs" / "gluemap.log"),
        },
    }
    write_json(run_dir / "build_report.json", report)
    lines = [
        f"# Football Field GLUEMAP Build: {args.site_name}",
        "",
        "## Input",
        "",
        f"- source manifest: `{extraction['source_manifest']}`",
        f"- source sampling fps: `{extraction['source_sampling_fps']}`",
        f"- selected images after quality: `{selection['selected_images']}/{selection['total_images']}`",
        f"- groups: `{manifest['groups']}`",
        f"- motion classes: `{manifest.get('motion_class_counts', {})}`",
        f"- motion roles: `{manifest.get('motion_role_counts', {})}`",
        "",
        "## GLUEMAP Model",
        "",
        f"- model dir: `{model_summary['model_dir']}`",
        f"- registered images: `{model_summary['registered_images']}`",
        f"- points3D: `{model_summary['points3D']}`",
        f"- mean reprojection error: `{model_summary['mean_reprojection_error']}`",
        f"- RGB points: `{ply_stats['rgb_points']}`",
        f"- PLY bytes: `{ply_stats['ply_bytes']}`",
        "",
        "## Baseline",
        "",
        f"- previous MV-RoMA/GLOMAP registered: `{baseline['registered_images']}`",
        f"- previous MV-RoMA/GLOMAP points3D: `{baseline['points3D']}`",
        f"- previous MV-RoMA/GLOMAP mean reprojection error: `{baseline['mean_reprojection_error']}`",
        "",
    ]
    (run_dir / "BUILD_GLUEMAP_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def preflight(args: argparse.Namespace, run_dir: Path) -> None:
    checks: dict[str, Any] = {}
    required_paths = {
        "video_dir": args.video_dir,
        "source_manifest": args.source_manifest,
        "source_build_config": args.source_build_config,
        "intrinsics_json": args.intrinsics_json,
        "repo": args.repo,
        "gluemap_env": args.gluemap_env,
        "river_helper": args.river_helper,
    }
    for key, path in required_paths.items():
        checks[key] = {"path": str(path), "exists": Path(path).exists()}
    checkpoints = [
        args.repo / "checkpoints" / "pi3.safetensors",
        args.repo / "checkpoints" / "dino_salad.ckpt",
        args.repo / "checkpoints" / "vggsfm_v2_0_0_track_predictor.bin",
        args.repo / "checkpoints" / "checkpoint-dg+visym.pth",
    ]
    checks["checkpoints"] = {str(p): p.exists() for p in checkpoints}
    checks["gluemap_demo"] = str(args.gluemap_env / "bin" / "gluemap-demo")
    checks["gluemap_demo_exists"] = (args.gluemap_env / "bin" / "gluemap-demo").exists()
    passed = all(v["exists"] for v in checks.values() if isinstance(v, dict) and "exists" in v)
    passed = passed and all(checks["checkpoints"].values()) and checks["gluemap_demo_exists"]
    strict_gate(run_dir, "preflight", passed, checks)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build GLUEMAP map for the football field from prior motion-selected frames.")
    ap.add_argument("--workspace", type=Path, default=Path("/media/cihcilab/新增磁碟區/sfm_system/建圖"))
    ap.add_argument("--repo", type=Path, default=Path("/media/cihcilab/新增磁碟區/河濱場域/gluemap_build/tools/gluemap"))
    ap.add_argument("--gluemap-env", type=Path, default=Path("/home/cihcilab/micromamba/envs/gluemap"))
    ap.add_argument("--river-helper", type=Path, default=Path("/media/cihcilab/新增磁碟區/河濱場域/gluemap_build/tools/run_highres_optimized_build.py"))
    ap.add_argument("--video-dir", type=Path, default=Path("/media/cihcilab/新增磁碟區/足球場/足球場域測試-20260707T061318Z-3-001/足球場域測試"))
    ap.add_argument("--source-manifest", type=Path, default=Path("/media/cihcilab/新增磁碟區/sfm_system/建圖/runs/football_field_20260707/frame_manifest.json"))
    ap.add_argument("--source-build-config", type=Path, default=Path("/media/cihcilab/新增磁碟區/sfm_system/建圖/runs/football_field_20260707/build_config.json"))
    ap.add_argument(
        "--intrinsics-json",
        type=Path,
        default=Path(
            "/media/cihcilab/新增磁碟區/sfm_system/configs/mapping/"
            "football_field_camera_init_1920x1080.json"
        ),
    )
    ap.add_argument("--run-name", default="football_field_gluemap_pi3_motion262_1920_20260707")
    ap.add_argument("--site-name", default="football_field_gluemap_pi3_motion262_1920")
    ap.add_argument("--target-width", type=int, default=1920)
    ap.add_argument("--target-height", type=int, default=1080)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--blur-drop-fraction", type=float, default=0.10)
    ap.add_argument("--duplicate-threshold", type=float, default=0.012)
    ap.add_argument("--num-track-per-img", type=int, default=2048)
    ap.add_argument("--num-neighbors", type=int, default=100)
    ap.add_argument("--num-neighbors-sequential", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--retrieval-batch-size", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--skip-doppelgangers", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--skip-gluemap", action="store_true", help="reuse an existing GLUEMAP model and only run post-processing/checks")
    ap.add_argument("--fixed-intrinsics-ba", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    run_dir = args.workspace / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    preflight(args, run_dir)

    helpers = import_river_helpers(args.river_helper)
    k, dist, intrinsics_meta = parse_full_opencv_intrinsics(args.intrinsics_json, args.target_width, args.target_height)
    write_json(run_dir / "intrinsics_scaled_debug.json", intrinsics_meta)

    started = time.time()
    extraction, motion_roles = extract_motion_manifest_frames(
        args.video_dir,
        args.source_manifest,
        args.source_build_config,
        run_dir / "images_all",
        args.target_width,
        args.target_height,
        k,
        dist,
        args.jpeg_quality,
        args.resume,
    )
    helpers.record_stage(run_dir, "extract_motion_manifest_undistort", started, "success")
    write_json(run_dir / "motion_selection_source.json", extraction)
    write_json(run_dir / "motion_roles.json", motion_roles)

    started = time.time()
    selection = helpers.prepare_clean_subset(
        run_dir / "images_all",
        run_dir / "images",
        run_dir / "quality",
        args.blur_drop_fraction,
        args.duplicate_threshold,
    )
    helpers.record_stage(run_dir, "quality_filter", started, "success")

    started = time.time()
    manifest = write_manifest_with_motion(run_dir, run_dir / "images", args.site_name, motion_roles, intrinsics_meta)
    seed_dir = helpers.write_colmap_seed(run_dir, run_dir / "images", k, args.target_width, args.target_height)
    config_path = write_config(run_dir, args.repo, run_dir / "images", seed_dir, args)
    helpers.record_stage(run_dir, "config", started, "success")
    validate_selection(run_dir, selection, manifest)

    if args.skip_gluemap:
        log("skip GLUEMAP stage and reuse existing model")
    else:
        run_logged_with_env(
            run_dir,
            "gluemap",
            [str(args.gluemap_env / "bin" / "gluemap-demo"), "--config", str(config_path)],
            args.repo,
            args.gluemap_env,
        )

    model_dir = helpers.find_model_dir(run_dir)
    seed_params = [float(k[0][0]), float(k[1][1]), float(k[0][2]), float(k[1][2])]
    if args.fixed_intrinsics_ba:
        model_dir = run_fixed_intrinsics_ba(
            run_dir,
            model_dir,
            seed_params,
            args.target_width,
            args.target_height,
            args.resume,
        )
    model = strict_model_summary(model_dir, seed_params)
    validate_model(run_dir, selection, model, manifest)

    started = time.time()
    ply_stats = helpers.export_rgb_ply(model_dir, run_dir / "images", run_dir / "deploy" / "map_rgb.ply", 8.0)
    helpers.record_stage(run_dir, "color", started, "success")
    validate_ply(run_dir, model, ply_stats)
    write_report(run_dir, args, extraction, selection, manifest, model, ply_stats)
    log(f"done: {run_dir}")
    log(f"report: {run_dir / 'BUILD_GLUEMAP_REPORT.md'}")
    log(f"ply: {run_dir / 'deploy' / 'map_rgb.ply'}")


if __name__ == "__main__":
    main()
