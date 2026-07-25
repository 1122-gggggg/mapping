#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sqlite3
import subprocess
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycolmap
import yaml


WORKSPACE = Path("/media/cihcilab/新增磁碟區/sfm_system/建圖")
DEFAULT_VIDEO_DIR = Path("/media/cihcilab/新增磁碟區/福和橋場域/福和橋下測試場域")
DEFAULT_RUN_NAME = "fuhe_bridge_gluemap_pi3_1fps_1920_20260707"
DEFAULT_REPO = Path("/media/cihcilab/新增磁碟區/河濱場域/gluemap_build/tools/gluemap")
DEFAULT_GLUEMAP_ENV = Path("/home/cihcilab/micromamba/envs/gluemap")
DEFAULT_RIVER_HELPER = Path("/media/cihcilab/新增磁碟區/河濱場域/gluemap_build/tools/run_highres_optimized_build.py")
DEFAULT_BASE_PIPELINE = WORKSPACE / "pipeline" / "run_football_gluemap_from_motion_manifest.py"
DEFAULT_CORE_PIPELINE = WORKSPACE / "pipeline" / "build_localizable_map_core.py"
DEFAULT_INTRINSICS = WORKSPACE / "configs" / "原始估計內參.json"


def log(message: str) -> None:
    print(f"[fuhe_gluemap] {message}", flush=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strict_gate(run_dir: Path, name: str, passed: bool, metrics: dict[str, Any], hard: bool = True) -> None:
    record = {
        "gate": name,
        "passed": bool(passed),
        "hard": bool(hard),
        "metrics": metrics,
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(run_dir / "gates" / f"{name}.json", record)
    status = "PASS" if passed else "FAIL"
    log(f"gate {name}: {status} {metrics}")
    if hard and not passed:
        raise SystemExit(f"strict gate failed: {name} {metrics}")


def record_stage(run_dir: Path, stage: str, started: float, status: str, **extra: Any) -> None:
    ended = time.time()
    record = {
        "stage": stage,
        "status": status,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "ended_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ended)),
        "duration_seconds": round(ended - started, 3),
        **extra,
    }
    write_json(run_dir / "stage_records" / f"{stage}.json", record)
    path = run_dir / "stage_times.json"
    data = read_json(path) if path.exists() else {"stages": []}
    data.setdefault("stages", []).append(record)
    data["total_seconds"] = round(sum(float(x.get("duration_seconds", 0.0)) for x in data["stages"]), 3)
    data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(path, data)


def parse_intrinsics(
    path: Path,
    width: int,
    height: int,
    variant: str,
    official_video_hfov_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    data = read_json(path)
    if "K" not in data:
        raise RuntimeError(f"expected K/dist intrinsics JSON: {path}")
    source_w = int(data["image_width"])
    source_h = int(data["image_height"])
    sx = width / source_w
    sy = height / source_h
    k = np.asarray(data["K"], dtype=np.float64).copy()
    k[0, 0] *= sx
    k[1, 1] *= sy
    k[0, 2] *= sx
    k[1, 2] *= sy
    dist_raw = np.asarray(data.get("dist", []), dtype=np.float64).reshape(-1)
    if dist_raw.size not in (4, 5, 8):
        raise RuntimeError(f"unsupported OpenCV distortion length: {dist_raw.size}")
    dist = np.zeros(8, dtype=np.float64)
    dist[: dist_raw.size] = dist_raw

    if variant == "current_undistort":
        map_k, roi = cv2.getOptimalNewCameraMatrix(k, dist, (width, height), 0.0, (width, height))
        note = "Frames are undistorted with scaled_K/dist and alpha=0 map_K before GLUEMAP receives shared PINHOLE images."
        source_model = "FULL_OPENCV"
        undistort_applied = True
    elif variant == "no_undistort_scaledK":
        map_k = k.copy()
        roi = (0, 0, width, height)
        dist = np.zeros(8, dtype=np.float64)
        note = "Frames are resized only; GLUEMAP receives shared PINHOLE images with linearly scaled calibration K. No OpenCV undistortion is applied."
        source_model = "PINHOLE"
        undistort_applied = False
    elif variant == "no_undistort_official69":
        fx = width / (2.0 * math.tan(math.radians(official_video_hfov_deg) / 2.0))
        fy = fx
        map_k = np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        roi = (0, 0, width, height)
        dist = np.zeros(8, dtype=np.float64)
        note = "Frames are resized only; GLUEMAP receives shared PINHOLE images from official Anafi video HFOV with centered principal point. No OpenCV undistortion is applied."
        source_model = "PINHOLE_OFFICIAL_HFOV"
        undistort_applied = False
    elif variant == "no_undistort_official69_calibpp":
        fx = width / (2.0 * math.tan(math.radians(official_video_hfov_deg) / 2.0))
        fy = fx
        map_k = np.array([[fx, 0.0, k[0, 2]], [0.0, fy, k[1, 2]], [0.0, 0.0, 1.0]], dtype=np.float64)
        roi = (0, 0, width, height)
        dist = np.zeros(8, dtype=np.float64)
        note = "Frames are resized only; GLUEMAP receives shared PINHOLE images from official Anafi video HFOV with calibration principal point. No OpenCV undistortion is applied."
        source_model = "PINHOLE_OFFICIAL_HFOV_CALIB_PP"
        undistort_applied = False
    else:
        raise RuntimeError(f"unsupported intrinsics variant: {variant}")

    meta = {
        "variant": variant,
        "source_format": "K_dist",
        "source_model": source_model,
        "source_width": source_w,
        "source_height": source_h,
        "target_width": int(width),
        "target_height": int(height),
        "scale_x": sx,
        "scale_y": sy,
        "scaled_K": k.tolist(),
        "map_K": map_k.tolist(),
        "undistort_applied": undistort_applied,
        "undistort_new_camera_matrix": "cv2.getOptimalNewCameraMatrix" if variant == "current_undistort" else None,
        "undistort_alpha": 0.0 if variant == "current_undistort" else None,
        "undistort_roi": [int(x) for x in roi],
        "dist": dist.tolist(),
        "source_dist": dist_raw.tolist(),
        "official_video_hfov_deg": official_video_hfov_deg if "official69" in variant else None,
        "distortion_coefficients_not_scaled": True,
        "note": note,
    }
    return k, dist, map_k, meta


def list_videos(video_dir: Path) -> list[Path]:
    videos: list[Path] = []
    for suffix in ("*.MP4", "*.mp4", "*.MOV", "*.mov", "*.mkv", "*.avi"):
        videos.extend(video_dir.glob(suffix))
    videos = sorted(set(videos))
    if not videos:
        raise RuntimeError(f"no videos found under {video_dir}")
    return videos


def probe_video(video: Path, target_width: int, target_height: int, sample_fps: float) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"path": str(video), "exists": video.exists(), "opened": False}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frames / fps if fps > 0 else 0.0
    cap.release()
    return {
        "path": str(video),
        "name": video.name,
        "sequence": safe_sequence_name(video),
        "exists": video.exists(),
        "opened": True,
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "duration_seconds": duration,
        "expected_extracted_frames": int(np.floor(duration * sample_fps)) + 1,
        "target_width": int(target_width),
        "target_height": int(target_height),
    }


def extract_frames_undistorted(
    videos: list[Path],
    images_all: Path,
    fps: float,
    width: int,
    height: int,
    source_k: np.ndarray,
    dist: np.ndarray,
    map_k: np.ndarray,
    undistort: bool,
    jpeg_quality: int,
    resume: bool,
) -> list[dict[str, Any]]:
    images_all.mkdir(parents=True, exist_ok=True)
    records = []
    for video in videos:
        seq = safe_sequence_name(video)
        out_dir = images_all / seq
        existing = sorted(out_dir.glob("*.jpg"))
        if existing and resume:
            log(f"reuse extracted frames {seq}: {len(existing)}")
            records.append({"video": str(video), "sequence": seq, "saved": len(existing), "reused": True})
            continue
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise FileNotFoundError(video)
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / src_fps if src_fps > 0 else 0.0
        log(
            f"extract {video.name}: {src_w}x{src_h}, src_fps={src_fps:.3f}, "
            f"frames={frame_count}, duration={duration:.1f}s, target={width}x{height}@{fps:g}fps, "
            f"undistort={undistort}"
        )
        next_t = 0.0
        frame_idx = 0
        saved = 0
        ok, frame = cap.read()
        while ok:
            t = frame_idx / src_fps
            if t + 1e-6 >= next_t:
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                undistorted = cv2.undistort(frame, source_k, dist, None, map_k) if undistort else frame
                saved += 1
                out = out_dir / f"{saved:06d}.jpg"
                cv2.imwrite(str(out), undistorted, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                next_t += 1.0 / fps
            frame_idx += 1
            ok, frame = cap.read()
        cap.release()
        if saved == 0:
            raise RuntimeError(f"no frames extracted from {video}")
        records.append({"video": str(video), "sequence": seq, "saved": saved, "reused": False, "undistort": undistort})
        log(f"  saved {saved} frames -> {out_dir}")
    return records


def read_name_list(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def apply_frame_filters(images_root: Path, allowlist_path: Path | None, denylist_path: Path | None) -> dict[str, Any]:
    allowlist = read_name_list(allowlist_path)
    denylist = read_name_list(denylist_path)
    all_images = sorted(images_root.glob("*/*.jpg"))
    removed: list[str] = []
    kept = 0
    for path in all_images:
        rel = str(path.relative_to(images_root))
        remove = False
        if allowlist and rel not in allowlist:
            remove = True
        if denylist and rel in denylist:
            remove = True
        if remove:
            path.unlink()
            removed.append(rel)
        else:
            kept += 1
    for seq_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        if not any(seq_dir.iterdir()):
            seq_dir.rmdir()
    return {
        "allowlist": str(allowlist_path) if allowlist_path else None,
        "denylist": str(denylist_path) if denylist_path else None,
        "allowlist_count": len(allowlist),
        "denylist_count": len(denylist),
        "input_images": len(all_images),
        "kept_images": kept,
        "removed_images": len(removed),
        "removed_sample": removed[:50],
    }


def safe_sequence_name(path: Path) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in path.stem).strip("_")
    return clean or "sequence"


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024.0 ** 3)


def preflight(args: argparse.Namespace, run_dir: Path, k: np.ndarray, dist: np.ndarray, intrinsics_meta: dict[str, Any]) -> list[Path]:
    videos = list_videos(args.video_dir)
    checks: dict[str, Any] = {
        "videos": [probe_video(v, args.target_width, args.target_height, args.fps) for v in videos],
        "paths": {
            "video_dir": {"path": str(args.video_dir), "exists": args.video_dir.exists()},
            "intrinsics_json": {"path": str(args.intrinsics_json), "exists": args.intrinsics_json.exists()},
            "repo": {"path": str(args.repo), "exists": args.repo.exists()},
            "gluemap_env": {"path": str(args.gluemap_env), "exists": args.gluemap_env.exists()},
            "river_helper": {"path": str(args.river_helper), "exists": args.river_helper.exists()},
            "base_pipeline": {"path": str(args.base_pipeline), "exists": args.base_pipeline.exists()},
            "core_pipeline": {"path": str(args.core_pipeline), "exists": args.core_pipeline.exists()},
        },
        "checkpoints": {},
        "gluemap_demo": str(args.gluemap_env / "bin" / "gluemap-demo"),
        "gluemap_demo_exists": (args.gluemap_env / "bin" / "gluemap-demo").exists(),
        "disk": {"free_gb": disk_free_gb(run_dir.parent)},
        "target_resolution": [args.target_width, args.target_height],
        "sample_fps": args.fps,
        "camera": {
            "model": "PINHOLE",
            "variant": intrinsics_meta.get("variant"),
            "params": [float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])],
            "source_model": intrinsics_meta.get("source_model"),
            "undistort_applied": intrinsics_meta.get("undistort_applied"),
            "dist": [float(x) for x in dist.tolist()],
        },
    }
    for ckpt in (
        args.repo / "checkpoints" / "pi3.safetensors",
        args.repo / "checkpoints" / "dino_salad.ckpt",
        args.repo / "checkpoints" / "vggsfm_v2_0_0_track_predictor.bin",
        args.repo / "checkpoints" / "checkpoint-dg+visym.pth",
    ):
        checks["checkpoints"][str(ckpt)] = ckpt.exists()
    passed = (
        len(videos) == 6
        and all(v.get("opened") and v.get("width") == 3840 and v.get("height") == 2160 for v in checks["videos"])
        and all(23.0 <= float(v.get("fps", 0.0)) <= 25.0 for v in checks["videos"])
        and all(item["exists"] for item in checks["paths"].values())
        and checks["gluemap_demo_exists"]
        and all(checks["checkpoints"].values())
        and checks["disk"]["free_gb"] >= 100.0
    )
    strict_gate(run_dir, "preflight", passed, checks)
    write_json(run_dir / "preflight_report.json", checks)
    return videos


def image_names(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.glob("*/*.jpg"))


def run_motion_gate(core: Any, run_dir: Path, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    rejected = run_dir / "rejected_frames"
    if rejected.exists():
        shutil.rmtree(rejected)
    cfg = argparse.Namespace(
        work_dir=str(run_dir),
        motion_gate=True,
        motion_action="filter",
        motion_min_tracks=args.motion_min_tracks,
        motion_min_flow_px=args.motion_min_flow_px,
        motion_rotation_h_over_f=args.motion_rotation_h_over_f,
        motion_rotation_min_inliers=args.motion_rotation_min_inliers,
        motion_keep_rotation_every=0,
        use_rotation_bridges=True,
        resume=False,
        overwrite=True,
    )
    core.motion_gate_frames(cfg)
    return core.load_motion_roles(cfg)


def rotation_bridge_keep_count(
    triangulation_count: int,
    bridge_count: int,
    total_input_images: int,
    max_bridge_ratio: float,
    min_parallax_ratio: float,
    min_selected_ratio: float,
) -> dict[str, Any]:
    if bridge_count <= 0:
        return {
            "target_bridge_count": 0,
            "lower_bound_for_selected_ratio": 0,
            "upper_bound_for_parallax_ratio": 0,
            "upper_bound_for_bridge_ratio": 0,
            "feasible": True,
        }
    if not 0.0 < max_bridge_ratio < 1.0:
        raise ValueError("max_bridge_ratio must be between 0 and 1")
    if not 0.0 < min_parallax_ratio < 1.0:
        raise ValueError("min_parallax_ratio must be between 0 and 1")
    if not 0.0 <= min_selected_ratio <= 1.0:
        raise ValueError("min_selected_ratio must be between 0 and 1")

    lower_bound = max(0, int(math.ceil(total_input_images * min_selected_ratio)) - triangulation_count)
    upper_by_parallax = max(0, int(math.floor(triangulation_count / min_parallax_ratio - triangulation_count)))
    upper_by_bridge_ratio = max(0, int(math.floor(triangulation_count * max_bridge_ratio / (1.0 - max_bridge_ratio))))
    upper_bound = min(bridge_count, upper_by_parallax, upper_by_bridge_ratio)
    feasible = lower_bound <= upper_bound
    target = upper_bound if feasible else min(bridge_count, upper_by_parallax)
    return {
        "target_bridge_count": int(target),
        "lower_bound_for_selected_ratio": int(lower_bound),
        "upper_bound_for_parallax_ratio": int(upper_by_parallax),
        "upper_bound_for_bridge_ratio": int(upper_by_bridge_ratio),
        "feasible": bool(feasible),
    }


def allocate_bridge_counts(counts_by_sequence: dict[str, int], target_total: int) -> dict[str, int]:
    total = sum(max(0, int(v)) for v in counts_by_sequence.values())
    if target_total >= total:
        return {k: max(0, int(v)) for k, v in counts_by_sequence.items()}
    if target_total <= 0 or total <= 0:
        return {k: 0 for k in counts_by_sequence}

    raw = {
        seq: (max(0, int(count)) * target_total / total)
        for seq, count in counts_by_sequence.items()
    }
    allocated = {seq: min(max(0, int(counts_by_sequence[seq])), int(math.floor(value))) for seq, value in raw.items()}

    nonzero = [seq for seq, count in counts_by_sequence.items() if count > 0]
    if target_total >= len(nonzero):
        for seq in nonzero:
            if allocated[seq] == 0:
                allocated[seq] = 1

    while sum(allocated.values()) > target_total:
        seq = max((s for s in allocated if allocated[s] > 0), key=lambda s: (allocated[s], counts_by_sequence[s], s))
        allocated[seq] -= 1

    while sum(allocated.values()) < target_total:
        candidates = [s for s in counts_by_sequence if allocated[s] < counts_by_sequence[s]]
        if not candidates:
            break
        seq = max(candidates, key=lambda s: (raw[s] - allocated[s], counts_by_sequence[s], s))
        allocated[seq] += 1

    return allocated


def select_evenly(items: list[str], count: int) -> list[str]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    last = len(items) - 1
    indices = [int(round(i * last / (count - 1))) for i in range(count)]
    return [items[i] for i in indices]


def downsample_rotation_bridges(
    run_dir: Path,
    images_dir: Path,
    roles: dict[str, dict[str, Any]],
    total_input_images: int,
    max_bridge_ratio: float,
    min_parallax_ratio: float,
    min_selected_ratio: float,
) -> dict[str, Any]:
    names = image_names(images_dir)
    triangulation = [
        name
        for name in names
        if str(roles.get(name, {}).get("motion_class", "parallax")) in {"seed", "parallax"}
    ]
    bridges_by_sequence: dict[str, list[str]] = {}
    for name in names:
        role = roles.get(name, {})
        if str(role.get("motion_class")) != "pure_rotation":
            continue
        if str(role.get("motion_role")) != "bridge_only":
            continue
        seq = name.split("/", 1)[0]
        bridges_by_sequence.setdefault(seq, []).append(name)

    bridge_count = sum(len(v) for v in bridges_by_sequence.values())
    keep_plan = rotation_bridge_keep_count(
        len(triangulation),
        bridge_count,
        total_input_images,
        max_bridge_ratio,
        min_parallax_ratio,
        min_selected_ratio,
    )
    allocations = allocate_bridge_counts({seq: len(v) for seq, v in bridges_by_sequence.items()}, keep_plan["target_bridge_count"])
    keep_bridges = {
        name
        for seq, seq_names in bridges_by_sequence.items()
        for name in select_evenly(sorted(seq_names), allocations.get(seq, 0))
    }
    remove_bridges = sorted(
        name
        for seq_names in bridges_by_sequence.values()
        for name in seq_names
        if name not in keep_bridges
    )

    rejected_root = run_dir / "rejected_frames" / "rotation_bridge_downsample"
    for name in remove_bridges:
        src = images_dir / name
        if not src.exists():
            continue
        dst = rejected_root / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    report_path = run_dir / "motion_quality.json"
    if report_path.exists():
        report = read_json(report_path)
        removed = set(remove_bridges)
        for seq, item in report.get("sequences", {}).items():
            for record in item.get("records", []):
                name = f"{seq}/{record.get('frame', '')}"
                if name not in removed:
                    continue
                record["kept"] = False
                reason = str(record.get("reason", "")).strip()
                record["reason"] = ";".join([x for x in (reason, "rotation_bridge_downsample") if x])
            item["kept"] = len(list((images_dir / seq).glob("*.jpg")))
            item["rejected"] = int(item.get("total_before", item["kept"])) - item["kept"]
            item["post_bridge_downsample"] = {
                "input_bridge_count": len(bridges_by_sequence.get(seq, [])),
                "kept_bridge_count": allocations.get(seq, 0),
                "removed_bridge_count": max(0, len(bridges_by_sequence.get(seq, [])) - allocations.get(seq, 0)),
            }
        report["post_bridge_downsample"] = {
            **keep_plan,
            "input_selected_images": len(names),
            "input_triangulation_count": len(triangulation),
            "input_bridge_count": bridge_count,
            "kept_bridge_count": len(keep_bridges),
            "removed_bridge_count": len(remove_bridges),
            "allocations": allocations,
            "max_bridge_ratio": max_bridge_ratio,
            "min_parallax_ratio": min_parallax_ratio,
            "min_selected_ratio": min_selected_ratio,
        }
        write_json(report_path, report)

    summary = {
        **keep_plan,
        "input_selected_images": len(names),
        "input_triangulation_count": len(triangulation),
        "input_bridge_count": bridge_count,
        "kept_bridge_count": len(keep_bridges),
        "removed_bridge_count": len(remove_bridges),
        "allocations": allocations,
        "output_selected_images": len(names) - len(remove_bridges),
        "output_parallax_or_seed_ratio": len(triangulation) / max(len(names) - len(remove_bridges), 1),
    }
    write_json(run_dir / "motion_bridge_downsample.json", summary)
    if remove_bridges:
        log(
            "rotation bridge downsample: "
            f"kept={len(keep_bridges)}/{bridge_count}, removed={len(remove_bridges)}, "
            f"parallax_or_seed_ratio={summary['output_parallax_or_seed_ratio']:.3f}"
        )
    return summary


def write_manifest_with_motion(
    run_dir: Path,
    images_dir: Path,
    site_name: str,
    roles: dict[str, dict[str, Any]],
    intrinsics_meta: dict[str, Any],
    sample_fps: float,
) -> dict[str, Any]:
    names = image_names(images_dir)
    groups: dict[str, int] = {}
    class_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    frames = []
    for name in names:
        img = cv2.imread(str(images_dir / name), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(images_dir / name)
        h, w = img.shape[:2]
        seq = name.split("/", 1)[0]
        groups[seq] = groups.get(seq, 0) + 1
        role = roles.get(name, {"motion_class": "parallax", "motion_role": "triangulation"})
        motion_class = str(role.get("motion_class", "parallax"))
        motion_role = str(role.get("motion_role", "triangulation"))
        class_counts[motion_class] += 1
        role_counts[motion_role] += 1
        frames.append(
            {
                "name": name,
                "width": int(w),
                "height": int(h),
                "motion_class": motion_class,
                "motion_role": motion_role,
            }
        )
    k = intrinsics_meta["map_K"]
    manifest = {
        "site_name": site_name,
        "image_root": str(images_dir),
        "fps": sample_fps,
        "total_frames": len(frames),
        "groups": groups,
        "motion_class_counts": dict(class_counts),
        "motion_role_counts": dict(role_counts),
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


def black_border_metrics(images_dir: Path, limit_per_sequence: int = 3) -> dict[str, Any]:
    ratios = []
    sampled = []
    by_seq: dict[str, list[Path]] = {}
    for path in sorted(images_dir.glob("*/*.jpg")):
        by_seq.setdefault(path.parent.name, []).append(path)
    for seq, paths in by_seq.items():
        picks = paths[:limit_per_sequence]
        if len(paths) > limit_per_sequence:
            picks.append(paths[len(paths) // 2])
            picks.append(paths[-1])
        for path in picks:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            b = max(4, int(round(min(h, w) * 0.02)))
            mask = np.zeros((h, w), dtype=bool)
            mask[:b, :] = True
            mask[-b:, :] = True
            mask[:, :b] = True
            mask[:, -b:] = True
            dark = np.all(img <= 5, axis=2)
            ratio = float(dark[mask].mean())
            ratios.append(ratio)
            sampled.append({"image": str(path.relative_to(images_dir)), "black_border_ratio": ratio})
    return {
        "sampled": sampled,
        "max_black_border_ratio": max(ratios) if ratios else None,
        "mean_black_border_ratio": float(np.mean(ratios)) if ratios else None,
    }


def validate_intrinsics(run_dir: Path, intrinsics_meta: dict[str, Any], images_dir: Path) -> None:
    border = black_border_metrics(images_dir)
    metrics = {**intrinsics_meta, "border_check": border}
    passed = (
        intrinsics_meta["target_width"] == 1920
        and intrinsics_meta["target_height"] == 1080
        and intrinsics_meta["source_model"] in {"FULL_OPENCV", "PINHOLE", "PINHOLE_OFFICIAL_HFOV", "PINHOLE_OFFICIAL_HFOV_CALIB_PP"}
        and bool(intrinsics_meta["distortion_coefficients_not_scaled"])
        and len(intrinsics_meta["dist"]) == 8
        and (border["max_black_border_ratio"] is None or border["max_black_border_ratio"] <= 0.30)
    )
    strict_gate(run_dir, "intrinsics", passed, metrics)


def validate_selection(
    run_dir: Path,
    selection: dict[str, Any],
    manifest: dict[str, Any],
    min_selected_images: int,
    min_group_images: int,
) -> None:
    total = int(selection["total_images"])
    selected = int(manifest["total_frames"])
    groups = {k: int(v) for k, v in manifest["groups"].items()}
    class_counts = manifest.get("motion_class_counts", {})
    role_counts = manifest.get("motion_role_counts", {})
    parallax = int(class_counts.get("parallax", 0)) + int(class_counts.get("seed", 0))
    hover = int(class_counts.get("hover", 0))
    bridge = int(role_counts.get("bridge_only", 0))
    metrics = {
        "min_selected_images": min_selected_images,
        "min_group_images": min_group_images,
        "total": total,
        "quality_selected": int(selection["selected_images"]),
        "selected": selected,
        "dropped": total - selected,
        "selected_ratio": selected / max(total, 1),
        "blur_drop_fraction": float(selection["blur_drop_fraction"]),
        "duplicate_threshold": float(selection["duplicate_threshold"]),
        "groups": groups,
        "motion_class_counts": class_counts,
        "motion_role_counts": role_counts,
        "parallax_or_seed_ratio": parallax / max(selected, 1),
        "hover_ratio": hover / max(selected, 1),
        "bridge_frames": bridge,
        "quality_report": selection.get("frame_quality_csv"),
    }
    passed = (
        selected >= min_selected_images
        and len(groups) == 6
        and all(count >= min_group_images for count in groups.values())
        and metrics["selected_ratio"] >= 0.65
        and metrics["blur_drop_fraction"] <= 0.20
        and metrics["parallax_or_seed_ratio"] >= 0.70
        and metrics["hover_ratio"] <= 0.05
    )
    strict_gate(run_dir, "selection_motion_quality", passed, metrics)


def write_gluemap_config(run_dir: Path, repo: Path, images_dir: Path, seed_dir: Path, args: argparse.Namespace) -> Path:
    config = {
        "chosen_model": "pi3",
        "path_feedforward": str(repo / "checkpoints" / "pi3.safetensors"),
        "path_retrieval": str(repo / "checkpoints" / "dino_salad.ckpt"),
        "path_tracker": str(repo / "checkpoints" / "vggsfm_v2_0_0_track_predictor.bin"),
        "path_dg": str(repo / "checkpoints" / "checkpoint-dg+visym.pth"),
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
        "skip_doppelgangers": not bool(args.use_doppelgangers),
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
    status = "success" if proc.returncode == 0 else "failed"
    record_stage(run_dir, stage, started, status, log=str(log_path), cmd=cmd)
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-120:])
        raise SystemExit(f"{stage} failed with exit {proc.returncode}; log tail:\n{tail}")


def read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    if not path.exists():
        return pairs
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    return pairs


def pair_graph_metrics(ref_names: list[str], pairs: list[tuple[str, str]]) -> dict[str, Any]:
    nodes = sorted(set(ref_names))
    known = set(nodes)
    adj = {n: set() for n in nodes}
    cross_sequence = 0
    valid_pairs = 0
    for a, b in pairs:
        if a not in known or b not in known or a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)
        valid_pairs += 1
        if a.split("/", 1)[0] != b.split("/", 1)[0]:
            cross_sequence += 1
    seen = set()
    components = []
    for n in nodes:
        if n in seen:
            continue
        q = deque([n])
        seen.add(n)
        size = 0
        while q:
            cur = q.popleft()
            size += 1
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        components.append(size)
    degrees = [len(adj[n]) for n in nodes]
    return {
        "total_frames": len(nodes),
        "pairs": valid_pairs,
        "connected_components": len(components),
        "component_sizes": sorted(components, reverse=True),
        "largest_component": max(components) if components else 0,
        "largest_component_ratio": (max(components) / len(nodes)) if nodes and components else 0.0,
        "isolated_images": sum(1 for d in degrees if d == 0),
        "median_pair_degree": float(np.median(degrees)) if degrees else 0.0,
        "min_pair_degree": min(degrees) if degrees else 0,
        "max_pair_degree": max(degrees) if degrees else 0,
        "cross_sequence_pairs": cross_sequence,
    }


def validate_pair_graph(run_dir: Path, manifest: dict[str, Any]) -> None:
    names = [frame["name"] for frame in manifest["frames"]]
    metrics = pair_graph_metrics(names, read_pairs(run_dir / "gluemap" / "pairs.txt"))
    passed = (
        metrics["connected_components"] == 1
        and metrics["isolated_images"] == 0
        and metrics["median_pair_degree"] >= 20.0
        and metrics["cross_sequence_pairs"] > 0
    )
    strict_gate(run_dir, "gluemap_pair_graph", passed, metrics)


def database_stats(path: Path) -> dict[str, Any]:
    con = sqlite3.connect(str(path))
    cur = con.cursor()
    stats: dict[str, Any] = {"path": str(path), "bytes": path.stat().st_size}
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


def validate_gluemap_database(run_dir: Path, selected_images: int) -> None:
    stats = database_stats(run_dir / "gluemap" / "database_merged.db")
    passed = (
        stats["cameras"] == 1
        and stats["images"] == selected_images
        and stats["keypoints"] == selected_images
        and stats["two_view_geometries"] >= selected_images * 8
        and stats["two_view_geometry_rows"]["mean"] >= 50.0
    )
    strict_gate(run_dir, "gluemap_database", passed, stats)


def validate_model(run_dir: Path, selected_images: int, summary: dict[str, Any]) -> None:
    registered = int(summary["registered_images"])
    points = int(summary["points3D"])
    reproj = summary["reprojection_stats"]
    drift = summary["intrinsics_drift"]
    metrics = {
        **summary,
        "selected_images": selected_images,
        "registered_ratio": registered / max(selected_images, 1),
        "points_per_registered": points / max(registered, 1),
    }
    passed = (
        registered >= int(np.ceil(selected_images * 0.95))
        and points >= registered * 500
        and float(summary["mean_reprojection_error"]) <= 4.5
        and float(reproj["p95_px"]) < 8.0
        and int(reproj["invalid_projection_count"]) <= max(100, int(reproj["observation_count"] * 0.001))
        and float(drift["max_focal_relative_drift"]) <= 1e-9
        and float(drift["max_principal_point_pixel_drift"]) <= 1e-9
    )
    strict_gate(run_dir, "gluemap_model", passed, metrics)


def validate_ply(run_dir: Path, summary: dict[str, Any], ply_stats: dict[str, Any]) -> None:
    points = max(int(summary["points3D"]), 1)
    rgb = int(ply_stats["rgb_points"])
    metrics = {**ply_stats, "model_points3D": int(summary["points3D"]), "rgb_ratio": rgb / points}
    passed = rgb >= int(points * 0.90) and int(ply_stats["ply_bytes"]) > 1_000_000
    strict_gate(run_dir, "rgb_ply", passed, metrics)


def write_report(
    run_dir: Path,
    args: argparse.Namespace,
    videos: list[dict[str, Any]],
    selection: dict[str, Any],
    manifest: dict[str, Any],
    model_summary: dict[str, Any],
    ply_stats: dict[str, Any],
) -> None:
    report = {
        "site_name": args.site_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "G1 build map only; holdout localization, RTX5060 replay, and safe-zone QA are out of this run.",
        "input": {"video_dir": str(args.video_dir), "videos": videos},
        "target_resolution": [args.target_width, args.target_height],
        "fps": args.fps,
        "selection": selection,
        "manifest_summary": {
            "groups": manifest["groups"],
            "motion_class_counts": manifest.get("motion_class_counts", {}),
            "motion_role_counts": manifest.get("motion_role_counts", {}),
        },
        "model_summary": model_summary,
        "ply_stats": ply_stats,
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
        f"# Fuhe Bridge GLUEMAP Build: {args.site_name}",
        "",
        "## Scope",
        "",
        "- G1 build-map QA only.",
        "- Holdout localization, RTX5060 replay, and safe-zone QA are not part of this run.",
        "",
        "## Input",
        "",
        f"- video dir: `{args.video_dir}`",
        f"- videos: `{len(videos)}`",
        f"- sample fps: `{args.fps}`",
        f"- target resolution: `{args.target_width}x{args.target_height}`",
        f"- selected images: `{manifest['total_frames']}/{selection['total_images']}`",
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
        f"- p95 reprojection error: `{model_summary['reprojection_stats']['p95_px']}`",
        f"- RGB points: `{ply_stats['rgb_points']}`",
        f"- PLY bytes: `{ply_stats['ply_bytes']}`",
    ]
    (run_dir / "BUILD_GLUEMAP_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build the Fuhe Bridge GLUEMAP map with strict G1 QA gates.")
    ap.add_argument("--workspace", type=Path, default=WORKSPACE)
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--gluemap-env", type=Path, default=DEFAULT_GLUEMAP_ENV)
    ap.add_argument("--river-helper", type=Path, default=DEFAULT_RIVER_HELPER)
    ap.add_argument("--base-pipeline", type=Path, default=DEFAULT_BASE_PIPELINE)
    ap.add_argument("--core-pipeline", type=Path, default=DEFAULT_CORE_PIPELINE)
    ap.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--intrinsics-json", type=Path, default=DEFAULT_INTRINSICS)
    ap.add_argument(
        "--intrinsics-variant",
        choices=("current_undistort", "no_undistort_scaledK", "no_undistort_official69", "no_undistort_official69_calibpp"),
        default="current_undistort",
    )
    ap.add_argument("--official-video-hfov-deg", type=float, default=69.0)
    ap.add_argument("--frame-allowlist", type=Path)
    ap.add_argument("--frame-denylist", type=Path)
    ap.add_argument("--min-selected-images", type=int, default=500)
    ap.add_argument("--min-group-images", type=int, default=40)
    ap.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    ap.add_argument("--site-name", default="fuhe_bridge_gluemap_pi3_1fps_1920")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--target-width", type=int, default=1920)
    ap.add_argument("--target-height", type=int, default=1080)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--blur-drop-fraction", type=float, default=0.15)
    ap.add_argument("--duplicate-threshold", type=float, default=0.012)
    ap.add_argument("--motion-min-tracks", type=int, default=20)
    ap.add_argument("--motion-min-flow-px", type=float, default=1.5)
    ap.add_argument("--motion-rotation-h-over-f", type=float, default=0.85)
    ap.add_argument("--motion-rotation-min-inliers", type=int, default=20)
    ap.add_argument("--rotation-bridge-ratio-cap", type=float, default=0.29)
    ap.add_argument("--min-parallax-or-seed-ratio", type=float, default=0.70)
    ap.add_argument("--min-selected-ratio", type=float, default=0.65)
    ap.add_argument("--num-track-per-img", type=int, default=2048)
    ap.add_argument("--num-neighbors", type=int, default=100)
    ap.add_argument("--num-neighbors-sequential", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--retrieval-batch-size", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--use-doppelgangers", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fixed-intrinsics-ba", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--skip-gluemap", action="store_true")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.workspace / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    river = import_module(args.river_helper, "river_gluemap_helper")
    base = import_module(args.base_pipeline, "football_gluemap_base")
    core = import_module(args.core_pipeline, "build_localizable_map_core")

    source_k, dist, map_k, intrinsics_meta = parse_intrinsics(
        args.intrinsics_json,
        args.target_width,
        args.target_height,
        args.intrinsics_variant,
        args.official_video_hfov_deg,
    )
    write_json(run_dir / "intrinsics_scaled_debug.json", intrinsics_meta)
    videos = preflight(args, run_dir, map_k, dist, intrinsics_meta)

    started = time.time()
    video_records = extract_frames_undistorted(
        videos,
        run_dir / "images_all",
        args.fps,
        args.target_width,
        args.target_height,
        source_k,
        dist,
        map_k,
        bool(intrinsics_meta["undistort_applied"]),
        args.jpeg_quality,
        args.resume,
    )
    record_stage(run_dir, "extract_undistort", started, "success")
    write_json(run_dir / "video_records.json", video_records)

    if args.frame_allowlist or args.frame_denylist:
        started = time.time()
        frame_filter = apply_frame_filters(run_dir / "images_all", args.frame_allowlist, args.frame_denylist)
        record_stage(run_dir, "frame_filter", started, "success", **frame_filter)
        write_json(run_dir / "frame_filter.json", frame_filter)

    started = time.time()
    selection = river.prepare_clean_subset(
        run_dir / "images_all",
        run_dir / "images",
        run_dir / "quality",
        args.blur_drop_fraction,
        args.duplicate_threshold,
    )
    record_stage(run_dir, "quality_filter", started, "success")

    started = time.time()
    roles = run_motion_gate(core, run_dir, args)
    bridge_downsample = downsample_rotation_bridges(
        run_dir,
        run_dir / "images",
        roles,
        int(selection["total_images"]),
        args.rotation_bridge_ratio_cap,
        args.min_parallax_or_seed_ratio,
        args.min_selected_ratio,
    )
    record_stage(run_dir, "motion_gate", started, "success", bridge_downsample=bridge_downsample)

    started = time.time()
    manifest = write_manifest_with_motion(run_dir, run_dir / "images", args.site_name, roles, intrinsics_meta, args.fps)
    seed_dir = river.write_colmap_seed(run_dir, run_dir / "images", map_k, args.target_width, args.target_height)
    config_path = write_gluemap_config(run_dir, args.repo, run_dir / "images", seed_dir, args)
    record_stage(run_dir, "config", started, "success")
    validate_intrinsics(run_dir, intrinsics_meta, run_dir / "images")
    validate_selection(run_dir, selection, manifest, args.min_selected_images, args.min_group_images)

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

    validate_pair_graph(run_dir, manifest)
    validate_gluemap_database(run_dir, int(manifest["total_frames"]))

    model_dir = river.find_model_dir(run_dir)
    seed_params = [float(map_k[0, 0]), float(map_k[1, 1]), float(map_k[0, 2]), float(map_k[1, 2])]
    if args.fixed_intrinsics_ba:
        model_dir = base.run_fixed_intrinsics_ba(
            run_dir,
            model_dir,
            seed_params,
            args.target_width,
            args.target_height,
            args.resume,
        )
    model = base.strict_model_summary(model_dir, seed_params)
    validate_model(run_dir, int(manifest["total_frames"]), model)

    started = time.time()
    ply_stats = river.export_rgb_ply(model_dir, run_dir / "images", run_dir / "deploy" / "map_rgb.ply", 8.0)
    record_stage(run_dir, "color", started, "success")
    validate_ply(run_dir, model, ply_stats)
    write_report(run_dir, args, [probe_video(v, args.target_width, args.target_height, args.fps) for v in videos], selection, manifest, model, ply_stats)
    log(f"done: {run_dir}")
    log(f"report: {run_dir / 'BUILD_GLUEMAP_REPORT.md'}")
    log(f"ply: {run_dir / 'deploy' / 'map_rgb.ply'}")


if __name__ == "__main__":
    main()
