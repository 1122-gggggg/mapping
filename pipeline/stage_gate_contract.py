#!/usr/bin/env python3
"""Evaluate whether a build run satisfies the one-click localizable-map contract."""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateRequirement:
    key: str
    title: str
    purpose: str
    gate_aliases: tuple[str, ...] = ()
    required: bool = True
    artifact_alternatives: tuple[tuple[str, ...], ...] = ()
    artifact_can_satisfy_missing_gate: bool = False
    operator_action: str = ""


SFM_MIN_REGISTERED_IMAGES = 60
SFM_MIN_REGISTERED_RATIO = 0.80
SFM_MIN_POINTS3D = 30000
SFM_MIN_POINTS_PER_REGISTERED = 500.0
SFM_MAX_MEAN_REPROJECTION_ERROR = 2.0
SFM_MODEL_GATE_PRIORITY = ("dense_retriangulated_model", "glomap", "gluemap_model")
VIEW_GRAPH_MIN_PAIRS = 500
VIEW_GRAPH_MIN_PAIRS_PER_FRAME = 4.0
VIEW_GRAPH_MIN_LARGEST_COMPONENT_RATIO = 0.90
VIEW_GRAPH_GATE_PRIORITY = ("gluemap_pair_graph", "pairs")
DATABASE_GATE_PRIORITY = ("db", "gluemap_database")
DATABASE_MIN_KEYPOINTS_PER_IMAGE = 50
DATABASE_MIN_AVG_KEYPOINTS_PER_IMAGE = 500.0
DATABASE_MIN_NONZERO_GEOMETRY_RATIO = 0.90
DATABASE_MIN_AVG_TWO_VIEW_INLIERS = 30.0
POINT_CLOUD_MIN_BYTES = 128 * 1024
POINT_CLOUD_MIN_VERTICES = 10000
POINT_CLOUD_MIN_VERTEX_RATIO_VS_POINTS3D = 0.30
REPORT_MIN_GATE_JSONS = 8
REPORT_REQUIRED_OUTPUT_KEYS = (
    "glomap_model",
    "rgb_point_cloud",
    "frame_manifest",
    "intrinsics",
    "config",
    "stage_times",
)
REPORT_BUNDLE_OUTPUT_KEYS = ("triangulated_bundle", "tracking_bundle", "snap_bundle")
PREFLIGHT_CORE_TOOLS = ("ffmpeg", "ffprobe", "glomap", "python_sfm", "python_sfmdb")
PREFLIGHT_MIN_DISK_FREE_GB = 50.0
PREFLIGHT_MIN_VIDEO_DURATION_SECONDS = 10.0
PREFLIGHT_MIN_VIDEO_FPS = 1.0
PREFLIGHT_MIN_TARGET_WIDTH = 640
PREFLIGHT_MIN_TARGET_HEIGHT = 480
PLY_MAX_HEADER_BYTES = 1024 * 1024
PLY_SCALAR_TYPE_BYTES = {
    "char": 1,
    "uchar": 1,
    "int8": 1,
    "uint8": 1,
    "short": 2,
    "ushort": 2,
    "int16": 2,
    "uint16": 2,
    "int": 4,
    "uint": 4,
    "int32": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}
FRAME_MIN_SELECTED = 60
FRAME_MIN_SELECTED_RATIO = 0.65
FRAME_MIN_PARALLAX_OR_SEED_RATIO = 0.65
FRAME_MAX_HOVER_RATIO = 0.05
FRAME_MIN_GROUP_IMAGES = 15
FRAME_MIN_EXTRACT_KEEP_RATIO = 0.15
FRAME_MIN_EXTRACT_PARALLAX_RATIO = 0.10
FRAME_GATE_PRIORITY = ("selection_motion_quality", "extract")


ONE_CLICK_REQUIREMENTS: tuple[GateRequirement, ...] = (
    GateRequirement(
        "preflight",
        "Input and environment preflight",
        "Verify videos, paths, tools, GPU, disk, target resolution, and camera settings before expensive work.",
        ("preflight",),
        operator_action="Fix missing videos/tools/GPU/disk or camera config before extracting frames.",
    ),
    GateRequirement(
        "preflight_quality",
        "Input video and environment quality metrics",
        "Hard-gate input video metadata, core tools, disk, GPU status, target resolution, and camera sanity before extraction.",
        ("preflight",),
        operator_action=(
            "Fix missing/short/unreadable videos, unavailable core tools, low disk, GPU failure, or invalid camera settings "
            "before frame extraction."
        ),
    ),
    GateRequirement(
        "frame_quality",
        "Frame extraction and motion quality",
        "Reject unusable hover/blur/rotation-only frame sets and keep enough parallax frames for triangulation.",
        ("extract", "selection_motion_quality"),
        operator_action="Re-record videos with more translation parallax or relax only diagnostic thresholds.",
    ),
    GateRequirement(
        "frame_motion_quality",
        "Frame motion quality metrics",
        "Hard-gate selected frame count, retained motion coverage, parallax/seed support, hover dominance, and bridge frames.",
        ("extract", "selection_motion_quality"),
        operator_action=(
            "Re-record with more translation parallax, slower passes, and explicit connector footage; do not promote "
            "hover/pure-rotation-dominated frame sets."
        ),
    ),
    GateRequirement(
        "intrinsics_manifest",
        "Frame manifest and camera intrinsics",
        "Preserve the image list, per-resolution intrinsics, and distortion/intrinsics decision used by the map.",
        ("manifest", "intrinsics"),
        operator_action="Regenerate manifest/intrinsics; do not continue with unknown camera calibration.",
    ),
    GateRequirement(
        "intrinsics_manifest_quality",
        "Manifest and intrinsics quality metrics",
        "Hard-gate frame manifest consistency, camera model support, intrinsics parameter sanity, and resize/undistort provenance.",
        ("manifest", "intrinsics"),
        operator_action=(
            "Regenerate frame_manifest.json and map_intrinsics.json from a known camera profile; run an intrinsics bake-off "
            "when distortion or firmware undistortion is ambiguous."
        ),
    ),
    GateRequirement(
        "pair_graph",
        "Retrieval and view-graph connectivity",
        "Ensure enough image pairs, connected components, cross-sequence links, and non-degenerate pair graph coverage.",
        ("pairs", "gluemap_pair_graph"),
        operator_action="Increase overlap, cross-video coverage, or retrieval/topk before dense matching.",
    ),
    GateRequirement(
        "view_graph_quality",
        "View graph quality metrics",
        "Hard-gate pair count, graph connectivity, and cross-video bridge evidence before dense matching/SfM handoff.",
        ("pairs", "gluemap_pair_graph"),
        operator_action=(
            "Increase retrieval topk, add temporal/cross-video bridge pairs, or collect overlapping connector footage "
            "before spending compute on dense matching or mapper sweeps."
        ),
    ),
    GateRequirement(
        "pair_filtering_optional",
        "Optional pair filtering safety checks",
        "When Doppelgangers++ or dense-match verification runs, failed filters must block promotion.",
        ("doppelgangers", "verify_pairs"),
        required=False,
        operator_action="Inspect repeated-structure or dense verification failures before reusing these pairs.",
    ),
    GateRequirement(
        "dense_matching_optional",
        "Dense/local correspondence checks",
        "When MV-RoMa or aggregation runs, match H5 files and aggregated features must be readable and non-empty.",
        ("mvroma", "aggregate"),
        required=False,
        operator_action="Rerun dense matching/aggregation or reduce workload only after preserving DB/H5 evidence.",
    ),
    GateRequirement(
        "database",
        "COLMAP database / match database",
        "Require a usable COLMAP-compatible DB with cameras, images, keypoints, matches, and two-view geometry.",
        ("db", "gluemap_database"),
        operator_action="Rebuild DB from preserved H5/matches; do not rerun expensive front-end unless DB evidence is bad.",
    ),
    GateRequirement(
        "database_quality",
        "COLMAP database quality metrics",
        "Hard-gate SQLite DB integrity, image/keypoint coverage, match-pair density, and verified two-view geometry.",
        ("db", "gluemap_database"),
        operator_action=(
            "Rebuild the COLMAP DB from preserved DB/H5/NPZ artifacts, or regenerate pair verification before mapper/BA sweeps."
        ),
    ),
    GateRequirement(
        "sparse_model",
        "Global SfM sparse model",
        "Require global SfM registration, points3D, reprojection statistics, and stable intrinsics.",
        ("dense_retriangulated_model", "glomap", "gluemap_model"),
        operator_action="Tune GLOMAP/BA/triangulation first; use LFOE only for bad translation-edge diagnostics.",
    ),
    GateRequirement(
        "sfm_reconstruction_quality",
        "Sparse reconstruction quality metrics",
        "Hard-gate registration coverage, point density, and reprojection error before any field handoff.",
        ("dense_retriangulated_model", "glomap", "gluemap_model"),
        operator_action=(
            "Sweep GLOMAP/BA/triangulation on the preserved DB/H5; if pair graph is connected but quality stays low, "
            "diagnose outlier edges or collect more overlap before promotion."
        ),
    ),
    GateRequirement(
        "point_cloud",
        "Colored inspection point cloud",
        "Require a non-empty RGB point cloud tied to the accepted sparse model.",
        ("color", "rgb_ply"),
        artifact_alternatives=(("deploy/map_rgb.ply",), ("deploy/map_rgb_dense_retriangulated.ply",)),
        operator_action="Regenerate colorization or inspect model/PLY mismatch.",
    ),
    GateRequirement(
        "point_cloud_quality",
        "Colored point cloud quality metrics",
        "Hard-gate PLY readability, RGB vertex properties, vertex count, and point-cloud coverage of the sparse model.",
        ("color", "rgb_ply"),
        artifact_alternatives=(("deploy/map_rgb.ply",), ("deploy/map_rgb_dense_retriangulated.ply",)),
        operator_action=(
            "Regenerate colorization from the accepted sparse model; do not hand off a tiny, non-RGB, or mismatched PLY."
        ),
    ),
    GateRequirement(
        "localization_bundle",
        "Localizable deployment bundle",
        "Require XFeat/MegaLoc bundle evidence so the output is not merely a point cloud, but a localizable map.",
        ("triangulate", "tracking"),
        artifact_alternatives=(("deploy/reloc_map_xfeat_tri.pt",), ("deploy/reloc_map_xfeat_tracking.pt",)),
        operator_action="Build tracking/triangulated bundle and verify MegaLoc 8448-d descriptors plus tracking metadata.",
    ),
    GateRequirement(
        "report_package",
        "Operator report and preserved provenance",
        "Require a human-readable and machine-readable record of parameters, outputs, gates, and timings.",
        ("report",),
        artifact_alternatives=(
            ("BUILD_LOCALIZABLE_MAP_REPORT.md",),
            ("BUILD_GLUEMAP_REPORT.md",),
            ("build_report.json",),
        ),
        artifact_can_satisfy_missing_gate=True,
        operator_action="Regenerate the report package before handing the map to a non-expert operator.",
    ),
    GateRequirement(
        "report_package_quality",
        "Report package and provenance quality",
        "Hard-gate machine-readable provenance, stage timing, config, output manifest, and gate JSON preservation.",
        ("report",),
        artifact_alternatives=(
            ("BUILD_LOCALIZABLE_MAP_REPORT.md",),
            ("BUILD_GLUEMAP_REPORT.md",),
            ("build_report.json",),
        ),
        artifact_can_satisfy_missing_gate=True,
        operator_action=(
            "Regenerate the build report package with build_report.json, build_config.json, stage_times.json, "
            "operator Markdown, output paths, and preserved gate JSON files."
        ),
    ),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def gate_path(run_dir: Path, name: str) -> Path:
    return run_dir / "gates" / f"{name}.json"


def normalize_gate(path: Path, alias: str) -> dict[str, Any]:
    try:
        data = read_json(path)
    except Exception as exc:  # pragma: no cover - exact parser errors vary.
        return {
            "name": alias,
            "path": str(path),
            "ok": False,
            "reasons": [f"cannot read gate JSON: {exc}"],
            "metrics": {},
        }
    ok = data.get("ok")
    if ok is None:
        ok = data.get("passed")
    ok = bool(ok)
    reasons = data.get("reasons")
    if not reasons and not ok:
        reasons = [f"{alias} gate failed"]
    return {
        "name": str(data.get("stage") or data.get("gate") or alias),
        "path": str(path),
        "ok": ok,
        "reasons": list(reasons or []),
        "metrics": data.get("metrics") or {},
    }


def artifact_evidence(run_dir: Path, alternatives: tuple[tuple[str, ...], ...]) -> tuple[bool, list[dict[str, Any]], list[str]]:
    if not alternatives:
        return True, [], []
    evidence: list[dict[str, Any]] = []
    missing_reasons: list[str] = []
    for alternative in alternatives:
        paths = [run_dir / rel for rel in alternative]
        existing = [path.exists() and path.stat().st_size > 0 for path in paths]
        evidence.extend(
            {
                "name": rel,
                "path": str(path),
                "ok": ok,
                "kind": "artifact",
            }
            for rel, path, ok in zip(alternative, paths, existing)
        )
        if all(existing):
            return True, evidence, []
    for alternative in alternatives:
        for rel in alternative:
            missing_reasons.append(f"missing {rel}")
    return False, evidence, missing_reasons


def evaluate_requirement(run_dir: Path, requirement: GateRequirement) -> dict[str, Any]:
    gate_evidence = [
        normalize_gate(gate_path(run_dir, alias), alias)
        for alias in requirement.gate_aliases
        if gate_path(run_dir, alias).exists()
    ]
    artifact_ok, artifacts, artifact_reasons = artifact_evidence(run_dir, requirement.artifact_alternatives)
    reasons: list[str] = []

    if gate_evidence:
        failed = [item for item in gate_evidence if not item["ok"]]
        if failed:
            for item in failed:
                reasons.extend(item["reasons"] or [f"{item['name']} gate failed"])
    elif requirement.required and requirement.gate_aliases:
        if not (requirement.artifact_can_satisfy_missing_gate and artifact_ok and artifacts):
            reasons.append(f"missing required gate: {' or '.join(requirement.gate_aliases)}")

    if requirement.required or gate_evidence:
        if not artifact_ok:
            reasons.extend(artifact_reasons)
    if requirement.key == "preflight_quality":
        reasons.extend(validate_preflight_quality(run_dir, gate_evidence))
    if requirement.key == "localization_bundle":
        reasons.extend(validate_localization_bundle_metadata(gate_evidence))
    if requirement.key == "frame_motion_quality":
        reasons.extend(validate_frame_motion_quality(gate_evidence))
    if requirement.key == "intrinsics_manifest_quality":
        reasons.extend(validate_intrinsics_manifest_quality(run_dir, gate_evidence))
    if requirement.key == "view_graph_quality":
        reasons.extend(validate_view_graph_quality(gate_evidence))
    if requirement.key == "database_quality":
        reasons.extend(validate_database_quality(run_dir, gate_evidence))
    if requirement.key == "sfm_reconstruction_quality":
        reasons.extend(validate_sfm_reconstruction_quality(gate_evidence))
    if requirement.key == "point_cloud_quality":
        reasons.extend(validate_point_cloud_quality(run_dir, gate_evidence))
    if requirement.key == "report_package_quality":
        reasons.extend(validate_report_package_quality(run_dir, gate_evidence + artifacts))

    if not requirement.required and not gate_evidence:
        status = "SKIP"
        ok = True
    else:
        status = "PASS" if not reasons else "FAIL"
        ok = not reasons

    return {
        "key": requirement.key,
        "title": requirement.title,
        "purpose": requirement.purpose,
        "required": requirement.required,
        "status": status,
        "ok": ok,
        "reasons": reasons,
        "operator_action": requirement.operator_action,
        "evidence": gate_evidence + artifacts,
    }


def validate_localization_bundle_metadata(evidence: list[dict[str, Any]]) -> list[str]:
    bundle_gates = [
        item for item in evidence
        if item.get("name") in {"triangulate", "tracking"} and item.get("ok")
    ]
    if not bundle_gates:
        return []
    reasons: list[str] = []
    for item in bundle_gates:
        name = str(item.get("name") or "localization_bundle")
        metrics = item.get("metrics") or {}
        meta = metrics.get("meta") if isinstance(metrics.get("meta"), dict) else {}
        refs = int(metrics.get("refs", 0) or 0)
        unique_refs = metrics.get("unique_refs")
        shape = metrics.get("ref_global_shape")
        total_anchors = int(metrics.get("total_3d_anchored_kp", meta.get("total_3d_anchored_kp", 0)) or 0)
        mean_anchors = float(
            metrics.get("mean_3d_anchored_per_ref", meta.get("mean_3d_anchored_per_ref", 0.0)) or 0.0
        )

        if refs <= 0:
            reasons.append(f"{name} refs=0; deployment bundle has no references")
        if unique_refs is None:
            reasons.append(f"{name} unique_refs missing")
        elif int(unique_refs) != refs:
            reasons.append(f"{name} unique_refs={int(unique_refs)} != refs={refs}")
        if not (isinstance(shape, list) and len(shape) == 2):
            reasons.append(f"{name} ref_global_shape missing or invalid: {shape}")
        else:
            if int(shape[0]) != refs:
                reasons.append(f"{name} ref_global_shape[0]={int(shape[0])} != refs={refs}")
            if int(shape[1]) != 8448:
                reasons.append(f"{name} ref_global is not MegaLoc 8448-d: {shape}")
        if not bool(meta.get("tracking_metadata")):
            reasons.append(f"{name} tracking_metadata missing")
        if total_anchors <= 0:
            reasons.append(f"{name} total_3d_anchored_kp=0")
        if mean_anchors < 50.0:
            reasons.append(f"{name} mean_3d_anchored_per_ref={mean_anchors:.1f} < 50.0")
    return reasons


def numeric_metric(metrics: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = metrics.get(name)
        if value is None or isinstance(value, bool):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def selected_view_graph_gates(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_gates = [
        item for item in evidence
        if item.get("ok") and item.get("name") in set(VIEW_GRAPH_GATE_PRIORITY)
    ]
    for name in VIEW_GRAPH_GATE_PRIORITY:
        selected = [item for item in ok_gates if item.get("name") == name]
        if selected:
            return selected
    return []


def selected_frame_motion_gates(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_gates = [
        item for item in evidence
        if item.get("ok") and item.get("name") in set(FRAME_GATE_PRIORITY)
    ]
    for name in FRAME_GATE_PRIORITY:
        selected = [item for item in ok_gates if item.get("name") == name]
        if selected:
            return selected
    return []


def validate_frame_motion_quality(evidence: list[dict[str, Any]]) -> list[str]:
    frame_gates = selected_frame_motion_gates(evidence)
    if not frame_gates:
        return []

    reasons: list[str] = []
    for item in frame_gates:
        name = str(item.get("name") or "frame_motion")
        metrics = item.get("metrics") or {}
        if name == "selection_motion_quality":
            reasons.extend(validate_selection_motion_metrics(name, metrics))
        else:
            reasons.extend(validate_extract_motion_metrics(name, metrics))
    return reasons


def load_json_for_quality(path: Path, label: str) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return None, [f"missing {label}"]
    try:
        data = read_json(path)
    except Exception as exc:
        return None, [f"cannot read {label}: {exc}"]
    if not isinstance(data, dict):
        return None, [f"{label} is not a JSON object"]
    return data, []


def parse_resolution_key(key: str) -> tuple[int, int] | None:
    if "x" not in str(key):
        return None
    left, right = str(key).lower().split("x", 1)
    try:
        width = int(left)
        height = int(right)
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def camera_core_params(model: str, params: list[Any]) -> tuple[float, float, float, float] | None:
    normalized = str(model).upper()
    try:
        values = [float(value) for value in params]
    except (TypeError, ValueError):
        return None
    if "SIMPLE_PINHOLE" in normalized or "SIMPLE_RADIAL" in normalized:
        if len(values) < 3:
            return None
        return values[0], values[0], values[1], values[2]
    if "PINHOLE" in normalized:
        if len(values) < 4:
            return None
        return values[0], values[1], values[2], values[3]
    if "OPENCV" in normalized:
        if len(values) < 8:
            return None
        return values[0], values[1], values[2], values[3]
    return None


def validate_camera_record(label: str, camera: dict[str, Any], width: int, height: int) -> list[str]:
    reasons: list[str] = []
    model = str(camera.get("model") or camera.get("camera_model") or "")
    params = camera.get("params")
    if not model:
        reasons.append(f"{label} camera model missing")
        return reasons
    if not isinstance(params, list):
        reasons.append(f"{label} params missing")
        return reasons
    core = camera_core_params(model, params)
    if core is None:
        reasons.append(f"{label} unsupported or incomplete camera model {model} params_len={len(params)}")
        return reasons
    fx, fy, cx, cy = core
    for axis, value in (("fx", fx), ("fy", fy)):
        if value <= 0:
            reasons.append(f"{label} {axis}={value:.1f} must be positive")
    if not (0 <= cx <= width):
        reasons.append(f"{label} cx={cx:.1f} outside image width {width}")
    if not (0 <= cy <= height):
        reasons.append(f"{label} cy={cy:.1f} outside image height {height}")
    return reasons


def selected_preflight_gates(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in evidence if item.get("ok") and item.get("name") == "preflight"]


def target_resolution_from_preflight(payload: dict[str, Any], gate_metrics: dict[str, Any]) -> tuple[int | None, int | None]:
    target = payload.get("target_resolution")
    if not isinstance(target, dict):
        target = gate_metrics.get("target_resolution")
    if not isinstance(target, dict):
        return None, None
    width = numeric_metric(target, "width", "target_width")
    height = numeric_metric(target, "height", "target_height")
    if width is None or height is None:
        return None, None
    return int(width), int(height)


def validate_preflight_tools(payload: dict[str, Any], gate_metrics: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    tool_exists = payload.get("tool_exists")
    if not isinstance(tool_exists, dict):
        tool_exists = gate_metrics.get("tool_exists")
    if not isinstance(tool_exists, dict) or not tool_exists:
        return ["preflight tool_exists missing"]
    metrics["tool_exists"] = {str(key): bool(value) for key, value in tool_exists.items()}
    for tool in PREFLIGHT_CORE_TOOLS:
        if tool_exists.get(tool) is not True:
            reasons.append(f"preflight tool {tool} missing")
    return reasons


def validate_preflight_videos(
    payload: dict[str, Any],
    gate_metrics: dict[str, Any],
    target_width: int | None,
    target_height: int | None,
    metrics: dict[str, Any],
) -> list[str]:
    videos = payload.get("videos")
    reasons: list[str] = []
    if not isinstance(videos, list) or not videos:
        return ["preflight videos missing"]
    metrics["video_count"] = len(videos)
    gate_video_count = numeric_metric(gate_metrics, "videos", "video_count")
    if gate_video_count is not None and int(gate_video_count) != len(videos):
        reasons.append(f"preflight video_count={len(videos)} != gate videos={int(gate_video_count)}")

    stems: set[str] = set()
    total_duration = 0.0
    min_expected_frames: int | None = None
    for idx, video in enumerate(videos):
        if not isinstance(video, dict):
            reasons.append(f"preflight video[{idx}] is not an object")
            continue
        if video.get("exists") is not True:
            reasons.append(f"preflight video[{idx}] exists=false")
        if not video.get("path"):
            reasons.append(f"preflight video[{idx}] path missing")
        if not video.get("codec_name"):
            reasons.append(f"preflight video[{idx}] codec_name missing")
        width = numeric_metric(video, "width")
        height = numeric_metric(video, "height")
        fps = numeric_metric(video, "fps")
        duration = numeric_metric(video, "duration")
        frames = numeric_metric(video, "nb_frames", "frames")
        expected = numeric_metric(video, "expected_extracted_frames")
        target_video_width = numeric_metric(video, "target_width")
        target_video_height = numeric_metric(video, "target_height")
        stem = str(video.get("sanitized_stem") or "")

        if width is None or height is None or int(width) <= 0 or int(height) <= 0:
            reasons.append(f"preflight video[{idx}] invalid source resolution")
        if fps is None or fps < PREFLIGHT_MIN_VIDEO_FPS:
            reasons.append(f"preflight video[{idx}] fps={fps if fps is not None else '<missing>'} < {PREFLIGHT_MIN_VIDEO_FPS:.1f}")
        if duration is None or duration < PREFLIGHT_MIN_VIDEO_DURATION_SECONDS:
            formatted = "<missing>" if duration is None else f"{duration:.1f}"
            reasons.append(
                f"preflight video[{idx}] duration={formatted} < {PREFLIGHT_MIN_VIDEO_DURATION_SECONDS:.1f}s"
            )
        else:
            total_duration += float(duration)
        if frames is None or int(frames) <= 0:
            reasons.append(f"preflight video[{idx}] nb_frames missing or non-positive")
        if expected is None:
            reasons.append(f"preflight video[{idx}] expected_extracted_frames missing")
        else:
            min_expected_frames = int(expected) if min_expected_frames is None else min(min_expected_frames, int(expected))
            if int(expected) < FRAME_MIN_GROUP_IMAGES:
                reasons.append(
                    f"preflight video[{idx}] expected_extracted_frames={int(expected)} < {FRAME_MIN_GROUP_IMAGES}"
                )
        if target_width is not None and target_video_width is not None and int(target_video_width) != target_width:
            reasons.append(f"preflight video[{idx}] target_width={int(target_video_width)} != {target_width}")
        if target_height is not None and target_video_height is not None and int(target_video_height) != target_height:
            reasons.append(f"preflight video[{idx}] target_height={int(target_video_height)} != {target_height}")
        if not stem:
            reasons.append(f"preflight video[{idx}] sanitized_stem missing")
        elif stem in stems:
            reasons.append(f"preflight video[{idx}] duplicate sanitized_stem={stem}")
        stems.add(stem)

    metrics["total_video_duration_seconds"] = round(total_duration, 3)
    metrics["min_expected_extracted_frames"] = min_expected_frames
    return reasons


def validate_preflight_quality(run_dir: Path, evidence: list[dict[str, Any]]) -> list[str]:
    preflight_gates = selected_preflight_gates(evidence)
    if not preflight_gates:
        return []
    gate = preflight_gates[0]
    gate_metrics = gate.get("metrics") if isinstance(gate.get("metrics"), dict) else {}
    metrics: dict[str, Any] = {}
    report, reasons = load_json_for_quality(run_dir / "preflight_report.json", "preflight_report.json")
    if report is None:
        gate.setdefault("metrics", {})["preflight_quality"] = metrics
        return reasons

    target_width, target_height = target_resolution_from_preflight(report, gate_metrics)
    metrics["target_resolution"] = {"width": target_width, "height": target_height}
    if target_width is None or target_height is None:
        reasons.append("preflight target_resolution missing")
    else:
        if target_width < PREFLIGHT_MIN_TARGET_WIDTH:
            reasons.append(f"preflight target_width={target_width} < {PREFLIGHT_MIN_TARGET_WIDTH}")
        if target_height < PREFLIGHT_MIN_TARGET_HEIGHT:
            reasons.append(f"preflight target_height={target_height} < {PREFLIGHT_MIN_TARGET_HEIGHT}")

    reasons.extend(validate_preflight_tools(report, gate_metrics, metrics))
    reasons.extend(validate_preflight_videos(report, gate_metrics, target_width, target_height, metrics))

    disk = report.get("disk") if isinstance(report.get("disk"), dict) else {}
    disk_free_gb = numeric_metric(disk, "free_gb")
    if disk_free_gb is None:
        disk_free_gb = numeric_metric(gate_metrics, "disk_free_gb")
    metrics["disk_free_gb"] = disk_free_gb
    if disk_free_gb is None or disk_free_gb < PREFLIGHT_MIN_DISK_FREE_GB:
        value = "<missing>" if disk_free_gb is None else f"{disk_free_gb:.1f}"
        reasons.append(f"preflight disk_free_gb={value} < {PREFLIGHT_MIN_DISK_FREE_GB:.1f}")

    gpu = report.get("gpu")
    if not isinstance(gpu, dict):
        gpu = gate_metrics.get("gpu")
    metrics["gpu"] = gpu if isinstance(gpu, dict) else None
    if not isinstance(gpu, dict):
        reasons.append("preflight gpu status missing")
    elif bool(gpu.get("required", False)) and gpu.get("ok") is not True:
        reasons.append("preflight gpu required but ok=false")

    camera = report.get("camera")
    if not isinstance(camera, dict):
        camera = gate_metrics.get("camera")
    if not isinstance(camera, dict):
        reasons.append("preflight camera missing")
    elif target_width is not None and target_height is not None:
        reasons.extend(validate_camera_record("preflight camera", camera, target_width, target_height))

    extract_fps = numeric_metric(report, "fps")
    metrics["extract_fps"] = extract_fps
    if extract_fps is None or extract_fps <= 0:
        reasons.append("preflight extraction fps missing or non-positive")

    gate.setdefault("metrics", {})["preflight_quality"] = metrics
    return reasons


def manifest_resolutions(manifest: dict[str, Any], reasons: list[str]) -> set[str]:
    total = numeric_metric(manifest, "total_frames")
    if total is None:
        reasons.append("frame_manifest total_frames missing")
    elif int(total) < FRAME_MIN_SELECTED:
        reasons.append(f"frame_manifest total_frames={int(total)} < {FRAME_MIN_SELECTED}")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        reasons.append("frame_manifest frames missing")
        return set()
    if total is not None and len(frames) != int(total):
        reasons.append(f"frame_manifest frames={len(frames)} != total_frames={int(total)}")
    resolutions: set[str] = set()
    for idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            reasons.append(f"frame_manifest frame[{idx}] is not an object")
            continue
        name = frame.get("name")
        if not name:
            reasons.append(f"frame_manifest frame[{idx}] missing name")
        width = numeric_metric(frame, "width")
        height = numeric_metric(frame, "height")
        if width is None or height is None:
            reasons.append(f"frame_manifest frame[{idx}] missing width/height")
            continue
        if int(width) <= 0 or int(height) <= 0:
            reasons.append(f"frame_manifest frame[{idx}] invalid size {int(width)}x{int(height)}")
            continue
        resolutions.add(f"{int(width)}x{int(height)}")
    return resolutions


def validate_map_intrinsics(intrinsics: dict[str, Any], manifest_resolution_keys: set[str]) -> list[str]:
    reasons: list[str] = []
    by_resolution = intrinsics.get("intrinsics_by_resolution")
    if isinstance(by_resolution, dict) and by_resolution:
        available = set()
        for key, camera in by_resolution.items():
            parsed = parse_resolution_key(str(key))
            if parsed is None:
                reasons.append(f"intrinsics_by_resolution key invalid: {key}")
                continue
            width, height = parsed
            available.add(f"{width}x{height}")
            if not isinstance(camera, dict):
                reasons.append(f"intrinsics_by_resolution {key} camera is not an object")
                continue
            reasons.extend(validate_camera_record(str(camera.get("model") or key), camera, width, height))
        missing = sorted(manifest_resolution_keys - available)
        if missing:
            reasons.append(f"map_intrinsics missing resolutions for manifest: {', '.join(missing)}")
        return reasons

    model = str(intrinsics.get("camera_model") or intrinsics.get("model") or "")
    params = intrinsics.get("params")
    width = numeric_metric(intrinsics, "image_width", "target_width", "width")
    height = numeric_metric(intrinsics, "image_height", "target_height", "height")
    if not model or not isinstance(params, list) or width is None or height is None:
        reasons.append("map_intrinsics missing intrinsics_by_resolution or shared camera_model/params/size")
        return reasons
    reasons.extend(validate_camera_record(model, {"model": model, "params": params}, int(width), int(height)))
    expected = f"{int(width)}x{int(height)}"
    if manifest_resolution_keys and manifest_resolution_keys != {expected}:
        reasons.append(
            "shared map_intrinsics resolution "
            f"{expected} does not match manifest resolutions {', '.join(sorted(manifest_resolution_keys))}"
        )

    has_source_distortion = bool(intrinsics.get("source_dist")) or str(intrinsics.get("source_format") or "").upper() in {
        "K_DIST",
        "FULL_OPENCV",
    }
    if has_source_distortion:
        has_decision = (
            "undistort_applied" in intrinsics
            or bool(intrinsics.get("undistort_new_camera_matrix"))
            or "PINHOLE" in str(intrinsics.get("source_model") or "").upper()
        )
        if not has_decision:
            reasons.append("map_intrinsics missing explicit undistort decision for distorted source intrinsics")
    for key in ("source_width", "source_height", "target_width", "target_height"):
        if key in intrinsics:
            value = numeric_metric(intrinsics, key)
            if value is None or value <= 0:
                reasons.append(f"map_intrinsics {key} invalid: {intrinsics.get(key)}")
    for key in ("scale_x", "scale_y"):
        if key in intrinsics:
            value = numeric_metric(intrinsics, key)
            if value is None or value <= 0:
                reasons.append(f"map_intrinsics {key} invalid: {intrinsics.get(key)}")
    return reasons


def validate_intrinsics_manifest_quality(run_dir: Path, evidence: list[dict[str, Any]]) -> list[str]:
    if not any(item.get("ok") and item.get("name") in {"manifest", "intrinsics"} for item in evidence):
        return []
    manifest, manifest_reasons = load_json_for_quality(run_dir / "frame_manifest.json", "frame_manifest.json")
    intrinsics, intrinsics_reasons = load_json_for_quality(run_dir / "map_intrinsics.json", "map_intrinsics.json")
    reasons = manifest_reasons + intrinsics_reasons
    manifest_resolution_keys: set[str] = set()
    if manifest is not None:
        manifest_resolution_keys = manifest_resolutions(manifest, reasons)
    if intrinsics is not None:
        reasons.extend(validate_map_intrinsics(intrinsics, manifest_resolution_keys))
    return reasons


def validate_selection_motion_metrics(name: str, metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    selected = numeric_metric(metrics, "selected", "selected_images", "quality_selected", "frames")
    total = numeric_metric(metrics, "total", "total_frames")
    selected_ratio = numeric_metric(metrics, "selected_ratio")
    if selected_ratio is None and selected is not None and total:
        selected_ratio = selected / total
    parallax_ratio = numeric_metric(metrics, "parallax_or_seed_ratio", "output_parallax_or_seed_ratio")
    hover_ratio = numeric_metric(metrics, "hover_ratio")
    bridge_frames = numeric_metric(metrics, "bridge_frames")
    groups = metrics.get("groups")

    if selected is None:
        reasons.append(f"{name} selected missing")
    elif int(selected) < FRAME_MIN_SELECTED:
        reasons.append(f"{name} selected={int(selected)} < {FRAME_MIN_SELECTED}")

    if selected_ratio is None:
        reasons.append(f"{name} selected_ratio missing")
    elif selected_ratio < FRAME_MIN_SELECTED_RATIO:
        reasons.append(f"{name} selected_ratio={selected_ratio:.3f} < {FRAME_MIN_SELECTED_RATIO:.3f}")

    if parallax_ratio is None:
        reasons.append(f"{name} parallax_or_seed_ratio missing")
    elif parallax_ratio < FRAME_MIN_PARALLAX_OR_SEED_RATIO:
        reasons.append(
            f"{name} parallax_or_seed_ratio={parallax_ratio:.3f} < {FRAME_MIN_PARALLAX_OR_SEED_RATIO:.3f}"
        )

    if hover_ratio is None:
        reasons.append(f"{name} hover_ratio missing")
    elif hover_ratio > FRAME_MAX_HOVER_RATIO:
        reasons.append(f"{name} hover_ratio={hover_ratio:.3f} > {FRAME_MAX_HOVER_RATIO:.3f}")

    if isinstance(groups, dict) and groups:
        low_groups = {
            str(group): int(count)
            for group, count in groups.items()
            if int(count or 0) < FRAME_MIN_GROUP_IMAGES
        }
        if low_groups:
            formatted = ", ".join(f"{group}={count}" for group, count in sorted(low_groups.items()))
            reasons.append(f"{name} groups below {FRAME_MIN_GROUP_IMAGES}: {formatted}")
        if len(groups) > 1:
            if bridge_frames is None:
                reasons.append(f"{name} bridge_frames missing for multi-group selection")
            elif int(bridge_frames) <= 0:
                reasons.append(f"{name} bridge_frames=0 for multi-group selection")
    else:
        reasons.append(f"{name} groups missing")
    return reasons


def validate_extract_motion_metrics(name: str, metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    frames = numeric_metric(metrics, "frames", "selected", "selected_images")
    motion_gate = metrics.get("motion_gate")
    if frames is None:
        reasons.append(f"{name} frames missing")
    elif int(frames) < FRAME_MIN_SELECTED:
        reasons.append(f"{name} frames={int(frames)} < {FRAME_MIN_SELECTED}")

    if not isinstance(motion_gate, dict) or not motion_gate:
        reasons.append(f"{name} motion_gate missing")
        return reasons

    total_kept = 0
    for group, item in motion_gate.items():
        if not isinstance(item, dict):
            reasons.append(f"{name} {group}: motion_gate item invalid")
            continue
        total = numeric_metric(item, "total_before", "total")
        kept = numeric_metric(item, "kept", "selected")
        if total is None:
            reasons.append(f"{name} {group}: total_before missing")
        if kept is None:
            reasons.append(f"{name} {group}: kept missing")
            continue
        total_kept += int(kept)
        if int(kept) < FRAME_MIN_GROUP_IMAGES:
            reasons.append(f"{name} {group}: kept={int(kept)} < {FRAME_MIN_GROUP_IMAGES}")
        if total:
            keep_ratio = kept / total
            if keep_ratio < FRAME_MIN_EXTRACT_KEEP_RATIO:
                reasons.append(f"{name} {group}: keep_ratio={keep_ratio:.3f} < {FRAME_MIN_EXTRACT_KEEP_RATIO:.3f}")
        classes = item.get("motion_classes")
        if isinstance(classes, dict) and classes:
            parallax = float(classes.get("parallax", 0) or 0) + float(classes.get("seed", 0) or 0)
            hover = float(classes.get("hover", 0) or 0)
            denom = total or sum(float(value or 0) for value in classes.values())
            if denom:
                parallax_ratio = parallax / denom
                hover_ratio = hover / denom
                if parallax_ratio < FRAME_MIN_EXTRACT_PARALLAX_RATIO:
                    reasons.append(
                        f"{name} {group}: parallax_or_seed_ratio={parallax_ratio:.3f} < "
                        f"{FRAME_MIN_EXTRACT_PARALLAX_RATIO:.3f}"
                    )
                if hover_ratio > 0.50:
                    reasons.append(f"{name} {group}: hover_ratio={hover_ratio:.3f} > 0.500")
    if frames is not None and total_kept and int(frames) != total_kept:
        reasons.append(f"{name} frames={int(frames)} != summed kept={total_kept}")
    return reasons


def frame_count_from_view_graph(metrics: dict[str, Any]) -> float | None:
    frame_count = numeric_metric(metrics, "total_frames", "frames", "image_count", "num_images")
    if frame_count is not None:
        return frame_count
    component_sizes = metrics.get("component_sizes")
    if isinstance(component_sizes, list) and component_sizes:
        try:
            return float(sum(int(value) for value in component_sizes))
        except (TypeError, ValueError):
            return None
    return numeric_metric(metrics, "largest_component")


def cross_edge_count(metrics: dict[str, Any]) -> float | None:
    direct = numeric_metric(
        metrics,
        "cross_sequence_pairs",
        "cross_video_pairs",
        "cross_sequence",
        "cross_video",
    )
    if direct is not None:
        return direct
    relations = metrics.get("relations")
    if isinstance(relations, dict):
        total = 0.0
        seen = False
        for key, value in relations.items():
            if "cross" in str(key):
                try:
                    total += float(value)
                    seen = True
                except (TypeError, ValueError):
                    continue
        if seen:
            return total
    pair_kinds = metrics.get("pair_kinds")
    if isinstance(pair_kinds, dict):
        total = 0.0
        seen = False
        for key, value in pair_kinds.items():
            if "cross" in str(key):
                try:
                    total += float(value)
                    seen = True
                except (TypeError, ValueError):
                    continue
        if seen:
            return total
    return None


def validate_view_graph_quality(evidence: list[dict[str, Any]]) -> list[str]:
    graph_gates = selected_view_graph_gates(evidence)
    if not graph_gates:
        return []

    reasons: list[str] = []
    for item in graph_gates:
        name = str(item.get("name") or "view_graph")
        metrics = item.get("metrics") or {}
        pairs = numeric_metric(metrics, "pairs", "pair_count", "num_pairs")
        frame_count = frame_count_from_view_graph(metrics)
        connected_components = numeric_metric(metrics, "connected_components", "num_connected_components")
        largest_ratio = numeric_metric(metrics, "largest_component_ratio")
        largest_component = numeric_metric(metrics, "largest_component")
        if largest_ratio is None and largest_component is not None and frame_count:
            largest_ratio = largest_component / frame_count
        isolated_images = numeric_metric(metrics, "isolated_images")
        cross_edges = cross_edge_count(metrics)

        if pairs is None:
            reasons.append(f"{name} pairs missing")
        else:
            min_pairs = float(VIEW_GRAPH_MIN_PAIRS)
            if frame_count:
                min_pairs = max(min_pairs, frame_count * VIEW_GRAPH_MIN_PAIRS_PER_FRAME)
            if pairs < min_pairs:
                reasons.append(f"{name} pairs={int(pairs)} < {int(min_pairs)}")
            if frame_count:
                pairs_per_frame = pairs / frame_count
                if pairs_per_frame < VIEW_GRAPH_MIN_PAIRS_PER_FRAME:
                    reasons.append(
                        f"{name} pairs_per_frame={pairs_per_frame:.2f} < {VIEW_GRAPH_MIN_PAIRS_PER_FRAME:.2f}"
                    )

        if connected_components is None:
            reasons.append(f"{name} connected_components missing")
        elif int(connected_components) > 1:
            reasons.append(f"{name} connected_components={int(connected_components)} > 1")

        if largest_ratio is None:
            reasons.append(f"{name} largest_component_ratio missing")
        elif largest_ratio < VIEW_GRAPH_MIN_LARGEST_COMPONENT_RATIO:
            reasons.append(
                f"{name} largest_component_ratio={largest_ratio:.3f} < {VIEW_GRAPH_MIN_LARGEST_COMPONENT_RATIO:.3f}"
            )

        if isolated_images is not None and int(isolated_images) > 0:
            reasons.append(f"{name} isolated_images={int(isolated_images)} > 0")

        if cross_edges is None:
            reasons.append(f"{name} cross_video_or_sequence_pairs missing")
        elif cross_edges <= 0:
            reasons.append(f"{name} cross_video_or_sequence_pairs=0")
    return reasons


def selected_database_gates(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_gates = [
        item for item in evidence
        if item.get("ok") and item.get("name") in set(DATABASE_GATE_PRIORITY)
    ]
    for name in DATABASE_GATE_PRIORITY:
        selected = [item for item in ok_gates if item.get("name") == name]
        if selected:
            return selected
    return []


def collect_database_paths_from_metrics(metrics: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for key in ("database_path", "db_path", "database", "db"):
        value = metrics.get(key)
        if isinstance(value, str) and value.endswith(".db"):
            paths.append(Path(value))
    outputs = metrics.get("outputs")
    if isinstance(outputs, dict):
        paths.extend(collect_database_paths_from_metrics(outputs))
    return paths


def database_path_candidates(run_dir: Path, evidence: list[dict[str, Any]]) -> list[Path]:
    candidates: list[Path] = []
    for item in evidence:
        metrics = item.get("metrics") or {}
        for path in collect_database_paths_from_metrics(metrics):
            candidates.append(path if path.is_absolute() else run_dir / path)
    candidates.extend(
        run_dir / rel for rel in (
            "glomap/0/database.db",
            "glomap/database_merged.db",
            "glomap/database_tracks.db",
            "glomap/database_sift.db",
            "work/mvroma/database_mvroma_forced.db",
            "work/tmp/database_mvroma_forced_tmp.db",
            "work/database.db",
            "database.db",
        )
    )
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def sqlite_table_names(cur: sqlite3.Cursor) -> set[str]:
    return {
        str(row[0])
        for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def scalar(cur: sqlite3.Cursor, sql: str) -> Any:
    return cur.execute(sql).fetchone()[0]


def min_required_pairs_for_images(images: int) -> int:
    return int(max(VIEW_GRAPH_MIN_PAIRS, images * VIEW_GRAPH_MIN_PAIRS_PER_FRAME))


def validate_database_file(db_path: Path, manifest_total_frames: int | None) -> tuple[list[str], dict[str, Any]]:
    metrics: dict[str, Any] = {
        "path": str(db_path),
        "bytes": db_path.stat().st_size if db_path.exists() else 0,
    }
    reasons: list[str] = []
    if not db_path.exists() or db_path.stat().st_size <= 0:
        return [f"missing database file: {db_path}"], metrics
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return [f"cannot open database {db_path}: {exc}"], metrics
    try:
        cur = con.cursor()
        tables = sqlite_table_names(cur)
        metrics["tables"] = sorted(tables)
        required_tables = {"cameras", "images", "keypoints", "matches", "two_view_geometries"}
        missing = sorted(required_tables - tables)
        if missing:
            return [f"missing required database tables: {', '.join(missing)}"], metrics

        cameras = int(scalar(cur, "SELECT COUNT(*) FROM cameras") or 0)
        images = int(scalar(cur, "SELECT COUNT(*) FROM images") or 0)
        keypoint_rows = int(scalar(cur, "SELECT COUNT(*) FROM keypoints") or 0)
        images_with_keypoints = int(
            scalar(
                cur,
                "SELECT COUNT(*) FROM images "
                "JOIN keypoints ON images.image_id = keypoints.image_id "
                "WHERE COALESCE(keypoints.rows, 0) > 0",
            )
            or 0
        )
        min_keypoints = scalar(cur, "SELECT MIN(rows) FROM keypoints")
        avg_keypoints = scalar(cur, "SELECT AVG(rows) FROM keypoints")
        matches = int(scalar(cur, "SELECT COUNT(*) FROM matches") or 0)
        nonzero_matches = int(scalar(cur, "SELECT COUNT(*) FROM matches WHERE COALESCE(rows, 0) > 0") or 0)
        two_view = int(scalar(cur, "SELECT COUNT(*) FROM two_view_geometries") or 0)
        nonzero_two_view = int(
            scalar(cur, "SELECT COUNT(*) FROM two_view_geometries WHERE COALESCE(rows, 0) > 0") or 0
        )
        avg_two_view_inliers = scalar(cur, "SELECT AVG(rows) FROM two_view_geometries")
        bad_camera_dims = int(
            scalar(cur, "SELECT COUNT(*) FROM cameras WHERE COALESCE(width, 0) <= 0 OR COALESCE(height, 0) <= 0")
            or 0
        )
        missing_camera_refs = int(
            scalar(
                cur,
                "SELECT COUNT(*) FROM images "
                "LEFT JOIN cameras ON images.camera_id = cameras.camera_id "
                "WHERE cameras.camera_id IS NULL",
            )
            or 0
        )
    except sqlite3.Error as exc:
        reasons.append(f"cannot inspect database {db_path}: {exc}")
        return reasons, metrics
    finally:
        con.close()

    min_pairs = min_required_pairs_for_images(images)
    nonzero_match_ratio = (nonzero_matches / matches) if matches else None
    nonzero_two_view_ratio = (nonzero_two_view / two_view) if two_view else None
    metrics.update({
        "cameras": cameras,
        "images": images,
        "manifest_total_frames": manifest_total_frames,
        "keypoint_rows": keypoint_rows,
        "images_with_keypoints": images_with_keypoints,
        "min_keypoints_per_image": min_keypoints,
        "avg_keypoints_per_image": avg_keypoints,
        "matches": matches,
        "nonzero_matches": nonzero_matches,
        "nonzero_match_ratio": nonzero_match_ratio,
        "two_view_geometries": two_view,
        "nonzero_two_view_geometries": nonzero_two_view,
        "nonzero_two_view_ratio": nonzero_two_view_ratio,
        "avg_two_view_inliers": avg_two_view_inliers,
        "min_required_pairs": min_pairs,
        "cameras_with_invalid_size": bad_camera_dims,
        "images_with_missing_camera": missing_camera_refs,
    })

    if cameras <= 0:
        reasons.append("cameras=0")
    if bad_camera_dims:
        reasons.append(f"cameras_with_invalid_size={bad_camera_dims}")
    if images < FRAME_MIN_SELECTED:
        reasons.append(f"images={images} < {FRAME_MIN_SELECTED}")
    if manifest_total_frames is not None and images != manifest_total_frames:
        reasons.append(f"images={images} != frame_manifest.total_frames={manifest_total_frames}")
    if missing_camera_refs:
        reasons.append(f"images_with_missing_camera={missing_camera_refs}")
    if keypoint_rows < images:
        reasons.append(f"keypoint_rows={keypoint_rows} < images={images}")
    if images_with_keypoints < images:
        reasons.append(f"images_with_keypoints={images_with_keypoints} < images={images}")
    if min_keypoints is None:
        reasons.append("keypoints rows missing")
    elif float(min_keypoints) < DATABASE_MIN_KEYPOINTS_PER_IMAGE:
        reasons.append(
            f"min_keypoints_per_image={float(min_keypoints):.0f} < {DATABASE_MIN_KEYPOINTS_PER_IMAGE}"
        )
    if avg_keypoints is None:
        reasons.append("avg_keypoints_per_image missing")
    elif float(avg_keypoints) < DATABASE_MIN_AVG_KEYPOINTS_PER_IMAGE:
        reasons.append(
            f"avg_keypoints_per_image={float(avg_keypoints):.1f} < {DATABASE_MIN_AVG_KEYPOINTS_PER_IMAGE:.1f}"
        )

    if matches < min_pairs:
        reasons.append(f"matches pairs={matches} < {min_pairs}")
    if nonzero_match_ratio is not None and nonzero_match_ratio < DATABASE_MIN_NONZERO_GEOMETRY_RATIO:
        reasons.append(
            f"matches nonzero_ratio={nonzero_match_ratio:.3f} < {DATABASE_MIN_NONZERO_GEOMETRY_RATIO:.3f}"
        )
    if two_view < min_pairs:
        reasons.append(f"two_view_geometries pairs={two_view} < {min_pairs}")
    if nonzero_two_view_ratio is not None:
        if nonzero_two_view_ratio < DATABASE_MIN_NONZERO_GEOMETRY_RATIO:
            reasons.append(
                f"two_view_geometries nonzero_ratio={nonzero_two_view_ratio:.3f} < "
                f"{DATABASE_MIN_NONZERO_GEOMETRY_RATIO:.3f}"
            )
    if avg_two_view_inliers is None:
        reasons.append("avg_two_view_inliers missing")
    elif float(avg_two_view_inliers) < DATABASE_MIN_AVG_TWO_VIEW_INLIERS:
        reasons.append(
            f"avg_two_view_inliers={float(avg_two_view_inliers):.1f} < {DATABASE_MIN_AVG_TWO_VIEW_INLIERS:.1f}"
        )
    return reasons, metrics


def manifest_total_frames(run_dir: Path) -> int | None:
    manifest, reasons = load_json_for_quality(run_dir / "frame_manifest.json", "frame_manifest.json")
    if reasons or manifest is None:
        return None
    total = numeric_metric(manifest, "total_frames")
    return int(total) if total is not None else None


def validate_database_quality(run_dir: Path, evidence: list[dict[str, Any]]) -> list[str]:
    database_gates = selected_database_gates(evidence)
    if not database_gates:
        return []
    candidates = database_path_candidates(run_dir, evidence)
    db_path = next((path for path in candidates if path.exists() and path.stat().st_size > 0), None)
    if db_path is None:
        formatted = ", ".join(str(path) for path in candidates)
        database_gates[0].setdefault("metrics", {})["database_quality"] = {
            "candidate_paths": [str(path) for path in candidates],
        }
        return [f"missing COLMAP database file; checked: {formatted}"]
    reasons, metrics = validate_database_file(db_path, manifest_total_frames(run_dir))
    metrics["candidate_paths"] = [str(path) for path in candidates]
    database_gates[0].setdefault("metrics", {})["database_quality"] = metrics
    return reasons


def selected_sfm_model_gates(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_gates = [
        item for item in evidence
        if item.get("ok") and item.get("name") in set(SFM_MODEL_GATE_PRIORITY)
    ]
    for name in SFM_MODEL_GATE_PRIORITY:
        selected = [item for item in ok_gates if item.get("name") == name]
        if selected:
            return selected
    return []


def validate_sfm_reconstruction_quality(evidence: list[dict[str, Any]]) -> list[str]:
    model_gates = selected_sfm_model_gates(evidence)
    if not model_gates:
        return []

    reasons: list[str] = []
    for item in model_gates:
        name = str(item.get("name") or "sfm_model")
        metrics = item.get("metrics") or {}
        registered = numeric_metric(metrics, "registered_images", "num_registered_images", "num_reg_images")
        registered_ratio = numeric_metric(metrics, "registered_ratio")
        selected_images = numeric_metric(metrics, "selected_images", "total_frames", "image_count")
        if registered_ratio is None and registered is not None and selected_images:
            registered_ratio = registered / selected_images
        points3d = numeric_metric(metrics, "points3D", "points3d", "num_points3D", "num_points3d")
        points_per_registered = numeric_metric(
            metrics,
            "points_per_registered",
            "points_per_registered_image",
            "points_per_registered_images",
        )
        if points_per_registered is None and points3d is not None and registered:
            points_per_registered = points3d / registered
        mean_reprojection_error = numeric_metric(
            metrics,
            "mean_reprojection_error",
            "pycolmap_mean_reprojection_error",
            "mean_reprojection_error_px",
        )

        if registered is None:
            reasons.append(f"{name} registered_images missing")
        elif int(registered) < SFM_MIN_REGISTERED_IMAGES:
            reasons.append(f"{name} registered_images={int(registered)} < {SFM_MIN_REGISTERED_IMAGES}")

        if registered_ratio is None:
            reasons.append(f"{name} registered_ratio missing")
        elif registered_ratio < SFM_MIN_REGISTERED_RATIO:
            reasons.append(f"{name} registered_ratio={registered_ratio:.3f} < {SFM_MIN_REGISTERED_RATIO:.3f}")

        if points3d is None:
            reasons.append(f"{name} points3D missing")
        elif int(points3d) < SFM_MIN_POINTS3D:
            reasons.append(f"{name} points3D={int(points3d)} < {SFM_MIN_POINTS3D}")

        if points_per_registered is None:
            reasons.append(f"{name} points_per_registered missing")
        elif points_per_registered < SFM_MIN_POINTS_PER_REGISTERED:
            reasons.append(
                f"{name} points_per_registered={points_per_registered:.1f} < {SFM_MIN_POINTS_PER_REGISTERED:.1f}"
            )

        if mean_reprojection_error is None:
            reasons.append(f"{name} mean_reprojection_error missing")
        elif mean_reprojection_error > SFM_MAX_MEAN_REPROJECTION_ERROR:
            reasons.append(
                f"{name} mean_reprojection_error={mean_reprojection_error:.3f} > "
                f"{SFM_MAX_MEAN_REPROJECTION_ERROR:.3f}"
            )
    return reasons


def selected_point_cloud_gates(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_gates = [
        item for item in evidence
        if item.get("ok") and item.get("name") in {"color", "rgb_ply"}
    ]
    for name in ("color", "rgb_ply"):
        selected = [item for item in ok_gates if item.get("name") == name]
        if selected:
            return selected
    return []


def collect_ply_paths_from_metrics(metrics: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for key in ("ply_path", "rgb_ply_path", "ply", "rgb_ply", "output_ply", "path"):
        value = metrics.get(key)
        if isinstance(value, str) and value.endswith(".ply"):
            paths.append(Path(value))
    outputs = metrics.get("outputs")
    if isinstance(outputs, dict):
        paths.extend(collect_ply_paths_from_metrics(outputs))
    return paths


def point_cloud_path_candidates(run_dir: Path, evidence: list[dict[str, Any]]) -> list[Path]:
    candidates: list[Path] = []
    for item in evidence:
        metrics = item.get("metrics") or {}
        for path in collect_ply_paths_from_metrics(metrics):
            candidates.append(path if path.is_absolute() else run_dir / path)
    candidates.extend(
        run_dir / rel for rel in (
            "deploy/map_rgb.ply",
            "deploy/map_rgb_dense_retriangulated.ply",
            "map_rgb.ply",
        )
    )
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def sparse_model_points3d(run_dir: Path) -> int | None:
    for alias in SFM_MODEL_GATE_PRIORITY:
        path = gate_path(run_dir, alias)
        if not path.exists():
            continue
        gate = normalize_gate(path, alias)
        if not gate.get("ok"):
            continue
        points = numeric_metric(gate.get("metrics") or {}, "points3D", "points3d", "num_points3D", "num_points3d")
        if points is not None:
            return int(points)
    return None


def parse_ply_header(path: Path) -> tuple[dict[str, Any], list[str]]:
    metrics: dict[str, Any] = {"path": str(path), "bytes": path.stat().st_size if path.exists() else 0}
    reasons: list[str] = []
    if not path.exists() or path.stat().st_size <= 0:
        return metrics, [f"missing PLY file: {path}"]
    try:
        with path.open("rb") as f:
            data = f.read(PLY_MAX_HEADER_BYTES)
    except OSError as exc:
        return metrics, [f"cannot read PLY header: {exc}"]
    marker = b"end_header\n"
    idx = data.find(marker)
    marker_len = len(marker)
    if idx < 0:
        marker = b"end_header\r\n"
        idx = data.find(marker)
        marker_len = len(marker)
    if idx < 0:
        return metrics, [f"PLY end_header missing within {PLY_MAX_HEADER_BYTES} bytes"]
    header_bytes = idx + marker_len
    try:
        header = data[:header_bytes].decode("ascii")
    except UnicodeDecodeError as exc:
        return metrics, [f"PLY header is not ASCII: {exc}"]

    lines = [line.strip() for line in header.splitlines() if line.strip()]
    metrics["header_bytes"] = header_bytes
    if not lines or lines[0] != "ply":
        reasons.append("PLY magic header missing")

    format_name = ""
    vertex_count: int | None = None
    vertex_properties: list[tuple[str, str]] = []
    current_element = ""
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            format_name = parts[1]
        elif parts[0] == "element" and len(parts) >= 3:
            current_element = parts[1]
            if current_element == "vertex":
                try:
                    vertex_count = int(parts[2])
                except ValueError:
                    reasons.append(f"PLY vertex count invalid: {parts[2]}")
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) >= 3 and parts[1] != "list":
                vertex_properties.append((parts[1], parts[2]))
            elif len(parts) >= 5 and parts[1] == "list":
                reasons.append(f"PLY vertex list property unsupported for quality gate: {' '.join(parts[2:])}")

    property_names = [name for _type, name in vertex_properties]
    has_xyz = all(name in property_names for name in ("x", "y", "z"))
    has_rgb = all(name in property_names for name in ("red", "green", "blue"))
    metrics.update({
        "format": format_name,
        "vertices": vertex_count,
        "vertex_properties": property_names,
        "has_xyz": has_xyz,
        "has_rgb": has_rgb,
    })
    if format_name not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        reasons.append(f"PLY format unsupported or missing: {format_name or '<missing>'}")
    if vertex_count is None:
        reasons.append("PLY vertex element missing")
    elif vertex_count < POINT_CLOUD_MIN_VERTICES:
        reasons.append(f"PLY vertices={vertex_count} < {POINT_CLOUD_MIN_VERTICES}")
    if not has_xyz:
        reasons.append("PLY missing XYZ vertex properties")
    if not has_rgb:
        reasons.append("PLY missing RGB vertex properties")
    if metrics["bytes"] < POINT_CLOUD_MIN_BYTES:
        reasons.append(f"PLY bytes={metrics['bytes']} < {POINT_CLOUD_MIN_BYTES}")

    if format_name.startswith("binary") and vertex_count is not None:
        stride = 0
        for prop_type, prop_name in vertex_properties:
            size = PLY_SCALAR_TYPE_BYTES.get(prop_type)
            if size is None:
                reasons.append(f"PLY vertex property {prop_name} has unsupported type {prop_type}")
                continue
            stride += size
        metrics["binary_vertex_stride"] = stride
        expected_bytes = header_bytes + vertex_count * stride
        metrics["min_expected_binary_bytes"] = expected_bytes
        if stride > 0 and metrics["bytes"] < expected_bytes:
            reasons.append(f"PLY bytes={metrics['bytes']} < expected binary bytes {expected_bytes}")
    return metrics, reasons


def validate_point_cloud_quality(run_dir: Path, evidence: list[dict[str, Any]]) -> list[str]:
    point_cloud_gates = selected_point_cloud_gates(evidence)
    if not point_cloud_gates:
        return []
    candidates = point_cloud_path_candidates(run_dir, evidence)
    ply_path = next((path for path in candidates if path.exists() and path.stat().st_size > 0), None)
    if ply_path is None:
        formatted = ", ".join(str(path) for path in candidates)
        point_cloud_gates[0].setdefault("metrics", {})["point_cloud_quality"] = {
            "candidate_paths": [str(path) for path in candidates],
        }
        return [f"missing PLY file; checked: {formatted}"]

    metrics, reasons = parse_ply_header(ply_path)
    model_points = sparse_model_points3d(run_dir)
    metrics["sparse_points3D"] = model_points
    vertices = metrics.get("vertices")
    if isinstance(vertices, int) and model_points:
        ratio = vertices / model_points
        metrics["vertex_ratio_vs_points3D"] = ratio
        if ratio < POINT_CLOUD_MIN_VERTEX_RATIO_VS_POINTS3D:
            reasons.append(
                f"PLY vertex_ratio_vs_points3D={ratio:.3f} < {POINT_CLOUD_MIN_VERTEX_RATIO_VS_POINTS3D:.3f}"
            )
    elif model_points is None:
        reasons.append("sparse model points3D missing for PLY ratio check")

    gate_metrics = point_cloud_gates[0].get("metrics") or {}
    gate_vertices = numeric_metric(gate_metrics, "ply_vertices", "vertices")
    if gate_vertices is not None and isinstance(vertices, int) and int(gate_vertices) != vertices:
        reasons.append(f"PLY header vertices={vertices} != gate ply_vertices={int(gate_vertices)}")
    gate_ratio = numeric_metric(gate_metrics, "ply_vertex_ratio_vs_points3D")
    if gate_ratio is not None and gate_ratio < POINT_CLOUD_MIN_VERTEX_RATIO_VS_POINTS3D:
        reasons.append(
            f"gate ply_vertex_ratio_vs_points3D={gate_ratio:.3f} < {POINT_CLOUD_MIN_VERTEX_RATIO_VS_POINTS3D:.3f}"
        )

    metrics["candidate_paths"] = [str(path) for path in candidates]
    point_cloud_gates[0].setdefault("metrics", {})["point_cloud_quality"] = metrics
    return reasons


def selected_report_package_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item for item in evidence
        if item.get("ok")
        and (
            item.get("name") == "report"
            or item.get("kind") == "artifact"
            or str(item.get("name") or "").endswith((".md", ".json"))
        )
    ]


def resolve_report_path(run_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def validate_output_manifest_paths(run_dir: Path, outputs: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    missing_keys = [key for key in REPORT_REQUIRED_OUTPUT_KEYS if not outputs.get(key)]
    if missing_keys:
        reasons.append(f"build_report outputs missing: {', '.join(missing_keys)}")
    if not any(outputs.get(key) for key in REPORT_BUNDLE_OUTPUT_KEYS):
        reasons.append(
            "build_report outputs missing localization bundle key: "
            + " or ".join(REPORT_BUNDLE_OUTPUT_KEYS)
        )

    checked: dict[str, str] = {}
    missing_paths: list[str] = []
    for key in (*REPORT_REQUIRED_OUTPUT_KEYS, *REPORT_BUNDLE_OUTPUT_KEYS):
        path = resolve_report_path(run_dir, outputs.get(key))
        if path is None:
            continue
        checked[key] = str(path)
        if not path.exists():
            missing_paths.append(f"{key}={path}")
        elif path.is_file() and path.stat().st_size <= 0:
            missing_paths.append(f"{key}={path} is empty")
    metrics["checked_output_paths"] = checked
    if missing_paths:
        reasons.append("build_report output paths missing or empty: " + "; ".join(missing_paths))
    return reasons


def validate_stage_times_payload(stage_times: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    stages = stage_times.get("stages")
    if not isinstance(stages, list) or not stages:
        return ["stage_times stages missing"]
    metrics["stage_times_count"] = len(stages)
    total_seconds = numeric_metric(stage_times, "total_seconds")
    metrics["stage_times_total_seconds"] = total_seconds
    if total_seconds is None or total_seconds <= 0:
        reasons.append("stage_times total_seconds missing or non-positive")

    latest_by_stage: dict[str, str] = {}
    failed_attempts: list[str] = []
    for item in stages:
        if not isinstance(item, dict):
            reasons.append("stage_times contains non-object item")
            continue
        stage = str(item.get("stage") or "")
        status = str(item.get("status") or "").lower()
        if not stage:
            reasons.append("stage_times item missing stage")
            continue
        latest_by_stage[stage] = status
        if status in {"failed", "fail", "error"}:
            failed_attempts.append(stage)
    metrics["stage_times_latest_status"] = latest_by_stage
    metrics["stage_times_failed_attempts"] = failed_attempts
    if latest_by_stage.get("report") not in {"success", "passed", "pass"}:
        reasons.append(f"stage_times latest report status={latest_by_stage.get('report') or '<missing>'}")
    return reasons


def validate_report_package_quality(run_dir: Path, evidence: list[dict[str, Any]]) -> list[str]:
    report_evidence = selected_report_package_evidence(evidence)
    if not report_evidence:
        return []

    metrics: dict[str, Any] = {}
    reasons: list[str] = []
    build_report, build_report_reasons = load_json_for_quality(run_dir / "build_report.json", "build_report.json")
    build_config, build_config_reasons = load_json_for_quality(run_dir / "build_config.json", "build_config.json")
    stage_times, stage_times_reasons = load_json_for_quality(run_dir / "stage_times.json", "stage_times.json")
    reasons.extend(build_report_reasons)
    reasons.extend(build_config_reasons)
    reasons.extend(stage_times_reasons)

    md_candidates = [run_dir / "BUILD_LOCALIZABLE_MAP_REPORT.md", run_dir / "BUILD_GLUEMAP_REPORT.md"]
    md_path = next((path for path in md_candidates if path.exists() and path.stat().st_size > 0), None)
    metrics["operator_report"] = str(md_path) if md_path else None
    if md_path is None:
        reasons.append("missing operator Markdown report")
    else:
        text = md_path.read_text(encoding="utf-8", errors="replace")
        metrics["operator_report_bytes"] = md_path.stat().st_size
        if "Output" not in text and "輸出" not in text:
            reasons.append("operator Markdown report missing outputs section")

    if build_report is not None:
        outputs = build_report.get("outputs")
        if not isinstance(outputs, dict) or not outputs:
            reasons.append("build_report outputs missing")
        else:
            metrics["build_report_output_keys"] = sorted(outputs)
            reasons.extend(validate_output_manifest_paths(run_dir, outputs, metrics))
        parameters = build_report.get("parameters")
        if not isinstance(parameters, dict) or not parameters:
            reasons.append("build_report parameters missing")
        else:
            metrics["build_report_parameter_keys"] = sorted(parameters)

    if build_config is not None:
        if not build_config:
            reasons.append("build_config is empty")
        metrics["build_config_keys"] = sorted(build_config) if isinstance(build_config, dict) else []

    if stage_times is not None:
        reasons.extend(validate_stage_times_payload(stage_times, metrics))

    gate_files = sorted((run_dir / "gates").glob("*.json"))
    metrics["gate_json_count"] = len(gate_files)
    metrics["gate_json_files"] = [path.name for path in gate_files]
    if len(gate_files) < REPORT_MIN_GATE_JSONS:
        reasons.append(f"gate JSON count={len(gate_files)} < {REPORT_MIN_GATE_JSONS}")

    report_evidence[0].setdefault("metrics", {})["report_package_quality"] = metrics
    return reasons


def skipped_external_stage(key: str, title: str, purpose: str, operator_action: str) -> dict[str, Any]:
    return {
        "key": key,
        "title": title,
        "purpose": purpose,
        "required": False,
        "status": "SKIP",
        "ok": True,
        "reasons": [],
        "operator_action": operator_action,
        "evidence": [],
    }


def external_stage_result(
    key: str,
    title: str,
    purpose: str,
    required: bool,
    reasons: list[str],
    evidence: list[dict[str, Any]],
    operator_action: str,
) -> dict[str, Any]:
    if not required and not evidence:
        return skipped_external_stage(key, title, purpose, operator_action)
    return {
        "key": key,
        "title": title,
        "purpose": purpose,
        "required": required,
        "status": "PASS" if not reasons else "FAIL",
        "ok": not reasons,
        "reasons": reasons,
        "operator_action": operator_action,
        "evidence": evidence,
    }


def evaluate_compare_json(path: Path, min_success: float, max_ok_to_fail: int,
                          max_final_fail_run: int) -> tuple[dict[str, Any], list[str]]:
    data = read_json(path)
    rows = data.get("rows", [])
    metrics = {
        "path": str(path),
        "rows": len(rows),
        "sets": [],
    }
    reasons: list[str] = []
    if not rows:
        reasons.append(f"{path}: eval JSON has no rows")
    for row in rows:
        name = str(row.get("set", ""))
        n = int(row.get("n", 0))
        base = float(row.get("base_success", 0.0))
        final = float(row.get("final_success", 0.0))
        ok_to_fail = int(row.get("ok_to_fail", 0))
        final_fail = int(row.get("final_max_fail_run", 0))
        base_fail = int(row.get("base_max_fail_run", max_final_fail_run))
        regression_ok = ok_to_fail <= max_ok_to_fail and final_fail <= max_final_fail_run
        absolute_ok = final >= min_success
        baseline_improved = (
            base < min_success
            and final >= base
            and final_fail <= base_fail
            and regression_ok
        )
        ok = regression_ok and (absolute_ok or baseline_improved)
        result = "PASS" if absolute_ok else "PASS_BASELINE_IMPROVED" if baseline_improved else "FAIL"
        if n <= 0:
            ok = False
            result = "FAIL"
        if not ok:
            reasons.append(
                f"{name or path.name}: final_success={final:.3f}, ok_to_fail={ok_to_fail}, final_fail_run={final_fail}"
            )
        metrics["sets"].append({
            "set": name,
            "n": n,
            "base_success": base,
            "final_success": final,
            "ok_to_fail": ok_to_fail,
            "final_max_fail_run": final_fail,
            "result": result,
        })
    return metrics, reasons


def percentile(values: list[float], pct: float) -> float | None:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * (float(pct) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def max_consecutive_failures(rows: list[dict[str, Any]]) -> int | None:
    if not rows:
        return None
    current = 0
    longest = 0
    for row in rows:
        if bool(row.get("success")):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def evaluate_holdout_localization(
    paths: list[Path],
    required: bool,
    min_success: float,
    max_ok_to_fail: int,
    max_final_fail_run: int,
) -> dict[str, Any]:
    title = "Holdout localization validation"
    purpose = "Prove the candidate bundle localizes held-out videos, not just build-time reference frames."
    action = "Run localize_pipeline.py --mode compare on held-out videos and attach the JSON result."
    if not paths and not required:
        return skipped_external_stage("holdout_localization", title, purpose, action)
    evidence: list[dict[str, Any]] = []
    reasons: list[str] = []
    if not paths:
        reasons.append("missing localization validation JSON")
    for path in paths:
        exists = path.exists() and path.stat().st_size > 0
        item = {"name": path.name, "path": str(path), "ok": False, "kind": "localization_compare", "metrics": {}}
        if not exists:
            reasons.append(f"missing localization validation JSON: {path}")
        else:
            try:
                metrics, row_reasons = evaluate_compare_json(path, min_success, max_ok_to_fail, max_final_fail_run)
                item["metrics"] = metrics
                item["ok"] = not row_reasons
                reasons.extend(row_reasons)
            except Exception as exc:
                reasons.append(f"{path}: cannot read localization JSON: {exc}")
        evidence.append(item)
    return external_stage_result("holdout_localization", title, purpose, required, reasons, evidence, action)


def evaluate_production_json(
    path: Path,
    min_success: float,
    max_wall_p90: float,
    min_frames: int,
    max_fail_run: int,
    min_inliers_p5: float,
) -> tuple[dict[str, Any], list[str]]:
    data = read_json(path)
    if data.get("kind") == "production_stream_preflight":
        metrics = {
            "path": str(path),
            "preflight_ok": bool(data.get("ok")),
            **(data.get("metrics") or {}),
        }
        reasons = ["production replay did not run; preflight-only artifact"]
        reasons.extend(str(reason) for reason in data.get("reasons", []))
        return metrics, reasons

    preflight_metrics, preflight_reasons = evaluate_paired_production_preflight(path, min_frames)
    summary = data.get("summary", {})
    rows = data.get("rows", []) or []
    wall_stats = summary.get("wall_ms", {}) or {}
    n = int(summary.get("n", 0))
    success = int(summary.get("success", 0))
    success_rate = float(summary.get("success_rate", 0.0))
    wall_p90 = wall_stats.get("p90")
    max_fail = max_consecutive_failures(rows)
    inliers_p5 = percentile(
        [float(row.get("inliers")) for row in rows if row.get("inliers") is not None],
        5.0,
    )
    metrics = {
        "path": str(path),
        "n": n,
        "success": success,
        "success_rate": success_rate,
        "wall_ms_p90": wall_p90,
        "max_failure_run": max_fail,
        "inliers_p5": inliers_p5,
        "preflight": preflight_metrics,
    }
    reasons: list[str] = list(preflight_reasons)
    if n <= 0:
        reasons.append("n=0; no production stream frames evaluated")
    if min_frames > 0 and n < min_frames:
        reasons.append(f"n={n} < min_frames={int(min_frames)}")
    if success < 0 or success > max(0, n):
        reasons.append(f"success={success} inconsistent with n={n}")
    if n > 0 and abs(success_rate - (success / n)) > 1e-3:
        reasons.append(f"success_rate={success_rate:.3f} inconsistent with success/n={success / n:.3f}")
    if success_rate < min_success:
        reasons.append(f"success_rate={success_rate:.3f} < {min_success:.3f}")
    if max_fail_run > 0:
        if max_fail is None:
            reasons.append("rows missing while max failure-run gate is enabled")
        elif max_fail > max_fail_run:
            reasons.append(f"max_failure_run={max_fail} > {int(max_fail_run)}")
    if min_inliers_p5 > 0:
        if inliers_p5 is None:
            reasons.append("inliers_p5 missing while inlier gate is enabled")
        elif inliers_p5 < min_inliers_p5:
            reasons.append(f"inliers_p5={inliers_p5:.1f} < {float(min_inliers_p5):.1f}")
    if max_wall_p90 > 0:
        if wall_p90 is None:
            reasons.append("wall_ms_p90 missing while latency gate is enabled")
        elif float(wall_p90) > max_wall_p90:
            reasons.append(f"wall_ms_p90={float(wall_p90):.1f} > {max_wall_p90:.1f}")
    return metrics, reasons


def evaluate_paired_production_preflight(path: Path, min_frames: int) -> tuple[dict[str, Any], list[str]]:
    preflight_path = path.with_suffix(".preflight.json")
    metrics: dict[str, Any] = {"path": str(preflight_path)}
    reasons: list[str] = []
    if not preflight_path.exists() or preflight_path.stat().st_size <= 0:
        return metrics, [f"missing paired production preflight JSON: {preflight_path}"]
    try:
        data = read_json(preflight_path)
    except Exception as exc:
        return metrics, [f"{preflight_path}: cannot read paired production preflight JSON: {exc}"]

    if data.get("kind") != "production_stream_preflight":
        reasons.append(f"paired preflight kind={data.get('kind')!r} is not production_stream_preflight")
    preflight_ok = bool(data.get("ok"))
    source_metrics = data.get("metrics") or {}
    metrics.update({
        "ok": preflight_ok,
        "direct_jpg_count": source_metrics.get("direct_jpg_count"),
        "numeric_frame_count": source_metrics.get("numeric_frame_count"),
        "max_frame_gap": source_metrics.get("max_frame_gap"),
    })
    if not preflight_ok:
        reasons.append("paired preflight ok=false")
    reasons.extend(str(reason) for reason in data.get("reasons", []))

    if metrics["direct_jpg_count"] is None:
        reasons.append("paired preflight missing direct_jpg_count")
    if metrics["numeric_frame_count"] is None:
        reasons.append("paired preflight missing numeric_frame_count")
    if metrics["max_frame_gap"] is None:
        reasons.append("paired preflight missing max_frame_gap")
    frame_count = metrics["numeric_frame_count"] if metrics["numeric_frame_count"] is not None else metrics["direct_jpg_count"]
    if frame_count is not None and min_frames > 0 and int(frame_count) < min_frames:
        reasons.append(f"paired preflight frame_count={int(frame_count)} < min_frames={int(min_frames)}")
    return metrics, reasons


def evaluate_production_replay(path: Path | None, required: bool, min_success: float,
                               max_wall_p90: float, min_frames: int,
                               max_fail_run: int, min_inliers_p5: float) -> dict[str, Any]:
    title = "Production tracker replay"
    purpose = "Verify the production state machine, cache, local/covis retrieval, and temporal behavior, not only per-frame MegaLoc top-k."
    action = "Run localize_pipeline.py --mode production-stream and attach the JSON result."
    if path is None and not required:
        return skipped_external_stage("production_replay", title, purpose, action)
    reasons: list[str] = []
    evidence: list[dict[str, Any]] = []
    if path is None:
        reasons.append("missing production replay JSON")
    else:
        item = {"name": path.name, "path": str(path), "ok": False, "kind": "production_stream", "metrics": {}}
        if not path.exists() or path.stat().st_size <= 0:
            reasons.append(f"missing production replay JSON: {path}")
        else:
            try:
                metrics, quality_reasons = evaluate_production_json(
                    path,
                    min_success,
                    max_wall_p90,
                    min_frames,
                    max_fail_run,
                    min_inliers_p5,
                )
                item["metrics"] = metrics
                item["ok"] = not quality_reasons
                reasons.extend(quality_reasons)
            except Exception as exc:
                reasons.append(f"{path}: cannot read production replay JSON: {exc}")
        evidence.append(item)
    return external_stage_result("production_replay", title, purpose, required, reasons, evidence, action)


def evaluate_package_verify(path: Path | None, required: bool) -> dict[str, Any]:
    title = "Package structure verification"
    purpose = "Ensure the handoff package has required paths, config syntax, symlinks, permissions, and runtime scaffolding."
    action = "Run tools/verify_package.py and attach package_verify.json."
    if path is None and not required:
        return skipped_external_stage("package_verify", title, purpose, action)
    reasons: list[str] = []
    evidence: list[dict[str, Any]] = []
    if path is None:
        reasons.append("missing package_verify.json")
    else:
        item = {"name": path.name, "path": str(path), "ok": False, "kind": "package_verify", "metrics": {}}
        if not path.exists() or path.stat().st_size <= 0:
            reasons.append(f"missing package_verify.json: {path}")
        else:
            try:
                data = read_json(path)
                failed = [check.get("name", "<unnamed>") for check in data.get("checks", []) if not check.get("ok")]
                if not data.get("ok", False):
                    reasons.append(f"package_verify ok=false; failed={failed}")
                item["metrics"] = {"ok": bool(data.get("ok")), "failed": failed, "checks": len(data.get("checks", []))}
                item["ok"] = bool(data.get("ok")) and not failed
            except Exception as exc:
                reasons.append(f"{path}: cannot read package_verify JSON: {exc}")
        evidence.append(item)
    return external_stage_result("package_verify", title, purpose, required, reasons, evidence, action)


def evaluate_system_verify(path: Path | None, required: bool) -> dict[str, Any]:
    title = "System readiness verification"
    purpose = "Ensure required package, map, bundle, update, localization, and mission smoke checks are green."
    action = "Run tools/system_verify.py --skip-runtime for fast packaging, and full system_verify before field deployment."
    if path is None and not required:
        return skipped_external_stage("system_verify", title, purpose, action)
    reasons: list[str] = []
    evidence: list[dict[str, Any]] = []
    if path is None:
        reasons.append("missing system_verify JSON")
    else:
        item = {"name": path.name, "path": str(path), "ok": False, "kind": "system_verify", "metrics": {}}
        if not path.exists() or path.stat().st_size <= 0:
            reasons.append(f"missing system_verify JSON: {path}")
        else:
            try:
                data = read_json(path)
                required_failures = [
                    check.get("name", "<unnamed>")
                    for check in data.get("checks", [])
                    if check.get("required") and check.get("status") not in {"PASS", "READY"}
                ]
                if required_failures:
                    reasons.append(f"required system checks failed: {', '.join(required_failures)}")
                item["metrics"] = {
                    "status": data.get("status"),
                    "checks": len(data.get("checks", [])),
                    "required_failures": required_failures,
                }
                item["ok"] = not required_failures
            except Exception as exc:
                reasons.append(f"{path}: cannot read system_verify JSON: {exc}")
        evidence.append(item)
    return external_stage_result("system_verify", title, purpose, required, reasons, evidence, action)


def evaluate_build_run(
    run_dir: str | Path,
    *,
    localization_jsons: list[str | Path] | None = None,
    production_json: str | Path | None = None,
    package_verify_json: str | Path | None = None,
    system_verify_json: str | Path | None = None,
    require_localization: bool = False,
    require_production: bool = False,
    require_package_verify: bool = False,
    require_system_verify: bool = False,
    min_success: float = 0.90,
    max_ok_to_fail: int = 0,
    max_final_fail_run: int = 30,
    min_production_success: float = 0.90,
    max_production_wall_p90: float = 0.0,
    min_production_frames: int = 1,
    max_production_fail_run: int = 30,
    min_production_inliers_p5: float = 0.0,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    stages = {
        req.key: evaluate_requirement(root, req)
        for req in ONE_CLICK_REQUIREMENTS
    }
    loc_paths = [Path(path) for path in (localization_jsons or [])]
    stages["holdout_localization"] = evaluate_holdout_localization(
        loc_paths,
        require_localization,
        min_success,
        max_ok_to_fail,
        max_final_fail_run,
    )
    stages["production_replay"] = evaluate_production_replay(
        Path(production_json) if production_json else None,
        require_production,
        min_production_success,
        max_production_wall_p90,
        min_production_frames,
        max_production_fail_run,
        min_production_inliers_p5,
    )
    stages["package_verify"] = evaluate_package_verify(
        Path(package_verify_json) if package_verify_json else None,
        require_package_verify,
    )
    stages["system_verify"] = evaluate_system_verify(
        Path(system_verify_json) if system_verify_json else None,
        require_system_verify,
    )
    next_blocked = next((key for key, value in stages.items() if not value["ok"]), None)
    failed = [key for key, value in stages.items() if not value["ok"]]
    passed = [key for key, value in stages.items() if value["status"] == "PASS"]
    skipped = [key for key, value in stages.items() if value["status"] == "SKIP"]
    return {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "contract": "one_click_localizable_map_v1",
        "run_dir": str(root),
        "overall_ok": not failed,
        "next_blocked_stage": next_blocked,
        "counts": {
            "pass": len(passed),
            "fail": len(failed),
            "skip": len(skipped),
            "total": len(stages),
        },
        "stages": stages,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# One-Click Map Build Gate Summary",
        "",
        f"- Contract: `{report['contract']}`",
        f"- Run dir: `{report['run_dir']}`",
        f"- Overall: `{'PASS' if report['overall_ok'] else 'FAIL'}`",
        f"- Next blocked stage: `{report['next_blocked_stage'] or ''}`",
        "",
        "| Stage | Status | Required | Evidence | Reasons | Operator action |",
        "|---|---|---:|---|---|---|",
    ]
    for key, stage in report["stages"].items():
        evidence = ", ".join(item["name"] for item in stage.get("evidence", []) if item.get("ok"))
        reasons = "; ".join(stage.get("reasons") or [])
        lines.append(
            f"| {key} | {stage['status']} | {'yes' if stage['required'] else 'no'} | "
            f"{evidence or '-'} | {reasons or '-'} | {stage['operator_action']} |"
        )
    lines.append("")
    lines.append("## Promotion Rule")
    lines.append("")
    lines.append(
        "Only a `PASS` overall result can be promoted to a non-expert operator. "
        "A colored point cloud without `localization_bundle` PASS is inspection-only, not deployable."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_files(report: dict[str, Any], json_out: Path, md_out: Path) -> tuple[Path, Path]:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, md_out)
    return json_out, md_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--localization-json", action="append", default=[])
    parser.add_argument("--production-json", default="")
    parser.add_argument("--package-verify-json", default="")
    parser.add_argument("--system-verify-json", default="")
    parser.add_argument("--require-localization", action="store_true")
    parser.add_argument("--require-production", action="store_true")
    parser.add_argument("--require-package-verify", action="store_true")
    parser.add_argument("--require-system-verify", action="store_true")
    parser.add_argument("--min-success", type=float, default=0.90)
    parser.add_argument("--max-ok-to-fail", type=int, default=0)
    parser.add_argument("--max-final-fail-run", type=int, default=30)
    parser.add_argument("--min-production-success", type=float, default=0.90)
    parser.add_argument("--max-production-wall-p90", type=float, default=0.0)
    parser.add_argument("--min-production-frames", type=int, default=1)
    parser.add_argument("--max-production-fail-run", type=int, default=30)
    parser.add_argument("--min-production-inliers-p5", type=float, default=0.0)
    args = parser.parse_args()

    report = evaluate_build_run(
        args.run_dir,
        localization_jsons=args.localization_json,
        production_json=args.production_json or None,
        package_verify_json=args.package_verify_json or None,
        system_verify_json=args.system_verify_json or None,
        require_localization=args.require_localization,
        require_production=args.require_production,
        require_package_verify=args.require_package_verify,
        require_system_verify=args.require_system_verify,
        min_success=args.min_success,
        max_ok_to_fail=args.max_ok_to_fail,
        max_final_fail_run=args.max_final_fail_run,
        min_production_success=args.min_production_success,
        max_production_wall_p90=args.max_production_wall_p90,
        min_production_frames=args.min_production_frames,
        max_production_fail_run=args.max_production_fail_run,
        min_production_inliers_p5=args.min_production_inliers_p5,
    )
    run_dir = Path(args.run_dir)
    json_out = Path(args.out_json) if args.out_json else run_dir / "BUILD_GATE_SUMMARY.json"
    md_out = Path(args.out_md) if args.out_md else run_dir / "BUILD_GATE_SUMMARY.md"
    write_summary_files(report, json_out, md_out)
    print(f"[stage_gate_contract] wrote {json_out}")
    print(f"[stage_gate_contract] wrote {md_out}")
    if not report["overall_ok"] and not args.allow_fail:
        raise SystemExit(f"one-click map gate failed at {report['next_blocked_stage']}")


if __name__ == "__main__":
    main()
