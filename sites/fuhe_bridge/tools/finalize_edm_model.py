#!/usr/bin/env python3
"""Remove pure-rotation observations, run fixed-intrinsics BA, and score S5."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from ts_common import (
    BUILD,
    Gate,
    read_fresh_gate_stage_metrics,
    required_check_ids,
)
from ts_intrinsics import FUHE_CX, FUHE_CY, FUHE_FX, FUHE_FY
from resource_guard import required_cli_path, run_global_heavy_job


MAX_REPROJECTION_ERROR_PX = 8.0
FIXED_CAMERA_KEY = ("PINHOLE", 1920, 1080)
FIXED_CAMERA_PARAMS = np.asarray(
    [FUHE_FX, FUHE_FY, FUHE_CX, FUHE_CY], dtype=np.float64
)
EXPECTED_SEQUENCES = frozenset(video.seq for video in BUILD)

MIN_REGISTERED_FRACTION = 0.95
MIN_PER_SEQUENCE_REGISTERED_FRACTION = 0.90
MIN_POINTS_PER_REGISTERED_IMAGE = 200.0
MAX_MEAN_REPROJECTION_ERROR_PX = 2.0
MAX_P95_REPROJECTION_ERROR_PX = 4.0
MAX_P99_REPROJECTION_ERROR_PX = 8.0
MAX_INTRINSICS_DRIFT = 1e-6
MIN_MEDIAN_TRIANGULATION_ANGLE_DEG = 5.0
MAX_FRACTION_TRIANGULATION_BELOW_1DEG = 0.02
FINAL_GATE_THRESHOLDS = {
    "registered_fraction_minimum": MIN_REGISTERED_FRACTION,
    "per_sequence_registered_fraction_minimum": (
        MIN_PER_SEQUENCE_REGISTERED_FRACTION
    ),
    "points_per_registered_image_minimum": MIN_POINTS_PER_REGISTERED_IMAGE,
    "mean_reprojection_error_px_maximum": MAX_MEAN_REPROJECTION_ERROR_PX,
    "p95_reprojection_error_px_maximum": MAX_P95_REPROJECTION_ERROR_PX,
    "p99_reprojection_error_px_maximum": MAX_P99_REPROJECTION_ERROR_PX,
    "invalid_reprojection_observations_maximum": 0,
    "maximum_intrinsics_delta": MAX_INTRINSICS_DRIFT,
    "median_triangulation_angle_deg_minimum": (
        MIN_MEDIAN_TRIANGULATION_ANGLE_DEG
    ),
    "fraction_triangulation_angle_below_1deg_maximum": (
        MAX_FRACTION_TRIANGULATION_BELOW_1DEG
    ),
    "active_component_fraction_minimum": 1.0,
}


def fixed_camera_evidence(reconstruction) -> dict:
    """Prove the model has exactly one Fuhe v2 fixed PINHOLE camera."""
    cameras = list(reconstruction.cameras.values())
    keys = [
        (str(camera.model.name), int(camera.width), int(camera.height))
        for camera in cameras
    ]
    params = [np.asarray(camera.params, dtype=np.float64) for camera in cameras]
    maximum_delta = math.inf
    params_valid = (
        len(params) == 1
        and params[0].shape == FIXED_CAMERA_PARAMS.shape
        and bool(np.isfinite(params[0]).all())
    )
    if params_valid:
        maximum_delta = float(np.max(np.abs(params[0] - FIXED_CAMERA_PARAMS)))
    ok = (
        len(cameras) == 1
        and keys == [FIXED_CAMERA_KEY]
        and params_valid
        and maximum_delta <= MAX_INTRINSICS_DRIFT
    )
    return {
        "ok": ok,
        "camera_count": len(cameras),
        "camera_keys": [list(key) for key in keys],
        "expected_key": list(FIXED_CAMERA_KEY),
        "expected_params": FIXED_CAMERA_PARAMS.tolist(),
        "actual_params": [value.tolist() for value in params],
        "maximum_intrinsics_delta": maximum_delta,
        "maximum_allowed_delta": MAX_INTRINSICS_DRIFT,
    }


def final_gate_checks(
    metrics: dict,
    *,
    fixed_camera_ok: bool,
    maximum_intrinsics_delta: float,
    remaining_quarantined_points: int,
    pure_rotation_complete: bool,
) -> dict[str, bool]:
    """Evaluate the formal Fuhe S5 final-model thresholds."""
    per_sequence = metrics.get("per_sequence_registered_fraction", {})
    registration_ok = (
        metrics.get("registered_fraction", 0.0) >= MIN_REGISTERED_FRACTION
        and set(per_sequence) == EXPECTED_SEQUENCES
        and all(
            float(value) >= MIN_PER_SEQUENCE_REGISTERED_FRACTION
            for value in per_sequence.values()
        )
    )
    return {
        "G5.1": registration_ok,
        "G5.2": metrics.get("points_per_registered_image", 0.0)
        >= MIN_POINTS_PER_REGISTERED_IMAGE,
        "G5.3": metrics.get("mean_reprojection_error_px", math.inf)
        <= MAX_MEAN_REPROJECTION_ERROR_PX
        and metrics.get("p95_reprojection_error_px", math.inf)
        <= MAX_P95_REPROJECTION_ERROR_PX
        and metrics.get("p99_reprojection_error_px", math.inf)
        <= MAX_P99_REPROJECTION_ERROR_PX
        and metrics.get("invalid_reprojection_observations", 1) == 0,
        "G5.4": metrics.get("largest_component_fraction", 0.0) == 1.0
        and metrics.get("zero_observation_registered", 1) == 0
        and metrics.get("short_track_points", 1) == 0
        and remaining_quarantined_points == 0,
        "G5.5": bool(fixed_camera_ok)
        and math.isfinite(maximum_intrinsics_delta)
        and maximum_intrinsics_delta <= MAX_INTRINSICS_DRIFT,
        "G5.6": metrics.get("median_triangulation_angle_deg", 0.0)
        >= MIN_MEDIAN_TRIANGULATION_ANGLE_DEG
        and metrics.get("fraction_triangulation_angle_below_1deg", 1.0)
        <= MAX_FRACTION_TRIANGULATION_BELOW_1DEG
        and metrics.get("forbidden_observations", 1) == 0
        and bool(pure_rotation_complete),
    }


def triangulation_angle_deg(xyz: np.ndarray, centers: np.ndarray) -> float:
    """Return the widest ray angle observing one 3D point."""
    rays = np.asarray(centers, dtype=np.float64) - np.asarray(xyz, dtype=np.float64)
    norms = np.linalg.norm(rays, axis=1)
    rays = rays[norms > 1e-12]
    if len(rays) < 2:
        return 0.0
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    minimum_dot = float(np.min(np.clip(rays @ rays.T, -1.0, 1.0)))
    return math.degrees(math.acos(minimum_dot))


def largest_image_component_ids(
    registered_ids: set[int], tracks: Iterable[Iterable[int]]
) -> set[int]:
    """Return the largest image component induced by shared 3D tracks."""
    if not registered_ids:
        return set()
    parent = {image_id: image_id for image_id in registered_ids}
    size = {image_id: 1 for image_id in registered_ids}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a == b:
            return
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]

    for track in tracks:
        ids = [int(image_id) for image_id in track if int(image_id) in parent]
        if len(ids) >= 2:
            anchor = ids[0]
            for image_id in ids[1:]:
                union(anchor, image_id)
    components: dict[int, set[int]] = defaultdict(set)
    for image_id in registered_ids:
        components[find(image_id)].add(image_id)
    return max(components.values(), key=lambda ids: (len(ids), -min(ids)))


def largest_image_component_fraction(
    registered_ids: set[int], tracks: Iterable[Iterable[int]]
) -> tuple[float, int]:
    """Measure connectivity induced by shared 3D tracks, including isolates."""
    largest = largest_image_component_ids(registered_ids, tracks)
    if not registered_ids:
        return 0.0, 0
    return len(largest) / len(registered_ids), len(largest)


def eligible_registered_ids(images: Iterable, forbidden_names: set[str]) -> set[int]:
    """Return images expected to participate in the 3D track graph."""
    return {
        int(image.image_id)
        for image in images
        if bool(getattr(image, "has_pose", True))
        and str(image.name) not in forbidden_names
    }


def _registered_images(rec) -> list:
    return [image for image in rec.images.values() if image.has_pose]


def _track_image_ids(rec) -> list[list[int]]:
    return [
        [int(element.image_id) for element in point.track.elements]
        for point in rec.points3D.values()
    ]


def _reconstruction_counts(rec) -> dict[str, int]:
    registered = _registered_images(rec)
    return {
        "registered_images": len(registered),
        "points3D": len(rec.points3D),
        "observations": sum(
            int(point.track.length()) for point in rec.points3D.values()
        ),
        "short_track_points": sum(
            point.track.length() < 3 for point in rec.points3D.values()
        ),
        "zero_observation_registered": sum(
            not any(point2d.has_point3D() for point2d in image.points2D)
            for image in registered
        ),
    }


def point_ids_spanning_sequence_edges(
    points: dict,
    image_sequences: dict[int, str],
    edges: set[tuple[str, str]],
) -> set[int]:
    """Return all 3D points co-observed across any quarantined sequence edge."""
    normalized_edges = {tuple(sorted(edge)) for edge in edges}
    selected = set()
    for point_id, point in points.items():
        sequences = {
            image_sequences[int(element.image_id)] for element in point.track.elements
        }
        if any(set(edge) <= sequences for edge in normalized_edges):
            selected.add(int(point_id))
    return selected


def _quarantine_sequence_edges(rec, edges: set[tuple[str, str]]) -> dict:
    if not edges:
        return {"points_deleted": 0, "observations_deleted": 0}
    image_sequences = {
        int(image.image_id): str(image.name).split("/", 1)[0]
        for image in rec.images.values()
    }
    point_ids = point_ids_spanning_sequence_edges(
        rec.points3D, image_sequences, edges
    )
    observations = sum(rec.points3D[point_id].track.length() for point_id in point_ids)
    for point_id in point_ids:
        rec.delete_point3D(point_id)
    return {
        "points_deleted": len(point_ids),
        "observations_deleted": int(observations),
    }


def _deregister_outside_largest_component(
    rec, forbidden_names: set[str]
) -> list[str]:
    """Remove active, localizable images not connected to the dominant map."""
    eligible = eligible_registered_ids(rec.images.values(), forbidden_names)
    keep = largest_image_component_ids(eligible, _track_image_ids(rec))
    remove = sorted(
        (
            image
            for image in _registered_images(rec)
            if image.name not in forbidden_names and int(image.image_id) not in keep
        ),
        key=lambda image: image.name,
    )
    names = [str(image.name) for image in remove]
    for image in remove:
        rec.deregister_frame(int(image.frame_id))
    return names


def reprojection_error_px(image, point2d, point3d) -> float | None:
    """Return finite positive-depth pixel error, or None for an invalid ray."""
    camera_point = np.asarray(
        image.cam_from_world() * np.asarray(point3d.xyz, dtype=np.float64),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(camera_point)) or camera_point[2] <= 1e-12:
        return None
    projected = image.project_point(point3d.xyz)
    if projected is None:
        return None
    delta = np.asarray(projected, dtype=np.float64) - np.asarray(
        point2d.xy, dtype=np.float64
    )
    error = float(np.linalg.norm(delta))
    return error if math.isfinite(error) else None


def _bridge_only_names(frame_manifest: Path) -> set[str]:
    payload = json.loads(frame_manifest.read_text(encoding="utf-8"))
    return {
        str(frame["name"])
        for frame in payload["frames"]
        if frame.get("role") == "bridge_only"
        or frame.get("motion_class") == "pure_rotation"
    }


def _camera_table(reconstruction) -> dict[tuple[str, int, int], np.ndarray]:
    table: dict[tuple[str, int, int], np.ndarray] = {}
    for camera in reconstruction.cameras.values():
        key = (str(camera.model.name), int(camera.width), int(camera.height))
        params = np.asarray(camera.params, dtype=np.float64)
        if key in table and not np.array_equal(table[key], params):
            raise ValueError(f"multiple incompatible cameras for {key}")
        table[key] = params
    return table


def restore_seed_intrinsics(
    reconstruction, seed_cameras: dict[tuple[str, int, int], np.ndarray]
) -> None:
    """Restore every camera to the exact seeded parameter vector."""
    seen: set[tuple[str, int, int]] = set()
    for camera in reconstruction.cameras.values():
        key = (str(camera.model.name), int(camera.width), int(camera.height))
        if key not in seed_cameras:
            raise ValueError(f"missing seeded camera for {key}")
        camera.params = np.asarray(seed_cameras[key], dtype=np.float64).copy()
        seen.add(key)
    missing = set(seed_cameras) - seen
    if missing:
        raise ValueError(f"seed cameras are unused by reconstruction: {sorted(missing)}")


def _remove_forbidden_observations(rec, forbidden_names: set[str]) -> tuple[int, int]:
    images_by_name = {image.name: image for image in rec.images.values()}
    removed = 0
    found = 0
    for name in sorted(forbidden_names):
        image = images_by_name.get(name)
        if image is None:
            continue
        found += 1
        indices = [
            index for index, point2d in enumerate(image.points2D) if point2d.has_point3D()
        ]
        for index in indices:
            rec.delete_observation(image.image_id, index)
            removed += 1
    return removed, found


def _delete_short_tracks(rec, minimum: int = 3) -> int:
    short = [
        int(point_id)
        for point_id, point in rec.points3D.items()
        if point.track.length() < minimum
    ]
    for point_id in short:
        rec.delete_point3D(point_id)
    return len(short)


def _deregister_zero_observation_images(rec) -> list[str]:
    """Deregister posed images that no longer contribute any 3D observation."""
    remove = sorted(
        (
            image
            for image in _registered_images(rec)
            if not any(point2d.has_point3D() for point2d in image.points2D)
        ),
        key=lambda image: str(image.name),
    )
    for image in remove:
        rec.deregister_frame(int(image.frame_id))
    return [str(image.name) for image in remove]


def _filter_reprojection_observations(
    rec, max_error_px: float = MAX_REPROJECTION_ERROR_PX
) -> dict[str, int]:
    """Remove invalid-depth and high-error observations without mutating mid-scan."""
    if not math.isfinite(max_error_px) or max_error_px <= 0:
        raise ValueError("max_error_px must be finite and positive")
    candidates: list[tuple[int, int]] = []
    invalid = 0
    excessive = 0
    for image in _registered_images(rec):
        for index, point2d in enumerate(image.points2D):
            if not point2d.has_point3D():
                continue
            point = rec.points3D[int(point2d.point3D_id)]
            error = reprojection_error_px(image, point2d, point)
            if error is None:
                invalid += 1
                candidates.append((int(image.image_id), index))
            elif error > max_error_px:
                excessive += 1
                candidates.append((int(image.image_id), index))

    removed = 0
    for image_id, index in candidates:
        point2d = rec.images[image_id].points2D[index]
        if not point2d.has_point3D():
            continue
        rec.delete_observation(image_id, index)
        removed += 1
    return {
        "invalid_candidates": invalid,
        "excessive_candidates": excessive,
        "observations_removed": removed,
    }


def _model_metrics(
    rec,
    expected_images: int,
    forbidden_names: set[str],
    expected_per_sequence: dict[str, int],
) -> dict:
    active_images = _registered_images(rec)
    registered = {int(image.image_id) for image in active_images}
    connectivity_ids = eligible_registered_ids(rec.images.values(), forbidden_names)
    tracks = _track_image_ids(rec)
    component_fraction, component_size = largest_image_component_fraction(
        connectivity_ids, tracks
    )
    centers = {
        int(image.image_id): np.asarray(image.projection_center(), dtype=np.float64)
        for image in active_images
    }
    angles = []
    for point in rec.points3D.values():
        point_centers = np.asarray(
            [centers[int(element.image_id)] for element in point.track.elements]
        )
        angles.append(triangulation_angle_deg(np.asarray(point.xyz), point_centers))
    angle_array = np.asarray(angles, dtype=np.float64)

    forbidden_observations = 0
    registered_per_sequence = Counter(
        str(image.name).split("/", 1)[0] for image in active_images
    )
    sequence_errors: dict[str, list[float]] = defaultdict(list)
    all_errors: list[float] = []
    invalid_reprojection_observations = 0
    zero_observation_registered = 0
    for image in active_images:
        is_forbidden = image.name in forbidden_names
        sequence = image.name.split("/", 1)[0]
        image_observations = 0
        for point2d in image.points2D:
            if not point2d.has_point3D():
                continue
            image_observations += 1
            if is_forbidden:
                forbidden_observations += 1
            point = rec.points3D[int(point2d.point3D_id)]
            error = reprojection_error_px(image, point2d, point)
            if error is None:
                invalid_reprojection_observations += 1
                continue
            sequence_errors[sequence].append(error)
            all_errors.append(error)
        if image_observations == 0:
            zero_observation_registered += 1
    per_sequence = {
        sequence: float(np.mean(errors))
        for sequence, errors in sorted(sequence_errors.items())
        if errors
    }
    error_array = np.asarray(all_errors, dtype=np.float64)
    mean_error = float(np.mean(error_array)) if len(error_array) else float("inf")
    p95_error = (
        float(np.percentile(error_array, 95)) if len(error_array) else float("inf")
    )
    p99_error = (
        float(np.percentile(error_array, 99)) if len(error_array) else float("inf")
    )
    per_sequence_registered_fraction = {
        sequence: registered_per_sequence.get(sequence, 0) / expected
        for sequence, expected in sorted(expected_per_sequence.items())
    }
    short_track_points = sum(
        point.track.length() < 3 for point in rec.points3D.values()
    )
    return {
        "registered": len(registered),
        "expected_images": expected_images,
        "registered_fraction": len(registered) / expected_images,
        "points3D": int(rec.num_points3D()),
        "points_per_registered_image": int(rec.num_points3D())
        / max(len(registered), 1),
        "mean_reprojection_error_px": mean_error,
        "p95_reprojection_error_px": p95_error,
        "p99_reprojection_error_px": p99_error,
        "invalid_reprojection_observations": invalid_reprojection_observations,
        "largest_component_images": component_size,
        "connectivity_eligible_images": len(connectivity_ids),
        "largest_component_fraction": component_fraction,
        "median_triangulation_angle_deg": (
            float(np.median(angle_array)) if len(angle_array) else 0.0
        ),
        "fraction_triangulation_angle_below_1deg": (
            float(np.mean(angle_array < 1.0)) if len(angle_array) else 1.0
        ),
        "forbidden_observations": forbidden_observations,
        "zero_observation_registered": zero_observation_registered,
        "short_track_points": short_track_points,
        "per_sequence_registered": dict(sorted(registered_per_sequence.items())),
        "per_sequence_expected": dict(sorted(expected_per_sequence.items())),
        "per_sequence_registered_fraction": per_sequence_registered_fraction,
        "per_sequence_mean_reprojection_error_px": per_sequence,
    }


def _main_locked() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--intrinsics-seed", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument(
        "--quarantine-edge",
        nargs=2,
        action="append",
        default=[],
        metavar=("SEQUENCE_A", "SEQUENCE_B"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    gate_dir = (
        args.metrics_out.parent
        if args.metrics_out.parent.name == "gates"
        else args.metrics_out.parent / "gates"
    )
    predecessor_gate = gate_dir / "S4_doppelgangers.json"
    read_fresh_gate_stage_metrics(
        predecessor_gate, expected_stage="S4_doppelgangers"
    )

    import pycolmap

    manifest = json.loads(args.frame_manifest.read_text(encoding="utf-8"))
    forbidden = _bridge_only_names(args.frame_manifest)
    if args.output_model.exists():
        if not args.overwrite:
            raise SystemExit(f"output already exists: {args.output_model}")
        shutil.rmtree(args.output_model)
    args.output_model.mkdir(parents=True)

    rec = pycolmap.Reconstruction(str(args.input_model))
    seed = pycolmap.Reconstruction(str(args.intrinsics_seed))
    seed_fixed_camera = fixed_camera_evidence(seed)
    if not seed_fixed_camera["ok"]:
        raise SystemExit("intrinsics seed is not the fixed Fuhe v2 camera")
    input_camera = fixed_camera_evidence(rec)
    input_key_ok = (
        input_camera["camera_count"] == 1
        and input_camera["camera_keys"] == [list(FIXED_CAMERA_KEY)]
        and len(input_camera["actual_params"]) == 1
        and len(input_camera["actual_params"][0]) == len(FIXED_CAMERA_PARAMS)
        and np.isfinite(input_camera["actual_params"][0]).all()
    )
    if not input_key_ok:
        raise SystemExit("input model is not a single finite 1920x1080 PINHOLE camera")
    cleanup_stages = {"input": _reconstruction_counts(rec)}
    seed_cameras = _camera_table(seed)
    removed, found = _remove_forbidden_observations(rec, forbidden)
    deleted_short_before = _delete_short_tracks(rec, minimum=3)
    deregistered_zero = _deregister_zero_observation_images(rec)
    restore_seed_intrinsics(rec, seed_cameras)
    prefilter = _filter_reprojection_observations(rec)
    deleted_short_prefilter = _delete_short_tracks(rec, minimum=3)
    deregistered_zero.extend(_deregister_zero_observation_images(rec))
    deregistered_prefilter = _deregister_outside_largest_component(rec, forbidden)
    quarantine_edges = {tuple(sorted(edge)) for edge in args.quarantine_edge}
    quarantine = _quarantine_sequence_edges(rec, quarantine_edges)
    deleted_short_after_component_prune = _delete_short_tracks(rec, minimum=3)
    deregistered_zero.extend(_deregister_zero_observation_images(rec))
    deregistered_after_quarantine = _deregister_outside_largest_component(
        rec, forbidden
    )
    cleanup_stages["pre_ba_clean"] = _reconstruction_counts(rec)

    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.refine_sensor_from_rig = False
    options.refine_points3D = True
    options.min_track_length = 3
    pycolmap.bundle_adjustment(rec, options)
    restore_seed_intrinsics(rec, seed_cameras)
    postfilter = _filter_reprojection_observations(rec)
    deleted_short_postfilter = _delete_short_tracks(rec, minimum=3)
    deregistered_zero.extend(_deregister_zero_observation_images(rec))
    deregistered_postfilter = _deregister_outside_largest_component(rec, forbidden)
    deleted_short_after_post_component_prune = _delete_short_tracks(rec, minimum=3)
    deregistered_zero.extend(_deregister_zero_observation_images(rec))
    restore_seed_intrinsics(rec, seed_cameras)
    cleanup_stages["final"] = _reconstruction_counts(rec)
    rec.write(str(args.output_model))

    final_cameras = _camera_table(rec)
    camera_keys_match = set(final_cameras) == set(seed_cameras)
    maximum_intrinsics_delta = float("inf")
    if camera_keys_match:
        maximum_intrinsics_delta = max(
            float(np.max(np.abs(final_cameras[key] - seed_cameras[key])))
            for key in seed_cameras
        )

    expected_per_sequence = dict(
        Counter(str(frame["seq"]) for frame in manifest["frames"])
    )
    metrics = _model_metrics(
        rec,
        int(manifest["n_frames"]),
        forbidden,
        expected_per_sequence,
    )
    final_image_sequences = {
        int(image.image_id): str(image.name).split("/", 1)[0]
        for image in rec.images.values()
    }
    remaining_quarantined_points = len(
        point_ids_spanning_sequence_edges(
            rec.points3D, final_image_sequences, quarantine_edges
        )
    )
    final_fixed_camera = fixed_camera_evidence(rec)
    epoch_gate = {
        "applicable": False,
        "status": "NOT_APPLICABLE",
        "epochs": ["2026-06-15"],
        "reason": "all five Fuhe build sequences belong to one capture epoch",
    }
    metrics.update(
        {
            "input_model": str(args.input_model.resolve()),
            "output_model": str(args.output_model.resolve()),
            "pure_rotation_names": sorted(forbidden),
            "pure_rotation_images_found": found,
            "observations_removed": removed,
            "reprojection_filter_threshold_px": MAX_REPROJECTION_ERROR_PX,
            "prefilter": prefilter,
            "postfilter": postfilter,
            "short_tracks_deleted": (
                deleted_short_before
                + deleted_short_prefilter
                + deleted_short_after_component_prune
                + deleted_short_postfilter
                + deleted_short_after_post_component_prune
            ),
            "short_tracks_deleted_by_stage": {
                "before_filter": deleted_short_before,
                "after_prefilter": deleted_short_prefilter,
                "after_component_prune": deleted_short_after_component_prune,
                "after_postfilter": deleted_short_postfilter,
                "after_post_component_prune": (
                    deleted_short_after_post_component_prune
                ),
            },
            "deregistered_outside_largest_component": sorted(
                set(deregistered_prefilter)
                | set(deregistered_after_quarantine)
                | set(deregistered_postfilter)
            ),
            "deregistered_zero_observation_images": sorted(set(deregistered_zero)),
            "quarantined_sequence_edges": [list(edge) for edge in sorted(quarantine_edges)],
            "quarantine": quarantine,
            "remaining_quarantined_edge_points": remaining_quarantined_points,
            "camera_keys_match_seed": camera_keys_match,
            "maximum_intrinsics_delta": maximum_intrinsics_delta,
            "fixed_camera": final_fixed_camera,
            "fixed_k_bundle_adjustment": {
                "refine_focal_length": False,
                "refine_principal_point": False,
                "refine_extra_params": False,
                "refine_sensor_from_rig": False,
            },
            "epoch_gate": epoch_gate,
            "gate_thresholds": FINAL_GATE_THRESHOLDS,
            "cleanup_stage_counts": cleanup_stages,
        }
    )
    checks = final_gate_checks(
        metrics,
        fixed_camera_ok=(
            camera_keys_match
            and final_fixed_camera["ok"]
            and seed_fixed_camera["ok"]
        ),
        maximum_intrinsics_delta=maximum_intrinsics_delta,
        remaining_quarantined_points=remaining_quarantined_points,
        pure_rotation_complete=found == len(forbidden),
    )
    metrics["checks"] = checks
    metrics["stage"] = "S5_fixed_intrinsics"
    metrics["status"] = "PASS" if all(checks.values()) else "FAIL"
    gate = Gate(
        "S5_fixed_intrinsics",
        required_check_ids("S5_fixed_intrinsics"),
        script_path=__file__,
        source_files=[
            Path(__file__).with_name("ts_common.py"),
            Path(__file__).with_name("ts_intrinsics.py"),
            Path(__file__).with_name("resource_guard.py"),
        ],
        input_artifacts={
            "input_model": args.input_model,
            "output_model": args.output_model,
            "frame_manifest": args.frame_manifest,
            "intrinsics_seed": args.intrinsics_seed,
        },
    )
    gate.record_predecessor_gate(
        "S4_doppelgangers",
        predecessor_gate,
        expected_stage="S4_doppelgangers",
    )
    for gid, passed in checks.items():
        gate.check(
            gid,
            passed,
            "final fixed-intrinsics model predicate recomputed",
            thresholds=FINAL_GATE_THRESHOLDS,
            observed_status=metrics["status"],
        )
    payload = gate.write(
        args.metrics_out.parent,
        output_path=args.metrics_out,
        stage_metrics=metrics,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    arguments = sys.argv[1:]
    if "--help" in arguments or "-h" in arguments:
        _main_locked()
        return
    disk_path = required_cli_path(arguments, "--output-model")
    run_global_heavy_job(disk_path, lambda _evidence: _main_locked())


if __name__ == "__main__":
    main()
