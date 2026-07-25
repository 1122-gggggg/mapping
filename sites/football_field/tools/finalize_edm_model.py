#!/usr/bin/env python3
"""Remove pure-rotation observations, run fixed-intrinsics BA, and score S5."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


MAX_REPROJECTION_ERROR_PX = 8.0


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
            # Re-check: deleting one observation can drop a track below the minimum
            # length, which makes COLMAP delete the whole Point3D. Sibling
            # observations of that same point in this image then already have
            # point3D_id == -1, and deleting them again raises. Seen on
            # end-of-flight hover frames whose points are short tracks confined
            # to those frames.
            if not image.points2D[index].has_point3D():
                continue
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


def _model_metrics(rec, expected_images: int, forbidden_names: set[str]) -> dict:
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
    sequence_errors: dict[str, list[float]] = defaultdict(list)
    all_errors: list[float] = []
    invalid_reprojection_observations = 0
    nonforbidden_zero_observation_registered = 0
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
        if not is_forbidden and image_observations == 0:
            nonforbidden_zero_observation_registered += 1
    per_sequence = {
        sequence: float(np.mean(errors))
        for sequence, errors in sorted(sequence_errors.items())
        if errors
    }
    mean_error = float(np.mean(all_errors)) if all_errors else float("inf")
    return {
        "registered": len(registered),
        "expected_images": expected_images,
        "registered_fraction": len(registered) / expected_images,
        "points3D": int(rec.num_points3D()),
        "mean_reprojection_error_px": mean_error,
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
        "nonforbidden_zero_observation_registered": (
            nonforbidden_zero_observation_registered
        ),
        "per_sequence_mean_reprojection_error_px": per_sequence,
    }


def main() -> None:
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
    seed_cameras = _camera_table(seed)
    removed, found = _remove_forbidden_observations(rec, forbidden)
    deleted_short_before = _delete_short_tracks(rec, minimum=3)
    restore_seed_intrinsics(rec, seed_cameras)
    prefilter = _filter_reprojection_observations(rec)
    deleted_short_prefilter = _delete_short_tracks(rec, minimum=3)
    deregistered_prefilter = _deregister_outside_largest_component(rec, forbidden)
    quarantine_edges = {tuple(sorted(edge)) for edge in args.quarantine_edge}
    quarantine = _quarantine_sequence_edges(rec, quarantine_edges)
    deleted_short_after_component_prune = _delete_short_tracks(rec, minimum=3)
    deregistered_after_quarantine = _deregister_outside_largest_component(
        rec, forbidden
    )

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
    deregistered_postfilter = _deregister_outside_largest_component(rec, forbidden)
    deleted_short_after_post_component_prune = _delete_short_tracks(rec, minimum=3)
    restore_seed_intrinsics(rec, seed_cameras)
    rec.write(str(args.output_model))

    final_cameras = _camera_table(rec)
    camera_keys_match = set(final_cameras) == set(seed_cameras)
    maximum_intrinsics_delta = float("inf")
    if camera_keys_match:
        maximum_intrinsics_delta = max(
            float(np.max(np.abs(final_cameras[key] - seed_cameras[key])))
            for key in seed_cameras
        )

    metrics = _model_metrics(rec, int(manifest["n_frames"]), forbidden)
    final_image_sequences = {
        int(image.image_id): str(image.name).split("/", 1)[0]
        for image in rec.images.values()
    }
    remaining_quarantined_points = len(
        point_ids_spanning_sequence_edges(
            rec.points3D, final_image_sequences, quarantine_edges
        )
    )
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
            "quarantined_sequence_edges": [list(edge) for edge in sorted(quarantine_edges)],
            "quarantine": quarantine,
            "remaining_quarantined_edge_points": remaining_quarantined_points,
            "camera_keys_match_seed": camera_keys_match,
            "maximum_intrinsics_delta": maximum_intrinsics_delta,
        }
    )
    checks = {
        "G5.1": metrics["registered_fraction"] >= 0.95,
        "G5.2": metrics["mean_reprojection_error_px"] <= 2.0
        and metrics["invalid_reprojection_observations"] == 0,
        "G5.3": metrics["largest_component_fraction"] == 1.0
        and metrics["nonforbidden_zero_observation_registered"] == 0
        and remaining_quarantined_points == 0,
        "G5.4": camera_keys_match and maximum_intrinsics_delta <= 1e-6,
        "G5.5": metrics["median_triangulation_angle_deg"] >= 2.0
        and metrics["fraction_triangulation_angle_below_1deg"] <= 0.05,
        "G5.6": metrics["forbidden_observations"] == 0
        and found == len(forbidden),
    }
    metrics["checks"] = checks
    metrics["status"] = "PASS" if all(checks.values()) else "FAIL"
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    if metrics["status"] != "PASS":
        raise SystemExit("S5.1-S5.6 gate failed")


if __name__ == "__main__":
    main()
