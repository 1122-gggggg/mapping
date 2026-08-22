#!/usr/bin/env python3
"""Fail-closed reachability audit before official G6.1 hotspot match injection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

def minimum_track_merges_to_p90(*, total_records: int, high_records: int) -> int:
    """Minimum removed exclusive records needed to make high/(total) <= 10%."""
    if total_records <= 0 or high_records < 0 or high_records > total_records:
        raise ValueError("invalid record counts")
    excess = high_records - 0.1 * total_records
    return max(0, int(math.ceil((excess / 0.9) - 1e-12)))


def _distance_to_route(xyz: np.ndarray, route: np.ndarray) -> float:
    if len(route) == 1:
        return float(np.linalg.norm(xyz - route[0]))
    starts = route[:-1]
    vectors = route[1:] - starts
    squared_lengths = np.sum(vectors * vectors, axis=1)
    parameters = np.divide(
        np.sum((xyz - starts) * vectors, axis=1),
        squared_lengths,
        out=np.zeros_like(squared_lengths),
        where=squared_lengths > 0,
    )
    projections = starts + np.clip(parameters, 0.0, 1.0)[:, None] * vectors
    return float(np.min(np.linalg.norm(projections - xyz, axis=1)))


def _route_masked(
    points: list[tuple[int, np.ndarray]],
    left_route: np.ndarray,
    right_route: np.ndarray,
    radius: float,
) -> list[tuple[int, np.ndarray]]:
    return [
        (point_id, xyz)
        for point_id, xyz in points
        if _distance_to_route(xyz, left_route) <= radius
        and _distance_to_route(xyz, right_route) <= radius
    ]


def _visible_in_any_target(
    reconstruction: object,
    target_images: list[object],
    xyz: np.ndarray,
    *,
    margin_px: float,
) -> bool:
    for image in target_images:
        rigid = image.cam_from_world()
        cam = (
            np.asarray(rigid.rotation.matrix(), dtype=np.float64) @ xyz
            + np.asarray(rigid.translation, dtype=np.float64)
        )
        if cam[2] <= 0:
            continue
        camera = reconstruction.cameras[int(image.camera_id)]
        pixel = camera.img_from_cam(cam)
        if pixel is None:
            continue
        if (
            margin_px <= pixel[0] < camera.width - margin_px
            and margin_px <= pixel[1] < camera.height - margin_px
        ):
            return True
    return False


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--geometry-gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    # Official G6.1 scored pairs: P109/P110 ↔ P111 (not the frozen P114 notes).
    parser.add_argument("--source-sequence", default="P1110111")
    parser.add_argument("--target-sequence", action="append", default=None)
    parser.add_argument("--maximum-p90-over-span", type=float, default=0.04)
    parser.add_argument("--route-support-radius-over-span", type=float, default=0.10)
    parser.add_argument("--max-points-per-cell", type=int, default=8)
    parser.add_argument("--roi-margin-px", type=float, default=32.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from audit_map_geometry import apply_official_hotspot_defaults

    apply_official_hotspot_defaults(args)
    if not args.model.is_dir() or not args.geometry_gate.is_file():
        raise SystemExit("model or geometry gate is missing")
    if args.maximum_p90_over_span <= 0 or args.route_support_radius_over_span <= 0:
        raise SystemExit("geometry thresholds must be positive")

    import pycolmap
    from audit_map_geometry import (
        _directed_voxel_halo_distances,
        robust_spatial_span,
        sequence_exclusive_point_clouds,
    )

    gate = json.loads(args.geometry_gate.read_text(encoding="utf-8"))
    gate_geometry = gate.get("stage_metrics", {}).get(
        "sequence_exclusive_ghost_geometry", {}
    )
    if gate_geometry.get("status") != "FAIL":
        raise SystemExit("reachability audit requires the current failing G6.1 evidence")
    if float(gate_geometry.get("maximum_worst_seq_nn_p90_over_S", -1)) != float(
        args.maximum_p90_over_span
    ):
        raise SystemExit("geometry threshold does not match the source gate")

    reconstruction = pycolmap.Reconstruction(str(args.model))
    images = {
        int(image_id): image
        for image_id, image in reconstruction.images.items()
        if image.has_pose
    }
    image_sequences = {
        image_id: image.name.split("/", 1)[0] for image_id, image in images.items()
    }
    names_by_sequence: dict[str, list[str]] = defaultdict(list)
    images_by_name = {image.name: image for image in images.values()}
    for image in images.values():
        names_by_sequence[image.name.split("/", 1)[0]].append(image.name)
    centers_by_sequence = {
        sequence: np.asarray(
            [
                np.asarray(images_by_name[name].projection_center(), dtype=np.float64)
                for name in sorted(names)
            ]
        )
        for sequence, names in names_by_sequence.items()
    }
    all_centers = np.concatenate(list(centers_by_sequence.values()))
    map_span = robust_spatial_span(all_centers)
    cell_size = args.maximum_p90_over_span * map_span
    route_radius = args.route_support_radius_over_span * map_span
    clouds = sequence_exclusive_point_clouds(
        reconstruction.points3D, image_sequences
    )

    edge_reports = {}
    for target_sequence in sorted(set(args.target_sequence)):
        if args.source_sequence not in clouds or target_sequence not in clouds:
            raise SystemExit(f"missing exclusive cloud: {target_sequence}")
        source_supported = _route_masked(
            clouds[args.source_sequence],
            centers_by_sequence[args.source_sequence],
            centers_by_sequence[target_sequence],
            route_radius,
        )
        target_supported = _route_masked(
            clouds[target_sequence],
            centers_by_sequence[args.source_sequence],
            centers_by_sequence[target_sequence],
            route_radius,
        )
        target_to_source = _directed_voxel_halo_distances(
            target_sequence,
            args.source_sequence,
            target_supported,
            source_supported,
            cell_size=cell_size,
            max_points_per_cell=args.max_points_per_cell,
        )
        source_to_target = _directed_voxel_halo_distances(
            args.source_sequence,
            target_sequence,
            source_supported,
            target_supported,
            cell_size=cell_size,
            max_points_per_cell=args.max_points_per_cell,
        )
        combined = target_to_source + source_to_target
        high_combined = [record for record in combined if record["distance"] > cell_size]
        high_source = [
            record for record in source_to_target if record["distance"] > cell_size
        ]
        minimum_repairs = minimum_track_merges_to_p90(
            total_records=len(combined), high_records=len(high_combined)
        )
        target_images = [
            images_by_name[name] for name in sorted(names_by_sequence[target_sequence])
        ]
        visible_zero = [
            record
            for record in high_source
            if _visible_in_any_target(
                reconstruction,
                target_images,
                np.asarray(record["point_xyz"], dtype=np.float64),
                margin_px=0.0,
            )
        ]
        visible_roi = [
            record
            for record in high_source
            if _visible_in_any_target(
                reconstruction,
                target_images,
                np.asarray(record["point_xyz"], dtype=np.float64),
                margin_px=args.roi_margin_px,
            )
        ]
        nearest_backed = [
            record for record in high_source if record["nearest_point3D_id"] is not None
        ]
        maximum_repairable = len(visible_roi)
        edge = "|".join(sorted((args.source_sequence, target_sequence)))
        observed_gate = gate_geometry.get("sequence_pairs", {}).get(edge, {})
        if int(observed_gate.get("sampled_directed_points", -1)) != len(combined):
            raise SystemExit(f"recomputed record count does not match G6.1 for {edge}")
        edge_reports[edge] = {
            "status": (
                "REACHABLE_FOR_EXACT_ROI_PROBE"
                if maximum_repairable >= minimum_repairs
                else "UNREACHABLE"
            ),
            "total_directed_records": len(combined),
            "high_distance_records": len(high_combined),
            "p114_directed_records": len(source_to_target),
            "p114_high_distance_tracks": len(high_source),
            "p114_high_with_sampled_target_neighbor": len(nearest_backed),
            "p114_high_visible_in_any_target_frame": len(visible_zero),
            "p114_high_visible_with_roi_margin": len(visible_roi),
            "roi_margin_px": args.roi_margin_px,
            "minimum_track_merges_required": minimum_repairs,
            "maximum_theoretical_repairable_tracks": maximum_repairable,
            "reachability_margin_tracks": maximum_repairable - minimum_repairs,
            "formal_reason": (
                "merging a P114-exclusive record removes one high record and one total "
                "record; require (high-m)/(total-m) <= 0.1"
            ),
            "hotspot_point3D_ids": [int(record["point3D_id"]) for record in high_source],
            "visible_hotspot_point3D_ids": [
                int(record["point3D_id"]) for record in visible_roi
            ],
        }

    status = (
        "UNREACHABLE"
        if any(report["status"] == "UNREACHABLE" for report in edge_reports.values())
        else "REACHABLE_FOR_EXACT_ROI_PROBE"
    )
    payload = {
        "schema_version": "fuhe-hotspot-repair-reachability-v1",
        "status": status,
        "promotion_allowed": False,
        "read_only_inputs": True,
        "database_modified": False,
        "model_modified": False,
        "model": {
            "path": str(args.model.resolve()),
            "sha256": _directory_sha256(args.model),
        },
        "geometry_gate": str(args.geometry_gate.resolve()),
        "map_span_S": map_span,
        "maximum_p90_over_span": args.maximum_p90_over_span,
        "edge_reports": edge_reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if status == "UNREACHABLE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
