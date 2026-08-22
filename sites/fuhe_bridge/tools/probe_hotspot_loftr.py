#!/usr/bin/env python3
"""Exploratory nearest-route LoFTR stress probe; never promotion evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def fundamental_from_poses(
    k0: np.ndarray,
    r0: np.ndarray,
    t0: np.ndarray,
    k1: np.ndarray,
    r1: np.ndarray,
    t1: np.ndarray,
) -> np.ndarray:
    """Return F for two world-to-camera poses and pixel-space intrinsics."""
    relative_rotation = np.asarray(r1) @ np.asarray(r0).T
    relative_translation = np.asarray(t1) - relative_rotation @ np.asarray(t0)
    essential = _skew(relative_translation) @ relative_rotation
    fundamental = np.linalg.inv(k1).T @ essential @ np.linalg.inv(k0)
    norm = float(np.linalg.norm(fundamental))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("degenerate fixed-pose fundamental matrix")
    return fundamental / norm


def sampson_errors_px(
    fundamental: np.ndarray, xy0: np.ndarray, xy1: np.ndarray
) -> np.ndarray:
    """Return square-root Sampson distances in pixels."""
    points0 = np.column_stack((np.asarray(xy0, dtype=np.float64), np.ones(len(xy0))))
    points1 = np.column_stack((np.asarray(xy1, dtype=np.float64), np.ones(len(xy1))))
    fx0 = (fundamental @ points0.T).T
    ftx1 = (fundamental.T @ points1.T).T
    numerator = np.sum(points1 * fx0, axis=1) ** 2
    denominator = fx0[:, 0] ** 2 + fx0[:, 1] ** 2 + ftx1[:, 0] ** 2 + ftx1[:, 1] ** 2
    squared = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.inf),
        where=denominator > 1e-15,
    )
    return np.sqrt(np.maximum(0.0, squared))


def triangulate_fixed_pose(
    k0: np.ndarray,
    r0: np.ndarray,
    t0: np.ndarray,
    k1: np.ndarray,
    r1: np.ndarray,
    t1: np.ndarray,
    xy0: np.ndarray,
    xy1: np.ndarray,
) -> dict[str, float | list[float]] | None:
    """Triangulate one match and return fixed-pose QA metrics."""
    projection0 = k0 @ np.column_stack((r0, t0))
    projection1 = k1 @ np.column_stack((r1, t1))
    rows = []
    for projection, xy in ((projection0, xy0), (projection1, xy1)):
        x, y = np.asarray(xy, dtype=np.float64)
        rows.extend((x * projection[2] - projection[0], y * projection[2] - projection[1]))
    _u, _s, vt = np.linalg.svd(np.asarray(rows))
    homogeneous = vt[-1]
    if abs(float(homogeneous[3])) <= 1e-12:
        return None
    xyz = homogeneous[:3] / homogeneous[3]
    depths = [float((r @ xyz + t)[2]) for r, t in ((r0, t0), (r1, t1))]
    if min(depths) <= 0 or not np.isfinite(xyz).all():
        return None

    errors = []
    for projection, xy in ((projection0, xy0), (projection1, xy1)):
        projected = projection @ np.r_[xyz, 1.0]
        if projected[2] <= 0:
            return None
        errors.append(float(np.linalg.norm(projected[:2] / projected[2] - xy)))
    center0 = -np.asarray(r0).T @ np.asarray(t0)
    center1 = -np.asarray(r1).T @ np.asarray(t1)
    ray0, ray1 = xyz - center0, xyz - center1
    cosine = float(np.dot(ray0, ray1) / (np.linalg.norm(ray0) * np.linalg.norm(ray1)))
    angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
    return {
        "xyz": xyz.tolist(),
        "maximum_reprojection_error_px": max(errors),
        "triangulation_angle_deg": angle,
    }


def select_nearest_pairs(
    centers: dict[str, np.ndarray],
    *,
    source_sequence: str,
    target_sequences: Iterable[str],
    source_frame_min: int,
    source_frame_max: int,
    pairs_per_source: int,
) -> list[tuple[str, str, float]]:
    if pairs_per_source <= 0 or source_frame_min > source_frame_max:
        raise ValueError("invalid pair-selection bounds")
    source_names = []
    for name in centers:
        if not name.startswith(f"{source_sequence}/"):
            continue
        try:
            frame = int(Path(name).stem)
        except ValueError:
            continue
        if source_frame_min <= frame <= source_frame_max:
            source_names.append(name)
    pairs = []
    for source_name in sorted(source_names):
        source_center = np.asarray(centers[source_name], dtype=np.float64)
        for target_sequence in sorted(set(target_sequences)):
            candidates = sorted(
                (
                    float(np.linalg.norm(source_center - np.asarray(center))),
                    name,
                )
                for name, center in centers.items()
                if name.startswith(f"{target_sequence}/")
            )[:pairs_per_source]
            pairs.extend(
                (source_name, target_name, round(distance, 12))
                for distance, target_name in candidates
            )
    return pairs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_observations(image: object) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        (np.asarray(point.xy, dtype=np.float64), int(point.point3D_id))
        for point in image.points2D
        if point.has_point3D()
    ]
    if not rows:
        return np.empty((0, 2)), np.empty(0, dtype=np.int64)
    return np.asarray([row[0] for row in rows]), np.asarray([row[1] for row in rows])


def _pose_and_k(reconstruction: object, image: object) -> tuple[np.ndarray, ...]:
    camera = reconstruction.cameras[int(image.camera_id)]
    if camera.model.name != "PINHOLE" or len(camera.params) != 4:
        raise ValueError("hotspot probe requires the fixed PINHOLE camera")
    rigid = image.cam_from_world()
    rotation = np.asarray(rigid.rotation.matrix(), dtype=np.float64)
    translation = np.asarray(rigid.translation, dtype=np.float64)
    k = camera_matrix(*map(float, camera.params))
    return k, rotation, translation


def _load_gray(path: Path, max_width: int) -> tuple[object, float, float]:
    import cv2
    import torch

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    height, width = image.shape
    output_width = min(width, max_width)
    output_height = max(8, int(round(height * output_width / width / 8)) * 8)
    resized = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float()[None, None] / 255.0
    return tensor, output_width / width, output_height / height


def _nearest_ids(
    model_xy: np.ndarray, model_ids: np.ndarray, query_xy: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    if not len(model_xy) or not len(query_xy):
        return np.full(len(query_xy), -1), np.full(len(query_xy), np.inf)
    distances, indexes = cKDTree(model_xy).query(query_xy, distance_upper_bound=radius)
    valid = np.isfinite(distances) & (indexes < len(model_ids))
    ids = np.full(len(query_xy), -1, dtype=np.int64)
    ids[valid] = model_ids[indexes[valid]]
    return ids, distances


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    # Official G6.1 scored pairs: P109/P110 ↔ P111 (not the frozen P114 notes).
    parser.add_argument("--source-sequence", default="P1110111")
    parser.add_argument("--target-sequence", action="append", default=None)
    parser.add_argument("--source-frame-min", type=int, default=29)
    parser.add_argument("--source-frame-max", type=int, default=45)
    parser.add_argument("--pairs-per-source", type=int, default=2)
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--confidence-min", type=float, default=0.5)
    parser.add_argument("--sampson-max-px", type=float, default=1.5)
    parser.add_argument("--association-radius-px", type=float, default=2.0)
    parser.add_argument("--reprojection-max-px", type=float, default=2.0)
    parser.add_argument("--triangulation-angle-min-deg", type=float, default=1.5)
    parser.add_argument("--ghost-distance-over-span", type=float, default=0.04)
    parser.add_argument("--minimum-multipair-tracks-per-edge", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from audit_map_geometry import apply_official_hotspot_defaults

    apply_official_hotspot_defaults(args)
    if not args.model.is_dir() or not args.image_root.is_dir() or not args.weights.is_file():
        raise SystemExit("model, image root, or pinned LoFTR weights are missing")

    import pycolmap
    import torch
    from kornia.feature import LoFTR

    reconstruction = pycolmap.Reconstruction(str(args.model))
    images = {image.name: image for image in reconstruction.images.values() if image.has_pose}
    centers = {
        name: np.asarray(image.projection_center(), dtype=np.float64)
        for name, image in images.items()
    }
    pairs = select_nearest_pairs(
        centers,
        source_sequence=args.source_sequence,
        target_sequences=args.target_sequence,
        source_frame_min=args.source_frame_min,
        source_frame_max=args.source_frame_max,
        pairs_per_source=args.pairs_per_source,
    )
    if not pairs:
        raise SystemExit("hotspot pair selection produced no registered pairs")

    all_centers = np.asarray(list(centers.values()))
    coordinate_median = np.median(all_centers, axis=0)
    map_span = 2.0 * float(
        np.percentile(np.linalg.norm(all_centers - coordinate_median, axis=1), 95)
    )
    image_sequences = {
        int(image_id): image.name.split("/", 1)[0]
        for image_id, image in reconstruction.images.items()
    }
    point_sequences = {
        int(point_id): {
            image_sequences[int(element.image_id)] for element in point.track.elements
        }
        for point_id, point in reconstruction.points3D.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LoFTR(pretrained=None)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    matcher.load_state_dict(checkpoint["state_dict"], strict=True)
    matcher = matcher.eval().to(device)

    pair_reports = []
    candidates: dict[str, dict[tuple[int, int], list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for pair_index, (source_name, target_name, center_distance) in enumerate(pairs, 1):
        source_image, target_image = images[source_name], images[target_name]
        input0, scale_x0, scale_y0 = _load_gray(
            args.image_root / source_name, args.max_width
        )
        input1, scale_x1, scale_y1 = _load_gray(
            args.image_root / target_name, args.max_width
        )
        with torch.inference_mode():
            raw = matcher({"image0": input0.to(device), "image1": input1.to(device)})
        xy0 = raw["keypoints0"].detach().cpu().numpy().astype(np.float64)
        xy1 = raw["keypoints1"].detach().cpu().numpy().astype(np.float64)
        confidence = raw["confidence"].detach().cpu().numpy().astype(np.float64)
        xy0 /= np.asarray([scale_x0, scale_y0])
        xy1 /= np.asarray([scale_x1, scale_y1])
        confidence_mask = confidence >= args.confidence_min

        k0, r0, t0 = _pose_and_k(reconstruction, source_image)
        k1, r1, t1 = _pose_and_k(reconstruction, target_image)
        fundamental = fundamental_from_poses(k0, r0, t0, k1, r1, t1)
        sampson = sampson_errors_px(fundamental, xy0, xy1)
        geometry_mask = confidence_mask & (sampson <= args.sampson_max_px)
        source_model_xy, source_model_ids = _model_observations(source_image)
        target_model_xy, target_model_ids = _model_observations(target_image)
        source_ids, source_assoc = _nearest_ids(
            source_model_xy, source_model_ids, xy0, args.association_radius_px
        )
        target_ids, target_assoc = _nearest_ids(
            target_model_xy, target_model_ids, xy1, args.association_radius_px
        )

        accepted = 0
        edge = "|".join(sorted((args.source_sequence, target_name.split("/", 1)[0])))
        for index in np.flatnonzero(geometry_mask):
            source_id, target_id = int(source_ids[index]), int(target_ids[index])
            if source_id < 0 or target_id < 0 or source_id == target_id:
                continue
            target_sequence = target_name.split("/", 1)[0]
            if point_sequences.get(source_id) != {args.source_sequence}:
                continue
            if point_sequences.get(target_id) != {target_sequence}:
                continue
            source_xyz = np.asarray(reconstruction.points3D[source_id].xyz)
            target_xyz = np.asarray(reconstruction.points3D[target_id].xyz)
            distance_over_span = float(np.linalg.norm(source_xyz - target_xyz) / map_span)
            if distance_over_span <= args.ghost_distance_over_span:
                continue
            triangulated = triangulate_fixed_pose(
                k0, r0, t0, k1, r1, t1, xy0[index], xy1[index]
            )
            if (
                triangulated is None
                or triangulated["maximum_reprojection_error_px"]
                > args.reprojection_max_px
                or triangulated["triangulation_angle_deg"]
                < args.triangulation_angle_min_deg
            ):
                continue
            candidates[edge][(source_id, target_id)].append(
                {
                    "source_image": source_name,
                    "target_image": target_name,
                    "confidence": float(confidence[index]),
                    "sampson_error_px": float(sampson[index]),
                    "source_association_px": float(source_assoc[index]),
                    "target_association_px": float(target_assoc[index]),
                    "existing_point_distance_over_S": distance_over_span,
                    **triangulated,
                }
            )
            accepted += 1
        pair_reports.append(
            {
                "pair_index": pair_index,
                "source": source_name,
                "target": target_name,
                "camera_center_distance": center_distance,
                "raw_matches": int(len(confidence)),
                "confidence_matches": int(confidence_mask.sum()),
                "fixed_pose_epipolar_matches": int(geometry_mask.sum()),
                "ghost_track_link_candidates": accepted,
            }
        )
        print(
            f"[{pair_index}/{len(pairs)}] {source_name} -> {target_name}: "
            f"raw={len(confidence)} epi={int(geometry_mask.sum())} links={accepted}",
            flush=True,
        )

    edge_reports = {}
    for edge in sorted(
        "|".join(sorted((args.source_sequence, target)))
        for target in args.target_sequence
    ):
        grouped = candidates.get(edge, {})
        unique_pairs = len(grouped)
        multipair = {
            pair: records
            for pair, records in grouped.items()
            if len({(r["source_image"], r["target_image"]) for r in records}) >= 2
        }
        edge_reports[edge] = {
            "status": (
                "DIAGNOSTIC_SUPPORT_FOUND"
                if len(multipair) >= args.minimum_multipair_tracks_per_edge
                else "NOT_PROVEN"
            ),
            "unique_candidate_track_pairs": unique_pairs,
            "multipair_supported_track_pairs": len(multipair),
            "support_histogram": dict(
                sorted(Counter(len(records) for records in grouped.values()).items())
            ),
            "examples": [
                {
                    "source_point3D_id": pair[0],
                    "target_point3D_id": pair[1],
                    "pair_support": len(records),
                    "records": records[:5],
                }
                for pair, records in sorted(
                    multipair.items(), key=lambda item: (-len(item[1]), item[0])
                )[:20]
            ],
        }
    status = (
        "DIAGNOSTIC_SUPPORT_FOUND"
        if edge_reports
        and all(
            item["status"] == "DIAGNOSTIC_SUPPORT_FOUND"
            for item in edge_reports.values()
        )
        else "NOT_PROVEN"
    )
    payload = {
        "schema_version": "fuhe-hotspot-loftr-exploratory-v2",
        "status": status,
        "promotion_allowed": False,
        "limitation": (
            "nearest-camera pairs are exploratory only; formal repair requires exact "
            "G6.1 hotspot extraction, greedy ROI coverage, and a full gate rerun"
        ),
        "read_only": True,
        "database_modified": False,
        "model_modified": False,
        "weights": {"path": str(args.weights.resolve()), "sha256": _sha256(args.weights)},
        "model": str(args.model.resolve()),
        "device": str(device),
        "map_span_S": map_span,
        "thresholds": {
            "confidence_min": args.confidence_min,
            "sampson_max_px": args.sampson_max_px,
            "association_radius_px": args.association_radius_px,
            "reprojection_max_px": args.reprojection_max_px,
            "triangulation_angle_min_deg": args.triangulation_angle_min_deg,
            "ghost_distance_over_span": args.ghost_distance_over_span,
            "minimum_multipair_tracks_per_edge": args.minimum_multipair_tracks_per_edge,
        },
        "pair_selection": {
            "policy": "nearest camera centers for each official G6.1 hotspot frame and target sequence",
            "source_frame_range": [args.source_frame_min, args.source_frame_max],
            "pairs_per_source_and_target": args.pairs_per_source,
            "pair_count": len(pairs),
        },
        "edge_reports": edge_reports,
        "pair_reports": pair_reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "edge_reports": edge_reports}, indent=2))


if __name__ == "__main__":
    main()
