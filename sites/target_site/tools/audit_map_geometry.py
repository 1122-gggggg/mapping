#!/usr/bin/env python3
"""Run independent Sim3 and anti-ghost audits on the finalized target map."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def estimate_sim3(source: np.ndarray, target: np.ndarray) -> dict:
    """Estimate target = scale * rotation @ source + translation (Umeyama)."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape (N, 3)")
    if len(source) < 3:
        raise ValueError("at least three correspondences are required")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if variance <= 1e-15:
        raise ValueError("source correspondences are geometrically degenerate")
    covariance = target_centered.T @ source_centered / len(source)
    left, singular, right_transposed = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(left @ right_transposed) < 0:
        sign[-1] = -1.0
    rotation = left @ np.diag(sign) @ right_transposed
    scale = float(np.sum(singular * sign) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    prediction = scale * (source @ rotation.T) + translation
    rmse = float(np.sqrt(np.mean(np.sum((prediction - target) ** 2, axis=1))))
    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "rmse": rmse,
    }


def nearest_neighbor_summary(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if not len(left) or not len(right):
        raise ValueError("nearest-neighbor sets must be non-empty")
    left_distances = cKDTree(right).query(left, k=1)[0]
    right_distances = cKDTree(left).query(right, k=1)[0]
    combined = np.concatenate((left_distances, right_distances))
    return {
        "left_to_right_median": float(np.median(left_distances)),
        "right_to_left_median": float(np.median(right_distances)),
        "symmetric_median": float(np.median(combined)),
        "symmetric_p90": float(np.percentile(combined, 90)),
    }


def robust_spatial_span(points: np.ndarray) -> float:
    """Return a rotation-invariant map scale resistant to sparse pose outliers."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("spatial span requires at least two 3D points")
    if not np.isfinite(values).all():
        raise ValueError("spatial span points must be finite")
    center = np.median(values, axis=0)
    radius = float(np.percentile(np.linalg.norm(values - center, axis=1), 95))
    span = 2.0 * radius
    if span <= 1e-12:
        raise ValueError("degenerate robust spatial span")
    return span


def filter_geometric_pairs(
    pairs: list[tuple[str, str]],
    observed_points: dict[str, set[int]],
    *,
    minimum_shared: int,
) -> list[tuple[str, str]]:
    """Keep only pairs that truly share final-map 3D tracks."""
    return [
        (left, right)
        for left, right in pairs
        if len(observed_points.get(left, set()) & observed_points.get(right, set()))
        >= minimum_shared
    ]


def _json_sim3(sim3: dict) -> dict:
    return {
        "scale": float(sim3["scale"]),
        "rotation": np.asarray(sim3["rotation"]).tolist(),
        "translation": np.asarray(sim3["translation"]).tolist(),
        "rmse": float(sim3["rmse"]),
    }


def _two_separated_clusters(
    pairs: list[tuple[str, str]], positions: dict[str, float]
) -> list[list[tuple[str, str]]]:
    coordinates = np.asarray(
        [[positions[left], positions[right]] for left, right in pairs], dtype=np.float64
    )
    if len(coordinates) < 6:
        raise ValueError("fewer than six bridge correspondences")
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
    first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
    if distances[first, second] < 0.25:
        raise ValueError("bridge correspondences are not spatially separated")
    seed_distances = np.column_stack((distances[:, first], distances[:, second]))
    labels = np.argmin(seed_distances, axis=1)
    clusters = [[pair for pair, label in zip(pairs, labels, strict=True) if label == index]
                for index in (0, 1)]
    if min(map(len, clusters)) < 3:
        raise ValueError("a separated bridge cluster has fewer than three pairs")
    return clusters


def _sim3_audit(
    robust_edges: list[list[str]],
    accepted_forced: dict[tuple[str, str], list[tuple[str, str]]],
    positions: dict[str, float],
    centers: dict[str, np.ndarray],
    map_span: float,
) -> tuple[dict, bool]:
    evidence = {}
    all_pass = True
    for raw_edge in robust_edges:
        edge = tuple(sorted(raw_edge))
        pairs = [
            pair
            for pair in accepted_forced.get(edge, [])
            if pair[0] in centers and pair[1] in centers
        ]
        try:
            clusters = _two_separated_clusters(pairs, positions)
            estimates = []
            for cluster in clusters:
                source = np.asarray([centers[left] for left, _ in cluster])
                target = np.asarray([centers[right] for _, right in cluster])
                estimates.append(estimate_sim3(source, target))
            scale_log_delta = abs(math.log(estimates[0]["scale"] / estimates[1]["scale"]))
            rotation_delta = math.degrees(
                (Rotation.from_matrix(estimates[0]["rotation"]).inv()
                 * Rotation.from_matrix(estimates[1]["rotation"])).magnitude()
            )
            translation_delta = float(
                np.linalg.norm(estimates[0]["translation"] - estimates[1]["translation"])
                / map_span
            )
            normalized_rmse = [float(item["rmse"] / map_span) for item in estimates]
            passed = (
                scale_log_delta <= 0.15
                and rotation_delta <= 15.0
                and translation_delta <= 0.08
                and max(normalized_rmse) <= 0.05
            )
            evidence["|".join(edge)] = {
                "status": "PASS" if passed else "FAIL",
                "cluster_sizes": [len(cluster) for cluster in clusters],
                "estimates": [_json_sim3(item) for item in estimates],
                "scale_log_delta": scale_log_delta,
                "rotation_delta_deg": rotation_delta,
                "translation_delta_over_span": translation_delta,
                "normalized_rmse": normalized_rmse,
            }
        except (ValueError, np.linalg.LinAlgError) as error:
            passed = False
            evidence["|".join(edge)] = {"status": "FAIL", "error": str(error)}
        all_pass &= passed
    return evidence, all_pass and bool(robust_edges)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--forced-pairs", type=Path, required=True)
    parser.add_argument("--forced-manifest", type=Path, required=True)
    parser.add_argument("--twoview", type=Path, required=True)
    parser.add_argument("--s4-gate", type=Path, required=True)
    parser.add_argument("--s5-metrics", type=Path, required=True)
    parser.add_argument("--s5-7-gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    import pycolmap
    import torch

    rec = pycolmap.Reconstruction(str(args.model))
    images = {
        image.name: image for image in rec.images.values() if image.has_pose
    }
    centers = {
        name: np.asarray(image.projection_center(), dtype=np.float64)
        for name, image in images.items()
    }
    all_centers = np.asarray(list(centers.values()))
    map_span = robust_spatial_span(all_centers)

    forced_manifest = json.loads(args.forced_manifest.read_text(encoding="utf-8"))
    directions = {
        **{name: "fwd" for name in forced_manifest["fwd"]},
        **{name: "rev" for name in forced_manifest["rev"]},
    }
    corpus = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
    epochs = {item["seq"]: item["epoch"] for item in corpus["build"]}
    by_sequence: dict[str, list[str]] = defaultdict(list)
    for name in sorted(centers):
        by_sequence[name.split("/", 1)[0]].append(name)
    positions = {}
    for names in by_sequence.values():
        denominator = max(1, len(names) - 1)
        positions.update({name: index / denominator for index, name in enumerate(names)})

    names = sorted(
        path.relative_to(args.image_root).as_posix()
        for path in args.image_root.glob("*/*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    two_view = torch.load(args.twoview, map_location="cpu", weights_only=False)
    pair_indexes = np.asarray(two_view["pairs"], dtype=np.int64)
    scores = np.asarray(two_view["scores"], dtype=np.float64)
    forced_lines = {
        tuple(sorted(fields))
        for line in args.forced_pairs.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
        for fields in [line.split()]
        if len(fields) == 2
    }
    accepted_forced: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for (left_index, right_index), score in zip(pair_indexes, scores, strict=True):
        if score <= args.threshold:
            continue
        left, right = names[int(left_index)], names[int(right_index)]
        if tuple(sorted((left, right))) not in forced_lines:
            continue
        left_sequence, right_sequence = left.split("/", 1)[0], right.split("/", 1)[0]
        if directions[left_sequence] == "rev":
            left, right = right, left
            left_sequence, right_sequence = right_sequence, left_sequence
        if directions[left_sequence] != directions[right_sequence]:
            accepted_forced[tuple(sorted((left_sequence, right_sequence)))].append((left, right))

    observed_points = {
        name: {
            int(point.point3D_id)
            for point in image.points2D
            if point.has_point3D()
        }
        for name, image in images.items()
    }
    geometric_forced = {
        edge: filter_geometric_pairs(pairs, observed_points, minimum_shared=5)
        for edge, pairs in accepted_forced.items()
    }
    geometric_pair_counts = {
        "|".join(edge): {
            "accepted_forced": len(accepted_forced[edge]),
            "shared_final_geometry": len(geometric_forced[edge]),
        }
        for edge in sorted(accepted_forced)
    }

    s4 = json.loads(args.s4_gate.read_text(encoding="utf-8"))
    independent_sim3 = json.loads(args.s5_7_gate.read_text(encoding="utf-8"))
    sim3_evidence = independent_sim3.get("edges", {})
    expected_sim3_edges = {
        "|".join(sorted(edge)) for edge in s4["robust_cross_direction_edges"]
    }
    sim3_pass = (
        independent_sim3.get("status") == "PASS"
        and set(sim3_evidence) == expected_sim3_edges
    )

    direction_centers = {
        direction: np.asarray(
            [centers[name] for sequence, names_in_sequence in by_sequence.items()
             if directions[sequence] == direction for name in names_in_sequence]
        )
        for direction in ("fwd", "rev")
    }
    direction_overlap = nearest_neighbor_summary(
        direction_centers["fwd"], direction_centers["rev"]
    )
    direction_overlap_normalized = {
        key: value / map_span for key, value in direction_overlap.items()
    }

    shared_direction_points = 0
    for point in rec.points3D.values():
        observed_directions = {
            directions[rec.images[int(element.image_id)].name.split("/", 1)[0]]
            for element in point.track.elements
        }
        shared_direction_points += observed_directions == {"fwd", "rev"}
    shared_fraction = shared_direction_points / max(1, rec.num_points3D())

    epoch_centers: dict[str, list[np.ndarray]] = defaultdict(list)
    for sequence, sequence_names in by_sequence.items():
        epoch_centers[epochs[sequence]].extend(centers[name] for name in sequence_names)
    epoch_overlap = {}
    epoch_pass = True
    epoch_names = sorted(epoch_centers)
    for index, left_epoch in enumerate(epoch_names):
        for right_epoch in epoch_names[index + 1 :]:
            summary = nearest_neighbor_summary(
                np.asarray(epoch_centers[left_epoch]), np.asarray(epoch_centers[right_epoch])
            )
            normalized = {key: value / map_span for key, value in summary.items()}
            # A shorter historical traversal may cover only a subset of a later flight.
            overlap_median = min(
                normalized["left_to_right_median"], normalized["right_to_left_median"]
            )
            passed = overlap_median <= 0.05
            epoch_pass &= passed
            epoch_overlap[f"{left_epoch}|{right_epoch}"] = {
                **normalized,
                "overlap_subset_median": overlap_median,
                "status": "PASS" if passed else "FAIL",
            }

    s5 = json.loads(args.s5_metrics.read_text(encoding="utf-8"))
    global_error = float(s5["mean_reprojection_error_px"])
    per_sequence = s5["per_sequence_mean_reprojection_error_px"]
    reprojection_pass = set(per_sequence) == set(directions) and all(
        float(error) <= 1.5 * global_error for error in per_sequence.values()
    )
    checks = {
        "G5.7": sim3_pass,
        "G6.1": direction_overlap_normalized["symmetric_median"] <= 0.05
        and direction_overlap_normalized["symmetric_p90"] <= 0.15,
        "G6.2": shared_direction_points >= 100 and shared_fraction >= 0.01,
        "G6.3": reprojection_pass,
        "G6.4": epoch_pass and len(epoch_overlap) == 3,
    }
    result = {
        "stage": "S5.7_S6_geometry_audit",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "map_span": map_span,
        "sim3_evidence": sim3_evidence,
        "sim3_method": independent_sim3.get("method"),
        "sim3_gate": str(args.s5_7_gate.resolve()),
        "geometric_pair_counts": geometric_pair_counts,
        "direction_overlap_normalized": direction_overlap_normalized,
        "shared_direction_points": shared_direction_points,
        "shared_direction_point_fraction": shared_fraction,
        "per_sequence_mean_reprojection_error_px": per_sequence,
        "global_mean_reprojection_error_px": global_error,
        "epoch_overlap_normalized": epoch_overlap,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if result["status"] != "PASS":
        raise SystemExit("S5.7/S6 gate failed")


if __name__ == "__main__":
    main()
