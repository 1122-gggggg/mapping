#!/usr/bin/env python3
"""Run independent Sim3 and anti-ghost audits on the finalized Fuhe map."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from finalize_edm_model import EXPECTED_SEQUENCES, fixed_camera_evidence
from resource_guard import required_cli_path, run_global_heavy_job
from ts_common import Gate, read_fresh_gate_stage_metrics, required_check_ids


REQUIRED_GHOST_SEQUENCE_PAIRS = frozenset(
    {
        ("P1090109_002", "P1110111"),
        ("P1100110_005", "P1110111"),
    }
)
MIN_REQUIRED_ROUTE_SUPPORTED_POINTS_PER_SIDE = 500
MIN_REQUIRED_SUPPORTED_CELLS = 5
MIN_REQUIRED_ROUTE_SUPPORT_COVERAGE = 0.10
MAX_GHOST_P90_OVER_SPAN = 0.040


def _required_ghost_thresholds(maximum_p90_over_span: float) -> dict[str, float | int]:
    return {
        "minimum_left_route_supported_points": (
            MIN_REQUIRED_ROUTE_SUPPORTED_POINTS_PER_SIDE
        ),
        "minimum_right_route_supported_points": (
            MIN_REQUIRED_ROUTE_SUPPORTED_POINTS_PER_SIDE
        ),
        "minimum_supported_cells": MIN_REQUIRED_SUPPORTED_CELLS,
        "minimum_route_support_coverage_fraction": (
            MIN_REQUIRED_ROUTE_SUPPORT_COVERAGE
        ),
        "maximum_seq_nn_p90_over_S": maximum_p90_over_span,
    }


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


def single_epoch_gate(epochs: dict[str, str]) -> dict:
    """Make the cross-epoch predicate explicitly non-applicable for Fuhe."""
    unique = sorted(set(epochs.values()))
    if len(unique) <= 1:
        return {
            "applicable": False,
            "status": "NOT_APPLICABLE",
            "epochs": unique,
            "reason": "all Fuhe build sequences belong to one capture epoch",
        }
    return {
        "applicable": True,
        "status": "REQUIRES_EVALUATION",
        "epochs": unique,
        "reason": "multiple capture epochs require a separate overlap audit",
    }


def sequence_exclusive_point_clouds(
    points: dict,
    image_sequences: dict[int, str],
) -> dict[str, list[tuple[int, np.ndarray]]]:
    """Return finite points observed by exactly one sequence.

    A shared final-map point is one track, not evidence of two independently
    reconstructed surfaces, and is therefore excluded from the ghost audit.
    """
    clouds: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for point_id, point in sorted(points.items(), key=lambda item: int(item[0])):
        sequences = {
            image_sequences[int(element.image_id)] for element in point.track.elements
        }
        xyz = np.asarray(point.xyz, dtype=np.float64)
        if len(sequences) != 1 or xyz.shape != (3,) or not np.isfinite(xyz).all():
            continue
        sequence = next(iter(sequences))
        clouds[sequence].append((int(point_id), xyz))
    return {sequence: values for sequence, values in sorted(clouds.items())}


def _sample_voxel_cells(
    points: list[tuple[int, np.ndarray]],
    *,
    cell_size: float,
    max_points_per_cell: int,
) -> list[tuple[int, np.ndarray, tuple[int, int, int]]]:
    cells: dict[tuple[int, int, int], list[tuple[int, np.ndarray]]] = defaultdict(list)
    for point_id, raw_xyz in points:
        xyz = np.asarray(raw_xyz, dtype=np.float64)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            continue
        cell = tuple(int(value) for value in np.floor(xyz / cell_size))
        cells[cell].append((int(point_id), xyz))
    sampled = []
    for cell in sorted(cells):
        for point_id, xyz in sorted(cells[cell], key=lambda item: item[0])[
            :max_points_per_cell
        ]:
            sampled.append((point_id, xyz, cell))
    return sampled


def _directed_voxel_halo_distances(
    source_sequence: str,
    target_sequence: str,
    source: list[tuple[int, np.ndarray]],
    target: list[tuple[int, np.ndarray]],
    *,
    cell_size: float,
    max_points_per_cell: int,
) -> list[dict]:
    sampled_source = _sample_voxel_cells(
        source,
        cell_size=cell_size,
        max_points_per_cell=max_points_per_cell,
    )
    sampled_target = _sample_voxel_cells(
        target,
        cell_size=cell_size,
        max_points_per_cell=max_points_per_cell,
    )
    target_cells: dict[
        tuple[int, int, int], list[tuple[int, np.ndarray]]
    ] = defaultdict(list)
    for point_id, xyz, cell in sampled_target:
        target_cells[cell].append((point_id, xyz))
    offsets = tuple(product((-1, 0, 1), repeat=3))
    fallback_distance = 2.0 * cell_size
    records = []
    for point_id, xyz, cell in sampled_source:
        candidates = [
            candidate
            for offset in offsets
            for candidate in target_cells.get(
                tuple(cell[axis] + offset[axis] for axis in range(3)), []
            )
            if candidate[0] != point_id
        ]
        if candidates:
            nearest_id, nearest_xyz = min(
                candidates,
                key=lambda item: (float(np.linalg.norm(xyz - item[1])), item[0]),
            )
            distance = float(np.linalg.norm(xyz - nearest_xyz))
            nearest_value: int | None = nearest_id
            nearest_coordinates: list[float] | None = nearest_xyz.tolist()
        else:
            distance = fallback_distance
            nearest_value = None
            nearest_coordinates = None
        records.append(
            {
                "source_sequence": source_sequence,
                "target_sequence": target_sequence,
                "point3D_id": point_id,
                "point_xyz": xyz.tolist(),
                "nearest_point3D_id": nearest_value,
                "nearest_xyz": nearest_coordinates,
                "distance": distance,
            }
        )
    return records


def _bidirectionally_supported_cell_count(
    left: list[tuple[int, np.ndarray]],
    right: list[tuple[int, np.ndarray]],
    *,
    cell_size: float,
) -> int:
    """Count spatial cells that have opposite-sequence support in the 27-cell halo."""

    def occupied(points: list[tuple[int, np.ndarray]]) -> set[tuple[int, int, int]]:
        return {
            tuple(int(value) for value in np.floor(np.asarray(xyz) / cell_size))
            for _point_id, xyz in points
        }

    left_cells = occupied(left)
    right_cells = occupied(right)
    offsets = tuple(product((-1, 0, 1), repeat=3))

    def supported(
        source: set[tuple[int, int, int]],
        target: set[tuple[int, int, int]],
    ) -> int:
        return sum(
            any(
                tuple(cell[axis] + offset[axis] for axis in range(3)) in target
                for offset in offsets
            )
            for cell in source
        )

    return min(supported(left_cells, right_cells), supported(right_cells, left_cells))


def audit_sequence_exclusive_geometry(
    clouds: dict[str, list[tuple[int, np.ndarray]]],
    camera_centers: np.ndarray,
    *,
    maximum_p90_over_span: float = MAX_GHOST_P90_OVER_SPAN,
    max_points_per_cell: int = 8,
    camera_centers_by_sequence: dict[str, np.ndarray] | None = None,
    route_supported_edges: set[tuple[str, str]] | None = None,
    route_support_radius_over_span: float = 0.10,
    expected_sequences: set[str] | None = None,
) -> dict:
    """Score only sequence-exclusive points inside both routes' visibility mask."""
    map_span = robust_spatial_span(camera_centers)
    if (
        maximum_p90_over_span <= 0
        or max_points_per_cell <= 0
        or route_support_radius_over_span <= 0
    ):
        raise ValueError("ghost-audit thresholds must be positive")
    cell_size = maximum_p90_over_span * map_span
    support_radius = route_support_radius_over_span * map_span
    normalized_route_edges = (
        {tuple(sorted(edge)) for edge in route_supported_edges}
        if route_supported_edges is not None
        else None
    )
    expected = EXPECTED_SEQUENCES if expected_sequences is None else expected_sequences
    required_pairs = {
        edge for edge in REQUIRED_GHOST_SEQUENCE_PAIRS if set(edge) <= expected
    }
    required_pair_thresholds = _required_ghost_thresholds(maximum_p90_over_span)

    def route_masked_points(
        points: list[tuple[int, np.ndarray]], left: str, right: str
    ) -> list[tuple[int, np.ndarray]]:
        if camera_centers_by_sequence is None:
            return points
        left_route = np.asarray(camera_centers_by_sequence.get(left, []), dtype=np.float64)
        right_route = np.asarray(camera_centers_by_sequence.get(right, []), dtype=np.float64)
        if (
            left_route.ndim != 2
            or right_route.ndim != 2
            or left_route.shape[1:] != (3,)
            or right_route.shape[1:] != (3,)
            or not len(left_route)
            or not len(right_route)
        ):
            return []
        def distance_to_route(xyz: np.ndarray, route: np.ndarray) -> float:
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

        supported = []
        for point_id, raw_xyz in points:
            xyz = np.asarray(raw_xyz, dtype=np.float64)
            if (
                xyz.shape == (3,)
                and np.isfinite(xyz).all()
                and distance_to_route(xyz, left_route) <= support_radius
                and distance_to_route(xyz, right_route) <= support_radius
            ):
                supported.append((point_id, xyz))
        return supported

    summaries: dict[str, dict] = {}
    all_records = []
    for left, right in combinations(sorted(clouds), 2):
        edge_tuple = tuple(sorted((left, right)))
        left_supported = route_masked_points(clouds[left], left, right)
        right_supported = route_masked_points(clouds[right], left, right)
        denominator = len(clouds[left]) + len(clouds[right])
        supported_count = len(left_supported) + len(right_supported)
        unsupported_count = denominator - supported_count
        coverage = supported_count / max(1, denominator)
        supported_cells = _bidirectionally_supported_cell_count(
            left_supported,
            right_supported,
            cell_size=cell_size,
        )
        edge = f"{left}|{right}"
        required = edge_tuple in required_pairs
        common_evidence = {
            "left_exclusive_points": len(clouds[left]),
            "right_exclusive_points": len(clouds[right]),
            "left_route_supported_points": len(left_supported),
            "right_route_supported_points": len(right_supported),
            "coverage_denominator_exclusive_points": denominator,
            "route_supported_exclusive_points": supported_count,
            "unsupported_exclusive_points": unsupported_count,
            "route_support_coverage_fraction": coverage,
            "supported_cells": supported_cells,
        }
        if required:
            common_evidence["required_pair_thresholds"] = required_pair_thresholds
        if normalized_route_edges is not None and edge_tuple not in normalized_route_edges:
            summaries[edge] = {
                "applicable": required,
                "status": "FAIL" if required else "NOT_APPLICABLE",
                "reason": (
                    "required Fuhe cross-direction pair lacks a trusted route edge"
                    if required
                    else "sequence pair is not backed by a trusted route edge"
                ),
                **common_evidence,
            }
            continue
        if not left_supported or not right_supported:
            summaries[edge] = {
                "applicable": required,
                "status": "FAIL" if required else "NOT_APPLICABLE",
                "reason": (
                    "required Fuhe pair has no bidirectional exclusive support in common route visibility"
                    if required
                    else "no bidirectional sequence-exclusive support in common route visibility"
                ),
                **common_evidence,
            }
            continue
        records = _directed_voxel_halo_distances(
            left,
            right,
            left_supported,
            right_supported,
            cell_size=cell_size,
            max_points_per_cell=max_points_per_cell,
        ) + _directed_voxel_halo_distances(
            right,
            left,
            right_supported,
            left_supported,
            cell_size=cell_size,
            max_points_per_cell=max_points_per_cell,
        )
        distances = np.asarray([record["distance"] for record in records])
        p90 = float(np.percentile(distances, 90)) if len(distances) else map_span
        ratio = p90 / map_span
        distance_gate_pass = ratio <= maximum_p90_over_span
        support_checks = {
            "left_route_supported_points": len(left_supported)
            >= MIN_REQUIRED_ROUTE_SUPPORTED_POINTS_PER_SIDE,
            "right_route_supported_points": len(right_supported)
            >= MIN_REQUIRED_ROUTE_SUPPORTED_POINTS_PER_SIDE,
            "supported_cells": supported_cells >= MIN_REQUIRED_SUPPORTED_CELLS,
            "route_support_coverage_fraction": coverage
            >= MIN_REQUIRED_ROUTE_SUPPORT_COVERAGE,
        }
        support_gate_pass = all(support_checks.values()) if required else True
        pair_pass = distance_gate_pass and support_gate_pass
        summaries[edge] = {
            "applicable": True,
            **common_evidence,
            "sampled_directed_points": len(records),
            "seq_nn_p90": p90,
            "seq_nn_p90_over_S": ratio,
            "maximum": maximum_p90_over_span,
            "distance_gate_pass": distance_gate_pass,
            "required_support_checks": support_checks if required else {},
            "required_support_gate_pass": support_gate_pass,
            "status": "PASS" if pair_pass else "FAIL",
        }
        for record in records:
            record["distance_over_S"] = record["distance"] / map_span
            record["sequence_pair"] = edge
        all_records.extend(records)
    worst_ratio = max(
        (
            summary["seq_nn_p90_over_S"]
            for summary in summaries.values()
            if summary.get("applicable") is True
            and "seq_nn_p90_over_S" in summary
        ),
        default=None,
    )
    hotspots = sorted(
        all_records,
        key=lambda record: (
            -record["distance_over_S"],
            record["sequence_pair"],
            record["source_sequence"],
            record["point3D_id"],
        ),
    )[:10]
    p110_p111 = "P1100110_005|P1110111"
    p109_p111 = "P1090109_002|P1110111"
    p114_edges = {
        edge: summary
        for edge, summary in summaries.items()
        if "P1140114" in edge.split("|")
    }
    required_diagnostics = {
        p110_p111: summaries.get(p110_p111, {"status": "MISSING"}),
        p109_p111: summaries.get(p109_p111, {"status": "MISSING"}),
        "P1140114_articulation": {
            "sequence": "P1140114",
            "edges": p114_edges,
            "worst_seq_nn_p90_over_S": max(
                (
                    summary["seq_nn_p90_over_S"]
                    for summary in p114_edges.values()
                    if summary.get("applicable") is True
                    and "seq_nn_p90_over_S" in summary
                ),
                default=None,
            ),
        },
    }
    missing_sequences = sorted(expected - set(clouds))
    unexpected_sequences = sorted(set(clouds) - expected)
    applicable_summaries = [
        summary for summary in summaries.values() if summary.get("applicable") is True
    ]
    failed_applicable = any(
        summary.get("status") == "FAIL" for summary in applicable_summaries
    )
    required_pair_pass = all(
        summaries.get("|".join(edge), {}).get("status") == "PASS"
        for edge in required_pairs
    )
    if (
        missing_sequences
        or unexpected_sequences
        or failed_applicable
        or not required_pair_pass
    ):
        status = "FAIL"
    elif applicable_summaries:
        status = "PASS"
    else:
        status = "NOT_APPLICABLE"
    return {
        "status": status,
        "map_scale_S": map_span,
        "scale_definition": "2*p95(camera-center distance from coordinate median)",
        "voxel_cell_size": cell_size,
        "voxel_cell_size_over_S": maximum_p90_over_span,
        "halo_neighbors": 27,
        "max_points_per_cell": max_points_per_cell,
        "route_support_radius": support_radius,
        "route_support_radius_over_S": route_support_radius_over_span,
        "overlap_mask_policy": (
            "point must lie within route-support radius of both sequence trajectories"
        ),
        "shared_track_policy": "exclude; only exactly-one-sequence tracks are audited",
        "required_pairs": ["|".join(edge) for edge in sorted(required_pairs)],
        "required_pair_thresholds": required_pair_thresholds,
        "required_pairs_pass": required_pair_pass,
        "sequence_pairs": summaries,
        "missing_sequences": missing_sequences,
        "unexpected_sequences": unexpected_sequences,
        "worst_seq_nn_p90_over_S": worst_ratio,
        "maximum_worst_seq_nn_p90_over_S": maximum_p90_over_span,
        "worst10_hotspots": hotspots,
        "required_diagnostics": required_diagnostics,
    }


def trajectory_jump_audit(
    camera_centers_by_sequence: dict[str, np.ndarray],
) -> dict:
    """Require zero consecutive camera steps above ten times route median."""
    evidence: dict[str, dict] = {}
    total_jumps = 0
    for sequence, raw_centers in sorted(camera_centers_by_sequence.items()):
        centers = np.asarray(raw_centers, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1:] != (3,) or len(centers) < 2:
            evidence[sequence] = {
                "status": "NOT_APPLICABLE",
                "applicable": False,
                "reason": "fewer than two registered cameras",
                "jump_indices": [],
            }
            continue
        steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        median_step = float(np.median(steps))
        threshold = 10.0 * median_step
        jump_indices = [
            int(index)
            for index, step in enumerate(steps)
            if float(step) > threshold
        ]
        total_jumps += len(jump_indices)
        evidence[sequence] = {
            "status": "PASS" if not jump_indices else "FAIL",
            "applicable": True,
            "n_steps": len(steps),
            "median_step": median_step,
            "jump_threshold": threshold,
            "maximum_step": float(np.max(steps)),
            "jump_indices": jump_indices,
        }
    return {
        "status": "PASS" if total_jumps == 0 else "FAIL",
        "maximum_jump_multiple": 10.0,
        "total_jumps_over_10x_median": total_jumps,
        "sequences": evidence,
    }


def route_cluster_audit(independent_sim3: dict) -> dict:
    """Revalidate two separated support clusters on at least two trusted routes."""
    trusted = independent_sim3.get("trusted_independent_edges", [])
    edges = independent_sim3.get("edges", {})
    errors = []
    for edge in trusted:
        payload = edges.get(edge, {})
        if (
            payload.get("route_cluster_count") != 2
            or payload.get("route_cluster_minimum_normalized_separation") != 0.25
            or len(payload.get("cluster_sizes", [])) != 2
            or min(payload.get("cluster_sizes", [0])) < 3
        ):
            errors.append(edge)
    passed = len(trusted) >= 2 and not errors
    return {
        "status": "PASS" if passed else "FAIL",
        "trusted_edges": trusted,
        "minimum_trusted_edges": 2,
        "required_clusters_per_edge": 2,
        "minimum_normalized_separation": 0.25,
        "invalid_edges": errors,
    }


def required_ghost_pairs_pass(ghost_geometry: dict) -> bool:
    """Recompute required Fuhe route support and ghost-distance predicates."""
    if ghost_geometry.get("status") != "PASS":
        return False
    summaries = ghost_geometry.get("sequence_pairs")
    if not isinstance(summaries, dict):
        return False
    thresholds = ghost_geometry.get("required_pair_thresholds")
    if not isinstance(thresholds, dict):
        return False
    try:
        thresholds_valid = (
            thresholds.get("minimum_left_route_supported_points")
            == MIN_REQUIRED_ROUTE_SUPPORTED_POINTS_PER_SIDE
            and thresholds.get("minimum_right_route_supported_points")
            == MIN_REQUIRED_ROUTE_SUPPORTED_POINTS_PER_SIDE
            and thresholds.get("minimum_supported_cells")
            == MIN_REQUIRED_SUPPORTED_CELLS
            and float(thresholds.get("minimum_route_support_coverage_fraction"))
            == MIN_REQUIRED_ROUTE_SUPPORT_COVERAGE
            and 0
            < float(thresholds.get("maximum_seq_nn_p90_over_S"))
            <= MAX_GHOST_P90_OVER_SPAN
        )
    except (TypeError, ValueError):
        return False
    if not thresholds_valid:
        return False

    for edge in REQUIRED_GHOST_SEQUENCE_PAIRS:
        summary = summaries.get("|".join(edge))
        if not isinstance(summary, dict):
            return False
        if summary.get("required_pair_thresholds") != thresholds:
            return False
        try:
            support_valid = (
                int(summary["left_route_supported_points"])
                >= MIN_REQUIRED_ROUTE_SUPPORTED_POINTS_PER_SIDE
                and int(summary["right_route_supported_points"])
                >= MIN_REQUIRED_ROUTE_SUPPORTED_POINTS_PER_SIDE
                and int(summary["supported_cells"])
                >= MIN_REQUIRED_SUPPORTED_CELLS
                and float(summary["route_support_coverage_fraction"])
                >= MIN_REQUIRED_ROUTE_SUPPORT_COVERAGE
            )
            ratio = float(summary["seq_nn_p90_over_S"])
            recorded_maximum = float(summary["maximum"])
            distance_valid = (
                math.isfinite(ratio)
                and math.isfinite(recorded_maximum)
                and ratio <= recorded_maximum
                and ratio <= MAX_GHOST_P90_OVER_SPAN
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if (
            summary.get("status") != "PASS"
            or summary.get("applicable") is not True
            or not support_valid
            or not distance_valid
        ):
            return False

    required_keys = {"|".join(edge) for edge in REQUIRED_GHOST_SEQUENCE_PAIRS}
    for edge, summary in summaries.items():
        if edge in required_keys:
            continue
        if not isinstance(summary, dict):
            return False
        state = summary.get("status")
        if state == "PASS" and summary.get("applicable") is True:
            continue
        if (
            state == "NOT_APPLICABLE"
            and summary.get("applicable") is False
            and isinstance(summary.get("reason"), str)
            and bool(summary["reason"].strip())
        ):
            continue
        return False
    return True


def formal_geometry_checks(
    *,
    sim3_pass: bool,
    ghost_geometry: dict,
    direction_overlap_normalized: dict[str, float],
    route_cluster_evidence: dict,
    trajectory_evidence: dict,
) -> dict[str, bool]:
    """Build the exact formal S5.7/S6 predicate set."""
    return {
        "G5.7": sim3_pass is True,
        "G6.1": required_ghost_pairs_pass(ghost_geometry),
        "G6.2": float(direction_overlap_normalized.get("symmetric_median", math.inf))
        <= 0.05
        and float(direction_overlap_normalized.get("symmetric_p90", math.inf))
        <= 0.15,
        "G6.3": route_cluster_evidence.get("status") == "PASS",
        "G6.4": trajectory_evidence.get("total_jumps_over_10x_median") == 0,
    }


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


def _main_locked() -> None:
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

    s4 = read_fresh_gate_stage_metrics(
        args.s4_gate, expected_stage="S4_doppelgangers"
    )
    s5 = read_fresh_gate_stage_metrics(
        args.s5_metrics, expected_stage="S5_fixed_intrinsics"
    )
    independent_sim3 = read_fresh_gate_stage_metrics(
        args.s5_7_gate, expected_stage="S5_7_independent_sim3"
    )

    import pycolmap
    import torch

    rec = pycolmap.Reconstruction(str(args.model))
    camera_evidence = fixed_camera_evidence(rec)
    if not camera_evidence["ok"]:
        raise SystemExit("geometry audit requires the single fixed Fuhe v2 camera")
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
    image_sequences = {
        int(image.image_id): str(image.name).split("/", 1)[0]
        for image in rec.images.values()
    }
    exclusive_clouds = sequence_exclusive_point_clouds(
        rec.points3D, image_sequences
    )
    camera_centers_by_sequence = {
        sequence: np.asarray([centers[name] for name in sequence_names])
        for sequence, sequence_names in by_sequence.items()
    }
    trusted_route_edges = {
        tuple(sorted(edge.split("|", 1)))
        for edge in independent_sim3.get("trusted_independent_edges", [])
    }
    ghost_geometry = audit_sequence_exclusive_geometry(
        exclusive_clouds,
        all_centers,
        camera_centers_by_sequence=camera_centers_by_sequence,
        route_supported_edges=trusted_route_edges,
    )
    epoch_gate = single_epoch_gate(epochs)
    trajectory_evidence = trajectory_jump_audit(camera_centers_by_sequence)
    route_cluster_evidence = route_cluster_audit(independent_sim3)

    global_error = float(s5["mean_reprojection_error_px"])
    per_sequence = s5["per_sequence_mean_reprojection_error_px"]
    reprojection_pass = set(per_sequence) == set(directions) and all(
        float(error) <= 1.5 * global_error for error in per_sequence.values()
    )
    checks = formal_geometry_checks(
        sim3_pass=sim3_pass,
        ghost_geometry=ghost_geometry,
        direction_overlap_normalized=direction_overlap_normalized,
        route_cluster_evidence=route_cluster_evidence,
        trajectory_evidence=trajectory_evidence,
    )
    result = {
        "stage": "S5.7_S6_geometry_audit",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "map_span": map_span,
        "fixed_camera": camera_evidence,
        "sim3_evidence": sim3_evidence,
        "sim3_method": independent_sim3.get("method"),
        "sim3_gate": str(args.s5_7_gate.resolve()),
        "geometric_pair_counts": geometric_pair_counts,
        "direction_overlap_normalized": direction_overlap_normalized,
        "direction_overlap_thresholds": {
            "maximum_symmetric_median_over_S": 0.05,
            "maximum_symmetric_p90_over_S": 0.15,
        },
        "sequence_exclusive_ghost_geometry": ghost_geometry,
        "route_cluster_evidence": route_cluster_evidence,
        "camera_trajectory_jump_evidence": trajectory_evidence,
        "shared_direction_points": shared_direction_points,
        "shared_direction_point_fraction": shared_fraction,
        "per_sequence_mean_reprojection_error_px": per_sequence,
        "global_mean_reprojection_error_px": global_error,
        "reprojection_diagnostic_pass": reprojection_pass,
        "epoch_gate": epoch_gate,
    }
    gate = Gate(
        "S5_7_S6_geometry",
        required_check_ids("S5_7_S6_geometry"),
        script_path=__file__,
        source_files=[
            Path(__file__).with_name("audit_independent_sim3.py"),
            Path(__file__).with_name("finalize_edm_model.py"),
            Path(__file__).with_name("resource_guard.py"),
            Path(__file__).with_name("ts_common.py"),
        ],
        input_artifacts={
            "final_model": args.model,
            "image_root": args.image_root,
            "frame_manifest": args.frame_manifest,
            "corpus_manifest": args.corpus_manifest,
            "conditional_pairs": args.forced_pairs,
            "conditional_pair_manifest": args.forced_manifest,
            "two_view_scores": args.twoview,
            "S4_gate": args.s4_gate,
            "S5_gate": args.s5_metrics,
        },
    )
    gate.record_predecessor_gate(
        "S5_7_independent_sim3",
        args.s5_7_gate,
        expected_stage="S5_7_independent_sim3",
    )
    for gid, passed in checks.items():
        gate.check(
            gid,
            passed,
            "formal final-map geometry predicate recomputed",
            direction_overlap_normalized=direction_overlap_normalized,
            route_cluster_evidence=route_cluster_evidence,
            trajectory_evidence=trajectory_evidence,
            ghost_status=ghost_geometry["status"],
        )
    payload = gate.write(
        args.out.parent,
        output_path=args.out,
        stage_metrics=result,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    arguments = sys.argv[1:]
    if "--help" in arguments or "-h" in arguments:
        _main_locked()
        return
    disk_path = required_cli_path(arguments, "--out")
    run_global_heavy_job(disk_path, lambda _evidence: _main_locked())


if __name__ == "__main__":
    main()
