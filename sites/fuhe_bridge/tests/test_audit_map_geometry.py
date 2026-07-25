from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from audit_map_geometry import (  # noqa: E402
    audit_sequence_exclusive_geometry,
    estimate_sim3,
    filter_geometric_pairs,
    formal_geometry_checks,
    nearest_neighbor_summary,
    robust_spatial_span,
    sequence_exclusive_point_clouds,
    trajectory_jump_audit,
)


def test_estimate_sim3_recovers_known_transform() -> None:
    source = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = 2.5 * (source @ rotation.T) + np.asarray([3.0, -2.0, 1.0])

    result = estimate_sim3(source, target)

    assert np.isclose(result["scale"], 2.5)
    assert np.allclose(result["rotation"], rotation)
    assert np.allclose(result["translation"], [3.0, -2.0, 1.0])
    assert result["rmse"] < 1e-12


def test_nearest_neighbor_summary_is_symmetric() -> None:
    left = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    right = np.asarray([[0.0, 0.0, 0.0]])

    result = nearest_neighbor_summary(left, right)

    assert result["left_to_right_median"] == 5.0
    assert result["right_to_left_median"] == 0.0
    assert result["symmetric_median"] == 0.0


def test_geometric_pair_filter_requires_shared_final_tracks() -> None:
    pairs = [("a", "b"), ("a", "c")]
    observations = {
        "a": {1, 2, 3, 4, 5},
        "b": {1, 2, 3, 4, 5, 6},
        "c": {1, 2},
    }

    assert filter_geometric_pairs(pairs, observations, minimum_shared=5) == [
        ("a", "b")
    ]


def test_robust_spatial_span_ignores_single_catastrophic_pose() -> None:
    centers = np.column_stack(
        (np.linspace(0.0, 10.0, 101), np.zeros(101), np.zeros(101))
    )
    centers[-1] = [1e9, -1e9, 1e9]

    span = robust_spatial_span(centers)

    assert 8.0 < span < 30.0


def test_sequence_exclusive_clouds_remove_every_shared_track() -> None:
    points = {
        1: SimpleNamespace(
            xyz=np.asarray([0.0, 0.0, 0.0]),
            track=SimpleNamespace(elements=[SimpleNamespace(image_id=1)]),
        ),
        2: SimpleNamespace(
            xyz=np.asarray([1.0, 0.0, 0.0]),
            track=SimpleNamespace(
                elements=[SimpleNamespace(image_id=1), SimpleNamespace(image_id=2)]
            ),
        ),
        3: SimpleNamespace(
            xyz=np.asarray([0.1, 0.0, 0.0]),
            track=SimpleNamespace(elements=[SimpleNamespace(image_id=2)]),
        ),
    }

    clouds = sequence_exclusive_point_clouds(points, {1: "A", 2: "B"})

    assert [point_id for point_id, _xyz in clouds["A"]] == [1]
    assert [point_id for point_id, _xyz in clouds["B"]] == [3]


def test_voxel_ghost_audit_is_deterministic_and_reports_fuhe_hotspots() -> None:
    sequences = (
        "P1090109_002",
        "P1100110_005",
        "P1110111",
        "P1120112",
        "P1140114",
    )
    base = np.column_stack(
        (np.linspace(0.0, 9.0, 500), np.zeros(500), np.zeros(500))
    )
    clouds = {
        sequence: [
            (offset * 1000 + index, xyz + np.asarray([0.0, offset * 0.01, 0.0]))
            for index, xyz in enumerate(base)
        ]
        for offset, sequence in enumerate(sequences)
    }
    camera_centers = np.column_stack(
        (np.linspace(0.0, 10.0, 100), np.zeros(100), np.zeros(100))
    )

    first = audit_sequence_exclusive_geometry(clouds, camera_centers)
    second = audit_sequence_exclusive_geometry(clouds, camera_centers)

    assert first == second
    assert first["status"] == "PASS"
    assert first["worst_seq_nn_p90_over_S"] <= 0.040
    assert len(first["worst10_hotspots"]) == 10
    assert set(first["required_diagnostics"]) == {
        "P1100110_005|P1110111",
        "P1090109_002|P1110111",
        "P1140114_articulation",
    }


def test_single_epoch_geometry_gate_is_explicitly_not_applicable() -> None:
    from audit_map_geometry import single_epoch_gate

    gate = single_epoch_gate({"P1090109_002": "2026-06-15", "P1110111": "2026-06-15"})

    assert gate["applicable"] is False
    assert gate["status"] == "NOT_APPLICABLE"


def test_ghost_audit_fails_closed_when_a_fuhe_sequence_has_no_exclusive_points() -> None:
    clouds = {
        "P1090109_002": [(1, np.asarray([0.0, 0.0, 0.0]))],
        "P1100110_005": [(2, np.asarray([0.0, 0.0, 0.0]))],
        "P1110111": [(3, np.asarray([0.0, 0.0, 0.0]))],
        "P1120112": [(4, np.asarray([0.0, 0.0, 0.0]))],
    }
    camera_centers = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    result = audit_sequence_exclusive_geometry(clouds, camera_centers)

    assert result["status"] == "FAIL"
    assert result["missing_sequences"] == ["P1140114"]


def test_ghost_audit_masks_to_common_route_visibility_and_reports_coverage() -> None:
    clouds = {
        "A": [
            (1, np.asarray([2.0, 0.01, 0.0])),
            (2, np.asarray([2.0, 100.0, 0.0])),
        ],
        "B": [
            (3, np.asarray([2.0, 0.02, 0.0])),
            (4, np.asarray([2.0, -100.0, 0.0])),
        ],
    }
    routes = {
        "A": np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        "B": np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
    }

    result = audit_sequence_exclusive_geometry(
        clouds,
        np.vstack(list(routes.values())),
        camera_centers_by_sequence=routes,
        route_supported_edges={("A", "B")},
        expected_sequences={"A", "B"},
    )

    pair = result["sequence_pairs"]["A|B"]
    assert pair["coverage_denominator_exclusive_points"] == 4
    assert pair["route_supported_exclusive_points"] == 2
    assert pair["unsupported_exclusive_points"] == 2
    assert pair["route_support_coverage_fraction"] == 0.5
    assert pair["status"] == "PASS"
    assert result["worst10_hotspots"][0]["point3D_id"] in {1, 3}


def test_unsupported_ghost_pair_is_not_applicable_instead_of_failed() -> None:
    clouds = {
        "A": [(1, np.asarray([0.0, 100.0, 0.0]))],
        "B": [(2, np.asarray([0.0, -100.0, 0.0]))],
    }
    routes = {
        "A": np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        "B": np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
    }

    result = audit_sequence_exclusive_geometry(
        clouds,
        np.vstack(list(routes.values())),
        camera_centers_by_sequence=routes,
        route_supported_edges={("A", "B")},
        expected_sequences={"A", "B"},
    )

    assert result["status"] == "NOT_APPLICABLE"
    assert result["sequence_pairs"]["A|B"]["status"] == "NOT_APPLICABLE"
    assert result["sequence_pairs"]["A|B"]["applicable"] is False


def test_trajectory_jump_audit_detects_more_than_ten_times_median_step() -> None:
    routes = {
        "A": np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [20.0, 0.0, 0.0]]
        ),
        "B": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    }

    result = trajectory_jump_audit(routes)

    assert result["status"] == "FAIL"
    assert result["total_jumps_over_10x_median"] == 1
    assert result["sequences"]["A"]["jump_indices"] == [2]


def _passing_required_ghost_geometry() -> dict:
    required = (
        "P1090109_002|P1110111",
        "P1100110_005|P1110111",
    )
    thresholds = {
        "minimum_left_route_supported_points": 500,
        "minimum_right_route_supported_points": 500,
        "minimum_supported_cells": 5,
        "minimum_route_support_coverage_fraction": 0.10,
        "maximum_seq_nn_p90_over_S": 0.04,
    }
    pairs = {
        edge: {
            "status": "PASS",
            "applicable": True,
            "left_route_supported_points": 500,
            "right_route_supported_points": 500,
            "supported_cells": 5,
            "route_support_coverage_fraction": 0.10,
            "seq_nn_p90_over_S": 0.04,
            "maximum": 0.04,
            "required_pair_thresholds": thresholds,
        }
        for edge in required
    }
    pairs["P1090109_002|P1100110_005"] = {
        "status": "NOT_APPLICABLE",
        "applicable": False,
        "reason": "same-direction pair is not a required ghost route",
    }
    return {
        "status": "PASS",
        "required_pair_thresholds": thresholds,
        "sequence_pairs": pairs,
    }


def test_formal_geometry_checks_require_both_fuhe_ghost_pairs() -> None:
    checks = formal_geometry_checks(
        sim3_pass=True,
        ghost_geometry=_passing_required_ghost_geometry(),
        direction_overlap_normalized={"symmetric_median": 0.04, "symmetric_p90": 0.14},
        route_cluster_evidence={"status": "PASS"},
        trajectory_evidence={"total_jumps_over_10x_median": 0},
    )

    assert checks == {
        "G5.7": True,
        "G6.1": True,
        "G6.2": True,
        "G6.3": True,
        "G6.4": True,
    }


def test_formal_geometry_g61_rejects_global_not_applicable() -> None:
    checks = formal_geometry_checks(
        sim3_pass=True,
        ghost_geometry={"status": "NOT_APPLICABLE", "sequence_pairs": {}},
        direction_overlap_normalized={"symmetric_median": 0.04, "symmetric_p90": 0.14},
        route_cluster_evidence={"status": "PASS"},
        trajectory_evidence={"total_jumps_over_10x_median": 0},
    )

    assert checks["G6.1"] is False


def test_formal_g61_requires_explicit_required_pair_threshold_record() -> None:
    ghost = _passing_required_ghost_geometry()
    ghost.pop("required_pair_thresholds")

    checks = formal_geometry_checks(
        sim3_pass=True,
        ghost_geometry=ghost,
        direction_overlap_normalized={"symmetric_median": 0.04, "symmetric_p90": 0.14},
        route_cluster_evidence={"status": "PASS"},
        trajectory_evidence={"total_jumps_over_10x_median": 0},
    )

    assert checks["G6.1"] is False


def test_required_fuhe_pairs_record_and_enforce_support_thresholds() -> None:
    sequences = ("P1090109_002", "P1100110_005", "P1110111")
    base = np.asarray(
        [
            [2.0 * cell + 0.001 * (index % 10), 0.0, 0.0]
            for cell in range(5)
            for index in range(100)
        ]
    )
    clouds = {
        sequence: [
            (offset * 1000 + index, xyz + np.asarray([0.0, offset * 0.001, 0.0]))
            for index, xyz in enumerate(base)
        ]
        for offset, sequence in enumerate(sequences)
    }
    routes = {
        sequence: np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        for sequence in sequences
    }
    required_edges = {
        ("P1090109_002", "P1110111"),
        ("P1100110_005", "P1110111"),
    }

    result = audit_sequence_exclusive_geometry(
        clouds,
        np.vstack(list(routes.values())),
        camera_centers_by_sequence=routes,
        route_supported_edges=required_edges,
        expected_sequences=set(sequences),
    )

    assert result["required_pair_thresholds"] == {
        "minimum_left_route_supported_points": 500,
        "minimum_right_route_supported_points": 500,
        "minimum_supported_cells": 5,
        "minimum_route_support_coverage_fraction": 0.10,
        "maximum_seq_nn_p90_over_S": 0.04,
    }
    for edge in ("P1090109_002|P1110111", "P1100110_005|P1110111"):
        pair = result["sequence_pairs"][edge]
        assert pair["status"] == "PASS"
        assert pair["left_route_supported_points"] >= 500
        assert pair["right_route_supported_points"] >= 500
        assert pair["supported_cells"] >= 5
        assert pair["route_support_coverage_fraction"] >= 0.10


def test_required_fuhe_pair_without_route_evidence_fails_instead_of_na() -> None:
    sequences = ("P1090109_002", "P1100110_005", "P1110111")
    clouds = {
        sequence: [(index, np.asarray([1.0, 0.0, 0.0]))]
        for index, sequence in enumerate(sequences)
    }
    routes = {
        sequence: np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        for sequence in sequences
    }

    result = audit_sequence_exclusive_geometry(
        clouds,
        np.vstack(list(routes.values())),
        camera_centers_by_sequence=routes,
        route_supported_edges=set(),
        expected_sequences=set(sequences),
    )

    for edge in ("P1090109_002|P1110111", "P1100110_005|P1110111"):
        pair = result["sequence_pairs"][edge]
        assert pair["status"] == "FAIL"
        assert pair["applicable"] is True
        assert pair["reason"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("left_route_supported_points", 499),
        ("right_route_supported_points", 499),
        ("supported_cells", 4),
        ("route_support_coverage_fraction", 0.099),
        ("seq_nn_p90_over_S", 0.041),
    ],
)
def test_formal_g61_recomputes_every_required_pair_threshold(
    field: str, value: float
) -> None:
    ghost = _passing_required_ghost_geometry()
    ghost["sequence_pairs"]["P1090109_002|P1110111"][field] = value

    checks = formal_geometry_checks(
        sim3_pass=True,
        ghost_geometry=ghost,
        direction_overlap_normalized={"symmetric_median": 0.04, "symmetric_p90": 0.14},
        route_cluster_evidence={"status": "PASS"},
        trajectory_evidence={"total_jumps_over_10x_median": 0},
    )

    assert checks["G6.1"] is False
