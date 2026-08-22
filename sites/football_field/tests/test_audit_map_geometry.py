from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from audit_map_geometry import (  # noqa: E402
    MAX_GHOST_P90_OVER_SPAN,
    REQUIRED_GHOST_SEQUENCE_PAIRS,
    audit_sequence_exclusive_geometry,
    estimate_sim3,
    filter_geometric_pairs,
    geometry_checks_pass,
    nearest_neighbor_summary,
    required_ghost_pairs_pass,
    robust_spatial_span,
    sequence_exclusive_point_clouds,
    trajectory_overlap_check,
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


def _exclusive_clouds(offset: float, count: int = 500):
    left, right = "P1270127", "P1290129"
    base = np.column_stack((np.linspace(0.0, 9.0, count), np.zeros(count), np.zeros(count)))
    return {
        left: [(index, xyz) for index, xyz in enumerate(base)],
        right: [
            (count + index, xyz + np.asarray([0.0, offset, 0.0]))
            for index, xyz in enumerate(base)
        ],
    }


def test_football_required_ghost_pair_is_the_two_build_sequences() -> None:
    assert REQUIRED_GHOST_SEQUENCE_PAIRS == frozenset({("P1270127", "P1290129")})


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

    clouds = sequence_exclusive_point_clouds(points, {1: "P1270127", 2: "P1290129"})

    assert [point_id for point_id, _xyz in clouds["P1270127"]] == [1]
    assert [point_id for point_id, _xyz in clouds["P1290129"]] == [3]


def test_exclusive_ghost_pair_fails_when_surfaces_are_separated() -> None:
    pairs = REQUIRED_GHOST_SEQUENCE_PAIRS
    camera_centers = np.column_stack(
        (np.linspace(0.0, 10.0, 100), np.zeros(100), np.zeros(100))
    )
    result = audit_sequence_exclusive_geometry(
        _exclusive_clouds(offset=5.0),
        camera_centers,
        expected_sequences={"P1270127", "P1290129"},
        required_pairs=pairs,
    )

    assert result["status"] == "FAIL"
    assert result["worst_seq_nn_p90_over_S"] > MAX_GHOST_P90_OVER_SPAN
    assert required_ghost_pairs_pass(result, required_pairs=pairs) is False


def test_exclusive_ghost_pair_passes_when_surfaces_coincide() -> None:
    pairs = REQUIRED_GHOST_SEQUENCE_PAIRS
    camera_centers = np.column_stack(
        (np.linspace(0.0, 10.0, 100), np.zeros(100), np.zeros(100))
    )
    result = audit_sequence_exclusive_geometry(
        _exclusive_clouds(offset=0.01),
        camera_centers,
        expected_sequences={"P1270127", "P1290129"},
        required_pairs=pairs,
    )

    assert result["status"] == "PASS"
    assert result["worst_seq_nn_p90_over_S"] <= MAX_GHOST_P90_OVER_SPAN
    assert required_ghost_pairs_pass(result, required_pairs=pairs) is True


def test_g62_is_not_applicable_without_fwd_rev() -> None:
    assert (
        trajectory_overlap_check(
            {"P1270127": "unknown", "P1290129": "unknown"}, {}
        )
        == "NOT_APPLICABLE"
    )
    assert geometry_checks_pass(
        {"G6.1": True, "G6.2": "NOT_APPLICABLE", "G6.3": True}
    )
    assert geometry_checks_pass({"G6.2": "NOT_APPLICABLE", "G6.1": False}) is False


