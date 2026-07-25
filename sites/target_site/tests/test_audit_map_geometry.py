from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from audit_map_geometry import (  # noqa: E402
    estimate_sim3,
    filter_geometric_pairs,
    nearest_neighbor_summary,
    robust_spatial_span,
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
