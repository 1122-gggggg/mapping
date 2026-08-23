import numpy as np

from sfm_diagnosis.fisher import (
    compute_fisher_metrics,
    compute_pose_uncertainty,
    weighted_bearing_fim,
)


def test_distributed_points_are_full_rank():
    points = np.array(
        [[x, y, z] for x in (-2, -1, 0, 1, 2) for y in (-1, 0, 1) for z in (5, 8)],
        dtype=float,
    )
    metrics = compute_fisher_metrics(weighted_bearing_fim(points))
    assert metrics.rank == 6
    assert metrics.lambda_min > 0
    assert metrics.condition_number < 1e6


def test_nearly_collinear_points_have_a_null_direction():
    points = np.array([[0.1 * i, 0.0, 5.0 + 0.1 * i] for i in range(-10, 11)], dtype=float)
    metrics = compute_fisher_metrics(weighted_bearing_fim(points))
    assert metrics.rank < 6
    assert metrics.condition_number > 1e10


def test_zero_weights_remove_information():
    points = np.array([[1, 0, 5], [-1, 0, 5], [0, 1, 6], [0, -1, 6]], dtype=float)
    full = weighted_bearing_fim(points)
    zero = weighted_bearing_fim(points, np.zeros(len(points)))
    assert np.trace(full) > 0
    assert np.allclose(zero, 0)


def test_covariance_exposes_worst_direction_of_degenerate_geometry():
    distributed = np.array(
        [[x, y, z] for x in (-2, -1, 0, 1, 2) for y in (-1, 0, 1) for z in (5, 8)],
        dtype=float,
    )
    collinear = np.array(
        [[0.1 * i, 0.0, 5.0 + 0.1 * i] for i in range(-10, 11)],
        dtype=float,
    )
    healthy = compute_pose_uncertainty(weighted_bearing_fim(distributed))
    weak = compute_pose_uncertainty(weighted_bearing_fim(collinear))
    assert weak.sigma_pose_worst_normalized > healthy.sigma_pose_worst_normalized
    assert weak.sigma_translation_worst_m > healthy.sigma_translation_worst_m
    assert np.isclose(
        weak.weakest_translation_fraction + weak.weakest_rotation_fraction,
        1.0,
    )
