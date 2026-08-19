from __future__ import annotations

import math

import pytest

from mapdoctor.recapture.compute import (
    bearing_fisher_information,
    compute_metric_bundle,
    convex_hull_area_fraction,
    fim_summary,
    grid_occupancy,
)
from mapdoctor.recapture.types import Availability

np = pytest.importorskip("numpy")


def test_spatial_coverage_not_just_point_count() -> None:
    spread = [(1, 1), (99, 1), (99, 99), (1, 99), (50, 50)]
    clustered = [(1, 1), (2, 1), (2, 2), (1, 2), (1.5, 1.5)]
    assert convex_hull_area_fraction(spread, 100, 100) > 0.90
    assert convex_hull_area_fraction(clustered, 100, 100) < 0.001
    assert grid_occupancy(spread, 100, 100)[0] > grid_occupancy(clustered, 100, 100)[0]


def test_spatial_coverage_ignores_nonfinite_and_out_of_image_points() -> None:
    valid = [(1, 1), (99, 1), (99, 99), (1, 99)]
    contaminated = valid + [(float("nan"), 50), (50, float("inf")), (-100, 50), (500, 500)]
    assert convex_hull_area_fraction(contaminated, 100, 100) == pytest.approx(
        convex_hull_area_fraction(valid, 100, 100)
    )
    assert grid_occupancy(contaminated, 100, 100) == grid_occupancy(valid, 100, 100)


def test_bearing_fim_detects_full_rank_and_degeneracy() -> None:
    points = np.array([[-2, -1, 4], [2, -1, 5], [-1.5, 2, 6], [2.5, 2, 7], [0.2, -2.5, 8], [0.5, 1, 3.5]], dtype=float)
    healthy = fim_summary(bearing_fisher_information(points))
    assert healthy["fim_rank"] == 6
    assert healthy["fim_lambda_min"] > 0
    degenerate = fim_summary(bearing_fisher_information(np.array([[0, 0, z] for z in (3, 4, 5, 6, 7)], dtype=float)))
    assert degenerate["fim_rank"] < 6
    assert math.isfinite(degenerate["fim_condition_number"])
    assert degenerate["fim_condition_number"] >= 1e7


def test_unobservable_fim_is_not_well_conditioned() -> None:
    zero = fim_summary(np.zeros((6, 6)))
    assert zero["fim_rank"] == 0
    assert zero["fim_condition_number"] >= 1e7
    bundle = compute_metric_bundle({"points_camera": []})
    assert bundle["fim_rank"].value == 0
    assert bundle["fim_condition_number"].value >= 1e7


def test_metric_bundle_does_not_emit_nan_or_count_invalid_camera_points() -> None:
    bundle = compute_metric_bundle(
        {
            "depths": [-1.0, 0.0, float("nan")],
            "points_camera": [[0, 0, 4], [float("nan"), 0, 2], [0, 0, 0]],
            "image_points": [[1, 1], [99, 1], [99, 99], [1, 99], [float("nan"), 50]],
            "image_size": [100, 100],
        }
    )
    assert bundle["depth_p50"].status == Availability.UNAVAILABLE
    assert bundle["visible_landmark_count"].value == 1
    assert math.isfinite(bundle["inlier_convex_hull_coverage"].value)
    for metric in bundle.values():
        if isinstance(metric.value, float):
            assert math.isfinite(metric.value)
