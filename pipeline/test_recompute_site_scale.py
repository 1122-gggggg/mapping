"""Tests for recompute_site_scale.

The load-bearing test here is test_pure_append_moves_S_even_with_frozen_poses:
that is the failure mode the whole tool exists to prevent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recompute_site_scale import (  # noqa: E402
    SPAN_NORMALIZED_TRACKER,
    robust_camera_span,
    tracker_params,
)


def ring(radius: float, count: int = 200, z: float = 0.0) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.full(count, z)])


def test_span_is_twice_p95_distance_from_median():
    centers = ring(3.0)

    # every camera sits at distance 3 from the centre, so p95 == 3
    assert robust_camera_span(centers) == pytest.approx(6.0, rel=1e-9)


def test_pure_append_moves_S_even_with_frozen_poses():
    """THE point of this tool.

    Append cameras covering new ground. Every original camera keeps its exact
    coordinates -- the gauge is bit-identical -- and S still moves, because it
    is a p95 over the whole set. Anyone who skips the recompute here ships
    stale tracker thresholds.
    """
    before = ring(3.0, 200)
    appended = np.vstack([before, ring(9.0, 60)])

    assert np.array_equal(appended[: len(before)], before)  # old poses untouched

    span_before = robust_camera_span(before)
    span_after = robust_camera_span(appended)

    assert span_after > span_before
    assert abs(span_after / span_before - 1.0) > 0.01  # past the reissue tolerance


def test_every_tracker_threshold_moves_with_S():
    before = tracker_params(robust_camera_span(ring(3.0, 200)))
    after = tracker_params(robust_camera_span(np.vstack([ring(3.0, 200), ring(9.0, 60)])))

    assert set(before) == set(SPAN_NORMALIZED_TRACKER)
    for name in SPAN_NORMALIZED_TRACKER:
        assert after[name] != before[name], f"{name} silently kept its old value"


def test_p95_ignores_a_single_stray_camera_but_max_would_not():
    centers = ring(3.0, 200)
    strayed = np.vstack([centers, [[500.0, 0.0, 0.0]]])

    span = robust_camera_span(strayed)
    origin = np.median(strayed, axis=0)
    span_if_max = 2.0 * float(np.linalg.norm(strayed - origin, axis=1).max())

    assert span == pytest.approx(6.0, rel=1e-2)  # p95 barely notices
    assert span_if_max > 900.0                   # max is completely captured


def test_median_origin_survives_a_lopsided_cluster():
    """componentwise median, not mean: a dense off-centre cluster must not drag
    the origin the way an average would."""
    centers = np.vstack([ring(3.0, 200), np.tile([40.0, 0.0, 0.0], (30, 1))])

    origin_median = np.median(centers, axis=0)
    origin_mean = centers.mean(axis=0)

    assert abs(origin_median[0]) < abs(origin_mean[0])


def test_gauge_rotation_alone_does_not_change_S():
    """S is rotation invariant, so a pure re-gauge leaves it alone. That is why
    an unchanged S is NOT evidence the gauge held -- use verify_gauge_invariance."""
    centers = ring(3.0, 200) + np.array([1.0, 2.0, 3.0])
    angle = np.radians(30.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )

    assert robust_camera_span(centers @ rotation.T) == pytest.approx(
        robust_camera_span(centers), rel=1e-9
    )


def test_ratios_match_the_deployment_contract():
    """Drift here silently repoints every deployed threshold."""
    assert SPAN_NORMALIZED_TRACKER == {
        "radius": 0.16,
        "max_jump": 0.40,
        "adaptive_jump_floor": 0.0006,
        "adaptive_jump_bootstrap": 0.004,
        "adaptive_jump_ceiling": 0.0016,
    }


def test_target_site_ratios_reproduce_the_edmconfig_defaults():
    """EDMConfig's defaults are these ratios at target_site's S = 5.0065 -- they
    are NOT universal constants, which is why a site without its own profile
    silently inherits target_site's scale."""
    params = tracker_params(5.006493)

    assert params["radius"] == pytest.approx(0.8, abs=2e-3)
    assert params["max_jump"] == pytest.approx(2.0, abs=3e-3)
    assert params["adaptive_jump_floor"] == pytest.approx(0.003, abs=1e-5)
    assert params["adaptive_jump_bootstrap"] == pytest.approx(0.02, abs=3e-5)
    assert params["adaptive_jump_ceiling"] == pytest.approx(0.008, abs=2e-5)


@pytest.mark.parametrize("bad", [np.zeros((1, 3)), np.zeros((4, 2)), np.full((4, 3), np.nan)])
def test_rejects_degenerate_input(bad):
    with pytest.raises(ValueError):
        robust_camera_span(bad)
