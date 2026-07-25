"""Tests for build_gravity_alignment.

These build synthetic camera fleets so the maths is checked against a KNOWN
gravity direction, without needing a COLMAP model on disk.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_gravity_alignment import (  # noqa: E402
    angle_between_deg,
    rotation_between,
    solve_gravity_irls,
    verify,
)


def camera_fleet(gravity: np.ndarray, yaws_deg, rolls_deg=None) -> np.ndarray:
    """Camera x-axes for a fleet that yaws about `gravity` with optional roll."""
    gravity = gravity / np.linalg.norm(gravity)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(gravity[0]) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    east = np.cross(gravity, helper)
    east /= np.linalg.norm(east)
    north = np.cross(gravity, east)
    rolls_deg = rolls_deg if rolls_deg is not None else [0.0] * len(yaws_deg)
    axes = []
    for yaw, roll in zip(yaws_deg, rolls_deg):
        yaw_r, roll_r = np.radians(yaw), np.radians(roll)
        horizontal = np.cos(yaw_r) * east + np.sin(yaw_r) * north
        axes.append(np.cos(roll_r) * horizontal + np.sin(roll_r) * gravity)
    return np.array(axes)


ARGS = dict(iters=12, soft_deg=2.0)


def test_recovers_known_gravity_from_zero_roll_fleet():
    truth = np.array([0.0, -1.0, 0.0])
    axes = camera_fleet(truth, np.linspace(0, 350, 36))

    estimate, singular, _ = solve_gravity_irls(axes, **ARGS)

    assert angle_between_deg(estimate, truth) < 1e-5
    assert singular[1] / singular[0] > 0.5  # x-axes span 2D
    assert singular[2] < 1e-9              # clean null direction


def test_recovers_tilted_gravity():
    truth = np.array([-0.009, -0.924, -0.382])
    truth /= np.linalg.norm(truth)
    axes = camera_fleet(truth, np.linspace(0, 350, 36))

    estimate, _, _ = solve_gravity_irls(axes, **ARGS)

    assert angle_between_deg(estimate, truth) < 1e-5


def test_irls_suppresses_rolled_outliers():
    """Outliers must be CLUSTERED in yaw to bias a plain SVD -- evenly spaced
    rolled frames cancel out and would make this test vacuous."""
    truth = np.array([0.0, -1.0, 0.0])
    yaws = list(np.linspace(0, 350, 36))
    rolls = [0.0] * 36
    for i in range(4):  # 4 badly rolled frames at adjacent headings
        rolls[i] = 25.0
    axes = camera_fleet(truth, yaws, rolls)

    robust, _, weights = solve_gravity_irls(axes, **ARGS)
    plain = np.linalg.svd(axes, full_matrices=False)[2][-1]

    assert angle_between_deg(robust, truth) < angle_between_deg(plain, truth)
    assert weights[0] < 0.5  # the outlier really was down-weighted


def test_single_heading_flight_is_caught_by_yaw_diversity_not_conditioning():
    """A straight single-heading flight cannot determine gravity: the null space
    is 2D, not 1D.

    REGRESSION: s2/s3 does NOT catch this. Both are ~0, so their ratio is
    numerical noise that reads as perfectly conditioned -- an earlier version of
    G-GRAV-1 passed this exact case. s2/s1 is the discriminator.
    """
    truth = np.array([0.0, -1.0, 0.0])
    axes = camera_fleet(truth, [10.0] * 30)

    _, singular, _ = solve_gravity_irls(axes, **ARGS)

    conditioning = singular[1] / singular[2] if singular[2] > 1e-12 else float("inf")
    assert conditioning >= 5.0, "s2/s3 looks fine here -- that is the trap"
    assert singular[1] / singular[0] < 0.10  # the gate that actually fires


def test_rotation_between_maps_gravity_to_minus_z():
    gravity = np.array([-0.009, -0.924, -0.382])
    gravity /= np.linalg.norm(gravity)

    rotation = rotation_between(gravity, np.array([0.0, 0.0, -1.0]))

    assert np.allclose(rotation @ gravity, [0.0, 0.0, -1.0], atol=1e-9)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(rotation), 1.0)  # proper rotation, no mirror


def test_rotation_between_handles_antiparallel():
    rotation = rotation_between(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]))

    assert np.allclose(rotation @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, -1.0], atol=1e-9)
    assert np.isclose(np.linalg.det(rotation), 1.0)


def test_verify_accepts_unchanged_gauge():
    payload = {"schema": "T_align_gravity/1", "gravity_in_map": [0.0, -1.0, 0.0]}

    ok, message = verify(payload, {"gravity_in_map": [0.0, -1.0, 0.0]}, 0.05)

    assert ok and "stable" in message


def test_verify_rejects_moved_gauge():
    """A BA that nudged old poses must be caught -- this is the G-U1 backstop."""
    payload = {"schema": "T_align_gravity/1", "gravity_in_map": [0.0, -1.0, 0.0]}
    moved = np.array([0.0, -np.cos(np.radians(0.5)), np.sin(np.radians(0.5))])

    ok, message = verify(payload, {"gravity_in_map": moved.tolist()}, 0.05)

    assert not ok and "drifted" in message


def test_verify_rejects_foreign_schema():
    ok, _ = verify({"schema": "other/9", "gravity_in_map": [0, -1, 0]},
                   {"gravity_in_map": [0, -1, 0]}, 0.05)

    assert not ok


@pytest.mark.parametrize("degrees", [0.5, 5.0, 30.0])
def test_verify_tolerance_scales(degrees):
    payload = {"schema": "T_align_gravity/1", "gravity_in_map": [0.0, -1.0, 0.0]}
    moved = np.array([0.0, -np.cos(np.radians(degrees)), np.sin(np.radians(degrees))])

    assert not verify(payload, {"gravity_in_map": moved.tolist()}, degrees * 0.5)[0]
    assert verify(payload, {"gravity_in_map": moved.tolist()}, degrees * 2.0)[0]
