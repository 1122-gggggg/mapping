"""Tests for verify_gauge_invariance.

Focus is residual_rotation_deg: the check that catches a coherent whole-map
rotation which per-element deltas under-report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_gauge_invariance import residual_rotation_deg  # noqa: E402


def rotation_about_z(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    return np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )


def cloud(count: int = 300, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(count, 3)) * 5.0


def test_identical_clouds_have_zero_rotation():
    centers = cloud()

    assert residual_rotation_deg(centers, centers) == pytest.approx(0.0, abs=1e-9)


def test_pure_translation_is_not_a_rotation():
    """Translation is gauge-irrelevant for gravity; it must not be reported."""
    centers = cloud()

    assert residual_rotation_deg(centers, centers + [10.0, -3.0, 7.0]) == pytest.approx(
        0.0, abs=1e-9
    )


@pytest.mark.parametrize("degrees", [0.001, 0.227, 5.0, 45.0])
def test_recovers_a_known_rotation(degrees):
    centers = cloud()

    recovered = residual_rotation_deg(centers, centers @ rotation_about_z(degrees).T)

    assert recovered == pytest.approx(degrees, rel=1e-6, abs=1e-9)


def test_detects_the_rotation_measured_on_the_real_s5_finalization():
    """target_site's gluemap_aba -> final_model really did rotate the map by
    0.227 deg. That is small enough to look like noise per element and large
    enough to make T_align_gravity wrong."""
    centers = cloud()

    recovered = residual_rotation_deg(centers, centers @ rotation_about_z(0.227).T)

    assert recovered == pytest.approx(0.227, rel=1e-6)
    assert recovered > 1e-6  # the default G-U1c threshold


def test_small_random_jitter_is_not_reported_as_a_large_rotation():
    """Independent per-camera noise must not masquerade as a coherent re-gauge."""
    centers = cloud()
    jittered = centers + np.random.default_rng(1).normal(size=centers.shape) * 1e-6

    assert residual_rotation_deg(centers, jittered) < 1e-3


def test_never_returns_a_reflection():
    """Kabsch without the det correction can return a mirror for adversarial
    input; a mirror is not a rotation and must not be reported as one."""
    centers = cloud()
    mirrored = centers * [1.0, 1.0, -1.0]

    recovered = residual_rotation_deg(centers, mirrored)

    assert 0.0 <= recovered <= 180.0
    assert np.isfinite(recovered)


def test_degenerate_tiny_input_does_not_crash():
    assert residual_rotation_deg(np.zeros((2, 3)), np.zeros((2, 3))) == 0.0
