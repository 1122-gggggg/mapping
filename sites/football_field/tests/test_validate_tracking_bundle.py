from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from validate_tracking_bundle import circular_bimodality  # noqa: E402


def test_numpy_gate_results_must_be_normalized_to_builtin_bool() -> None:
    result = bool(np.asarray([1.0, 2.0]).all())

    assert type(result) is bool


def test_circular_bimodality_accepts_opposite_heading_clusters() -> None:
    yaws = np.deg2rad(np.asarray([-5, 0, 5, 175, 180, 185], dtype=np.float64))

    result = circular_bimodality(yaws)

    assert result["passed"] is True
    assert result["separation_deg"] > 170


def test_circular_bimodality_rejects_one_heading_cluster() -> None:
    yaws = np.deg2rad(np.asarray([-10, -5, 0, 5, 10, 15], dtype=np.float64))

    assert circular_bimodality(yaws)["passed"] is False
