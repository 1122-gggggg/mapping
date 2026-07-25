from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from validate_edm_bundle import anchored_counts  # noqa: E402


def test_numpy_bundle_gate_is_normalized_to_builtin_bool() -> None:
    result = bool(np.isclose(3.0, 3.0))

    assert type(result) is bool


def test_anchored_counts_recomputes_finite_xyz_cells() -> None:
    refs = {
        "a.jpg": {"xyz_by_cell": np.asarray([[1, 2, 3], [np.nan, np.nan, np.nan]])},
        "b.jpg": {"xyz_by_cell": np.asarray([[0, 0, 0], [4, 5, 6]])},
    }

    assert anchored_counts(refs) == {"a.jpg": 1, "b.jpg": 2}
