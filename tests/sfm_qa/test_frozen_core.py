"""Contract tests for sfm_qa.frozen_core. Production lives elsewhere."""

from __future__ import annotations

import pytest

from sfm_qa.frozen_core import (
    FrozenBinsChangedError,
    admit_fringe_only_if,
    assert_frozen_bins_unchanged,
    classify_update_vs_appearance,
)


def _role(value):
    return value.role if hasattr(value, "role") else value


def test_hash_mismatch_raises():
    before = {
        "cameras.bin": "aaa111",
        "images.bin": "bbb222",
        "points3D.bin": "ccc333",
    }
    assert_frozen_bins_unchanged(before, dict(before))

    changed = dict(before)
    changed["images.bin"] = "zzz999"
    with pytest.raises(FrozenBinsChangedError):
        assert_frozen_bins_unchanged(before, changed)

    missing = {"cameras.bin": "aaa111", "images.bin": "bbb222"}
    with pytest.raises(FrozenBinsChangedError):
        assert_frozen_bins_unchanged(before, missing)


def test_admit_fringe_only_if_fails_closed():
    assert admit_fringe_only_if(0.80, 3, 0.0) is True
    assert admit_fringe_only_if(0.10, 3, 0.0) is False
    assert admit_fringe_only_if(0.80, 1, 0.0) is False
    assert admit_fringe_only_if(0.80, 0, 0.0) is False
    assert admit_fringe_only_if(0.80, 3, 1.5) is False
    assert admit_fringe_only_if(0.49, 4, 0.0) is False


def test_classify_update_vs_appearance_loc_covered_no_change():
    role = _role(classify_update_vs_appearance(True, True, False))
    assert role == "APPEARANCE_REF"
    assert _role(classify_update_vs_appearance(True, True, True)) == "UPDATE_CANDIDATE"
    assert _role(classify_update_vs_appearance(False, True, False)) != "APPEARANCE_REF"
    assert _role(classify_update_vs_appearance(True, False, False)) != "UPDATE_CANDIDATE"
