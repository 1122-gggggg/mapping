from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from build_bundle_seed import active_ref_names, validate_descriptors  # noqa: E402


class FakeImage:
    def __init__(self, name: str, *, has_pose: bool = True) -> None:
        self.name = name
        self.has_pose = has_pose


class FakeReconstruction:
    def __init__(self, names: list[str]) -> None:
        self.images = {i + 1: FakeImage(name) for i, name in enumerate(names)}


def test_active_ref_names_are_sorted_and_exclude_bridge_only(tmp_path: Path) -> None:
    for name in ("S02/000002.jpg", "S01/000003.jpg", "S01/000001.jpg"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"jpeg")
    rec = FakeReconstruction(
        ["S02/000002.jpg", "S01/000003.jpg", "S01/000001.jpg", "missing.jpg"]
    )

    names = active_ref_names(rec, tmp_path, {"S01/000003.jpg"})

    assert names == ["S01/000001.jpg", "S02/000002.jpg"]


def test_active_ref_names_excludes_deregistered_images(tmp_path: Path) -> None:
    for name in ("S01/active.jpg", "S01/deregistered.jpg"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"jpeg")
    rec = FakeReconstruction([])
    rec.images = {
        1: FakeImage("S01/active.jpg"),
        2: FakeImage("S01/deregistered.jpg", has_pose=False),
    }

    assert active_ref_names(rec, tmp_path, set()) == ["S01/active.jpg"]


def test_validate_descriptors_requires_finite_normalized_rows() -> None:
    descriptors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    result = validate_descriptors(descriptors, expected_rows=2)

    assert result.dtype == np.float32
    assert result.shape == (2, 2)


@pytest.mark.parametrize(
    "descriptors, message",
    [
        (np.zeros((1, 2), np.float32), "not L2-normalized"),
        (np.asarray([[np.nan, 0.0]], np.float32), "non-finite"),
        (np.eye(2, dtype=np.float32), "row count"),
    ],
)
def test_validate_descriptors_rejects_bad_output(
    descriptors: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_descriptors(descriptors, expected_rows=1)
