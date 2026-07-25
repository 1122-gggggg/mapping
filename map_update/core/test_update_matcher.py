"""Tests for the backend-neutral update matcher layer.

Covers the dedup/aggregation logic and the EDM cell identity. The real backends
talk to the localization deploy tree (GPU + EDM weights) and are not importable
here, so they are exercised through a fake localizer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from update_matcher import (  # noqa: E402
    EDM_CELL,
    GRID_W,
    N_CELLS,
    EDMUpdateMatcher,
    QueryRecord,
    RefMatch,
    build_matcher,
    cell_keys,
    dedup_anchored,
)


def row(name, index, xy, xyz, keys, ref_keys=None):
    return RefMatch(name, index, np.asarray(xy, float), np.asarray(xyz, float),
                    np.asarray(keys), None if ref_keys is None else np.asarray(ref_keys))


# ---------------------------------------------------------------- dedup


def test_one_anchor_per_query_observation():
    """The inlier count must never exceed the number of distinct observations."""
    rows = [
        row("a", 0, [[1, 1], [2, 2]], [[0, 0, 1], [0, 0, 2]], [7, 8]),
        row("b", 1, [[1, 1], [3, 3]], [[9, 9, 9], [0, 0, 3]], [7, 9]),  # 7 repeats
    ]

    result = dedup_anchored(rows)

    assert len(result.p2) == 3
    assert sorted(m["qidx"] for m in result.meta) == [7, 8, 9]


def test_first_reference_wins_so_retrieval_rank_decides_ties():
    rows = [
        row("best", 0, [[1, 1]], [[1, 1, 1]], [5]),
        row("worse", 1, [[1, 1]], [[2, 2, 2]], [5]),
    ]

    result = dedup_anchored(rows)

    assert result.meta[0]["ref_name"] == "best"
    assert result.p3[0].tolist() == [1.0, 1.0, 1.0]


def test_non_finite_anchors_are_dropped():
    rows = [row("a", 0, [[1, 1], [2, 2]], [[np.nan, 0, 0], [0, 0, 1]], [1, 2])]

    result = dedup_anchored(rows)

    assert len(result.p2) == 1
    assert result.meta[0]["qidx"] == 2


def test_empty_input_returns_the_none_sentinel():
    """Callers branch on `P2 is None`, so this contract must hold."""
    p2, p3, meta = dedup_anchored([]).as_tuple()

    assert p2 is None and p3 is None and meta == []


def test_ragged_rows_are_rejected_loudly():
    bad = RefMatch("a", 0, np.zeros((3, 2)), np.zeros((2, 3)), np.arange(3))

    with pytest.raises(ValueError, match="ragged"):
        dedup_anchored([bad])


def test_tuple_cell_keys_dedup_the_same_way_as_integer_keys():
    """EDM keys are (cx, cy) tuples, XFeat keys are ints. Same behaviour."""
    rows = [
        row("a", 0, [[8, 8], [16, 16]], [[0, 0, 1], [0, 0, 2]], [[1, 1], [2, 2]]),
        row("b", 1, [[8, 9]], [[9, 9, 9]], [[1, 1]]),  # same cell as the first
    ]

    result = dedup_anchored(rows)

    assert len(result.p2) == 2
    assert all(isinstance(m["qidx"], tuple) for m in result.meta)


# ---------------------------------------------------------------- cell identity


def test_cell_key_is_round_not_floor():
    """round(kpt/8), matching build_reloc_map_edm. floor would shift the grid."""
    assert cell_keys(np.array([[12.0, 12.0]])).tolist() == [[2, 2]]
    assert cell_keys(np.array([[11.0, 11.0]])).tolist() == [[1, 1]]


def test_subpixel_spread_inside_a_cell_maps_to_one_key():
    """Measured spread is ~2.07 px median; it must not split an observation."""
    base = 8.0 * 10
    points = np.array([[base - 2.0, base], [base, base], [base + 2.0, base]])

    assert len({tuple(k) for k in cell_keys(points)}) == 1


def test_cell_grid_covers_the_edm_canvas_exactly():
    assert GRID_W == 1024 // EDM_CELL
    assert N_CELLS == GRID_W * (576 // EDM_CELL) == 9216


# ---------------------------------------------------------------- EDM backend


class FakeLocalizer:
    """Stands in for EDMLocalizer. Returns CAMERA-pixel correspondences."""

    def __init__(self, rows, scale=1.25):
        self.rows = rows
        self.scale = scale
        self.prepared = []

    def prepare_query(self, frame):
        self.prepared.append(frame)
        return {"frame": frame}

    def correspondences_by_ref(self, query, ref_names):
        return [self.rows.get(n, (np.zeros((0, 2)), np.zeros((0, 3)), 0)) for n in ref_names]


def test_edm_keys_are_quantised_in_canvas_not_camera_pixels():
    """correspondences_by_ref returns camera px (1280 wide); the cell id is only
    meaningful after dividing by scale back to the 1024-wide EDM canvas.
    Quantising camera px directly would index the wrong xyz_by_cell slot."""
    camera_xy = np.array([[100.0, 200.0]])
    matcher = EDMUpdateMatcher(FakeLocalizer({"r": (camera_xy, np.array([[1.0, 2.0, 3.0]]), 1)}))
    query = matcher.prepare_query(np.zeros((576, 1024, 3), np.uint8))

    rows = matcher.correspondences(query, ["r"], [0])

    expected = cell_keys(camera_xy / 1.25)
    assert rows[0].query_keys.tolist() == expected.tolist()
    assert rows[0].query_keys.tolist() != cell_keys(camera_xy).tolist()


def test_edm_synthesises_a_keypoint_table_from_matched_cells():
    """EDM has no keypoints until matching; the update loop still needs a table
    for tile statistics, so it is the union of matched query cells."""
    matcher = EDMUpdateMatcher(
        FakeLocalizer({
            "a": (np.array([[10.0, 10.0], [50.0, 50.0]]), np.zeros((2, 3)), 2),
            "b": (np.array([[10.0, 10.0], [90.0, 90.0]]), np.zeros((2, 3)), 2),
        })
    )
    query = matcher.prepare_query(np.zeros((576, 1024, 3), np.uint8))

    assert query.keypoints is None  # nothing known before matching
    matcher.correspondences(query, ["a", "b"], [0, 1])

    assert len(query.keypoints) == 3  # 4 matches, one shared cell


def test_edm_bundle_keyframe_matches_the_reloc_map_schema():
    matcher = EDMUpdateMatcher(FakeLocalizer({}))
    query = QueryRecord(width=1024, height=576)
    meta = [{"qidx": (3, 4), "xyz": np.array([1.0, 2.0, 3.0])}]

    entry = matcher.bundle_keyframe(query, meta, np.zeros((576, 1024, 3), np.uint8), "f.jpg")

    assert set(entry) == {"xyz_by_cell", "image_jpg"}
    assert entry["xyz_by_cell"].shape == (N_CELLS, 3)
    assert entry["xyz_by_cell"].dtype == np.float32
    assert entry["xyz_by_cell"][4 * GRID_W + 3].tolist() == [1.0, 2.0, 3.0]
    # every other cell stays unanchored
    assert np.isfinite(entry["xyz_by_cell"][:, 0]).sum() == 1
    assert entry["image_jpg"].dtype == np.uint8


def test_edm_bundle_keyframe_rejects_integer_keys():
    """Mixing an XFeat-shaped meta into the EDM writer would silently produce a
    bundle with anchors in the wrong cells."""
    matcher = EDMUpdateMatcher(FakeLocalizer({}))

    with pytest.raises(ValueError, match="cell keys"):
        matcher.bundle_keyframe(
            QueryRecord(width=1024, height=576),
            [{"qidx": 17, "xyz": np.zeros(3)}],
            np.zeros((576, 1024, 3), np.uint8),
            "f.jpg",
        )


def test_edm_bundle_meta_declares_the_feature_reloc_map_checks():
    assert EDMUpdateMatcher.bundle_meta()["feature"] == "edm"


def test_build_matcher_rejects_an_unknown_backend():
    with pytest.raises(SystemExit, match="unknown matcher backend"):
        build_matcher("superglue")
