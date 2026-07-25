from __future__ import annotations

from changed_region_evidence import aggregate_changed_region_evidence


def stats(seq: str, tile: str, low_support_frames: int, support_ratio: float = 0.01) -> dict:
    return {
        "changed_region_candidates": [{
            "seq": seq,
            "tile": tile,
            "low_support_frames": low_support_frames,
            "query_kp": 100,
            "matches": 10,
            "inliers": 1,
            "support_ratio": support_ratio,
        }]
    }


def test_single_session_candidate_is_not_promoted():
    report = aggregate_changed_region_evidence(
        [("obs1.json", stats("P125", "1,2", 3))],
        min_sessions=2,
        min_low_support_frames=2,
    )

    assert report["promoted"] == []
    assert report["observed"][0]["decision"] == "hold"


def test_same_tile_across_two_sessions_is_not_promoted_by_default():
    report = aggregate_changed_region_evidence(
        [
            ("obs1.json", stats("P125", "1,2", 3)),
            ("obs2.json", stats("P126", "1,2", 2, support_ratio=0.02)),
        ],
        min_sessions=2,
        min_low_support_frames=2,
    )

    assert report["promoted"] == []


def test_same_tile_across_two_sessions_can_be_promoted_when_grouped_by_tile():
    report = aggregate_changed_region_evidence(
        [
            ("obs1.json", stats("P125", "1,2", 3)),
            ("obs2.json", stats("P126", "1,2", 2, support_ratio=0.02)),
        ],
        min_sessions=2,
        min_low_support_frames=2,
        group_by="tile",
    )

    assert len(report["promoted"]) == 1
    row = report["promoted"][0]
    assert row["key"] == "1,2"
    assert row["sessions"] == ["P125", "P126"]
    assert row["total_low_support_frames"] == 5
