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


def test_empty_candidates_are_insufficient_not_stability():
    report = aggregate_changed_region_evidence([])

    assert report["promoted"] == []
    assert report["observed"] == []
    assert report["hard_status"] == "VALID"
    assert report["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert report["proposal_status"] == "EMPTY_NO_CANDIDATES"
    assert report["authority"] == "review_only"
    assert report["geometry_invalidated"] is False
    assert report["independence_attested"] is False
    assert report["lineage_attested"] is False
    assert "no_candidate_rows" in report["decision_reasons"]
    assert "independence_not_attested" in report["decision_reasons"]
    assert "lineage_not_attested" in report["decision_reasons"]
    assert report["candidate_count"] == 0
    assert report["suppressed_count"] == 0
    assert report["unscorable_count"] == 0
    assert report["evidence_status"] != "PASS"


def test_malformed_nan_and_bool_rows_are_unscorable():
    payload = {
        "changed_region_candidates": [
            {
                "seq": "P125",
                "tile": "1,2",
                "low_support_frames": float("nan"),
                "query_kp": 100,
                "matches": 10,
                "inliers": 1,
                "support_ratio": 0.01,
            },
            {
                "seq": "P126",
                "tile": "1,2",
                "low_support_frames": True,
                "query_kp": 100,
                "matches": 10,
                "inliers": 1,
                "support_ratio": 0.01,
            },
            {
                "seq": "P127",
                "tile": "1,2",
                "low_support_frames": 4,
                "query_kp": 100,
                "matches": 10,
                "inliers": 1,
                "support_ratio": float("inf"),
            },
        ]
    }
    report = aggregate_changed_region_evidence(
        [("obs.json", payload)],
        min_sessions=1,
        min_low_support_frames=1,
        group_by="tile",
    )

    assert report["promoted"] == []
    assert report["observed"] == []
    assert report["unscorable_count"] == 3
    assert report["candidate_count"] == 3
    assert report["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert report["proposal_status"] == "INSUFFICIENT_EVIDENCE"
    assert "unscorable_rows" in report["decision_reasons"]
    assert report["geometry_invalidated"] is False


def test_low_session_is_insufficient_with_signed_margins():
    report = aggregate_changed_region_evidence(
        [("obs1.json", stats("P125", "1,2", 3))],
        min_sessions=2,
        min_low_support_frames=2,
    )

    assert report["promoted"] == []
    row = report["observed"][0]
    assert row["decision"] == "hold"
    assert row["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert row["decision_reason"] == "low_session_count"
    assert row["session_count_margin"] == -1
    assert row["total_low_support_margin"] == -1
    assert report["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert report["authority"] == "review_only"


def test_low_total_support_is_quality_shortfall():
    report = aggregate_changed_region_evidence(
        [
            ("obs1.json", stats("P125", "1,2", 3)),
            ("obs2.json", stats("P126", "1,2", 2, support_ratio=0.02)),
        ],
        min_sessions=2,
        min_low_support_frames=2,
        min_total_low_support_frames=10,
        group_by="tile",
    )

    assert report["promoted"] == []
    row = report["observed"][0]
    assert row["decision"] == "hold"
    assert row["evidence_status"] == "QUALITY_SHORTFALL"
    assert row["decision_reason"] == "low_total_support"
    assert row["session_count_margin"] == 0
    assert row["total_low_support_margin"] == -5
    assert report["evidence_status"] == "QUALITY_SHORTFALL"
    assert report["proposal_status"] == "QUALITY_SHORTFALL"
    assert report["thresholds"]["min_total_source"] == "explicit"


def test_promoted_review_candidate_keeps_existing_key():
    report = aggregate_changed_region_evidence(
        [
            ("obs1.json", stats("P125", "1,2", 3)),
            ("obs2.json", stats("P126", "1,2", 2, support_ratio=0.02)),
        ],
        min_sessions=2,
        min_low_support_frames=2,
        group_by="tile",
    )

    assert [row["key"] for row in report["promoted"]] == ["1,2"]
    row = report["promoted"][0]
    assert row["decision"] == "promote_for_review"
    assert row["evidence_status"] == "REVIEW_CANDIDATE"
    assert row["decision_reason"] == "review_candidate"
    assert row["sessions"] == ["P125", "P126"]
    assert row["total_low_support_frames"] == 5
    assert row["session_count_margin"] == 0
    assert row["total_low_support_margin"] == 1
    assert report["evidence_status"] == "WARN"
    assert report["proposal_status"] == "REVIEW_CANDIDATES"
    assert report["authority"] == "review_only"
    assert report["hard_status"] == "VALID"
    assert report["geometry_invalidated"] is False
    assert report["independence_attested"] is False
    assert report["lineage_attested"] is False


def test_default_seq_tile_grouping_does_not_enlarge_promotion():
    report = aggregate_changed_region_evidence(
        [
            ("obs1.json", stats("P125", "1,2", 3)),
            ("obs2.json", stats("P126", "1,2", 2, support_ratio=0.02)),
        ],
        min_sessions=2,
        min_low_support_frames=2,
    )

    assert report["promoted"] == []
    assert {row["key"] for row in report["observed"]} == {"P125:1,2", "P126:1,2"}
    assert all(row["decision"] == "hold" for row in report["observed"])
    assert all(row["evidence_status"] == "INSUFFICIENT_EVIDENCE" for row in report["observed"])
