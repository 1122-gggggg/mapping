from __future__ import annotations

import pytest

from point_evidence_ledger import PointEvidenceLedger, make_point_id


def test_old_single_match_decays_when_new_sessions_arrive() -> None:
    ledger = PointEvidenceLedger()
    point_id = make_point_id(colmap_pid=1)
    ledger.record_session("s1", [{"point_id": point_id, "matched": True}])
    assert ledger.points[point_id]["stability"] == pytest.approx(1.0)

    ledger.record_session("s2", [])
    assert ledger.points[point_id]["stability"] == pytest.approx(0.5)

    ledger.record_session("s3", [])
    assert ledger.points[point_id]["stability"] == pytest.approx(0.25)


def test_match_resets_consecutive_unmatched_streak() -> None:
    ledger = PointEvidenceLedger()
    point_id = make_point_id(colmap_pid=2)
    ledger.record_session("s1", [{"point_id": point_id, "visible": True, "matched": False}])
    assert ledger.points[point_id]["unmatched_streak"] == 1
    assert ledger.points[point_id]["state"] == "suspected_stale"

    ledger.record_session("s2", [{"point_id": point_id, "visible": True, "matched": True}])
    assert ledger.points[point_id]["unmatched_streak"] == 0
    assert ledger.points[point_id]["state"] == "active"

    ledger.record_session("s3", [{"point_id": point_id, "visible": True, "matched": False}])
    assert ledger.points[point_id]["unmatched_streak"] == 1
    assert ledger.points[point_id]["state"] == "suspected_stale"


def test_same_session_reprocessing_is_idempotent() -> None:
    ledger = PointEvidenceLedger()
    point_id = make_point_id(colmap_pid=3)
    observation = [{"point_id": point_id, "visible": True, "matched": False}]
    ledger.record_session("s1", observation)
    ledger.record_session("s1", observation)

    record = ledger.points[point_id]
    assert record["negative_weight"] == pytest.approx(1.0)
    assert record["unmatched_streak"] == 1
    assert record["unmatched_visible_sessions"] == ["s1"]


def test_consistency_separates_positive_and_negative_recency() -> None:
    ledger = PointEvidenceLedger()
    point_id = make_point_id(colmap_pid=4)
    ledger.record_session("s1", [{"point_id": point_id, "visible": True, "matched": True}])
    ledger.record_session("s2", [{"point_id": point_id, "visible": True, "matched": False}])

    record = ledger.points[point_id]
    assert record["positive_recency"] == pytest.approx(0.5)
    assert record["negative_recency"] == pytest.approx(1.0)
    assert record["evidence_consistency"] == pytest.approx(1.0 / 3.0)
