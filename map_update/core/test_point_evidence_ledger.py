from __future__ import annotations

import json
from pathlib import Path

from point_evidence_ledger import (
    PointEvidenceLedger,
    make_point_id,
)


def test_make_point_id_prefers_colmap_pid():
    assert make_point_id(colmap_pid=42) == "pid:42"
    assert make_point_id(ref_name="P1/a.jpg", cell_cx=3, cell_cy=7) == "cell:P1/a.jpg:3:7"


def test_one_visible_unmatched_session_is_not_retired():
    ledger = PointEvidenceLedger()
    point_id = make_point_id(colmap_pid=1)
    ledger.record_session(
        "s1",
        [
            {"point_id": point_id, "visible": True, "matched": False},
            {"point_id": point_id, "visible": True, "matched": False},
        ],
    )
    assert ledger.points[point_id]["state"] == "suspected_stale"
    assert ledger.points[point_id]["last_seen_session"] is None


def test_two_sessions_visible_unmatched_retires_without_deleting(tmp_path: Path):
    ledger = PointEvidenceLedger()
    point_id = make_point_id(ref_name="ref.jpg", cell_cx=1, cell_cy=2)
    ledger.record_session("s1", [{"point_id": point_id, "visible": True, "matched": False}])
    ledger.record_session("s2", [{"point_id": point_id, "visible": True, "matched": False}])
    assert ledger.points[point_id]["state"] == "retired"
    assert point_id in ledger.points

    out = tmp_path / "evidence" / "point_stability.json"
    ledger.write(out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "point_evidence_ledger/1"
    assert payload["points"][point_id]["state"] == "retired"
    loaded = PointEvidenceLedger.load(out)
    assert loaded.points[point_id]["state"] == "retired"


def test_matched_session_stays_active_and_updates_last_seen():
    ledger = PointEvidenceLedger()
    point_id = make_point_id(colmap_pid="7")
    ledger.record_session("s1", [{"point_id": point_id, "visible": True, "matched": True}])
    assert ledger.points[point_id]["state"] == "active"
    assert ledger.points[point_id]["last_seen_session"] == "s1"
    assert ledger.points[point_id]["positive_weight"] == 1.0


def test_not_visible_is_ignored():
    ledger = PointEvidenceLedger()
    point_id = make_point_id(colmap_pid=9)
    ledger.record_session("s1", [{"point_id": point_id, "visible": False, "matched": False}])
    assert ledger.points[point_id]["state"] == "active"
    assert ledger.points[point_id]["visible_sessions"] == []
    assert ledger.points[point_id]["negative_weight"] == 0.0
