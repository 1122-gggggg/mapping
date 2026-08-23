#!/usr/bin/env python3
"""Session-level point evidence ledger with recency decay.

Expected observation rows::

    {"point_id": str, "visible": bool, "matched": bool}

``point_id`` comes from :func:`make_point_id`: a COLMAP point id, or an EDM
``(reference image, cell_x, cell_y)`` identity.

This is an ExMaps-inspired evidence ledger, not a verbatim reproduction of its
score.  Evidence is decayed by session age and positive/negative histories are
kept separately.  A point is retired only after *consecutive* visible-but-
unmatched sessions; an intervening successful match resets that streak.

The live update path must supply real point identities and should-be-visible
observations.  Frame counts alone are not landmark evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


SCHEMA = "point_evidence_ledger/1"
SESSION_DECAY = 0.5
RETIRE_UNMATCHED_SESSIONS = 2
STATES = ("active", "suspected_stale", "retired")


def make_point_id(
    *,
    colmap_pid: int | str | None = None,
    ref_name: str | None = None,
    cell_cx: int | float | None = None,
    cell_cy: int | float | None = None,
) -> str:
    if colmap_pid is not None and str(colmap_pid) != "":
        return f"pid:{colmap_pid}"
    if ref_name is not None and cell_cx is not None and cell_cy is not None:
        return f"cell:{ref_name}:{int(cell_cx)}:{int(cell_cy)}"
    raise ValueError("need colmap_pid or (ref_name, cell_cx, cell_cy)")


def _empty_point(point_id: str, session_id: str) -> dict:
    return {
        "point_id": point_id,
        "first_seen_session": session_id,
        "last_seen_session": None,
        "last_observation_session": None,
        "visible_sessions": [],
        "matched_sessions": [],
        "positive_weight": 0.0,
        "negative_weight": 0.0,
        "positive_recency": 0.0,
        "negative_recency": 0.0,
        "evidence_consistency": None,
        "stability": 0.0,
        "state": "active",
        "unmatched_visible_sessions": [],
        "unmatched_streak": 0,
    }


def _decayed_sum(session_ids: Iterable[str], session_index: dict[str, int], current: int) -> float:
    total = 0.0
    for session_id in session_ids:
        index = session_index.get(str(session_id))
        if index is None:
            continue
        age = max(0, current - index)
        total += SESSION_DECAY ** age
    return float(total)


class PointEvidenceLedger:
    def __init__(self) -> None:
        self.sessions: list[str] = []
        self.points: dict[str, dict] = {}

    def _refresh_scores(self) -> None:
        if not self.sessions:
            return
        index = {session_id: i for i, session_id in enumerate(self.sessions)}
        current = len(self.sessions) - 1
        for record in self.points.values():
            positive = _decayed_sum(record.get("matched_sessions", ()), index, current)
            negative = _decayed_sum(
                record.get("unmatched_visible_sessions", ()), index, current
            )
            record["positive_recency"] = positive
            record["negative_recency"] = negative
            denominator = positive + negative
            record["evidence_consistency"] = (
                float(positive / denominator) if denominator > 0.0 else None
            )
            # Recency/persistence signal. One current positive observation is 1;
            # an old isolated observation decays. Repeated recent observations
            # saturate at 1 rather than growing without bound.
            record["stability"] = float(min(1.0, positive))

    def record_session(self, session_id: str, observations: Iterable[dict]) -> None:
        is_new_session = session_id not in self.sessions
        if is_new_session:
            self.sessions.append(session_id)

        aggregated: dict[str, dict[str, bool]] = {}
        for row in observations:
            point_id = str(row["point_id"])
            visible = bool(row.get("visible"))
            matched = bool(row.get("matched"))
            # A successful match is evidence that the landmark was visible even
            # if the caller omitted ``visible=True``.
            visible = visible or matched
            current = aggregated.setdefault(point_id, {"visible": False, "matched": False})
            current["visible"] = current["visible"] or visible
            current["matched"] = current["matched"] or matched

        for point_id, flags in aggregated.items():
            record = self.points.setdefault(point_id, _empty_point(point_id, session_id))
            visible = flags["visible"]
            matched = flags["matched"]
            if not visible and not matched:
                continue

            record["last_observation_session"] = session_id
            if visible and session_id not in record["visible_sessions"]:
                record["visible_sessions"].append(session_id)

            if matched:
                if session_id not in record["matched_sessions"]:
                    record["matched_sessions"].append(session_id)
                    record["positive_weight"] += 1.0
                record["last_seen_session"] = session_id
                record["unmatched_streak"] = 0
                record["state"] = "active"
                continue

            # Visible and unmatched: caller explicitly says the point should have
            # been observable but failed to match. Count once per session.
            if session_id not in record["unmatched_visible_sessions"]:
                record["unmatched_visible_sessions"].append(session_id)
                record["negative_weight"] += 1.0
                # Increment only for a new session; repeated processing of the
                # same session must be idempotent.
                record["unmatched_streak"] = int(record.get("unmatched_streak", 0)) + 1
            if int(record.get("unmatched_streak", 0)) >= RETIRE_UNMATCHED_SESSIONS:
                record["state"] = "retired"
            elif record["state"] == "active":
                record["state"] = "suspected_stale"

        # Even points not observed this session lose recency; do this once the
        # session is present in the global timeline.
        if is_new_session:
            self._refresh_scores()
        else:
            # Re-processing the same session can add previously omitted point
            # observations, so scores still need recomputation without extra age.
            self._refresh_scores()

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "parameters": {
                "session_decay": SESSION_DECAY,
                "retire_unmatched_sessions": RETIRE_UNMATCHED_SESSIONS,
                "stability_semantics": "min(1, sum(decay**session_age for matched_sessions))",
            },
            "sessions": list(self.sessions),
            "points": {key: dict(value) for key, value in self.points.items()},
        }

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        if path.suffix == ".npz":
            import numpy as np

            np.savez_compressed(path, payload=json.dumps(payload))
            return
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PointEvidenceLedger":
        path = Path(path)
        if path.suffix == ".npz":
            import numpy as np

            payload = json.loads(str(np.load(path, allow_pickle=False)["payload"]))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
        ledger = cls()
        ledger.sessions = list(payload.get("sessions", []))
        ledger.points = {
            str(key): dict(value) for key, value in payload.get("points", {}).items()
        }
        # Backward-compatible loading of v1 files written before recency fields
        # were added. Refresh derives the new values from stored session lists.
        for point_id, record in ledger.points.items():
            defaults = _empty_point(point_id, record.get("first_seen_session") or "")
            for key, value in defaults.items():
                record.setdefault(key, value)
        ledger._refresh_scores()
        return ledger
