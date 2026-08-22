#!/usr/bin/env python3
"""Session-unit point evidence ledger.

Expected observation rows:
  {"point_id": str, "visible": bool, "matched": bool}

`point_id` comes from make_point_id() — a COLMAP pid, or (ref_name, cell_cx, cell_cy).
map_update_tool.record_observation currently stores tile grids and ref_name#ref_kp
anchor hits, not those identities, so this module is not hooked there. Do not invent
point ids from frame counts.
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
        "visible_sessions": [],
        "matched_sessions": [],
        "positive_weight": 0.0,
        "negative_weight": 0.0,
        "stability": 1.0,
        "state": "active",
        "unmatched_visible_sessions": [],
    }


class PointEvidenceLedger:
    def __init__(self) -> None:
        self.sessions: list[str] = []
        self.points: dict[str, dict] = {}

    def record_session(self, session_id: str, observations: Iterable[dict]) -> None:
        if session_id not in self.sessions:
            for record in self.points.values():
                record["stability"] *= SESSION_DECAY
            self.sessions.append(session_id)

        aggregated: dict[str, dict[str, bool]] = {}
        for row in observations:
            point_id = str(row["point_id"])
            visible = bool(row.get("visible"))
            matched = bool(row.get("matched"))
            current = aggregated.setdefault(point_id, {"visible": False, "matched": False})
            current["visible"] = current["visible"] or visible
            current["matched"] = current["matched"] or matched

        for point_id, flags in aggregated.items():
            record = self.points.setdefault(point_id, _empty_point(point_id, session_id))
            visible = flags["visible"]
            matched = flags["matched"]
            if not visible and not matched:
                continue
            if visible and session_id not in record["visible_sessions"]:
                record["visible_sessions"].append(session_id)
            if matched:
                if session_id not in record["matched_sessions"]:
                    record["matched_sessions"].append(session_id)
                record["positive_weight"] += 1.0
                record["last_seen_session"] = session_id
                record["stability"] = 1.0
                record["state"] = "active"
                continue
            # visible and unmatched — caller said should-be-visible
            record["negative_weight"] += 1.0
            if session_id not in record["unmatched_visible_sessions"]:
                record["unmatched_visible_sessions"].append(session_id)
            unmatched = record["unmatched_visible_sessions"]
            if len(unmatched) >= RETIRE_UNMATCHED_SESSIONS:
                record["state"] = "retired"
            elif record["state"] == "active":
                record["state"] = "suspected_stale"

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
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
        return ledger
