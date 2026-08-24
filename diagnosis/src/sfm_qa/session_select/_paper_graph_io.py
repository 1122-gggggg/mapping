"""Input/output helpers for paper graph hardening."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._paper_graph_util import (
    _artifact_row,
    _csv_cell,
    _json_default,
    _pair,
    _write_lines,
    _write_pairs,
)


def write_hardening_outputs(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write report, hardened CSV, pair lists, and optimization schedule."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in report.get("edge_rows", [])]
    edges_path = destination / "session_edges_hardened.csv"
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with edges_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in columns})
    payload = {key: value for key, value in report.items() if key != "edge_rows"}
    report_path = destination / "paper_graph_report.json"
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    schedule_path = destination / "mso_optimization_schedule.json"
    schedule_path.write_text(
        json.dumps(report.get("optimization_schedule", []), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    retained_path = destination / "retained_geometry_pairs.txt"
    pruned_path = destination / "quarantined_geometry_pairs.txt"
    sessions_path = destination / "quarantined_sessions.txt"
    submaps_path = destination / "new_submap_candidates.txt"
    _write_pairs(retained_path, report.get("retained_pairs", []))
    _write_pairs(pruned_path, report.get("pruned_pairs", []))
    _write_lines(sessions_path, report.get("quarantined_sessions", []))
    _write_lines(submaps_path, report.get("new_submap_candidates", []))
    return {
        "report": str(report_path),
        "hardened_edges": str(edges_path),
        "schedule": str(schedule_path),
        "retained_pairs": str(retained_path),
        "quarantined_pairs": str(pruned_path),
        "quarantined_sessions": str(sessions_path),
        "new_submaps": str(submaps_path),
    }


def load_edge_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load edge rows from CSV, JSON, or JSONL."""

    source = Path(path)
    if source.suffix.lower() == ".csv":
        with source.open(encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if source.suffix.lower() in {".jsonl", ".ndjson"}:
        return [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("edge_rows", "edges", "pairs", "artifacts"):
            values = payload.get(key)
            if isinstance(values, list):
                return [_artifact_row(row) for row in values if isinstance(row, Mapping)]
    raise ValueError(f"Unsupported edge-table shape: {source}")


def merge_probe_metrics(
    edge_rows: Sequence[Mapping[str, Any]], probe_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Merge optional homography summaries into edge rows by undirected pair."""

    probes = {}
    for raw in probe_rows:
        row = _artifact_row(raw)
        pair = _pair(row.get("session_a"), row.get("session_b"))
        if pair:
            probes.setdefault(pair, {}).update(
                {key: value for key, value in row.items() if value not in (None, "")}
            )
    merged = []
    for raw in edge_rows:
        row = dict(raw)
        pair = _pair(row.get("session_a"), row.get("session_b"))
        if pair in probes:
            row.update(probes[pair])
        merged.append(row)
    return merged
