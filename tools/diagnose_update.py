#!/usr/bin/env python3
"""Advisory honesty check for map_update_summary.json.

Does not run SfM, triangulation, or G-U1. Labels routes that over-claim
incremental update capability and records whether gauge proof is still due.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TILE_ROUTES = frozenset({"changed-region", "changed", "tile_replace"})
FAILING_LABELS = frozenset({"UNIMPLEMENTED_TILE", "REGISTER_POINTS_CONTRACT_BROKEN"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="map_update_summary.json (or update_summary.json with a rows list)",
    )
    parser.add_argument(
        "--after-model",
        type=Path,
        default=None,
        help="Updated sparse model path; presence means G-U1 still needs proof",
    )
    parser.add_argument("--output", type=Path, required=True, help="Honesty report JSON")
    return parser


def _row_labels(row: dict[str, Any]) -> list[str]:
    route = str(row.get("route", ""))
    status = str(row.get("status", ""))
    labels: list[str] = []
    if route == "register":
        labels.append("REGISTER_PNP_ONLY")
        if row.get("points_added", 0) != 0:
            labels.append("REGISTER_POINTS_CONTRACT_BROKEN")
    if route in TILE_ROUTES or status == "needs_tile_replace":
        labels.append("UNIMPLEMENTED_TILE")
    if route == "submap":
        labels.append("SUBMAP_NOT_INCREMENTAL")
    return labels


def diagnose_rows(rows: list[dict[str, Any]], after_model: Path | str | None = None) -> dict[str, Any]:
    labeled_rows: list[dict[str, Any]] = []
    seen: list[str] = []
    for row in rows:
        labels = _row_labels(row)
        for label in labels:
            if label not in seen:
                seen.append(label)
        labeled_rows.append(
            {
                "seq": row.get("seq"),
                "route": row.get("route"),
                "status": row.get("status"),
                "points_added": row.get("points_added"),
                "labels": labels,
            }
        )
    return {
        "ok": not any(label in FAILING_LABELS for label in seen),
        "G-U1": "NOT_APPLICABLE" if after_model is None else "REQUIRES_G_U1",
        "labels": seen,
        "rows": labeled_rows,
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"Summary does not exist: {resolved}")
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"cannot load summary: {exc}") from exc
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("rows", [])
    else:
        raise SystemExit("summary must be an object with rows or a list")
    if not isinstance(rows, list):
        raise SystemExit("summary rows must be a list")
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = diagnose_rows(load_rows(args.summary), after_model=args.after_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
