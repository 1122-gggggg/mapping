"""Command-line entry point for paper-derived session-graph hardening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .paper_graph import (
    harden_session_graph,
    load_edge_rows,
    merge_probe_metrics,
    write_hardening_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Harden exact-pair session geometry with community pruning, spectral "
            "minimum-range backbones, and a local-neighbor-global BA schedule."
        )
    )
    parser.add_argument("--edges", required=True, help="session_edges CSV/JSON/JSONL")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--sessions", help="optional newline/JSON session list")
    parser.add_argument("--probe-metrics", help="optional exact-pair homography/probe metrics")
    parser.add_argument("--config", help="optional YAML/JSON full or paper_graph config")
    parser.add_argument("--protected-sessions", help="optional newline/JSON protected session list")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    edge_rows = load_edge_rows(args.edges)
    if args.probe_metrics:
        edge_rows = merge_probe_metrics(edge_rows, load_edge_rows(args.probe_metrics))
    sessions = _load_names(args.sessions) if args.sessions else _infer_sessions(edge_rows)
    protected = _load_names(args.protected_sessions) if args.protected_sessions else []
    config = _load_config(args.config) if args.config else None
    report = harden_session_graph(sessions, edge_rows, config, protected_sessions=protected)
    outputs = write_hardening_outputs(report, args.output)
    summary = {
        "schema_version": report["schema_version"],
        "counts": report["counts"],
        "outputs": outputs,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def _infer_sessions(rows: list[dict[str, Any]]) -> list[str]:
    sessions = set()
    for row in rows:
        for key in ("session_a", "session_b"):
            value = row.get(key)
            if value not in (None, ""):
                sessions.add(str(value))
    return sorted(sessions)


def _load_names(path: str | Path) -> list[str]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            return [str(item) for item in payload]
        if isinstance(payload, dict):
            for key in ("sessions", "protected_sessions", "names"):
                if isinstance(payload.get(key), list):
                    return [str(item) for item in payload[key]]
        raise ValueError(f"Unsupported session-list JSON shape: {source}")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def _load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be an object: {source}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
