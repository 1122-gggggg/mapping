#!/usr/bin/env python3
"""Copy a Fuhe matches DB and inject only official G6.1 hotspot edges.

Official G6.1 scored pairs: P109/P110 ↔ P111. Frozen P114 research notes do
not supersede this gate. P112 edges are refused until new footage.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from audit_dg_graph import LOFTR_SEQUENCE_ALLOWLIST, loftr_trigger_contract
from audit_map_geometry import official_g61_edges


FIXED_BA = (
    Path(__file__).resolve().parents[3] / "pipeline" / "repair_fuhe_gluemap_fixed_ba.py"
)
P112 = "P1120112"
REACHABLE = "REACHABLE_FOR_EXACT_ROI_PROBE"


def normalize_edge(value: object) -> tuple[str, str]:
    if isinstance(value, str):
        fields = value.split("|")
    else:
        fields = list(value)  # type: ignore[arg-type]
    if len(fields) != 2 or not all(fields) or fields[0] == fields[1]:
        raise ValueError(f"invalid sequence edge: {value!r}")
    return tuple(sorted((str(fields[0]), str(fields[1]))))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def failing_official_edges(geometry: dict[str, Any] | None) -> set[tuple[str, str]]:
    official = official_g61_edges()
    if not geometry:
        return set(official)
    ghost = geometry.get("stage_metrics", {}).get(
        "sequence_exclusive_ghost_geometry", {}
    )
    pairs = ghost.get("sequence_pairs") or geometry.get("sequence_pairs") or {}
    failing: set[tuple[str, str]] = set()
    for edge in official:
        summary = pairs.get("|".join(edge), {})
        if not isinstance(summary, dict) or summary.get("status") != "PASS":
            failing.add(edge)
    return failing


def geometry_gate_promotable(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("schema_version") != "sfm-gate-v2":
        return False
    if payload.get("status") != "PASS" or payload.get("ok") is not True:
        return False
    checks = payload.get("checks")
    if isinstance(checks, list):
        for check in checks:
            if check.get("id") == "G6.1":
                return check.get("state") == "PASS" and check.get("ok") is True
        return False
    if isinstance(checks, dict):
        value = checks.get("G6.1")
        if value is True:
            return True
        return isinstance(value, dict) and value.get("state") == "PASS"
    return False


def planned_repair_edges(
    *,
    geometry: dict[str, Any] | None,
    requested: Iterable[object] | None = None,
) -> tuple[set[tuple[str, str]], list[str]]:
    official = official_g61_edges()
    failing = failing_official_edges(geometry)
    planned = set(official & failing)
    if requested is not None:
        planned &= {normalize_edge(item) for item in requested}
    refused = sorted(
        "|".join(edge) for edge in planned if P112 in edge
    )
    planned = {edge for edge in planned if P112 not in edge}
    return planned, refused


def evaluate_repair_preflight(
    *,
    reachability: dict[str, Any] | None,
    trigger: dict[str, Any] | None,
    geometry: dict[str, Any] | None,
    requested_edges: Iterable[object] | None = None,
) -> dict[str, Any]:
    planned, refused_p112 = planned_repair_edges(
        geometry=geometry, requested=requested_edges
    )
    reasons: list[str] = []
    if reachability is None:
        reasons.append("missing reachability JSON")
    elif reachability.get("status") != REACHABLE:
        reasons.append(
            f"reachability status {reachability.get('status')!r} is not {REACHABLE}"
        )
    if trigger is None or trigger.get("authorized") is not True:
        reasons.append("loftr_trigger_contract.authorized is required")
    if refused_p112:
        reasons.append(
            "P112 edges are refused until new footage: " + ", ".join(refused_p112)
        )
    extra = planned - LOFTR_SEQUENCE_ALLOWLIST
    if extra:
        reasons.append(
            "planned edges are outside the official G6.1 allowlist: "
            + ", ".join("|".join(edge) for edge in sorted(extra))
        )
    ok = not reasons and bool(planned)
    if not planned and not reasons:
        reasons.append("no allowlisted G6.1 failure edges to repair")
        ok = False
    return {
        "ok": ok,
        "reasons": reasons,
        "planned_edges": ["|".join(edge) for edge in sorted(planned)],
        "refused_p112_edges": refused_p112,
    }


def re_gate_command(work_dir: Path) -> str:
    gate = work_dir / "gates" / "S5_7_S6_geometry.json"
    return (
        "python3 sites/fuhe_bridge/tools/audit_map_geometry.py "
        f"--model {work_dir / 'final_fixed'} "
        f"--image-root {work_dir / 'images'} "
        f"--frame-manifest {work_dir / 'frame_manifest.json'} "
        f"--corpus-manifest {work_dir / 'corpus_manifest.json'} "
        f"--forced-pairs {work_dir / 'forced_bridges.txt'} "
        f"--forced-manifest {work_dir / 'forced_bridges.json'} "
        f"--twoview {work_dir / 'twoview.pt'} "
        f"--s4-gate {work_dir / 'gates' / 'S4_doppelgangers.json'} "
        f"--s5-metrics {work_dir / 'gates' / 'S5_fixed_intrinsics.json'} "
        f"--s5-7-gate {work_dir / 'gates' / 'S5_7_independent_sim3.json'} "
        f"--out {gate}"
    )


def _filter_matches(
    matches: Iterable[dict[str, Any]], planned: set[tuple[str, str]]
) -> list[dict[str, Any]]:
    injected: list[dict[str, Any]] = []
    for item in matches:
        try:
            edge = normalize_edge(item.get("edge"))
        except (TypeError, ValueError):
            continue
        if edge not in planned or P112 in edge:
            continue
        injected.append({"edge": list(edge), "pairs": item.get("pairs", [])})
    return injected


def _copy_database(source_db: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source_db.name
    if dest.resolve() == source_db.resolve():
        raise ValueError("refusing in-place database mutation")
    shutil.copy2(source_db, dest)
    return dest


def _record_injection(dest_db: Path, injected: list[dict[str, Any]]) -> None:
    sidecar = dest_db.parent / "injected_matches.json"
    sidecar.write_text(
        json.dumps({"edges": injected}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        connection = sqlite3.connect(dest_db)
    except sqlite3.Error:
        return
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS hotspot_injected_edges "
            "(edge TEXT PRIMARY KEY, pair_count INTEGER NOT NULL)"
        )
        connection.executemany(
            "INSERT OR REPLACE INTO hotspot_injected_edges(edge, pair_count) "
            "VALUES (?, ?)",
            [
                ("|".join(item["edge"]), len(item.get("pairs") or []))
                for item in injected
            ],
        )
        connection.commit()
    except sqlite3.Error:
        connection.rollback()
    finally:
        connection.close()


def repair_hotspot_tracks(
    *,
    source_db: Path,
    dest_dir: Path,
    reachability: dict[str, Any] | None,
    trigger: dict[str, Any] | None,
    geometry: dict[str, Any] | None,
    matches: Iterable[dict[str, Any]] | None = None,
    requested_edges: Iterable[object] | None = None,
    geometry_gate_path: Path | None = None,
    run_ba: bool = False,
    input_model: Path | None = None,
) -> dict[str, Any]:
    preflight = evaluate_repair_preflight(
        reachability=reachability,
        trigger=trigger,
        geometry=geometry,
        requested_edges=requested_edges,
    )
    report: dict[str, Any] = {
        "schema_version": "fuhe-hotspot-track-repair-v1",
        "promotion_allowed": False,
        "database_modified": False,
        "model_modified": False,
        "source_database": str(source_db),
        "dest_dir": str(dest_dir),
        "preflight": preflight,
        "injected_edges": [],
        "re_gate_command": re_gate_command(dest_dir),
    }
    if not preflight["ok"]:
        return report

    dest_db = _copy_database(source_db, dest_dir)
    planned = {normalize_edge(edge) for edge in preflight["planned_edges"]}
    injected = _filter_matches(matches or [], planned)
    _record_injection(dest_db, injected)
    report["database_modified"] = True
    report["copied_database"] = str(dest_db)
    report["injected_edges"] = ["|".join(item["edge"]) for item in injected]

    if run_ba and input_model is not None and input_model.is_dir() and FIXED_BA.is_file():
        output_model = dest_dir / "gluemap" / "gluemap_fixed_intrinsics_ba_repaired"
        subprocess.run(
            [
                sys.executable,
                str(FIXED_BA),
                "--run-dir",
                str(dest_dir),
                "--input-model",
                str(input_model),
                "--output-model",
                str(output_model),
            ],
            check=False,
        )
        report["model_modified"] = output_model.is_dir()
        report["fixed_ba_command"] = [
            sys.executable,
            str(FIXED_BA),
            "--run-dir",
            str(dest_dir),
            "--input-model",
            str(input_model),
            "--output-model",
            str(output_model),
        ]

    gate_path = geometry_gate_path or (dest_dir / "gates" / "S5_7_S6_geometry.json")
    if gate_path.is_file():
        report["promotion_allowed"] = geometry_gate_promotable(_read_json(gate_path))
        report["geometry_gate"] = str(gate_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--dest-dir", type=Path, required=True)
    parser.add_argument("--reachability", type=Path, required=True)
    parser.add_argument("--trigger", type=Path, required=False)
    parser.add_argument("--geometry-gate", type=Path, required=False)
    parser.add_argument("--matches", type=Path, required=False)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-ba", action="store_true")
    parser.add_argument("--input-model", type=Path, required=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reachability = _read_json(args.reachability) if args.reachability.is_file() else None
    trigger = _read_json(args.trigger) if args.trigger and args.trigger.is_file() else None
    geometry = (
        _read_json(args.geometry_gate)
        if args.geometry_gate and args.geometry_gate.is_file()
        else None
    )
    matches_payload = (
        _read_json(args.matches) if args.matches and args.matches.is_file() else {}
    )
    matches = matches_payload.get("matches", matches_payload.get("edges", []))
    if trigger is None and geometry is not None:
        failing = failing_official_edges(geometry)
        if failing:
            trigger = loftr_trigger_contract(
                next(iter(sorted(failing))),
                {"G5.1": True, "G5.7": True, "G6.1": False, "G6.3": True},
                ghost_check_id="G6.1",
                blocking_edges=failing,
            )
    report = repair_hotspot_tracks(
        source_db=args.source_db,
        dest_dir=args.dest_dir,
        reachability=reachability,
        trigger=trigger,
        geometry=geometry,
        matches=matches if isinstance(matches, list) else [],
        run_ba=args.run_ba,
        input_model=args.input_model,
        geometry_gate_path=(
            args.dest_dir / "gates" / "S5_7_S6_geometry.json"
        ),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"re-gate: {report['re_gate_command']}", flush=True)
    if not report["preflight"]["ok"]:
        raise SystemExit("; ".join(report["preflight"]["reasons"]))


if __name__ == "__main__":
    main()
