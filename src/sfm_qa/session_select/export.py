"""Write session-selection artifacts. Site-agnostic; no ranking by timestamp."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROLE_FILES = {
    "BASE_CORE": "base_core_sessions.txt",
    "BASE_SUPPORT": "base_support_sessions.txt",
    "APPEARANCE_REF": "appearance_ref_sessions.txt",
    "GEOMETRY_REINFORCEMENT": "geometry_reinforcement_sessions.txt",
    "UPDATE_CANDIDATE": "update_candidate_sessions.txt",
    "NEW_SUBMAP": "new_submap_sessions.txt",
    "QUARANTINE": "quarantine_sessions.txt",
    "REJECT": "reject_sessions.txt",
    "VALIDATION_ONLY": "validation_only_sessions.txt",
}

SESSION_ROLE_COLUMNS = [
    "session_id",
    "timestamp",
    "internal_status",
    "internal_quality_score",
    "num_frames",
    "num_keyframes",
    "tracks",
    "observations",
    "coverage_score",
    "information_score",
    "graph_score",
    "redundancy_score",
    "risk_score",
    "track_cost",
    "base_information_gain",
    "base_coverage_gain",
    "num_strong_session_edges",
    "num_independent_bridge_groups",
    "critical_bridge_dependency",
    "change_score",
    "role",
    "reason",
]


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    return str(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in columns})


def _row_mapping(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    payload = getattr(row, "__dict__", None)
    if isinstance(payload, dict):
        return dict(payload)
    return {}


def _quality_columns(qualities: Sequence[Any]) -> list[str]:
    if qualities:
        payload = _row_mapping(qualities[0])
        if payload:
            return list(payload.keys())
    return [
        "session_id",
        "timestamp",
        "num_frames",
        "num_keyframes",
        "internal_quality_score",
        "internal_status",
        "reasons",
    ]


def _edge_columns(edges: Sequence[Any]) -> list[str]:
    if edges:
        payload = _row_mapping(edges[0])
        if payload:
            return list(payload.keys())
    return [
        "session_a",
        "session_b",
        "num_candidate_pairs",
        "num_verified_pairs",
        "independent_bridge_groups",
        "edge_quality_score",
        "status",
        "reasons",
    ]


def write_session_quality_csv(path: Path, qualities: Sequence[Any]) -> None:
    rows = [_row_mapping(row) for row in qualities]
    _write_csv(path, rows, _quality_columns(qualities))


def write_session_edges_csv(path: Path, edges: Sequence[Any]) -> None:
    rows = [_row_mapping(row) for row in edges]
    _write_csv(path, rows, _edge_columns(edges))


def write_role_lists(output_dir: Path, roles: Mapping[str, str]) -> None:
    grouped: dict[str, list[str]] = {role: [] for role in ROLE_FILES}
    for session_id, role in roles.items():
        grouped.setdefault(role, []).append(session_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    for role, filename in ROLE_FILES.items():
        path = output_dir / filename
        names = sorted(grouped.get(role) or [])
        if names:
            path.write_text("\n".join(names) + "\n", encoding="utf-8")
        else:
            path.write_text("# none\n", encoding="utf-8")


def _objective_terms(
    qualities: Sequence[Any],
    edges: Sequence[Any],
    selected: Sequence[str],
    config: Mapping[str, Any] | None,
) -> dict[str, float]:
    empty = {
        "coverage": 0.0,
        "information": 0.0,
        "connectivity": 0.0,
        "redundancy": 0.0,
        "risk": 0.0,
        "track_cost": 0.0,
        "observations": 0.0,
        "tracks": 0.0,
    }
    try:
        from sfm_qa.session_select.objective import compute_objective_terms

        terms = compute_objective_terms(qualities, edges, selected, config or {})
        if isinstance(terms, Mapping):
            merged = dict(empty)
            for key in empty:
                if key in terms and terms[key] is not None:
                    merged[key] = float(terms[key])
            return merged
    except Exception:
        pass
    return empty


def build_role_rows(
    qualities: Sequence[Any],
    edges: Sequence[Any],
    roles: Mapping[str, str],
    reasons: Mapping[str, str],
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    extra = extra or {}
    change_score = extra.get("change_score") or {}
    core = [sid for sid, role in roles.items() if role in {"BASE_CORE", "BASE_SUPPORT"}]
    base_terms = _objective_terms(qualities, edges, core, config) if core else {}
    rows: list[dict[str, Any]] = []
    for row in qualities:
        sid = getattr(row, "session_id", None)
        if sid is None:
            continue
        alone = _objective_terms(qualities, edges, [sid], config)
        with_base = _objective_terms(qualities, edges, core + [sid], config) if core else alone
        incident = [
            edge
            for edge in edges
            if sid in {getattr(edge, "session_a", None), getattr(edge, "session_b", None)}
        ]
        strong = sum(1 for edge in incident if getattr(edge, "status", None) == "STRONG")
        groups = max((getattr(edge, "independent_bridge_groups", 0) or 0 for edge in incident), default=0)
        critical = any(getattr(edge, "is_critical_bridge", False) for edge in incident)
        info_gain = 0.0
        cov_gain = 0.0
        if core and sid not in core:
            info_gain = with_base["information"] - base_terms.get("information", 0.0)
            cov_gain = with_base["coverage"] - base_terms.get("coverage", 0.0)
        rows.append(
            {
                "session_id": sid,
                "timestamp": getattr(row, "timestamp", None) or "",
                "internal_status": getattr(row, "internal_status", ""),
                "internal_quality_score": getattr(row, "internal_quality_score", 0.0),
                "num_frames": getattr(row, "num_frames", 0),
                "num_keyframes": getattr(row, "num_keyframes", 0),
                "tracks": getattr(row, "num_tracks", None),
                "observations": getattr(row, "num_observations", None),
                "coverage_score": alone["coverage"],
                "information_score": alone["information"],
                "graph_score": alone["connectivity"],
                "redundancy_score": alone["redundancy"],
                "risk_score": alone["risk"],
                "track_cost": alone["track_cost"],
                "base_information_gain": info_gain,
                "base_coverage_gain": cov_gain,
                "num_strong_session_edges": strong,
                "num_independent_bridge_groups": groups,
                "critical_bridge_dependency": bool(critical),
                "change_score": change_score.get(sid, 0.0),
                "role": roles.get(sid, "QUARANTINE"),
                "reason": reasons.get(sid, "; ".join(getattr(row, "reasons", ()) or ())),
            }
        )
    return rows


def write_session_roles_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_csv(path, rows, SESSION_ROLE_COLUMNS)


def _fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_selection_report(
    *,
    qualities: Sequence[Any],
    edges: Sequence[Any],
    roles: Mapping[str, str],
    reasons: Mapping[str, str],
    selection: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> str:
    del config
    selection = dict(selection or {})
    by_role: dict[str, list[str]] = {role: [] for role in ROLE_FILES}
    for sid, role in roles.items():
        by_role.setdefault(role, []).append(sid)
    quality_by_id = {getattr(row, "session_id", None): row for row in qualities}
    lines = [
        "# Session selection report",
        "",
        "Video-first selection. No COLMAP model is required.",
        "Numeric gates are **heuristic**, not fitted site cutoffs.",
        "VPR / retrieval is **not** a geometric edge.",
        "Timestamp is recorded but **never** used to rank Base.",
        "Fail-closed: uncertain → QUARANTINE; no reliable edge → NEW_SUBMAP;",
        "never force-merge on one critical or AMBIGUOUS bridge.",
        "",
        "## BASE_CORE",
        "",
    ]
    seed = selection.get("seed")
    lines.append(
        f"- Seed (max internal quality among STRONG/USABLE): `{seed}`. "
        "Timestamp was not a ranking key. Seed is not automatically the whole Base."
    )
    lines.append(f"- Greedy selected: {list(selection.get('selected') or []) or '(none)'}.")
    lines.append(f"- Stop reason: `{selection.get('stop_reason')}`.")
    core = sorted(by_role.get("BASE_CORE") or [])
    if not core:
        lines.append("- No BASE_CORE.")
    for sid in core:
        row = quality_by_id.get(sid)
        if row is None:
            lines.append(f"- `{sid}`")
            continue
        lines.append(
            f"- `{sid}`: status={row.internal_status}, "
            f"score={_fmt(row.internal_quality_score)}, "
            f"sharpness_p10={_fmt(getattr(row, 'sharpness_p10', None))}. "
            f"Timestamp `{getattr(row, 'timestamp', None)}` unused as rank."
        )
        extra = "; ".join(getattr(row, "reasons", ()) or ())
        if extra:
            lines.append(f"  - {extra}")
    lines.extend(["", "## Other roles", ""])
    for role in ROLE_FILES:
        if role == "BASE_CORE":
            continue
        names = sorted(by_role.get(role) or [])
        lines.append(f"### {role}")
        if not names:
            lines.append("- (none)")
        for sid in names:
            lines.append(f"- `{sid}`: {reasons.get(sid, '')}".rstrip())
        lines.append("")
    lines.extend(
        [
            "## Edges",
            "",
            f"- Count: {len(edges)}.",
            "- STRONG requires verified pairs **and** independent bridge groups.",
            "- Zero verified + zero independent groups is never STRONG.",
            "",
        ]
    )
    if not edges:
        lines.append("- No geometric edges (video-only / candidate-only run).")
    for edge in edges:
        lines.append(
            f"- `{edge.session_a}`–`{edge.session_b}`: status={edge.status}, "
            f"verified={getattr(edge, 'num_verified_pairs', 0)}, "
            f"groups={getattr(edge, 'independent_bridge_groups', 0)}, "
            f"candidates={getattr(edge, 'num_candidate_pairs', 0)}."
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def write_selection_outputs(
    output_dir: str | Path,
    *,
    qualities: Sequence[Any],
    edges: Sequence[Any],
    roles: Mapping[str, str],
    reasons: Mapping[str, str] | None = None,
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the Stage-0 CSV / role-list / markdown artifacts."""

    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    reasons = dict(reasons or {})
    for row in qualities:
        sid = getattr(row, "session_id", None)
        if sid and sid not in reasons:
            reasons[sid] = "; ".join(getattr(row, "reasons", ()) or ())
    role_rows = build_role_rows(qualities, edges, roles, reasons, config, extra)
    write_session_quality_csv(dest / "session_quality.csv", qualities)
    write_session_edges_csv(dest / "session_edges.csv", edges)
    write_session_roles_csv(dest / "session_roles.csv", role_rows)
    write_role_lists(dest, roles)
    report = render_selection_report(
        qualities=qualities,
        edges=edges,
        roles=roles,
        reasons=reasons,
        selection=selection,
        config=config,
    )
    (dest / "selection_report.md").write_text(report, encoding="utf-8")
    written = {
        "session_quality.csv": dest / "session_quality.csv",
        "session_edges.csv": dest / "session_edges.csv",
        "session_roles.csv": dest / "session_roles.csv",
        "selection_report.md": dest / "selection_report.md",
    }
    for filename in ROLE_FILES.values():
        written[filename] = dest / filename
    return written


__all__ = [
    "ROLE_FILES",
    "SESSION_ROLE_COLUMNS",
    "build_role_rows",
    "render_selection_report",
    "write_role_lists",
    "write_selection_outputs",
    "write_session_edges_csv",
    "write_session_quality_csv",
    "write_session_roles_csv",
]
