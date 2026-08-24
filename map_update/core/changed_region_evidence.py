#!/usr/bin/env python3
"""Promote changed-region candidates only after multi-session evidence.

This does not invalidate or replace map points.  It turns one or more
observation_stats.json files into an explicit review artifact that a future
tile-replace workflow can consume.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

COUNT_FIELDS = ("low_support_frames", "query_kp", "matches", "inliers")


def _nonneg_count(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    as_int = int(value)
    if as_int != value:
        return None
    return as_int


def _finite_number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _score_candidate_row(row: dict) -> tuple[dict | None, str | None]:
    scored = dict(row)
    for field in COUNT_FIELDS:
        count = _nonneg_count(row.get(field, 0))
        if count is None:
            return None, f"unscorable_{field}"
        scored[field] = count
    if row.get("support_ratio") is not None:
        ratio = _finite_number(row.get("support_ratio"))
        if ratio is None:
            return None, "unscorable_support_ratio"
        scored["support_ratio"] = ratio
    return scored, None


def iter_candidate_rows(payload: dict, source: str) -> list[dict]:
    rows = []
    explicit = payload.get("changed_region_candidates") or []
    if explicit:
        for row in explicit:
            item = dict(row)
            item.setdefault("source", source)
            rows.append(item)
        return rows

    for seq, sess in (payload.get("sessions") or {}).items():
        for tile in sess.get("tiles", []):
            parsed_low = _nonneg_count(tile.get("low_support_frames", 0))
            if parsed_low is not None and parsed_low <= 0:
                continue
            item = dict(tile)
            item.update({
                "seq": seq,
                "source": source,
                "reason": "derived_from_session_tile_low_support",
            })
            rows.append(item)
    return rows


def candidate_key(row: dict, group_by: str) -> str:
    if group_by == "seq_tile":
        return f"{row.get('seq', '')}:{row.get('tile', '')}"
    if group_by == "tile":
        return str(row.get("tile", ""))
    raise ValueError(f"unknown group_by {group_by}")


def _row_evidence(session_count: int, total_low: int, min_sessions: int, min_total: int) -> tuple[str, str, str]:
    if session_count < min_sessions:
        return "hold", "INSUFFICIENT_EVIDENCE", "low_session_count"
    if total_low < min_total:
        return "hold", "QUALITY_SHORTFALL", "low_total_support"
    return "promote_for_review", "REVIEW_CANDIDATE", "review_candidate"


def aggregate_changed_region_evidence(
    payloads: list[tuple[str, dict]],
    min_sessions: int = 2,
    min_low_support_frames: int = 2,
    min_total_low_support_frames: int | None = None,
    group_by: str = "seq_tile",
) -> dict:
    explicit_total = min_total_low_support_frames is not None
    min_total = (
        int(min_total_low_support_frames)
        if explicit_total
        else int(min_sessions) * int(min_low_support_frames)
    )
    groups: dict[str, dict] = {}
    suppressed: list[dict] = []
    unscorable: list[dict] = []
    candidate_count = 0
    discarded_subthreshold_frames = 0
    for source, payload in payloads:
        for row in iter_candidate_rows(payload, source):
            candidate_count += 1
            scored, reason = _score_candidate_row(row)
            if scored is None:
                unscorable.append({
                    "source": str(row.get("source", source)),
                    "key": candidate_key(row, group_by),
                    "decision_reason": reason,
                })
                continue
            low = scored["low_support_frames"]
            if low < min_low_support_frames:
                discarded_subthreshold_frames += low
                suppressed.append({
                    "source": str(scored.get("source", source)),
                    "key": candidate_key(scored, group_by),
                    "low_support_frames": low,
                    "decision_reason": "below_min_low_support_frames",
                })
                continue
            key = candidate_key(scored, group_by)
            grp = groups.setdefault(key, {
                "key": key,
                "tile": scored.get("tile"),
                "sessions": set(),
                "sources": set(),
                "total_low_support_frames": 0,
                "total_query_kp": 0,
                "total_matches": 0,
                "total_inliers": 0,
                "min_support_ratio": None,
                "evidence": [],
            })
            seq = str(scored.get("seq", "unknown"))
            grp["sessions"].add(seq)
            grp["sources"].add(str(scored.get("source", source)))
            grp["total_low_support_frames"] += low
            grp["total_query_kp"] += scored["query_kp"]
            grp["total_matches"] += scored["matches"]
            grp["total_inliers"] += scored["inliers"]
            support_ratio = scored.get("support_ratio")
            if support_ratio is not None:
                current = grp["min_support_ratio"]
                grp["min_support_ratio"] = (
                    support_ratio if current is None else min(current, support_ratio)
                )
            grp["evidence"].append(scored)

    observed = []
    promoted = []
    for grp in groups.values():
        session_count = len(grp["sessions"])
        total_low = grp["total_low_support_frames"]
        decision, evidence_status, decision_reason = _row_evidence(
            session_count, total_low, int(min_sessions), int(min_total)
        )
        row = {
            **{k: v for k, v in grp.items() if k not in {"sessions", "sources"}},
            "sessions": sorted(grp["sessions"]),
            "sources": sorted(grp["sources"]),
            "session_count": session_count,
            "source_count": len(grp["sources"]),
            "decision": decision,
            "decision_reason": decision_reason,
            "evidence_status": evidence_status,
            "session_count_margin": session_count - int(min_sessions),
            "total_low_support_margin": total_low - int(min_total),
        }
        if decision == "promote_for_review":
            promoted.append(row)
        observed.append(row)

    observed.sort(key=lambda r: (-r["session_count"], -r["total_low_support_frames"], str(r["key"])))
    promoted.sort(key=lambda r: (-r["session_count"], -r["total_low_support_frames"], str(r["key"])))

    decision_reasons: list[str] = []
    if candidate_count == 0:
        proposal_status = "EMPTY_NO_CANDIDATES"
        evidence_status = "INSUFFICIENT_EVIDENCE"
        decision_reasons.append("no_candidate_rows")
    elif promoted:
        proposal_status = "REVIEW_CANDIDATES"
        evidence_status = "WARN"
        decision_reasons.append("review_candidate")
    elif any(row["evidence_status"] == "QUALITY_SHORTFALL" for row in observed) and all(
        row["evidence_status"] != "INSUFFICIENT_EVIDENCE" for row in observed
    ):
        proposal_status = "QUALITY_SHORTFALL"
        evidence_status = "QUALITY_SHORTFALL"
        decision_reasons.append("low_total_support")
    else:
        proposal_status = "INSUFFICIENT_EVIDENCE"
        evidence_status = "INSUFFICIENT_EVIDENCE"
        if unscorable:
            decision_reasons.append("unscorable_rows")
        if suppressed and not observed:
            decision_reasons.append("below_min_low_support_frames")
        if any(row["evidence_status"] == "INSUFFICIENT_EVIDENCE" for row in observed):
            decision_reasons.append("low_session_count")
        if not decision_reasons:
            decision_reasons.append("no_scorable_groups")
    decision_reasons.append("independence_not_attested")
    decision_reasons.append("lineage_not_attested")

    return {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hard_status": "VALID",
        "evidence_status": evidence_status,
        "proposal_status": proposal_status,
        "authority": "review_only",
        "geometry_invalidated": False,
        "independence_attested": False,
        "lineage_attested": False,
        "decision_reasons": decision_reasons,
        "candidate_count": candidate_count,
        "suppressed_count": len(suppressed),
        "unscorable_count": len(unscorable),
        "discarded_subthreshold_rows": len(suppressed),
        "discarded_subthreshold_frames": discarded_subthreshold_frames,
        "suppressed": suppressed,
        "unscorable": unscorable,
        "thresholds": {
            "min_sessions": int(min_sessions),
            "min_low_support_frames": int(min_low_support_frames),
            "min_total_low_support_frames": int(min_total),
            "min_total_source": "explicit" if explicit_total else "product",
            "group_by": group_by,
        },
        "promoted": promoted,
        "observed": observed,
        "note": "No map points are invalidated by this report.",
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = ["# Changed Region Evidence", ""]
    th = report["thresholds"]
    lines.append(
        f"- Gate: sessions >= {th['min_sessions']}, low_support_frames/session >= "
        f"{th['min_low_support_frames']}, total low_support_frames >= "
        f"{th['min_total_low_support_frames']}, group_by={th['group_by']}"
    )
    lines.append(f"- hard_status: {report.get('hard_status', 'VALID')}")
    lines.append(f"- evidence_status: {report.get('evidence_status')}")
    if report.get("proposal_status") is not None:
        lines.append(f"- proposal_status: {report['proposal_status']}")
    lines.append(f"- authority: {report.get('authority', 'review_only')}")
    lines.append(
        f"- independence_attested: {str(bool(report.get('independence_attested'))).lower()}"
    )
    lines.append(
        f"- lineage_attested: {str(bool(report.get('lineage_attested'))).lower()}"
    )
    lines.append(
        "- Independence and lineage are not attested; this report is review-only "
        "and is not stability proof."
    )
    reasons = report.get("decision_reasons") or []
    lines.append(f"- decision_reasons: {', '.join(reasons) if reasons else '-'}")
    lines.append(
        f"- counts: candidates={report.get('candidate_count', 0)}, "
        f"suppressed={report.get('suppressed_count', 0)}, "
        f"unscorable={report.get('unscorable_count', 0)}"
    )
    lines.append("- Action: review only; no old points are deleted or invalidated.")
    lines.append("")
    lines.append(
        "| Key | Sessions | Total low-support frames | Min support ratio | "
        "Session margin | Total margin | Evidence | Reason | Decision |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|---|---|")
    for row in report["observed"]:
        min_ratio = row.get("min_support_ratio")
        min_ratio_text = "-" if min_ratio is None else f"{float(min_ratio):.4f}"
        lines.append(
            f"| {row['key']} | {row['session_count']} | {row['total_low_support_frames']} | "
            f"{min_ratio_text} | {row.get('session_count_margin', '-')} | "
            f"{row.get('total_low_support_margin', '-')} | "
            f"{row.get('evidence_status', '-')} | {row.get('decision_reason', '-')} | "
            f"{row['decision']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation_stats", nargs="+", help="One or more observation_stats.json files.")
    parser.add_argument("--min-sessions", type=int, default=2)
    parser.add_argument("--min-low-support-frames", type=int, default=2)
    parser.add_argument("--min-total-low-support-frames", type=int, default=0)
    parser.add_argument("--group-by", choices=["seq_tile", "tile"], default="seq_tile")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    payloads = []
    for item in args.observation_stats:
        path = Path(item)
        payloads.append((str(path), json.loads(path.read_text(encoding="utf-8"))))
    report = aggregate_changed_region_evidence(
        payloads,
        min_sessions=args.min_sessions,
        min_low_support_frames=args.min_low_support_frames,
        min_total_low_support_frames=(
            args.min_total_low_support_frames if args.min_total_low_support_frames > 0 else None
        ),
        group_by=args.group_by,
    )
    out_json = Path(args.out_json) if args.out_json else Path(args.observation_stats[0]).with_name("changed_region_evidence.json")
    out_md = Path(args.out_md) if args.out_md else out_json.with_suffix(".md")
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, out_md)
    print(f"[changed_region_evidence] wrote {out_json}")
    print(f"[changed_region_evidence] wrote {out_md}")


if __name__ == "__main__":
    main()
