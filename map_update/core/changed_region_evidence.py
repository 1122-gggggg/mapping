#!/usr/bin/env python3
"""Promote changed-region candidates only after multi-session evidence.

This does not invalidate or replace map points.  It turns one or more
observation_stats.json files into an explicit review artifact that a future
tile-replace workflow can consume.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


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
            low = int(tile.get("low_support_frames", 0))
            if low <= 0:
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


def aggregate_changed_region_evidence(
    payloads: list[tuple[str, dict]],
    min_sessions: int = 2,
    min_low_support_frames: int = 2,
    min_total_low_support_frames: int | None = None,
    group_by: str = "seq_tile",
) -> dict:
    min_total = (
        int(min_total_low_support_frames)
        if min_total_low_support_frames is not None
        else int(min_sessions) * int(min_low_support_frames)
    )
    groups: dict[str, dict] = {}
    for source, payload in payloads:
        for row in iter_candidate_rows(payload, source):
            low = int(row.get("low_support_frames", 0))
            if low < min_low_support_frames:
                continue
            key = candidate_key(row, group_by)
            grp = groups.setdefault(key, {
                "key": key,
                "tile": row.get("tile"),
                "sessions": set(),
                "sources": set(),
                "total_low_support_frames": 0,
                "total_query_kp": 0,
                "total_matches": 0,
                "total_inliers": 0,
                "min_support_ratio": None,
                "evidence": [],
            })
            seq = str(row.get("seq", "unknown"))
            grp["sessions"].add(seq)
            grp["sources"].add(str(row.get("source", source)))
            grp["total_low_support_frames"] += low
            grp["total_query_kp"] += int(row.get("query_kp", 0))
            grp["total_matches"] += int(row.get("matches", 0))
            grp["total_inliers"] += int(row.get("inliers", 0))
            support_ratio = row.get("support_ratio")
            if support_ratio is not None:
                support_ratio = float(support_ratio)
                current = grp["min_support_ratio"]
                grp["min_support_ratio"] = support_ratio if current is None else min(current, support_ratio)
            grp["evidence"].append(row)

    observed = []
    promoted = []
    for grp in groups.values():
        row = {
            **{k: v for k, v in grp.items() if k not in {"sessions", "sources"}},
            "sessions": sorted(grp["sessions"]),
            "sources": sorted(grp["sources"]),
            "session_count": len(grp["sessions"]),
            "source_count": len(grp["sources"]),
            "decision": "hold",
        }
        if row["session_count"] >= min_sessions and row["total_low_support_frames"] >= min_total:
            row["decision"] = "promote_for_review"
            promoted.append(row)
        observed.append(row)

    observed.sort(key=lambda r: (-r["session_count"], -r["total_low_support_frames"], str(r["key"])))
    promoted.sort(key=lambda r: (-r["session_count"], -r["total_low_support_frames"], str(r["key"])))
    return {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "thresholds": {
            "min_sessions": int(min_sessions),
            "min_low_support_frames": int(min_low_support_frames),
            "min_total_low_support_frames": int(min_total),
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
    lines.append("- Action: review only; no old points are deleted or invalidated.")
    lines.append("")
    lines.append("| Key | Sessions | Total low-support frames | Min support ratio | Decision |")
    lines.append("|---|---:|---:|---:|---|")
    for row in report["observed"]:
        min_ratio = row.get("min_support_ratio")
        min_ratio_text = "-" if min_ratio is None else f"{float(min_ratio):.4f}"
        lines.append(
            f"| {row['key']} | {row['session_count']} | {row['total_low_support_frames']} | "
            f"{min_ratio_text} | {row['decision']} |"
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
