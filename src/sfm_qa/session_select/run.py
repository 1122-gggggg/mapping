"""Video-first session selection runner. Does not invent STRONG geometry."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

from sfm_qa.session_select.edges import build_session_edges
from sfm_qa.session_select.export import write_selection_outputs
from sfm_qa.session_select.image_qa import evaluate_video
from sfm_qa.session_select.ingest import discover_sessions
from sfm_qa.session_select.motion import scan_video

try:
    from sfm_qa.session_select.types import ROLES, SessionQuality
except ImportError:  # sibling types may land later
    ROLES = (
        "BASE_CORE",
        "BASE_SUPPORT",
        "APPEARANCE_REF",
        "GEOMETRY_REINFORCEMENT",
        "UPDATE_CANDIDATE",
        "NEW_SUBMAP",
        "QUARANTINE",
        "REJECT",
        "VALIDATION_ONLY",
    )

    @dataclass
    class SessionQuality:  # type: ignore[no-redef]
        session_id: str
        timestamp: str | None
        num_frames: int
        num_keyframes: int
        registered_ratio: float | None
        sharpness_median: float | None
        sharpness_p10: float | None
        num_tracks: int | None
        num_observations: int | None
        median_track_length: float | None
        long_track_ratio: float | None
        reprojection_rmse: float | None
        reprojection_p90: float | None
        parallax_median_deg: float | None
        parallax_p10_deg: float | None
        positive_depth_ratio: float | None
        convex_hull_coverage: float | None
        grid_occupancy_4x4: float | None
        fim_condition_number: float | None
        fim_logdet: float | None
        rotation_cycle_error: float | None
        translation_consistency: float | None
        connected_components: int | None
        average_degree: float | None
        num_bridges: int | None
        num_articulation_points: int | None
        fiedler_value: float | None
        internal_quality_score: float
        internal_status: str
        reasons: tuple[str, ...]


def _construct(cls: type, **kwargs: Any) -> Any:
    if is_dataclass(cls):
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in kwargs.items() if key in allowed})
    return cls(**kwargs)


def _lookup(config: Mapping[str, Any] | None, dotted: str, default: Any = None) -> Any:
    if config:
        try:
            from sfm_qa.session_select.config import lookup

            return lookup(dict(config), dotted, default)
        except Exception:
            node: Any = config
            for part in dotted.split("."):
                if not isinstance(node, Mapping) or part not in node:
                    return default
                node = node[part]
            return node
    return default


def _resolve_config(config: Any) -> dict[str, Any]:
    if config is None:
        try:
            from sfm_qa.session_select.config import load_config

            loaded = load_config()
            return dict(loaded) if isinstance(loaded, Mapping) else {}
        except Exception:
            return {}
    if isinstance(config, (str, Path)):
        path = Path(config)
        try:
            from sfm_qa.session_select.config import load_config

            loaded = load_config(path)
            return dict(loaded) if isinstance(loaded, Mapping) else {}
        except Exception:
            try:
                import yaml

                payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                return dict(payload) if isinstance(payload, Mapping) else {}
            except Exception:
                return {}
    if isinstance(config, Mapping):
        return dict(config)
    return {}


def _video_only_status(
    qa: Mapping[str, Any],
    motion: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[str, float, list[str]]:
    reasons = [str(item) for item in (qa.get("reasons") or ())]
    reasons.extend(str(item) for item in (motion.get("reasons") or ()))
    reasons.append("video_only_never_strong_geometry")
    sharp = qa.get("sharpness_p10")
    reject_sharp = float(_lookup(config, "internal_status.reject_sharpness_p10_below", 15.0) or 15.0)
    if sharp is not None and float(sharp) < reject_sharp:
        reasons.append(f"sharpness_p10_below_heuristic_{reject_sharp}")
        return "REJECT", 0.05, reasons

    histogram = motion.get("histogram") or motion.get("classes") or {}
    total = sum(int(value) for value in histogram.values()) if isinstance(histogram, Mapping) else 0
    parallax_n = int(histogram.get("parallax", 0) or 0) if isinstance(histogram, Mapping) else 0
    has_parallax = total > 0 and (parallax_n / total) >= 0.25
    missing = sharp is None or "missing_opencv" in reasons or "unreadable_video" in reasons
    if missing:
        reasons.append("video_only_missing_geometry_fail_closed")
        return "WEAK", 0.2, reasons

    under = qa.get("underexposed_ratio")
    over = qa.get("overexposed_ratio")
    if under is not None and float(under) >= 0.85:
        reasons.append("mostly_underexposed")
        return "WEAK", 0.25, reasons
    if over is not None and float(over) >= 0.85:
        reasons.append("mostly_overexposed")
        return "WEAK", 0.25, reasons

    if float(sharp) >= reject_sharp and has_parallax:
        score = min(0.75, 0.4 + min(1.0, float(sharp) / 200.0) * 0.35)
        reasons.append("sharp_with_parallax_motion")
        return "USABLE", float(score), reasons

    reasons.append("video_only_not_sharp_and_parallax")
    score = min(0.45, 0.2 + (min(1.0, float(sharp) / 200.0) * 0.2 if sharp is not None else 0.0))
    return "WEAK", float(score), reasons


def _session_quality(record: Any, qa: Mapping[str, Any], motion: Mapping[str, Any], config: Mapping[str, Any]) -> Any:
    status, score, reasons = _video_only_status(qa, motion, config)
    sampled = int(qa.get("sampled") or 0)
    num_frames = int(getattr(record, "num_frames", 0) or 0)
    return _construct(
        SessionQuality,
        session_id=str(record.session_id),
        timestamp=getattr(record, "timestamp", None),
        num_frames=num_frames,
        num_keyframes=sampled,
        registered_ratio=None,
        sharpness_median=qa.get("sharpness_median"),
        sharpness_p10=qa.get("sharpness_p10"),
        num_tracks=None,
        num_observations=None,
        median_track_length=None,
        long_track_ratio=None,
        reprojection_rmse=None,
        reprojection_p90=None,
        parallax_median_deg=None,
        parallax_p10_deg=None,
        positive_depth_ratio=None,
        convex_hull_coverage=None,
        grid_occupancy_4x4=None,
        fim_condition_number=None,
        fim_logdet=None,
        rotation_cycle_error=None,
        translation_consistency=None,
        connected_components=None,
        average_degree=None,
        num_bridges=None,
        num_articulation_points=None,
        fiedler_value=None,
        internal_quality_score=float(score),
        internal_status=status,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _greedy_select(qualities: list[Any], edges: list[Any], config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from sfm_qa.session_select.select_core import greedy_select_core

        result = greedy_select_core(qualities, edges, config)
        if isinstance(result, Mapping):
            return dict(result)
    except Exception:
        pass
    eligible = [row for row in qualities if getattr(row, "internal_status", None) in {"STRONG", "USABLE"}]
    if not eligible:
        return {
            "core": [],
            "support": [],
            "selected": [],
            "scores": {},
            "stop_reason": "no_strong_or_usable_seed",
            "seed": None,
        }
    # Timestamp is intentionally omitted from the ranking key.
    seed_row = max(eligible, key=lambda row: (float(row.internal_quality_score), row.session_id))
    return {
        "core": [seed_row.session_id],
        "support": [],
        "selected": [seed_row.session_id],
        "scores": {},
        "stop_reason": "video_only_seed_only_no_admissible_neighbor",
        "seed": seed_row.session_id,
    }


def _classify(
    qualities: list[Any],
    edges: list[Any],
    core: list[str],
    support: list[str],
    config: Mapping[str, Any],
    extra: Mapping[str, Any],
) -> dict[str, str]:
    try:
        from sfm_qa.session_select.classify_remainder import classify_remainder

        assigned = classify_remainder(qualities, edges, core, support, config, extra=extra)
        if isinstance(assigned, Mapping):
            return {str(key): str(value) for key, value in assigned.items()}
    except Exception:
        pass
    roles: dict[str, str] = {}
    for sid in core:
        roles[sid] = "BASE_CORE"
    for sid in support:
        roles[sid] = "BASE_SUPPORT"
    for row in qualities:
        sid = row.session_id
        if sid in roles:
            continue
        status = getattr(row, "internal_status", "WEAK")
        if status in {"REJECT", "INCONSISTENT"}:
            roles[sid] = "REJECT"
            continue
        if status == "WEAK" and row.sharpness_p10 is not None and float(row.sharpness_p10) <= 0:
            roles[sid] = "REJECT"
            continue
        if status in {"STRONG", "USABLE"}:
            roles[sid] = "NEW_SUBMAP"
        else:
            roles[sid] = "QUARANTINE"
    return roles


def _reason_map(qualities: list[Any], roles: Mapping[str, str]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for row in qualities:
        role = roles.get(row.session_id, "QUARANTINE")
        bits = [f"role={role}", f"status={row.internal_status}"]
        bits.extend(list(getattr(row, "reasons", ()) or ())[:6])
        if role == "NEW_SUBMAP":
            bits.append("no_reliable_geometric_edge")
        reasons[row.session_id] = "; ".join(bits)
    return reasons


def select_sessions(
    video_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    config: Any = None,
    maps_dir: str | Path | None = None,
    **aliases: Any,
) -> dict[str, Any]:
    """QA videos, score fail-closed edges, select core, classify leftovers, export."""

    if video_dir is None:
        video_dir = aliases.get("videos")
    if output_dir is None:
        output_dir = aliases.get("output")
    if maps_dir is None:
        maps_dir = aliases.get("maps")
    if isinstance(config, (str, Path)) and maps_dir is None and aliases.get("config") is None:
        # Positional (videos, output, maps) from callers that treat the third slot as maps.
        maybe_maps = Path(config)
        if maybe_maps.exists() and not str(maybe_maps).lower().endswith((".yaml", ".yml", ".json")):
            maps_dir = maybe_maps
            config = aliases.get("config")

    if video_dir is None or output_dir is None:
        raise TypeError("select_sessions requires video_dir/output_dir (or videos/output)")

    cfg = _resolve_config(config)
    records = discover_sessions(video_dir, cfg)
    qualities = []
    for record in records:
        qa = evaluate_video(record.video_path)
        motion = scan_video(record.video_path)
        qualities.append(_session_quality(record, qa, motion, cfg))

    session_ids = [row.session_id for row in qualities]
    edges = build_session_edges(session_ids, maps_dir=maps_dir, config=cfg)
    for edge in edges:
        if getattr(edge, "status", None) == "STRONG" and (
            int(getattr(edge, "num_verified_pairs", 0) or 0) <= 0
            or int(getattr(edge, "independent_bridge_groups", 0) or 0) <= 0
        ):
            edge.status = "REJECT"
            if hasattr(edge, "reasons"):
                edge.reasons = tuple(edge.reasons) + ("maps_absent_or_unverified_not_strong",)

    extra = {
        "change_score": {row.session_id: 0.0 for row in qualities},
        "loc": {},
        "influences": {},
        "appearance_shift": {},
    }
    selection = _greedy_select(qualities, edges, cfg)
    core = list(selection.get("core") or [])
    support = list(selection.get("support") or [])
    if not core and support:
        core, support = [support[0]], support[1:]
    roles = _classify(qualities, edges, core, support, cfg, extra)
    for sid, role in list(roles.items()):
        if role not in ROLES:
            roles[sid] = "QUARANTINE"
    reasons = _reason_map(qualities, roles)

    dest = Path(output_dir)
    write_selection_outputs(
        dest,
        qualities=qualities,
        edges=edges,
        roles=roles,
        reasons=reasons,
        config=cfg,
        extra=extra,
        selection=selection,
    )
    role_counts = {name: 0 for name in ROLES}
    role_counts.update(Counter(roles.values()))
    assignments = [
        {"session_id": sid, "role": roles[sid], "reason": reasons.get(sid, "")}
        for sid in session_ids
    ]
    return {
        "sessions": session_ids,
        "roles": roles,
        "assignments": assignments,
        "role_counts": role_counts,
        "output": str(dest),
        "selection": selection,
        "qualities": qualities,
        "edges": edges,
    }


__all__ = ["select_sessions"]
