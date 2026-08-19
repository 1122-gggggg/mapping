"""Session-graph edges. VPR/retrieval is never geometric proof."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

try:
    from sfm_qa.session_select.types import SessionEdgeQuality as _ImportedEdge
except ImportError:  # sibling types may land later
    _ImportedEdge = None  # type: ignore[misc, assignment]


try:
    from dataclasses import dataclass

    @dataclass
    class _LocalEdge:
        session_a: str
        session_b: str
        num_candidate_pairs: int
        num_verified_pairs: int
        num_cross_session_tracks: int | None
        num_cross_session_observations: int | None
        independent_bridge_groups: int
        inlier_count: int | None
        inlier_ratio: float | None
        rotation_consensus_deg: float | None
        translation_direction_consensus_deg: float | None
        scale_consensus: float | None
        cross_session_reprojection_error: float | None
        spatial_coverage: float | None
        cycle_support: int | None
        cycle_error: float | None
        edge_quality_score: float
        is_bridge: bool
        is_critical_bridge: bool
        status: str
        reasons: tuple[str, ...]

except Exception:  # pragma: no cover
    _LocalEdge = None  # type: ignore[misc, assignment]


def _edge_cls() -> type:
    if _ImportedEdge is not None:
        return _ImportedEdge
    if _LocalEdge is not None:
        return _LocalEdge
    raise RuntimeError("SessionEdgeQuality is unavailable")


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


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def classify_session_edge(
    session_a: str,
    session_b: str,
    num_verified_pairs: int = 0,
    independent_bridge_groups: int = 0,
    num_candidate_pairs: int = 0,
    *,
    num_cross_session_tracks: int | None = None,
    num_cross_session_observations: int | None = None,
    inlier_count: int | None = None,
    inlier_ratio: float | None = None,
    rotation_consensus_deg: float | None = None,
    translation_direction_consensus_deg: float | None = None,
    scale_consensus: float | None = None,
    cross_session_reprojection_error: float | None = None,
    spatial_coverage: float | None = None,
    cycle_support: int | None = None,
    cycle_error: float | None = None,
    consensus_ambiguous: bool = False,
    trusted_geometry: bool = False,
    config: Mapping[str, Any] | None = None,
    reasons: Sequence[str] | None = None,
    **aliases: Any,
) -> Any:
    """Fail-closed edge status. Zero verified + zero groups is never STRONG."""

    verified = _as_int(num_verified_pairs, 0)
    for key in ("verified", "n_verified", "num_verified"):
        if key in aliases and aliases[key] is not None:
            verified = _as_int(aliases[key], 0)
            break
    if verified <= 0 and num_cross_session_tracks:
        verified = _as_int(num_cross_session_tracks, 0)
    groups = _as_int(independent_bridge_groups, 0)
    for key in ("groups", "n_groups", "independent_bridges"):
        if key in aliases and aliases[key] is not None:
            groups = _as_int(aliases[key], 0)
            break
    candidates = _as_int(num_candidate_pairs, 0)
    for key in ("candidates", "vpr_candidates", "num_vpr"):
        if key in aliases and aliases[key] is not None:
            candidates = _as_int(aliases[key], 0)
            break
    tracks = num_cross_session_tracks if num_cross_session_tracks is not None else (
        verified if verified else None
    )
    notes = [str(item) for item in (reasons or ())]
    min_tracks = int(_lookup(config, "edge.min_cross_tracks_for_verified", 30) or 30)
    min_groups = int(
        _lookup(config, "selection.min_independent_bridges_for_strong_edge", 2) or 2
    )
    min_pairs = int(_lookup(config, "edge.min_verified_pairs_for_usable", 8) or 8)
    max_rot = float(_lookup(config, "edge.max_rotation_consensus_deg", 5.0) or 5.0)
    max_scale = float(_lookup(config, "edge.max_scale_consensus_rel", 0.15) or 0.15)

    status = "REJECT"
    score = 0.0
    is_bridge = False

    if verified <= 0 and groups <= 0:
        notes.append("vpr_is_not_a_geometric_edge")
        if candidates > 0:
            notes.append("vpr_candidates_only_not_an_edge")
        else:
            notes.append("no_verified_geometry")
        # Retrieval is not a geometric edge. WEAK would be admissible to core.
        status = "REJECT"
        score = 0.05 if candidates else 0.0
    elif not trusted_geometry and verified <= 0:
        notes.append("untrusted_geometry_fail_closed")
        status = "REJECT"
        score = 0.05
    else:
        if verified < min_tracks:
            notes.append(f"cross_tracks_below_heuristic_{min_tracks}")
        if groups < 1:
            notes.append("no_independent_bridge_group")
        rot = rotation_consensus_deg
        if consensus_ambiguous or (rot is not None and float(rot) > max_rot):
            notes.append("cross_session_pose_consensus_ambiguous")
            status = "AMBIGUOUS"
            score = 0.2
        elif scale_consensus is not None and float(scale_consensus) > max_scale:
            notes.append("cross_session_scale_consensus_ambiguous")
            status = "AMBIGUOUS"
            score = 0.2
        elif verified <= 0 or groups < 1:
            status = "AMBIGUOUS"
            score = 0.15
        else:
            score = min(1.0, verified / 200.0) * 0.5 + min(1.0, groups / max(min_groups, 1)) * 0.5
            if groups >= min_groups and verified >= min_tracks:
                notes.append("verified_multi_bridge_geometry")
                status = "STRONG"
                is_bridge = True
            elif verified >= min_pairs:
                notes.append("verified_but_below_strong_bridge_heuristic")
                status = "USABLE"
                is_bridge = True
            else:
                notes.append("weak_verified_support")
                status = "WEAK"
                score = min(0.4, score)

    if status == "STRONG" and (verified <= 0 or groups <= 0):
        status = "REJECT"
        is_bridge = False
        score = 0.05
        notes.append("strong_blocked_without_verified_bridges")

    return _construct(
        _edge_cls(),
        session_a=str(session_a),
        session_b=str(session_b),
        num_candidate_pairs=candidates,
        num_verified_pairs=verified,
        num_cross_session_tracks=tracks,
        num_cross_session_observations=num_cross_session_observations,
        independent_bridge_groups=groups,
        inlier_count=inlier_count if inlier_count is not None else (verified or None),
        inlier_ratio=inlier_ratio,
        rotation_consensus_deg=rotation_consensus_deg,
        translation_direction_consensus_deg=translation_direction_consensus_deg,
        scale_consensus=scale_consensus,
        cross_session_reprojection_error=cross_session_reprojection_error,
        spatial_coverage=spatial_coverage,
        cycle_support=cycle_support,
        cycle_error=cycle_error,
        edge_quality_score=float(score),
        is_bridge=bool(is_bridge),
        is_critical_bridge=False,
        status=status,
        reasons=tuple(dict.fromkeys(notes)),
    )


def _session_for_image(name: str, session_ids: Sequence[str]) -> str | None:
    text = str(name).replace("\\", "/")
    parts = text.split("/")
    stem = Path(text).stem
    for sid in session_ids:
        if not sid:
            continue
        if sid in parts:
            return sid
        if stem == sid:
            return sid
        if stem.startswith(sid + "_") or stem.startswith(sid + "-"):
            return sid
    return None


def _parse_colmap_text(
    model_dir: Path,
    session_ids: Sequence[str],
    *,
    frame_sep: int = 30,
) -> dict[tuple[str, str], dict[str, Any]]:
    images_path = model_dir / "images.txt"
    points_path = model_dir / "points3D.txt"
    if not images_path.is_file() or not points_path.is_file():
        return {}
    image_session: dict[int, str] = {}
    try:
        lines = images_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            image_id = int(parts[0])
        except ValueError:
            continue
        name = parts[9]
        sid = _session_for_image(name, session_ids)
        if sid:
            image_session[image_id] = sid
        if index < len(lines) and (not lines[index].strip().startswith("#")):
            # skip POINTS2D line
            index += 1
    if len(set(image_session.values())) < 2:
        return {}

    pair_tracks: dict[tuple[str, str], int] = {}
    pair_obs: dict[tuple[str, str], int] = {}
    pair_links: dict[tuple[str, str], list[tuple[int, int]]] = {}
    try:
        point_lines = points_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    for line in point_lines:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        if len(parts) < 8:
            continue
        track = parts[8:]
        observed: dict[str, list[int]] = {}
        for offset in range(0, len(track) - 1, 2):
            try:
                image_id = int(track[offset])
            except ValueError:
                continue
            sid = image_session.get(image_id)
            if sid:
                observed.setdefault(sid, []).append(image_id)
        sids = sorted(observed)
        if len(sids) < 2:
            continue
        for a, b in combinations(sids, 2):
            key = (a, b)
            pair_tracks[key] = pair_tracks.get(key, 0) + 1
            pair_obs[key] = pair_obs.get(key, 0) + len(observed[a]) + len(observed[b])
            for ia in observed[a]:
                for ib in observed[b]:
                    pair_links.setdefault(key, []).append((ia, ib))

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, count in pair_tracks.items():
        links = pair_links.get(key) or []
        groups = _cluster_bridge_pairs(links, frame_sep)
        out[key] = {
            "num_verified_pairs": count,
            "num_cross_session_tracks": count,
            "num_cross_session_observations": pair_obs.get(key),
            "independent_bridge_groups": groups,
            "trusted_geometry": True,
        }
    return out


def _cluster_bridge_pairs(pairs: Sequence[tuple[int, int]], sep: int) -> int:
    if not pairs:
        return 0
    parent = list(range(len(pairs)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_l, root_r = find(left), find(right)
        if root_l != root_r:
            parent[root_r] = root_l

    for i, (ai, bi) in enumerate(pairs):
        for j in range(i):
            aj, bj = pairs[j]
            if abs(ai - aj) < sep and abs(bi - bj) < sep:
                union(i, j)
    return len({find(i) for i in range(len(pairs))})


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _candidate_from_mapping(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    a = payload.get("session_a") or payload.get("a") or payload.get("query")
    b = payload.get("session_b") or payload.get("b") or payload.get("reference")
    if not a or not b or a == b:
        return None
    return {
        "session_a": str(a),
        "session_b": str(b),
        "num_candidate_pairs": _as_int(
            _first(
                payload.get("num_candidate_pairs"),
                payload.get("mutual_top1"),
                payload.get("mutual"),
                payload.get("candidates"),
                1,
            ),
            1,
        ),
        "num_verified_pairs": _as_int(
            _first(
                payload.get("num_verified_pairs"),
                payload.get("verified"),
                payload.get("num_cross_session_tracks"),
                0,
            ),
            0,
        ),
        "independent_bridge_groups": _as_int(
            _first(
                payload.get("independent_bridge_groups"),
                payload.get("independent_bridges"),
                0,
            ),
            0,
        ),
    }


def load_vpr_candidates(payload: Mapping[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    """Parse retrieval-only pair lists. Never promotes them to geometry."""

    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    if not payload:
        return pairs

    def add(row: Mapping[str, Any]) -> None:
        parsed = _candidate_from_mapping(row)
        if parsed is None:
            return
        a, b = parsed["session_a"], parsed["session_b"]
        key = (a, b) if a <= b else (b, a)
        current = pairs.get(key)
        if current is None or parsed["num_candidate_pairs"] > current["num_candidate_pairs"]:
            pairs[key] = parsed

    if isinstance(payload.get("pairs"), list):
        for row in payload["pairs"]:
            if isinstance(row, Mapping):
                add(row)
    for key, value in payload.items():
        if key in {"pairs", "nodes", "sessions"}:
            continue
        if isinstance(value, Mapping):
            nested = _candidate_from_mapping(value)
            if nested:
                add(nested)
                continue
            for other, inner in value.items():
                if isinstance(inner, Mapping):
                    row = dict(inner)
                    row.setdefault("session_a", str(key))
                    row.setdefault("session_b", str(other))
                    add(row)
                elif isinstance(inner, (int, float)):
                    add({"session_a": str(key), "session_b": str(other), "num_candidate_pairs": int(inner)})
    return pairs


def _discover_map_models(maps_dir: Path) -> list[Path]:
    found: list[Path] = []
    if (maps_dir / "images.txt").is_file() and (maps_dir / "points3D.txt").is_file():
        found.append(maps_dir)
    try:
        children = list(maps_dir.rglob("images.txt"))
    except OSError:
        children = []
    for images in children:
        model = images.parent
        if (model / "points3D.txt").is_file() and model not in found:
            found.append(model)
    return found[:8]


def _load_maps_dir_candidates(maps_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    patterns = ("*vpr*.json", "*pairs*.json", "*overlap*.json", "session_edges.json")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(maps_dir.glob(pattern))
        files.extend(maps_dir.glob(f"*/{pattern}"))
    for path in files:
        payload = _load_json(path)
        if isinstance(payload, Mapping):
            for key, row in load_vpr_candidates(payload).items():
                pairs.setdefault(key, row)
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, Mapping):
                    for key, row in load_vpr_candidates({"pairs": [item]}).items():
                        pairs.setdefault(key, row)
    return pairs


def build_session_edges(
    session_ids: Sequence[str],
    *,
    maps_dir: str | Path | None = None,
    vpr_payload: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> list[Any]:
    """Candidate-only unless a map supplies verified tracks + independent groups."""

    ids = [str(sid) for sid in session_ids if sid]
    if len(ids) < 2:
        return []

    evidence: dict[tuple[str, str], dict[str, Any]] = {}
    for key, row in load_vpr_candidates(vpr_payload).items():
        evidence[key] = dict(row)

    if maps_dir is not None:
        root = Path(maps_dir)
        if root.is_dir():
            for key, row in _load_maps_dir_candidates(root).items():
                current = evidence.get(key, {})
                merged = dict(current)
                merged.update({k: v for k, v in row.items() if v not in (None, 0, "")})
                if current.get("num_candidate_pairs"):
                    merged["num_candidate_pairs"] = max(
                        int(current.get("num_candidate_pairs") or 0),
                        int(row.get("num_candidate_pairs") or 0),
                    )
                evidence[key] = merged
            sep = int(_lookup(config, "edge.query_frame_separation", 30) or 30)
            for model in _discover_map_models(root):
                parsed = _parse_colmap_text(model, ids, frame_sep=sep)
                for key, row in parsed.items():
                    current = evidence.get(key, {})
                    merged = dict(current)
                    merged.update(row)
                    evidence[key] = merged

    edges: list[Any] = []
    # Without maps, do not invent an all-pairs graph. Emit only pairs that have
    # candidate or verified evidence; those stay non-STRONG unless geometry exists.
    keys = sorted(evidence) if evidence else []
    for a, b in keys:
        if a not in ids or b not in ids:
            continue
        row = evidence[(a, b)]
        verified = _as_int(row.get("num_verified_pairs") or row.get("num_cross_session_tracks"), 0)
        groups = _as_int(row.get("independent_bridge_groups"), 0)
        trusted = bool(row.get("trusted_geometry")) and verified > 0
        edges.append(
            classify_session_edge(
                a,
                b,
                num_verified_pairs=verified,
                independent_bridge_groups=groups,
                num_candidate_pairs=_as_int(row.get("num_candidate_pairs"), 0),
                num_cross_session_tracks=row.get("num_cross_session_tracks"),
                num_cross_session_observations=row.get("num_cross_session_observations"),
                trusted_geometry=trusted,
                config=config,
            )
        )

    try:
        from sfm_qa.session_select.critical_bridges import classify_critical_bridges

        marked = {
            (row["session_a"], row["session_b"])
            if row["session_a"] <= row["session_b"]
            else (row["session_b"], row["session_a"]): row
            for row in classify_critical_bridges(ids, edges)
        }
    except Exception:
        marked = {}
    if not marked:
        return edges
    out = []
    for edge in edges:
        key = (
            (edge.session_a, edge.session_b)
            if edge.session_a <= edge.session_b
            else (edge.session_b, edge.session_a)
        )
        info = marked.get(key)
        if info and info.get("is_critical_bridge"):
            payload = dict(edge.__dict__)
            payload["is_bridge"] = True
            payload["is_critical_bridge"] = True
            payload["reasons"] = tuple(edge.reasons) + ("unique_connector_groups_ge2",)
            out.append(_construct(_edge_cls(), **payload))
        else:
            out.append(edge)
    return out


def annotate_critical_bridges(session_ids: Iterable[str], edges: Sequence[Any]) -> list[Any]:
    del session_ids
    return list(edges)


__all__ = [
    "build_session_edges",
    "classify_session_edge",
    "load_vpr_candidates",
]
