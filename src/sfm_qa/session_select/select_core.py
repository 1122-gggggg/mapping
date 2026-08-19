"""Greedy BASE_CORE / BASE_SUPPORT selection. Timestamp is never a ranking key."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .config import lookup
from .critical_bridges import session_graph_diagnostics
from .objective import compute_objective_terms, delta_utility, efficiency_coverage, efficiency_info
from .types import SessionEdgeQuality, SessionQuality, edge_is_vpr_only

_CORE_OK = frozenset({"STRONG", "USABLE"})
_BLOCKED_EDGE = frozenset({"AMBIGUOUS", "REJECT"})


def _qualities(rows: Iterable[SessionQuality]) -> dict[str, SessionQuality]:
    return {row.session_id: row for row in rows}


def _edges(rows: Iterable[SessionEdgeQuality]) -> list[SessionEdgeQuality]:
    return list(rows)


def connecting_edges(
    session_id: str,
    selected: set[str],
    edges: Sequence[SessionEdgeQuality],
) -> list[SessionEdgeQuality]:
    found: list[SessionEdgeQuality] = []
    for edge in edges:
        ends = {edge.session_a, edge.session_b}
        if session_id in ends and (ends - {session_id}) & selected:
            found.append(edge)
    return found


def _link_is_geometric(edge: SessionEdgeQuality) -> bool:
    if edge.status in _BLOCKED_EDGE:
        return False
    if edge_is_vpr_only(edge):
        return False
    return True


def connection_is_admissible(
    session_id: str,
    selected: set[str],
    edges: Sequence[SessionEdgeQuality],
) -> tuple[bool, str]:
    """Fail-closed: no REJECT / AMBIGUOUS / single-critical-bridge merge into the core."""

    if not selected:
        return True, "empty_core"
    links = connecting_edges(session_id, selected, edges)
    if not links:
        return False, "no_connection_to_core"
    blocked = [edge for edge in links if edge.status in _BLOCKED_EDGE]
    usable = [edge for edge in links if _link_is_geometric(edge)]
    if not usable:
        if blocked:
            return False, "only_blocked_edges"
        return False, "no_geometric_edge"
    critical_only = [edge for edge in usable if edge.is_critical_bridge]
    noncritical = [edge for edge in usable if not edge.is_critical_bridge]
    if critical_only and not noncritical and len(critical_only) == 1:
        return False, "single_critical_bridge"
    if any(edge.is_critical_bridge and edge.status != "STRONG" for edge in usable) and not noncritical:
        return False, "unverified_critical_bridge"
    return True, "admissible"


def seed_session(
    qualities: Iterable[SessionQuality],
    config: Mapping[str, Any] | None = None,
    exclude: Iterable[str] = (),
) -> str | None:
    """Max internal quality among STRONG/USABLE. Timestamp is not used."""

    del config
    blocked = set(exclude)
    eligible = [
        row
        for row in qualities
        if row.internal_status in _CORE_OK and row.session_id not in blocked
    ]
    if not eligible:
        return None

    def key(row: SessionQuality) -> tuple[float, float, float, float]:
        # Timestamp is intentionally absent from this key.
        coverage = 0.0
        if row.convex_hull_coverage is not None:
            coverage += float(row.convex_hull_coverage)
        if row.grid_occupancy_4x4 is not None:
            coverage += float(row.grid_occupancy_4x4)
        info = float(row.fim_logdet) if row.fim_logdet is not None else 0.0
        consistency = 0.0 if row.rotation_cycle_error is None else -float(row.rotation_cycle_error)
        return (row.internal_quality_score, coverage, info, consistency)

    return max(eligible, key=key).session_id


def _budget_ok(
    selected: set[str],
    candidate: SessionQuality,
    qualities: Mapping[str, SessionQuality],
    config: Mapping[str, Any] | None,
) -> bool:
    cfg = dict(config or {})
    max_sessions = lookup(cfg, "selection.max_base_sessions")
    if max_sessions is not None and len(selected) + 1 > int(max_sessions):
        return False
    max_tracks = lookup(cfg, "selection.max_total_tracks")
    max_obs = lookup(cfg, "selection.max_total_observations")
    max_per = lookup(cfg, "selection.max_tracks_per_session")
    if max_per is not None and candidate.num_tracks is not None and candidate.num_tracks > int(max_per):
        return False
    tracks = sum((qualities[sid].num_tracks or 0) for sid in selected) + (candidate.num_tracks or 0)
    obs = sum((qualities[sid].num_observations or 0) for sid in selected) + (candidate.num_observations or 0)
    if max_tracks is not None and tracks > int(max_tracks):
        return False
    if max_obs is not None and obs > int(max_obs):
        return False
    return True


def split_core_vs_support(
    selected: Iterable[str],
    qualities: Iterable[SessionQuality],
    edges: Iterable[SessionEdgeQuality],
    config: Mapping[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    """BASE_CORE if removing it drops coverage/connectivity/info a lot; else SUPPORT."""

    cfg = dict(config or {})
    min_info = float(lookup(cfg, "selection.min_information_gain", 0.02) or 0.02)
    min_cov = float(lookup(cfg, "selection.min_coverage_gain", 0.02) or 0.02)
    quality_by_id = _qualities(qualities)
    edge_rows = _edges(edges)
    chosen = [sid for sid in selected if sid in quality_by_id]
    if not chosen:
        return [], []
    if len(chosen) == 1:
        return list(chosen), []

    full = compute_objective_terms(quality_by_id.values(), edge_rows, chosen, cfg)
    core: list[str] = []
    support: list[str] = []
    for sid in chosen:
        remainder = [other for other in chosen if other != sid]
        reduced = compute_objective_terms(quality_by_id.values(), edge_rows, remainder, cfg)
        drop_cov = full["coverage"] - reduced["coverage"]
        drop_info = full["information"] - reduced["information"]
        drop_conn = full["connectivity"] - reduced["connectivity"]
        drop_red = full["redundancy"] - reduced["redundancy"]
        if drop_cov >= min_cov or drop_info >= min_info or drop_conn > 0.05 or drop_red > 0.15:
            core.append(sid)
        else:
            support.append(sid)
    if not core:
        best = max(chosen, key=lambda sid: quality_by_id[sid].internal_quality_score)
        core = [best]
        support = [sid for sid in chosen if sid != best]
    return core, support


def _close_cycles(
    selected: list[str],
    leftover: Sequence[SessionQuality],
    qualities: Mapping[str, SessionQuality],
    edges: Sequence[SessionEdgeQuality],
    config: Mapping[str, Any],
) -> list[str]:
    """If the selected graph is a tree, add one USABLE closer that does not explode tracks."""

    if len(selected) < 2:
        return selected
    inner = [
        edge
        for edge in edges
        if edge.session_a in selected
        and edge.session_b in selected
        and edge.status in {"STRONG", "USABLE"}
        and not edge_is_vpr_only(edge)
    ]
    diagnostics = session_graph_diagnostics(selected, inner)
    tree_like = diagnostics["usable_edges"] if "usable_edges" in diagnostics else len(inner)
    n = len(selected)
    if tree_like >= n:
        return selected
    closer_ids: list[str] = []
    for row in leftover:
        if row.internal_status not in _CORE_OK:
            continue
        links = connecting_edges(row.session_id, set(selected), edges)
        usable = [
            edge
            for edge in links
            if edge.status in {"STRONG", "USABLE"}
            and not edge.is_critical_bridge
            and not edge_is_vpr_only(edge)
        ]
        if len(usable) >= 2 and _budget_ok(set(selected), row, qualities, config):
            closer_ids.append(row.session_id)
    if closer_ids:
        pick = max(closer_ids, key=lambda sid: qualities[sid].internal_quality_score)
        return selected + [pick]
    return selected


def greedy_select_core(
    qualities: Iterable[SessionQuality],
    edges: Iterable[SessionEdgeQuality],
    config: Mapping[str, Any] | None = None,
    exclude: Iterable[str] = (),
) -> dict[str, Any]:
    """Seed by internal quality, then add argmax ΔU under fail-closed connection rules."""

    cfg = dict(config or {})
    quality_by_id = _qualities(qualities)
    edge_rows = _edges(edges)
    min_info = float(lookup(cfg, "selection.min_information_gain", 0.02) or 0.02)
    min_cov = float(lookup(cfg, "selection.min_coverage_gain", 0.02) or 0.02)
    weights = lookup(cfg, "selection.weights")
    blocked = set(exclude)

    seed = seed_session(quality_by_id.values(), cfg, exclude=blocked)
    if seed is None:
        return {
            "core": [],
            "support": [],
            "selected": [],
            "scores": {},
            "stop_reason": "no_strong_or_usable_seed",
            "seed": None,
        }

    selected = [seed]
    selected_set = {seed}
    remaining = [sid for sid in quality_by_id if sid != seed and sid not in blocked]
    stop_reason = "exhausted_candidates"

    while remaining:
        current = compute_objective_terms(quality_by_id.values(), edge_rows, selected, cfg)
        ranked: list[tuple[float, float, str, dict[str, float], str]] = []
        for sid in remaining:
            row = quality_by_id[sid]
            if row.internal_status not in _CORE_OK:
                continue
            if not _budget_ok(selected_set, row, quality_by_id, cfg):
                continue
            ok, why = connection_is_admissible(sid, selected_set, edge_rows)
            if not ok:
                continue
            after = compute_objective_terms(quality_by_id.values(), edge_rows, selected + [sid], cfg)
            delta_u = delta_utility(current, after, weights)
            d_obs = after["observations"] - current["observations"]
            d_cov = after["coverage"] - current["coverage"]
            d_info = after["information"] - current["information"]
            eff_i = efficiency_info(d_info, d_obs) or 0.0
            eff_c = efficiency_coverage(d_cov, d_obs) or 0.0
            ranked.append((delta_u, eff_i + eff_c, sid, after, why))
        if not ranked:
            stop_reason = "no_admissible_neighbor"
            break
        ranked.sort(reverse=True)
        delta_u, _eff, sid, after, _why = ranked[0]
        d_cov = after["coverage"] - current["coverage"]
        d_info = after["information"] - current["information"]
        d_red = after["redundancy"] - current["redundancy"]
        d_tracks = after["tracks"] - current["tracks"]
        d_obs = after["observations"] - current["observations"]
        cheap_gain = d_cov < min_cov and d_info < min_info and d_red < min_cov
        expensive = d_tracks > 0 or d_obs > 0
        if cheap_gain and expensive and delta_u <= 0:
            stop_reason = "diminishing_returns"
            break
        if delta_u <= 0 and cheap_gain:
            stop_reason = "nonpositive_delta_u"
            break
        selected.append(sid)
        selected_set.add(sid)
        remaining.remove(sid)

    leftover_rows = [
        quality_by_id[sid]
        for sid in quality_by_id
        if sid not in selected_set and sid not in blocked
    ]
    selected = _close_cycles(selected, leftover_rows, quality_by_id, edge_rows, cfg)
    core, support = split_core_vs_support(selected, quality_by_id.values(), edge_rows, cfg)
    scores = compute_objective_terms(quality_by_id.values(), edge_rows, selected, cfg)
    scores["seed"] = seed
    return {
        "core": core,
        "support": support,
        "selected": selected,
        "scores": scores,
        "stop_reason": stop_reason,
        "seed": seed,
    }
