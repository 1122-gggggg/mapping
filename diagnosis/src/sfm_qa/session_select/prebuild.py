"""Pre-build video admission planning.

This module is deliberately proposal-only. Retrieval and video QA can rank which
sessions are worth spending geometry on, but they never authorize an SfM merge.
The final BASE_CORE/BASE_SUPPORT selection still requires verified geometric
session edges in :mod:`select_core`.

The graph objective follows two ideas from the SfM literature:

* camera-triplet edge scoring: compare an edge with the strongest edge in each
  triangle, then aggregate across triangles;
* budgeted coverage: greedily prefer sessions that add new graph neighbourhood
  coverage instead of adding every redundant video.

Both are heuristic at session level here. Image-pair geometry remains the
authoritative gate later in S3/S4.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from typing import Any

from .config import lookup
from .types import SessionEdgeQuality, SessionQuality

_BLOCKED_STATUS = frozenset({"REJECT", "INCONSISTENT"})


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    if not math.isfinite(number):
        return low
    return max(low, min(high, number))


def _edge_value(edge: SessionEdgeQuality | Mapping[str, Any], field: str) -> float:
    if isinstance(edge, Mapping):
        value = edge.get(field)
    else:
        value = getattr(edge, field, None)
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _edge_ends(edge: SessionEdgeQuality | Mapping[str, Any]) -> tuple[str, str]:
    if isinstance(edge, Mapping):
        left = str(edge.get("session_a") or edge.get("a") or "")
        right = str(edge.get("session_b") or edge.get("b") or "")
    else:
        left, right = str(edge.session_a), str(edge.session_b)
    return (left, right) if left <= right else (right, left)


def camera_triplet_scores(
    edges: Iterable[SessionEdgeQuality | Mapping[str, Any]],
    *,
    count_field: str = "num_candidate_pairs",
) -> dict[tuple[str, str], dict[str, float]]:
    """Return the session-edge analogue of camera-triplet support.

    For every complete triangle ``t`` and edge ``e`` in that triangle,

    ``q(e,t) = n_e / max(n_1, n_2, n_3)``.

    The final score is the mean over triangles containing the edge.  With
    ``count_field='num_candidate_pairs'`` this is proposal evidence only; with
    verified counts it becomes geometric support evidence.  A score of zero
    means "no complete triplet evidence", not "false edge".
    """

    counts: dict[tuple[str, str], float] = {}
    nodes: set[str] = set()
    for edge in edges:
        left, right = _edge_ends(edge)
        if not left or not right or left == right:
            continue
        count = _edge_value(edge, count_field)
        if count <= 0.0:
            continue
        counts[(left, right)] = max(counts.get((left, right), 0.0), count)
        nodes.update((left, right))

    sums: dict[tuple[str, str], float] = {key: 0.0 for key in counts}
    support: dict[tuple[str, str], int] = {key: 0 for key in counts}
    ordered = sorted(nodes)
    for a, b, c in combinations(ordered, 3):
        triangle = ((a, b), (a, c), (b, c))
        if any(key not in counts for key in triangle):
            continue
        maximum = max(counts[key] for key in triangle)
        if maximum <= 0.0:
            continue
        for key in triangle:
            sums[key] += counts[key] / maximum
            support[key] += 1

    return {
        key: {
            "score": (sums[key] / support[key]) if support[key] else 0.0,
            "triplets": float(support[key]),
            "count": float(counts[key]),
        }
        for key in counts
    }


def _motion_profile(row: SessionQuality) -> tuple[float, ...] | None:
    values = (
        row.parallax_ratio,
        row.low_parallax_ratio,
        row.hover_ratio,
        row.pure_rotation_ratio,
        row.fast_motion_ratio,
        row.unproven_ratio,
    )
    if all(value is None for value in values):
        return None
    array = [max(0.0, float(value or 0.0)) for value in values]
    total = sum(array)
    if total <= 0.0:
        return None
    return tuple(value / total for value in array)


def motion_profile_distance(left: SessionQuality, right: SessionQuality) -> float:
    """Total-variation distance between two measured motion histograms."""

    first = _motion_profile(left)
    second = _motion_profile(right)
    if first is None or second is None:
        return 0.0
    return 0.5 * sum(abs(a - b) for a, b in zip(first, second))


def video_risk(row: SessionQuality) -> float:
    """Bounded risk proxy from degeneracy, blur/exposure and motion consistency."""

    low = _clamp(row.low_parallax_ratio)
    degeneracy = _clamp(
        _clamp(row.hover_ratio)
        + _clamp(row.pure_rotation_ratio)
        + _clamp(row.fast_motion_ratio)
        + 0.5 * low
    )
    duplicate = _clamp(row.near_duplicate_ratio)
    exposure = _clamp(_clamp(row.underexposed_ratio) + _clamp(row.overexposed_ratio))
    epipolar = _clamp(row.epipolar_outlier_ratio_median)
    unproven = _clamp(row.unproven_ratio)
    return _clamp(
        0.35 * degeneracy
        + 0.20 * duplicate
        + 0.15 * exposure
        + 0.20 * epipolar
        + 0.10 * unproven
    )


def video_admission_score(
    row: SessionQuality,
    config: Mapping[str, Any] | None = None,
) -> float:
    """Score whether a video is worth geometry verification before SfM."""

    cfg = dict(config or {})
    sharp_reference = float(lookup(cfg, "prebuild.sharpness_reference", 100.0) or 100.0)
    sharpness = _clamp((row.sharpness_p10 or 0.0) / max(sharp_reference, 1e-9))
    parallax = _clamp(_clamp(row.parallax_ratio) + 0.35 * _clamp(row.low_parallax_ratio))
    motion_evidence = 1.0 - _clamp(row.unproven_ratio)
    duplicate_quality = 1.0 - _clamp(row.near_duplicate_ratio)
    exposure_quality = 1.0 - _clamp(
        _clamp(row.underexposed_ratio) + _clamp(row.overexposed_ratio)
    )
    epipolar_quality = 1.0 - _clamp(row.epipolar_outlier_ratio_median)
    internal = _clamp(row.internal_quality_score)

    weights = lookup(cfg, "prebuild.video_weights", {}) or {}
    w_internal = float(weights.get("internal_quality", 0.20))
    w_parallax = float(weights.get("parallax", 0.25))
    w_sharp = float(weights.get("sharpness", 0.15))
    w_motion = float(weights.get("motion_evidence", 0.10))
    w_duplicate = float(weights.get("non_duplicate", 0.10))
    w_exposure = float(weights.get("exposure", 0.10))
    w_epipolar = float(weights.get("epipolar_consistency", 0.10))
    denominator = max(
        1e-9,
        w_internal + w_parallax + w_sharp + w_motion + w_duplicate + w_exposure + w_epipolar,
    )
    score = (
        w_internal * internal
        + w_parallax * parallax
        + w_sharp * sharpness
        + w_motion * motion_evidence
        + w_duplicate * duplicate_quality
        + w_exposure * exposure_quality
        + w_epipolar * epipolar_quality
    ) / denominator
    return _clamp(score)


def _proposal_graph(
    edges: Sequence[SessionEdgeQuality | Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str], float],
    dict[str, dict[str, float]],
    dict[tuple[str, str], dict[str, float]],
]:
    raw: dict[tuple[str, str], float] = {}
    for edge in edges:
        left, right = _edge_ends(edge)
        if not left or not right or left == right:
            continue
        count = _edge_value(edge, "num_candidate_pairs")
        if count > 0:
            raw[(left, right)] = max(raw.get((left, right), 0.0), count)
    maximum = max(raw.values(), default=0.0)
    strengths: dict[tuple[str, str], float] = {}
    adjacency: dict[str, dict[str, float]] = {}
    for key, count in raw.items():
        if maximum <= 1.0:
            strength = _clamp(count / max(maximum, 1.0))
        else:
            strength = math.log1p(count) / math.log1p(maximum)
        strengths[key] = _clamp(strength)
        left, right = key
        adjacency.setdefault(left, {})[right] = strengths[key]
        adjacency.setdefault(right, {})[left] = strengths[key]
    return strengths, adjacency, camera_triplet_scores(edges, count_field="num_candidate_pairs")


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def propose_prebuild_set(
    qualities: Iterable[SessionQuality],
    edges: Iterable[SessionEdgeQuality | Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Propose which videos deserve S3 geometric verification.

    This *never* returns merge authority.  ``proposed_base_sessions`` is a
    budgeted candidate set; every cross-session relation is emitted again in
    ``verification_pairs`` and must be verified geometrically before the normal
    BASE_CORE/BASE_SUPPORT selector may admit it.
    """

    cfg = dict(config or {})
    rows = {row.session_id: row for row in qualities}
    edge_rows = list(edges)
    min_video_score = float(lookup(cfg, "prebuild.min_video_score", 0.35) or 0.35)
    min_marginal = float(lookup(cfg, "prebuild.min_marginal_gain", 0.15) or 0.15)
    min_base = int(lookup(cfg, "prebuild.min_base_sessions", 2) or 2)
    max_sessions_cfg = lookup(cfg, "prebuild.max_sessions")
    max_sessions = int(max_sessions_cfg) if max_sessions_cfg is not None else max(1, len(rows))
    max_no_graph = int(lookup(cfg, "prebuild.max_no_graph_sessions", 3) or 3)
    validation_count = int(lookup(cfg, "prebuild.validation_candidates", 1) or 1)
    allow_weak = bool(lookup(cfg, "prebuild.allow_weak_video_candidates", False))

    video_scores = {sid: video_admission_score(row, cfg) for sid, row in rows.items()}
    risks = {sid: video_risk(row) for sid, row in rows.items()}
    eligible: list[str] = []
    rejected: dict[str, str] = {}
    for sid, row in rows.items():
        if row.internal_status in _BLOCKED_STATUS:
            rejected[sid] = f"internal_status={row.internal_status}"
            continue
        if row.internal_status == "WEAK" and not allow_weak:
            rejected[sid] = "weak_video_requires_explicit_override_or_geometry"
            continue
        if video_scores[sid] < min_video_score:
            rejected[sid] = f"video_score_below_heuristic_{min_video_score:.3f}"
            continue
        eligible.append(sid)

    strengths, adjacency, triplets = _proposal_graph(edge_rows)
    graph_available = bool(strengths)
    if not eligible:
        return {
            "proposed_base_sessions": [],
            "validation_candidates": [],
            "verification_pairs": [],
            "session_scores": {},
            "rejected": rejected,
            "proposal_confidence": "NONE",
            "proposal_graph_available": graph_available,
            "requires_geometric_verification": True,
            "notes": [
                "No video passed proposal-level QA.",
                "Retrieval is never geometric merge authority.",
            ],
        }

    max_frames = max((rows[sid].num_frames for sid in eligible), default=1) or 1
    weights = lookup(cfg, "prebuild.weights", {}) or {}
    w_video = float(weights.get("video_quality", 0.45))
    w_bridge = float(weights.get("bridgeability", 0.25))
    w_graph_cov = float(weights.get("graph_coverage", 0.15))
    w_diversity = float(weights.get("motion_diversity", 0.10))
    w_triplet = float(weights.get("triplet_support", 0.10))
    w_multi = float(weights.get("multi_link", 0.10))
    w_risk = float(weights.get("risk", 0.20))
    w_cost = float(weights.get("frame_cost", 0.05))

    seed = max(
        eligible,
        key=lambda sid: (
            video_scores[sid] - 0.5 * risks[sid],
            rows[sid].num_keyframes,
            sid,
        ),
    )
    selected = [seed]
    selected_set = {seed}
    ranked_steps: list[dict[str, Any]] = [
        {
            "session_id": seed,
            "marginal_score": video_scores[seed] - 0.5 * risks[seed],
            "reason": "highest_video_admission_score",
        }
    ]

    def covered_neighbourhood(chosen: set[str]) -> set[str]:
        covered = set(chosen)
        for sid in chosen:
            covered.update(adjacency.get(sid, {}))
        return covered

    remaining = [sid for sid in eligible if sid != seed]
    # Preserve an untouched validation candidate whenever the pool is large enough.
    # This is proposal-stage isolation only; the final S0 corpus lock still owns
    # the authoritative build/test split and content-hash proof.
    reserve = validation_count if len(eligible) > min_base else 0
    proposal_capacity = max(min_base, len(eligible) - max(0, reserve))
    budget = min(max_sessions, proposal_capacity, len(eligible))
    if not graph_available:
        budget = min(budget, max_no_graph)

    while remaining and len(selected) < budget:
        covered = covered_neighbourhood(selected_set)
        total_weight = sum(video_scores[sid] for sid in eligible) or 1.0
        candidates: list[tuple[float, str, dict[str, float]]] = []
        for sid in remaining:
            links = [
                strengths.get(_pair_key(sid, other), 0.0)
                for other in selected
                if _pair_key(sid, other) in strengths
            ]
            bridgeability = max(links, default=0.0)
            multi_link = _clamp(len([value for value in links if value > 0.0]) / 2.0)
            triplet_support = max(
                (
                    triplets.get(_pair_key(sid, other), {}).get("score", 0.0)
                    for other in selected
                ),
                default=0.0,
            )
            profile_distances = [motion_profile_distance(rows[sid], rows[other]) for other in selected]
            diversity = (
                sum(profile_distances) / len(profile_distances) if profile_distances else 0.0
            )
            new_nodes = ({sid} | set(adjacency.get(sid, {}))) - covered
            graph_coverage = (
                sum(video_scores.get(node, 0.0) for node in new_nodes) / total_weight
                if graph_available
                else 0.0
            )
            frame_cost = _clamp(rows[sid].num_frames / max_frames)
            terms = {
                "video_quality": video_scores[sid],
                "bridgeability": bridgeability,
                "graph_coverage": graph_coverage,
                "motion_diversity": diversity,
                "triplet_support": float(triplet_support),
                "multi_link": multi_link,
                "risk": risks[sid],
                "frame_cost": frame_cost,
            }
            marginal = (
                w_video * terms["video_quality"]
                + w_bridge * terms["bridgeability"]
                + w_graph_cov * terms["graph_coverage"]
                + w_diversity * terms["motion_diversity"]
                + w_triplet * terms["triplet_support"]
                + w_multi * terms["multi_link"]
                - w_risk * terms["risk"]
                - w_cost * terms["frame_cost"]
            )
            candidates.append((marginal, sid, terms))

        candidates.sort(key=lambda item: (item[0], video_scores[item[1]], item[1]), reverse=True)
        marginal, sid, terms = candidates[0]
        if len(selected) >= min_base and marginal < min_marginal:
            break
        selected.append(sid)
        selected_set.add(sid)
        remaining.remove(sid)
        ranked_steps.append(
            {
                "session_id": sid,
                "marginal_score": float(marginal),
                "terms": terms,
                "reason": "greedy_budgeted_proposal",
            }
        )

    remaining_ranked = sorted(
        (sid for sid in eligible if sid not in selected_set),
        key=lambda sid: (
            max(
                (strengths.get(_pair_key(sid, other), 0.0) for other in selected),
                default=0.0,
            )
            * 0.6
            + video_scores[sid] * 0.4,
            video_scores[sid],
            sid,
        ),
        reverse=True,
    )
    validation = remaining_ranked[: max(0, validation_count)]

    verification_pairs: list[dict[str, Any]] = []
    for left, right in combinations(selected, 2):
        key = _pair_key(left, right)
        candidate_count = 0
        for edge in edge_rows:
            if _edge_ends(edge) == key:
                candidate_count = max(
                    candidate_count,
                    int(_edge_value(edge, "num_candidate_pairs")),
                )
        triplet_score = float(triplets.get(key, {}).get("score", 0.0))
        strength = strengths.get(key, 0.0)
        priority = strength + 0.5 * triplet_score
        verification_pairs.append(
            {
                "session_a": left,
                "session_b": right,
                "candidate_pairs": candidate_count,
                "proposal_strength": float(strength),
                "triplet_score": triplet_score,
                "priority": float(priority),
                "forced_probe": candidate_count <= 0,
                "requires_geometric_verification": True,
                "reason": (
                    "retrieval_candidate_requires_geometry"
                    if candidate_count > 0
                    else "no_retrieval_support_force_geometry_probe"
                ),
            }
        )
    verification_pairs.sort(
        key=lambda row: (
            bool(row["forced_probe"]),
            -float(row["priority"]),
            str(row["session_a"]),
            str(row["session_b"]),
        )
    )

    if not graph_available:
        confidence = "LOW"
    else:
        selected_edges = [
            key
            for key, strength in strengths.items()
            if strength > 0.0 and key[0] in selected_set and key[1] in selected_set
        ]
        connected_nodes = {node for edge in selected_edges for node in edge}
        has_triangle = any(
            triplets.get(key, {}).get("triplets", 0.0) > 0.0 for key in selected_edges
        )
        if selected_set <= connected_nodes and has_triangle:
            confidence = "HIGH_PROPOSAL_ONLY"
        elif selected_set <= connected_nodes:
            confidence = "MEDIUM_PROPOSAL_ONLY"
        else:
            confidence = "LOW"

    session_scores = {
        sid: {
            "video_admission_score": float(video_scores[sid]),
            "risk": float(risks[sid]),
            "selected_for_geometry": sid in selected_set,
            "validation_candidate": sid in validation,
        }
        for sid in rows
    }
    return {
        "proposed_base_sessions": selected,
        "validation_candidates": validation,
        "verification_pairs": verification_pairs,
        "session_scores": session_scores,
        "ranked_steps": ranked_steps,
        "rejected": rejected,
        "proposal_confidence": confidence,
        "proposal_graph_available": graph_available,
        "requires_geometric_verification": True,
        "notes": [
            "This is a pre-build proposal, not merge authority.",
            "VPR/candidate counts rank verification effort only; they are never geometric edges.",
            "Final BASE_CORE/BASE_SUPPORT still requires verified multi-bridge geometry.",
        ],
    }


__all__ = [
    "camera_triplet_scores",
    "motion_profile_distance",
    "propose_prebuild_set",
    "video_admission_score",
    "video_risk",
]
