#!/usr/bin/env python3
"""ExMaps-style ref/keyframe stability scoring for update bundles.

The score is intentionally conservative: old refs decay when they are not
recently observed, refs supported by new observation sessions get a bounded
boost, and newly-added refs inherit a route-quality prior from the update
report. A score of 1.0 is neutral.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def prefix_of(name: str) -> str:
    return str(name).split("/", 1)[0]


def decay_for_half_life(half_life_sessions: float) -> float:
    half_life = max(float(half_life_sessions), 1e-6)
    return float(0.5 ** (1.0 / half_life))


def observation_ref_hits(observation_stats: dict) -> dict[str, float]:
    """Return normalized ref hit counts from map_update_tool observation stats."""
    hits: dict[str, float] = defaultdict(float)
    sessions = observation_stats.get("sessions") or {}
    for sess in sessions.values():
        for item in sess.get("top_base_ref_hits", []):
            key = str(item.get("key", ""))
            if key:
                hits[key] += float(item.get("count", 0.0))
    if not hits:
        for item in observation_stats.get("global_top_base_ref_hits", []):
            key = str(item.get("key", ""))
            if key:
                hits[key] += float(item.get("count", 0.0))
    return dict(hits)


def route_score(row: dict) -> float:
    """Initial stability prior for keyframes added by one update route."""
    route = str(row.get("route", ""))
    status = str(row.get("status", ""))
    if route == "submap":
        bridges = float(row.get("bridges") or 0.0)
        geom = float(row.get("bridge_geometry") or 0.0)
        ratio = float(row.get("bridge_median_inlier_ratio") or 0.0)
        if "retrieval_high_but_inliers_low" in status or geom < 4:
            return 0.65
        return float(np.clip(0.90 + min(0.25, bridges / 200.0) + min(0.15, ratio * 0.25), 0.8, 1.3))
    if route == "register":
        return 0.85
    if route == "connector_only":
        return 0.70
    return 1.0


def route_scores_by_prefix(report_rows: Iterable[dict]) -> dict[str, float]:
    return {str(row.get("seq")): route_score(row) for row in report_rows if row.get("seq")}


def build_ref_stability(
    ref_names: Iterable[str],
    prior_names: Iterable[str],
    prior_scores: Iterable[float],
    observation_stats: dict,
    report_rows: Iterable[dict],
    half_life_sessions: float = 4.0,
    observed_bonus: float = 0.45,
    min_score: float = 0.05,
    max_score: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """Build ref_stability aligned with ref_names.

    Existing refs are decayed by one update session, then recently observed refs
    receive a normalized support boost. New refs get a route-quality prior.
    """
    names = [str(name) for name in ref_names]
    prior_lut = {str(name): float(score) for name, score in zip(prior_names, prior_scores)}
    route_lut = route_scores_by_prefix(report_rows)
    decay = decay_for_half_life(half_life_sessions)
    hits = observation_ref_hits(observation_stats)
    max_hits = max(hits.values(), default=0.0)

    scores = np.empty(len(names), dtype=np.float32)
    observed = 0
    for i, name in enumerate(names):
        if name in prior_lut:
            score = prior_lut[name] * decay
        else:
            score = route_lut.get(prefix_of(name), 1.0)
        if max_hits > 0 and name in hits:
            score += float(observed_bonus) * float(hits[name]) / max_hits
            observed += 1
        scores[i] = float(np.clip(score, min_score, max_score))

    meta = {
        "version": 1,
        "model": "exponential_decay_ref_stability",
        "half_life_sessions": float(half_life_sessions),
        "decay_per_update": decay,
        "observed_bonus": float(observed_bonus),
        "min_score": float(min_score),
        "max_score": float(max_score),
        "observed_ref_count": int(observed),
        "route_prefix_scores": route_lut,
    }
    return scores, meta


def rerank_indices_by_stability(
    similarities: np.ndarray,
    stability: np.ndarray | None,
    topk: int,
    candidate_multiplier: int = 3,
    stability_weight: float = 0.05,
) -> np.ndarray:
    """Return retrieval indices with stability as a mild tie-breaker.

    The candidate pool is still selected by raw VPR similarity, then reranked by
    similarity + weight * log(stability). This avoids letting stability rescue
    globally dissimilar refs.
    """
    sims = np.asarray(similarities, dtype=np.float32)
    k = min(len(sims), max(1, int(topk)))
    pool_k = min(len(sims), max(k, k * max(1, int(candidate_multiplier))))
    if pool_k == 0:
        return np.asarray([], dtype=np.int64)
    pool = np.argsort(-sims)[:pool_k]
    if stability is None or float(stability_weight) == 0.0:
        return pool[:k].astype(np.int64)
    stab = np.asarray(stability, dtype=np.float32)
    if stab.shape[0] != sims.shape[0]:
        return pool[:k].astype(np.int64)
    adjusted = sims[pool] + float(stability_weight) * np.log(np.clip(stab[pool], 1e-6, None))
    order = np.argsort(-adjusted)
    return pool[order[:k]].astype(np.int64)


def stability_summary(scores: np.ndarray) -> dict:
    arr = np.asarray(scores, dtype=np.float32)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "below_neutral": int((arr < 1.0).sum()),
        "above_neutral": int((arr > 1.0).sum()),
    }
