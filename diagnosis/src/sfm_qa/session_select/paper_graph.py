"""Fail-closed graph hardening adapted from G-MASt3R-SfM and Planar-SfM.

Only independently verified exact-pair geometry enters this graph. The module may
retain or downgrade an edge, but never promotes retrieval/shared-map evidence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import networkx as nx
import numpy as np

from ._paper_graph_io import load_edge_rows, merge_probe_metrics, write_hardening_outputs
from ._paper_graph_topology import _communities, _component_plan, _features, _schedule
from ._paper_graph_util import (
    _append_reason,
    _clip,
    _deep_update,
    _eligible,
    _first_number,
    _gaussian,
    _items,
    _json_value,
    _number,
    _pair,
    _prune_reason,
    _row_dict,
    _truthy,
)

SCHEMA_VERSION = "PAPER_GRAPH_HARDENING_V1"

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 17,
    "geometric_statuses": ["STRONG", "USABLE"],
    "reliability": {
        "support_reference": 200.0,
        "rotation_scale_deg": 5.0,
        "translation_scale_deg": 15.0,
        "scale_consensus_scale": 0.15,
        "reprojection_scale_px": 5.0,
    },
    "planar": {
        "rotation_agreement_scale_deg": 5.0,
        "normal_angle_scale_deg": 15.0,
        "minimum_observed_terms": 2,
        "validated_score": 0.65,
    },
    "community": {
        "resolution": 1.0,
        "layout_iterations": 200,
        "separation_threshold": 1.5,
        "maximum_outlier_fraction": 0.25,
        "maximum_outlier_sessions": 2,
        "maximum_external_weight_ratio": 0.20,
        "minimum_keep_reliability": 0.55,
        "minimum_new_submap_reliability": 0.65,
        "minimum_new_submap_sessions": 2,
    },
    "embedding": {
        "dense_eigendecomposition_limit": 512,
        "exact_minimum_range_edge_limit": 600,
        "prune_margin": 1.5,
        "relative_feature_anomaly_threshold": 3.5,
        "low_reliability_ratio": 0.85,
        "maximum_cycle_augmentations": 64,
    },
    "optimization": {
        "recent_error_window": 5,
        "minimum_relative_improvement": 1e-4,
        "local_max_iterations": 50,
        "neighbor_max_iterations": 30,
        "global_max_iterations": 100,
        "rollback_on_error_increase": True,
    },
}


def paper_graph_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Merge defaults with either a full config or its ``paper_graph`` section."""

    merged = deepcopy(DEFAULT_CONFIG)
    if config:
        source = config.get("paper_graph", config)
        if isinstance(source, Mapping):
            _deep_update(merged, source)
    return merged


def harden_session_graph(
    session_ids: Iterable[str],
    edges: Sequence[Mapping[str, Any] | Any],
    config: Mapping[str, Any] | None = None,
    *,
    protected_sessions: Iterable[str] = (),
) -> dict[str, Any]:
    """Return hardened edge rows, communities, backbone, and BA schedule."""

    cfg = paper_graph_config(config)
    rows = [_row_dict(row) for row in edges]
    sessions = {str(item) for item in session_ids if str(item)}
    protected = {str(item) for item in protected_sessions if str(item)}
    pair_info: dict[tuple[str, str], dict[str, Any]] = {}

    for index, row in enumerate(rows):
        pair = _pair(row.get("session_a"), row.get("session_b"))
        if pair:
            sessions.update(pair)
        eligible, reason = _eligible(row, cfg)
        planar = planar_consistency(row, cfg)
        reliability = edge_reliability(row, cfg, planar)
        row.update(
            paper_graph_eligible=eligible,
            paper_graph_reason=reason,
            graph_reliability=reliability,
            planar_consistency_score=planar["score"],
            planar_evidence_completeness=planar["completeness"],
            planar_validated=planar["validated"],
        )
        if not pair or not eligible:
            continue
        candidate = {
            "row": row,
            "row_index": index,
            "reliability": reliability,
            "planar": planar,
            "features": _features(row, cfg, planar),
        }
        current = pair_info.get(pair)
        key = (reliability, planar["completeness"], -index)
        if current is None or key > current["rank"]:
            candidate["rank"] = key
            pair_info[pair] = candidate

    graph = nx.Graph()
    graph.add_nodes_from(sorted(sessions))
    for pair, info in pair_info.items():
        graph.add_edge(*pair, weight=info["reliability"])

    community = _communities(graph, pair_info, cfg, protected)
    quarantined = set(community["quarantined_sessions"])
    components = []
    embeddings: dict[tuple[str, str], float] = {}
    anomalies: dict[tuple[str, str], float] = {}
    backbone: set[tuple[str, str]] = set()
    cycle_edges: set[tuple[str, str]] = set()
    spectral_pruned: set[tuple[str, str]] = set()
    pair_component: dict[tuple[str, str], int] = {}

    connected = [
        set(nodes)
        for nodes in nx.connected_components(graph)
        if len(nodes) > 1 and graph.subgraph(nodes).number_of_edges() > 0
    ]
    connected.sort(key=lambda nodes: (-len(nodes), sorted(nodes)))
    for component_id, nodes in enumerate(connected):
        subgraph = graph.subgraph(nodes).copy()
        report = _component_plan(component_id, subgraph, pair_info, cfg)
        components.append(report)
        for row in report["edge_embeddings"]:
            pair = tuple(row["pair"])
            embeddings[pair] = row["embedding"]
            anomalies[pair] = row["feature_anomaly"]
            pair_component[pair] = component_id
        backbone.update(tuple(pair) for pair in report["backbone_pairs"])
        cycle_edges.update(tuple(pair) for pair in report["cycle_pairs"])
        spectral_pruned.update(tuple(pair) for pair in report["pruned_pairs"])

    community_pruned = {
        pair for pair in pair_info if pair[0] in quarantined or pair[1] in quarantined
    }
    pruned = spectral_pruned | community_pruned
    hardened = []
    downgraded = 0
    for row in rows:
        item = dict(row)
        pair = _pair(item.get("session_a"), item.get("session_b"))
        item["paper_graph_schema"] = SCHEMA_VERSION
        if pair:
            info = pair_info.get(pair)
            item.update(
                graph_community_a=community["session_community"].get(pair[0]),
                graph_community_b=community["session_community"].get(pair[1]),
                graph_component_id=pair_component.get(pair),
                graph_pair_embedding=embeddings.get(pair),
                graph_feature_anomaly=anomalies.get(pair),
                graph_backbone=pair in backbone,
                graph_cycle_edge=pair in cycle_edges,
                graph_community_quarantined=pair in community_pruned,
                graph_pruned=pair in pruned,
                graph_prune_reason=_prune_reason(pair, spectral_pruned, community_pruned),
            )
            if info:
                item["graph_reliability"] = info["reliability"]
            original = str(item.get("status") or "").upper()
            item["status_before_paper_graph"] = original
            if pair in pruned and original in {"STRONG", "USABLE"}:
                item["status"] = "AMBIGUOUS"
                item["is_bridge"] = False
                item["reasons"] = _append_reason(
                    item.get("reasons"),
                    "paper_graph_fail_closed_downgrade:" + item["graph_prune_reason"],
                )
                downgraded += 1
        hardened.append(item)

    retained = sorted(pair for pair in pair_info if pair not in pruned)
    schedule = _schedule(graph, retained, community, cfg, quarantined)
    diagnostics = [
        {
            "pair": list(pair),
            "reliability": info["reliability"],
            "planar_score": info["planar"]["score"],
            "planar_validated": info["planar"]["validated"],
            "embedding": embeddings.get(pair),
            "feature_anomaly": anomalies.get(pair),
            "backbone": pair in backbone,
            "cycle_edge": pair in cycle_edges,
            "pruned": pair in pruned,
            "prune_reason": _prune_reason(pair, spectral_pruned, community_pruned),
        }
        for pair, info in sorted(pair_info.items())
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "source_methods": {
            "G-MASt3R-SfM": "arXiv:2606.22856",
            "Planar-SfM": "arXiv:2606.31979v2",
        },
        "invariants": [
            "retrieval-only edges never enter the geometric graph",
            "edge statuses are never promoted",
            "original graph bridges are not spectrally removed",
            "strong disconnected communities remain new-submap candidates",
            "optimization output is a schedule, not a BA receipt",
        ],
        "config": cfg,
        "counts": {
            "input_sessions": len(sessions),
            "input_edge_rows": len(rows),
            "eligible_pairs": len(pair_info),
            "retained_pairs": len(retained),
            "pruned_pairs": len(pruned),
            "downgraded_rows": downgraded,
        },
        "sessions": sorted(sessions),
        "communities": community["communities"],
        "quarantined_sessions": sorted(quarantined),
        "new_submap_candidates": community["new_submap_candidates"],
        "retained_pairs": [list(pair) for pair in retained],
        "pruned_pairs": [list(pair) for pair in sorted(pruned)],
        "backbone_pairs": [list(pair) for pair in sorted(backbone)],
        "cycle_pairs": [list(pair) for pair in sorted(cycle_edges)],
        "components": components,
        "edge_diagnostics": diagnostics,
        "optimization_schedule": schedule,
        "edge_rows": hardened,
    }


def planar_consistency(
    row: Mapping[str, Any], config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Score shared inliers, plane normals, and essential-rotation agreement."""

    cfg = paper_graph_config(config)["planar"]
    hypotheses = _json_value(
        row.get("homography_hypotheses")
        or row.get("planar_hypotheses")
        or row.get("homographies")
    )
    candidates = hypotheses if isinstance(hypotheses, list) else [row]
    scores = []
    weights = []
    best_terms: dict[str, float] = {}
    for raw in candidates:
        if not isinstance(raw, Mapping):
            continue
        merged = dict(row)
        if raw is not row:
            merged.update(raw)
        terms: dict[str, float] = {}
        support = _first_number(
            merged,
            "homography_shared_inlier_ratio",
            "homography_inlier_ratio",
            "homography_support",
        )
        if support is not None:
            if support > 1.0:
                denominator = _first_number(
                    merged, "num_matches", "num_verified_pairs", "inlier_count"
                )
                if denominator and denominator > 0:
                    support /= denominator
            terms["shared_inliers"] = _clip(support)
        normal_similarity = _first_number(
            merged, "plane_normal_similarity", "homography_normal_similarity"
        )
        if normal_similarity is not None:
            terms["plane_normal"] = _clip(abs(normal_similarity))
        normal_angle = _first_number(
            merged, "plane_normal_angle_deg", "homography_normal_angle_deg"
        )
        if normal_angle is not None:
            terms["plane_normal"] = _gaussian(normal_angle, cfg["normal_angle_scale_deg"])
        rotation_error = _first_number(
            merged,
            "homography_rotation_error_deg",
            "essential_rotation_agreement_deg",
            "homography_essential_rotation_error_deg",
        )
        if rotation_error is not None:
            terms["essential_rotation"] = _gaussian(
                rotation_error, cfg["rotation_agreement_scale_deg"]
            )
        if terms:
            scores.append(float(np.mean(list(terms.values()))))
            weights.append(max(_first_number(merged, "confidence", "weight") or 1.0, 1e-6))
            if len(terms) > len(best_terms):
                best_terms = terms
    score = float(np.average(scores, weights=weights)) if scores else None
    observed = len(best_terms)
    completeness = _clip(observed / max(int(cfg["minimum_observed_terms"]), 1))
    return {
        "score": score,
        "completeness": completeness,
        "validated": bool(
            score is not None
            and observed >= int(cfg["minimum_observed_terms"])
            and score >= float(cfg["validated_score"])
        ),
        "terms": best_terms,
        "hypotheses": len(scores),
    }


def edge_reliability(
    row: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    planar: Mapping[str, Any] | None = None,
) -> float:
    """Compute a bounded quality score from complete, observed evidence only."""

    cfg = paper_graph_config(config)
    scale = cfg["reliability"]
    values = []
    weights = []

    def add(value: float | None, weight: float) -> None:
        if value is not None:
            values.append(_clip(value))
            weights.append(weight)

    add(_number(row.get("edge_quality_score")), 0.15)
    support = _first_number(row, "num_verified_pairs", "num_cross_session_tracks", "inlier_count")
    if support and support > 0:
        reference = max(float(scale["support_reference"]), 1.0)
        add(math.log1p(support) / math.log1p(reference), 0.15)
    add(_first_number(row, "holdout_inlier_ratio", "inlier_ratio"), 0.15)
    geometry = []
    for field, key in (
        ("rotation_consensus_deg", "rotation_scale_deg"),
        ("translation_direction_consensus_deg", "translation_scale_deg"),
        ("scale_consensus", "scale_consensus_scale"),
        ("cross_session_reprojection_error", "reprojection_scale_px"),
    ):
        value = _number(row.get(field))
        if value is not None:
            geometry.append(_gaussian(value, scale[key]))
    for field in ("edge_positive_depth_ratio", "spatial_coverage"):
        value = _number(row.get(field))
        if value is not None:
            geometry.append(_clip(value))
    add(float(np.mean(geometry)) if geometry else None, 0.35)
    planar = planar or planar_consistency(row, cfg)
    add(planar["score"], 0.20)
    if not values:
        return 0.0
    score = float(np.average(values, weights=weights))
    completeness = sum(weights)
    score *= 0.65 + 0.35 * min(completeness, 1.0)
    flags = {item.upper() for item in _items(row.get("degeneracy_flags"))}
    if flags & {"DEGENERATE", "PURE_ROTATION", "PANORAMIC", "PLANAR_DOMINANT_UNVALIDATED"}:
        score *= 0.5
    if _truthy(row.get("planar_dominant")) and not planar["validated"]:
        score *= 0.75
    return _clip(score)


__all__ = [
    "DEFAULT_CONFIG",
    "SCHEMA_VERSION",
    "edge_reliability",
    "harden_session_graph",
    "load_edge_rows",
    "merge_probe_metrics",
    "paper_graph_config",
    "planar_consistency",
    "write_hardening_outputs",
]
