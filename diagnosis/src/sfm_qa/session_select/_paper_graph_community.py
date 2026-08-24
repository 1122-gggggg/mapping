"""Community-aware scene-graph pruning helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import networkx as nx
import numpy as np


def _communities(
    graph: nx.Graph,
    pair_info: Mapping[tuple[str, str], Mapping[str, Any]],
    cfg: Mapping[str, Any],
    protected: set[str],
) -> dict[str, Any]:
    active = graph.subgraph([node for node in graph if graph.degree(node) > 0]).copy()
    if graph.number_of_nodes() == 0:
        return {
            "communities": [],
            "session_community": {},
            "quarantined_sessions": [],
            "new_submap_candidates": [],
        }
    if not active:
        groups = [{node} for node in sorted(graph)]
        membership = {node: index for index, group in enumerate(groups) for node in group}
        metrics = []
        quarantined = []
        for index, group in enumerate(groups):
            node = next(iter(group))
            keep = node in protected
            if not keep:
                quarantined.append(node)
            metrics.append(
                {
                    "community_id": index,
                    "sessions": [node],
                    "size": 1,
                    "component_id": None,
                    "separation_score": None,
                    "mean_internal_reliability": 0.0,
                    "external_weight_ratio": 0.0,
                    "contains_protected_session": keep,
                    "size_ratio_to_anchor": 0.0,
                    "role": "PROTECTED_KEEP" if keep else "QUARANTINE_OUTLIER",
                }
            )
        return {
            "communities": metrics,
            "session_community": membership,
            "quarantined_sessions": sorted(quarantined),
            "new_submap_candidates": [],
        }
    try:
        groups = list(
            nx.community.louvain_communities(
                active,
                weight="weight",
                resolution=float(cfg["community"]["resolution"]),
                seed=int(cfg["seed"]),
            )
        )
    except AttributeError:
        groups = list(nx.community.greedy_modularity_communities(active, weight="weight"))
    groups.extend({node} for node in sorted(set(graph) - set(active)))
    groups = [set(group) for group in groups]
    groups.sort(key=lambda group: (-len(group), sorted(group)))
    membership = {node: index for index, group in enumerate(groups) for node in group}
    positions = (
        nx.spring_layout(
            active,
            seed=int(cfg["seed"]),
            weight="weight",
            iterations=int(cfg["community"]["layout_iterations"]),
        )
        if active
        else {}
    )
    lengths = [
        float(np.linalg.norm(np.asarray(positions[a]) - np.asarray(positions[b])))
        for a, b in active.edges
    ]
    layout_scale = max(float(np.median(lengths)) if lengths else 1.0, 1e-9)
    components = list(nx.connected_components(active)) if active else []
    components.sort(key=lambda group: (-len(group), sorted(group)))
    component_id = {node: index for index, group in enumerate(components) for node in group}

    metrics = []
    masses = []
    for index, group in enumerate(groups):
        internal = []
        external = []
        for pair, info in pair_info.items():
            inside = (pair[0] in group, pair[1] in group)
            if all(inside):
                internal.append(info["reliability"])
            elif any(inside):
                external.append(info["reliability"])
        internal_weight = float(sum(internal))
        external_weight = float(sum(external))
        total = internal_weight + external_weight
        outside = set(active) - group
        if outside and group & set(positions):
            distance = min(
                float(np.linalg.norm(np.asarray(positions[u]) - np.asarray(positions[v])))
                for u in group
                if u in positions
                for v in outside
                if v in positions
            )
            separation = distance / layout_scale / max(math.log1p(len(group)), 1e-9)
        else:
            separation = None
        metrics.append(
            {
                "community_id": index,
                "sessions": sorted(group),
                "size": len(group),
                "component_id": min(
                    (component_id[node] for node in group if node in component_id), default=None
                ),
                "separation_score": separation,
                "mean_internal_reliability": float(np.mean(internal)) if internal else 0.0,
                "external_weight_ratio": external_weight / total if total else 0.0,
                "contains_protected_session": bool(group & protected),
            }
        )
        masses.append(internal_weight + 0.5 * external_weight + len(group) * 1e-6)

    protected_groups = {membership[node] for node in protected if node in membership}
    anchor = max(protected_groups, key=masses.__getitem__) if protected_groups else max(
        range(len(groups)), key=masses.__getitem__
    )
    anchor_component = metrics[anchor]["component_id"]
    anchor_size = max(len(groups[anchor]), 1)
    quarantined: set[str] = set()
    new_submaps: set[str] = set()
    ccfg = cfg["community"]
    for metric in metrics:
        group = groups[metric["community_id"]]
        ratio = len(group) / anchor_size
        small = len(group) <= int(ccfg["maximum_outlier_sessions"]) or ratio <= float(
            ccfg["maximum_outlier_fraction"]
        )
        separated = metric["separation_score"] is None or metric["separation_score"] >= float(
            ccfg["separation_threshold"]
        )
        weak = metric["mean_internal_reliability"] < float(ccfg["minimum_keep_reliability"])
        weakly_attached = metric["external_weight_ratio"] <= float(
            ccfg["maximum_external_weight_ratio"]
        )
        if metric["contains_protected_session"]:
            role = "PROTECTED_KEEP"
        elif metric["community_id"] == anchor:
            role = "BASE_CONNECTED"
        elif metric["component_id"] != anchor_component:
            strong = len(group) >= int(ccfg["minimum_new_submap_sessions"]) and metric[
                "mean_internal_reliability"
            ] >= float(ccfg["minimum_new_submap_reliability"])
            if strong:
                role = "NEW_SUBMAP_CANDIDATE"
                new_submaps.update(group)
            elif small:
                role = "QUARANTINE_OUTLIER"
                quarantined.update(group)
            else:
                role = "NEW_SUBMAP_CANDIDATE"
                new_submaps.update(group)
        elif separated and small and weak and weakly_attached:
            role = "QUARANTINE_OUTLIER"
            quarantined.update(group)
        else:
            role = "SUPPORT_CONNECTED"
        metric.update(size_ratio_to_anchor=ratio, role=role)
    return {
        "communities": metrics,
        "session_community": membership,
        "quarantined_sessions": sorted(quarantined),
        "new_submap_candidates": sorted(new_submaps),
    }
