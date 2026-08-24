"""Community-local to component-global optimization scheduling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx


def _schedule(
    graph: nx.Graph,
    retained: Sequence[tuple[str, str]],
    community: Mapping[str, Any],
    cfg: Mapping[str, Any],
    quarantined: set[str],
) -> list[dict[str, Any]]:
    groups = {
        row["community_id"]: set(row["sessions"])
        for row in community["communities"]
        if row["role"] != "QUARANTINE_OUTLIER"
    }
    membership = community["session_community"]
    ocfg = cfg["optimization"]
    common = {
        "recent_error_window": int(ocfg["recent_error_window"]),
        "minimum_relative_improvement": float(ocfg["minimum_relative_improvement"]),
        "rollback_on_error_increase": bool(ocfg["rollback_on_error_increase"]),
        "execution_authority": "SCHEDULE_ONLY_REQUIRES_REAL_BA_RECEIPT",
    }
    stages = []
    for community_id, sessions in sorted(groups.items()):
        active = sessions - quarantined
        if active:
            stages.append(
                {
                    **common,
                    "stage": "LOCAL",
                    "community_id": community_id,
                    "sessions": sorted(active),
                    "pairs": [list(pair) for pair in retained if set(pair) <= active],
                    "max_iterations": int(ocfg["local_max_iterations"]),
                }
            )
    for community_id, sessions in sorted(groups.items()):
        neighbors = {community_id}
        for a, b in retained:
            if membership.get(a) == community_id:
                neighbors.add(membership.get(b))
            if membership.get(b) == community_id:
                neighbors.add(membership.get(a))
        neighbors.discard(None)
        active = set().union(*(groups.get(index, set()) for index in neighbors)) - quarantined
        if active:
            stages.append(
                {
                    **common,
                    "stage": "NEIGHBOR",
                    "community_id": community_id,
                    "neighbor_communities": sorted(neighbors),
                    "sessions": sorted(active),
                    "pairs": [list(pair) for pair in retained if set(pair) <= active],
                    "max_iterations": int(ocfg["neighbor_max_iterations"]),
                }
            )
    retained_graph = nx.Graph()
    retained_graph.add_nodes_from(node for node in graph if node not in quarantined)
    retained_graph.add_edges_from(retained)
    components = [
        set(nodes)
        for nodes in nx.connected_components(retained_graph)
        if retained_graph.subgraph(nodes).number_of_edges() > 0
    ]
    components.sort(key=lambda nodes: (-len(nodes), sorted(nodes)))
    for component_id, nodes in enumerate(components):
        stages.append(
            {
                **common,
                "stage": "GLOBAL_COMPONENT",
                "component_id": component_id,
                "sessions": sorted(nodes),
                "pairs": [list(pair) for pair in retained if set(pair) <= nodes],
                "max_iterations": int(ocfg["global_max_iterations"]),
            }
        )
    return stages
