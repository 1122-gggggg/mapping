"""Pose-graph initialization from Global-Aware Edge Prioritization (Wei et al., CVPR 2026).

The production S3 forced reverse-direction grid stays load-bearing. These tests
specify the *retrieval/same-direction* selector that can replace per-image kNN
without dropping required VPR-blind bridges.
"""
from __future__ import annotations

from pose_graph_init import (
    ScoredEdge,
    knn_select,
    select_pose_graph,
)


def _undirected(pairs: set[tuple[str, str]] | tuple[tuple[str, str], ...]) -> set[tuple[str, str]]:
    return {tuple(sorted(pair)) for pair in pairs}


def _two_cluster_edges() -> tuple[list[str], list[ScoredEdge]]:
    left = [f"L{i}" for i in range(5)]
    right = [f"R{i}" for i in range(5)]
    nodes = left + right
    edges: list[ScoredEdge] = []
    for cluster in (left, right):
        for i, a in enumerate(cluster):
            for j, b in enumerate(cluster):
                if i >= j:
                    continue
                # Nearby same-cluster views look like strong retrieval neighbors.
                gap = j - i
                score = 0.95 - 0.05 * (gap - 1)
                edges.append(ScoredEdge(a, b, score, source="retrieval"))
    edges.append(ScoredEdge("L4", "R0", 0.18, source="retrieval"))
    return nodes, edges


def test_mst_keeps_the_only_inter_cluster_bridge_that_knn_drops() -> None:
    nodes, edges = _two_cluster_edges()
    knn = knn_select(nodes, edges, k=2)
    mst = select_pose_graph(nodes, edges, k_msts=1)

    assert ("L4", "R0") not in _undirected(knn.edges)
    assert ("L4", "R0") in _undirected(mst.edges)
    assert mst.components == 1
    assert knn.components == 2


def test_required_vpr_blind_bridges_survive_zero_retrieval_score() -> None:
    nodes = ["F0", "F1", "R0", "R1"]
    edges = [
        ScoredEdge("F0", "F1", 0.9, source="retrieval"),
        ScoredEdge("R0", "R1", 0.9, source="retrieval"),
        ScoredEdge("F0", "R0", 0.12, source="retrieval"),
        ScoredEdge("F1", "R1", 0.11, source="retrieval"),
    ]
    result = select_pose_graph(
        nodes,
        edges,
        k_msts=1,
        required_edges=[("F1", "R0")],
    )
    selected = _undirected(result.edges)
    assert ("F1", "R0") in selected
    assert ("F0", "R0") not in selected
    assert ("F1", "R1") not in selected
    assert result.n_required == 1
    assert result.components == 1


def test_second_mst_plus_modulation_prefers_a_long_range_chord() -> None:
    nodes = [f"C{i}" for i in range(6)]
    edges = [
        ScoredEdge(f"C{i}", f"C{i + 1}", 0.92, source="retrieval")
        for i in range(5)
    ]
    edges.append(ScoredEdge("C0", "C5", 0.55, source="retrieval"))
    edges.append(ScoredEdge("C2", "C4", 0.88, source="retrieval"))
    edges.append(ScoredEdge("C3", "C5", 0.87, source="retrieval"))
    edges.append(ScoredEdge("C1", "C3", 0.82, source="retrieval"))
    edges.append(ScoredEdge("C0", "C2", 0.80, source="retrieval"))
    edges.append(ScoredEdge("C2", "C5", 0.78, source="retrieval"))
    edges.append(ScoredEdge("C0", "C3", 0.75, source="retrieval"))
    edges.append(ScoredEdge("C1", "C5", 0.72, source="retrieval"))
    edges.append(ScoredEdge("C1", "C4", 0.70, source="retrieval"))
    edges.append(ScoredEdge("C0", "C4", 0.68, source="retrieval"))

    first = select_pose_graph(nodes, edges, k_msts=1, modulation_lambda=0.0)
    unmodulated = select_pose_graph(
        nodes,
        edges,
        k_msts=2,
        modulation_lambda=0.0,
    )
    modulated = select_pose_graph(
        nodes,
        edges,
        k_msts=2,
        modulation_lambda=0.6,
    )

    assert ("C0", "C5") not in _undirected(first.edges)
    assert ("C0", "C5") not in _undirected(unmodulated.edges)
    assert ("C0", "C5") in _undirected(modulated.edges)
    assert modulated.diameter is not None
    assert first.diameter is not None
    assert modulated.diameter < first.diameter
    assert modulated.n_selected > first.n_selected


def test_select_pose_graph_is_undirected_and_ignores_self_pairs() -> None:
    nodes = ["a", "b", "c"]
    edges = [
        ScoredEdge("a", "b", 0.8, source="retrieval"),
        ScoredEdge("b", "a", 0.4, source="retrieval"),
        ScoredEdge("b", "c", 0.7, source="retrieval"),
        ScoredEdge("a", "a", 1.0, source="retrieval"),
    ]
    result = select_pose_graph(nodes, edges, k_msts=1)
    assert result.edges == (("a", "b"), ("b", "c"))
    assert result.isolated == ()


def test_modulation_can_promote_a_low_score_diametral_chord_on_a_long_path() -> None:
    nodes = [f"C{i:02d}" for i in range(12)]
    edges = [
        ScoredEdge(nodes[i], nodes[i + 1], 0.95, source="retrieval")
        for i in range(11)
    ]
    for i in range(10):
        edges.append(ScoredEdge(nodes[i], nodes[i + 2], 0.80, source="retrieval"))
    for i in range(9):
        edges.append(ScoredEdge(nodes[i], nodes[i + 3], 0.70, source="retrieval"))
    edges.append(ScoredEdge(nodes[0], nodes[-1], 0.20, source="retrieval"))

    unmodulated = select_pose_graph(nodes, edges, k_msts=2, modulation_lambda=0.0)
    modulated = select_pose_graph(nodes, edges, k_msts=2, modulation_lambda=0.7)

    chord = (nodes[0], nodes[-1])
    assert chord not in _undirected(unmodulated.edges)
    assert chord in _undirected(modulated.edges)


def test_scored_endpoints_outside_nodes_are_ignored() -> None:
    result = select_pose_graph(
        ["a", "b", "c"],
        [
            ScoredEdge("a", "b", 0.9, source="retrieval"),
            ScoredEdge("b", "c", 0.8, source="retrieval"),
            ScoredEdge("c", "ghost", 0.99, source="retrieval"),
        ],
        k_msts=2,
        modulation_lambda=0.5,
    )
    selected = _undirected(result.edges)
    assert ("c", "ghost") not in selected
    assert result.n_nodes == 3


def test_required_endpoints_are_added_even_when_missing_from_nodes() -> None:
    result = select_pose_graph(
        ["F0", "F1"],
        [ScoredEdge("F0", "F1", 0.9, source="retrieval")],
        required_edges=[("F1", "R0")],
    )
    assert ("F1", "R0") in _undirected(result.edges)
    assert result.n_nodes == 3
    assert result.components == 1
    assert result.isolated == ()


def test_empty_candidates_keep_required_edges_and_report_isolates() -> None:
    result = select_pose_graph(
        ["a", "b", "c"],
        [],
        required_edges=[("a", "b")],
    )
    assert result.edges == (("a", "b"),)
    assert result.isolated == ("c",)
    assert result.components == 2
