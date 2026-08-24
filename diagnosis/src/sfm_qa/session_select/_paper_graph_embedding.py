"""Planar-evidence spectral embedding and robust graph backbone helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from ._paper_graph_util import (
    _UnionFind,
    _clip,
    _first_number,
    _number,
    _pair,
    _standardize,
)


def _component_plan(
    component_id: int,
    graph: nx.Graph,
    pair_info: Mapping[tuple[str, str], Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    pairs = sorted(_pair(a, b) for a, b in graph.edges)
    pairs = [pair for pair in pairs if pair]
    embedding, embedding_method = _spectral_embedding(pairs, pair_info, cfg)
    anomaly = _feature_anomalies(pairs, pair_info)
    tree, window, exact = _minimum_range_tree(graph, embedding, pair_info, cfg)
    tree_pairs = {_pair(a, b) for a, b in tree.edges} - {None}
    augmented, cycle = _augment_cycles(tree, graph, embedding, anomaly, pair_info, cfg)
    original_bridges = {_pair(a, b) for a, b in nx.bridges(graph)} - {None}
    median_reliability = float(np.median([pair_info[pair]["reliability"] for pair in pairs]))
    ecfg = cfg["embedding"]
    pruned = set()
    for pair in pairs:
        if pair in tree_pairs or pair in cycle or pair in original_bridges:
            continue
        distance = max(window[0] - embedding[pair], 0.0, embedding[pair] - window[1])
        outlier = distance > float(ecfg["prune_margin"]) or anomaly[pair] >= float(
            ecfg["relative_feature_anomaly_threshold"]
        )
        low_reliability = pair_info[pair]["reliability"] < median_reliability * float(
            ecfg["low_reliability_ratio"]
        )
        explicit = _explicit_inconsistency(pair_info[pair]["row"], pair_info[pair]["planar"], cfg)
        if outlier and (low_reliability or explicit):
            pruned.add(pair)
    return {
        "component_id": component_id,
        "sessions": sorted(graph),
        "embedding_method": embedding_method,
        "minimum_range_exact": exact,
        "minimum_range_window": list(window),
        "backbone_pairs": [list(pair) for pair in sorted(tree_pairs)],
        "cycle_pairs": [list(pair) for pair in sorted(cycle)],
        "pruned_pairs": [list(pair) for pair in sorted(pruned)],
        "biconnected_after_augmentation": bool(
            augmented.number_of_nodes() > 2 and nx.is_biconnected(augmented)
        ),
        "articulation_points_after_augmentation": sorted(nx.articulation_points(augmented)),
        "edge_embeddings": [
            {
                "pair": list(pair),
                "embedding": embedding[pair],
                "feature_anomaly": anomaly[pair],
                "reliability": pair_info[pair]["reliability"],
            }
            for pair in pairs
        ],
    }


def _spectral_embedding(
    pairs: Sequence[tuple[str, str]],
    pair_info: Mapping[tuple[str, str], Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], float], str]:
    count = len(pairs)
    if count <= 1:
        return {pair: 0.0 for pair in pairs}, "SINGLE_EDGE"
    rows = []
    cols = []
    values = []
    for left in range(count):
        for right in range(left + 1, count):
            if not set(pairs[left]) & set(pairs[right]):
                continue
            similarity = math.sqrt(
                pair_info[pairs[left]]["reliability"] * pair_info[pairs[right]]["reliability"]
            )
            similarity *= 0.25 + 0.75 * _feature_similarity(
                pair_info[pairs[left]]["features"], pair_info[pairs[right]]["features"]
            )
            rows.extend((left, right))
            cols.extend((right, left))
            values.extend((similarity, similarity))
    adjacency = sparse.csr_matrix((values, (rows, cols)), shape=(count, count))
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(degree)
    valid = degree > 1e-12
    inverse[valid] = 1.0 / np.sqrt(degree[valid])
    laplacian = sparse.eye(count) - sparse.diags(inverse) @ adjacency @ sparse.diags(inverse)
    if count <= int(cfg["embedding"]["dense_eigendecomposition_limit"]):
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
        vector = eigenvectors[:, np.argsort(eigenvalues)[1]]
        method = "DENSE_EIGH"
    else:
        try:
            eigenvalues, eigenvectors = eigsh(
                laplacian,
                k=2,
                which="SM",
                v0=np.linspace(1.0, 2.0, count),
            )
            vector = eigenvectors[:, np.argsort(eigenvalues)[1]]
            method = "SPARSE_EIGSH"
        except (ArpackNoConvergence, RuntimeError, ValueError):
            # Fail safely: a deterministic reliability ordering keeps the stage
            # operational without manufacturing spectral confidence.
            vector = np.asarray([pair_info[pair]["reliability"] for pair in pairs])
            method = "RELIABILITY_FALLBACK"
    nonzero = np.flatnonzero(np.abs(vector) > 1e-12)
    if len(nonzero) and vector[nonzero[0]] < 0:
        vector = -vector
    vector = _standardize(np.asarray(vector, dtype=float))
    return (
        {pair: float(value) for pair, value in zip(pairs, vector, strict=True)},
        method,
    )


def _minimum_range_tree(
    graph: nx.Graph,
    embedding: Mapping[tuple[str, str], float],
    pair_info: Mapping[tuple[str, str], Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> tuple[nx.Graph, tuple[float, float], bool]:
    pairs = sorted(embedding, key=lambda pair: (embedding[pair], pair))
    exact = len(pairs) <= int(cfg["embedding"]["exact_minimum_range_edge_limit"])
    best = None
    if exact:
        for left in range(len(pairs)):
            union = _UnionFind(graph.nodes)
            for right in range(left, len(pairs)):
                union.union(*pairs[right])
                if union.count == 1:
                    candidate = (
                        embedding[pairs[right]] - embedding[pairs[left]],
                        -sum(pair_info[pair]["reliability"] for pair in pairs[left : right + 1]),
                        left,
                        right,
                    )
                    if best is None or candidate < best:
                        best = candidate
                    break
    if best is None:
        selected = sorted(
            pairs,
            key=lambda pair: (-pair_info[pair]["reliability"], abs(embedding[pair]), pair),
        )
        window_graph = nx.Graph()
        window_graph.add_nodes_from(graph)
        used = []
        for pair in selected:
            window_graph.add_edge(*pair, weight=pair_info[pair]["reliability"])
            used.append(pair)
            if nx.is_connected(window_graph):
                break
        low = min(embedding[pair] for pair in used)
        high = max(embedding[pair] for pair in used)
        exact = False
    else:
        used = pairs[best[2] : best[3] + 1]
        low = embedding[used[0]]
        high = embedding[used[-1]]
        window_graph = nx.Graph()
        window_graph.add_nodes_from(graph)
        for pair in used:
            window_graph.add_edge(*pair, weight=pair_info[pair]["reliability"])
    if not nx.is_connected(window_graph):
        window_graph = graph.copy()
        low, high, exact = min(embedding.values()), max(embedding.values()), False
    tree = nx.maximum_spanning_tree(window_graph, weight="weight")
    return tree, (float(low), float(high)), exact


def _augment_cycles(
    tree: nx.Graph,
    original: nx.Graph,
    embedding: Mapping[tuple[str, str], float],
    anomaly: Mapping[tuple[str, str], float],
    pair_info: Mapping[tuple[str, str], Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> tuple[nx.Graph, set[tuple[str, str]]]:
    augmented = tree.copy()
    if augmented.number_of_nodes() <= 2:
        return augmented, set()
    tree_pairs = {_pair(a, b) for a, b in tree.edges} - {None}
    low = min(embedding[pair] for pair in tree_pairs)
    high = max(embedding[pair] for pair in tree_pairs)
    reliabilities = [info["reliability"] for info in pair_info.values()]
    median = float(np.median(reliabilities)) if reliabilities else 0.0
    ecfg = cfg["embedding"]
    candidates = []
    for a, b in original.edges:
        pair = _pair(a, b)
        if not pair or pair in tree_pairs:
            continue
        distance = max(low - embedding[pair], 0.0, embedding[pair] - high)
        outlier = distance > float(ecfg["prune_margin"]) or anomaly[pair] >= float(
            ecfg["relative_feature_anomaly_threshold"]
        )
        low_reliability = pair_info[pair]["reliability"] < median * float(
            ecfg["low_reliability_ratio"]
        )
        explicit = _explicit_inconsistency(pair_info[pair]["row"], pair_info[pair]["planar"], cfg)
        if outlier and (low_reliability or explicit):
            continue
        candidates.append((distance, -pair_info[pair]["reliability"], pair))
    selected = set()
    for _, _, pair in sorted(candidates):
        if nx.is_biconnected(augmented):
            break
        before = len(list(nx.articulation_points(augmented)))
        trial = augmented.copy()
        trial.add_edge(*pair, weight=pair_info[pair]["reliability"])
        if len(list(nx.articulation_points(trial))) < before:
            augmented = trial
            selected.add(pair)
            if len(selected) >= int(ecfg["maximum_cycle_augmentations"]):
                break
    return augmented, selected


def _feature_anomalies(
    pairs: Sequence[tuple[str, str]], pair_info: Mapping[tuple[str, str], Mapping[str, Any]]
) -> dict[tuple[str, str], float]:
    fields = sorted({field for pair in pairs for field in pair_info[pair]["features"]})
    median = {}
    scale = {}
    for field in fields:
        values = [
            pair_info[pair]["features"][field]
            for pair in pairs
            if field in pair_info[pair]["features"]
        ]
        median[field] = float(np.median(values))
        mad = float(np.median(np.abs(np.asarray(values) - median[field])))
        scale[field] = max(1.4826 * mad, 0.10)
    result = {}
    for pair in pairs:
        z = [
            abs((value - median[field]) / scale[field])
            for field, value in pair_info[pair]["features"].items()
        ]
        result[pair] = float(math.sqrt(np.mean(np.square(z)))) if z else 0.0
    return result


def _features(
    row: Mapping[str, Any], cfg: Mapping[str, Any], planar: Mapping[str, Any]
) -> dict[str, float]:
    scales = cfg["reliability"]
    result = {}
    for field, key in (
        ("rotation_consensus_deg", "rotation_scale_deg"),
        ("translation_direction_consensus_deg", "translation_scale_deg"),
        ("scale_consensus", "scale_consensus_scale"),
        ("cross_session_reprojection_error", "reprojection_scale_px"),
    ):
        value = _number(row.get(field))
        if value is not None:
            result[field] = value / max(float(scales[key]), 1e-9)
    inlier = _first_number(row, "holdout_inlier_ratio", "inlier_ratio")
    if inlier is not None:
        result["inlier_error"] = 1.0 - _clip(inlier)
    if planar["score"] is not None:
        result["planar_error"] = 1.0 - float(planar["score"])
    return result


def _feature_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    common = set(left) & set(right)
    if not common:
        return 0.35
    distance = float(np.mean([(left[key] - right[key]) ** 2 for key in common]))
    return float(math.exp(-0.5 * distance))


def _explicit_inconsistency(
    row: Mapping[str, Any], planar: Mapping[str, Any], cfg: Mapping[str, Any]
) -> bool:
    scales = cfg["reliability"]
    for field, key in (
        ("rotation_consensus_deg", "rotation_scale_deg"),
        ("translation_direction_consensus_deg", "translation_scale_deg"),
        ("scale_consensus", "scale_consensus_scale"),
        ("cross_session_reprojection_error", "reprojection_scale_px"),
    ):
        value = _number(row.get(field))
        if value is not None and value > 1.25 * float(scales[key]):
            return True
    return bool(planar["score"] is not None and planar["score"] < 0.35)
