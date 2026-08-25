"""Globally informed pose-graph initialization.

Implements the *selection* half of Wei, Tolias, Matas & Barath, "Global-Aware
Edge Prioritization for Pose Graph Initialization", CVPR 2026:

1. rank candidate edges (caller supplies scores; GNN weights are optional);
2. build the graph as the union of k minimum spanning trees;
3. modulate later trees with hop-distance so weak regions and long chains get
   extra chords.

This module does **not** replace S3 forced reverse-direction bridges or S4
Doppelgangers++. MegaLoc kNN is blind to forward/reverse of the same route;
those edges stay `required_edges`. Downstream geometric verification still
owns admission.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ScoredEdge:
    image_a: str
    image_b: str
    score: float
    source: str = "retrieval"

    def normalized(self) -> tuple[str, str]:
        a, b = self.image_a, self.image_b
        if a == b:
            raise ValueError("self-pair is not a pose-graph edge")
        return (a, b) if a < b else (b, a)


@dataclass(frozen=True)
class PoseGraphInit:
    edges: tuple[tuple[str, str], ...]
    n_nodes: int
    n_selected: int
    n_required: int
    components: int
    diameter: int | None
    isolated: tuple[str, ...]
    method: str


def knn_select(
    nodes: Sequence[str],
    edges: Sequence[ScoredEdge],
    k: int,
) -> PoseGraphInit:
    """Per-image top-k retrieval, the baseline Wei et al. replace at initialization."""
    if k < 1:
        raise ValueError("k must be >= 1")
    adjacency: dict[str, list[tuple[float, str]]] = {node: [] for node in nodes}
    for pair, edge in _unique_undirected(edges).items():
        a, b = pair
        if a not in adjacency or b not in adjacency:
            continue
        adjacency[a].append((edge.score, b))
        adjacency[b].append((edge.score, a))
    selected: set[tuple[str, str]] = set()
    for node, neighbors in adjacency.items():
        neighbors.sort(key=lambda item: (-item[0], item[1]))
        for _, other in neighbors[:k]:
            selected.add((node, other) if node < other else (other, node))
    return _finalize(nodes, selected, n_required=0, method=f"knn_k{k}")


def select_pose_graph(
    nodes: Sequence[str],
    edges: Sequence[ScoredEdge],
    *,
    k_msts: int = 2,
    modulation_lambda: float = 0.5,
    required_edges: Sequence[tuple[str, str]] = (),
) -> PoseGraphInit:
    """Union of k MSTs with optional hop-distance modulation.

    ``score`` is higher-better. Scores are min-max normalized to [0, 1]. Kruskal
    then prefers high score (equivalent to ``w = 1 - score``). Already chosen
    edges are removed from later trees. Required reverse-direction bridges are
    inserted first so they contribute to hop-distance and cannot be dropped.

    The candidate ``edges`` list *is* the pool. Hop-distance reweights every
    remaining candidate against the original normalized score. Wei et al. cap
    modulation at top-5/image because they rank a complete graph; this repo
    never builds that complete graph, and MegaLoc will not have already ranked
    the long-range chord into a local top-5.
    """
    if k_msts < 1:
        raise ValueError("k_msts must be >= 1")
    if not 0.0 <= modulation_lambda <= 1.0:
        raise ValueError("modulation_lambda must be in [0, 1]")

    node_set = list(dict.fromkeys(nodes))
    required = {_ordered(pair) for pair in required_edges if pair[0] != pair[1]}
    known = set(node_set)
    for a, b in required:
        if a not in known:
            node_set.append(a)
            known.add(a)
        if b not in known:
            node_set.append(b)
            known.add(b)
    ranked = _normalize_scores(_unique_undirected(edges, known=known))
    selected: set[tuple[str, str]] = set(required)
    adjacency = _adjacency(node_set, selected)

    base_scores = {
        pair: score for pair, score in ranked.items() if pair not in selected
    }
    remaining_keys = set(base_scores)
    for mst_index in range(k_msts):
        current = {pair: base_scores[pair] for pair in remaining_keys}
        if mst_index > 0 and modulation_lambda > 0.0 and current:
            current = _modulate(
                current,
                adjacency,
                modulation_lambda=modulation_lambda,
            )
        tree = _kruskal_mst(
            node_set,
            current,
            preconnected=required if mst_index == 0 else (),
        )
        if not tree:
            break
        selected.update(tree)
        for a, b in tree:
            adjacency[a].add(b)
            adjacency[b].add(a)
        remaining_keys.difference_update(tree)

    return _finalize(
        node_set,
        selected,
        n_required=len(required),
        method=f"multi_mst_k{k_msts}",
    )


def _unique_undirected(
    edges: Sequence[ScoredEdge],
    *,
    known: set[str] | None = None,
) -> dict[tuple[str, str], ScoredEdge]:
    unique: dict[tuple[str, str], ScoredEdge] = {}
    for edge in edges:
        if edge.image_a == edge.image_b:
            continue
        if known is not None and (
            edge.image_a not in known or edge.image_b not in known
        ):
            continue
        pair = edge.normalized()
        previous = unique.get(pair)
        if previous is None or edge.score > previous.score:
            unique[pair] = ScoredEdge(pair[0], pair[1], edge.score, edge.source)
    return unique


def _normalize_scores(
    unique: dict[tuple[str, str], ScoredEdge],
) -> dict[tuple[str, str], float]:
    if not unique:
        return {}
    values = [edge.score for edge in unique.values()]
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 0.0:
        return {pair: 1.0 for pair in unique}
    return {pair: (edge.score - lo) / span for pair, edge in unique.items()}


def _kruskal_mst(
    nodes: Sequence[str],
    remaining: dict[tuple[str, str], float],
    *,
    preconnected: Iterable[tuple[str, str]] = (),
) -> list[tuple[str, str]]:
    uf = _UnionFind(nodes)
    for a, b in preconnected:
        if a in uf.parent and b in uf.parent:
            uf.union(a, b)
    ordered = sorted(remaining.items(), key=lambda item: (-item[1], item[0]))
    tree: list[tuple[str, str]] = []
    for pair, _score in ordered:
        a, b = pair
        if a not in uf.parent or b not in uf.parent:
            continue
        if uf.union(a, b):
            tree.append(pair)
            if uf.component_count <= 1:
                break
    return tree


def _modulate(
    remaining: dict[tuple[str, str], float],
    adjacency: dict[str, set[str]],
    *,
    modulation_lambda: float,
) -> dict[tuple[str, str], float]:
    hops, diameter = _hop_stats(adjacency, remaining)
    if diameter is None or diameter <= 0:
        return dict(remaining)

    updated: dict[tuple[str, str], float] = {}
    for pair, score in remaining.items():
        hop = hops.get(pair)
        distance = 1.0 if hop is None else hop / diameter
        updated[pair] = (1.0 - modulation_lambda) * score + modulation_lambda * distance
    return updated


def _hop_stats(
    adjacency: dict[str, set[str]],
    pairs: Iterable[tuple[str, str]],
) -> tuple[dict[tuple[str, str], int], int | None]:
    needed: dict[str, set[str]] = defaultdict(set)
    for a, b in pairs:
        needed[a].add(b)
        needed[b].add(a)
    hops: dict[tuple[str, str], int] = {}
    for source, targets in needed.items():
        distances = _bfs(adjacency, source)
        for target in targets:
            dist = distances.get(target)
            if dist is None:
                continue
            hops[_ordered((source, target))] = dist
    if not hops:
        return hops, None
    return hops, max(hops.values())


def _bfs(adjacency: dict[str, set[str]], source: str) -> dict[str, int | None]:
    distances: dict[str, int | None] = {node: None for node in adjacency}
    distances[source] = 0
    queue = deque([source])
    while queue:
        node = queue.popleft()
        node_dist = distances[node]
        if node_dist is None:
            continue
        for nxt in adjacency.get(node, ()):
            if distances.get(nxt) is None:
                distances[nxt] = node_dist + 1
                queue.append(nxt)
    return distances


def _adjacency(
    nodes: Sequence[str],
    edges: Iterable[tuple[str, str]],
) -> dict[str, set[str]]:
    adjacency = {node: set() for node in nodes}
    for a, b in edges:
        if a in adjacency and b in adjacency and a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    return adjacency


def _finalize(
    nodes: Sequence[str],
    selected: set[tuple[str, str]],
    *,
    n_required: int,
    method: str,
) -> PoseGraphInit:
    adjacency = _adjacency(nodes, selected)
    components = _components(adjacency)
    isolated = tuple(sorted(node for node, nbrs in adjacency.items() if not nbrs))
    diameter = _diameter(adjacency) if len(components) == 1 and isolated == () else None
    ordered = tuple(sorted(selected))
    return PoseGraphInit(
        edges=ordered,
        n_nodes=len(nodes),
        n_selected=len(ordered),
        n_required=n_required,
        components=len(components),
        diameter=diameter,
        isolated=isolated,
        method=method,
    )


def _components(adjacency: dict[str, set[str]]) -> list[list[str]]:
    seen: set[str] = set()
    groups: list[list[str]] = []
    for node in adjacency:
        if node in seen:
            continue
        queue = deque([node])
        seen.add(node)
        group = []
        while queue:
            cur = queue.popleft()
            group.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        groups.append(group)
    return groups


def _diameter(adjacency: dict[str, set[str]]) -> int:
    farthest = 0
    for source in adjacency:
        distances = _bfs(adjacency, source)
        farthest = max(farthest, max(d for d in distances.values() if d is not None))
    return farthest


def _ordered(pair: tuple[str, str]) -> tuple[str, str]:
    a, b = pair
    return (a, b) if a < b else (b, a)


class _UnionFind:
    def __init__(self, nodes: Sequence[str]) -> None:
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}
        self.component_count = len(self.parent)

    def find(self, node: str) -> str:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.component_count -= 1
        return True


def load_scored_edges(path: str | Path) -> list[ScoredEdge]:
    import csv

    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = str(row.get("source") or "retrieval")
            rows.append(
                ScoredEdge(
                    str(row["image_a"]),
                    str(row["image_b"]),
                    float(row["score"]),
                    source,
                )
            )
    return rows


def load_required_pairs(path: str | Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"{path}:{lineno}: expected two image names")
        pairs.append((parts[0], parts[1]))
    return pairs


def write_pairs(path: str | Path, pairs: Iterable[tuple[str, str]]) -> None:
    text = "".join(f"{a} {b}\n" for a, b in pairs)
    Path(path).write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="CSV with image_a,image_b,score")
    parser.add_argument("--nodes", help="optional text file of image names, one per line")
    parser.add_argument("--required", help="VPR-blind forced pairs to keep")
    parser.add_argument("--k-msts", type=int, default=2)
    parser.add_argument("--modulation-lambda", type=float, default=0.5)
    parser.add_argument("--output", required=True, help="selected pairs.txt")
    parser.add_argument("--report", help="optional JSON metrics path")
    args = parser.parse_args(argv)

    edges = load_scored_edges(args.scores)
    required = load_required_pairs(args.required) if args.required else []
    if args.nodes:
        nodes = [
            line.strip()
            for line in Path(args.nodes).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        nodes = []
        seen = set()
        for edge in edges:
            for name in (edge.image_a, edge.image_b):
                if name not in seen:
                    seen.add(name)
                    nodes.append(name)
        for a, b in required:
            for name in (a, b):
                if name not in seen:
                    seen.add(name)
                    nodes.append(name)

    result = select_pose_graph(
        nodes,
        edges,
        k_msts=args.k_msts,
        modulation_lambda=args.modulation_lambda,
        required_edges=required,
    )
    write_pairs(args.output, result.edges)
    report = {
        "method": result.method,
        "n_nodes": result.n_nodes,
        "n_selected": result.n_selected,
        "n_required": result.n_required,
        "components": result.components,
        "diameter": result.diameter,
        "isolated": list(result.isolated),
        "paper": "Wei et al., Global-Aware Edge Prioritization, CVPR 2026",
        "note": (
            "This selector ranks and sparsifies candidate edges. It does not replace "
            "S4 Doppelgangers++ or geometric verification."
        ),
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
