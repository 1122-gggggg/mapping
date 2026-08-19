from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .models import MapData


@dataclass(frozen=True)
class CovisibilityGraph:
    image_support: np.ndarray
    pair_counts: dict[tuple[int, int], int]
    adjacency: tuple[tuple[int, ...], ...]
    degrees: np.ndarray
    strong_edges: int
    components: tuple[tuple[int, ...], ...]

    def shared_points(self, a: int, b: int) -> int:
        if a > b:
            a, b = b, a
        return int(self.pair_counts.get((a, b), 0))


def build_covisibility_graph(
    map_data: MapData,
    *,
    min_shared_points: int = 15,
    max_track_for_pair_expansion: int = 20,
) -> CovisibilityGraph:
    """Build a registered-image covisibility graph from SfM tracks.

    Nodes are registered images. An undirected edge is considered strong when
    at least ``min_shared_points`` reconstructed landmarks are observed by both
    images. Very long tracks can create quadratic pair expansion, so they are
    counted toward per-image support but omitted from pair expansion above
    ``max_track_for_pair_expansion``.
    """
    image_lookup = map_data.image_index()
    image_support = np.zeros(map_data.num_images, dtype=int)
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)

    for obs in map_data.track_image_ids:
        indices = sorted({image_lookup[int(i)] for i in obs if int(i) in image_lookup})
        for i in indices:
            image_support[i] += 1
        if 2 <= len(indices) <= max_track_for_pair_expansion:
            for a, b in combinations(indices, 2):
                pair_counts[(a, b)] += 1

    adjacency_lists: list[list[int]] = [[] for _ in range(map_data.num_images)]
    strong_edges = 0
    for (a, b), shared in pair_counts.items():
        if shared >= min_shared_points:
            adjacency_lists[a].append(b)
            adjacency_lists[b].append(a)
            strong_edges += 1

    adjacency = tuple(tuple(sorted(v)) for v in adjacency_lists)
    degrees = np.asarray([len(v) for v in adjacency], dtype=int)
    components = connected_components(adjacency)
    return CovisibilityGraph(
        image_support=image_support,
        pair_counts=dict(pair_counts),
        adjacency=adjacency,
        degrees=degrees,
        strong_edges=strong_edges,
        components=components,
    )


def connected_components(adjacency: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    seen: set[int] = set()
    components: list[tuple[int, ...]] = []
    for start in range(len(adjacency)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            u = stack.pop()
            component.append(u)
            for v in adjacency[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        components.append(tuple(sorted(component)))
    return tuple(components)


def nearest_neighbor_distances(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if len(xyz) < 2:
        return np.asarray([], dtype=float)
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(xyz).query(xyz, k=2)
    return np.asarray(distances[:, 1], dtype=float)
