from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .config import BridgeConfig
from .geometry import pose_distance, rotation_angle_deg
from .models import BridgeEdge, MatchSet, Observation, Pose, Sim3


@dataclass
class BridgePath:
    nodes: list[str]
    cost: float
    confidence: float


@dataclass
class BridgeValidation:
    passed: bool
    anchor_count: int
    disjoint_path_count: int
    rotation_cycle_error_deg: float | None = None
    translation_cycle_error: float | None = None
    scale_cycle_error_fraction: float | None = None
    failed_gates: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RelativePoseEdge:
    """Measured camera transform ``T_target_source``."""

    source: str
    target: str
    T_target_source: Pose
    weight: float = 1.0


def edge_confidence(
    num_matches: int,
    num_inliers: int,
    inlier_ratio: float,
    spatial_coverage: float,
    median_epipolar_error: float | None = None,
) -> float:
    match_term = 1.0 - np.exp(-max(num_matches, 0) / 100.0)
    inlier_term = 1.0 - np.exp(-max(num_inliers, 0) / 40.0)
    ratio_term = np.clip(inlier_ratio, 0.0, 1.0)
    coverage_term = np.clip(spatial_coverage / 0.25, 0.0, 1.0)
    epipolar_term = (
        1.0
        if median_epipolar_error is None
        else float(np.exp(-max(median_epipolar_error, 0.0) / 2.0))
    )
    score = (
        0.20 * match_term
        + 0.25 * inlier_term
        + 0.25 * ratio_term
        + 0.20 * coverage_term
        + 0.10 * epipolar_term
    )
    return float(np.clip(score, 1e-6, 1.0))


class BridgeGraph:
    def __init__(self) -> None:
        self.graph = nx.Graph()

    def add_edge(self, edge: BridgeEdge) -> None:
        self.graph.add_edge(edge.source, edge.target, edge=edge, weight=edge.cost)

    def add_edges(self, edges: Iterable[BridgeEdge]) -> None:
        for edge in edges:
            self.add_edge(edge)

    def valid_edges(self, config: BridgeConfig) -> list[BridgeEdge]:
        output: list[BridgeEdge] = []
        for _, _, data in self.graph.edges(data=True):
            edge: BridgeEdge = data["edge"]
            if (
                edge.num_matches >= config.min_edge_matches
                and edge.num_inliers >= config.min_edge_inliers
                and edge.inlier_ratio >= config.min_edge_inlier_ratio
                and edge.spatial_coverage >= config.min_edge_spatial_coverage
                and edge.confidence >= config.min_edge_confidence
            ):
                output.append(edge)
        return output

    def filtered(self, config: BridgeConfig) -> "BridgeGraph":
        result = BridgeGraph()
        result.add_edges(self.valid_edges(config))
        return result

    def best_path(self, source: str, anchors: set[str], max_depth: int | None = None) -> BridgePath | None:
        best: BridgePath | None = None
        for anchor in anchors:
            if source not in self.graph or anchor not in self.graph:
                continue
            try:
                nodes = nx.shortest_path(self.graph, source, anchor, weight="weight")
                if max_depth is not None and len(nodes) - 1 > max_depth:
                    continue
                cost = float(nx.path_weight(self.graph, nodes, weight="weight"))
                path = BridgePath(nodes=nodes, cost=cost, confidence=float(np.exp(-cost)))
                if best is None or path.cost < best.cost:
                    best = path
            except nx.NetworkXNoPath:
                continue
        return best

    def paths_to_distinct_anchors(
        self, source: str, anchors: set[str], max_depth: int | None = None
    ) -> list[BridgePath]:
        paths: list[BridgePath] = []
        for anchor in sorted(anchors):
            path = self.best_path(source, {anchor}, max_depth=max_depth)
            if path is not None:
                paths.append(path)
        return sorted(paths, key=lambda item: item.cost)

    def edge_disjoint_paths_to_anchors(
        self, source: str, anchors: set[str], max_paths: int = 4
    ) -> list[BridgePath]:
        if source not in self.graph:
            return []
        augmented = self.graph.copy()
        sink = "__CURRENT_ANCHOR_SINK__"
        augmented.add_node(sink)
        for anchor in anchors:
            if anchor in augmented:
                augmented.add_edge(anchor, sink, weight=0.0)
        try:
            raw_paths = list(nx.edge_disjoint_paths(augmented, source, sink))
        except (nx.NetworkXNoPath, nx.NetworkXError):
            return []
        output: list[BridgePath] = []
        for nodes in raw_paths[:max_paths]:
            trimmed = nodes[:-1]
            if len(trimmed) < 2:
                continue
            cost = float(nx.path_weight(self.graph, trimmed, weight="weight"))
            output.append(BridgePath(trimmed, cost, float(np.exp(-cost))))
        return sorted(output, key=lambda item: item.cost)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, int, int], tuple[str, int, int]] = {}

    def add(self, item: tuple[str, int, int]) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: tuple[str, int, int]) -> tuple[str, int, int]:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, a: tuple[str, int, int], b: tuple[str, int, int]) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


class CorrespondenceTrackGraph:
    """Propagate fixed current point IDs through a network of image correspondences.

    Coordinates are quantized only for track construction. Conflicting current point seeds in
    one component are rejected instead of selecting an arbitrary ID.
    """

    def __init__(self, quantization_px: float = 1.0):
        if quantization_px <= 0:
            raise ValueError("quantization_px must be positive")
        self.quantization_px = quantization_px
        self.uf = _UnionFind()
        self.node_xy: dict[tuple[str, int, int], np.ndarray] = {}
        self.seeds: dict[tuple[str, int, int], set[int]] = {}
        self.node_confidence: dict[tuple[str, int, int], float] = {}

    def _node(self, image_id: str, xy: np.ndarray) -> tuple[str, int, int]:
        value = np.asarray(xy, dtype=np.float64).reshape(2)
        qx, qy = np.rint(value / self.quantization_px).astype(int)
        node = (str(image_id), int(qx), int(qy))
        self.uf.add(node)
        self.node_xy.setdefault(node, value)
        return node

    def add_matches(self, matches: MatchSet) -> None:
        for query_xy, reference_xy, confidence in zip(
            matches.query_xy, matches.reference_xy, matches.confidence, strict=True
        ):
            query_node = self._node(matches.query_id, query_xy)
            reference_node = self._node(matches.reference_id, reference_xy)
            self.uf.union(query_node, reference_node)
            self.node_confidence[query_node] = max(
                self.node_confidence.get(query_node, 0.0), float(confidence)
            )
            self.node_confidence[reference_node] = max(
                self.node_confidence.get(reference_node, 0.0), float(confidence)
            )

    def seed_observations(self, image_id: str, observations: Sequence[Observation]) -> None:
        for observation in observations:
            node = self._node(image_id, observation.xy)
            self.seeds.setdefault(node, set()).add(observation.point3d_id)

    def propagated_point_ids(self) -> tuple[dict[str, list[tuple[np.ndarray, int, float]]], int]:
        component_seeds: dict[tuple[str, int, int], set[int]] = {}
        component_nodes: dict[tuple[str, int, int], list[tuple[str, int, int]]] = {}
        for node in self.uf.parent:
            root = self.uf.find(node)
            component_nodes.setdefault(root, []).append(node)
            component_seeds.setdefault(root, set()).update(self.seeds.get(node, set()))
        output: dict[str, list[tuple[np.ndarray, int, float]]] = {}
        conflicts = 0
        for root, nodes in component_nodes.items():
            point_ids = component_seeds.get(root, set())
            if len(point_ids) != 1:
                if len(point_ids) > 1:
                    conflicts += 1
                continue
            point_id = next(iter(point_ids))
            for node in nodes:
                image_id = node[0]
                confidence = self.node_confidence.get(node, 1.0 if node in self.seeds else 0.5)
                output.setdefault(image_id, []).append((self.node_xy[node], point_id, confidence))
        return output, conflicts


def sim3_cycle_errors(transform_a: Sim3, transform_b: Sim3) -> tuple[float, float, float]:
    delta = transform_a.compose(transform_b.inverse())
    rotation = rotation_angle_deg(delta.R)
    translation = float(np.linalg.norm(delta.t))
    scale_fraction = abs(delta.scale - 1.0)
    return rotation, translation, scale_fraction


def validate_bridge(
    anchor_ids: set[str],
    disjoint_paths: Sequence[BridgePath],
    config: BridgeConfig,
    cycle_transforms: tuple[Sim3, Sim3] | None = None,
) -> BridgeValidation:
    failed: list[str] = []
    if len(anchor_ids) < config.min_anchor_count:
        failed.append("min_anchor_count")
    if len(disjoint_paths) < config.min_disjoint_paths:
        failed.append("min_disjoint_paths")
    rotation = translation = scale = None
    if cycle_transforms is not None:
        rotation, translation, scale = sim3_cycle_errors(*cycle_transforms)
        if rotation > config.max_rotation_cycle_deg:
            failed.append("max_rotation_cycle_deg")
        if translation > config.max_translation_cycle:
            failed.append("max_translation_cycle")
        if scale > config.max_scale_cycle_fraction:
            failed.append("max_scale_cycle_fraction")
    else:
        failed.append("missing_cycle_evidence")
    return BridgeValidation(
        passed=not failed,
        anchor_count=len(anchor_ids),
        disjoint_path_count=len(disjoint_paths),
        rotation_cycle_error_deg=rotation,
        translation_cycle_error=translation,
        scale_cycle_error_fraction=scale,
        failed_gates=failed,
    )


def _pose_to_vector(pose: Pose) -> np.ndarray:
    return np.concatenate([Rotation.from_matrix(pose.R_cw).as_rotvec(), pose.t_cw])


def _vector_to_pose(vector: np.ndarray) -> Pose:
    return Pose(Rotation.from_rotvec(vector[:3]).as_matrix(), vector[3:6])


def optimize_anchored_pose_graph(
    initial_poses: Mapping[str, Pose],
    anchors: set[str],
    edges: Sequence[RelativePoseEdge],
    translation_scale: float = 1.0,
) -> dict[str, Pose]:
    """Optimize historical poses while keeping all current anchors fixed.

    Edge convention: ``T_target_source = T_target_world @ inverse(T_source_world)``.
    The residual uses rotation-vector and translation components of the transform error.
    """

    unknown = [node for node in initial_poses if node not in anchors]
    if not unknown:
        return dict(initial_poses)
    index = {node: idx for idx, node in enumerate(unknown)}
    initial = np.concatenate([_pose_to_vector(initial_poses[node]) for node in unknown])

    def unpack(parameters: np.ndarray) -> dict[str, Pose]:
        result = dict(initial_poses)
        for node, idx in index.items():
            result[node] = _vector_to_pose(parameters[6 * idx : 6 * idx + 6])
        return result

    def residual(parameters: np.ndarray) -> np.ndarray:
        poses = unpack(parameters)
        values: list[np.ndarray] = []
        for edge in edges:
            source = poses[edge.source].as_matrix()
            target = poses[edge.target].as_matrix()
            predicted = target @ np.linalg.inv(source)
            measured = edge.T_target_source.as_matrix()
            error = np.linalg.inv(measured) @ predicted
            rotation_error = Rotation.from_matrix(error[:3, :3]).as_rotvec()
            translation_error = error[:3, 3] / max(translation_scale, 1e-9)
            values.append(np.sqrt(max(edge.weight, 1e-9)) * np.concatenate([rotation_error, translation_error]))
        return np.concatenate(values) if values else np.empty(0, dtype=np.float64)

    if not edges:
        return dict(initial_poses)
    result = least_squares(residual, initial, loss="huber", f_scale=1.0, max_nfev=500)
    return unpack(result.x)


def pose_cycle_error(pose_a: Pose, pose_b: Pose) -> tuple[float, float]:
    return pose_distance(pose_a, pose_b)
