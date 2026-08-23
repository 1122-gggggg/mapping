import numpy as np
from scipy.spatial.transform import Rotation

from update_map.bridge import BridgeGraph, CorrespondenceTrackGraph, validate_bridge
from update_map.config import BridgeConfig
from update_map.geometry import ransac_sim3
from update_map.models import BridgeEdge, MatchSet, Observation, Sim3


def test_ransac_sim3_recovers_alignment() -> None:
    rng = np.random.default_rng(5)
    source = rng.normal(size=(50, 3))
    rotation = Rotation.from_euler("zyx", [20, -5, 3], degrees=True).as_matrix()
    expected = Sim3(1.25, rotation, np.array([2.0, -1.0, 0.5]))
    target = expected.transform(source)
    target[:5] += 5.0
    estimated, inliers, _ = ransac_sim3(source, target, threshold=0.05, random_seed=3)
    assert inliers.sum() >= 45
    assert abs(estimated.scale - expected.scale) < 1e-3
    assert np.linalg.norm(estimated.t - expected.t) < 1e-3


def test_multi_anchor_bridge_passes_and_single_anchor_fails() -> None:
    graph = BridgeGraph()
    graph.add_edges(
        [
            BridgeEdge("q", "a1", 0.9, 100, 70, 0.7, 0.3),
            BridgeEdge("a1", "anchor_a", 0.9, 100, 70, 0.7, 0.3),
            BridgeEdge("q", "b1", 0.9, 100, 70, 0.7, 0.3),
            BridgeEdge("b1", "anchor_b", 0.9, 100, 70, 0.7, 0.3),
        ]
    )
    paths = graph.edge_disjoint_paths_to_anchors("q", {"anchor_a", "anchor_b"})
    config = BridgeConfig()
    valid = validate_bridge(
        {"anchor_a", "anchor_b"}, paths, config, (Sim3.identity(), Sim3.identity())
    )
    assert valid.passed
    invalid = validate_bridge(
        {"anchor_a"}, paths[:1], config, (Sim3.identity(), Sim3.identity())
    )
    assert not invalid.passed
    assert "min_anchor_count" in invalid.failed_gates


def test_current_point_id_propagation_rejects_conflicting_track() -> None:
    graph = CorrespondenceTrackGraph(quantization_px=0.5)
    graph.add_matches(
        MatchSet(
            "old",
            "current",
            np.array([[5.0, 5.0], [10.0, 10.0]]),
            np.array([[1.0, 1.0], [2.0, 2.0]]),
            np.ones(2),
        )
    )
    graph.seed_observations(
        "current",
        [
            Observation("current", 0, np.array([1.0, 1.0]), 100),
            Observation("current", 1, np.array([2.0, 2.0]), 200),
        ],
    )
    propagated, conflicts = graph.propagated_point_ids()
    assert conflicts == 0
    assert {item[1] for item in propagated["old"]} == {100, 200}

    conflict_graph = CorrespondenceTrackGraph(quantization_px=0.5)
    conflict_graph.add_matches(
        MatchSet(
            "old",
            "current",
            np.array([[5.0, 5.0], [5.1, 5.1]]),
            np.array([[1.0, 1.0], [2.0, 2.0]]),
            np.ones(2),
        )
    )
    conflict_graph.seed_observations(
        "current",
        [
            Observation("current", 0, np.array([1.0, 1.0]), 100),
            Observation("current", 1, np.array([2.0, 2.0]), 200),
        ],
    )
    propagated, conflicts = conflict_graph.propagated_point_ids()
    assert conflicts == 1
    assert "old" not in propagated
