from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .adapters.base import Matcher, Retriever
from .bridge import BridgeGraph, validate_bridge
from .change import save_mask_bundle
from .config import UpdateMapConfig
from .experiments import aggregate_query_results, compare_to_baseline
from .geometry import project_points
from .models import (
    BaseMap,
    BridgeEdge,
    Camera,
    HistoricalReference,
    ImageRecord,
    Landmark,
    MapImage,
    MaskBundle,
    MatchSet,
    Pose,
    PoseQuality,
    QueryResult,
    ReferenceCandidate,
    RetrievalResult,
    Sim3,
    UtilityBreakdown,
)
from .pipeline import HistoricalAugmentationPipeline
from .reporting import write_json
from .selection import greedy_select_references
from .states import (
    ImageSource,
    MaskLabel,
    QualityStatus,
    ReferenceProvenance,
    ReferenceState,
)


class SyntheticRetriever(Retriever):
    def __init__(self, references: list[str]):
        self.references = references

    def retrieve(self, query_id: str, query_path: Path, top_k: int) -> list[RetrievalResult]:
        return [RetrievalResult(reference, 1.0 - 0.01 * idx) for idx, reference in enumerate(self.references[:top_k])]


class SyntheticMatcher(Matcher):
    def __init__(self, matches: dict[tuple[str, str], MatchSet]):
        self.matches = matches

    def match(
        self,
        query_id: str,
        query_path: Path,
        reference_id: str,
        reference_path: Path,
    ) -> MatchSet:
        return self.matches[(query_id, reference_id)]


def _pose_from_center(center: np.ndarray, yaw_deg: float = 0.0) -> Pose:
    rotation_wc = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    rotation_cw = rotation_wc.T
    return Pose(rotation_cw, -(rotation_cw @ np.asarray(center, dtype=np.float64)))


def create_synthetic_base_map() -> tuple[BaseMap, Camera, dict[str, np.ndarray]]:
    camera = Camera(1, "PINHOLE", 1280, 720, np.array([850.0, 850.0, 640.0, 360.0]))
    xs = np.linspace(-3.0, 3.0, 10)
    ys = np.linspace(-1.8, 1.8, 7)
    points_xyz = np.array([[x, y, 9.0 + 0.15 * np.sin(x)] for y in ys for x in xs], dtype=np.float64)
    points = {
        index + 1: Landmark(index + 1, xyz, np.array([180, 180, 180], dtype=np.uint8), 0.2)
        for index, xyz in enumerate(points_xyz)
    }
    poses = {
        "ref_a.jpg": _pose_from_center(np.array([-0.8, 0.0, 0.0])),
        "ref_b.jpg": _pose_from_center(np.array([0.8, 0.0, 0.0])),
    }
    images: dict[int, MapImage] = {}
    projected_by_name: dict[str, np.ndarray] = {}
    for image_id, (name, pose) in enumerate(poses.items(), start=1):
        xys, depth = project_points(points_xyz, pose, camera)
        valid = (
            (depth > 0)
            & (xys[:, 0] >= 0)
            & (xys[:, 0] < camera.width)
            & (xys[:, 1] >= 0)
            & (xys[:, 1] < camera.height)
        )
        point_ids = np.arange(1, len(points_xyz) + 1, dtype=np.int64)[valid]
        images[image_id] = MapImage(
            image_id=image_id,
            name=name,
            camera_id=1,
            pose=pose,
            xys=xys[valid],
            point3d_ids=point_ids,
        )
        projected_by_name[name] = xys[valid]
        for point2d_idx, point_id in enumerate(point_ids):
            points[int(point_id)].track.append((image_id, point2d_idx))
    return BaseMap({1: camera}, images, points, root=None, source_format="synthetic"), camera, projected_by_name


def build_synthetic_matches(
    base_map: BaseMap,
    camera: Camera,
    query_id: str,
    query_pose: Pose,
    noise_px: float = 0.25,
    seed: int = 7,
) -> dict[tuple[str, str], MatchSet]:
    rng = np.random.default_rng(seed)
    output: dict[tuple[str, str], MatchSet] = {}
    all_points = np.stack([base_map.points3d[idx].xyz for idx in sorted(base_map.points3d)], axis=0)
    query_xy, query_depth = project_points(all_points, query_pose, camera)
    for image in base_map.images.values():
        ids = image.point3d_ids
        query_selected = query_xy[ids - 1]
        valid = (
            (query_depth[ids - 1] > 0)
            & (query_selected[:, 0] >= 0)
            & (query_selected[:, 0] < camera.width)
            & (query_selected[:, 1] >= 0)
            & (query_selected[:, 1] < camera.height)
        )
        query_values = query_selected[valid] + rng.normal(0.0, noise_px, size=(int(valid.sum()), 2))
        reference_values = image.xys[valid] + rng.normal(0.0, noise_px * 0.25, size=(int(valid.sum()), 2))
        output[(query_id, image.name)] = MatchSet(
            query_id=query_id,
            reference_id=image.name,
            query_xy=query_values,
            reference_xy=reference_values,
            confidence=np.full(int(valid.sum()), 0.95, dtype=np.float64),
        )
    return output


def run_synthetic_demo(output_dir: str | Path) -> dict[str, object]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    base_map, camera, _ = create_synthetic_base_map()
    for name in ("ref_a.jpg", "ref_b.jpg", "historical_query.jpg"):
        cv2.imwrite(str(destination / name), np.full((camera.height, camera.width, 3), 127, dtype=np.uint8))
    query_id = "historical_query"
    query_pose = _pose_from_center(np.array([0.0, 0.15, 0.0]), yaw_deg=1.0)
    matches = build_synthetic_matches(base_map, camera, query_id, query_pose)
    config = UpdateMapConfig()
    config.pose.gate.min_unique_point3d = 20
    config.pose.gate.min_independent_reference_support = 2
    config.pose.gate.max_fim_condition_number = 1e8
    config.pose.gate.max_translation_std = 10.0
    config.pose.gate.max_rotation_std_deg = 20.0
    config.pose.characteristic_length = 5.0
    config.lifting.snap_radius_px = 2.0
    pipeline = HistoricalAugmentationPipeline(
        base_map=base_map,
        config=config,
        retriever=SyntheticRetriever(["ref_a.jpg", "ref_b.jpg"]),
        matcher=SyntheticMatcher(matches),
        reference_paths={
            "ref_a.jpg": destination / "ref_a.jpg",
            "ref_b.jpg": destination / "ref_b.jpg",
        },
    )
    record = ImageRecord(
        image_id=query_id,
        path=destination / "historical_query.jpg",
        source=ImageSource.HISTORICAL_UPDATE,
        session_id="old_session_001",
        quality_status=QualityStatus.GOOD,
    )
    direct = pipeline.direct_register(record, camera)
    labels = np.full((camera.height, camera.width), int(MaskLabel.STABLE), dtype=np.uint8)
    # Mark a deterministic region changed to verify association filtering.
    labels[250:360, 500:650] = int(MaskLabel.CHANGED)
    mask = MaskBundle(labels)
    mask_path = destination / "stable_mask.npz"
    save_mask_bundle(mask, mask_path)
    reference = pipeline.apply_historical_stable_mask(direct, mask, mask_path)

    graph = BridgeGraph()
    bridge_edges = [
        BridgeEdge("old_far", "mid_a", 0.93, 180, 120, 0.67, 0.35),
        BridgeEdge("mid_a", "ref_a.jpg", 0.95, 190, 135, 0.71, 0.40),
        BridgeEdge("old_far", "mid_b", 0.91, 170, 110, 0.65, 0.32),
        BridgeEdge("mid_b", "ref_b.jpg", 0.94, 200, 145, 0.73, 0.42),
    ]
    graph.add_edges(bridge_edges)
    disjoint = graph.edge_disjoint_paths_to_anchors("old_far", {"ref_a.jpg", "ref_b.jpg"})
    bridge_validation = validate_bridge(
        {"ref_a.jpg", "ref_b.jpg"},
        disjoint,
        config.bridge,
        cycle_transforms=(Sim3.identity(), Sim3.identity()),
    )

    if reference is None:
        raise RuntimeError("Synthetic direct registration did not produce a reference")
    reference.state = ReferenceState.HIST_STABLE
    candidate_direct = ReferenceCandidate(
        reference=reference,
        supports_cells={"route:p0:y0", "route:p1:y0"},
        utility=UtilityBreakdown(
            viewpoint_gain=0.8,
            edm_success_gain=0.2,
            pose_information_gain=0.4,
            stable_ratio=reference.stable_ratio,
            runtime_cost=0.1,
            risk_penalty=0.05,
        ),
        visible_point3d_ids=reference.current_point3d_ids,
    )
    bridged_reference = HistoricalReference(
        reference_id="old_far",
        image_path=destination / "historical_query.jpg",
        pose=query_pose,
        provenance=ReferenceProvenance.BRIDGED,
        state=ReferenceState.HIST_STABLE,
        stable_ratio=0.9,
        current_point3d_ids=set(list(base_map.points3d)[:40]),
        bridge_depth=2,
        anchor_ids={"ref_a.jpg", "ref_b.jpg"},
        bridge_path_count=2,
    )
    candidate_bridge = ReferenceCandidate(
        reference=bridged_reference,
        supports_cells={"route:weak:p2:y3"},
        utility=UtilityBreakdown(
            viewpoint_gain=1.0,
            edm_success_gain=0.5,
            pose_information_gain=0.6,
            stable_ratio=0.9,
            runtime_cost=0.15,
            risk_penalty=0.1,
        ),
        visible_point3d_ids=bridged_reference.current_point3d_ids,
    )
    selection = greedy_select_references(
        [candidate_direct, candidate_bridge],
        config.selection,
        current_coverage={"route:p0:y0": 2, "route:p1:y0": 1, "route:weak:p2:y3": 0},
        cell_weights={"route:weak:p2:y3": 5.0},
    )

    quality_ok = PoseQuality(num_inliers=80, num_unique_point3d=80, passed=True)
    quality_fail = PoseQuality(num_inliers=0, num_unique_point3d=0, passed=False)
    baseline_queries = [
        QueryResult("q0", True, query_pose, quality_ok, route_cell="healthy"),
        QueryResult("q1", False, None, quality_fail, route_cell="weak"),
        QueryResult("q2", False, None, quality_fail, route_cell="weak"),
    ]
    candidate_queries = [
        QueryResult("q0", True, query_pose, quality_ok, route_cell="healthy"),
        QueryResult("q1", True, query_pose, quality_ok, route_cell="weak"),
        QueryResult("q2", True, query_pose, quality_ok, route_cell="weak"),
    ]
    from .models import ExperimentResult

    baseline = ExperimentResult("E0_BASE_CURRENT_ONLY", baseline_queries, aggregate_query_results(baseline_queries))
    augmented = ExperimentResult("E5_PRODUCTION_CANDIDATE", candidate_queries, aggregate_query_results(candidate_queries))
    config.validation.min_failure_run_reduction = 1
    config.validation.min_weak_cell_success_gain = 0.1
    regression = compare_to_baseline(baseline, augmented, config.validation)

    report = {
        "direct_status": direct.pose_estimate.status.value,
        "direct_passed": direct.pose_estimate.quality.passed,
        "direct_associations_before_mask": len(direct.aggregated_correspondences),
        "direct_associations_after_mask": len(direct.accepted_associations),
        "stable_ratio": reference.stable_ratio,
        "bridge_validation": asdict(bridge_validation),
        "selected_references": [item.reference.reference_id for item in selection.selected],
        "uncovered_cells": selection.uncovered_cells,
        "regression": asdict(regression),
        "core_immutable": pipeline.verify_core_immutable(),
    }
    write_json(report, destination / "synthetic_report.json")
    pipeline.export_sidecar(
        [reference, bridged_reference],
        direct.accepted_associations,
        destination / "sidecar",
    )
    return report
