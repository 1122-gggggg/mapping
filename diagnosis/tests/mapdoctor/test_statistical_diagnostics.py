from __future__ import annotations

import pytest

from mapdoctor.benchmark import QueryLocalizationResult
from mapdoctor.config import LocalizationThresholds
from mapdoctor.diagnostics.graph import analyze_covisibility_fragility
from mapdoctor.diagnostics.regions import RegionDiagnosisConfig, diagnose_regions
from mapdoctor.diagnostics.risk_coverage import evaluate_risk_coverage
from mapdoctor.diagnostics.statistics import wilson_interval
from mapdoctor.model import ImageRecord, MapModel, Point3D, TrackElement


def query(
    name: str,
    *,
    passes: bool,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
) -> QueryLocalizationResult:
    return QueryLocalizationResult(
        query=name,
        success=passes,
        inliers=60 if passes else 0,
        inlier_ratio=0.6 if passes else 0.0,
        reproj_p90_px=1.5 if passes else None,
        hull_coverage=0.4 if passes else 0.0,
        grid4_occupancy=10 if passes else 0,
        positive_depth_ratio=1.0 if passes else 0.0,
        pose_consensus=0.9 if passes else 0.0,
        x=x,
        y=y,
        z=z,
    )


def test_wilson_interval_does_not_treat_one_failure_as_certain():
    interval = wilson_interval(1, 1)
    assert interval.low < 0.30
    assert interval.high == pytest.approx(1.0)


def test_single_failure_is_insufficient_evidence_not_weak():
    report = diagnose_regions(
        [query("q1", passes=False)],
        LocalizationThresholds(),
    )
    assert report.regions[0].status == "INSUFFICIENT_EVIDENCE"
    assert report.regions[0].strict_failures == 1


def test_repeated_failures_become_weak_and_repeated_successes_become_healthy():
    results = [
        query(f"bad-{index}", passes=False, x=0.0)
        for index in range(12)
    ] + [
        query(f"good-{index}", passes=True, x=20.0)
        for index in range(30)
    ]
    report = diagnose_regions(
        results,
        LocalizationThresholds(),
        config=RegionDiagnosisConfig(
            weak_failure_rate=0.30,
            healthy_failure_rate=0.15,
            min_samples=8,
        ),
    )
    by_region = {region.region_id: region for region in report.regions}
    assert by_region["grid:0:0:0"].status == "WEAK"
    assert by_region["grid:4:0:0"].status == "HEALTHY"


def test_risk_coverage_rewards_failure_ranking_and_is_tie_invariant():
    results = [
        query("s1", passes=True),
        query("s2", passes=True),
        query("f1", passes=False),
        query("f2", passes=False),
    ]
    perfect = evaluate_risk_coverage(
        results,
        {"s1": 0.1, "s2": 0.2, "f1": 0.8, "f2": 0.9},
        LocalizationThresholds(),
    )
    inverse = evaluate_risk_coverage(
        results,
        {"s1": 0.9, "s2": 0.8, "f1": 0.2, "f2": 0.1},
        LocalizationThresholds(),
    )
    assert perfect.aurc == pytest.approx(perfect.oracle_aurc)
    assert perfect.failure_auroc == pytest.approx(1.0)
    assert inverse.aurc > perfect.aurc

    tied_a = evaluate_risk_coverage(
        results,
        {result.query: 0.5 for result in results},
        LocalizationThresholds(),
    )
    reversed_results = list(reversed(results))
    tied_b = evaluate_risk_coverage(
        reversed_results,
        {result.query: 0.5 for result in reversed_results},
        LocalizationThresholds(),
    )
    assert tied_a.aurc == pytest.approx(tied_b.aurc)
    assert tied_a.aurc == pytest.approx(0.5)


def test_risk_coverage_fails_closed_on_missing_scores():
    with pytest.raises(ValueError, match="missing required queries"):
        evaluate_risk_coverage(
            [query("q1", passes=True), query("q2", passes=False)],
            {"q1": 0.1},
            LocalizationThresholds(),
        )


def model_from_edges(edges: list[tuple[int, int]], node_count: int) -> MapModel:
    images = {
        image_id: ImageRecord(
            id=image_id,
            camera_id=1,
            name=f"im{image_id}.jpg",
            center=(float(image_id), 0.0, 0.0),
            viewing_direction=(0.0, 0.0, 1.0),
        )
        for image_id in range(1, node_count + 1)
    }
    points = {
        point_id: Point3D(
            id=point_id,
            xyz=(float(point_id), 0.0, 5.0),
            rgb=(255, 255, 255),
            error=0.5,
            track=(
                TrackElement(image_id=image_a, point2d_idx=0),
                TrackElement(image_id=image_b, point2d_idx=0),
            ),
        )
        for point_id, (image_a, image_b) in enumerate(edges, start=1)
    }
    return MapModel(source="synthetic", format="test", images=images, points3d=points)


def test_chain_covisibility_graph_exposes_articulations_and_bridges():
    model = model_from_edges([(1, 2), (2, 3), (3, 4)], 4)
    report = analyze_covisibility_fragility(
        model,
        minimum_shared_landmarks=1,
        route_image_names=["im1.jpg", "im4.jpg"],
    )
    assert {row.image_id for row in report.articulation_images} == {2, 3}
    assert len(report.bridge_edges) == 3
    assert all(edge.splits_route for edge in report.bridge_edges)


def test_cycle_has_no_single_point_of_failure_and_isolated_nodes_are_preserved():
    model = model_from_edges([(1, 2), (2, 3), (1, 3)], 4)
    report = analyze_covisibility_fragility(model, minimum_shared_landmarks=1)
    assert not report.articulation_images
    assert not report.bridge_edges
    assert report.component_sizes == (3, 1)
    assert report.isolated_images == ({"image_id": 4, "image_name": "im4.jpg"},)


def test_operating_points_do_not_split_equal_risk_groups():
    results = [
        query("s1", passes=True),
        query("s2", passes=True),
        query("f1", passes=False),
        query("f2", passes=False),
    ]
    report = evaluate_risk_coverage(
        results,
        {result.query: 0.5 for result in results},
        LocalizationThresholds(),
        target_failure_rates=(0.25,),
    )
    assert report.operating_points["0.25"] is None
    assert all(
        point.randomized_within_tie
        for point in report.curve[:-1]
    )
    assert report.curve[-1].randomized_within_tie is False


def test_region_assignments_fail_on_unknown_query():
    with pytest.raises(ValueError, match="outside the benchmark"):
        diagnose_regions(
            [query("q1", passes=True)],
            LocalizationThresholds(),
            assignments={"q1": "route-a", "typo": "route-a"},
        )
