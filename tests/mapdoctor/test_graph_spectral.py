from __future__ import annotations

import itertools

import pytest

from mapdoctor.diagnostics.graph import analyze_covisibility_fragility
from mapdoctor.model import ImageRecord, MapModel, Point3D, TrackElement


def model_from_weighted_edges(
    node_count: int,
    edges: list[tuple[int, int, int]],
) -> MapModel:
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
    points = {}
    point_id = 1
    for image_a, image_b, support in edges:
        for _ in range(support):
            points[point_id] = Point3D(
                id=point_id,
                xyz=(float(point_id), 0.0, 5.0),
                rgb=(255, 255, 255),
                error=0.5,
                track=(
                    TrackElement(image_id=image_a, point2d_idx=0),
                    TrackElement(image_id=image_b, point2d_idx=0),
                ),
            )
            point_id += 1
    return MapModel(
        source="synthetic",
        format="test",
        images=images,
        points3d=points,
    )


def test_existing_articulation_and_bridge_behavior_is_preserved():
    model = model_from_weighted_edges(
        4,
        [(1, 2, 1), (2, 3, 1), (3, 4, 1)],
    )
    report = analyze_covisibility_fragility(
        model,
        minimum_shared_landmarks=1,
        route_image_names=["im1.jpg", "im4.jpg"],
    )
    assert {row.image_id for row in report.articulation_images} == {2, 3}
    assert len(report.bridge_edges) == 3
    assert all(edge.splits_route for edge in report.bridge_edges)
    assert (
        report.shared_landmark_backend
        == "scipy_sparse_block_incidence_product"
    )
    assert report.estimated_pair_expansions == 3


def test_spectral_lambda2_detects_soft_bottleneck_without_exact_bridge():
    complete_edges = [
        (a, b, 10)
        for a, b in itertools.combinations(range(1, 7), 2)
    ]
    complete = analyze_covisibility_fragility(
        model_from_weighted_edges(6, complete_edges),
        minimum_shared_landmarks=1,
    )

    bottleneck_edges = []
    for cluster in ((1, 2, 3), (4, 5, 6)):
        bottleneck_edges.extend(
            (a, b, 10) for a, b in itertools.combinations(cluster, 2)
        )
    bottleneck_edges.extend([(2, 4, 1), (3, 5, 1)])
    bottleneck = analyze_covisibility_fragility(
        model_from_weighted_edges(6, bottleneck_edges),
        minimum_shared_landmarks=1,
    )
    assert not bottleneck.bridge_edges
    assert not bottleneck.articulation_images
    assert (
        bottleneck.spectral_connectivity.normalized_laplacian_lambda2
        < complete.spectral_connectivity.normalized_laplacian_lambda2 / 5
    )


def test_threshold_sensitivity_exposes_support_collapse():
    edges = []
    for cluster in ((1, 2, 3), (4, 5, 6)):
        edges.extend(
            (a, b, 20) for a, b in itertools.combinations(cluster, 2)
        )
    edges.extend([(2, 4, 7), (3, 5, 7)])
    report = analyze_covisibility_fragility(
        model_from_weighted_edges(6, edges),
        minimum_shared_landmarks=5,
    )
    profile = {
        row.minimum_shared_landmarks: row
        for row in report.threshold_sensitivity
    }
    assert profile[5].component_count == 1
    assert profile[10].component_count == 2
    assert profile[10].largest_component_ratio == pytest.approx(0.5)


def test_long_track_counts_each_image_pair_exactly_once():
    images = {
        image_id: ImageRecord(
            id=image_id,
            camera_id=1,
            name=f"im{image_id}.jpg",
            center=(0.0, 0.0, 0.0),
            viewing_direction=(0.0, 0.0, 1.0),
        )
        for image_id in range(1, 8)
    }
    point = Point3D(
        id=1,
        xyz=(0.0, 0.0, 5.0),
        rgb=(255, 255, 255),
        error=0.1,
        track=tuple(
            TrackElement(image_id=i, point2d_idx=0)
            for i in range(1, 8)
        ),
    )
    report = analyze_covisibility_fragility(
        MapModel(
            source="synthetic",
            format="test",
            images=images,
            points3d={1: point},
        ),
        minimum_shared_landmarks=1,
    )
    assert report.edge_count == 21
    assert report.estimated_pair_expansions == 21
    assert report.track_observations == 7
