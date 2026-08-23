import numpy as np

from sfm_diagnosis.graph import build_covisibility_graph
from sfm_diagnosis.models import CameraIntrinsics, MapData
from sfm_diagnosis.report import map_health_summary
from sfm_diagnosis.weak_regions import analyze_weak_regions


def _map_from_tracks(image_ids: np.ndarray, tracks: list[np.ndarray]) -> MapData:
    num_points = len(tracks)
    centers = np.asarray(
        [[float(index), 0.0, 0.0] for index in range(len(image_ids))],
        dtype=float,
    )
    return MapData(
        point_ids=np.arange(num_points),
        points_xyz=np.asarray(
            [[float(index), 0.0, 5.0] for index in range(num_points)],
            dtype=float,
        ),
        point_rgb=np.zeros((num_points, 3), dtype=np.uint8),
        point_errors=np.full(num_points, 0.5),
        track_lengths=np.asarray([len(track) for track in tracks], dtype=np.int32),
        track_image_ids=tracks,
        image_ids=image_ids,
        image_names=[f"im_{image_id}.jpg" for image_id in image_ids],
        image_camera_ids=np.zeros(len(image_ids), dtype=int),
        image_centers=centers,
        image_R_wc=np.repeat(np.eye(3)[None], len(image_ids), axis=0),
        cameras={
            0: CameraIntrinsics(0, "PINHOLE", 1000, 800, 500, 500, 500, 400)
        },
    )


def test_exact_sparse_support_handles_many_long_tracks_without_isolation():
    image_ids = np.arange(100, 121)
    tracks = [image_ids.copy() for _ in range(15)]
    graph = build_covisibility_graph(_map_from_tracks(image_ids, tracks), min_shared_points=15)

    assert graph.support_mode == "exact"
    assert graph.omitted_long_track_count == 0
    assert graph.strong_edges == 210
    assert len(graph.components) == 1
    assert np.all(graph.degrees == 20)
    assert np.sum(graph.degrees == 0) == 0

    report = map_health_summary(_map_from_tracks(image_ids, tracks))
    assert report["covisibility"]["strong_edges"] == 210
    assert report["covisibility"]["connected_components"] == 1
    assert report["covisibility"]["isolated_images"] == 0
    assert report["covisibility"]["support_mode"] == "exact"

    weak = analyze_weak_regions(_map_from_tracks(image_ids, tracks))
    assert weak.summary["covisibility"]["support_mode"] == "exact"
    assert weak.summary["covisibility"]["omitted_long_track_count"] == 0
    assert weak.summary["num_weak_images"] == 0
    assert all(row["covisibility_degree"] > 0 for row in weak.images)


def test_exact_and_explicit_legacy_modes_match_on_small_tracks():
    image_ids = np.arange(1, 5)
    tracks = [
        np.asarray([1, 2]),
        np.asarray([2, 3]),
        np.asarray([3, 4]),
        np.asarray([4, 1]),
        np.asarray([1, 3]),
    ]
    map_data = _map_from_tracks(image_ids, tracks)
    exact = build_covisibility_graph(map_data, min_shared_points=1)
    legacy = build_covisibility_graph(
        map_data,
        min_shared_points=1,
        max_track_for_pair_expansion=4,
    )

    assert exact.support_mode == "exact"
    assert legacy.support_mode == "legacy_approximation"
    assert legacy.omitted_long_track_count == 0
    assert exact.pair_counts == legacy.pair_counts
    assert exact.adjacency == legacy.adjacency
    assert np.array_equal(exact.degrees, legacy.degrees)
    assert exact.components == legacy.components


def test_exact_pair_counts_are_threshold_retained():
    image_ids = np.arange(1, 3)
    map_data = _map_from_tracks(image_ids, [np.asarray([1, 2])])
    exact = build_covisibility_graph(map_data, min_shared_points=2)
    legacy = build_covisibility_graph(
        map_data,
        min_shared_points=2,
        max_track_for_pair_expansion=2,
    )

    assert exact.pair_counts == {}
    assert exact.shared_points(0, 1) == 0
    assert legacy.pair_counts == {(0, 1): 1}
    assert legacy.shared_points(0, 1) == 1
