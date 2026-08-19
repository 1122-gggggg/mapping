import numpy as np

from sfm_diagnosis.evidence import BuildEvidence
from sfm_diagnosis.models import CameraIntrinsics, MapData
from sfm_diagnosis.weak_regions import (
    WeakRegionCause,
    WeakRegionConfig,
    analyze_weak_regions,
)


def mixed_map() -> MapData:
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [8.0, 0.0, 0.0],
            [8.15, 0.0, 0.0],
            [8.30, 0.0, 0.0],
        ],
        dtype=float,
    )
    image_ids = np.arange(1, 8)
    points = []
    tracks = []
    errors = []

    for x in np.linspace(-1, 2, 8):
        for y in np.linspace(-1, 2, 6):
            points.append([x, y, 6.0])
            tracks.append(image_ids[:4].copy())
            errors.append(0.5)

    weak_xy = [(x, y) for x in np.linspace(7.5, 8.8, 6) for y in (-0.4, 0.0, 0.4)]
    for k, (x, y) in enumerate(weak_xy):
        points.append([x, y, 7.0])
        tracks.append(
            np.array([image_ids[4], image_ids[5]])
            if k % 2 == 0
            else np.array([image_ids[5], image_ids[6]])
        )
        errors.append(0.8)

    points = np.asarray(points, dtype=float)
    return MapData(
        point_ids=np.arange(len(points)),
        points_xyz=points,
        point_rgb=np.zeros((len(points), 3), np.uint8),
        point_errors=np.asarray(errors, dtype=float),
        track_lengths=np.asarray([len(track) for track in tracks]),
        track_image_ids=[np.asarray(track) for track in tracks],
        image_ids=image_ids,
        image_names=[f"im_{i}.jpg" for i in image_ids],
        image_camera_ids=np.zeros(len(centers), dtype=int),
        image_centers=centers,
        image_R_wc=np.repeat(np.eye(3)[None], len(centers), axis=0),
        cameras={
            0: CameraIntrinsics(0, "PINHOLE", 1000, 800, 500, 500, 500, 400)
        },
    )


def test_weak_region_explains_track_parallax_and_graph_failures():
    cfg = WeakRegionConfig(
        covisibility_min_shared=5,
        cluster_radius=1.0,
        anchor_radius=10.0,
        weak_image_score_threshold=0.30,
    )
    result = analyze_weak_regions(mixed_map(), config=cfg)
    assert result.summary["num_weak_regions"] == 1
    region = result.regions[0]
    causes = set(region["root_causes"])
    assert WeakRegionCause.VIEW_GRAPH_ISOLATION.value in causes
    assert WeakRegionCause.TRACK_FRAGMENTATION.value in causes
    assert WeakRegionCause.LOW_PARALLAX.value in causes
    actions = {item["action"] for item in region["repair_sequence"]}
    assert "TARGETED_BRIDGE_PAIR_SELECTION" in actions
    assert "MULTIVIEW_TRACK_REPAIR" in actions
    assert "TARGETED_LATERAL_OBLIQUE_RECAPTURE" in actions


def test_pair_evidence_distinguishes_matching_ambiguity():
    m = mixed_map()
    pairs = [
        {
            "image_i": "im_5.jpg",
            "image_j": "im_1.jpg",
            "selected": True,
            "attempted": True,
            "num_matches": 200,
            "num_inliers": 15,
            "inlier_ratio": 0.075,
            "verified": True,
        },
        {
            "image_i": "im_6.jpg",
            "image_j": "im_2.jpg",
            "selected": True,
            "attempted": True,
            "num_matches": 180,
            "num_inliers": 20,
            "inlier_ratio": 0.111,
            "verified": True,
        },
    ]
    cfg = WeakRegionConfig(
        covisibility_min_shared=5,
        cluster_radius=1.0,
        anchor_radius=10.0,
        weak_image_score_threshold=0.30,
    )
    result = analyze_weak_regions(
        m,
        evidence=BuildEvidence(pair_rows=pairs),
        config=cfg,
    )
    causes = set(result.regions[0]["root_causes"])
    assert WeakRegionCause.MATCHING_AMBIGUITY.value in causes
    assert result.summary["diagnostic_mode"] == "MAP_PLUS_PARTIAL_BUILD_EVIDENCE"


def test_positioned_unregistered_images_seed_a_region():
    m = mixed_map()
    manifest = [
        {
            "image_name": "failed_001.jpg",
            "registered": False,
            "x": 8.1,
            "y": 0.1,
            "z": 0.0,
            "route_id": "route_B",
        }
    ]
    cfg = WeakRegionConfig(
        covisibility_min_shared=5,
        cluster_radius=1.0,
        anchor_radius=2.0,
        weak_image_score_threshold=0.95,
    )
    result = analyze_weak_regions(
        m,
        evidence=BuildEvidence(image_rows=manifest),
        config=cfg,
    )
    assert result.summary["positioned_unregistered_images"] == 1
    assert any(
        region["metrics"]["positioned_unregistered_images"] == 1
        for region in result.regions
    )


def _pair_cfg() -> WeakRegionConfig:
    return WeakRegionConfig(
        covisibility_min_shared=5,
        cluster_radius=1.0,
        anchor_radius=10.0,
        weak_image_score_threshold=0.30,
    )


def test_planar_config_is_planar_pair_dominance():
    pairs = [
        {
            "image_i": "im_5.jpg",
            "image_j": "im_6.jpg",
            "selected": True,
            "attempted": True,
            "num_matches": 200,
            "num_inliers": 160,
            "inlier_ratio": 0.80,
            "verified": True,
            "two_view_config": 4,
        },
        {
            "image_i": "im_6.jpg",
            "image_j": "im_7.jpg",
            "selected": True,
            "attempted": True,
            "num_matches": 180,
            "num_inliers": 150,
            "inlier_ratio": 0.83,
            "verified": True,
            "two_view_config": 4,
        },
    ]
    result = analyze_weak_regions(
        mixed_map(), evidence=BuildEvidence(pair_rows=pairs), config=_pair_cfg()
    )
    causes = set(result.regions[0]["root_causes"])
    assert WeakRegionCause.PLANAR_PAIR_DOMINANCE.value in causes
    assert WeakRegionCause.MATCHING_AMBIGUITY.value not in causes


def test_homography_vs_essential_is_planar_not_ambiguity():
    pairs = [
        {
            "image_i": "im_5.jpg",
            "image_j": "im_1.jpg",
            "selected": True,
            "attempted": True,
            "num_matches": 200,
            "num_inliers": 160,
            "inlier_ratio": 0.80,
            "verified": True,
            "homography_support": 0.85,
            "essential_support": 0.10,
        },
        {
            "image_i": "im_6.jpg",
            "image_j": "im_2.jpg",
            "selected": True,
            "attempted": True,
            "num_matches": 180,
            "num_inliers": 150,
            "inlier_ratio": 0.83,
            "verified": True,
            "homography_support": 0.80,
            "essential_support": 0.12,
        },
    ]
    result = analyze_weak_regions(
        mixed_map(), evidence=BuildEvidence(pair_rows=pairs), config=_pair_cfg()
    )
    causes = set(result.regions[0]["root_causes"])
    assert WeakRegionCause.PLANAR_PAIR_DOMINANCE.value in causes
    assert WeakRegionCause.MATCHING_AMBIGUITY.value not in causes


def test_inconsistent_two_view_pose_is_flagged():
    pairs = [
        {
            "image_i": "im_5.jpg",
            "image_j": "im_6.jpg",
            "selected": True,
            "attempted": True,
            "num_matches": 120,
            "num_inliers": 90,
            "inlier_ratio": 0.75,
            "verified": True,
            "two_view_qw": 0.70710678,
            "two_view_qx": 0.0,
            "two_view_qy": 0.0,
            "two_view_qz": 0.70710678,
            "two_view_tx": -1.0,
            "two_view_ty": 0.0,
            "two_view_tz": 0.0,
        },
        {
            "image_i": "im_6.jpg",
            "image_j": "im_7.jpg",
            "selected": True,
            "attempted": True,
            "num_matches": 110,
            "num_inliers": 80,
            "inlier_ratio": 0.73,
            "verified": True,
        },
    ]
    result = analyze_weak_regions(
        mixed_map(), evidence=BuildEvidence(pair_rows=pairs), config=_pair_cfg()
    )
    causes = set(result.regions[0]["root_causes"])
    assert WeakRegionCause.RELATIVE_POSE_INCONSISTENT.value in causes
    assert result.regions[0]["pair_evidence"]["pose_inconsistent_fraction"] == 1.0


def test_identity_two_view_pose_is_not_inconsistent():
    pairs = [
        {
            "image_i": "im_5.jpg",
            "image_j": "im_6.jpg",
            "selected": True,
            "attempted": True,
            "num_matches": 120,
            "num_inliers": 90,
            "inlier_ratio": 0.75,
            "verified": True,
            "two_view_qw": 1.0,
            "two_view_qx": 0.0,
            "two_view_qy": 0.0,
            "two_view_qz": 0.0,
            "two_view_tx": -1.0,
            "two_view_ty": 0.0,
            "two_view_tz": 0.0,
        },
        {
            "image_i": "im_6.jpg",
            "image_j": "im_7.jpg",
            "selected": True,
            "attempted": True,
            "num_matches": 110,
            "num_inliers": 80,
            "inlier_ratio": 0.73,
            "verified": True,
            "two_view_qw": 1.0,
            "two_view_qx": 0.0,
            "two_view_qy": 0.0,
            "two_view_qz": 0.0,
            "two_view_tx": -1.0,
            "two_view_ty": 0.0,
            "two_view_tz": 0.0,
        },
    ]
    result = analyze_weak_regions(
        mixed_map(), evidence=BuildEvidence(pair_rows=pairs), config=_pair_cfg()
    )
    causes = set(result.regions[0]["root_causes"])
    assert WeakRegionCause.RELATIVE_POSE_INCONSISTENT.value not in causes


def test_database_without_qvec_keeps_pose_fraction_null(tmp_path):
    import sqlite3

    from sfm_diagnosis.evidence import COLMAP_MAX_IMAGE_ID, load_colmap_database

    db = tmp_path / "database.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute(
            "CREATE TABLE two_view_geometries("
            "pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB, config INTEGER)"
        )
        conn.execute("INSERT INTO images VALUES(5, 'im_5.jpg')")
        conn.execute("INSERT INTO images VALUES(6, 'im_6.jpg')")
        pair_id = 5 * COLMAP_MAX_IMAGE_ID + 6
        conn.execute(
            "INSERT INTO two_view_geometries VALUES(?, 80, 2, NULL, 2)",
            (pair_id,),
        )
        conn.commit()
    finally:
        conn.close()
    _, pairs = load_colmap_database(db)
    assert pairs[0]["two_view_config"] == 2
    assert pairs[0]["two_view_qvec"] is None
    result = analyze_weak_regions(
        mixed_map(),
        evidence=BuildEvidence(pair_rows=pairs),
        config=_pair_cfg(),
    )
    assert result.regions[0]["pair_evidence"]["pose_inconsistent_fraction"] is None
