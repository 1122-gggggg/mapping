import numpy as np

from sfm_diagnosis.diagnose import DiagnosisCode, diagnose_pose
from sfm_diagnosis.geometry_stats import compute_query_geometry_stats
from sfm_diagnosis.models import CameraIntrinsics, MapData, Pose
from test_diagnose import healthy_map

def short_history_map() -> MapData:
    points = np.array(
        [
            [x, y, z]
            for x in np.linspace(-3, 3, 9)
            for y in np.linspace(-2, 2, 7)
            for z in (6.0, 9.0)
        ],
        dtype=float,
    )
    n = len(points)
    centers = np.array(
        [
            [0.0, 0.0, 4.0],
            [2.0, 0.0, 4.0],
            [-2.0, 0.0, 4.0],
            [0.0, 2.0, 4.0],
            [0.0, -2.0, 4.0],
        ],
        dtype=float,
    )
    m = len(centers)
    return MapData(
        point_ids=np.arange(n),
        points_xyz=points,
        point_rgb=np.zeros((n, 3), np.uint8),
        point_errors=np.full(n, 0.5),
        track_lengths=np.full(n, m),
        track_image_ids=[np.arange(m) for _ in range(n)],
        image_ids=np.arange(m),
        image_names=[f"im_{i}.jpg" for i in range(m)],
        image_camera_ids=np.zeros(m, dtype=int),
        image_centers=centers,
        image_R_wc=np.repeat(np.eye(3)[None], m, axis=0),
        cameras={0: CameraIntrinsics(0, "PINHOLE", 1000, 800, 500, 500, 500, 400)},
    )


def test_same_mapping_view_is_not_scale_weak():
    m = healthy_map()
    pose = Pose(np.zeros(3), np.eye(3))
    d = diagnose_pose(m, pose)
    stats = d.query_geometry
    assert stats.scale_ratio_p50 is not None
    assert abs(stats.scale_ratio_p50 - 1.0) < 0.25
    assert stats.scale_extrapolated_fraction == 0.0
    assert DiagnosisCode.OBSERVATION_SCALE_WEAK not in d.codes
    assert d.primary == DiagnosisCode.HEALTHY


def test_far_standoff_is_observation_scale_weak_not_view_coverage():
    m = short_history_map()
    pose = Pose(np.zeros(3), np.eye(3))
    d = diagnose_pose(m, pose)
    assert d.visible_points >= 40
    assert d.query_geometry.scale_extrapolated_fraction >= 0.50
    assert d.primary == DiagnosisCode.OBSERVATION_SCALE_WEAK
    assert DiagnosisCode.GEOMETRY_WEAK not in d.codes
    assert DiagnosisCode.DATA_SPARSE not in d.codes
    assert DiagnosisCode.VIEW_COVERAGE_WEAK not in {d.primary}


def test_coincident_mapping_cameras_are_query_parallax_weak():
    m = healthy_map()
    m.image_centers[:] = 0.0
    m.image_centers += np.array(
        [[1e-4 * i, 0.0, 0.0] for i in range(m.num_images)], dtype=float
    )
    d = diagnose_pose(m, Pose(np.zeros(3), np.eye(3)))
    assert d.fim.rank == 6
    assert d.query_geometry.triangulation_angle_p50_deg is not None
    assert d.query_geometry.triangulation_angle_p50_deg < 1.0
    assert DiagnosisCode.QUERY_PARALLAX_WEAK in d.codes
    assert d.primary == DiagnosisCode.QUERY_PARALLAX_WEAK
    assert DiagnosisCode.GEOMETRY_WEAK not in d.codes
    assert d.primary != DiagnosisCode.VIEW_COVERAGE_WEAK


def test_empty_visible_set_stays_data_sparse():
    m = healthy_map()
    R_wc = np.diag([-1.0, 1.0, -1.0])
    d = diagnose_pose(m, Pose(np.zeros(3), R_wc))
    assert d.primary == DiagnosisCode.DATA_SPARSE
    assert DiagnosisCode.OBSERVATION_SCALE_WEAK not in d.codes
    assert DiagnosisCode.QUERY_PARALLAX_WEAK not in d.codes
    empty = compute_query_geometry_stats(m, Pose(np.zeros(3), R_wc), np.array([], dtype=int))
    assert empty.scale_extrapolated_fraction == 0.0
    assert empty.triangulation_angle_p50_deg is None
