import numpy as np

from sfm_diagnosis.diagnose import DiagnosisCode, diagnose_pose
from sfm_diagnosis.logs import LocalizationHistory
from sfm_diagnosis.models import CameraIntrinsics, MapData, Pose


def healthy_map() -> MapData:
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
        [[0, 0, 0], [2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0]], dtype=float
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


def test_healthy_geometry_is_not_flagged():
    d = diagnose_pose(healthy_map(), Pose(np.zeros(3), np.eye(3)))
    assert d.primary == DiagnosisCode.HEALTHY
    assert d.visible_points > 100
    assert d.grid_occupancy >= 6
    assert d.hull_coverage >= 0.15


def test_actual_failures_with_good_geometry_are_matching_weak():
    history = LocalizationHistory(
        [
            {
                "x": 0,
                "y": 0,
                "z": 0,
                "qx": 0,
                "qy": 0,
                "qz": 0,
                "qw": 1,
                "success": 0,
                "registration_confidence": 0.9,
                "retrieval_score": 0.8,
                "matches": 100,
                "unique_tracks": 15,
                "reference_count": 3,
                "pnp_inliers": 10,
                "inlier_ratio": 0.10,
                "reproj_p90": 4.0,
                "grid_occupancy": 4,
                "hull_coverage": 0.08,
                "positive_depth_ratio": 1.0,
                "pose_consensus_translation_m": 0.1,
                "pose_consensus_rotation_deg": 1.0,
            }
        ]
    )
    d = diagnose_pose(healthy_map(), Pose(np.zeros(3), np.eye(3)), history=history)
    assert d.primary == DiagnosisCode.MATCHING_WEAK
    assert DiagnosisCode.GEOMETRY_WEAK not in d.codes


def test_sparse_view_is_classified_as_data_sparse():
    m = healthy_map()
    # Turn the camera away from all positive-Z points.
    R_wc = np.diag([-1.0, 1.0, -1.0])
    d = diagnose_pose(m, Pose(np.zeros(3), R_wc))
    assert d.primary == DiagnosisCode.DATA_SPARSE

def test_far_but_oriented_pose_is_not_view_coverage_weak():
    from test_geometry_stats import short_history_map

    d = diagnose_pose(short_history_map(), Pose(np.zeros(3), np.eye(3)))
    assert d.primary == DiagnosisCode.OBSERVATION_SCALE_WEAK
    assert d.primary != DiagnosisCode.VIEW_COVERAGE_WEAK
    assert DiagnosisCode.GEOMETRY_WEAK not in d.codes
