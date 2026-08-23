import numpy as np

from sfm_diagnosis.diagnose import DiagnosisCode, diagnose_pose
from sfm_diagnosis.logs import LocalizationHistory
from sfm_diagnosis.models import CameraIntrinsics, MapData, Pose


def _healthy_map() -> MapData:
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
        [[0, 0, 0], [2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0]],
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


def test_reference_disagreement_with_good_support_flags_aliasing_suspect():
    history = LocalizationHistory(
        [
            {
                "x": 0,
                "y": 0,
                "z": 0,
                "success": 1,
                "registration_confidence": 0.95,
                "retrieval_score": 0.9,
                "unique_tracks": 80,
                "reference_count": 8,
                "reference_dispersion_m": 1.2,
                "reference_consensus_sigma_m": 0.08,
                "reference_rotation_dispersion_deg": 2.0,
                "reference_covariance_eligible_ratio": 1.0,
            }
        ]
    )
    d = diagnose_pose(_healthy_map(), Pose(np.zeros(3), np.eye(3)), history=history)
    assert d.primary == DiagnosisCode.PERCEPTUAL_ALIASING_SUSPECTED
    assert DiagnosisCode.REFERENCE_DISAGREEMENT in d.codes
    assert DiagnosisCode.GEOMETRY_WEAK not in d.codes


def test_agreeing_but_uncertain_references_are_observability_weak():
    history = LocalizationHistory(
        [
            {
                "x": 0,
                "y": 0,
                "z": 0,
                "success": 1,
                "registration_confidence": 0.95,
                "retrieval_score": 0.9,
                "unique_tracks": 80,
                "reference_count": 8,
                "reference_dispersion_m": 0.05,
                "reference_consensus_sigma_m": 1.2,
                "reference_rotation_dispersion_deg": 1.0,
                "reference_covariance_eligible_ratio": 1.0,
            }
        ]
    )
    d = diagnose_pose(_healthy_map(), Pose(np.zeros(3), np.eye(3)), history=history)
    assert d.primary == DiagnosisCode.REFERENCE_OBSERVABILITY_WEAK
    assert DiagnosisCode.REFERENCE_DISAGREEMENT not in d.codes
