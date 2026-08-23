import numpy as np

from update_map.config import PoseConfig
from update_map.geometry import pose_distance, project_points
from update_map.models import Camera, LiftedCorrespondence, Pose
from update_map.pose import localize_with_reference_groups
from update_map.states import RegistrationStatus


def make_correspondences(reference_id: str, pose: Pose, camera: Camera):
    points = np.array(
        [[x, y, z] for z in (7.0, 9.0) for y in np.linspace(-1.5, 1.5, 5) for x in np.linspace(-2.5, 2.5, 8)],
        dtype=float,
    )
    xy, depth = project_points(points, pose, camera)
    valid = (
        (depth > 0)
        & (xy[:, 0] > 5)
        & (xy[:, 0] < camera.width - 5)
        & (xy[:, 1] > 5)
        & (xy[:, 1] < camera.height - 5)
    )
    return [
        LiftedCorrespondence(
            query_xy=point_xy,
            reference_xy=point_xy,
            point3d_id=index + 1,
            xyz_w=point_xyz,
            confidence=0.95,
            reference_id=reference_id,
        )
        for index, (point_xyz, point_xy) in enumerate(zip(points[valid], xy[valid], strict=True))
    ]


def relaxed_config() -> PoseConfig:
    config = PoseConfig(characteristic_length=5.0)
    config.gate.min_unique_point3d = 10
    config.gate.min_independent_reference_support = 2
    config.gate.max_fim_condition_number = 1e9
    config.gate.max_translation_std = 10.0
    config.gate.max_rotation_std_deg = 30.0
    return config


def test_pose_recovery_from_two_references() -> None:
    camera = Camera(1, "PINHOLE", 1280, 720, np.array([850.0, 850.0, 640.0, 360.0]))
    ground_truth = Pose(np.eye(3), np.array([-0.2, 0.1, 0.0]))
    groups = {
        "ref_a": make_correspondences("ref_a", ground_truth, camera),
        "ref_b": make_correspondences("ref_b", ground_truth, camera),
    }
    result = localize_with_reference_groups("q", groups, camera, relaxed_config())
    assert result.status == RegistrationStatus.DIRECT_STRONG
    assert result.pose is not None
    rotation, translation = pose_distance(result.pose, ground_truth)
    assert rotation < 0.1
    assert translation < 0.01


def test_competitive_pose_modes_fail_closed() -> None:
    camera = Camera(1, "PINHOLE", 1280, 720, np.array([850.0, 850.0, 640.0, 360.0]))
    pose_a = Pose(np.eye(3), np.array([0.0, 0.0, 0.0]))
    pose_b = Pose(np.eye(3), np.array([-1.0, 0.0, 0.0]))
    groups = {
        "ref_a": make_correspondences("ref_a", pose_a, camera),
        "ref_b": make_correspondences("ref_b", pose_b, camera),
    }
    config = relaxed_config()
    config.cluster_translation = 0.2
    config.dominant_cluster_ratio = 2.0
    result = localize_with_reference_groups("q", groups, camera, config)
    assert result.status == RegistrationStatus.AMBIGUOUS_MULTIMODAL
    assert result.quality.pose_mode_count == 2
