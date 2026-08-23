import numpy as np
import pytest

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


@pytest.mark.parametrize(
    ("model", "params", "distortion"),
    [
        ("PINHOLE", (700.0, 500.0, 320.0, 240.0), (0.0, 0.0, 0.0, 0.0)),
        ("SIMPLE_RADIAL", (600.0, 320.0, 240.0, 0.08), (0.08, 0.0, 0.0, 0.0)),
        ("RADIAL", (600.0, 320.0, 240.0, 0.08, -0.02), (0.08, -0.02, 0.0, 0.0)),
        (
            "OPENCV",
            (700.0, 500.0, 320.0, 240.0, 0.08, -0.02, 0.03, -0.01),
            (0.08, -0.02, 0.03, -0.01),
        ),
    ],
)
def test_projection_uses_colmap_parameter_order(
    model: str, params: tuple[float, ...], distortion: tuple[float, ...]
) -> None:
    camera = Camera(1, model, 640, 480, np.array(params))
    pose = Pose(np.eye(3), np.zeros(3))
    points = np.array([[1.5, -0.75, 4.0], [-2.0, 1.0, 5.0]], dtype=float)

    projected, depth = project_points(points, pose, camera)

    normalized = points[:, :2] / points[:, 2, None]
    radius_squared = np.sum(normalized**2, axis=1)
    k1, k2, p1, p2 = distortion
    radial = 1.0 + k1 * radius_squared + k2 * radius_squared**2
    x, y = normalized.T
    distorted = np.column_stack(
        (
            x * radial + 2.0 * p1 * x * y + p2 * (radius_squared + 2.0 * x**2),
            y * radial + p1 * (radius_squared + 2.0 * y**2) + 2.0 * p2 * x * y,
        )
    )
    if model in {"SIMPLE_RADIAL", "RADIAL"}:
        fx = fy = params[0]
        cx, cy = params[1:3]
    else:
        fx, fy, cx, cy = params[:4]
    expected = distorted * np.array([fx, fy]) + np.array([cx, cy])
    assert np.allclose(projected, expected, atol=1e-10)
    assert np.allclose(depth, points[:, 2])


@pytest.mark.parametrize(
    ("model", "params", "required"),
    [
        ("SIMPLE_RADIAL", (600.0, 320.0, 240.0), 4),
        ("RADIAL", (600.0, 320.0, 240.0, 0.08), 5),
        ("OPENCV", (700.0, 500.0, 320.0, 240.0, 0.08, -0.02, 0.03), 8),
    ],
)
def test_projection_rejects_incomplete_colmap_params(
    model: str, params: tuple[float, ...], required: int
) -> None:
    camera = Camera(1, model, 640, 480, np.array(params))

    with pytest.raises(ValueError, match=rf"needs {required} params"):
        project_points(np.array([[0.5, -0.25, 2.0]]), Pose.identity(), camera)


def test_fisheye_projection_requires_dedicated_model() -> None:
    camera = Camera(
        1,
        "OPENCV_FISHEYE",
        640,
        480,
        np.array([500.0, 500.0, 320.0, 240.0, 0.01, -0.02, 0.003, -0.001]),
    )

    with pytest.raises(ValueError, match="dedicated model"):
        project_points(np.array([[0.5, -0.25, 2.0]]), Pose.identity(), camera)


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
