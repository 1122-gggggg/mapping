import numpy as np

from sfm_diagnosis.models import CameraIntrinsics, MapData, Pose
from sfm_diagnosis.view_support import ViewSupportConfig, compute_view_support


def _map() -> MapData:
    points = np.array([[0.0, 0.0, 8.0], [1.0, 0.0, 8.0], [-1.0, 0.0, 8.0]])
    centers = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    return MapData(
        point_ids=np.arange(3),
        points_xyz=points,
        point_rgb=np.zeros((3, 3), dtype=np.uint8),
        point_errors=np.full(3, 0.5),
        track_lengths=np.full(3, 3),
        track_image_ids=[np.array([10, 11, 12]) for _ in range(3)],
        image_ids=np.array([10, 11, 12]),
        image_names=["a.jpg", "b.jpg", "c.jpg"],
        image_camera_ids=np.zeros(3, dtype=int),
        image_centers=centers,
        image_R_wc=np.repeat(np.eye(3)[None], 3, axis=0),
        cameras={0: CameraIntrinsics(0, "PINHOLE", 1000, 800, 500, 500, 500, 400)},
    )


def test_repeated_local_observation_support_scores_high():
    result = compute_view_support(
        _map(),
        Pose(np.zeros(3), np.eye(3)),
        np.arange(3),
        config=ViewSupportConfig(neighbors=3, min_track_observations=3),
    )
    assert result.redetectable_points == 3
    assert result.paper_raw_score == 9.0
    assert result.weighted_visible_support_fraction == 1.0


def test_pose_neighbor_selection_is_orientation_aware():
    m = _map()
    m.image_centers[:] = 0.0
    m.image_R_wc[1] = np.diag([-1.0, 1.0, -1.0])
    result = compute_view_support(
        m,
        Pose(np.zeros(3), np.eye(3)),
        np.arange(3),
        config=ViewSupportConfig(neighbors=1, min_track_observations=1),
    )
    assert result.neighbor_image_ids == [10]


def test_large_viewpoint_extrapolation_is_rejected():
    m = _map()
    result = compute_view_support(
        m,
        Pose(np.array([8.0, 0.0, 0.0]), np.eye(3)),
        np.arange(3),
        config=ViewSupportConfig(
            neighbors=3,
            min_track_observations=3,
            max_observation_angle_deg=10.0,
            range_expansion=0.95,
        ),
    )
    assert result.redetectable_points == 0
    assert result.angle_extrapolated_fraction == 1.0
