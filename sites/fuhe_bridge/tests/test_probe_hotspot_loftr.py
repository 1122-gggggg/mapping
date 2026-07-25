from __future__ import annotations

import numpy as np

from tools.probe_hotspot_loftr import (
    camera_matrix,
    fundamental_from_poses,
    sampson_errors_px,
    select_nearest_pairs,
    triangulate_fixed_pose,
)


def test_select_nearest_pairs_is_deterministic_and_bounded() -> None:
    centers = {
        "P114/000029.jpg": np.asarray([0.0, 0.0, 0.0]),
        "P114/000030.jpg": np.asarray([1.0, 0.0, 0.0]),
        "P112/000001.jpg": np.asarray([0.1, 0.0, 0.0]),
        "P112/000002.jpg": np.asarray([0.2, 0.0, 0.0]),
        "P112/000003.jpg": np.asarray([1.1, 0.0, 0.0]),
    }

    pairs = select_nearest_pairs(
        centers,
        source_sequence="P114",
        target_sequences=["P112"],
        source_frame_min=29,
        source_frame_max=30,
        pairs_per_source=1,
    )

    assert pairs == [
        ("P114/000029.jpg", "P112/000001.jpg", 0.1),
        ("P114/000030.jpg", "P112/000003.jpg", 0.1),
    ]


def test_fixed_pose_epipolar_and_triangulation_accept_consistent_match() -> None:
    k = camera_matrix(800.0, 800.0, 320.0, 240.0)
    r0 = np.eye(3)
    t0 = np.zeros(3)
    r1 = np.eye(3)
    t1 = np.asarray([-1.0, 0.0, 0.0])
    xyz = np.asarray([[0.2, 0.1, 5.0]])

    def project(r: np.ndarray, t: np.ndarray) -> np.ndarray:
        cam = (r @ xyz.T + t[:, None]).T
        uvw = (k @ cam.T).T
        return uvw[:, :2] / uvw[:, 2:]

    xy0 = project(r0, t0)
    xy1 = project(r1, t1)
    fundamental = fundamental_from_poses(k, r0, t0, k, r1, t1)

    assert sampson_errors_px(fundamental, xy0, xy1)[0] < 1e-8
    result = triangulate_fixed_pose(k, r0, t0, k, r1, t1, xy0[0], xy1[0])
    assert result is not None
    assert result["maximum_reprojection_error_px"] < 1e-8
    assert result["triangulation_angle_deg"] > 1.5


def test_sampson_error_rejects_off_epipolar_match() -> None:
    k = camera_matrix(800.0, 800.0, 320.0, 240.0)
    fundamental = fundamental_from_poses(
        k,
        np.eye(3),
        np.zeros(3),
        k,
        np.eye(3),
        np.asarray([-1.0, 0.0, 0.0]),
    )

    error = sampson_errors_px(
        fundamental,
        np.asarray([[320.0, 240.0]]),
        np.asarray([[300.0, 270.0]]),
    )[0]

    assert error > 20.0
