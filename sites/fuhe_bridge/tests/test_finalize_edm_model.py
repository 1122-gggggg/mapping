from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from finalize_edm_model import (  # noqa: E402
    FIXED_CAMERA_KEY,
    FIXED_CAMERA_PARAMS,
    FINAL_GATE_THRESHOLDS,
    final_gate_checks,
    fixed_camera_evidence,
    eligible_registered_ids,
    largest_image_component_ids,
    largest_image_component_fraction,
    point_ids_spanning_sequence_edges,
    reprojection_error_px,
    restore_seed_intrinsics,
    triangulation_angle_deg,
)


def test_triangulation_angle_uses_widest_observation_baseline() -> None:
    xyz = np.asarray([0.0, 1.0, 0.0])
    centers = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    assert np.isclose(triangulation_angle_deg(xyz, centers), 90.0)


def test_triangulation_angle_is_zero_with_fewer_than_two_valid_rays() -> None:
    xyz = np.asarray([0.0, 0.0, 0.0])
    centers = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    assert triangulation_angle_deg(xyz, centers) == 0.0


def test_largest_component_fraction_includes_isolated_registered_images() -> None:
    registered = {1, 2, 3, 4, 5}
    tracks = [[1, 2, 3], [3, 4]]

    fraction, largest = largest_image_component_fraction(registered, tracks)

    assert largest == 4
    assert fraction == 0.8


def test_largest_component_fraction_for_empty_model_is_zero() -> None:
    assert largest_image_component_fraction(set(), []) == (0.0, 0)


def test_connectivity_excludes_intentional_zero_observation_images() -> None:
    images = [
        SimpleNamespace(image_id=1, name="S01/frame.jpg"),
        SimpleNamespace(image_id=2, name="S01/pure_rotation.jpg"),
    ]

    assert eligible_registered_ids(images, {"S01/pure_rotation.jpg"}) == {1}


def test_connectivity_excludes_deregistered_images() -> None:
    images = [
        SimpleNamespace(image_id=1, name="S01/active.jpg", has_pose=True),
        SimpleNamespace(image_id=2, name="S01/deregistered.jpg", has_pose=False),
    ]

    assert eligible_registered_ids(images, set()) == {1}


def test_largest_component_ids_returns_members_not_only_size() -> None:
    registered = {1, 2, 3, 4, 5, 6}
    tracks = [[1, 2, 3], [3, 4], [5, 6]]

    assert largest_image_component_ids(registered, tracks) == {1, 2, 3, 4}


def test_quarantine_selects_every_track_spanning_suspect_edge() -> None:
    def point(point_id: int, image_ids: list[int]):
        return (
            point_id,
            SimpleNamespace(
                track=SimpleNamespace(
                    elements=[SimpleNamespace(image_id=image_id) for image_id in image_ids]
                )
            ),
        )

    points = dict(
        [point(10, [1, 2]), point(11, [1, 2, 3]), point(12, [1, 3])]
    )
    sequences = {1: "S03", 2: "S06", 3: "S05"}

    assert point_ids_spanning_sequence_edges(
        points, sequences, {("S03", "S06")}
    ) == {10, 11}


def test_reprojection_error_rejects_nonpositive_depth() -> None:
    class Pose:
        def __mul__(self, xyz):
            return np.asarray([xyz[0], xyz[1], -1.0])

    image = SimpleNamespace(
        cam_from_world=lambda: Pose(),
        project_point=lambda xyz: np.asarray([2.0, 3.0]),
    )
    point = SimpleNamespace(xyz=np.asarray([1.0, 2.0, 3.0]))
    point2d = SimpleNamespace(xy=np.asarray([2.0, 3.0]))

    assert reprojection_error_px(image, point2d, point) is None


def test_reprojection_error_is_finite_pixel_distance() -> None:
    class Pose:
        def __mul__(self, xyz):
            return np.asarray([xyz[0], xyz[1], 1.0])

    image = SimpleNamespace(
        cam_from_world=lambda: Pose(),
        project_point=lambda xyz: np.asarray([5.0, 7.0]),
    )
    point = SimpleNamespace(xyz=np.asarray([1.0, 2.0, 3.0]))
    point2d = SimpleNamespace(xy=np.asarray([2.0, 3.0]))

    assert reprojection_error_px(image, point2d, point) == 5.0


def test_restore_seed_intrinsics_is_exact() -> None:
    camera = SimpleNamespace(
        model=SimpleNamespace(name="PINHOLE"),
        width=1280,
        height=720,
        params=np.asarray([930.0, 930.0, 640.0, 360.0]),
    )
    rec = SimpleNamespace(cameras={1: camera})
    seed = {("PINHOLE", 1280, 720): np.asarray([931.0, 931.0, 640.0, 360.0])}

    restore_seed_intrinsics(rec, seed)

    assert np.array_equal(camera.params, seed[("PINHOLE", 1280, 720)])


def test_fuhe_finalizer_accepts_only_the_single_fixed_working_camera() -> None:
    camera = SimpleNamespace(
        model=SimpleNamespace(name="PINHOLE"),
        width=1920,
        height=1080,
        params=FIXED_CAMERA_PARAMS.copy(),
    )

    evidence = fixed_camera_evidence(SimpleNamespace(cameras={7: camera}))

    assert FIXED_CAMERA_KEY == ("PINHOLE", 1920, 1080)
    assert evidence["ok"] is True
    assert evidence["camera_count"] == 1
    assert evidence["maximum_intrinsics_delta"] == 0.0

    camera.params[0] += 1e-5
    assert fixed_camera_evidence(SimpleNamespace(cameras={7: camera}))["ok"] is False
    assert fixed_camera_evidence(
        SimpleNamespace(cameras={7: camera, 8: camera})
    )["ok"] is False


def test_fuhe_final_gate_threshold_manifest_is_exact() -> None:
    assert FINAL_GATE_THRESHOLDS == {
        "registered_fraction_minimum": 0.95,
        "per_sequence_registered_fraction_minimum": 0.90,
        "points_per_registered_image_minimum": 200.0,
        "mean_reprojection_error_px_maximum": 2.0,
        "p95_reprojection_error_px_maximum": 4.0,
        "p99_reprojection_error_px_maximum": 8.0,
        "invalid_reprojection_observations_maximum": 0,
        "maximum_intrinsics_delta": 1e-6,
        "median_triangulation_angle_deg_minimum": 5.0,
        "fraction_triangulation_angle_below_1deg_maximum": 0.02,
        "active_component_fraction_minimum": 1.0,
    }


def test_final_gate_checks_enforce_every_fuhe_geometry_threshold() -> None:
    metrics = {
        "registered_fraction": 0.95,
        "per_sequence_registered_fraction": {
            "P1090109_002": 0.90,
            "P1100110_005": 0.90,
            "P1110111": 0.90,
            "P1120112": 0.90,
            "P1140114": 0.90,
        },
        "points_per_registered_image": 200.0,
        "mean_reprojection_error_px": 2.0,
        "p95_reprojection_error_px": 4.0,
        "p99_reprojection_error_px": 8.0,
        "invalid_reprojection_observations": 0,
        "largest_component_fraction": 1.0,
        "zero_observation_registered": 0,
        "median_triangulation_angle_deg": 5.0,
        "fraction_triangulation_angle_below_1deg": 0.02,
        "forbidden_observations": 0,
        "short_track_points": 0,
    }

    checks = final_gate_checks(
        metrics,
        fixed_camera_ok=True,
        maximum_intrinsics_delta=1e-6,
        remaining_quarantined_points=0,
        pure_rotation_complete=True,
    )

    assert all(checks.values())

    for field, bad_value in (
        ("registered_fraction", 0.949),
        ("points_per_registered_image", 199.9),
        ("p95_reprojection_error_px", 4.01),
        ("p99_reprojection_error_px", 8.01),
        ("median_triangulation_angle_deg", 4.99),
        ("fraction_triangulation_angle_below_1deg", 0.021),
    ):
        failing = dict(metrics)
        failing[field] = bad_value
        assert not all(
            final_gate_checks(
                failing,
                fixed_camera_ok=True,
                maximum_intrinsics_delta=0.0,
                remaining_quarantined_points=0,
                pure_rotation_complete=True,
            ).values()
        )


def test_final_gate_checks_reject_one_underregistered_sequence() -> None:
    metrics = {
        "registered_fraction": 0.99,
        "per_sequence_registered_fraction": {
            "P1090109_002": 0.89,
            "P1100110_005": 1.0,
            "P1110111": 1.0,
            "P1120112": 1.0,
            "P1140114": 1.0,
        },
        "points_per_registered_image": 300.0,
        "mean_reprojection_error_px": 1.0,
        "p95_reprojection_error_px": 2.0,
        "p99_reprojection_error_px": 3.0,
        "invalid_reprojection_observations": 0,
        "largest_component_fraction": 1.0,
        "zero_observation_registered": 0,
        "median_triangulation_angle_deg": 8.0,
        "fraction_triangulation_angle_below_1deg": 0.0,
        "forbidden_observations": 0,
        "short_track_points": 0,
    }

    checks = final_gate_checks(
        metrics,
        fixed_camera_ok=True,
        maximum_intrinsics_delta=0.0,
        remaining_quarantined_points=0,
        pure_rotation_complete=True,
    )

    assert checks["G5.1"] is False
