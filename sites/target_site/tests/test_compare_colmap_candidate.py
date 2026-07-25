from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "compare_colmap_candidate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("compare_colmap_candidate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_reconstruction(*, changed_track: bool = False):
    images = {
        1: SimpleNamespace(image_id=1, name="a.jpg", camera_id=7, has_pose=True),
        2: SimpleNamespace(image_id=2, name="b.jpg", camera_id=7, has_pose=True),
    }
    track = [SimpleNamespace(image_id=1, point2D_idx=3)]
    track.append(SimpleNamespace(image_id=2, point2D_idx=5 if changed_track else 4))
    points = {9: SimpleNamespace(track=SimpleNamespace(elements=track))}
    return SimpleNamespace(images=images, points3D=points)


def test_topology_signature_is_stable_but_detects_track_mutation() -> None:
    module = load_module()
    baseline = module.topology_signature(fake_reconstruction())

    assert baseline == module.topology_signature(fake_reconstruction())
    assert baseline["registered_images"] == 2
    assert baseline["points3D"] == 1
    assert baseline["observations"] == 2
    assert baseline["sha256"] != module.topology_signature(
        fake_reconstruction(changed_track=True)
    )["sha256"]


def test_distribution_summary_rejects_nonfinite_geometry() -> None:
    module = load_module()

    assert module.distribution_summary([1.0, 2.0, 3.0]) == {
        "median": 2.0,
        "p95": pytest.approx(2.9),
        "max": 3.0,
    }
    with pytest.raises(ValueError, match="non-finite"):
        module.distribution_summary([1.0, np.nan])


def test_rotation_delta_is_zero_for_equal_rotations_and_90_for_quarter_turn() -> None:
    module = load_module()
    identity = np.eye(3)
    quarter_turn = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    assert module.rotation_delta_degrees(identity, identity) == pytest.approx(0.0)
    assert module.rotation_delta_degrees(identity, quarter_turn) == pytest.approx(90.0)


def test_reprojection_metrics_are_recomputed_from_geometry_not_stored_error() -> None:
    module = load_module()

    class IdentityPose:
        def __mul__(self, points):
            return np.asarray(points)

    class Camera:
        def img_from_cam(self, points):
            points = np.asarray(points)
            return points[:, :2] / points[:, 2:3]

    points2d = [
        SimpleNamespace(point3D_id=11, xy=np.array([0.0, 0.0]), has_point3D=lambda: True),
        SimpleNamespace(point3D_id=12, xy=np.array([2.0, 0.0]), has_point3D=lambda: True),
    ]
    image = SimpleNamespace(
        camera_id=7,
        has_pose=True,
        points2D=points2d,
        cam_from_world=lambda: IdentityPose(),
    )
    reconstruction = SimpleNamespace(
        images={1: image},
        cameras={7: Camera()},
        points3D={
            11: SimpleNamespace(xyz=np.array([0.0, 0.0, 1.0]), error=999.0),
            12: SimpleNamespace(xyz=np.array([1.0, 0.0, 1.0]), error=999.0),
        },
    )

    metrics = module.reprojection_metrics(reconstruction)

    assert metrics["observations"] == 2
    assert metrics["invalid_or_behind"] == 0
    assert metrics["mean_px"] == pytest.approx(0.5)
    assert metrics["max_px"] == pytest.approx(1.0)
