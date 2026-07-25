from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from export_scrstudio_dataset import (  # noqa: E402
    camera_intrinsics,
    camera_to_world_matrix,
    select_reference_images,
)


class _Camera:
    model_name = "PINHOLE"
    params = np.array([900.0, 910.0, 640.0, 360.0])


class _InversePose:
    def matrix(self) -> np.ndarray:
        return np.hstack([np.eye(3), np.array([[1.0], [2.0], [3.0]])])


class _Pose:
    def inverse(self) -> _InversePose:
        return _InversePose()


class _Image:
    def __init__(self, name: str, has_pose: bool = True) -> None:
        self.name = name
        self.has_pose = has_pose
        self.cam_from_world = _Pose()


class _MethodImage:
    name = "S01/method.jpg"
    has_pose = True

    def cam_from_world(self) -> _Pose:
        return _Pose()


def test_scrstudio_export_uses_full_pinhole_intrinsics_and_camera_to_world() -> None:
    intrinsics = camera_intrinsics(_Camera())
    pose = camera_to_world_matrix(_Image("S01/a.jpg"))

    np.testing.assert_allclose(
        intrinsics,
        [[900.0, 0.0, 640.0], [0.0, 910.0, 360.0], [0.0, 0.0, 1.0]],
    )
    np.testing.assert_allclose(pose[:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(pose[3], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        camera_to_world_matrix(_MethodImage()),
        pose,
    )


def test_scrstudio_export_rejects_non_pinhole_cameras() -> None:
    camera = _Camera()
    camera.model_name = "OPENCV"

    with pytest.raises(ValueError, match="PINHOLE"):
        camera_intrinsics(camera)


def test_reference_selection_is_sorted_exact_and_pose_checked() -> None:
    images = {
        1: _Image("S02/b.jpg"),
        2: _Image("S01/a.jpg"),
        3: _Image("S03/c.jpg", has_pose=False),
    }

    selected = select_reference_images(
        images,
        required_names={"S02/b.jpg", "S01/a.jpg"},
    )

    assert [image.name for image in selected] == ["S01/a.jpg", "S02/b.jpg"]
    with pytest.raises(ValueError, match="missing"):
        select_reference_images(images, required_names={"missing.jpg"})
