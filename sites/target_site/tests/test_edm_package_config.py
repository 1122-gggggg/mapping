from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SFMSYSTEM = Path(__file__).resolve().parents[3]
PACKAGE_SCRIPT = SFMSYSTEM / "EDM定位測試/build/make_transfer_package.py"


def load_package_module():
    spec = importlib.util.spec_from_file_location("make_transfer_package", PACKAGE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_site_query_camera_is_scaled_fixed_pinhole() -> None:
    site = load_package_module().SITES["target_site"]
    camera = site["pnp_camera"]
    expected_focal = 1955.532134535766 * 1280 / 2688

    assert camera["model"] == "PINHOLE"
    assert camera["width"] == 1280
    assert camera["height"] == 720
    assert np.allclose(
        camera["params"], [expected_focal, expected_focal, 640.0, 360.0], atol=1e-12
    )


def test_target_site_package_points_to_v1_outputs() -> None:
    site = load_package_module().SITES["target_site"]

    assert site["model"].name == "final_model"
    assert site["bundle"].name == "target_site_v1_reloc_map_edm.pt"


def test_export_ref_poses_excludes_deregistered_images(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_package_module()

    class Pose:
        rotation = SimpleNamespace(matrix=lambda: np.eye(3))
        translation = np.zeros(3)

    active = SimpleNamespace(
        name="S01/active.jpg",
        camera_id=1,
        has_pose=True,
        cam_from_world=lambda: Pose(),
    )
    inactive = SimpleNamespace(
        name="S01/inactive.jpg",
        camera_id=1,
        has_pose=False,
        cam_from_world=lambda: (_ for _ in ()).throw(AssertionError("must not read pose")),
    )
    camera = SimpleNamespace(
        model=SimpleNamespace(name="PINHOLE"),
        width=1280,
        height=720,
        params=np.asarray([931.0, 931.0, 640.0, 360.0]),
    )
    reconstruction = SimpleNamespace(images={1: active, 2: inactive}, cameras={1: camera})
    monkeypatch.setattr(
        module.pycolmap, "Reconstruction", lambda _model: reconstruction
    )
    output = tmp_path / "poses.json"

    module.export_ref_poses(tmp_path / "model", output)

    payload = json.loads(output.read_text())
    assert set(payload["poses"]) == {"S01/active.jpg"}
