from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SFMSYSTEM = Path(__file__).resolve().parents[3]
PACKAGE_SCRIPT = SFMSYSTEM / "EDM定位測試/build/make_transfer_package.py"
TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from ts_common import RUN_ID  # noqa: E402
from verify_final_release import release_artifact_paths  # noqa: E402


def load_package_module():
    spec = importlib.util.spec_from_file_location("make_transfer_package", PACKAGE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_football_package_points_to_run_id_outputs(tmp_path: Path) -> None:
    paths = release_artifact_paths(tmp_path / "run", tmp_path / "package")

    assert RUN_ID == "football_field_v1"
    assert paths["edm_bundle"].name == f"{RUN_ID}_reloc_map_edm.pt"
    assert paths["package_bundle"].name == f"{RUN_ID}_reloc_map_edm.pt"
    assert "target_site" not in paths["edm_bundle"].name
    assert "target_site" not in paths["package_bundle"].name


@pytest.mark.skipif(not PACKAGE_SCRIPT.is_file(), reason="shared package script is absent")
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
