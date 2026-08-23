from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from mapdoctor.adapters import get_adapter
from mapdoctor.model import Camera, MapModel

from sfm_qa.bridge import map_model_to_map_data


def _generate_demo(tmp_path: Path):
    path = Path(__file__).resolve().parents[2] / "examples" / "reproducible_demo" / "generate_demo.py"
    spec = importlib.util.spec_from_file_location("generate_demo", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.generate(tmp_path)


def test_map_model_to_map_data_preserves_demo_geometry(tmp_path):
    paths = _generate_demo(tmp_path)
    model = get_adapter("gluemap").load(paths["model"])
    map_data = map_model_to_map_data(model)

    assert map_data.num_images == 8
    assert map_data.num_points == 80
    assert len(model.images) == 8
    assert len(model.points3d) == 80

    camera = next(iter(map_data.cameras.values()))
    assert camera.fx == 500
    assert camera.fy == 500
    assert camera.cx == 320
    assert camera.cy == 240

    for image, R_wc in zip(
        [model.images[i] for i in sorted(model.images)],
        map_data.image_R_wc,
    ):
        assert R_wc.shape == (3, 3)
        assert np.allclose(R_wc.T @ R_wc, np.eye(3), atol=1e-8)
        assert np.isclose(np.linalg.det(R_wc), 1.0, atol=1e-8)
        assert np.allclose(R_wc[:, 2], image.viewing_direction, atol=1e-8)


def test_map_model_to_map_data_parses_colmap_radial_intrinsics() -> None:
    model = MapModel(
        source="synthetic",
        format="memory",
        cameras={
            7: Camera(
                id=7,
                model="RADIAL",
                width=640,
                height=480,
                params=(500.0, 320.0, 240.0, 0.08, -0.02),
            )
        },
    )

    camera = map_model_to_map_data(model).cameras[7]

    assert camera.fx == 500.0
    assert camera.fy == 500.0
    assert camera.cx == 320.0
    assert camera.cy == 240.0


@pytest.mark.parametrize(
    ("model", "params", "required"),
    [
        ("SIMPLE_RADIAL", (500.0, 320.0, 240.0), 4),
        ("RADIAL", (500.0, 320.0, 240.0, 0.08), 5),
        ("OPENCV", (500.0, 500.0, 320.0, 240.0, 0.08, -0.02, 0.01), 8),
    ],
)
def test_map_model_to_map_data_rejects_incomplete_camera_params(
    model: str, params: tuple[float, ...], required: int
) -> None:
    map_model = MapModel(
        source="synthetic",
        format="memory",
        cameras={1: Camera(id=1, model=model, width=640, height=480, params=params)},
    )

    with pytest.raises(ValueError, match=rf"needs {required} params"):
        map_model_to_map_data(map_model)
