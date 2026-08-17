from pathlib import Path

import numpy as np

from update_map.io.colmap import load_colmap_reconstruction
from update_map.states import GeometryProvenance


def test_load_colmap_text_and_provenance(tmp_path: Path) -> None:
    (tmp_path / "cameras.txt").write_text("1 PINHOLE 640 480 500 500 320 240\n")
    (tmp_path / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 image.jpg\n"
        "320 240 1 100 100 2\n"
    )
    (tmp_path / "points3D.txt").write_text(
        "1 0 0 5 255 0 0 0.1 1 0\n"
        "2 1 0 5 0 255 0 0.2 1 1\n"
    )
    (tmp_path / "virtual_point_ids.txt").write_text("2\n")
    model = load_colmap_reconstruction(tmp_path)
    assert model.source_format == "text"
    assert len(model.cameras) == 1
    assert len(model.images) == 1
    assert np.allclose(model.images[1].pose.R_cw, np.eye(3))
    assert model.points3d[2].provenance == GeometryProvenance.VIRTUAL_BA_ONLY
    assert model.real_point_ids() == {1}
