from pathlib import Path

from mapdoctor.io.colmap import _is_valid_point3d_id, load_colmap_compatible


FIXTURE = Path(__file__).parent / "fixtures" / "colmap_text"


def test_load_text_model():
    model = load_colmap_compatible(FIXTURE, source="gluemap")
    assert model.source == "gluemap"
    assert model.format == "colmap-text"
    assert len(model.cameras) == 1
    assert len(model.images) == 3
    assert len(model.points3d) == 5
    assert model.images[1].center == (0.0, 0.0, 0.0)
    assert model.images[2].center == (1.0, 0.0, 0.0)
    assert model.images[1].viewing_direction == (0.0, 0.0, 1.0)


def test_negative_point3d_id_is_invalid_without_pycolmap_helpers():
    assert _is_valid_point3d_id(-1, -1, None) is False
    assert _is_valid_point3d_id(3, -1, None) is True
    assert _is_valid_point3d_id(3, 3, None) is False
    assert _is_valid_point3d_id(-1, -1, lambda: True) is False
    assert _is_valid_point3d_id(3, -1, lambda: False) is False
