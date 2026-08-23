from pathlib import Path

from mapdoctor.io.colmap import load_colmap_compatible
from mapdoctor.metrics import analyze
from mapdoctor.model import Camera, ImageRecord, MapModel, Observation2D
from mapdoctor.scoring import score


FIXTURE = Path(__file__).parent / "fixtures" / "colmap_text"


def test_metrics_are_computed():
    model = load_colmap_compatible(FIXTURE, source="glomap")
    metrics = analyze(model)
    assert metrics.registered_images == 3
    assert metrics.points3d == 5
    assert metrics.track_length_median == 3.0
    assert metrics.reprojection_error_p90_px is not None
    assert metrics.largest_covisibility_component_ratio == 1.0
    assert metrics.nearest_camera_baseline_median == 1.0


def test_score_shape():
    metrics = analyze(load_colmap_compatible(FIXTURE, source="colmap"))
    result = score(metrics)
    assert 0 <= result.score <= 100
    assert result.grade in {"A", "B", "C", "D"}
    assert "reprojection_error_p90_px" in result.checks


def test_missing_camera_is_not_dropped():
    camera = Camera(id=1, model="PINHOLE", width=100, height=100, params=(50.0, 50.0, 50.0, 50.0))
    observations = tuple(
        Observation2D(x=float(i * 5), y=float(i * 5), point3d_id=i) for i in range(20)
    )
    valid = ImageRecord(
        id=1,
        camera_id=1,
        name="good.jpg",
        center=(0.0, 0.0, 0.0),
        viewing_direction=(0.0, 0.0, 1.0),
        observations=observations,
    )
    orphan = ImageRecord(
        id=2,
        camera_id=99,
        name="orphan.jpg",
        center=(1.0, 0.0, 0.0),
        viewing_direction=(0.0, 0.0, 1.0),
        observations=observations,
    )
    metrics = analyze(
        MapModel(
            source="test",
            format="colmap",
            cameras={1: camera},
            images={1: valid, 2: orphan},
        )
    )
    weak = next(item for item in metrics.weak_images if item["name"] == "orphan.jpg")
    assert "missing_camera" in weak["reasons"]
    assert weak["observations"] == 0
    assert any(item["reference_image"] == "orphan.jpg" for item in metrics.recapture_suggestions)
    assert metrics.observations_per_image_p10 != 20.0
    assert metrics.observations_per_image_p10 is not None
    assert metrics.observations_per_image_p10 < 20.0
