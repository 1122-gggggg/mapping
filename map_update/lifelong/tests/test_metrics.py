import numpy as np

from update_map.metrics import compute_fim, convex_hull_ratio, grid_occupancy
from update_map.models import Camera, Pose


def test_spatial_metrics_cover_image() -> None:
    points = np.array([[0, 0], [99, 0], [99, 99], [0, 99], [50, 50]], dtype=float)
    assert convex_hull_ratio(points, 100, 100) > 0.95
    assert grid_occupancy(points, 100, 100) >= 5


def test_fim_is_finite_for_well_distributed_points() -> None:
    camera = Camera(1, "PINHOLE", 640, 480, np.array([500.0, 500.0, 320.0, 240.0]))
    points = np.array(
        [[x, y, z] for z in (5.0, 7.0) for y in (-1.0, 0.0, 1.0) for x in (-1.5, 0.0, 1.5)]
    )
    metrics = compute_fim(points, Pose.identity(), camera, characteristic_length=5.0)
    assert np.all(metrics.eigenvalues > 0)
    assert np.isfinite(metrics.condition_number)
    assert np.all(np.isfinite(metrics.marginal_std))
