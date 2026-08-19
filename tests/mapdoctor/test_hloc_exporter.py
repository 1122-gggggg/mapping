from types import SimpleNamespace

import pytest

from mapdoctor.integrations.hloc import _coverage, _query_result, export_hloc_logs, write_hloc_results
from mapdoctor.metrics import convex_hull_area_fraction, grid_coverage


def test_hloc_exporter_requires_explicit_pickle_trust(tmp_path):
    logs = tmp_path / "logs.pkl"
    logs.write_bytes(b"not loaded because trust is false")
    with pytest.raises(ValueError, match="explicit trust"):
        export_hloc_logs(logs, tmp_path / "model", trust_pickle=False)


def test_hloc_geometry_export(tmp_path):
    np = pytest.importorskip("numpy")
    pycolmap = pytest.importorskip("pycolmap")

    camera = pycolmap.Camera(
        model="PINHOLE",
        width=640,
        height=480,
        params=[500.0, 500.0, 320.0, 240.0],
    )
    pose = pycolmap.Rigid3d()
    world = np.array(
        [
            [-1.0, -1.0, 5.0],
            [1.0, -1.0, 5.0],
            [-1.0, 1.0, 5.0],
            [1.0, 1.0, 5.0],
            [0.0, -1.5, 6.0],
            [0.0, 1.5, 6.0],
        ],
        dtype=float,
    )
    keypoints = np.asarray(camera.img_from_cam(world), dtype=float)
    ids = list(range(1, len(world) + 1))
    reconstruction = SimpleNamespace(
        points3D={point_id: SimpleNamespace(xyz=xyz) for point_id, xyz in zip(ids, world)}
    )
    ret = {
        "cam_from_world": pose,
        "camera": camera,
        "inliers": np.ones(len(world), dtype=bool),
        "num_inliers": len(world),
    }
    selected = {
        "PnP_ret": ret,
        "keypoints_query": keypoints,
        "points3D_ids": ids,
        "num_matches": len(world),
    }

    result = _query_result(
        "query.jpg",
        selected,
        [selected],
        reconstruction,
        max_translation=1.0,
        max_rotation_deg=5.0,
    )

    assert result.success is True
    assert result.inliers == len(world)
    assert result.inlier_ratio == pytest.approx(1.0)
    assert result.reproj_p90_px == pytest.approx(0.0, abs=1e-8)
    assert result.positive_depth_ratio == pytest.approx(1.0)
    assert result.hull_coverage > 0
    assert result.grid4_occupancy > 0
    assert result.pose_consensus == pytest.approx(1.0)

    output = write_hloc_results([result], tmp_path / "results.csv")
    assert output.exists()
    assert "query.jpg" in output.read_text(encoding="utf-8")


def test_coverage_matches_canonical_and_ignores_oob_and_nan():
    width, height = 100, 100
    quad = [(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)]
    hull, occupied = _coverage(quad, width, height)
    expected_hull = convex_hull_area_fraction(quad, width, height)
    expected_occupied, _ = grid_coverage(quad, width, height, rows=4, cols=4)
    assert hull == expected_hull
    assert occupied == expected_occupied

    with_oob = quad + [(-10.0, 50.0), (150.0, 50.0), (50.0, -5.0)]
    hull_oob, occupied_oob = _coverage(with_oob, width, height)
    assert occupied_oob == occupied
    assert hull_oob == hull

    with_nan = quad + [(float("nan"), 10.0), (10.0, float("nan"))]
    hull_nan, occupied_nan = _coverage(with_nan, width, height)
    assert hull_nan == hull
    assert occupied_nan == occupied
