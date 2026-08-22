from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from ts_common import BUILD, GLUEMAP_REPO, RUNS, RUN_ID  # noqa: E402
import s3_pairs as s3  # noqa: E402


ENGINE_REGRESSION = GLUEMAP_REPO / "tests" / "test_multi_sequence_extra_pairs.py"
FOOTBALL_RUN = RUNS / RUN_ID
HAS_FOOTBALL_S3 = (FOOTBALL_RUN / "gluemap_config.yaml").is_file()


def test_s3_expected_counts_follow_football_corpus() -> None:
    assert len(BUILD) == 2
    assert s3.EXPECTED_CAMERAS == 1
    assert s3.EXPECTED_SEQUENCE_PAIRS == 0
    assert s3.EXPECTED_FORCED_PAIRS == 0
    assert all(video.direction == "unknown" for video in BUILD)


def test_engine_regression_does_not_bypass_the_real_constructor() -> None:
    pytest.importorskip("torch")
    if not ENGINE_REGRESSION.is_file():
        pytest.skip("gluemap engine regression test is not present")
    source = ENGINE_REGRESSION.read_text(encoding="utf-8")

    assert "MultiSequencePairs.__new__" not in source
    assert "MultiSequencePairs(args" in source


@pytest.mark.skipif(not HAS_FOOTBALL_S3, reason="football S3 run is not present")
def test_production_config_and_g36_bind_literal_zero_num_workers() -> None:
    run_dir = FOOTBALL_RUN
    config = yaml.safe_load(
        (run_dir / "gluemap_config.yaml").read_text(encoding="utf-8")
    )
    gate = json.loads(
        (run_dir / "gates" / "S3_pairs.json").read_text(encoding="utf-8")
    )
    g36 = next(check for check in gate["checks"] if check["id"] == "G3.6")

    assert type(config.get("num_workers")) is int
    assert config["num_workers"] == 0
    assert type(g36["metrics"].get("num_workers")) is int
    assert g36["metrics"]["num_workers"] == 0


@pytest.mark.skipif(not HAS_FOOTBALL_S3, reason="football S3 run is not present")
def test_production_config_binds_memory_safe_feature_budgets() -> None:
    run_dir = FOOTBALL_RUN
    config = yaml.safe_load(
        (run_dir / "gluemap_config.yaml").read_text(encoding="utf-8")
    )
    gate = json.loads(
        (run_dir / "gates" / "S3_pairs.json").read_text(encoding="utf-8")
    )
    g36 = next(check for check in gate["checks"] if check["id"] == "G3.6")

    expected = {
        "num_track_per_img": 512,
        "sift_max_num_features": 2048,
        "max_num_tracks": 400000,
    }
    assert {field: config.get(field) for field in expected} == expected
    assert {field: g36["metrics"].get(field) for field in expected} == expected
    launcher = Path(config["memory_safe_launcher"])
    assert launcher.is_absolute() and launcher.is_file()
    assert g36["metrics"]["memory_safe_launcher"] == str(launcher)


@pytest.mark.skipif(not HAS_FOOTBALL_S3, reason="football S3 run is not present")
def test_production_config_injects_exact_bridges_and_loads_one_gt_camera(
    monkeypatch,
) -> None:
    argparse = pytest.importorskip("argparse")
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(GLUEMAP_REPO))
    from gluemap.datasets import multi_sequence_twoview as multi
    from gluemap.utils.colmap import extract_gt_intrinsics

    run_dir = FOOTBALL_RUN
    config = yaml.safe_load(
        (run_dir / "gluemap_config.yaml").read_text(encoding="utf-8")
    )
    args = argparse.Namespace(**config)
    datasets = sorted(path.name for path in Path(args.images_path).iterdir() if path.is_dir())

    def fake_descriptors(self, _args, _dataset, image_names):
        count = len(image_names)
        neighbors = np.arange(count, dtype=np.int64).reshape(count, 1)
        descriptors = torch.zeros((count, 1), dtype=torch.float32)
        return neighbors, descriptors

    monkeypatch.setattr(
        multi.MultiSequencePairs,
        "_retrieve_descriptors_and_neighbors",
        fake_descriptors,
    )
    monkeypatch.setattr(
        multi,
        "retrieve_global_neighbors",
        lambda *_args, **_kwargs: np.empty((0, 2), dtype=np.int64),
    )

    dataset = multi.MultiSequencePairs(args, datasets)
    loaded_pairs = {tuple(pair) for pair in dataset.pairs.tolist()}

    expected_pairs = set()
    index_of = {name: index for index, name in enumerate(dataset.images_list)}
    for line in Path(args.extra_pairs_path).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        first, second = line.split()
        i, j = index_of[first], index_of[second]
        expected_pairs.add((min(i, j), max(i, j)))

    gt_intrinsics = extract_gt_intrinsics(
        args.gt_intrinsics_path,
        dataset.images_list,
        dataset.intrinsics_mapping,
    )

    assert Path(args.images_path).is_absolute()
    assert len(expected_pairs) == s3.EXPECTED_FORCED_PAIRS
    assert loaded_pairs == expected_pairs
    assert len(gt_intrinsics) == s3.EXPECTED_CAMERAS
    assert sum(item is not None for item in gt_intrinsics) == s3.EXPECTED_CAMERAS
