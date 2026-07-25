from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import resource_guard  # noqa: E402
import s3_pairs as s3  # noqa: E402
from ts_common import BUILD, GLUEMAP_REPO, Gate  # noqa: E402


ENGINE_REGRESSION = GLUEMAP_REPO / "tests" / "test_multi_sequence_extra_pairs.py"


def test_engine_regression_does_not_bypass_the_real_constructor() -> None:
    source = ENGINE_REGRESSION.read_text(encoding="utf-8")

    assert "MultiSequencePairs.__new__" not in source
    assert "MultiSequencePairs(args" in source


def test_clean_config_binds_zero_workers_and_memory_safe_feature_budgets(
    tmp_path: Path,
) -> None:
    forced = tmp_path / "forced_bridges.txt"
    forced.write_text("# natural first\n", encoding="utf-8")
    config = s3.build_gluemap_config(tmp_path, forced)

    assert type(config.get("num_workers")) is int
    assert config["num_workers"] == 0
    assert {
        field: config.get(field)
        for field in (
            "num_track_per_img",
            "sift_max_num_features",
            "sift_max_num_orientations",
            "max_num_tracks",
        )
    } == {
        "num_track_per_img": 512,
        "sift_max_num_features": 2048,
        "sift_max_num_orientations": 1,
        "max_num_tracks": 400000,
    }
    assert Path(config["memory_safe_launcher"]).is_file()
    assert "rerun_from" not in config


def test_config_binds_one_camera_natural_first_and_global_resource_lock(
    tmp_path: Path,
) -> None:
    forced = tmp_path / "forced_bridges.txt"
    forced.write_text("# natural first\n", encoding="utf-8")
    config = s3.build_gluemap_config(tmp_path, forced)
    contract = resource_guard.contract_from_config(config, tmp_path)

    assert config["camera_model"] == "PINHOLE"
    assert config["intrinsics_mode"] == "SHARED"
    assert config["is_sequential"] is True
    assert config["num_neighbors"] > 0
    assert Path(config["resource_lock_path"]) == resource_guard.GLOBAL_HEAVY_LOCK
    assert not Path(config["resource_lock_path"]).is_relative_to(tmp_path.resolve())
    assert Path(config["resource_guard_log_path"]).is_relative_to(tmp_path.resolve())
    assert contract["lock"]["exclusive"] is True
    assert contract["lock"]["scope"] == "global_sfm_heavy"


def test_s3_gate_source_freshness_covers_the_memory_safe_launcher(
    tmp_path: Path,
) -> None:
    gate = s3.stage_gate(tmp_path)

    assert s3.MEMORY_SAFE_LAUNCHER.resolve() in set(gate._source_paths.values())


def _probe_payload(*, loader_pairs: int, forced_pairs: int) -> dict:
    n_images = 240
    return {
        "n_gt_camera_slots": 1,
        "n_non_null_gt_cameras": 1,
        "n_images": n_images,
        "n_unique_images": n_images,
        "n_dimension_matches": n_images,
        "missing_seed_names": [],
        "dimension_mismatches": [],
        "n_loader_pairs": loader_pairs,
        "pair_density": loader_pairs / n_images,
        "datasets": sorted(video.seq for video in BUILD),
        "images_path_is_absolute": True,
        "images_path": "/absolute/images",
        "n_forced_expected": forced_pairs,
        "n_forced_injected": forced_pairs,
        "missing_forced_pairs": 0,
        "missing_forced_endpoints": [],
    }


def _probe_gate() -> Gate:
    return Gate(
        "S3_pairs",
        {"G3.1", "G3.2", "G3.3", "G3.4", "G3.5d"},
    )


def test_natural_only_pre_retrieval_density_is_not_applicable() -> None:
    fwd = sorted(video.seq for video in BUILD if video.direction == "fwd")
    rev = sorted(video.seq for video in BUILD if video.direction == "rev")
    policy = s3.decide_gap_bridge_policy(fwd, rev, None)
    gate = _probe_gate()

    s3._emit_probe_checks(gate, _probe_payload(loader_pairs=0, forced_pairs=0), "", policy)

    check = next(item for item in gate.checks if item["id"] == "G3.3")
    assert check["state"] == "NOT_APPLICABLE"
    assert check["applicable"] is False
    assert check["metrics"]["retrieval_executed"] is False
    assert check["metrics"]["force_gap_bridges"] is False
    assert check["metrics"]["n_forced_pairs"] == 0


def test_forced_mode_thresholds_conditional_pairs_not_loader_density() -> None:
    fwd = ["P1090109_002"]
    rev = ["P1110111"]
    policy = s3.decide_gap_bridge_policy(
        fwd,
        rev,
        {"P1090109_002|P1110111": {"route_clusters": []}},
    )
    gate = _probe_gate()

    s3._emit_probe_checks(
        gate,
        _probe_payload(loader_pairs=10_000, forced_pairs=0),
        "",
        policy,
    )

    check = next(item for item in gate.checks if item["id"] == "G3.3")
    assert check["state"] == "FAIL"
    assert check["metrics"]["force_gap_bridges"] is True
    assert check["metrics"]["conditional_pair_density"] == 0.0
