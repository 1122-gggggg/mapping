from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from update_pipeline import (
    DEFAULT_FPS,
    DEFAULT_KEEP_ROTATION_EVERY,
    DEFAULT_MAX_FLOW_PX,
    DEFAULT_MIN_FLOW_PX,
    build_sparsify_command,
    build_validation_command,
    inspect_flight_bundle,
    inspect_update_summary,
    write_quality_report,
)


def test_frame_preparation_defaults_match_validated_split_run():
    assert DEFAULT_FPS == 3.0
    assert DEFAULT_MIN_FLOW_PX == 0.8
    assert DEFAULT_MAX_FLOW_PX == 180.0
    assert DEFAULT_KEEP_ROTATION_EVERY == 8


def test_build_sparsify_command_includes_stats_and_keep_prefixes(tmp_path: Path):
    obs = tmp_path / "observation_stats.json"
    obs.write_text("{}", encoding="utf-8")
    args = Namespace(
        python="/usr/bin/python3.12",
        sparsify_target_fraction=0.7,
        sparsify_min_per_prefix=25,
        sparsify_keep_prefix=["P1230123", "P1260126"],
    )

    cmd = build_sparsify_command(args, tmp_path / "full.pt", tmp_path / "slim.pt", obs)

    assert "--observation-stats" in cmd
    assert str(obs) in cmd
    assert cmd.count("--keep-prefix") == 2
    assert "P1230123" in cmd
    assert "P1260126" in cmd


def test_inspect_flight_bundle_rejects_bad_megaloc_descriptor_dim(tmp_path: Path):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    bundle = tmp_path / "bundle.pt"
    torch.save({
        "ref_names": ["a"],
        "refs": {"a": {}},
        "ref_global": np.ones((1, 1), dtype=np.float32),
        "ref_centers": np.zeros((1, 3), dtype=np.float32),
        "ref_yaws": np.zeros((1,), dtype=np.float32),
        "covis": {"a": []},
        "meta": {"bundle_vpr": "megaloc", "tracking_metadata": True},
    }, bundle)

    _detail, reasons = inspect_flight_bundle(bundle)

    assert any("ref_global must be finite MegaLoc descriptors" in r for r in reasons)


def test_inspect_flight_bundle_rejects_bad_ref_stability_shape(tmp_path: Path):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    bundle = tmp_path / "bundle.pt"
    torch.save({
        "ref_names": ["a", "b"],
        "refs": {"a": {}, "b": {}},
        "ref_global": np.ones((2, 8448), dtype=np.float32),
        "ref_centers": np.zeros((2, 3), dtype=np.float32),
        "ref_yaws": np.zeros((2,), dtype=np.float32),
        "ref_stability": np.ones((1,), dtype=np.float32),
        "covis": {"a": [1], "b": [0]},
        "meta": {"bundle_vpr": "megaloc", "tracking_metadata": True},
    }, bundle)

    _detail, reasons = inspect_flight_bundle(bundle)

    assert any("ref_stability must be finite" in r for r in reasons)


def test_update_quality_report_fails_empty_rows(tmp_path: Path):
    result = tmp_path / "validation.json"
    result.write_text('{"rows": []}', encoding="utf-8")

    ok = write_quality_report(result, min_success=0.9, max_ok_to_fail=0, max_final_fail_run=30)

    assert not ok


def test_update_quality_report_marks_regression_failure_even_above_success_target(tmp_path: Path):
    result = tmp_path / "validation.json"
    result.write_text(
        '{"rows": [{"set": "heldout", "n": 10, "base_success": 0.9, '
        '"final_success": 1.0, "gain_pp": 10.0, "ok_to_fail": 1, '
        '"final_max_fail_run": 1, "base_max_fail_run": 1}]}',
        encoding="utf-8",
    )

    ok = write_quality_report(result, min_success=0.9, max_ok_to_fail=0, max_final_fail_run=30)

    report = result.with_suffix(".quality_report.md").read_text(encoding="utf-8")
    assert not ok
    assert "FAIL_REGRESSION" in report


def test_build_validation_command_uses_requested_base_megaloc_cache(tmp_path: Path):
    args = Namespace(
        python="/usr/bin/python3.12",
        base_bundle="/maps/base.pt",
        base_megaloc_cache="/custom/base_megaloc.npz",
        validate_dir="/validation",
        validate_stride=10,
        validate_resize="1280x720",
    )

    cmd = build_validation_command(args, tmp_path / "final.pt", tmp_path / "validation.json")

    cache_index = cmd.index("--base-megaloc-cache")
    assert cmd[cache_index + 1] == "/custom/base_megaloc.npz"


def test_inspect_update_summary_rejects_unimplemented_changed_route(tmp_path: Path):
    out_dir = tmp_path
    (out_dir / "update_summary.json").write_text(
        '{"rows": [{"seq": "P2000200", "route": "changed-region", '
        '"status": "needs_tile_replace"}]}',
        encoding="utf-8",
    )

    detail, reasons = inspect_update_summary(out_dir)

    assert detail["summary_exists"]
    assert any("tile replacement is not implemented" in reason for reason in reasons)
