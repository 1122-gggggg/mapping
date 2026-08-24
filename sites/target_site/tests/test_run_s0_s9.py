from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from run_s0_s9 import gate_passed, plan_stages, resolve_stage_command  # noqa: E402


def _args(tmp_path: Path, **overrides) -> Namespace:
    values = dict(
        run_dir=tmp_path / "target_site_v1",
        python=sys.executable,
        twoview=None,
        input_model=None,
        output_model=None,
        intrinsics_seed=None,
        database=None,
        glomap_bin=None,
        model=None,
        s5_metrics=None,
        tracking_bundle=None,
        edm_bundle=None,
        baseline_bundle=None,
        result=None,
        package_bundle=None,
        package_config=None,
        start_from="S0",
    )
    values.update(overrides)
    return Namespace(**values)


def test_plan_includes_readme_s0_to_s9_order(tmp_path: Path) -> None:
    run_dir = tmp_path / "target_site_v1"
    run_dir.mkdir()
    stages = [spec["stage"] for spec in plan_stages(_args(tmp_path))]
    assert stages == [
        "S0",
        "S1",
        "S1b",
        "S2",
        "S2b",
        "S3",
        "S4",
        "S5",
        "S5.7",
        "S6",
        "S7",
        "S8",
        "S9",
    ]
    assert "--run-name" in plan_stages(_args(tmp_path))[0]["cmd"]
    assert plan_stages(_args(tmp_path))[0]["cmd"][-1] == "target_site_v1"


def test_s4_fails_closed_without_twoview(tmp_path: Path) -> None:
    run_dir = tmp_path / "target_site_v1"
    run_dir.mkdir()
    spec = next(item for item in plan_stages(_args(tmp_path)) if item["stage"] == "S4")
    with pytest.raises(SystemExit, match="S4"):
        resolve_stage_command(spec)


def test_gate_passed_requires_status_pass(tmp_path: Path) -> None:
    missing = tmp_path / "gates" / "S0_corpus.json"
    ok, detail = gate_passed(missing)
    assert ok is False
    assert "missing" in detail
    missing.parent.mkdir()
    missing.write_text('{"status": "FAIL"}', encoding="utf-8")
    assert gate_passed(missing)[0] is False
    missing.write_text('{"status": "PASS"}', encoding="utf-8")
    assert gate_passed(missing)[0] is True


def test_s9_canonical_command_wires_lineage_inputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "target_site_v1"
    run_dir.mkdir()
    (run_dir / "forced_bridges.json").write_text("{}", encoding="utf-8")
    (run_dir / "corpus_manifest.json").write_text("{}", encoding="utf-8")
    edm_dir = run_dir / "edm"
    edm_dir.mkdir()
    tracking = edm_dir / "target_site_v1_seed_tracking.pt"
    edm = edm_dir / "target_site_v1_reloc_map_edm.pt"
    tracking.write_bytes(b"tracking")
    edm.write_bytes(b"edm")
    result_a = tmp_path / "t01.json"
    result_b = tmp_path / "t02.json"
    result_a.write_text("{}", encoding="utf-8")
    result_b.write_text("{}", encoding="utf-8")

    spec = next(
        item
        for item in plan_stages(_args(tmp_path, result=[result_a, result_b]))
        if item["stage"] == "S9"
    )
    cmd = resolve_stage_command(spec)
    assert "--corpus-manifest" in cmd
    assert "--edm-bundle" in cmd
    assert "--tracking-bundle" in cmd
    assert "--package-bundle" not in cmd
    assert "--package-config" not in cmd
    assert str((run_dir / "corpus_manifest.json").resolve()) in cmd
    assert str(edm.resolve()) in cmd
    assert str(tracking.resolve()) in cmd
    assert str(result_a) in cmd
    assert str(result_b) in cmd

    missing = next(
        item for item in plan_stages(_args(tmp_path)) if item["stage"] == "S9"
    )
    with pytest.raises(SystemExit, match="S9"):
        resolve_stage_command(missing)
