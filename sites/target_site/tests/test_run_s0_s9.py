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
