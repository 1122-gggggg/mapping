from __future__ import annotations

import json
import sys
from pathlib import Path


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from verify_final_release import (  # noqa: E402
    GATE_FILENAMES,
    gate_statuses_pass,
    localization_deliverables_ok,
    release_artifact_paths,
    release_lineage_checks,
    s8_s9_freshness_ok,
)
from ts_common import hash_artifact  # noqa: E402


def test_gate_statuses_require_every_gate_to_pass(tmp_path: Path) -> None:
    paths = []
    for index in range(3):
        path = tmp_path / f"gate{index}.json"
        path.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
        paths.append(path)

    assert gate_statuses_pass(paths)[0] is True

    paths[1].write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")
    passed, evidence = gate_statuses_pass(paths)
    assert passed is False
    assert evidence[str(paths[1])] == "FAIL"


def test_gate_filenames_include_independent_sim3() -> None:
    assert "S5_7_independent_sim3.json" in GATE_FILENAMES


def test_missing_site_scale_or_gravity_fails_release_checks(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    package_dir = tmp_path / "package"
    run_dir.mkdir()
    (run_dir / "T_align_gravity.json").write_text(
        json.dumps({"schema": "T_align_gravity/1"}), encoding="utf-8"
    )
    ok, evidence = localization_deliverables_ok(run_dir)
    assert ok is False
    assert evidence["site_scale"] == "missing"

    (run_dir / "site_scale.json").write_text(
        json.dumps({"schema": "site_scale/1"}), encoding="utf-8"
    )
    (run_dir / "T_align_gravity.json").unlink()
    ok, evidence = localization_deliverables_ok(run_dir)
    assert ok is False
    assert evidence["T_align_gravity"] == "missing"

    records = {
        name: hash_artifact(path)
        for name, path in release_artifact_paths(run_dir, package_dir).items()
    }
    assert records["site_scale"]["sha256"]
    assert records["T_align_gravity"]["sha256"] is None


def test_present_hashed_localization_deliverables_pass(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    package_dir = tmp_path / "package"
    run_dir.mkdir()
    (run_dir / "site_scale.json").write_text(
        json.dumps({"schema": "site_scale/1", "metric_scale": False}),
        encoding="utf-8",
    )
    (run_dir / "T_align_gravity.json").write_text(
        json.dumps({"schema": "T_align_gravity/1", "ok": True}),
        encoding="utf-8",
    )
    ok, evidence = localization_deliverables_ok(run_dir)
    assert ok is True
    assert evidence == {"site_scale": "ok", "T_align_gravity": "ok"}

    records = {
        name: hash_artifact(path)
        for name, path in release_artifact_paths(run_dir, package_dir).items()
        if name in {"site_scale", "T_align_gravity"}
    }
    assert all(record.get("sha256") for record in records.values())


def test_s8_s9_freshness_requires_provenance(tmp_path: Path) -> None:
    gate_dir = tmp_path / "gates"
    gate_dir.mkdir()
    (gate_dir / "S8_edm_bundle.json").write_text(
        json.dumps({"stage": "S8_edm_bundle", "status": "PASS"}), encoding="utf-8"
    )
    (gate_dir / "S9_heldout_localization.json").write_text(
        json.dumps({"stage": "S9_heldout_localization", "status": "PASS"}),
        encoding="utf-8",
    )
    ok, error = s8_s9_freshness_ok(gate_dir)
    assert ok is False
    assert error and "provenance" in error


def test_release_lineage_binds_s8_hashes_when_present() -> None:
    tracking = "a" * 64
    edm = "b" * 64
    artifacts = {
        "tracking_bundle": {"sha256": tracking},
        "edm_bundle": {"sha256": edm},
        "package_bundle": {"sha256": edm},
    }
    s8 = {
        "provenance": {
            "input_artifacts": {
                "edm_bundle": {"sha256": edm},
                "tracking_bundle": {"sha256": tracking},
            }
        }
    }
    checks = release_lineage_checks(artifacts=artifacts, s8_gate=s8, s9_gate={})
    assert checks["s8_hashes_bound"] is True
    assert checks["package_edm_matches_run"] is True
    artifacts["edm_bundle"]["sha256"] = "c" * 64
    assert release_lineage_checks(artifacts=artifacts, s8_gate=s8, s9_gate={})[
        "s8_hashes_bound"
    ] is False


