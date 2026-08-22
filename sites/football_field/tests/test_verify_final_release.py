from __future__ import annotations

import json
import sys
from pathlib import Path


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from ts_common import RUN_ID, hash_artifact  # noqa: E402
from verify_final_release import (  # noqa: E402
    gate_statuses_pass,
    localization_deliverables_ok,
    release_artifact_paths,
)


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


def test_release_artifacts_use_football_names_and_do_not_reference_target_site(
    tmp_path: Path,
) -> None:
    paths = release_artifact_paths(tmp_path / "run", tmp_path / "package")

    assert RUN_ID == "football_field_v1"
    assert paths["tracking_bundle"].name == f"{RUN_ID}_seed_tracking.pt"
    assert paths["edm_bundle"].name == f"{RUN_ID}_reloc_map_edm.pt"
    assert paths["package_ref_poses"].name == "football_field_ref_poses.json"
    assert paths["package_bundle"].name == f"{RUN_ID}_reloc_map_edm.pt"
    assert all("target_site" not in str(path) for path in paths.values())
    assert paths["site_scale"].name == "site_scale.json"
    assert paths["T_align_gravity"].name == "T_align_gravity.json"


def test_missing_site_scale_or_gravity_fails_release_checks(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
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
        for name, path in release_artifact_paths(run_dir, tmp_path / "package").items()
        if name in {"site_scale", "T_align_gravity"}
    }
    assert records["site_scale"]["sha256"]
    assert records["T_align_gravity"]["sha256"] is None


def test_present_hashed_localization_deliverables_pass(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
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
        for name, path in release_artifact_paths(run_dir, tmp_path / "package").items()
        if name in {"site_scale", "T_align_gravity"}
    }
    assert all(record.get("sha256") for record in records.values())
