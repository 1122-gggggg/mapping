from __future__ import annotations

import json
import sys
from pathlib import Path


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from verify_final_release import (  # noqa: E402
    EXPECTED_BASELINE_SHA256,
    RUN_ID,
    gate_statuses_pass,
    localization_deliverables_ok,
    release_lineage_checks,
    release_artifact_paths,
)
from ts_common import hash_artifact  # noqa: E402


def test_gate_statuses_require_every_gate_to_pass(tmp_path: Path) -> None:
    paths = []
    for index in range(3):
        path = tmp_path / f"gate{index}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "sfm-gate-v2",
                    "status": "PASS",
                    "ok": True,
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    assert gate_statuses_pass(paths)[0] is True

    paths[1].write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")
    passed, evidence = gate_statuses_pass(paths)
    assert passed is False
    assert evidence[str(paths[1])] == "FAIL"


def test_gate_statuses_reject_legacy_pass_payload(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")

    assert gate_statuses_pass([legacy])[0] is False


def test_release_artifacts_use_fuhe_names_and_do_not_reference_target_site(
    tmp_path: Path,
) -> None:
    paths = release_artifact_paths(tmp_path / "run", tmp_path / "package")

    assert RUN_ID == "fuhe_bridge_v2"
    assert paths["tracking_bundle"].name == f"{RUN_ID}_seed_tracking.pt"
    assert paths["edm_bundle"].name == f"{RUN_ID}_reloc_map_edm.pt"
    assert paths["package_ref_poses"].name == "fuhe_bridge_ref_poses.json"
    assert paths["package_bundle"].name == f"{RUN_ID}_reloc_map_edm.pt"
    assert all("target_site" not in str(path) for path in paths.values())
    assert paths["site_scale"].name == "site_scale.json"
    assert paths["T_align_gravity"].name == "T_align_gravity.json"


def test_release_lineage_cross_binds_s8_s9_package_and_results() -> None:
    tracking_sha = "a" * 64
    edm_sha = "b" * 64
    config_sha = "c" * 64
    manifest_sha = "d" * 64
    artifacts = {
        "tracking_bundle": {"sha256": tracking_sha},
        "edm_bundle": {"sha256": edm_sha},
        "package_bundle": {"sha256": edm_sha},
        "anchor_density_baseline": {"sha256": EXPECTED_BASELINE_SHA256},
        "package_config": {"sha256": config_sha},
        "package_manifest": {"sha256": manifest_sha},
    }
    config = {
        "schema": "edm-localization/v2",
        "source_provenance": {
            "tracking_bundle_sha256": tracking_sha,
            "bundle_source_tracking_bundle_sha256": tracking_sha,
            "edm_bundle_sha256": edm_sha,
            "anchor_density_baseline": {
                "path": "outputs/river_site_reloc_map_edm.pt",
                "sha256": EXPECTED_BASELINE_SHA256,
                "median_3d_anchored_per_ref": 3811.5,
            },
        },
    }
    s8 = {
        "provenance": {
            "input_artifacts": {
                "tracking_bundle": {"sha256": tracking_sha},
                "edm_bundle": {"sha256": edm_sha},
                "anchor_density_baseline": {"sha256": EXPECTED_BASELINE_SHA256},
            }
        }
    }
    s9 = {
        "provenance": {
            "input_artifacts": {
                "tracking_bundle": {"sha256": tracking_sha},
                "package_edm_bundle": {"sha256": edm_sha},
                "anchor_density_baseline": {"sha256": EXPECTED_BASELINE_SHA256},
                "package_config": {"sha256": config_sha},
                "package_manifest": {"sha256": manifest_sha},
                "benchmark_result_0": {"sha256": "e" * 64},
            }
        }
    }

    checks = release_lineage_checks(
        package_config=config,
        artifacts=artifacts,
        s8_gate=s8,
        s9_gate=s9,
    )
    assert all(checks.values())

    artifacts["tracking_bundle"]["sha256"] = "f" * 64
    assert not all(
        release_lineage_checks(
            package_config=config,
            artifacts=artifacts,
            s8_gate=s8,
            s9_gate=s9,
        ).values()
    )


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

