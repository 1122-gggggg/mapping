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
    release_lineage_checks,
    release_artifact_paths,
)


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
