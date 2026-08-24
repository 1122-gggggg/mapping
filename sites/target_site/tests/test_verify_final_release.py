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


def _s8_gate(edm: str, tracking: str) -> dict:
    return {
        "provenance": {
            "input_artifacts": {
                "edm_bundle": {"sha256": edm},
                "tracking_bundle": {"sha256": tracking},
            }
        }
    }


def _s9_gate(
    *,
    edm: str | None = None,
    tracking: str | None = None,
    package: str | None = None,
    results: bool = True,
) -> dict:
    artifacts: dict[str, dict[str, str]] = {}
    if results:
        artifacts["benchmark_result_0"] = {"sha256": "d" * 64}
        artifacts["benchmark_result_1"] = {"sha256": "e" * 64}
    if edm is not None:
        artifacts["edm_bundle"] = {"sha256": edm}
    if tracking is not None:
        artifacts["tracking_bundle"] = {"sha256": tracking}
    if package is not None:
        artifacts["package_edm_bundle"] = {"sha256": package}
    return {"provenance": {"input_artifacts": artifacts}}


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


def test_release_lineage_requires_s8_and_complete_s9_hashes() -> None:
    tracking = "a" * 64
    edm = "b" * 64
    artifacts = {
        "tracking_bundle": {"sha256": tracking},
        "edm_bundle": {"sha256": edm},
        "package_bundle": {"sha256": edm},
    }
    s8 = _s8_gate(edm, tracking)
    s9 = _s9_gate(edm=edm, tracking=tracking, package=edm)
    checks = release_lineage_checks(artifacts=artifacts, s8_gate=s8, s9_gate=s9)
    assert checks["s8_hashes_bound"] is True
    assert checks["s9_hashes_bound"] is True
    assert checks["s9_result_hashes_bound"] is True
    assert checks["s9_package_bound"] is True
    assert checks["package_edm_matches_run"] is True
    assert all(checks.values()) is True

    artifacts["edm_bundle"]["sha256"] = "c" * 64
    mismatched = release_lineage_checks(artifacts=artifacts, s8_gate=s8, s9_gate=s9)
    assert mismatched["s8_hashes_bound"] is False
    assert mismatched["s9_hashes_bound"] is False
    assert mismatched["package_edm_matches_run"] is False


def test_status_only_s9_lineage_cannot_pass() -> None:
    tracking = "a" * 64
    edm = "b" * 64
    artifacts = {
        "tracking_bundle": {"sha256": tracking},
        "edm_bundle": {"sha256": edm},
        "package_bundle": {"sha256": edm},
    }
    checks = release_lineage_checks(
        artifacts=artifacts,
        s8_gate=_s8_gate(edm, tracking),
        s9_gate={"stage": "S9_heldout_localization", "status": "PASS", "ok": True},
    )
    assert checks["s8_hashes_bound"] is True
    assert checks["s9_hashes_bound"] is False
    assert checks["s9_result_hashes_bound"] is False
    assert checks["s9_package_bound"] is False
    assert all(checks.values()) is False


def test_missing_s9_lineage_cannot_pass() -> None:
    tracking = "a" * 64
    edm = "b" * 64
    artifacts = {
        "tracking_bundle": {"sha256": tracking},
        "edm_bundle": {"sha256": edm},
        "package_bundle": {"sha256": edm},
    }
    s9 = {
        "stage": "S9_heldout_localization",
        "status": "PASS",
        "provenance": {
            "input_artifacts": {
                "forced_manifest": {"sha256": "f" * 64},
            }
        },
    }
    checks = release_lineage_checks(
        artifacts=artifacts,
        s8_gate=_s8_gate(edm, tracking),
        s9_gate=s9,
    )
    assert checks["s9_hashes_bound"] is False
    assert checks["s9_result_hashes_bound"] is False
    assert checks["s9_package_bound"] is False
    assert all(checks.values()) is False


def test_canonical_s9_provenance_satisfies_release_lineage(tmp_path: Path) -> None:
    from ts_common import TEST
    from validate_heldout_localization import main as s9_main

    test_ids = [video.seq for video in TEST]
    result_paths = []
    for seq in test_ids:
        payload = {
            "video": seq,
            "frames": 1000,
            "localized": 970,
            "rate": 0.97,
            "inliers_p05": 40.0,
            "step_median": 0.01,
            "step_p95": 0.05,
            "jumps_gt_10x_median": 0,
            "rejections": {"jump": 1},
            "reference_sequence_counts": {seq: 100},
        }
        path = tmp_path / f"{seq}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        result_paths.append(path)

    corpus = tmp_path / "corpus_manifest.json"
    corpus.write_text(
        json.dumps(
            {
                "test": [
                    {
                        "seq": test_ids[0],
                        "sha256": "a" * 64,
                        "probed": {"nb_frames": 1000},
                    },
                    {
                        "seq": test_ids[1],
                        "sha256": "b" * 64,
                        "probed": {"nb_frames": 1000},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    forced = tmp_path / "forced_bridges.json"
    forced.write_text(json.dumps({"rev": []}), encoding="utf-8")
    edm = tmp_path / "target_site_v1_reloc_map_edm.pt"
    tracking = tmp_path / "target_site_v1_seed_tracking.pt"
    edm.write_bytes(b"edm-bytes")
    tracking.write_bytes(b"tracking-bytes")
    out = tmp_path / "S9_heldout_localization.json"

    argv = [
        "validate_heldout_localization.py",
        *[item for path in result_paths for item in ("--result", str(path))],
        "--forced-manifest",
        str(forced),
        "--corpus-manifest",
        str(corpus),
        "--edm-bundle",
        str(edm),
        "--tracking-bundle",
        str(tracking),
        "--out",
        str(out),
    ]
    previous = sys.argv
    sys.argv = argv
    try:
        s9_main()
    finally:
        sys.argv = previous

    s9_gate = json.loads(out.read_text(encoding="utf-8"))
    artifacts = {
        "tracking_bundle": hash_artifact(tracking),
        "edm_bundle": hash_artifact(edm),
        "package_bundle": hash_artifact(edm),
    }
    s8 = {
        "provenance": {
            "input_artifacts": {
                "edm_bundle": hash_artifact(edm),
                "tracking_bundle": hash_artifact(tracking),
            }
        }
    }
    checks = release_lineage_checks(artifacts=artifacts, s8_gate=s8, s9_gate=s9_gate)
    assert s9_gate["status"] == "PASS"
    assert s9_gate["hard_status"] == "VALID"
    provenance = s9_gate["provenance"]["input_artifacts"]
    assert provenance["edm_bundle"]["sha256"] == artifacts["edm_bundle"]["sha256"]
    assert provenance["tracking_bundle"]["sha256"] == artifacts["tracking_bundle"]["sha256"]
    assert provenance["package_edm_bundle"]["sha256"] == artifacts["package_bundle"]["sha256"]
    assert provenance["package_edm_bundle"]["binding"] == "transitive_edm_identity"
    assert checks["s8_hashes_bound"] is True
    assert checks["s9_hashes_bound"] is True
    assert checks["s9_result_hashes_bound"] is True
    assert checks["s9_package_bound"] is True
    assert checks["package_edm_matches_run"] is True
    assert all(checks.values()) is True
