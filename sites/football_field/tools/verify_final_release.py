#!/usr/bin/env python3
"""Issue the final football_field release only when every S0-S9 gate is green."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_common import RUN_ID, assert_gate_fresh, hash_artifact, verify_predecessor_chain


SITE_SCALE_SCHEMA = "site_scale/1"
GRAVITY_ALIGN_SCHEMA = "T_align_gravity/1"

GATE_FILENAMES = (
    "S0_S3_release.json",
    "S4_doppelgangers.json",
    "S5_fixed_intrinsics.json",
    "S5_7_S6_geometry.json",
    "S7_tracking_bundle.json",
    "S8_edm_bundle.json",
    "S9_heldout_localization.json",
)


def release_artifact_paths(run_dir: Path, package_dir: Path) -> dict[str, Path]:
    tracking_bundle_name = f"{RUN_ID}_seed_tracking.pt"
    edm_bundle_name = f"{RUN_ID}_reloc_map_edm.pt"
    paths = {
        "final_model": run_dir / "final_model",
        "tracking_bundle": run_dir / "edm" / tracking_bundle_name,
        "edm_bundle": run_dir / "edm" / edm_bundle_name,
        "site_scale": run_dir / "site_scale.json",
        "T_align_gravity": run_dir / "T_align_gravity.json",
        "package_config": package_dir / "config.json",
        "package_ref_poses": package_dir / "maps" / "football_field_ref_poses.json",
        "package_bundle": package_dir / "bundles" / edm_bundle_name,
    }
    leaked = [str(path) for path in paths.values() if "target_site" in str(path)]
    if leaked:
        raise ValueError(
            f"football release paths must not contain target_site: {leaked}"
        )
    return paths


def localization_deliverables_ok(run_dir: Path) -> tuple[bool, dict[str, str | None]]:
    specs = {
        "site_scale": (run_dir / "site_scale.json", SITE_SCALE_SCHEMA),
        "T_align_gravity": (run_dir / "T_align_gravity.json", GRAVITY_ALIGN_SCHEMA),
    }
    evidence: dict[str, str | None] = {}
    ok = True
    for name, (path, schema) in specs.items():
        if not path.is_file():
            evidence[name] = "missing"
            ok = False
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as error:
            evidence[name] = f"invalid json: {error}"
            ok = False
            continue
        if payload.get("schema") != schema:
            evidence[name] = f"schema {payload.get('schema')!r} != {schema!r}"
            ok = False
            continue
        evidence[name] = "ok"
    return ok, evidence


def gate_statuses_pass(paths: list[Path]) -> tuple[bool, dict[str, str | None]]:
    evidence = {}
    for path in paths:
        try:
            status = json.loads(path.read_text(encoding="utf-8")).get("status")
        except (OSError, ValueError, TypeError):
            status = None
        evidence[str(path)] = status
    return bool(paths) and all(status == "PASS" for status in evidence.values()), evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    gate_dir = args.run_dir / "gates"
    gate_paths = [gate_dir / name for name in GATE_FILENAMES]
    statuses_ok, statuses = gate_statuses_pass(gate_paths)

    stage_chain = [
        gate_dir / name
        for name in (
            "S0_corpus.json",
            "S1_motion.json",
            "S2_extract.json",
            "S2b_intrinsics.json",
            "S3_pairs.json",
        )
    ]
    provenance_error = None
    try:
        for path in [*stage_chain, gate_dir / "S0_S3_release.json"]:
            assert_gate_fresh(path)
        verify_predecessor_chain(stage_chain)
    except (OSError, ValueError, RuntimeError) as error:
        provenance_error = str(error)

    artifacts = release_artifact_paths(args.run_dir, args.package_dir)
    artifact_records = {name: hash_artifact(path) for name, path in artifacts.items()}
    artifacts_ok = all(record.get("sha256") for record in artifact_records.values())
    deliverables_ok, deliverable_evidence = localization_deliverables_ok(args.run_dir)
    checks = {
        "all_S0_S9_gates_pass": statuses_ok,
        "S0_S3_provenance_fresh": provenance_error is None,
        "release_artifacts_present_and_hashed": artifacts_ok,
        "localization_deliverables_valid": deliverables_ok,
    }
    result = {
        "stage": "football_field_final_release",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "gate_statuses": statuses,
        "provenance_error": provenance_error,
        "artifacts": artifact_records,
        "localization_deliverables": deliverable_evidence,
    }
    output = args.out or gate_dir / "final_release.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if result["status"] != "PASS":
        raise SystemExit("final release gate failed")


if __name__ == "__main__":
    main()
