#!/usr/bin/env python3
"""Issue the final target-site release only when every S0-S9 gate is green."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_common import assert_gate_fresh, hash_artifact, verify_predecessor_chain


GATE_FILENAMES = (
    "S0_S3_release.json",
    "S4_doppelgangers.json",
    "S5_fixed_intrinsics.json",
    "S5_7_S6_geometry.json",
    "S7_tracking_bundle.json",
    "S8_edm_bundle.json",
    "S9_heldout_localization.json",
)


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

    artifacts = {
        "final_model": args.run_dir / "final_model",
        "tracking_bundle": args.run_dir / "edm" / "target_site_v1_seed_tracking.pt",
        "edm_bundle": args.run_dir / "edm" / "target_site_v1_reloc_map_edm.pt",
        "package_config": args.package_dir / "config.json",
        "package_ref_poses": args.package_dir / "maps" / "target_site_ref_poses.json",
        "package_bundle": args.package_dir
        / "bundles"
        / "target_site_v1_reloc_map_edm.pt",
    }
    artifact_records = {name: hash_artifact(path) for name, path in artifacts.items()}
    artifacts_ok = all(record.get("sha256") for record in artifact_records.values())
    checks = {
        "all_S0_S9_gates_pass": statuses_ok,
        "S0_S3_provenance_fresh": provenance_error is None,
        "release_artifacts_present_and_hashed": artifacts_ok,
    }
    result = {
        "stage": "target_site_final_release",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "gate_statuses": statuses,
        "provenance_error": provenance_error,
        "artifacts": artifact_records,
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
