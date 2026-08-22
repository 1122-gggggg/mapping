#!/usr/bin/env python3
"""Issue the final target-site release only when every S0-S9 gate is green."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_common import assert_gate_fresh, hash_artifact, verify_predecessor_chain


SITE_SCALE_SCHEMA = "site_scale/1"
GRAVITY_ALIGN_SCHEMA = "T_align_gravity/1"

GATE_FILENAMES = (
    "S0_S3_release.json",
    "S4_doppelgangers.json",
    "S5_fixed_intrinsics.json",
    "S5_7_independent_sim3.json",
    "S5_7_S6_geometry.json",
    "S7_tracking_bundle.json",
    "S8_edm_bundle.json",
    "S9_heldout_localization.json",
)


def release_artifact_paths(run_dir: Path, package_dir: Path) -> dict[str, Path]:
    return {
        "final_model": run_dir / "final_model",
        "tracking_bundle": run_dir / "edm" / "target_site_v1_seed_tracking.pt",
        "edm_bundle": run_dir / "edm" / "target_site_v1_reloc_map_edm.pt",
        "site_scale": run_dir / "site_scale.json",
        "T_align_gravity": run_dir / "T_align_gravity.json",
        "S5_7_independent_sim3": run_dir / "gates" / "S5_7_independent_sim3.json",
        "package_config": package_dir / "config.json",
        "package_ref_poses": package_dir / "maps" / "target_site_ref_poses.json",
        "package_bundle": package_dir
        / "bundles"
        / "target_site_v1_reloc_map_edm.pt",
    }


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

def _gate_input_sha256(payload: dict, label: str) -> str | None:
    return (
        payload.get("provenance", {})
        .get("input_artifacts", {})
        .get(label, {})
        .get("sha256")
    )


def s8_s9_freshness_ok(gate_dir: Path) -> tuple[bool, str | None]:
    errors: list[str] = []
    for name, stage in (
        ("S8_edm_bundle.json", "S8_edm_bundle"),
        ("S9_heldout_localization.json", "S9_heldout_localization"),
    ):
        path = gate_dir / name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("stage") != stage:
                errors.append(f"{name} stage {payload.get('stage')!r} != {stage!r}")
            if payload.get("status") != "PASS":
                errors.append(f"{name} status {payload.get('status')!r}")
            if not isinstance(payload.get("provenance"), dict):
                errors.append(f"{name} has no provenance")
            else:
                assert_gate_fresh(path)
        except (OSError, ValueError, RuntimeError) as error:
            errors.append(f"{name}: {error}")
    return not errors, ("; ".join(errors) if errors else None)


def release_lineage_checks(
    *,
    artifacts: dict[str, dict],
    s8_gate: dict,
    s9_gate: dict,
) -> dict[str, bool]:
    tracking_sha = artifacts["tracking_bundle"].get("sha256")
    edm_sha = artifacts["edm_bundle"].get("sha256")
    package_edm_sha = artifacts["package_bundle"].get("sha256")
    checks: dict[str, bool] = {}
    s8_edm = _gate_input_sha256(s8_gate, "edm_bundle")
    s8_tracking = _gate_input_sha256(s8_gate, "tracking_bundle")
    if s8_edm is not None or s8_tracking is not None:
        checks["s8_hashes_bound"] = bool(
            edm_sha
            and s8_edm == edm_sha
            and tracking_sha
            and s8_tracking == tracking_sha
        )
    s9_edm = _gate_input_sha256(s9_gate, "edm_bundle") or _gate_input_sha256(
        s9_gate, "package_edm_bundle"
    )
    s9_tracking = _gate_input_sha256(s9_gate, "tracking_bundle")
    if s9_edm is not None or s9_tracking is not None:
        checks["s9_hashes_bound"] = bool(
            (s9_edm is None or s9_edm == edm_sha)
            and (s9_tracking is None or s9_tracking == tracking_sha)
        )
    if edm_sha and package_edm_sha:
        checks["package_edm_matches_run"] = package_edm_sha == edm_sha
    return checks




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
    s8_s9_ok, s8_s9_error = s8_s9_freshness_ok(gate_dir)
    s8_gate = {}
    s9_gate = {}
    try:
        s8_gate = json.loads((gate_dir / "S8_edm_bundle.json").read_text(encoding="utf-8"))
        s9_gate = json.loads(
            (gate_dir / "S9_heldout_localization.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError):
        s8_s9_ok = False
        s8_s9_error = s8_s9_error or "S8/S9 gate JSON unreadable"
    lineage = release_lineage_checks(
        artifacts=artifact_records,
        s8_gate=s8_gate,
        s9_gate=s9_gate,
    )
    checks = {
        "all_S0_S9_gates_pass": statuses_ok,
        "S0_S3_provenance_fresh": provenance_error is None,
        "S8_S9_provenance_fresh": s8_s9_ok,
        "release_artifacts_present_and_hashed": artifacts_ok,
        "localization_deliverables_valid": deliverables_ok,
        "release_lineage_bound": bool(lineage) and all(lineage.values()),
    }
    result = {
        "stage": "target_site_final_release",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "gate_statuses": statuses,
        "provenance_error": provenance_error,
        "artifacts": artifact_records,
        "localization_deliverables": deliverable_evidence,
        "s8_s9_error": s8_s9_error,
        "lineage": lineage,
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
