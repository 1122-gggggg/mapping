#!/usr/bin/env python3
"""Issue the final Fuhe Bridge release only when every S0-S9 gate is green."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edm_gate_contract import require_fresh_v2_gate
from ts_common import Gate, RUN_ID, hash_artifact, verify_predecessor_chain
from validate_edm_bundle import (
    EXPECTED_BASELINE_MEDIAN,
    EXPECTED_BASELINE_PATH,
    EXPECTED_BASELINE_SHA256,
)

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


def gate_statuses_pass(paths: list[Path]) -> tuple[bool, dict[str, str | None]]:
    evidence = {}
    valid = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            status = payload.get("status")
            valid.append(
                payload.get("schema_version") == "sfm-gate-v2"
                and status == "PASS"
                and payload.get("ok") is True
            )
        except (OSError, ValueError, TypeError):
            status = None
            valid.append(False)
        evidence[str(path)] = status
    return bool(paths) and all(valid), evidence


def release_artifact_paths(run_dir: Path, package_dir: Path) -> dict[str, Path]:
    tracking_bundle_name = f"{RUN_ID}_seed_tracking.pt"
    edm_bundle_name = f"{RUN_ID}_reloc_map_edm.pt"
    return {
        "final_model": run_dir / "final_model",
        "tracking_bundle": run_dir / "edm" / tracking_bundle_name,
        "edm_bundle": run_dir / "edm" / edm_bundle_name,
        "site_scale": run_dir / "site_scale.json",
        "T_align_gravity": run_dir / "T_align_gravity.json",
        "anchor_density_baseline": EXPECTED_BASELINE_PATH,
        "package_config": package_dir / "config.json",
        "package_manifest": package_dir / "MANIFEST.sha256",
        "package_ref_poses": package_dir / "maps" / "fuhe_bridge_ref_poses.json",
        "package_bundle": package_dir / "bundles" / edm_bundle_name,
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


def release_lineage_checks(
    *,
    package_config: dict,
    artifacts: dict[str, dict],
    s8_gate: dict,
    s9_gate: dict,
) -> dict[str, bool]:
    """Cross-bind release artifacts, package declarations, and S8/S9 evidence."""
    source = package_config.get("source_provenance", {})
    baseline = source.get("anchor_density_baseline", {})
    tracking_sha = artifacts["tracking_bundle"].get("sha256")
    edm_sha = artifacts["edm_bundle"].get("sha256")
    package_edm_sha = artifacts["package_bundle"].get("sha256")
    baseline_sha = artifacts["anchor_density_baseline"].get("sha256")
    config_sha = artifacts["package_config"].get("sha256")
    manifest_sha = artifacts["package_manifest"].get("sha256")
    return {
        "package_schema_v2": package_config.get("schema")
        == "edm-localization/v2",
        "tracking_hash_bound": bool(
            tracking_sha
            and source.get("tracking_bundle_sha256") == tracking_sha
            and source.get("bundle_source_tracking_bundle_sha256") == tracking_sha
            and _gate_input_sha256(s8_gate, "tracking_bundle") == tracking_sha
            and _gate_input_sha256(s9_gate, "tracking_bundle") == tracking_sha
        ),
        "edm_hash_bound": bool(
            edm_sha
            and edm_sha == package_edm_sha
            and source.get("edm_bundle_sha256") == edm_sha
            and _gate_input_sha256(s8_gate, "edm_bundle") == edm_sha
            and _gate_input_sha256(s9_gate, "package_edm_bundle") == edm_sha
        ),
        "baseline_hash_and_median_bound": bool(
            baseline_sha == EXPECTED_BASELINE_SHA256
            and baseline.get("path") == "outputs/river_site_reloc_map_edm.pt"
            and baseline.get("sha256") == EXPECTED_BASELINE_SHA256
            and float(
                baseline.get("median_3d_anchored_per_ref", float("nan"))
            )
            == EXPECTED_BASELINE_MEDIAN
            and _gate_input_sha256(s8_gate, "anchor_density_baseline")
            == EXPECTED_BASELINE_SHA256
            and _gate_input_sha256(s9_gate, "anchor_density_baseline")
            == EXPECTED_BASELINE_SHA256
        ),
        "package_and_results_bound": bool(
            config_sha
            and manifest_sha
            and _gate_input_sha256(s9_gate, "package_config") == config_sha
            and _gate_input_sha256(s9_gate, "package_manifest") == manifest_sha
            and any(
                label.startswith("benchmark_result_")
                and record.get("sha256")
                for label, record in s9_gate.get("provenance", {})
                .get("input_artifacts", {})
                .items()
            )
        ),
    }


def package_manifest_is_fresh(package_dir: Path) -> tuple[bool, str | None]:
    manifest = package_dir / "MANIFEST.sha256"
    try:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            target = (package_dir / relative).resolve(strict=True)
            if package_dir.resolve() not in (target, *target.parents):
                raise ValueError(f"manifest path escapes package: {relative}")
            if hash_artifact(target).get("sha256") != expected:
                raise ValueError(f"manifest SHA-256 drift: {relative}")
    except (OSError, ValueError, TypeError) as error:
        return False, str(error)
    return True, None


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
    artifacts = release_artifact_paths(args.run_dir, args.package_dir)
    artifact_records = {name: hash_artifact(path) for name, path in artifacts.items()}
    gate_records = {
        path.name: hash_artifact(path) for path in [*stage_chain, *gate_paths]
    }
    artifacts_ok = all(record.get("sha256") for record in artifact_records.values())
    gates_hashed = all(record.get("sha256") for record in gate_records.values())

    freshness_errors = []
    for path in [*stage_chain, *gate_paths]:
        try:
            require_fresh_v2_gate(path, expected_stage=path.stem)
        except (OSError, ValueError, RuntimeError) as error:
            freshness_errors.append(f"{path.name}: {error}")

    chain_errors = []
    for chain in (
        stage_chain,
        [
            gate_dir / "S3_pairs.json",
            gate_dir / "S4_doppelgangers.json",
            gate_dir / "S5_fixed_intrinsics.json",
            gate_dir / "S5_7_independent_sim3.json",
            gate_dir / "S5_7_S6_geometry.json",
        ],
        [
            gate_dir / "S5_7_S6_geometry.json",
            gate_dir / "S7_tracking_bundle.json",
            gate_dir / "S8_edm_bundle.json",
            gate_dir / "S9_heldout_localization.json",
        ],
    ):
        try:
            verify_predecessor_chain(chain)
        except (OSError, ValueError, RuntimeError) as error:
            chain_errors.append(str(error))

    package_config = json.loads(
        artifacts["package_config"].read_text(encoding="utf-8")
    )
    s8_gate = json.loads(
        (gate_dir / "S8_edm_bundle.json").read_text(encoding="utf-8")
    )
    s9_gate = json.loads(
        (gate_dir / "S9_heldout_localization.json").read_text(encoding="utf-8")
    )
    lineage = release_lineage_checks(
        package_config=package_config,
        artifacts=artifact_records,
        s8_gate=s8_gate,
        s9_gate=s9_gate,
    )
    manifest_ok, manifest_error = package_manifest_is_fresh(args.package_dir)
    deliverable_ok, deliverable_evidence = localization_deliverables_ok(args.run_dir)


    output = args.out or gate_dir / "final_release.json"
    gate = Gate(
        "fuhe_bridge_final_release",
        {
            "release/gates",
            "release/freshness",
            "release/chains",
            "release/artifacts",
            "release/lineage",
            "release/package_manifest",
            "release/localization_deliverables",
        },
        script_path=__file__,
        source_files=[
            Path(__file__).with_name("edm_gate_contract.py"),
            Path(__file__).with_name("validate_edm_bundle.py"),
        ],
        input_artifacts={
            **{f"artifact/{name}": path for name, path in artifacts.items()},
            **{
                f"gate/{path.stem}": path
                for path in [*stage_chain, *gate_paths]
            },
        },
    )
    gate.record_predecessor_gate(
        "S9_heldout_localization",
        gate_dir / "S9_heldout_localization.json",
        expected_stage="S9_heldout_localization",
    )
    gate.check(
        "release/gates",
        statuses_ok,
        "every declared S0-S9 release gate is sfm-gate-v2 PASS",
        statuses=statuses,
    )
    gate.check(
        "release/freshness",
        not freshness_errors,
        "all source/input/predecessor material hashes still match",
        errors=freshness_errors,
    )
    gate.check(
        "release/chains",
        not chain_errors,
        "S0-S3, S3-S6, and S6-S9 predecessor chains revalidate by exact gate hash",
        errors=chain_errors,
    )
    gate.check(
        "release/artifacts",
        artifacts_ok and gates_hashed,
        "all release artifacts and gate evidence are present and hashed",
        artifacts=artifact_records,
        gates=gate_records,
    )
    gate.check(
        "release/lineage",
        all(lineage.values()),
        "tracking, EDM, baseline, package, and results are mutually hash-bound",
        lineage=lineage,
    )
    gate.check(
        "release/package_manifest",
        manifest_ok,
        "every package manifest entry still matches its file bytes",
        error=manifest_error,
    )
    gate.check(
        "release/localization_deliverables",
        deliverable_ok,
        "site_scale/1 and T_align_gravity/1 are present valid JSON",
        evidence=deliverable_evidence,
    )
    gate.write(args.run_dir, output_path=output)


if __name__ == "__main__":
    main()
