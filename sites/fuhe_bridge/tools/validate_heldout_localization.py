#!/usr/bin/env python3
"""Apply the S9 acceptance contract to held-out EDM video benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from edm_gate_contract import require_fresh_v2_gate
from ts_common import Gate
from validate_edm_bundle import (
    EXPECTED_BASELINE_MEDIAN,
    EXPECTED_BASELINE_PATH,
    EXPECTED_BASELINE_SHA256,
)


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REQUIRED_SCOPES = (
    "full",
    "diagnostic_first_half",
    "diagnostic_second_half",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def heldout_contract_from_manifest(path: Path) -> dict[str, str | int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    heldout = payload.get("test")
    if not isinstance(heldout, list) or len(heldout) != 1:
        raise ValueError("held-out contract requires exactly one independent video")
    item = heldout[0]
    digest = str(item.get("source_sha256") or item.get("sha256") or "")
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError("held-out video requires a lowercase SHA-256 digest")
    probed = item.get("probed", {})
    total_frames = int(probed.get("nb_frames", 0))
    if total_frames <= 0:
        raise ValueError("held-out video requires a positive probed frame count")
    video_id = str(item.get("seq", ""))
    if not video_id:
        raise ValueError("held-out video requires a sequence ID")
    return {
        "video_id": video_id,
        "video_sha256": digest,
        "total_frames": total_frames,
    }


def _validation_semantics(
    results: list[dict],
    *,
    expected_video_id: str,
    expected_video_sha256: str,
    expected_total_frames: int,
    expected_profile_sha256: str,
    expected_package_config_sha256: str,
    expected_edm_bundle_sha256: str,
    expected_tracking_bundle_sha256: str,
    expected_baseline_sha256: str,
    expected_baseline_median: float,
    expected_camera_contract_sha256: str,
) -> tuple[bool, dict]:
    midpoint = expected_total_frames // 2
    expected_ranges = {
        "full": [0, expected_total_frames],
        "diagnostic_first_half": [0, midpoint],
        "diagnostic_second_half": [midpoint, expected_total_frames],
    }
    scopes: dict[str, list[dict]] = {scope: [] for scope in REQUIRED_SCOPES}
    identities: set[tuple[str, str]] = set()
    rows_ok = True
    for result in results:
        evaluation = result.get("evaluation")
        runtime = result.get("runtime")
        if not isinstance(evaluation, dict) or not isinstance(runtime, dict):
            rows_ok = False
            continue
        scope = str(evaluation.get("scope", ""))
        video_id = str(evaluation.get("source_video_id", ""))
        digest = str(evaluation.get("source_video_sha256", ""))
        identities.add((video_id, digest))
        if scope not in scopes:
            rows_ok = False
            continue
        scopes[scope].append(evaluation)
        rows_ok = rows_ok and bool(
            video_id == expected_video_id
            and digest == expected_video_sha256
            and SHA256_PATTERN.fullmatch(digest)
            and int(evaluation.get("source_total_frames", -1))
            == expected_total_frames
            and evaluation.get("source_frame_range") == expected_ranges[scope]
            and evaluation.get("complete_scope") is True
            and evaluation.get("diagnostic_segment") is (scope != "full")
            and evaluation.get("independent_flight") is (scope == "full")
            and runtime.get("deployment_profile_sha256")
            == expected_profile_sha256
            and SHA256_PATTERN.fullmatch(expected_profile_sha256)
            and runtime.get("package_config_sha256")
            == expected_package_config_sha256
            and runtime.get("edm_bundle_sha256") == expected_edm_bundle_sha256
            and runtime.get("tracking_bundle_sha256")
            == expected_tracking_bundle_sha256
            and runtime.get("anchor_density_baseline_sha256")
            == expected_baseline_sha256
            and float(runtime.get("anchor_density_baseline_median", float("nan")))
            == expected_baseline_median
            and runtime.get("camera_contract") == "benchmark_camera_contract"
            and runtime.get("benchmark_camera_contract_sha256")
            == expected_camera_contract_sha256
            and all(
                SHA256_PATTERN.fullmatch(value)
                for value in (
                    expected_package_config_sha256,
                    expected_edm_bundle_sha256,
                    expected_tracking_bundle_sha256,
                    expected_baseline_sha256,
                    expected_camera_contract_sha256,
                )
            )
        )
    scope_cardinality_ok = all(len(scopes[scope]) == 1 for scope in REQUIRED_SCOPES)
    semantics = {
        "independent_video_count": len(identities),
        "diagnostic_segment_count": sum(
            len(scopes[scope]) for scope in REQUIRED_SCOPES if scope != "full"
        ),
        "diagnostic_segments_are_independent_flights": False,
        "source_video_id": expected_video_id,
        "source_video_sha256": expected_video_sha256,
    }
    return bool(rows_ok and scope_cardinality_ok and len(identities) == 1), semantics


def evaluate_results(
    results: list[dict],
    *,
    reverse_sequences: set[str],
    expected_video_id: str,
    expected_video_sha256: str,
    expected_total_frames: int,
    expected_profile_sha256: str,
    expected_package_config_sha256: str,
    expected_edm_bundle_sha256: str,
    expected_tracking_bundle_sha256: str,
    expected_baseline_sha256: str,
    expected_baseline_median: float,
    expected_camera_contract_sha256: str,
) -> dict:
    semantics_ok, validation_semantics = _validation_semantics(
        results,
        expected_video_id=expected_video_id,
        expected_video_sha256=expected_video_sha256,
        expected_total_frames=expected_total_frames,
        expected_profile_sha256=expected_profile_sha256,
        expected_package_config_sha256=expected_package_config_sha256,
        expected_edm_bundle_sha256=expected_edm_bundle_sha256,
        expected_tracking_bundle_sha256=expected_tracking_bundle_sha256,
        expected_baseline_sha256=expected_baseline_sha256,
        expected_baseline_median=expected_baseline_median,
        expected_camera_contract_sha256=expected_camera_contract_sha256,
    )
    evidence = []
    for result in results:
        reference_counts = result.get("reference_sequence_counts", {})
        reference_total = sum(int(value) for value in reference_counts.values())
        reverse_total = sum(
            int(value)
            for sequence, value in reference_counts.items()
            if sequence in reverse_sequences
        )
        reverse_fraction = reverse_total / max(1, reference_total)
        frames = int(result.get("frames", 0))
        rejected_jumps = int(result.get("rejections", {}).get("jump", 0))
        median = result.get("step_median")
        p95 = result.get("step_p95")
        continuity_ok = (
            median is not None
            and p95 is not None
            and float(median) > 0
            and float(p95) <= 10.0 * float(median)
            and int(result.get("jumps_gt_10x_median", -1)) == 0
        )
        evidence.append(
            {
                "video": result.get("video"),
                "evaluation": result.get("evaluation"),
                "frames": frames,
                "localized": int(result.get("localized", 0)),
                "rate": float(result.get("rate", 0.0)),
                "inliers_p05": result.get("inliers_p05"),
                "continuity_ok": continuity_ok,
                "step_median": median,
                "step_p95": p95,
                "jumps_gt_10x_median": result.get("jumps_gt_10x_median"),
                "rejected_jumps": rejected_jumps,
                "rejected_jump_fraction": rejected_jumps / max(1, frames),
                "reverse_reference_fraction": reverse_fraction,
                "reference_sequence_counts": reference_counts,
            }
        )

    enough_runs = semantics_ok and len(evidence) == len(REQUIRED_SCOPES)
    all_rates = enough_runs and all(item["rate"] >= 0.95 for item in evidence)
    all_continuous = enough_runs and all(item["continuity_ok"] for item in evidence)
    reverse_supported = all_rates and any(
        item["reverse_reference_fraction"] >= 0.50 for item in evidence
    )
    inliers_supported = enough_runs and all(
        item["inliers_p05"] is not None and float(item["inliers_p05"]) >= 30.0
        for item in evidence
    )
    no_ghost_teleports = enough_runs and all(
        int(item["jumps_gt_10x_median"] or 0) == 0
        and item["rejected_jump_fraction"] <= 0.005
        for item in evidence
    )
    return {
        "stage": "S9_heldout_localization",
        "checks": {
            "G9.0": semantics_ok,
            "G9.1": all_rates,
            "G9.2": all_continuous,
            "G9.3": reverse_supported,
            "G9.4": inliers_supported,
            "G9.5": no_ghost_teleports,
        },
        "runs": evidence,
        "validation_semantics": validation_semantics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--forced-manifest", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--package-config", type=Path, required=True)
    parser.add_argument("--tracking-bundle", type=Path, required=True)
    parser.add_argument(
        "--baseline-bundle", type=Path, default=EXPECTED_BASELINE_PATH
    )
    parser.add_argument("--predecessor-gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.result]
    forced = json.loads(args.forced_manifest.read_text(encoding="utf-8"))
    heldout = heldout_contract_from_manifest(args.corpus_manifest)
    corpus_manifest_sha256 = _file_sha256(args.corpus_manifest)
    package_config = json.loads(args.package_config.read_text(encoding="utf-8"))
    if package_config.get("schema") != "edm-localization/v2":
        raise ValueError("Fuhe S9 requires edm-localization/v2 package config")
    live = package_config.get("live_camera_contract", {})
    if (
        live.get("validated") is not False
        or live.get("status") != "UNVALIDATED"
        or live.get("production_claim_allowed") is not False
    ):
        raise ValueError("live camera contract must remain UNVALIDATED")
    expected_camera = {
        "model": "PINHOLE",
        "width": 1280,
        "height": 720,
        "params": [931.2057783503648, 931.2057783503648, 640.0, 360.0],
    }
    benchmark_camera = package_config.get("benchmark_camera_contract", {})
    if (
        benchmark_camera
        != {
            "source": "P1130113.MP4",
            "input_size": [3840, 2160],
            "resize": {
                "size": [1280, 720],
                "interpolation": "cv2.INTER_AREA",
                "crop": None,
                "eis": False,
            },
            "pnp_camera": expected_camera,
        }
        or package_config.get("pnp_camera") != expected_camera
    ):
        raise ValueError("Fuhe benchmark_camera_contract is stale")

    package_root = args.package_config.parent.resolve(strict=True)
    package_bundle = (
        package_root / str(package_config.get("paths", {}).get("bundle", ""))
    ).resolve(strict=True)
    package_manifest = (package_root / "MANIFEST.sha256").resolve(strict=True)
    config_sha256 = _file_sha256(args.package_config)
    edm_bundle_sha256 = _file_sha256(package_bundle)
    tracking_bundle_sha256 = _file_sha256(args.tracking_bundle)
    baseline_path = args.baseline_bundle.expanduser().resolve(strict=True)
    baseline_sha256 = _file_sha256(baseline_path)
    if (
        baseline_path != EXPECTED_BASELINE_PATH.resolve(strict=True)
        or baseline_sha256 != EXPECTED_BASELINE_SHA256
    ):
        raise ValueError("anchor-density baseline path or SHA-256 was replaced")

    import torch

    baseline_payload = torch.load(
        baseline_path, map_location="cpu", weights_only=False
    )
    baseline_median = float(
        baseline_payload["meta"]["median_3d_anchored_per_ref"]
    )
    if baseline_median != EXPECTED_BASELINE_MEDIAN:
        raise ValueError("anchor-density baseline median is not the predeclared 3811.5")

    source_provenance = package_config.get("source_provenance", {})
    expected_source_values = {
        "tracking_bundle_sha256": tracking_bundle_sha256,
        "bundle_source_tracking_bundle_sha256": tracking_bundle_sha256,
        "edm_bundle_sha256": edm_bundle_sha256,
    }
    stale_source_fields = sorted(
        field
        for field, expected in expected_source_values.items()
        if source_provenance.get(field) != expected
    )
    baseline_record = source_provenance.get("anchor_density_baseline", {})
    if (
        baseline_record.get("path") != "outputs/river_site_reloc_map_edm.pt"
        or baseline_record.get("sha256") != baseline_sha256
        or float(
            baseline_record.get("median_3d_anchored_per_ref", float("nan"))
        )
        != baseline_median
    ):
        stale_source_fields.append("anchor_density_baseline")
    if stale_source_fields:
        raise ValueError(f"package source provenance is stale: {stale_source_fields}")

    expected_camera_contract_sha256 = _canonical_json_sha256(benchmark_camera)
    expected_profile_sha256 = str(
        package_config.get("deployment_profile", {}).get("sha256", "")
    )
    if SHA256_PATTERN.fullmatch(expected_profile_sha256) is None:
        raise ValueError("package config lacks a deployment profile SHA-256")
    packaged_heldout = package_config.get("heldout_validation", {})
    expected_packaged_contract = {
        "source_video_id": heldout["video_id"],
        "source_video_sha256": heldout["video_sha256"],
        "source_total_frames": heldout["total_frames"],
        "corpus_manifest_sha256": corpus_manifest_sha256,
    }
    stale_fields = sorted(
        field
        for field, expected in expected_packaged_contract.items()
        if packaged_heldout.get(field) != expected
    )
    if stale_fields:
        raise ValueError(f"package held-out contract is stale: {stale_fields}")
    report = evaluate_results(
        results,
        reverse_sequences=set(forced["rev"]),
        expected_video_id=str(heldout["video_id"]),
        expected_video_sha256=str(heldout["video_sha256"]),
        expected_total_frames=int(heldout["total_frames"]),
        expected_profile_sha256=expected_profile_sha256,
        expected_package_config_sha256=config_sha256,
        expected_edm_bundle_sha256=edm_bundle_sha256,
        expected_tracking_bundle_sha256=tracking_bundle_sha256,
        expected_baseline_sha256=baseline_sha256,
        expected_baseline_median=baseline_median,
        expected_camera_contract_sha256=expected_camera_contract_sha256,
    )

    gate = Gate(
        "S9_heldout_localization",
        {"G9.predecessor", *report["checks"]},
        script_path=__file__,
        source_files=[
            Path(__file__).with_name("edm_gate_contract.py"),
            Path(__file__).with_name("validate_edm_bundle.py"),
        ],
        input_artifacts={
            **{
                f"benchmark_result_{index}": path
                for index, path in enumerate(args.result)
            },
            "forced_manifest": args.forced_manifest,
            "corpus_manifest": args.corpus_manifest,
            "package_config": args.package_config,
            "package_manifest": package_manifest,
            "package_edm_bundle": package_bundle,
            "tracking_bundle": args.tracking_bundle,
            "anchor_density_baseline": baseline_path,
        },
    )
    gate.record_predecessor_gate(
        "S8_edm_bundle",
        args.predecessor_gate,
        expected_stage="S8_edm_bundle",
    )
    predecessor_error = None
    try:
        require_fresh_v2_gate(
            args.predecessor_gate, expected_stage="S8_edm_bundle"
        )
    except (OSError, ValueError, RuntimeError) as error:
        predecessor_error = str(error)
    gate.check(
        "G9.predecessor",
        predecessor_error is None,
        "S8 predecessor is a fresh sfm-gate-v2 PASS",
        error=predecessor_error,
    )
    details = {
        "G9.0": "benchmark semantics and all package/result hashes are bound",
        "G9.1": "every required scope meets localization-rate acceptance",
        "G9.2": "every required scope has continuous trajectory steps",
        "G9.3": "reverse-direction reference support is demonstrated",
        "G9.4": "PnP inlier support meets acceptance",
        "G9.5": "no ghost teleport acceptance threshold is exceeded",
    }
    for gid, passed in report["checks"].items():
        gate.check(
            gid,
            passed,
            details[gid],
            runs=report["runs"],
            validation_semantics=report["validation_semantics"],
        )
    gate.write(args.out.parent.parent, output_path=args.out)


if __name__ == "__main__":
    main()
