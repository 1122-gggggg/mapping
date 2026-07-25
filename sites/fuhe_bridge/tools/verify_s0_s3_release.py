#!/usr/bin/env python3
"""Initialize and independently verify the Fuhe Bridge v2 S0--S3 chain."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from ts_common import (
    BUILD,
    PROBE_FRAME_COUNT,
    RUN_ID,
    SUBFOLDER_REGEX,
    TEST,
    WORKING_HEIGHT,
    WORKING_WIDTH,
    GLUEMAP_ENV,
    GLUEMAP_PY,
    GLUEMAP_REPO,
    RUNS,
    Gate,
    GateFreshnessError,
    assert_gate_fresh,
    hash_artifact,
    initialize_not_run_gate,
    read_json,
    required_check_ids,
    sha256,
    stage_material_artifacts,
    verify_predecessor_chain,
    write_json,
)
from ts_env import verify_pycolmap_runtime
from resource_guard import GLOBAL_HEAVY_LOCK


STAGES = (
    "S0_corpus",
    "S1_motion",
    "S2_extract",
    "S2b_intrinsics",
    "S3_pairs",
)
STAGE_SCRIPTS = {
    "S0_corpus": "s0_corpus_lock.py",
    "S1_motion": "s1_motion_scan.py",
    "S2_extract": "s2_extract.py",
    "S2b_intrinsics": "s2b_intrinsics_bakeoff.py",
    "S3_pairs": "s3_pairs.py",
}
RELEASE_REQUIRED_IDS = frozenset(
    {
        "release/runtime_lock",
        "release/source_lock",
        "release/stage_files",
        "release/no_archived_inputs",
        "release/predecessor_chain",
    }
    | {f"release/exact/{stage}" for stage in STAGES}
    | {f"release/semantic/{stage}" for stage in STAGES}
    | {f"release/artifacts/{stage}" for stage in STAGES}
    | {f"release/fresh/{stage}" for stage in STAGES}
)
ARCHIVE_TOKENS = ("/_backup_pre_", "/stale_gates/")
TOOLS = Path(__file__).resolve().parent
TS_COMMON = TOOLS / "ts_common.py"
TS_ENV = TOOLS / "ts_env.py"
EXPECTED_RUNTIME_PREFIX = Path(
    "/home/cihcilab/micromamba/envs/target-site-gluemap-run"
)
EXTERNAL_CAMERA_RECORD = Path(
    "/media/cihcilab/新增磁碟區/福和橋場域/fuhe_submaps/"
    "FUHE_BRIDGE_PROJECT_COMPLETE_RECORD.md"
).resolve()
EXTERNAL_CAMERA_RECORD_SHA256 = (
    "65b1b50dff22935711263ab9b546cbbe1dc0f2c3443782e83c3be7e4def03903"
)


def material_artifacts(stage: str, run_dir: Path) -> dict[str, Path]:
    """Return the exact material closure a stage must bind in its gate."""
    return stage_material_artifacts(stage, run_dir)


def _require(metrics: Mapping[str, Any], *names: str) -> None:
    missing = [name for name in names if name not in metrics]
    if missing:
        raise ValueError(f"missing typed metrics: {missing}")


def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"expected finite number, got {value!r}")
    return result


def _s3_g33_not_applicable(metrics: Mapping[str, Any]) -> bool:
    """Recompute the only valid natural-first G3.3 non-applicable state."""
    _require(
        metrics,
        "policy_mode",
        "force_gap_bridges",
        "retrieval_executed",
        "n_forced_pairs",
        "n_images",
        "conditional_pair_density",
        "minimum",
        "missing_sequence_pairs",
        "pending_sequence_pairs",
        "required_route_clusters",
        "minimum_normalized_separation",
    )
    pending = metrics["pending_sequence_pairs"]
    missing = metrics["missing_sequence_pairs"]
    retrieval_executed = metrics["retrieval_executed"]
    evidence_state_valid = (
        retrieval_executed is False and bool(pending) and missing == []
    ) or (retrieval_executed is True and pending == [] and missing == [])
    return (
        metrics["policy_mode"] == "natural_retrieval_first"
        and metrics["force_gap_bridges"] is False
        and type(metrics["n_forced_pairs"]) is int
        and metrics["n_forced_pairs"] == 0
        and metrics["n_images"] == PROBE_FRAME_COUNT
        and _finite(metrics["conditional_pair_density"]) == 0.0
        and _finite(metrics["minimum"]) == 4.0
        and metrics["required_route_clusters"] == 2
        and _finite(metrics["minimum_normalized_separation"]) == 0.25
        and evidence_state_valid
    )


def _s2b_camera_policy_valid(policy: Mapping[str, Any]) -> bool:
    camera_policy = policy.get("camera_policy", {})
    return (
        policy.get("schema_version") == "fuhe-intrinsics-policy-v2"
        and isinstance(camera_policy, dict)
        and camera_policy.get("state") == "PASS"
        and camera_policy.get("applicable") is True
        and camera_policy.get("decision") == "official69_fixed_pinhole"
        and camera_policy.get("external_record")
        == {
            "path": str(EXTERNAL_CAMERA_RECORD),
            "sha256": EXTERNAL_CAMERA_RECORD_SHA256,
        }
        and camera_policy.get("locked_external_record_sha256")
        == EXTERNAL_CAMERA_RECORD_SHA256
        and camera_policy.get("camera")
        == {
            "model": "PINHOLE",
            "width": 1920,
            "height": 1080,
            "params": [
                1396.8086675255472,
                1396.8086675255472,
                960.0,
                540.0,
            ],
        }
        and camera_policy.get("official_hfov_deg") == 69.0
        and camera_policy.get("resize") == "raw INTER_AREA to 1920x1080"
        and camera_policy.get("undistort") is False
        and camera_policy.get("fixed_intrinsics_ba") is True
    )


def _s2b_not_applicable(gid: str, metrics: Mapping[str, Any]) -> bool:
    expected = {
        "G2.7/1920x1080": ("two_seed", {"resolution": "1920x1080"}),
        "G2.8": (
            "cross_resolution",
            {"working_resolutions": ["1920x1080"]},
        ),
    }
    if gid not in expected:
        return False
    diagnostic, fields = expected[gid]
    return (
        metrics.get("diagnostic") == diagnostic
        and metrics.get("external_record_sha256")
        == EXTERNAL_CAMERA_RECORD_SHA256
        and all(metrics.get(name) == value for name, value in fields.items())
    )


EXACT_NOT_APPLICABLE_IDS: dict[str, frozenset[str]] = {
    "S2b_intrinsics": frozenset({"G2.7/1920x1080", "G2.8"}),
    "S3_pairs": frozenset({"G3.3"}),
}


def _exact_not_applicable_check(stage: str, check: Mapping[str, Any]) -> bool:
    """Accept only the three typed, evidence-complete release N/A predicates."""
    gid = check.get("id")
    metrics = check.get("metrics")
    evidence = check.get("evidence")
    reason = check.get("reason")
    if (
        gid not in EXACT_NOT_APPLICABLE_IDS.get(stage, frozenset())
        or check.get("state") != "NOT_APPLICABLE"
        or check.get("ok") is not False
        or check.get("applicable") is not False
        or not isinstance(reason, str)
        or not reason.strip()
        or not isinstance(metrics, dict)
        or not metrics
        or not isinstance(evidence, dict)
        or evidence != metrics
    ):
        return False
    try:
        if stage == "S2b_intrinsics":
            return _s2b_not_applicable(str(gid), metrics)
        if stage == "S3_pairs" and gid == "G3.3":
            return _s3_g33_not_applicable(metrics)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return False


def _exact_stage_gate_pass(stage: str, payload: Mapping[str, Any]) -> bool:
    """Validate the exact ID/state contract, including narrowly approved N/A."""
    expected_ids = required_check_ids(stage)
    required_ids = payload.get("required_ids", [])
    stage_checks = payload.get("checks", [])
    if not isinstance(stage_checks, list) or not all(
        isinstance(item, dict) for item in stage_checks
    ):
        return False
    emitted_ids = [item.get("id") for item in stage_checks]
    check_states_valid = all(
        (
            item.get("state") == "PASS"
            and item.get("ok") is True
        )
        or _exact_not_applicable_check(stage, item)
        for item in stage_checks
    )
    return (
        payload.get("stage") == stage
        and payload.get("status") == "PASS"
        and payload.get("ok") is True
        and isinstance(required_ids, list)
        and set(required_ids) == expected_ids
        and len(required_ids) == len(expected_ids)
        and set(emitted_ids) == expected_ids
        and len(emitted_ids) == len(expected_ids)
        and check_states_valid
    )


def _runtime_evidence_ok(runtime: Any) -> bool:
    if not isinstance(runtime, dict):
        return False
    try:
        module_path = Path(runtime["module_path"]).resolve()
        executable = Path(runtime["python_executable"]).resolve()
        prefix = Path(runtime["sys_prefix"]).resolve()
        required = runtime["required_apis"]
        forbidden = runtime["forbidden_apis"]
    except (KeyError, TypeError, OSError):
        return False
    return (
        runtime.get("version") == "4.0.4"
        and runtime.get("providers")
        == [{"name": "pycolmap-cuda12", "version": "4.0.4"}]
        and prefix == EXPECTED_RUNTIME_PREFIX.resolve()
        and module_path.is_relative_to(prefix)
        and executable.is_relative_to(prefix)
        and isinstance(required, dict)
        and bool(required)
        and all(value is True for value in required.values())
        and isinstance(forbidden, dict)
        and bool(forbidden)
        and all(value is False for value in forbidden.values())
    )


def _typed_predicate(stage: str, gid: str, metrics: Mapping[str, Any]) -> bool:
    """Recompute one gate predicate solely from its persisted typed evidence."""
    if stage == "S0_corpus":
        if gid == "G0.1a":
            _require(metrics, "n_build", "mismatched")
            return metrics["n_build"] == len(BUILD) and metrics["mismatched"] == []
        if gid == "G0.1b":
            _require(metrics, "n_hashed", "n_unique")
            expected = len(BUILD) + len(TEST)
            return metrics["n_hashed"] == expected and metrics["n_unique"] == expected
        if gid == "G0.2a":
            _require(metrics, "build_paths", "test_paths", "path_overlap")
            return (
                len(metrics["build_paths"]) == len(BUILD)
                and len(metrics["test_paths"]) == len(TEST)
                and metrics["path_overlap"] == []
            )
        if gid == "G0.2b":
            _require(metrics, "n_build_hashes", "n_test_hashes", "hash_overlap")
            return (
                metrics["n_build_hashes"] == len(BUILD)
                and metrics["n_test_hashes"] == len(TEST)
                and metrics["hash_overlap"] == []
            )
        if gid == "G0.3":
            _require(metrics, "worst_aspect_err")
            return _finite(metrics["worst_aspect_err"]) <= 1e-6
        if gid == "G0.4":
            _require(metrics, "dropped")
            return metrics["dropped"] == []
        if gid == "G0.5":
            _require(metrics, "heldout_sequences", "heldout_paths", "sequence_overlap", "path_overlap")
            return (
                metrics["heldout_sequences"] == sorted(video.seq for video in TEST)
                and metrics["heldout_paths"] == sorted(video.rel for video in TEST)
                and metrics["sequence_overlap"] == []
                and metrics["path_overlap"] == []
            )
        if gid == "G0.6":
            _require(metrics, "regex")
            return metrics["regex"] == SUBFOLDER_REGEX
        if gid == "G0.7":
            _require(metrics, "source_groups", "working_groups")
            return metrics["working_groups"] == {
                f"{WORKING_WIDTH}x{WORKING_HEIGHT}": len(BUILD)
            }
        if gid == "G0.8":
            _require(metrics, "directions", "unknown")
            directions = metrics["directions"]
            return (
                isinstance(directions, dict)
                and len(directions) == len(BUILD)
                and set(directions.values()) == {"fwd", "rev"}
                and sorted(metrics["unknown"])
                == sorted(key for key, value in directions.items() if value == "unknown")
            )

    if stage == "S1_motion":
        if gid == "G0.2":
            _require(
                metrics,
                "expected_sources",
                "observed_sources",
                "expected_test_sources",
                "observed_test_sources",
                "hashes_complete",
                "source_overlap",
                "hash_overlap",
            )
            return (
                metrics["expected_sources"] == metrics["observed_sources"]
                and metrics["expected_test_sources"] == metrics["observed_test_sources"]
                and metrics["hashes_complete"] is True
                and metrics["source_overlap"] == []
                and metrics["hash_overlap"] == []
            )
        if gid.startswith("G1.1/"):
            _require(metrics, "expected", "actual", "coverage")
            expected = int(metrics["expected"])
            actual = int(metrics["actual"])
            coverage = _finite(metrics["coverage"])
            return expected > 0 and abs(coverage - actual / expected) <= 1e-12 and coverage >= 0.99
        if gid.startswith("G1.2/"):
            _require(metrics, "parallax", "low_parallax")
            return _finite(metrics["parallax"]) + _finite(metrics["low_parallax"]) >= 0.65
        if gid == "G1.4a":
            _require(metrics, "resolved", "unresolved")
            return len(metrics["resolved"]) == len(BUILD) and metrics["unresolved"] == []
        if gid == "G1.4b":
            _require(metrics, "conflicts")
            return metrics["conflicts"] == []
        if gid == "G1.4c":
            _require(metrics, "resolved")
            values = list(metrics["resolved"].values())
            return (
                len(values) == len(BUILD)
                and values.count("fwd") == sum(video.direction == "fwd" for video in BUILD)
                and values.count("rev") == sum(video.direction == "rev" for video in BUILD)
            )
        if gid == "G1.5":
            _require(metrics, "finite_records")
            return (
                set(metrics["finite_records"]) == {video.seq for video in BUILD}
                and all(int(value) > 0 for value in metrics["finite_records"].values())
            )
        if gid == "G1.6":
            _require(metrics, "n_records", "n_pure", "n_violations", "violations")
            return (
                metrics["n_records"] > 0
                and 0 <= metrics["n_pure"] <= metrics["n_records"]
                and metrics["n_violations"] == len(metrics["violations"]) == 0
            )

    if stage == "S2_extract":
        if gid == "G0.2":
            _require(metrics, "expected_lineage", "actual_lineage", "forbidden_lineage", "missing_on_disk", "unmanifested_on_disk")
            return (
                len(metrics["expected_lineage"]) == len(BUILD)
                and metrics["expected_lineage"] == metrics["actual_lineage"]
                and metrics["forbidden_lineage"] == []
                and metrics["missing_on_disk"] == []
                and metrics["unmanifested_on_disk"] == []
            )
        if gid.startswith("G1.3/"):
            _require(metrics, "n_frames", "expected_frames", "n_hover", "hover_ratio", "max_hover_ratio")
            n_frames = int(metrics["n_frames"])
            n_hover = int(metrics["n_hover"])
            ratio = _finite(metrics["hover_ratio"])
            return (
                n_frames == int(metrics["expected_frames"])
                and n_frames > 0
                and abs(ratio - n_hover / n_frames) <= 1e-12
                and ratio <= _finite(metrics["max_hover_ratio"])
            )
        if gid == "G2.1":
            _require(metrics, "samples")
            samples = metrics["samples"]
            return bool(samples) and all(
                _finite(row["mad_vs_raw"]) < 0.35 * _finite(row["mad_vs_undistorted"])
                for row in samples
            )
        if gid == "G2.2":
            _require(metrics, "n_frames", "bad_dimensions", "unreadable", "hash_mismatches")
            return metrics["n_frames"] == PROBE_FRAME_COUNT and all(metrics[name] == [] for name in ("bad_dimensions", "unreadable", "hash_mismatches"))
        if gid == "G2.3a":
            _require(metrics, "shapes")
            return metrics["shapes"] == [f"{WORKING_WIDTH}x{WORKING_HEIGHT}"]
        if gid == "G2.4":
            _require(metrics, "n_frames", "expected_frames", "est_hours")
            return metrics["n_frames"] == metrics["expected_frames"] == PROBE_FRAME_COUNT and _finite(metrics["est_hours"]) > 0
        if gid == "G2.5":
            _require(metrics, "roles", "bad_role_records")
            roles = metrics["roles"]
            return isinstance(roles, dict) and sum(roles.values()) == PROBE_FRAME_COUNT and set(roles) <= {"triangulation", "bridge_only"} and metrics["bad_role_records"] == []
        if gid == "G2.6":
            _require(metrics, "seeded", "missing_seed_names", "extra_seed_names", "seed_model_sha256")
            hashes = metrics["seed_model_sha256"]
            return (
                metrics["seeded"] == [f"{WORKING_WIDTH}x{WORKING_HEIGHT}"]
                and metrics["missing_seed_names"] == []
                and metrics["extra_seed_names"] == []
                and set(hashes) == {"cameras.txt", "images.txt", "points3D.txt"}
                and all(isinstance(value, str) and len(value) == 64 for value in hashes.values())
            )

    if stage == "S2b_intrinsics":
        if gid == "G2.7/results_complete":
            _require(metrics, "policy")
            policy = metrics["policy"]
            return (
                isinstance(policy, dict)
                and _s2b_camera_policy_valid(policy)
                and EXTERNAL_CAMERA_RECORD.is_file()
                and sha256(EXTERNAL_CAMERA_RECORD)
                == EXTERNAL_CAMERA_RECORD_SHA256
            )

    if stage == "S3_pairs":
        if gid == "G0.2":
            _require(metrics, "expected_lineage", "actual_lineage", "forbidden_lineage")
            return len(metrics["expected_lineage"]) == len(BUILD) and metrics["expected_lineage"] == metrics["actual_lineage"] and metrics["forbidden_lineage"] == []
        if gid == "G3.0":
            _require(metrics, "runtime_fingerprint")
            return _runtime_evidence_ok(metrics["runtime_fingerprint"])
        if gid == "G3.1":
            _require(metrics, "n_camera_slots", "n_non_null")
            return metrics["n_camera_slots"] == metrics["n_non_null"] == 1
        if gid == "G3.2":
            _require(metrics, "n_images", "n_unique_images", "missing_seed_names", "dimension_mismatches")
            return metrics["n_images"] == metrics["n_unique_images"] == PROBE_FRAME_COUNT and metrics["missing_seed_names"] == [] and metrics["dimension_mismatches"] == []
        if gid == "G3.3":
            _require(
                metrics,
                "policy_mode",
                "force_gap_bridges",
                "retrieval_executed",
                "n_forced_pairs",
                "n_images",
                "conditional_pair_density",
                "minimum",
                "missing_sequence_pairs",
                "pending_sequence_pairs",
                "required_route_clusters",
                "minimum_normalized_separation",
                "cluster_policy_valid",
            )
            density = metrics["n_forced_pairs"] / metrics["n_images"]
            return (
                metrics["policy_mode"] == "natural_retrieval_first"
                and metrics["force_gap_bridges"] is True
                and metrics["retrieval_executed"] is True
                and metrics["cluster_policy_valid"] is True
                and metrics["n_images"] == PROBE_FRAME_COUNT
                and bool(metrics["missing_sequence_pairs"])
                and metrics["pending_sequence_pairs"] == []
                and metrics["required_route_clusters"] == 2
                and _finite(metrics["minimum_normalized_separation"]) == 0.25
                and abs(density - _finite(metrics["conditional_pair_density"]))
                <= 1e-12
                and density >= _finite(metrics["minimum"])
            )
        if gid == "G3.4":
            _require(metrics, "expected", "discovered")
            return len(metrics["expected"]) == len(BUILD) and metrics["expected"] == metrics["discovered"]
        if gid == "G3.5a":
            _require(metrics, "fwd", "rev", "expected")
            return len(metrics["fwd"]) == sum(video.direction == "fwd" for video in BUILD) and len(metrics["rev"]) == sum(video.direction == "rev" for video in BUILD) and set(metrics["fwd"]).isdisjoint(metrics["rev"]) and set(metrics["fwd"]) | set(metrics["rev"]) == set(metrics["expected"])
        if gid == "G3.5b":
            _require(metrics, "n_pairs", "n_parsed", "n_sequence_pairs", "max_pairs", "per_video_pair", "policy")
            per_pair = metrics["per_video_pair"]
            policy = metrics["policy"]
            return metrics["n_pairs"] == metrics["n_parsed"] and metrics["n_sequence_pairs"] == len(per_pair) == 6 and metrics["n_pairs"] <= metrics["max_pairs"] <= 12000 and sum(row["n_pairs"] for row in per_pair.values()) == metrics["n_pairs"] and policy.get("mode") == "natural_retrieval_first"
        if gid == "G3.5c":
            _require(metrics, "missing_manifest", "missing_disk")
            return metrics["missing_manifest"] == metrics["missing_disk"] == []
        if gid == "G3.5d":
            _require(metrics, "images_path", "expected", "injected", "missing_pairs", "missing_endpoints")
            return Path(metrics["images_path"]).is_absolute() and metrics["expected"] == metrics["injected"] and metrics["missing_pairs"] == 0 and metrics["missing_endpoints"] == []
        if gid == "G3.6":
            _require(
                metrics,
                "config",
                "extra_pairs_path",
                "forced_pairs_sha256",
                "parsed_pairs",
                "injected_pairs",
                "refine_intrinsics",
                "skip_doppelgangers",
                "num_workers",
                "num_track_per_img",
                "sift_max_num_features",
                "sift_max_num_orientations",
                "sift_max_rows_per_image",
                "max_num_tracks",
                "memory_safe_launcher",
                "rerun_from",
                "resource_lock_path",
                "resource_guard_path",
                "resource_guard_log_path",
            )
            config_path = Path(metrics["config"]).resolve()
            run_dir = config_path.parent
            lock_path = Path(metrics["resource_lock_path"]).resolve()
            log_path = Path(metrics["resource_guard_log_path"]).resolve()
            return (
                Path(metrics["config"]).is_absolute()
                and Path(metrics["extra_pairs_path"]).is_absolute()
                and Path(metrics["memory_safe_launcher"]).is_absolute()
                and Path(metrics["memory_safe_launcher"]).is_file()
                and isinstance(metrics["forced_pairs_sha256"], str)
                and len(metrics["forced_pairs_sha256"]) == 64
                and metrics["parsed_pairs"] == metrics["injected_pairs"]
                and metrics["refine_intrinsics"] is False
                and metrics["skip_doppelgangers"] is False
                and type(metrics["num_workers"]) is int
                and metrics["num_workers"] == 0
                and metrics["num_track_per_img"] == 512
                and metrics["sift_max_num_features"] == 2048
                and type(metrics["sift_max_num_orientations"]) is int
                and metrics["sift_max_num_orientations"] == 1
                and metrics["sift_max_rows_per_image"] == 2048
                and metrics["max_num_tracks"] == 400000
                and metrics["rerun_from"] is None
                and lock_path == GLOBAL_HEAVY_LOCK
                and not lock_path.is_relative_to(run_dir)
                and Path(metrics["resource_guard_path"]).is_file()
                and log_path.is_relative_to(run_dir)
            )

    raise ValueError(f"no typed predicate for {stage}/{gid}")


def validate_typed_stage_evidence(stage: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reject literal-True chains and recompute every exact required predicate."""
    by_id = {item.get("id"): item for item in payload.get("checks", []) if isinstance(item, dict)}
    errors: dict[str, str] = {}
    for gid in sorted(required_check_ids(stage)):
        check = by_id.get(gid)
        if check is None:
            errors[gid] = "required check is absent"
            continue
        try:
            if check.get("state") == "NOT_APPLICABLE":
                if (
                    stage == "S2b_intrinsics"
                    and isinstance(check.get("metrics"), dict)
                    and _s2b_not_applicable(gid, check["metrics"])
                    and check.get("applicable") is False
                    and isinstance(check.get("reason"), str)
                    and bool(check["reason"].strip())
                ):
                    continue
                if (
                    stage == "S3_pairs"
                    and gid == "G3.3"
                    and check.get("applicable") is False
                    and isinstance(check.get("reason"), str)
                    and bool(check["reason"].strip())
                    and isinstance(check.get("metrics"), dict)
                    and _s3_g33_not_applicable(check["metrics"])
                ):
                    continue
                raise ValueError("NOT_APPLICABLE is not allowed for this predicate")
            metrics = check.get("metrics")
            if not isinstance(metrics, dict) or not metrics:
                raise ValueError("metrics must be a non-empty object")
            predicate = _typed_predicate(stage, gid, metrics)
            if predicate is not True:
                raise ValueError("typed predicate recomputed false")
            if check.get("state") != "PASS" or check.get("ok") is not True:
                raise ValueError("persisted state is not PASS")
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            errors[gid] = str(exc)
    return _check(
        f"release/semantic/{stage}",
        not errors,
        "every required typed predicate recomputed true" if not errors else "typed evidence is missing, malformed, or false",
        n_required=len(required_check_ids(stage)),
        n_recomputed=len(required_check_ids(stage)) - len(errors),
        errors=errors,
    )


def _check(check_id: str, ok: bool, detail: str, **metrics: Any) -> dict[str, Any]:
    evidence = metrics or {"predicate": bool(ok)}
    return {
        "id": check_id,
        "state": "PASS" if ok else "FAIL",
        "ok": bool(ok),
        "detail": detail,
        "metrics": evidence,
        "evidence": evidence,
    }


def validate_s2_image_corpus(run_dir: Path) -> dict[str, Any]:
    """Re-read the complete image corpus against frame_manifest set and hashes."""
    try:
        manifest = read_json(Path(run_dir) / "frame_manifest.json")
        frames = manifest.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError("frame_manifest.frames must be a non-empty list")
        names = [frame.get("name") for frame in frames if isinstance(frame, dict)]
        if len(names) != len(frames) or any(
            not isinstance(name, str) or not name or Path(name).is_absolute()
            for name in names
        ):
            raise ValueError("frame names must be non-empty relative strings")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        expected = set(names)
        images_dir = Path(run_dir) / "images"
        actual = {
            path.relative_to(images_dir).as_posix()
            for path in images_dir.rglob("*.jpg")
            if path.is_file()
        } if images_dir.is_dir() else set()
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        hash_mismatches: list[str] = []
        malformed_hashes: list[str] = []
        for frame in frames:
            name = frame["name"]
            recorded = frame.get("image_sha256")
            if not isinstance(recorded, str) or len(recorded) != 64:
                malformed_hashes.append(name)
                continue
            path = images_dir / name
            if path.is_file() and sha256(path) != recorded:
                hash_mismatches.append(name)
        ok = (
            not duplicates
            and not missing
            and not unexpected
            and not hash_mismatches
            and not malformed_hashes
            and manifest.get("n_frames") == len(frames)
        )
        return _check(
            "release/material_semantic/S2_image_corpus",
            ok,
            "frame manifest and complete JPEG corpus sets/hashes agree"
            if ok
            else "frame manifest and JPEG corpus differ",
            n_manifest=len(frames),
            n_disk=len(actual),
            duplicates=duplicates,
            missing=missing,
            unexpected=unexpected,
            hash_mismatches=hash_mismatches,
            malformed_hashes=malformed_hashes,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        return _check(
            "release/material_semantic/S2_image_corpus",
            False,
            str(exc),
            missing=["frame_manifest_or_images"],
            unexpected=[],
            hash_mismatches=[],
            malformed_hashes=[],
        )


def validate_intrinsics_seed(run_dir: Path) -> dict[str, Any]:
    """Reopen the persisted seed and bind its cameras/images/file hashes."""
    try:
        import pycolmap

        root = Path(run_dir)
        manifest = read_json(root / "frame_manifest.json")
        seed_manifest = read_json(root / "intrinsics_seed.json")
        seed_dir = root / "intrinsics_seed"
        expected_names = {frame["name"] for frame in manifest["frames"]}
        file_drift = []
        recorded_hashes = seed_manifest.get("model_files_sha256", {})
        for name in ("cameras.txt", "images.txt", "points3D.txt"):
            path = seed_dir / name
            if not path.is_file() or recorded_hashes.get(name) != sha256(path):
                file_drift.append(name)
        reconstruction = pycolmap.Reconstruction(str(seed_dir))
        actual_names = {image.name for image in reconstruction.images.values()}
        shapes = {
            (int(camera.width), int(camera.height))
            for camera in reconstruction.cameras.values()
        }
        camera_models = {
            camera.model_name for camera in reconstruction.cameras.values()
        }
        ok = (
            not file_drift
            and len(reconstruction.cameras) == seed_manifest.get("n_cameras") == 1
            and len(reconstruction.images) == seed_manifest.get("n_images") == len(expected_names)
            and actual_names == expected_names
            and shapes == {(WORKING_WIDTH, WORKING_HEIGHT)}
            and camera_models == {"PINHOLE"}
            and Path(seed_manifest.get("seed_dir", "")).resolve() == seed_dir.resolve()
        )
        return _check(
            "release/material_semantic/intrinsics_seed",
            ok,
            "seed reopens as one PINHOLE camera over the exact frame set"
            if ok
            else "seed model, manifest, or frame coverage differs",
            n_cameras=len(reconstruction.cameras),
            n_images=len(reconstruction.images),
            missing_names=sorted(expected_names - actual_names),
            unexpected_names=sorted(actual_names - expected_names),
            shapes=[f"{w}x{h}" for w, h in sorted(shapes)],
            camera_models=sorted(camera_models),
            file_drift=file_drift,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        return _check(
            "release/material_semantic/intrinsics_seed",
            False,
            str(exc),
            n_cameras=0,
            n_images=0,
            missing_names=["seed_or_manifest"],
            unexpected_names=[],
            shapes=[],
            camera_models=[],
            file_drift=[],
        )


def _rehash_generating_provenance(
    result: Mapping[str, Any], cache: dict[str, dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    provenance = result.get("generating_provenance")
    if not isinstance(provenance, dict) or provenance.get("mode") != "fresh_solve":
        return ["fresh_solve generating provenance is absent"]
    if "imported_from" in result:
        errors.append("imported_from is forbidden")
    if provenance.get("runtime_fingerprint") != result.get("runtime_fingerprint"):
        errors.append("generating/runtime fingerprint mismatch")
    if provenance.get("scientific_config") != result.get("scientific_config"):
        errors.append("generating/scientific config mismatch")
    for bucket_name in ("sources", "checkpoints"):
        bucket = provenance.get(bucket_name)
        if not isinstance(bucket, dict) or not bucket:
            errors.append(f"{bucket_name} bundle is empty")
            continue
        for label, record in bucket.items():
            if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
                errors.append(f"{bucket_name}/{label}: malformed artifact record")
                continue
            path = str(Path(record["path"]).resolve())
            if path not in cache:
                cache[path] = hash_artifact(path)
            actual = cache[path]
            if actual.get("sha256") != record.get("sha256"):
                errors.append(f"{bucket_name}/{label}: SHA-256 drift")
    git_record = provenance.get("gluemap_git")
    if not isinstance(git_record, dict):
        errors.append("generating gluemap_git record is absent")
    else:
        current = _git_source_state()
        current_porcelain = [
            f"{item['status']} {item['path']}" for item in current["dirty"]
        ]
        if git_record.get("commit") != current["commit"]:
            errors.append("generating gluemap commit drift")
        if git_record.get("porcelain_v1") != current_porcelain:
            errors.append("generating gluemap porcelain drift")
    return errors


def validate_s2b_semantics(run_dir: Path) -> dict[str, Any]:
    """Validate the external, hash-bound official69 frozen-camera decision."""
    root = Path(run_dir)
    try:
        policy = read_json(root / "intrinsics_policy.json")
        errors: list[str] = []
        if not _s2b_camera_policy_valid(policy):
            errors.append("external official69 camera policy differs")
        if not EXTERNAL_CAMERA_RECORD.is_file():
            errors.append("external camera record is absent")
        elif sha256(EXTERNAL_CAMERA_RECORD) != EXTERNAL_CAMERA_RECORD_SHA256:
            errors.append("external camera record SHA-256 drift")
        diagnostics = policy.get("diagnostics", {})
        expected_diagnostics = {
            "two_seed": "camera is externally frozen; seed convergence is not estimated",
            "cross_resolution": "adapter has one 1920x1080 working resolution",
        }
        for name, reason in expected_diagnostics.items():
            if diagnostics.get(name) != {
                "state": "NOT_APPLICABLE",
                "applicable": False,
                "reason": reason,
            }:
                errors.append(f"{name} diagnostic applicability differs")
        return _check(
            "release/material_semantic/S2b_intrinsics",
            not errors,
            "external official69 camera policy and diagnostic N/A states revalidate"
            if not errors
            else "external frozen-camera policy differs",
            policy=policy,
            errors=errors,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return _check(
            "release/material_semantic/S2b_intrinsics",
            False,
            str(exc),
            policy={},
            errors=[str(exc)],
        )

    # Historical v1 re-opener retained below for archived-artifact inspection.
    try:
        import pycolmap
        import s2b_intrinsics_bakeoff as s2b

        root = Path(run_dir)
        runtime = verify_pycolmap_runtime()
        probes = {
            shape: s2b.probe_input(root, s2b.shape_tuple(shape))
            for shape in s2b.REQUIRED_SHAPES
        }
        probe_hashes = {
            shape: probe["probe_input_sha256"] for shape, probe in probes.items()
        }
        results, statuses = s2b.load_result_markers(
            root / "intrinsics_bakeoff",
            validate=True,
            expected_probe_hashes=probe_hashes,
        )
        errors: list[str] = []
        cache: dict[str, dict[str, Any]] = {}
        recomputed: dict[str, dict[str, Any]] = {}
        if set(results) != s2b.REQUIRED_RESULT_KEYS or set(statuses) != s2b.REQUIRED_RESULT_KEYS:
            errors.append("result/status logical key set is not the exact six")
        for key in sorted(s2b.REQUIRED_RESULT_KEYS):
            result = results.get(key)
            if result is None:
                continue
            work = root / "intrinsics_bakeoff" / f"{key[0]}__{key[1]}"
            model = work / "gluemap" / "gluemap_aba"
            if Path(result.get("model_path", "")).resolve() != model.resolve():
                errors.append(f"{s2b.keystr(key)}: model path escapes solve tree")
                continue
            errors.extend(
                f"{s2b.keystr(key)}: {error}"
                for error in _rehash_generating_provenance(result, cache)
            )
            if result.get("runtime_fingerprint") != runtime:
                errors.append(f"{s2b.keystr(key)}: generating runtime is not current lock")
            reconstruction = pycolmap.Reconstruction(str(model))
            expected_names = {row["name"] for row in probes[key[0]]["frames"]}
            actual_names = {image.name for image in reconstruction.images.values()}
            if len(reconstruction.cameras) != 1 or actual_names != expected_names:
                errors.append(f"{s2b.keystr(key)}: model completeness differs")
                continue
            camera = next(iter(reconstruction.cameras.values()))
            mean_reproj = float(reconstruction.compute_mean_reprojection_error())
            fx_over_w = float(camera.params[0]) / int(camera.width)
            observed = {
                "camera_model": camera.model_name,
                "camera_count": len(reconstruction.cameras),
                "registered": int(reconstruction.num_reg_images()),
                "n_frames": len(expected_names),
                "points3D": int(reconstruction.num_points3D()),
                "mean_reproj": mean_reproj,
                "fx_over_w": fx_over_w,
            }
            recomputed[s2b.keystr(key)] = observed
            for field in ("camera_model", "camera_count", "registered", "n_frames", "points3D"):
                if result.get(field) != observed[field]:
                    errors.append(f"{s2b.keystr(key)}: {field} differs from reopened model")
            for field in ("mean_reproj", "fx_over_w"):
                if not math.isclose(float(result.get(field, math.nan)), observed[field], rel_tol=1e-10, abs_tol=1e-10):
                    errors.append(f"{s2b.keystr(key)}: {field} differs from reopened model")

        aggregate = read_json(root / "intrinsics_bakeoff.json")
        if aggregate.get("required_result_keys") != sorted(
            s2b.keystr(key) for key in s2b.REQUIRED_RESULT_KEYS
        ):
            errors.append("aggregate required_result_keys differs")
        if aggregate.get("runs") != [results[key] for key in sorted(results)]:
            errors.append("aggregate runs differ from result markers")
        if aggregate.get("statuses") != {
            s2b.keystr(key): statuses[key] for key in sorted(statuses)
        }:
            errors.append("aggregate statuses differ from status markers")
        replay_gate = Gate("S2b_intrinsics", s2b.S2B_REQUIRED_IDS)
        s2b.emit_aggregate_checks(
            replay_gate, results, statuses, runtime_fingerprint=runtime
        )
        non_pass = [
            check["id"] for check in replay_gate.checks if check["state"] != "PASS"
        ]
        if non_pass:
            errors.append(f"recomputed aggregate predicates are non-PASS: {non_pass}")
        return _check(
            "release/material_semantic/S2b_intrinsics",
            not errors,
            "historical diagnostic models and completeness predicates recomputed"
            if not errors
            else "calibration closure or recomputed predicates differ",
            n_results=len(results),
            n_statuses=len(statuses),
            n_models_reopened=len(recomputed),
            recomputed=recomputed,
            errors=errors,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        return _check(
            "release/material_semantic/S2b_intrinsics",
            False,
            str(exc),
            n_results=0,
            n_statuses=0,
            n_models_reopened=0,
            recomputed={},
            errors=[str(exc)],
        )


def validate_s3_semantics(run_dir: Path) -> dict[str, Any]:
    """Reparse S3 outputs and rerun the live loader independently."""
    try:
        import yaml
        import s3_pairs as s3

        root = Path(run_dir)
        forced_path = root / "forced_bridges.txt"
        config_path = root / "gluemap_config.yaml"
        manifest = read_json(root / "forced_bridges.json")
        persisted_probe = read_json(root / "s3_loader_probe.json")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        pairs = s3.parse_pair_file(forced_path)
        pair_set = {tuple(sorted(pair)) for pair in pairs}
        (root / "s3_loader_probe.log").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="fuhe_s3_verify_") as temp:
            replay = s3.real_loader_probe(root, Path(temp) / "probe.json")
        core_fields = (
            "datasets",
            "n_images",
            "n_unique_images",
            "n_gt_camera_slots",
            "n_non_null_gt_cameras",
            "n_dimension_matches",
            "missing_seed_names",
            "dimension_mismatches",
            "n_loader_pairs",
            "n_forced_expected",
            "n_forced_injected",
            "missing_forced_endpoints",
            "missing_forced_pairs",
        )
        replay_drift = {
            field: {"persisted": persisted_probe.get(field), "replayed": replay.get(field)}
            for field in core_fields
            if persisted_probe.get(field) != replay.get(field)
        }
        errors = []
        if len(pairs) != len(pair_set) or len(pair_set) > 12000:
            errors.append("conditional pair text is duplicated or exceeds the hard cap")
        if manifest.get("n_pairs") != len(pair_set) or manifest.get("forced_pairs_sha256") != sha256(forced_path):
            errors.append("forced manifest does not bind the conditional pair text")
        policy = manifest.get("policy", {})
        if policy.get("mode") != "natural_retrieval_first":
            errors.append("pair policy is not natural-retrieval-first")
        if manifest.get("config_sha256") != sha256(config_path):
            errors.append("forced manifest config hash differs")
        if Path(manifest.get("forced_pairs_path", "")).resolve() != forced_path.resolve():
            errors.append("forced manifest path differs")
        if Path(manifest.get("config_path", "")).resolve() != config_path.resolve():
            errors.append("forced manifest config path differs")
        if Path(config.get("extra_pairs_path", "")).resolve() != forced_path.resolve():
            errors.append("config extra_pairs_path differs")
        if config.get("refine_intrinsics") is not False or config.get("skip_doppelgangers") is not False:
            errors.append("production config intrinsics/Doppelgangers policy differs")
        if type(config.get("num_workers")) is not int or config["num_workers"] != 0:
            errors.append("production config num_workers must be literal integer zero")
        if config.get("num_track_per_img") != 512:
            errors.append("production config num_track_per_img must be 512")
        if config.get("sift_max_num_features") != 2048:
            errors.append("production config sift_max_num_features must be 2048")
        if (
            type(config.get("sift_max_num_orientations")) is not int
            or config["sift_max_num_orientations"] != 1
        ):
            errors.append(
                "production config sift_max_num_orientations must be literal "
                "integer 1 for a 2048-row hard cap"
            )
        if config.get("max_num_tracks") != 400000:
            errors.append("production config max_num_tracks must be 400000")
        if "rerun_from" in config:
            errors.append("clean production config must not carry rerun_from")
        launcher = Path(config.get("memory_safe_launcher", ""))
        if not launcher.is_absolute() or not launcher.is_file():
            errors.append("production config memory_safe_launcher is invalid")
        if replay_drift:
            errors.append("persisted loader probe differs from independent replay")
        if not (
            replay.get("n_forced_expected") == replay.get("n_forced_injected") == len(pair_set)
            and replay.get("missing_forced_pairs") == 0
            and replay.get("n_gt_camera_slots") == replay.get("n_non_null_gt_cameras") == 1
        ):
            errors.append("independent loader replay differs from conditional pairs or one camera")
        return _check(
            "release/material_semantic/S3_pairs",
            not errors,
            "S3 natural-first files and one-camera loader replay revalidate"
            if not errors
            else "S3 material closure or independent loader replay differs",
            n_pairs=len(pair_set),
            replay_forced_expected=replay.get("n_forced_expected"),
            replay_forced_injected=replay.get("n_forced_injected"),
            replay_gt_slots=replay.get("n_gt_camera_slots"),
            replay_gt_non_null=replay.get("n_non_null_gt_cameras"),
            replay_drift=replay_drift,
            errors=errors,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        return _check(
            "release/material_semantic/S3_pairs",
            False,
            str(exc),
            n_pairs=0,
            replay_forced_expected=0,
            replay_forced_injected=0,
            replay_gt_slots=0,
            replay_gt_non_null=0,
            replay_drift={},
            errors=[str(exc)],
        )


def validate_s0_semantics(run_dir: Path) -> dict[str, Any]:
    try:
        manifest = read_json(Path(run_dir) / "corpus_manifest.json")
        build = manifest.get("build", [])
        test = manifest.get("test", [])
        build_by_seq = {record.get("seq"): record for record in build}
        test_by_seq = {record.get("seq"): record for record in test}
        errors = []
        if set(build_by_seq) != {video.seq for video in BUILD}:
            errors.append("build sequence set differs")
        expected_test = {video.seq for video in TEST}
        if set(test_by_seq) != expected_test:
            errors.append("test sequence set differs")
        for video in BUILD:
            record = build_by_seq.get(video.seq, {})
            if record.get("rel") != video.rel:
                errors.append(f"{video.seq}: relative path differs")
            recorded = record.get("source_sha256", record.get("sha256"))
            if not isinstance(recorded, str) or len(recorded) != 64:
                errors.append(f"{video.seq}: source hash is absent")
        hashes = [
            record.get("source_sha256", record.get("sha256"))
            for record in [*build, *test]
        ]
        expected_hashes = len(BUILD) + len(TEST)
        if len(hashes) != expected_hashes or len(set(hashes)) != expected_hashes:
            errors.append("declared corpus hashes are not present and unique")
        return _check(
            "release/material_semantic/S0_corpus",
            not errors,
            f"corpus manifest independently resolves the exact {len(BUILD)}+{len(TEST)} split"
            if not errors
            else "corpus manifest split/hash schema differs",
            n_build=len(build),
            n_test=len(test),
            errors=errors,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return _check(
            "release/material_semantic/S0_corpus",
            False,
            str(exc),
            n_build=0,
            n_test=0,
            errors=[str(exc)],
        )


def validate_s1_semantics(run_dir: Path) -> dict[str, Any]:
    try:
        import s1_motion_scan as s1

        manifest = read_json(Path(run_dir) / "motion_manifest.json")
        sequences = manifest.get("sequences", {})
        errors = []
        violations = []
        n_records = 0
        for video in BUILD:
            sequence = sequences.get(video.seq)
            if not isinstance(sequence, dict) or not sequence.get("records"):
                errors.append(f"{video.seq}: records are absent")
                continue
            for record in sequence["records"]:
                n_records += 1
                if record.get("motion_class") == "pure_rotation" and not s1.is_pure_rotation(record):
                    violations.append({"seq": video.seq, "t": record.get("t")})
        directions = {video.seq: video.direction for video in BUILD}
        direction_records = manifest.get("directions", {})
        if not isinstance(direction_records, dict):
            errors.append("motion direction records are malformed")
        else:
            for seq, record in direction_records.items():
                if seq not in directions or not isinstance(record, dict):
                    errors.append(f"{seq}: direction record is unexpected or malformed")
                    continue
                directions[seq] = record.get("direction")
        if set(sequences) != {video.seq for video in BUILD}:
            errors.append("motion sequence set differs")
        if (
            len(directions) != len(BUILD)
            or list(directions.values()).count("fwd")
            != sum(video.direction == "fwd" for video in BUILD)
            or list(directions.values()).count("rev")
            != sum(video.direction == "rev" for video in BUILD)
        ):
            errors.append("resolved direction cardinality differs")
        if violations:
            errors.append("pure_rotation contract violations exist")
        return _check(
            "release/material_semantic/S1_motion",
            not errors,
            "motion manifest replays all records and declared direction partition"
            if not errors
            else "motion manifest semantics differ",
            n_records=n_records,
            n_pure_violations=len(violations),
            violations=violations,
            directions=directions,
            errors=errors,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return _check(
            "release/material_semantic/S1_motion",
            False,
            str(exc),
            n_records=0,
            n_pure_violations=0,
            violations=[],
            directions={},
            errors=[str(exc)],
        )


def validate_s2_manifest_semantics(run_dir: Path) -> dict[str, Any]:
    try:
        import s2_extract as s2

        root = Path(run_dir)
        manifest = read_json(root / "frame_manifest.json")
        frames = manifest["frames"]
        errors = []
        groups = {seq: 0 for seq in s2.EXPECTED_FRAME_COUNTS}
        hover = {seq: 0 for seq in s2.EXPECTED_FRAME_COUNTS}
        roles: dict[str, int] = {}
        for frame in frames:
            seq = frame["seq"]
            groups[seq] = groups.get(seq, 0) + 1
            hover[seq] = hover.get(seq, 0) + (frame.get("motion_class") == "hover")
            role = frame.get("role")
            roles[role] = roles.get(role, 0) + 1
            expected_role = "bridge_only" if frame.get("motion_class") in s2.BRIDGE_ONLY_CLASSES else "triangulation"
            if role != expected_role:
                errors.append(f"{frame.get('name')}: role differs")
        if groups != s2.EXPECTED_FRAME_COUNTS:
            errors.append("per-sequence frame-count vector differs")
        for seq, count in groups.items():
            if not count or hover[seq] / count > s2.MAX_HOVER_RATIO:
                errors.append(f"{seq}: hover cap differs")
        if manifest.get("n_frames") != len(frames) or len(frames) != PROBE_FRAME_COUNT:
            errors.append("frame total differs")
        if manifest.get("roles") != roles:
            errors.append("manifest role aggregate differs")
        replay_gate = Gate("S2_extract_replay", {"G2.1"})
        s2.prove_no_undistortion(replay_gate, root / "images", frames)
        if replay_gate.checks[0]["state"] != "PASS":
            errors.append("raw-vs-undistorted discriminative replay failed")
        return _check(
            "release/material_semantic/S2_extract",
            not errors,
            "frame plan, roles, hover caps, and raw-pixel replay recompute"
            if not errors
            else "S2 manifest semantics differ",
            n_frames=len(frames),
            groups=groups,
            hover=hover,
            roles=roles,
            raw_pixel_evidence=replay_gate.checks[0]["evidence"],
            errors=errors,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return _check(
            "release/material_semantic/S2_extract",
            False,
            str(exc),
            n_frames=0,
            groups={},
            hover={},
            roles={},
            raw_pixel_evidence={},
            errors=[str(exc)],
        )
def validate_stage_material_contract(
    stage: str, run_dir: Path, payload: Mapping[str, Any] | None
) -> dict[str, Any]:
    expected = {
        label: str(path.resolve())
        for label, path in material_artifacts(stage, run_dir).items()
    }
    records = (
        payload.get("provenance", {}).get("input_artifacts", {})
        if isinstance(payload, Mapping)
        else {}
    )
    actual = {
        label: record.get("path")
        for label, record in records.items()
        if isinstance(record, dict)
    }
    missing_labels = sorted(set(expected) - set(actual))
    unexpected_labels = sorted(set(actual) - set(expected))
    wrong_paths = {
        label: {"expected": expected[label], "actual": actual.get(label)}
        for label in set(expected) & set(actual)
        if actual.get(label) != expected[label]
    }
    absent = sorted(
        label for label, path in material_artifacts(stage, run_dir).items() if not path.exists()
    )
    image_check = (
        validate_s2_image_corpus(run_dir)
        if stage in {"S2_extract", "S2b_intrinsics", "S3_pairs"}
        else None
    )
    s0_check = validate_s0_semantics(run_dir) if stage == "S0_corpus" and not absent else None
    s1_check = validate_s1_semantics(run_dir) if stage == "S1_motion" and not absent else None
    s2_check = validate_s2_manifest_semantics(run_dir) if stage == "S2_extract" and not absent else None
    seed_check = (
        validate_intrinsics_seed(run_dir)
        if stage in {"S2_extract", "S2b_intrinsics", "S3_pairs"} and not absent
        else None
    )
    s2b_check = (
        validate_s2b_semantics(run_dir)
        if stage == "S2b_intrinsics" and not absent
        else None
    )
    s3_check = (
        validate_s3_semantics(run_dir)
        if stage == "S3_pairs" and not absent
        else None
    )
    semantic_checks = [
        check
        for check in (
            s0_check,
            s1_check,
            image_check,
            seed_check,
            s2_check,
            s2b_check,
            s3_check,
        )
        if check is not None
    ]
    ok = (
        not missing_labels
        and not unexpected_labels
        and not wrong_paths
        and not absent
        and all(check["ok"] for check in semantic_checks)
    )
    return _check(
        f"release/artifacts/{stage}",
        ok,
        "exact material artifact labels/paths exist" if ok else "material artifact closure is incomplete or mismatched",
        n_expected=len(expected),
        missing_labels=missing_labels,
        unexpected_labels=unexpected_labels,
        wrong_paths=wrong_paths,
        absent=absent,
        semantic_replays={check["id"]: check["evidence"] for check in semantic_checks}
        or {"not_applicable": True},
    )


def _all_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _all_strings(key)
            yield from _all_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_strings(child)


def validate_stage_chain(run_dir: Path) -> list[dict[str, Any]]:
    """Validate exact stage semantics, provenance freshness, and hash chaining."""
    gate_paths = [run_dir / "gates" / f"{stage}.json" for stage in STAGES]
    checks: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    missing = [str(path) for path in gate_paths if not path.is_file()]
    checks.append(
        _check(
            "release/stage_files",
            not missing,
            "all five stage gates exist" if not missing else "stage gates are missing",
            missing=missing,
        )
    )
    for stage, path in zip(STAGES, gate_paths, strict=True):
        if not path.is_file():
            checks.append(
                _check(
                    f"release/exact/{stage}",
                    False,
                    "stage gate is missing",
                    path=str(path),
                )
            )
            checks.append(
                _check(
                    f"release/semantic/{stage}",
                    False,
                    "stage gate is missing",
                    errors={"gate": "missing"},
                )
            )
            checks.append(validate_stage_material_contract(stage, run_dir, None))
            checks.append(
                _check(
                    f"release/fresh/{stage}",
                    False,
                    "stage gate is missing",
                    path=str(path),
                )
            )
            continue
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            checks.append(
                _check(
                    f"release/exact/{stage}", False, "stage gate is unreadable", error=str(exc)
                )
            )
            checks.append(
                _check(
                    f"release/semantic/{stage}", False, "stage gate is unreadable", error=str(exc)
                )
            )
            checks.append(validate_stage_material_contract(stage, run_dir, None))
            checks.append(
                _check(
                    f"release/fresh/{stage}", False, "stage gate is unreadable", error=str(exc)
                )
            )
            continue
        payloads.append(payload)
        required_ids = payload.get("required_ids", [])
        raw_checks = payload.get("checks", [])
        emitted_ids = (
            [item.get("id") if isinstance(item, dict) else None for item in raw_checks]
            if isinstance(raw_checks, list)
            else []
        )
        exact = _exact_stage_gate_pass(stage, payload)
        checks.append(
            _check(
                f"release/exact/{stage}",
                exact,
                "PASS with the complete exact check set" if exact else "gate is not an exact substantive PASS",
                status=payload.get("status"),
                required_ids=required_ids,
                emitted_ids=emitted_ids,
            )
        )
        checks.append(validate_typed_stage_evidence(stage, payload))
        checks.append(validate_stage_material_contract(stage, run_dir, payload))
        try:
            fresh = assert_gate_fresh(path)
            detail = "all recorded provenance re-hashes"
        except (GateFreshnessError, OSError, ValueError) as exc:
            fresh = False
            detail = str(exc)
        checks.append(_check(f"release/fresh/{stage}", fresh, detail))

    archived_paths = sorted(
        {
            value
            for payload in payloads
            for value in _all_strings(payload.get("provenance", {}))
            if any(token in value for token in ARCHIVE_TOKENS)
        }
    )
    checks.append(
        _check(
            "release/no_archived_inputs",
            not archived_paths,
            "no active gate provenance references archived pre-fix evidence"
            if not archived_paths
            else "active provenance references archived pre-fix evidence",
            archived_paths=archived_paths,
        )
    )

    try:
        chained = len(payloads) == len(STAGES) and verify_predecessor_chain(gate_paths)
        detail = "S0 -> S1 -> S2 -> S2b -> S3 predecessor hashes revalidate"
    except (GateFreshnessError, OSError, ValueError) as exc:
        chained = False
        detail = str(exc)
    checks.append(_check("release/predecessor_chain", chained, detail))
    return checks


def write_runtime_env_lock(run_dir: Path) -> Path:
    """Persist the active interpreter/package inventory used by regenerated stages."""
    runtime = verify_pycolmap_runtime()
    pip = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=True,
        check=True,
    )
    conda = subprocess.run(
        ["micromamba", "list", "-p", sys.prefix, "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python_executable": str(Path(sys.executable).resolve()),
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "pycolmap_runtime": runtime,
        "source_selected_runtime": {
            "environment": str(GLUEMAP_ENV.resolve()),
            "python": str(GLUEMAP_PY.resolve()),
        },
        "pip_freeze": sorted(line for line in pip.stdout.splitlines() if line.strip()),
        "micromamba_list": json.loads(conda.stdout),
    }
    providers = [
        line
        for line in payload["pip_freeze"]
        if line.lower().startswith(("pycolmap==", "pycolmap-cuda12==", "easy-gravity=="))
    ]
    if providers != ["pycolmap-cuda12==4.0.4"]:
        raise RuntimeError(f"runtime lock has invalid pycolmap providers: {providers}")
    if Path(sys.prefix).resolve() != GLUEMAP_ENV.resolve() or Path(sys.executable).resolve() != GLUEMAP_PY.resolve():
        raise RuntimeError(
            "runtime lock generation must run under the source-selected canonical interpreter"
        )
    path = run_dir / "runtime_env_lock.json"
    write_json(path, payload)
    return path


def _git_source_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=GLUEMAP_REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=GLUEMAP_REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    dirty: list[dict[str, Any]] = []
    for line in status:
        rel = line[3:]
        if " -> " in rel:
            rel = rel.split(" -> ", 1)[1]
        path = GLUEMAP_REPO / rel
        dirty.append({"status": line[:2], "path": rel, "artifact": hash_artifact(path)})
    return {"commit": commit, "dirty": dirty}


def write_source_lock(run_dir: Path, runtime_lock: Path) -> Path:
    """Hash all Phase B decision source plus the active environment inventory."""
    gluemap_sources = (
        GLUEMAP_REPO / "gluemap" / "cli.py",
        GLUEMAP_REPO / "gluemap" / "datasets" / "multi_sequence_twoview.py",
        GLUEMAP_REPO / "gluemap" / "utils" / "colmap.py",
        GLUEMAP_REPO / "gluemap" / "utils" / "cli.py",
        GLUEMAP_REPO / "gluemap" / "estimators" / "augmented_bundle_adjustment.py",
        GLUEMAP_REPO / "gluemap" / "controllers" / "augmented_bundle_adjustment.py",
        GLUEMAP_REPO / "gluemap" / "controllers" / "global_refinement.py",
        GLUEMAP_REPO / "gluemap" / "controllers" / "gluemap_impl.py",
        GLUEMAP_REPO / "gluemap" / "math" / "reprojection_error.py",
        GLUEMAP_REPO / "pyproject.toml",
    )
    checkpoints = [
        GLUEMAP_REPO / "checkpoints" / "pi3.safetensors",
        GLUEMAP_REPO / "checkpoints" / "dino_salad.ckpt",
        GLUEMAP_REPO / "checkpoints" / "vggsfm_v2_0_0_track_predictor.bin",
        GLUEMAP_REPO / "checkpoints" / "checkpoint-dg+visym.pth",
    ]
    source_paths = (
        sorted(TOOLS.glob("*.py"))
        + list(gluemap_sources)
        + checkpoints
        + [runtime_lock]
    )
    records = [hash_artifact(path) for path in source_paths]
    missing = [record["path"] for record in records if record["sha256"] is None]
    if missing:
        raise FileNotFoundError(f"source-lock inputs are missing: {missing}")
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "sources": records,
        "gluemap_git": _git_source_state(),
    }
    path = run_dir / "source_lock.json"
    write_json(path, payload)
    return path


def validate_source_lock(path: Path) -> dict[str, Any]:
    try:
        payload = read_json(path)
        drift = []
        for record in payload.get("sources", []):
            actual = hash_artifact(record["path"])
            if actual["sha256"] != record.get("sha256"):
                drift.append(
                    {
                        "path": record.get("path"),
                        "expected": record.get("sha256"),
                        "actual": actual["sha256"],
                    }
                )
        expected_git = payload.get("gluemap_git")
        actual_git = _git_source_state()
        git_drift: dict[str, Any] = {}
        if not isinstance(expected_git, dict):
            git_drift["schema"] = "gluemap_git must be a non-empty object"
        else:
            if expected_git.get("commit") != actual_git.get("commit"):
                git_drift["commit"] = {
                    "expected": expected_git.get("commit"),
                    "actual": actual_git.get("commit"),
                }
            if expected_git.get("dirty") != actual_git.get("dirty"):
                git_drift["dirty"] = {
                    "expected": expected_git.get("dirty"),
                    "actual": actual_git.get("dirty"),
                }
        ok = bool(payload.get("sources")) and not drift and not git_drift
        return _check(
            "release/source_lock",
            ok,
            (
                "every source-lock hash and the exact gluemap git state revalidate"
                if ok
                else "source-lock or gluemap git-state drift"
            ),
            n_sources=len(payload.get("sources", [])),
            drift=drift,
            git_drift=git_drift,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return _check("release/source_lock", False, str(exc))


def validate_runtime_lock(path: Path) -> dict[str, Any]:
    """Revalidate the persisted runtime against source selection and live import."""
    try:
        payload = read_json(path)
        locked_runtime = payload.get("pycolmap_runtime")
        live_runtime = verify_pycolmap_runtime()
        locked_python = Path(payload["python_executable"]).resolve()
        locked_prefix = Path(payload["sys_prefix"]).resolve()
        selected_record = payload.get("source_selected_runtime", {})
        selected_runtime_matches = (
            locked_python == GLUEMAP_PY.resolve()
            and locked_prefix == GLUEMAP_ENV.resolve()
            and locked_prefix == EXPECTED_RUNTIME_PREFIX.resolve()
            and Path(selected_record.get("environment", "")).resolve()
            == GLUEMAP_ENV.resolve()
            and Path(selected_record.get("python", "")).resolve()
            == GLUEMAP_PY.resolve()
        )
        providers = [
            line
            for line in payload.get("pip_freeze", [])
            if str(line).lower().startswith(
                ("pycolmap==", "pycolmap-cuda12==", "easy-gravity==")
            )
        ]
        fingerprint_matches = (
            _runtime_evidence_ok(locked_runtime)
            and locked_runtime == live_runtime
        )
        ok = (
            selected_runtime_matches
            and fingerprint_matches
            and providers == ["pycolmap-cuda12==4.0.4"]
        )
        return _check(
            "release/runtime_lock",
            ok,
            "source selection, lock, and live runtime fingerprint agree"
            if ok
            else "runtime lock differs from source selection or live preflight",
            selected_runtime_matches=selected_runtime_matches,
            fingerprint_matches=fingerprint_matches,
            locked_python=str(locked_python),
            locked_prefix=str(locked_prefix),
            selected_python=str(GLUEMAP_PY.resolve()),
            selected_prefix=str(GLUEMAP_ENV.resolve()),
            providers=providers,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return _check(
            "release/runtime_lock",
            False,
            str(exc),
            selected_runtime_matches=False,
            fingerprint_matches=False,
        )


def initialize_fresh_run(run_dir: Path) -> None:
    forbidden = [run_dir / "images", run_dir / "intrinsics_seed"]
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise RuntimeError(f"fresh-run roots still exist: {present}")
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_lock = write_runtime_env_lock(run_dir)
    write_source_lock(run_dir, runtime_lock)
    prerequisite = run_dir / ".not_run"
    prerequisite.write_text("Phase B initialized; stages have not run yet.\n", encoding="utf-8")
    for stage in STAGES:
        initialize_not_run_gate(
            run_dir,
            stage,
            "Phase B initialized; this stage has not run yet",
            script_path=TOOLS / STAGE_SCRIPTS[stage],
        )


def issue_release_gate(run_dir: Path) -> dict[str, Any]:
    checks = [
        validate_runtime_lock(run_dir / "runtime_env_lock.json"),
        validate_source_lock(run_dir / "source_lock.json"),
    ]
    checks.extend(validate_stage_chain(run_dir))
    inputs = {
        "source_lock": hash_artifact(run_dir / "source_lock.json"),
        "runtime_env_lock": hash_artifact(run_dir / "runtime_env_lock.json"),
    }
    inputs.update(
        {
            stage: hash_artifact(run_dir / "gates" / f"{stage}.json")
            for stage in STAGES
        }
    )
    gate = Gate(
        "S0_S3_release",
        RELEASE_REQUIRED_IDS,
        script_path=Path(__file__),
        input_artifacts={label: record["path"] for label, record in inputs.items()},
        source_files=[TS_COMMON, TS_ENV],
    )
    for check in checks:
        gate.check(
            check["id"],
            check["ok"],
            check["detail"],
            **check["evidence"],
        )
    payload = gate.write(
        run_dir,
        fail_hard=False,
        output_path=run_dir / "gates" / "S0_S3_release.json",
    )
    payload["release_index"] = [inputs[stage] for stage in STAGES]
    write_json(run_dir / "gates" / "S0_S3_release.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=RUN_ID)
    parser.add_argument("--init", action="store_true")
    args = parser.parse_args()
    run_dir = RUNS / args.run_name
    if args.init:
        initialize_fresh_run(run_dir)
        print(f"initialized Phase B run: {run_dir}")
        return
    result = issue_release_gate(run_dir)
    print(f"S0-S3 release: {result['status']} ({len(result['checks'])} checks)")
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
