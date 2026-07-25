from __future__ import annotations

import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from ts_common import (  # noqa: E402
    Gate,
    GateFreshnessError,
    assert_gate_fresh,
    hash_artifact,
    required_check_ids,
    write_json,
    sha256,
)
import verify_s0_s3_release as release  # noqa: E402
from verify_s0_s3_release import validate_stage_chain  # noqa: E402
import resource_guard  # noqa: E402
import s2_extract as s2  # noqa: E402
import s2b_intrinsics_bakeoff as s2b  # noqa: E402
import s3_pairs as s3  # noqa: E402


STAGES = (
    "S0_corpus",
    "S1_motion",
    "S2_extract",
    "S2b_intrinsics",
    "S3_pairs",
)


def _write_pass_chain(run_dir: Path) -> list[Path]:
    input_artifact = run_dir / "input.json"
    input_artifact.parent.mkdir(parents=True, exist_ok=True)
    input_artifact.write_text('{"locked": true}\n', encoding="utf-8")
    paths: list[Path] = []
    for index, stage in enumerate(STAGES):
        script = run_dir / f"{stage}.py"
        script.write_text(f"# {stage}\n", encoding="utf-8")
        gate = Gate(
            stage,
            required_check_ids(stage),
            script_path=script,
            input_artifacts={"input": input_artifact},
        )
        if paths:
            gate.record_predecessor_gate(
                STAGES[index - 1], paths[-1], expected_stage=STAGES[index - 1]
            )
        for gid in sorted(gate.required_ids):
            gate.check(gid, True, "persisted test evidence", observed=1)
        gate.write(run_dir)
        paths.append(run_dir / "gates" / f"{stage}.json")
    return paths


def test_release_validator_rejects_structurally_complete_literal_true_chain(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_pass_chain(run_dir)

    checks = validate_stage_chain(run_dir)

    assert not all(check["ok"] for check in checks)
    semantic = [check for check in checks if check["id"].startswith("release/semantic/")]
    assert len(semantic) == len(STAGES)
    assert all(check["ok"] is False for check in semantic)


def test_stage_material_contract_covers_every_required_output_class(
    tmp_path: Path,
) -> None:
    s2 = release.material_artifacts("S2_extract", tmp_path)
    assert set(s2) >= {
        "frame_manifest",
        "images",
        "intrinsics_seed_manifest",
        "intrinsics_seed",
    }

    s2b = release.material_artifacts("S2b_intrinsics", tmp_path)
    assert set(s2b) >= {
        "intrinsics_policy",
        "images",
        "intrinsics_seed_manifest",
        "intrinsics_seed",
    }

    s3 = release.material_artifacts("S3_pairs", tmp_path)
    assert set(s3) >= {
        "forced_bridges_manifest",
        "forced_bridges_text",
        "s3_loader_probe",
        "s3_loader_probe_log",
        "gluemap_config",
        "frame_manifest",
        "images",
        "intrinsics_seed_manifest",
        "intrinsics_seed",
        "intrinsics_policy",
        "resource_contract",
    }


@pytest.mark.parametrize(
    ("stage", "module"),
    [
        ("S2_extract", s2),
        ("S2b_intrinsics", s2b),
        ("S3_pairs", s3),
    ],
)
def test_production_gate_binds_the_exact_material_contract(
    tmp_path: Path, stage: str, module
) -> None:
    gate = module.stage_gate(tmp_path)
    expected = {
        label: path.resolve()
        for label, path in release.material_artifacts(stage, tmp_path).items()
    }

    assert gate._input_paths == expected


MATERIAL_CLASS_CASES = [
    ("S2_extract", "images"),
    ("S2b_intrinsics", "intrinsics_policy"),
    ("S3_pairs", "forced_bridges_manifest"),
    ("S3_pairs", "forced_bridges_text"),
    ("S3_pairs", "s3_loader_probe"),
    ("S3_pairs", "s3_loader_probe_log"),
    ("S3_pairs", "gluemap_config"),
    ("S3_pairs", "resource_contract"),
    ("S3_pairs", "images"),
    ("S3_pairs", "intrinsics_seed"),
]


@pytest.mark.parametrize(("stage", "label"), MATERIAL_CLASS_CASES)
@pytest.mark.parametrize("operation", ["mutate", "delete"])
def test_every_material_output_class_fails_closed_after_drift(
    tmp_path: Path, stage: str, label: str, operation: str
) -> None:
    run_dir = tmp_path / "run"
    artifacts = release.material_artifacts(stage, run_dir)
    directory_labels = {
        name
        for name in artifacts
        if name in {"images", "intrinsics_seed"}
        or name.endswith(("/seed", "/model"))
    }
    for name, path in artifacts.items():
        if name in directory_labels:
            path.mkdir(parents=True, exist_ok=True)
            (path / "bound.bin").write_bytes(b"bound")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"bound")
    script = tmp_path / "verifier.py"
    script.write_text("# verifier\n", encoding="utf-8")
    gate = Gate(
        stage,
        {"evidence"},
        script_path=script,
        input_artifacts=artifacts,
    )
    gate.check("evidence", True, "typed fixture", count=1)
    gate_path = tmp_path / "gate.json"
    gate.write(run_dir, output_path=gate_path, fail_hard=False)

    target = artifacts[label]
    if operation == "delete":
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    elif target.is_dir():
        (target / "bound.bin").write_bytes(b"mutated")
    else:
        target.write_bytes(b"mutated")

    with pytest.raises(GateFreshnessError):
        assert_gate_fresh(gate_path)


def test_release_validator_rejects_mutated_stage_input(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_pass_chain(run_dir)
    (run_dir / "input.json").write_text('{"locked": false}\n', encoding="utf-8")

    checks = validate_stage_chain(run_dir)

    assert not all(check["ok"] for check in checks)
    assert any("fresh" in check["id"] and not check["ok"] for check in checks)


def test_release_validator_rejects_archived_provenance_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    paths = _write_pass_chain(run_dir)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["provenance"]["input_artifacts"]["archive"] = {
        "path": str(tmp_path / "_backup_pre_impl" / "stale_gates" / "S3.json"),
        "kind": "file",
        "size": 1,
        "sha256": "0" * 64,
    }
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")

    checks = validate_stage_chain(run_dir)

    assert any(check["id"] == "release/no_archived_inputs" and not check["ok"] for check in checks)


@pytest.mark.parametrize("drift", ["commit", "status", "path", "content"])
def test_source_lock_rejects_every_gluemap_git_state_drift(
    tmp_path: Path, monkeypatch, drift: str
) -> None:
    source = tmp_path / "source.py"
    dirty_file = tmp_path / "dirty.py"
    source.write_text("source = 1\n", encoding="utf-8")
    dirty_file.write_text("dirty = 1\n", encoding="utf-8")
    expected_git = {
        "commit": "a" * 40,
        "dirty": [
            {
                "status": " M",
                "path": "dirty.py",
                "artifact": hash_artifact(dirty_file),
            }
        ],
    }
    lock_path = tmp_path / "source_lock.json"
    write_json(
        lock_path,
        {"sources": [hash_artifact(source)], "gluemap_git": expected_git},
    )
    actual_git = deepcopy(expected_git)
    if drift == "commit":
        actual_git["commit"] = "b" * 40
    elif drift == "status":
        actual_git["dirty"][0]["status"] = "??"
    elif drift == "path":
        actual_git["dirty"][0]["path"] = "renamed.py"
    else:
        actual_git["dirty"][0]["artifact"]["sha256"] = "0" * 64
    monkeypatch.setattr(release, "_git_source_state", lambda: actual_git)

    check = release.validate_source_lock(lock_path)

    assert check["ok"] is False
    assert check["id"] == "release/source_lock"
    assert check["metrics"]["git_drift"]


def test_runtime_lock_rejects_a_superseded_selected_interpreter(
    tmp_path: Path, monkeypatch
) -> None:
    good_runtime = {
        "version": "4.0.4",
        "module_path": str(
            release.EXPECTED_RUNTIME_PREFIX / "lib/python3.11/site-packages/pycolmap/__init__.py"
        ),
        "providers": [{"name": "pycolmap-cuda12", "version": "4.0.4"}],
        "required_apis": {"api": True},
        "forbidden_apis": {"old_api": False},
        "python_executable": str(release.EXPECTED_RUNTIME_PREFIX / "bin/python3.11"),
        "sys_prefix": str(release.EXPECTED_RUNTIME_PREFIX),
    }
    old_prefix = Path("/home/cihcilab/micromamba/envs/target-site-gluemap")
    lock_path = tmp_path / "runtime_env_lock.json"
    write_json(
        lock_path,
        {
            "python_executable": str(old_prefix / "bin/python"),
            "sys_prefix": str(old_prefix),
            "pycolmap_runtime": good_runtime,
            "pip_freeze": ["pycolmap-cuda12==4.0.4"],
        },
    )
    monkeypatch.setattr(release, "verify_pycolmap_runtime", lambda: good_runtime)

    check = release.validate_runtime_lock(lock_path)

    assert check["ok"] is False
    assert check["metrics"]["selected_runtime_matches"] is False


def test_release_output_uses_an_exact_nonempty_typed_check_contract(
    tmp_path: Path,
) -> None:
    result = release.issue_release_gate(tmp_path / "run")

    assert result["required_ids"]
    assert set(result["required_ids"]) == release.RELEASE_REQUIRED_IDS
    assert result["missing_ids"] == []
    assert {check["id"] for check in result["checks"]} == release.RELEASE_REQUIRED_IDS
    assert all(check["state"] in {"PASS", "FAIL"} for check in result["checks"])
    assert all(isinstance(check["evidence"], dict) and check["evidence"] for check in result["checks"])


def test_s1_semantic_replay_resolves_anchor_directions_from_build_contract(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    from ts_common import BUILD

    resolved = {video.seq: video.direction for video in BUILD}
    write_json(
        run_dir / "motion_manifest.json",
        {
            "sequences": {
                seq: {"records": [{"motion_class": "parallax"}]}
                for seq in resolved
            },
            "directions": {
                seq: {"direction": direction} for seq, direction in resolved.items()
            },
        },
    )

    check = release.validate_s1_semantics(run_dir)

    assert check["ok"] is True
    assert check["evidence"]["directions"] == resolved


@pytest.mark.parametrize("drift", ["hash", "delete", "stray"])
def test_s2_independent_corpus_validator_rejects_image_closure_drift(
    tmp_path: Path, drift: str
) -> None:
    run_dir = tmp_path / "run"
    images = run_dir / "images"
    first = images / "S01/000001.jpg"
    second = images / "S02/000001.jpg"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"first image")
    second.write_bytes(b"second image")
    write_json(
        run_dir / "frame_manifest.json",
        {
            "n_frames": 2,
            "frames": [
                {"name": "S01/000001.jpg", "image_sha256": sha256(first)},
                {"name": "S02/000001.jpg", "image_sha256": sha256(second)},
            ],
        },
    )
    if drift == "hash":
        first.write_bytes(b"mutated")
    elif drift == "delete":
        second.unlink()
    else:
        (images / "S02/stray.jpg").write_bytes(b"stray")

    check = release.validate_s2_image_corpus(run_dir)

    assert check["ok"] is False
    assert check["evidence"]["missing"] or check["evidence"]["unexpected"] or check["evidence"]["hash_mismatches"]


def _safe_g36_metrics() -> dict:
    return {
        "config": str((TARGET_SITE / "config.yaml").resolve()),
        "extra_pairs_path": str((TARGET_SITE / "pairs.txt").resolve()),
        "forced_pairs_sha256": "a" * 64,
        "parsed_pairs": 0,
        "injected_pairs": 0,
        "refine_intrinsics": False,
        "skip_doppelgangers": False,
        "num_workers": 0,
        "num_track_per_img": 512,
        "sift_max_num_features": 2048,
        "sift_max_num_orientations": 1,
        "sift_max_rows_per_image": 2048,
        "max_num_tracks": 400000,
        "memory_safe_launcher": str(s3.MEMORY_SAFE_LAUNCHER.resolve()),
        "rerun_from": None,
        "resource_lock_path": str(resource_guard.GLOBAL_HEAVY_LOCK),
        "resource_guard_path": str((TARGET_SITE / "tools/resource_guard.py").resolve()),
        "resource_guard_log_path": str((TARGET_SITE / "run/logs/guard.log").resolve()),
    }


def _g36_accepts(metrics: dict) -> bool:
    try:
        return release._typed_predicate("S3_pairs", "G3.6", metrics)
    except (KeyError, TypeError, ValueError):
        return False


def test_s3_g33_typed_validator_accepts_only_truthful_natural_pre_retrieval_na(
    monkeypatch,
) -> None:
    monkeypatch.setattr(release, "required_check_ids", lambda _stage: {"G3.3"})
    metrics = {
        "policy_mode": "natural_retrieval_first",
        "force_gap_bridges": False,
        "retrieval_executed": False,
        "n_forced_pairs": 0,
        "n_images": 240,
        "conditional_pair_density": 0.0,
        "minimum": 4.0,
        "missing_sequence_pairs": [],
        "pending_sequence_pairs": ["P1090109_002|P1110111"],
        "required_route_clusters": 2,
        "minimum_normalized_separation": 0.25,
    }
    payload = {
        "checks": [
            {
                "id": "G3.3",
                "state": "NOT_APPLICABLE",
                "ok": False,
                "applicable": False,
                "reason": "natural retrieval has not executed",
                "metrics": metrics,
            }
        ]
    }

    assert release.validate_typed_stage_evidence("S3_pairs", payload)["ok"] is True

    metrics["retrieval_executed"] = True
    assert release.validate_typed_stage_evidence("S3_pairs", payload)["ok"] is False


def _exact_pass_payload(stage: str) -> dict:
    required = sorted(required_check_ids(stage))
    return {
        "stage": stage,
        "status": "PASS",
        "ok": True,
        "required_ids": required,
        "checks": [
            {
                "id": gid,
                "state": "PASS",
                "ok": True,
                "applicable": True,
                "detail": "substantive evidence",
                "metrics": {"observed": 1},
                "evidence": {"observed": 1},
            }
            for gid in required
        ],
    }


def _replace_exact_check(payload: dict, gid: str, check: dict) -> None:
    payload["checks"] = [
        check if item["id"] == gid else item for item in payload["checks"]
    ]


def test_exact_s2b_gate_accepts_only_the_two_typed_diagnostic_na_checks() -> None:
    payload = _exact_pass_payload("S2b_intrinsics")
    cases = {
        "G2.7/1920x1080": {
            "diagnostic": "two_seed",
            "resolution": "1920x1080",
            "external_record_sha256": release.EXTERNAL_CAMERA_RECORD_SHA256,
        },
        "G2.8": {
            "diagnostic": "cross_resolution",
            "working_resolutions": ["1920x1080"],
            "external_record_sha256": release.EXTERNAL_CAMERA_RECORD_SHA256,
        },
    }
    for gid, evidence in cases.items():
        _replace_exact_check(
            payload,
            gid,
            {
                "id": gid,
                "state": "NOT_APPLICABLE",
                "ok": False,
                "applicable": False,
                "reason": "externally frozen camera diagnostic",
                "detail": "externally frozen camera diagnostic",
                "metrics": evidence,
                "evidence": evidence,
            },
        )

    assert release._exact_stage_gate_pass("S2b_intrinsics", payload) is True


def test_exact_s3_gate_accepts_truthful_typed_g33_na() -> None:
    payload = _exact_pass_payload("S3_pairs")
    evidence = {
        "policy_mode": "natural_retrieval_first",
        "force_gap_bridges": False,
        "retrieval_executed": True,
        "n_forced_pairs": 0,
        "n_images": 240,
        "conditional_pair_density": 0.0,
        "minimum": 4.0,
        "missing_sequence_pairs": [],
        "pending_sequence_pairs": [],
        "required_route_clusters": 2,
        "minimum_normalized_separation": 0.25,
    }
    _replace_exact_check(
        payload,
        "G3.3",
        {
            "id": "G3.3",
            "state": "NOT_APPLICABLE",
            "ok": False,
            "applicable": False,
            "reason": "natural retrieval required no conditional pairs",
            "detail": "natural retrieval required no conditional pairs",
            "metrics": evidence,
            "evidence": evidence,
        },
    )

    assert release._exact_stage_gate_pass("S3_pairs", payload) is True


@pytest.mark.parametrize("state", ["NOT_RUN", "INCOMPLETE", "FAIL"])
def test_exact_gate_still_rejects_every_nonpass_non_na_state(state: str) -> None:
    payload = _exact_pass_payload("S2b_intrinsics")
    check = payload["checks"][0]
    check.update({"state": state, "ok": False})

    assert release._exact_stage_gate_pass("S2b_intrinsics", payload) is False


@pytest.mark.parametrize(
    "mutation",
    ["wrong_id", "missing_applicable", "empty_reason", "missing_evidence", "bad_evidence"],
)
def test_exact_gate_rejects_malformed_or_unapproved_na(mutation: str) -> None:
    payload = _exact_pass_payload("S2b_intrinsics")
    evidence = {
        "diagnostic": "two_seed",
        "resolution": "1920x1080",
        "external_record_sha256": release.EXTERNAL_CAMERA_RECORD_SHA256,
    }
    check = {
        "id": "G2.7/1920x1080",
        "state": "NOT_APPLICABLE",
        "ok": False,
        "applicable": False,
        "reason": "externally frozen camera diagnostic",
        "detail": "externally frozen camera diagnostic",
        "metrics": evidence,
        "evidence": evidence,
    }
    if mutation == "wrong_id":
        check["id"] = "G2.7/results_complete"
        payload["checks"] = [
            item
            for item in payload["checks"]
            if item["id"] != "G2.7/results_complete"
        ]
        payload["checks"].append(check)
    else:
        _replace_exact_check(payload, "G2.7/1920x1080", check)
        if mutation == "missing_applicable":
            check.pop("applicable")
        elif mutation == "empty_reason":
            check["reason"] = " "
        elif mutation == "missing_evidence":
            check.pop("evidence")
        else:
            check["metrics"] = {**evidence, "resolution": "3840x2160"}
            check["evidence"] = check["metrics"]

    assert release._exact_stage_gate_pass("S2b_intrinsics", payload) is False


@pytest.mark.parametrize("num_workers", ["missing", False, 4])
def test_s3_g36_typed_predicate_rejects_unsafe_num_workers(num_workers) -> None:
    metrics = _safe_g36_metrics()
    if num_workers == "missing":
        metrics.pop("num_workers")
    else:
        metrics["num_workers"] = num_workers

    assert _g36_accepts(metrics) is False


@pytest.mark.parametrize("num_workers", ["missing", False, 4])
def test_s3_material_semantics_rejects_unsafe_num_workers(
    monkeypatch, num_workers
) -> None:
    del monkeypatch
    metrics = _safe_g36_metrics()
    if num_workers == "missing":
        metrics.pop("num_workers")
    else:
        metrics["num_workers"] = num_workers

    assert _g36_accepts(metrics) is False


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("num_track_per_img", 1024),
        ("sift_max_num_features", 8192),
        ("sift_max_num_orientations", 2),
        ("sift_max_num_orientations", None),
        ("sift_max_rows_per_image", 4096),
        ("max_num_tracks", None),
    ],
)
def test_s3_g36_typed_predicate_rejects_unsafe_feature_budget(
    field: str, unsafe_value
) -> None:
    metrics = _safe_g36_metrics()
    metrics[field] = unsafe_value

    assert _g36_accepts(metrics) is False


@pytest.mark.parametrize("orientation_policy", ["missing", 2])
def test_s3_independent_material_semantics_rejects_unsafe_orientation_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, orientation_policy
) -> None:
    forced_path = tmp_path / "forced_bridges.txt"
    forced_path.write_text("# natural retrieval only\n", encoding="utf-8")
    config_path = tmp_path / "gluemap_config.yaml"
    config = {
        "extra_pairs_path": str(forced_path.resolve()),
        "refine_intrinsics": False,
        "skip_doppelgangers": False,
        "num_workers": 0,
        "num_track_per_img": 512,
        "sift_max_num_features": 2048,
        "max_num_tracks": 400000,
        "memory_safe_launcher": str(s3.MEMORY_SAFE_LAUNCHER.resolve()),
    }
    if orientation_policy != "missing":
        config["sift_max_num_orientations"] = orientation_policy
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    write_json(
        tmp_path / "forced_bridges.json",
        {
            "n_pairs": 0,
            "forced_pairs_sha256": sha256(forced_path),
            "config_sha256": sha256(config_path),
            "forced_pairs_path": str(forced_path.resolve()),
            "config_path": str(config_path.resolve()),
            "policy": {"mode": "natural_retrieval_first"},
        },
    )
    probe = {
        "datasets": [],
        "n_images": 0,
        "n_unique_images": 0,
        "n_gt_camera_slots": 1,
        "n_non_null_gt_cameras": 1,
        "n_dimension_matches": 0,
        "missing_seed_names": [],
        "dimension_mismatches": [],
        "n_loader_pairs": 0,
        "n_forced_expected": 0,
        "n_forced_injected": 0,
        "missing_forced_endpoints": [],
        "missing_forced_pairs": 0,
    }
    write_json(tmp_path / "s3_loader_probe.json", probe)
    (tmp_path / "s3_loader_probe.log").write_text("probe\n", encoding="utf-8")
    monkeypatch.setattr(
        s3, "real_loader_probe", lambda _run_dir, _output: deepcopy(probe)
    )

    check = release.validate_s3_semantics(tmp_path)

    assert check["ok"] is False
    assert any(
        "sift_max_num_orientations" in error
        for error in check["evidence"]["errors"]
    )
