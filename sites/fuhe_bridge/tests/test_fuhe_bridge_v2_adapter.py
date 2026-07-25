from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import resource_guard  # noqa: E402
import s1b_bridge_feasibility as s1b  # noqa: E402
import s2_extract as s2  # noqa: E402
import s3_pairs as s3  # noqa: E402
from ts_common import (  # noqa: E402
    BUILD,
    TEST,
    Gate,
    GateFreshnessError,
    RUN_ID,
    assert_no_test_leakage,
    assert_gate_fresh,
    corpus_invariants,
    stage_material_artifacts,
)
from ts_intrinsics import cameras_for  # noqa: E402


BUILD_SEQUENCES = {
    "P1090109_002",
    "P1100110_005",
    "P1110111",
    "P1120112",
    "P1140114",
}


def test_p0_contract_is_the_fixed_fuhe_v2_probe() -> None:
    assert RUN_ID == "fuhe_bridge_v2"
    assert {video.seq for video in BUILD} == BUILD_SEQUENCES
    assert {video.seq for video in TEST} == {"P1130113"}
    assert s2.EXPECTED_FRAME_COUNTS == {seq: 48 for seq in BUILD_SEQUENCES}
    assert sum(s2.EXPECTED_FRAME_COUNTS.values()) == 240
    assert (s2.OUTPUT_WIDTH, s2.OUTPUT_HEIGHT) == (1920, 1080)
    assert s2.RESIZE_INTERPOLATION == cv2.INTER_AREA
    assert s2.UNDISTORT is False


def test_p0_invariants_are_derived_from_the_declared_corpus() -> None:
    invariants = corpus_invariants(BUILD, TEST)

    assert invariants["n_build"] == len(BUILD) == 5
    assert invariants["n_test"] == len(TEST) == 1
    assert invariants["build_sequences"] == sorted(BUILD_SEQUENCES)
    assert invariants["test_sequences"] == ["P1130113"]
    assert invariants["directions"] == {"fwd": 3, "rev": 2}
    assert invariants["source_resolution_groups"] == {"3840x2160": 5}
    assert invariants["working_resolution_groups"] == {"1920x1080": 5}
    assert invariants["epochs"] == {"2026-06-15": 5}
    assert invariants["cross_resolution_gate"]["applicable"] is False
    assert invariants["epoch_gate"]["applicable"] is False


def test_fixed_working_camera_has_exact_calibrated_pinhole() -> None:
    cameras = cameras_for("fuhe_v2_fixed")

    assert list(cameras) == [(1920, 1080)]
    camera = cameras[(1920, 1080)]
    assert camera.params == pytest.approx(
        (1396.8086675255472, 1396.8086675255472, 960.0, 540.0), rel=0, abs=1e-12
    )
    assert "PINHOLE 1920 1080" in camera.colmap_line(1)


def test_resize_is_exact_inter_area_and_never_undistorts(monkeypatch) -> None:
    calls: list[int] = []
    original_resize = cv2.resize

    def recording_resize(image, size, *, interpolation):
        calls.append(interpolation)
        return original_resize(image, size, interpolation=interpolation)

    monkeypatch.setattr(s2.cv2, "resize", recording_resize)
    source = np.zeros((2160, 3840, 3), dtype=np.uint8)

    resized = s2.to_working_resolution(source)

    assert resized.shape == (1080, 1920, 3)
    assert calls == [cv2.INTER_AREA]


def test_gate_can_record_a_non_applicable_predicate_without_making_gate_red(
    tmp_path: Path,
) -> None:
    script = tmp_path / "stage.py"
    artifact = tmp_path / "input.json"
    script.write_text("# stage\n", encoding="utf-8")
    artifact.write_text("{}\n", encoding="utf-8")
    gate = Gate(
        "single_resolution",
        {"G2.8"},
        script_path=script,
        input_artifacts={"input": artifact},
    )

    gate.not_applicable("G2.8", "only one working-resolution camera exists")
    payload = gate.write(tmp_path / "run", fail_hard=False)

    check = payload["checks"][0]
    assert payload["status"] == "PASS"
    assert check["state"] == "NOT_APPLICABLE"
    assert check["applicable"] is False
    assert check["reason"] == "only one working-resolution camera exists"
    assert payload["schema_version"] == "sfm-gate-v2"
    assert payload["material_hash"]["algorithm"] == "sha256"
    assert len(payload["material_hash"]["digest"]) == 64


def test_v2_material_schema_rejects_gate_digest_tampering(tmp_path: Path) -> None:
    script = tmp_path / "stage.py"
    artifact = tmp_path / "input.json"
    script.write_text("# stage\n", encoding="utf-8")
    artifact.write_text("{}\n", encoding="utf-8")
    gate = Gate(
        "material",
        {"G"},
        script_path=script,
        input_artifacts={"input": artifact},
    )
    gate.check("G", True, "material is bound", count=1)
    path = tmp_path / "gate.json"
    gate.write(tmp_path, output_path=path, fail_hard=False)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["material_hash"]["digest"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GateFreshnessError, match="material hash"):
        assert_gate_fresh(path)


def test_s2b_and_s3_material_closures_bind_policy_and_resource_contract(
    tmp_path: Path,
) -> None:
    s2b = stage_material_artifacts("S2b_intrinsics", tmp_path)
    s3 = stage_material_artifacts("S3_pairs", tmp_path)

    assert "intrinsics_policy" in s2b
    assert "intrinsics_bakeoff" not in s2b
    assert "intrinsics_policy" in s3
    assert "resource_contract" in s3


def test_heldout_leakage_is_derived_from_test_corpus(tmp_path: Path) -> None:
    images = tmp_path / "images"
    (images / "P1090109_002").mkdir(parents=True)
    (images / "P1090109_002" / "000001.jpg").write_bytes(b"build")
    (images / "P1130113").mkdir()
    (images / "P1130113" / "000001.jpg").write_bytes(b"heldout")
    script = tmp_path / "stage.py"
    script.write_text("# stage\n", encoding="utf-8")
    gate = Gate(
        "leakage",
        {"G0.2"},
        script_path=script,
        input_artifacts={"images": images},
    )

    assert assert_no_test_leakage(gate, images) is False
    assert gate.checks[0]["metrics"]["heldout_sequences"] == ["P1130113"]


def test_s1b_cache_is_run_local_and_bound_to_material_hash(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / RUN_ID
    source = tmp_path / "corpus.json"
    source.write_text('{"version": 1}\n', encoding="utf-8")

    cache, metadata = s1b.cache_paths(run_dir)
    s1b.write_cache_metadata(metadata, cache, [source], {"topk": 5})
    first = json.loads(metadata.read_text(encoding="utf-8"))
    source.write_text('{"version": 2}\n', encoding="utf-8")

    assert cache.is_relative_to(run_dir)
    assert metadata.is_relative_to(run_dir)
    assert first["schema_version"] == "run-local-cache-v1"
    assert not s1b.cache_is_fresh(metadata, [source], {"topk": 5})


def _route_evidence(n_clusters: int) -> dict[str, list[dict[str, float]]]:
    clusters = [{"fwd_normalized": 0.1, "rev_normalized": 0.1}]
    if n_clusters == 2:
        clusters.append({"fwd_normalized": 0.5, "rev_normalized": 0.5})
    return {"route_clusters": clusters}


def test_gap_bridge_policy_is_natural_first_and_only_fills_unproven_routes() -> None:
    fwd = ["P1090109_002", "P1100110_005", "P1120112"]
    rev = ["P1110111", "P1140114"]
    complete = {f"{a}|{b}": _route_evidence(2) for a in fwd for b in rev}

    natural = s3.decide_gap_bridge_policy(fwd, rev, complete)
    assert natural["mode"] == "natural_retrieval_first"
    assert natural["missing_sequence_pairs"] == []
    assert natural["force_gap_bridges"] is False

    complete["P1090109_002|P1110111"] = _route_evidence(1)
    conditional = s3.decide_gap_bridge_policy(fwd, rev, complete)
    assert conditional["missing_sequence_pairs"] == ["P1090109_002|P1110111"]
    assert conditional["force_gap_bridges"] is True
    assert conditional["minimum_normalized_separation"] == 0.25


def test_gap_bridge_generation_is_deterministic_and_hard_capped() -> None:
    frames = {
        seq: [
            {"name": f"{seq}/{index:06d}.jpg", "seq": seq, "t": float(index)}
            for index in range(48)
        ]
        for seq in BUILD_SEQUENCES
    }
    missing = ["P1090109_002|P1110111", "P1100110_005|P1140114"]

    first = s3.deterministic_gap_pairs(frames, missing, max_pairs=12_000)
    second = s3.deterministic_gap_pairs(frames, missing, max_pairs=12_000)

    assert first == second
    assert 0 < len(first) <= 12_000


def test_clean_s3_config_has_one_camera_no_rerun_and_global_resource_contract(
    tmp_path: Path,
) -> None:
    run_dir = (tmp_path / RUN_ID).resolve()
    run_dir.mkdir()
    forced = run_dir / "forced_bridges.txt"
    forced.write_text("# natural retrieval first\n", encoding="utf-8")

    config = s3.build_gluemap_config(run_dir, forced)
    contract = resource_guard.contract_from_config(config, run_dir)

    assert "rerun_from" not in config
    assert config["camera_model"] == "PINHOLE"
    assert config["intrinsics_mode"] == "SHARED"
    assert config["use_gt_intrinsics"] is True
    assert config["num_workers"] == 0
    assert Path(config["resource_lock_path"]) == resource_guard.GLOBAL_HEAVY_LOCK
    assert not Path(config["resource_lock_path"]).is_relative_to(run_dir)
    assert Path(config["resource_guard_log_path"]).is_relative_to(run_dir)
    assert contract["lock"]["exclusive"] is True
    assert contract["lock"]["scope"] == "global_sfm_heavy"
    assert contract["guard"]["low_memory_gib"] == resource_guard.LOW_MEMORY_GIB
    assert contract["guard"]["sustained_swap_samples"] == 2


def test_p0_to_s3_sources_do_not_retain_target_site_identifiers() -> None:
    inspected = [
        "s0_corpus_lock.py",
        "s1_motion_scan.py",
        "s1b_bridge_feasibility.py",
        "s2_extract.py",
        "s3_pairs.py",
    ]
    source = "\n".join((TOOLS / name).read_text(encoding="utf-8") for name in inspected)

    for stale in ("target_site_v1", "S01_ABrot", "S02_BA", "P0710071"):
        assert stale not in source
