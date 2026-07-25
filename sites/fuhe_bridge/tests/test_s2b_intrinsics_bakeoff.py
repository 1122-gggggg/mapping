from __future__ import annotations

import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import s2b_intrinsics_bakeoff as s2b  # noqa: E402
import verify_s0_s3_release as release  # noqa: E402
from ts_common import Gate, write_json  # noqa: E402


LITERAL_KEYS = frozenset(
    {
        ("1920x1080", "official69"),
        ("1920x1080", "charuco"),
    }
)


def _result(shape: str, seed: str, fx_over_w: float = 0.728) -> dict:
    return {
        "logical_key": f"{shape}/{seed}",
        "shape": shape,
        "seed": seed,
        "fx_over_w": fx_over_w,
        "moved_pct": 0.0,
        "camera_model": "PINHOLE",
        "camera_count": 1,
        "registered": 10,
        "n_frames": 10,
        "mean_reproj": 1.0,
        "probe_input_sha256": "probe-hash",
    }


def _gate() -> Gate:
    return Gate("S2b_intrinsics", s2b.S2B_REQUIRED_IDS)


def _write_complete_marker(root: Path, shape: str, seed: str) -> None:
    work = root / f"{shape}__{seed}"
    attempt_id = f"attempt-{shape}-{seed}"
    write_json(
        work / "result.json",
        {**_result(shape, seed), "attempt_id": attempt_id},
    )
    write_json(
        work / "status.json",
        {
            "logical_key": f"{shape}/{seed}",
            "state": "COMPLETE",
            "attempt_id": attempt_id,
        },
    )


def test_legacy_result_keys_are_limited_to_the_working_resolution() -> None:
    assert s2b.REQUIRED_RESULT_KEYS == LITERAL_KEYS


def test_failed_import_preserves_the_salad_descriptor_cache(tmp_path: Path) -> None:
    source = tmp_path / "archive" / "3840x2160__charuco"
    target = tmp_path / "fresh" / "3840x2160__charuco"
    cache = source / "gluemap" / "S03_BA2" / "salad_descriptors.pt"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"validated descriptor cache")

    record = s2b._copy_salad_descriptor_cache(source, target)

    copied = target / "gluemap" / "S03_BA2" / "salad_descriptors.pt"
    assert copied.read_bytes() == cache.read_bytes()
    assert record["sha256"] == s2b.sha256(cache)


def test_working_resolution_results_emit_every_legacy_helper_id(
    tmp_path: Path,
) -> None:
    for seed in s2b.SEEDS:
        _write_complete_marker(tmp_path, "1920x1080", seed)
    results, statuses = s2b.load_result_markers(tmp_path, validate=False)
    gate = _gate()

    s2b.emit_aggregate_checks(gate, results, statuses)

    assert {check["id"] for check in gate.checks} == s2b.S2B_REQUIRED_IDS
    assert gate.missing_ids == set()
    assert all(check["state"] == "PASS" for check in gate.checks)


def test_six_consistent_results_can_pass_every_required_check(tmp_path: Path) -> None:
    for shape, seed in LITERAL_KEYS:
        _write_complete_marker(tmp_path, shape, seed)
    results, statuses = s2b.load_result_markers(tmp_path, validate=False)
    gate = _gate()

    s2b.emit_aggregate_checks(gate, results, statuses)

    assert {check["id"] for check in gate.checks} == s2b.S2B_REQUIRED_IDS
    assert all(check["state"] == "PASS" for check in gate.checks)


@pytest.mark.parametrize("state", ["RUNNING", "FAILED", "NOT_RUN"])
def test_noncomplete_attempt_state_invalidates_an_older_result(
    tmp_path: Path, state: str
) -> None:
    work = tmp_path / "1920x1080__official69"
    write_json(
        work / "result.json",
        {**_result("1920x1080", "official69"), "attempt_id": "old"},
    )
    write_json(
        work / "status.json",
        {
            "logical_key": "1920x1080/official69",
            "state": state,
            "attempt_id": "new",
        },
    )

    results, statuses = s2b.load_result_markers(tmp_path, validate=False)

    assert results == {}
    assert statuses[("1920x1080", "official69")]["state"] == state


def test_duplicate_logical_keys_are_rejected(tmp_path: Path) -> None:
    for dirname in ("first", "second"):
        write_json(
            tmp_path / dirname / "status.json",
            {
                "logical_key": "1920x1080/official69",
                "state": "NOT_RUN",
                "attempt_id": dirname,
            },
        )

    with pytest.raises(ValueError, match="duplicate logical key"):
        s2b.load_result_markers(tmp_path, validate=False)


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"fx_over_w": float("nan")}, "finite"),
        ({"camera_model": "SIMPLE_PINHOLE"}, "PINHOLE"),
        ({"camera_count": 2}, "one camera"),
        ({"probe_input_sha256": "wrong"}, "probe hash"),
    ],
)
def test_result_payload_rejects_invalid_measurements(update: dict, match: str) -> None:
    payload = _result("1920x1080", "official69")
    payload.update(update)

    with pytest.raises(ValueError, match=match):
        s2b.validate_result_payload(payload, expected_probe_hash="probe-hash")


def test_complete_result_requires_fresh_generating_provenance() -> None:
    payload = _result("1920x1080", "official69")

    with pytest.raises(ValueError, match="generating provenance"):
        s2b.validate_result_payload(payload, expected_probe_hash="probe-hash")


def test_complete_result_rejects_imported_from_laundering() -> None:
    payload = _result("1920x1080", "official69")
    payload["generating_provenance"] = {
        "mode": "fresh_solve",
        "runtime_fingerprint": {"version": "4.0.4"},
        "scientific_config": {"refine_intrinsics": True},
        "sources": {"solver": {"sha256": "a" * 64}},
        "checkpoints": {"pi3": {"sha256": "b" * 64}},
    }
    payload["imported_from"] = "/archive/old-result"

    with pytest.raises(ValueError, match="imported_from"):
        s2b.validate_result_payload(payload, expected_probe_hash="probe-hash")


def test_external_official69_camera_policy_is_hash_bound_and_exact() -> None:
    policy = s2b.external_camera_policy()

    assert policy["schema_version"] == "fuhe-intrinsics-policy-v2"
    assert policy["camera_policy"]["state"] == "PASS"
    assert policy["camera_policy"]["applicable"] is True
    assert policy["camera_policy"]["external_record"] == {
        "path": str(s2b.EXTERNAL_CAMERA_RECORD),
        "sha256": "65b1b50dff22935711263ab9b546cbbe1dc0f2c3443782e83c3be7e4def03903",
    }
    assert policy["camera_policy"]["camera"] == {
        "model": "PINHOLE",
        "width": 1920,
        "height": 1080,
        "params": [1396.8086675255472, 1396.8086675255472, 960.0, 540.0],
    }
    assert policy["camera_policy"]["undistort"] is False
    assert policy["camera_policy"]["resize"] == "raw INTER_AREA to 1920x1080"
    assert policy["diagnostics"]["two_seed"]["applicable"] is False
    assert policy["diagnostics"]["cross_resolution"]["applicable"] is False


def test_s2b_gate_records_camera_policy_pass_and_only_diagnostics_na() -> None:
    gate = _gate()

    s2b.emit_external_camera_policy_checks(gate, s2b.external_camera_policy())

    states = {check["id"]: check["state"] for check in gate.checks}
    assert states == {
        "G2.7/results_complete": "PASS",
        "G2.7/1920x1080": "NOT_APPLICABLE",
        "G2.8": "NOT_APPLICABLE",
    }


def test_release_verifier_rejects_external_camera_policy_tampering(
    tmp_path: Path,
) -> None:
    policy = s2b.external_camera_policy()
    write_json(tmp_path / "intrinsics_policy.json", policy)

    assert release.validate_s2b_semantics(tmp_path)["ok"] is True

    policy["camera_policy"]["external_record"]["sha256"] = "0" * 64
    write_json(tmp_path / "intrinsics_policy.json", policy)
    assert release.validate_s2b_semantics(tmp_path)["ok"] is False

    policy = s2b.external_camera_policy()
    policy["camera_policy"]["camera"]["params"][0] += 1e-9
    write_json(tmp_path / "intrinsics_policy.json", policy)
    assert release.validate_s2b_semantics(tmp_path)["ok"] is False
