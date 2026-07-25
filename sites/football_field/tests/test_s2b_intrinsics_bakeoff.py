from __future__ import annotations

import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import s2b_intrinsics_bakeoff as s2b  # noqa: E402
from ts_common import Gate, write_json  # noqa: E402


LITERAL_KEYS = frozenset(
    {
        ("1920x1080", "official69"),
        ("1920x1080", "charuco"),
        ("2688x1512", "official69"),
        ("2688x1512", "charuco"),
        ("3840x2160", "official69"),
        ("3840x2160", "charuco"),
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


def test_required_result_keys_are_the_literal_six_key_experiment() -> None:
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


def test_partial_results_emit_every_required_id_and_remain_incomplete(
    tmp_path: Path,
) -> None:
    for seed in s2b.SEEDS:
        _write_complete_marker(tmp_path, "1920x1080", seed)
    results, statuses = s2b.load_result_markers(tmp_path, validate=False)
    gate = _gate()

    s2b.emit_aggregate_checks(gate, results, statuses)

    assert {check["id"] for check in gate.checks} == s2b.S2B_REQUIRED_IDS
    assert gate.missing_ids == set()
    assert any(check["state"] == "INCOMPLETE" for check in gate.checks)
    assert next(c for c in gate.checks if c["id"] == "G2.8")["state"] == "INCOMPLETE"


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
