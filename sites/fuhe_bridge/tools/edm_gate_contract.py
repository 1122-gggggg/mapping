"""Shared fail-closed provenance checks for the Fuhe EDM release stages."""

from __future__ import annotations

from pathlib import Path

from ts_common import GateFreshnessError, assert_gate_fresh, read_json


def require_fresh_v2_gate(
    path: Path | str, *, expected_stage: str
) -> dict:
    """Require a green ``sfm-gate-v2`` gate whose material still hashes exactly."""
    gate_path = Path(path).expanduser().resolve(strict=True)
    payload = read_json(gate_path)
    if payload.get("schema_version") != "sfm-gate-v2":
        raise GateFreshnessError(
            f"predecessor must use sfm-gate-v2: {gate_path}"
        )
    if payload.get("stage") != expected_stage:
        raise GateFreshnessError(
            f"predecessor stage mismatch: expected {expected_stage!r}, "
            f"got {payload.get('stage')!r}"
        )
    if payload.get("status") != "PASS" or payload.get("ok") is not True:
        raise GateFreshnessError(f"predecessor gate is not PASS: {gate_path}")
    assert_gate_fresh(gate_path)
    return payload
