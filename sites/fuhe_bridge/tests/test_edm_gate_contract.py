from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


FUHE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FUHE / "tools"))

from edm_gate_contract import require_fresh_v2_gate  # noqa: E402
from ts_common import Gate, GateFreshnessError  # noqa: E402


def _passing_v2_gate(tmp_path: Path) -> Path:
    script = tmp_path / "producer.py"
    source = tmp_path / "source.py"
    material = tmp_path / "material.bin"
    script.write_text("# producer\n", encoding="utf-8")
    source.write_text("VALUE = 1\n", encoding="utf-8")
    material.write_bytes(b"material-v1")
    gate = Gate(
        "S7_tracking_bundle",
        {"G7.test"},
        script_path=script,
        source_files=[source],
        input_artifacts={"material": material},
    )
    gate.check("G7.test", True, "fixture passes", value=1)
    output = tmp_path / "S7_tracking_bundle.json"
    gate.write(tmp_path, output_path=output)
    return output


def test_require_fresh_v2_gate_rejects_legacy_green_payload(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps({"stage": "S7_tracking_bundle", "status": "PASS", "ok": True}),
        encoding="utf-8",
    )

    with pytest.raises(GateFreshnessError, match="sfm-gate-v2"):
        require_fresh_v2_gate(legacy, expected_stage="S7_tracking_bundle")


def test_require_fresh_v2_gate_detects_material_drift(tmp_path: Path) -> None:
    gate_path = _passing_v2_gate(tmp_path)
    require_fresh_v2_gate(gate_path, expected_stage="S7_tracking_bundle")
    (tmp_path / "material.bin").write_bytes(b"material-v2")

    with pytest.raises(GateFreshnessError, match="sha256 drift"):
        require_fresh_v2_gate(gate_path, expected_stage="S7_tracking_bundle")
