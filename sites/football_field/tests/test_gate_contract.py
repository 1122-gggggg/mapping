from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from ts_common import (  # noqa: E402
    Gate,
    GateDefinitionError,
    GateFreshnessError,
    assert_gate_fresh,
    required_check_ids,
    verify_predecessor_chain,
)


def make_gate(tmp_path: Path, required_ids: set[str]) -> Gate:
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "stage.py"
    artifact = tmp_path / "input.json"
    script.write_text("print('stage')\n", encoding="utf-8")
    artifact.write_text('{"input": true}\n', encoding="utf-8")
    return Gate(
        "test_stage",
        required_ids=required_ids,
        script_path=script,
        input_artifacts={"input": artifact},
    )


def test_zero_checks_is_not_green(tmp_path: Path) -> None:
    gate = make_gate(tmp_path, {"G1"})

    assert gate.ok is False
    assert gate.write(tmp_path / "run", fail_hard=False)["status"] == "NOT_RUN"


def test_empty_required_id_set_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(GateDefinitionError, match="required_ids must be non-empty"):
        Gate("test_stage", required_ids=[])


def test_duplicate_check_id_is_rejected(tmp_path: Path) -> None:
    gate = make_gate(tmp_path, {"G1"})
    gate.check("G1", True, "first evidence", count=1)

    with pytest.raises(GateDefinitionError, match="duplicate check id G1"):
        gate.check("G1", True, "duplicate evidence", count=1)


def test_unexpected_check_id_is_rejected(tmp_path: Path) -> None:
    gate = make_gate(tmp_path, {"G1"})

    with pytest.raises(GateDefinitionError, match="unexpected check id G2"):
        gate.check("G2", True, "wrong contract")


def test_missing_required_id_is_incomplete(tmp_path: Path) -> None:
    gate = make_gate(tmp_path, {"G1", "G2"})
    gate.check("G1", True, "substantive evidence", count=1)

    result = gate.write(tmp_path / "run", fail_hard=False)

    assert result["status"] == "INCOMPLETE"
    assert result["ok"] is False
    assert result["missing_ids"] == ["G2"]


def test_missing_input_artifact_is_not_run(tmp_path: Path) -> None:
    script = tmp_path / "stage.py"
    script.write_text("print('stage')\n", encoding="utf-8")
    gate = Gate(
        "test_stage",
        required_ids={"G1"},
        script_path=script,
        input_artifacts={"missing": tmp_path / "absent.json"},
    )
    gate.check("G1", True, "predicate would otherwise pass", count=1)

    result = gate.write(tmp_path / "run", fail_hard=False)

    assert result["status"] == "NOT_RUN"
    assert result["ok"] is False
    assert result["provenance"]["input_artifacts"]["missing"]["sha256"] is None


def test_complete_exact_set_passes(tmp_path: Path) -> None:
    gate = make_gate(tmp_path, {"G1", "G2"})
    gate.check("G1", True, "count matches", actual=7, expected=7)
    gate.check("G2", True, "hash matches", sha256="a" * 64)

    result = gate.write(tmp_path / "run", fail_hard=False)

    assert gate.ok is True
    assert result["status"] == "PASS"
    assert result["ok"] is True
    assert [check["id"] for check in result["checks"]] == ["G1", "G2"]
    assert (
        result["provenance"]["input_manifests"]
        == result["provenance"]["input_artifacts"]
    )


def test_release_stage_required_id_sets_are_exact() -> None:
    sequences = {
        "S01_ABrot",
        "S02_BA",
        "S03_BA2",
        "S04_ab",
        "S05_P1220122",
        "S06_P1240124",
        "S07_P1250125",
    }

    assert required_check_ids("S0_corpus") == {
        "G0.1a",
        "G0.1b",
        "G0.2a",
        "G0.2b",
        "G0.3",
        "G0.4",
        "G0.5",
        "G0.6",
        "G0.7",
        "G0.8",
    }
    assert required_check_ids("S1_motion") == (
        {"G0.2", "G1.4a", "G1.4b", "G1.4c", "G1.5", "G1.6"}
        | {f"G1.1/{seq}" for seq in sequences}
        | {f"G1.2/{seq}" for seq in sequences}
    )
    assert required_check_ids("S2_extract") == (
        {"G0.2", "G2.1", "G2.2", "G2.3a", "G2.4", "G2.5", "G2.6"}
        | {f"G1.3/{seq}" for seq in sequences}
    )
    assert required_check_ids("S2b_intrinsics") == {
        "G2.7/results_complete",
        "G2.7/1920x1080",
        "G2.7/2688x1512",
        "G2.7/3840x2160",
        "G2.8",
    }
    assert required_check_ids("S3_pairs") == {
        "G0.2",
        "G3.0",
        "G3.1",
        "G3.2",
        "G3.3",
        "G3.4",
        "G3.5a",
        "G3.5b",
        "G3.5c",
        "G3.5d",
        "G3.6",
    }


def test_fail_hard_rejects_every_non_pass_status(tmp_path: Path) -> None:
    gate = make_gate(tmp_path, {"G1", "G2"})
    gate.check("G1", True, "only one required check emitted", count=1)

    with pytest.raises(SystemExit, match="GATE INCOMPLETE"):
        gate.write(tmp_path / "run")


@pytest.mark.parametrize("mutated", ["script", "input"])
def test_freshness_rejects_mutation_after_gate_write(
    tmp_path: Path, mutated: str
) -> None:
    gate = make_gate(tmp_path, {"G1"})
    gate.check("G1", True, "fresh at write", count=1)
    gate_path = tmp_path / "run" / "gates" / "test_stage.json"
    gate.write(tmp_path / "run", fail_hard=False)
    target = tmp_path / ("stage.py" if mutated == "script" else "input.json")
    target.write_text("mutated\n", encoding="utf-8")

    with pytest.raises(GateFreshnessError, match="sha256 drift"):
        assert_gate_fresh(gate_path)


def test_skipped_optional_check_is_incomplete(tmp_path: Path) -> None:
    gate = make_gate(tmp_path, {"G1", "G_optional"})
    gate.check("G1", True, "required evidence", count=1)
    gate.incomplete("G_optional", "diagnostic mode skipped this evidence")

    result = gate.write(tmp_path / "run", fail_hard=False)

    assert result["status"] == "INCOMPLETE"
    assert result["ok"] is False


def test_absent_prerequisite_check_is_not_run(tmp_path: Path) -> None:
    gate = make_gate(tmp_path, {"G1"})
    gate.not_run("G1", "required upstream artifact is absent")

    result = gate.write(tmp_path / "run", fail_hard=False)

    assert result["status"] == "NOT_RUN"
    assert result["ok"] is False


def test_partial_s2b_result_set_is_incomplete(tmp_path: Path) -> None:
    gate = make_gate(tmp_path, {"G2.7/a", "G2.7/b", "G2.8"})
    gate.check("G2.7/a", True, "first solve complete", seeds=2)

    result = gate.write(tmp_path / "run", fail_hard=False)

    assert result["status"] == "INCOMPLETE"
    assert result["missing_ids"] == ["G2.7/b", "G2.8"]


def test_complete_threshold_violation_is_fail(tmp_path: Path) -> None:
    gate = make_gate(tmp_path, {"G2.7/a", "G2.8"})
    gate.check("G2.7/a", True, "both solves complete", seeds=2)
    gate.check("G2.8", False, "focal spread exceeds threshold", spread=0.2)

    result = gate.write(tmp_path / "run", fail_hard=False)

    assert result["status"] == "FAIL"
    assert result["ok"] is False


def test_predecessor_gate_hash_chain_detects_drift(tmp_path: Path) -> None:
    first = make_gate(tmp_path / "first", {"G1"})
    first.check("G1", True, "first stage evidence", count=1)
    first_path = tmp_path / "run" / "gates" / "first.json"
    first.write(tmp_path / "run", output_path=first_path, fail_hard=False)

    second_script = tmp_path / "second.py"
    second_input = tmp_path / "second_input.json"
    second_script.write_text("print('second')\n", encoding="utf-8")
    second_input.write_text("{}\n", encoding="utf-8")
    second = Gate(
        "second",
        required_ids={"G2"},
        script_path=second_script,
        input_artifacts={"input": second_input},
    )
    second.record_predecessor_gate("first", first_path, expected_stage="test_stage")
    second.check("G2", True, "second stage evidence", count=1)
    second_path = tmp_path / "run" / "gates" / "second.json"
    second.write(tmp_path / "run", output_path=second_path, fail_hard=False)

    assert verify_predecessor_chain([first_path, second_path]) is True

    payload = json.loads(first_path.read_text(encoding="utf-8"))
    payload["created_at"] = "tampered"
    first_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GateFreshnessError, match="predecessor hash mismatch"):
        verify_predecessor_chain([first_path, second_path])
