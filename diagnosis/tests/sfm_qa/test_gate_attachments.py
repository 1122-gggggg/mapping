from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from sfm_qa.gate_attachments import load_run_attachments
from sfm_qa.pipeline import analyze


def _generate_demo(tmp_path: Path):
    path = Path(__file__).resolve().parents[2] / "examples" / "reproducible_demo" / "generate_demo.py"
    spec = importlib.util.spec_from_file_location("generate_demo", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.generate(tmp_path)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_list_checks_become_findings(tmp_path):
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "gates" / "S0_S3_release.json",
        {
            "checks": [
                {"id": "G0.1", "ok": True},
                {"id": "G0.2", "ok": False},
                {"name": "G0.3", "state": "PASS"},
            ]
        },
    )

    findings = load_run_attachments(run_dir)["findings"]

    assert findings == [
        {"id": "G0.1", "ok": True, "source": "gates/S0_S3_release.json"},
        {"id": "G0.2", "ok": False, "source": "gates/S0_S3_release.json"},
        {"id": "G0.3", "ok": True, "source": "gates/S0_S3_release.json"},
    ]


def test_dict_checks_become_findings(tmp_path):
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "verify_final_release.json",
        {"checks": {"all_S0_S9_gates_pass": True, "release_lineage_bound": False}},
    )
    _write_json(
        run_dir / "nested" / "validate_heldout.json",
        {"checks": {"G9.1": {"ok": True}, "G9.2": {"pass": False}}},
    )
    _write_json(
        run_dir / "audit_map_geometry.json",
        {"checks": {"G6.1": "PASS", "G6.2": "FAIL"}},
    )

    findings = load_run_attachments(run_dir)["findings"]
    by_id = {item["id"]: item for item in findings}

    assert by_id["all_S0_S9_gates_pass"] == {
        "id": "all_S0_S9_gates_pass",
        "ok": True,
        "source": "verify_final_release.json",
    }
    assert by_id["release_lineage_bound"]["ok"] is False
    assert by_id["G9.1"] == {
        "id": "G9.1",
        "ok": True,
        "source": "nested/validate_heldout.json",
    }
    assert by_id["G9.2"]["ok"] is False
    assert by_id["G6.1"]["ok"] is True
    assert by_id["G6.2"]["ok"] is False
    assert by_id["G6.2"]["source"] == "audit_map_geometry.json"


def test_missing_run_dir_returns_empty_findings(tmp_path):
    assert load_run_attachments(tmp_path / "missing") == {"findings": []}
    assert load_run_attachments(None) == {"findings": []}


def test_corrupt_json_is_skipped(tmp_path):
    run_dir = tmp_path / "run"
    gates = run_dir / "gates"
    gates.mkdir(parents=True)
    (gates / "broken.json").write_text("{not-json", encoding="utf-8")
    _write_json(gates / "good.json", {"checks": {"kept": True}})

    findings = load_run_attachments(run_dir)["findings"]

    assert findings == [{"id": "kept", "ok": True, "source": "gates/good.json"}]


def test_analyze_includes_attachments_without_flipping_ready(tmp_path):
    paths = _generate_demo(tmp_path / "demo")
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "gates" / "S9_heldout_localization.json",
        {"checks": {"heldout_rate": False, "identity_bound": True}},
    )

    report = analyze(
        paths["model"],
        backend="gluemap",
        logs_path=paths["base"],
        run_dir=run_dir,
    )

    assert report["overall_status"] == "READY"
    findings = report["gate_attachments"]["findings"]
    assert {"id": "heldout_rate", "ok": False, "source": "gates/S9_heldout_localization.json"} in findings
    assert any(item["id"] == "identity_bound" and item["ok"] is True for item in findings)
