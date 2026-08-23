from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import diagnose_map


def _install_fake_analyze(monkeypatch: pytest.MonkeyPatch, captured: list[dict], status: str = "MAP_SCREENED_LOCALIZATION_UNCHECKED") -> None:
    def fake_analyze(model_path, **kwargs):
        captured.append({"model_path": Path(model_path), **kwargs})
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "report.json").write_text("{}", encoding="utf-8")
        return {"overall_status": status}

    def fake_is_success(overall_status: str) -> bool:
        return overall_status in {"READY", "MAP_SCREENED_LOCALIZATION_UNCHECKED"}

    pkg = types.ModuleType("sfm_qa")
    pipeline = types.ModuleType("sfm_qa.pipeline")
    pipeline.analyze = fake_analyze
    pipeline.is_success_status = fake_is_success
    monkeypatch.setitem(sys.modules, "sfm_qa", pkg)
    monkeypatch.setitem(sys.modules, "sfm_qa.pipeline", pipeline)


def test_missing_diagnosis_install_message(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "cameras.bin").touch()
    monkeypatch.setitem(sys.modules, "sfm_qa", None)
    monkeypatch.setitem(sys.modules, "sfm_qa.pipeline", None)
    with pytest.raises(SystemExit, match="pip install") as excinfo:
        diagnose_map.main(["--model", str(model), "--output", str(tmp_path / "out")])
    assert "Diagnosis packages" in str(excinfo.value)


def test_nested_model_is_resolved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parent = tmp_path / "sparse"
    nested = parent / "0"
    nested.mkdir(parents=True)
    (nested / "cameras.bin").touch()
    output = tmp_path / "out"
    captured: list[dict] = []
    _install_fake_analyze(monkeypatch, captured)

    rc = diagnose_map.main(["--model", str(parent), "--output", str(output), "--backend", "colmap"])

    assert rc == 0
    assert captured[0]["model_path"] == nested
    assert captured[0]["backend"] == "colmap"
    assert captured[0]["output_dir"] == output


def test_missing_model_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[dict] = []
    _install_fake_analyze(monkeypatch, captured)
    with pytest.raises(SystemExit, match="cameras"):
        diagnose_map.main(["--model", str(tmp_path / "missing"), "--output", str(tmp_path / "out")])
    assert captured == []


def test_failed_status_exits_nonzero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "cameras.bin").touch()
    captured: list[dict] = []
    _install_fake_analyze(monkeypatch, captured, status="MAP_SCREENING_FAILED")
    rc = diagnose_map.main(["--model", str(model), "--output", str(tmp_path / "out")])
    assert rc == 1
