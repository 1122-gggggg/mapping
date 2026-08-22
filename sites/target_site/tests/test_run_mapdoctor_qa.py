from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

import run_mapdoctor_qa  # noqa: E402


def _install_fake_mapdoctor(monkeypatch: pytest.MonkeyPatch, captured: list[list[str]]) -> None:
    def fake_main(argv: list[str]) -> int:
        captured.append(list(argv))
        return 0

    pkg = types.ModuleType("mapdoctor")
    cli = types.ModuleType("mapdoctor.cli")
    cli.main = fake_main
    monkeypatch.setitem(sys.modules, "mapdoctor", pkg)
    monkeypatch.setitem(sys.modules, "mapdoctor.cli", cli)


def test_missing_mapdoctor_raises_install_message(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "cameras.bin").touch()
    monkeypatch.setitem(sys.modules, "mapdoctor", None)
    monkeypatch.setitem(sys.modules, "mapdoctor.cli", None)

    with pytest.raises(SystemExit, match="MapDoctor") as excinfo:
        run_mapdoctor_qa.main(["--model", str(model), "--output", str(tmp_path / "out")])
    assert "install" in str(excinfo.value).lower()


def test_output_parent_is_created(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "cameras.bin").touch()
    output = tmp_path / "reports" / "nested" / "out"
    captured: list[list[str]] = []
    _install_fake_mapdoctor(monkeypatch, captured)

    rc = run_mapdoctor_qa.main(["--model", str(model), "--output", str(output)])

    assert rc == 0
    assert output.is_dir()
    assert captured == [["gluemap", str(model), "--output", str(output)]]


def test_parent_dir_with_nested_zero_model_is_resolved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parent = tmp_path / "sparse"
    nested = parent / "0"
    nested.mkdir(parents=True)
    (nested / "cameras.bin").touch()
    output = tmp_path / "out"
    captured: list[list[str]] = []
    _install_fake_mapdoctor(monkeypatch, captured)

    rc = run_mapdoctor_qa.main(["--model", str(parent), "--output", str(output)])

    assert rc == 0
    assert captured == [["gluemap", str(nested), "--output", str(output)]]


def test_missing_model_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[list[str]] = []
    _install_fake_mapdoctor(monkeypatch, captured)
    missing = tmp_path / "does-not-exist"
    with pytest.raises(SystemExit, match="cameras"):
        run_mapdoctor_qa.main(["--model", str(missing), "--output", str(tmp_path / "out")])
    assert captured == []
