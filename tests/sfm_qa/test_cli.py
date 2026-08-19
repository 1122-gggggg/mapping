from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from sfm_qa.cli import main
from sfm_qa.pipeline import is_success_status


def _generate_demo(tmp_path: Path):
    path = Path(__file__).resolve().parents[2] / "examples" / "reproducible_demo" / "generate_demo.py"
    spec = importlib.util.spec_from_file_location("generate_demo", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.generate(tmp_path)


def test_cli_check_base_exits_consistently(tmp_path):
    paths = _generate_demo(tmp_path / "demo")
    out = tmp_path / "out"
    code = main(
        [
            "check",
            str(paths["model"]),
            "--backend",
            "gluemap",
            "--logs",
            str(paths["base"]),
            "--output",
            str(out),
        ]
    )
    report_path = out / "report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["overall_status"] == "READY"
    assert code == 0
    assert code == (0 if is_success_status(report["overall_status"]) else 1)


def test_cli_check_candidate_exits_one(tmp_path):
    paths = _generate_demo(tmp_path / "demo")
    out = tmp_path / "out"
    code = main(
        [
            "check",
            str(paths["model"]),
            "--backend",
            "gluemap",
            "--logs",
            str(paths["candidate"]),
            "--output",
            str(out),
        ]
    )
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["overall_status"] == "LOCALIZATION_FAILED"
    assert code == 1
    assert code == (0 if is_success_status(report["overall_status"]) else 1)


def test_cli_analyze_without_logs_prints_stages(tmp_path, capsys):
    paths = _generate_demo(tmp_path / "demo")
    code = main(
        [
            "analyze",
            str(paths["model"]),
            "--backend",
            "gluemap",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr().out
    assert code == 0
    stage1 = captured.index("=== Stage 1: map diagnosis ===")
    stage2 = captured.index("=== Stage 2: SfM localization ===")
    skipped = captured.index("(skipped: no --logs)")
    overall = captured.index("=== overall_status:")
    assert stage1 < stage2 < skipped < overall
