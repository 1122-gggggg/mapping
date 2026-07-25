from __future__ import annotations

import sys
from pathlib import Path

import pytest


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

import audit_independent_sim3  # noqa: E402
import audit_map_geometry  # noqa: E402
import finalize_edm_model  # noqa: E402


@pytest.mark.parametrize(
    ("module", "disk_option"),
    [
        (finalize_edm_model, "--output-model"),
        (audit_independent_sim3, "--work-dir"),
        (audit_map_geometry, "--out"),
    ],
)
def test_s5_s57_s6_cli_enters_global_guard_before_locked_body(
    monkeypatch, tmp_path: Path, module, disk_option: str
) -> None:
    events: list[str] = []
    monkeypatch.setattr(sys, "argv", [module.__file__, disk_option, str(tmp_path)])
    monkeypatch.setattr(module, "_main_locked", lambda: events.append("body"))

    def guarded(path, operation):
        events.append(f"guard:{Path(path).resolve()}")
        return operation({"ok": True})

    monkeypatch.setattr(module, "run_global_heavy_job", guarded)

    module.main()

    assert events == [f"guard:{tmp_path.resolve()}", "body"]
