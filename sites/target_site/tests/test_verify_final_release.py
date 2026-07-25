from __future__ import annotations

import json
import sys
from pathlib import Path


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from verify_final_release import gate_statuses_pass  # noqa: E402


def test_gate_statuses_require_every_gate_to_pass(tmp_path: Path) -> None:
    paths = []
    for index in range(3):
        path = tmp_path / f"gate{index}.json"
        path.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
        paths.append(path)

    assert gate_statuses_pass(paths)[0] is True

    paths[1].write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")
    passed, evidence = gate_statuses_pass(paths)
    assert passed is False
    assert evidence[str(paths[1])] == "FAIL"
