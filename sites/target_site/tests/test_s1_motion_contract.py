from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from s1_motion_scan import (  # noqa: E402
    MIN_INLIERS,
    classify,
    emit_pure_rotation_replay_gate,
    smooth,
)
from ts_common import Gate  # noqa: E402


def metrics(
    *,
    inliers: int = MIN_INLIERS,
    rot_deg_s: float = 6.0,
    parallax_deg: float = 0.2,
) -> dict[str, float | int]:
    return {
        "tracks": 100,
        "median_flow_px": 10.0,
        "inliers": inliers,
        "rot_deg_s": rot_deg_s,
        "parallax_deg": parallax_deg,
    }


def test_low_evidence_records_are_unproven_not_pure_rotation() -> None:
    low_inlier_class, _ = classify(metrics(inliers=MIN_INLIERS - 1), 10.0)
    nonfinite_rot_class, _ = classify(metrics(rot_deg_s=np.nan), 10.0)
    nonfinite_parallax_class, _ = classify(metrics(parallax_deg=np.nan), 10.0)

    assert low_inlier_class == "unproven"
    assert nonfinite_rot_class == "unproven"
    assert nonfinite_parallax_class == "unproven"
    assert "pure_rotation" not in {
        low_inlier_class,
        nonfinite_rot_class,
        nonfinite_parallax_class,
    }


def test_threshold_violations_are_never_pure_rotation() -> None:
    low_rotation_class, _ = classify(metrics(rot_deg_s=4.999), 10.0)
    high_parallax_class, _ = classify(metrics(parallax_deg=0.35), 10.0)
    valid_class, _ = classify(metrics(rot_deg_s=5.0, parallax_deg=0.349), 10.0)

    assert low_rotation_class != "pure_rotation"
    assert high_parallax_class != "pure_rotation"
    assert valid_class == "pure_rotation"


def test_smoothing_cannot_manufacture_pure_rotation() -> None:
    classes = ["pure_rotation", "parallax", "pure_rotation"]

    assert smooth(classes) == classes


def test_g1_6_rejects_violating_manifest_and_accepts_clean_manifest(
    tmp_path: Path,
) -> None:
    script = tmp_path / "stage.py"
    input_manifest = tmp_path / "input.json"
    script.write_text("# synthetic stage\n", encoding="utf-8")
    input_manifest.write_text("{}\n", encoding="utf-8")

    violating = {
        "sequences": {
            "S01_ABrot": {
                "records": [
                    {
                        "t": 1.0,
                        "motion_class": "pure_rotation",
                        "rot_deg_s": 4.0,
                        "parallax_deg": 0.2,
                    }
                ]
            }
        }
    }
    failing_gate = Gate(
        "test_s1",
        {"G1.6"},
        script_path=script,
        input_artifacts={"manifest": input_manifest},
    )
    emit_pure_rotation_replay_gate(failing_gate, violating)

    assert failing_gate.checks[0]["id"] == "G1.6"
    assert failing_gate.checks[0]["state"] == "FAIL"
    assert failing_gate.checks[0]["metrics"]["n_violations"] == 1

    clean = {
        "sequences": {
            "S01_ABrot": {
                "records": [
                    {
                        "t": 1.0,
                        "motion_class": "pure_rotation",
                        "rot_deg_s": 5.0,
                        "parallax_deg": 0.349,
                    },
                    {
                        "t": 2.0,
                        "motion_class": "parallax",
                        "rot_deg_s": 4.0,
                        "parallax_deg": 0.8,
                    },
                ]
            }
        }
    }
    passing_gate = Gate(
        "test_s1",
        {"G1.6"},
        script_path=script,
        input_artifacts={"manifest": input_manifest},
    )
    emit_pure_rotation_replay_gate(passing_gate, clean)

    assert passing_gate.checks[0]["state"] == "PASS"
    assert passing_gate.checks[0]["metrics"]["n_records"] == 2
    assert passing_gate.checks[0]["metrics"]["n_pure"] == 1
