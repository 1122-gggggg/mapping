from __future__ import annotations

import sys
from pathlib import Path


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from validate_heldout_localization import evaluate_results  # noqa: E402


def passing_result(sequence: str) -> dict:
    return {
        "frames": 1000,
        "localized": 970,
        "rate": 0.97,
        "inliers_p05": 40.0,
        "step_median": 0.01,
        "step_p95": 0.05,
        "jumps_gt_10x_median": 0,
        "rejections": {"jump": 1},
        "reference_sequence_counts": {sequence: 100},
    }


def test_two_green_runs_with_reverse_support_pass_all_gates() -> None:
    result = evaluate_results(
        [passing_result("FWD"), passing_result("REV")], reverse_sequences={"REV"}
    )

    assert all(result["checks"].values())


def test_low_localization_rate_fails_acceptance() -> None:
    first = passing_result("FWD")
    first["rate"] = 0.94

    result = evaluate_results([first, passing_result("REV")], reverse_sequences={"REV"})

    assert result["checks"]["G9.1"] is False
