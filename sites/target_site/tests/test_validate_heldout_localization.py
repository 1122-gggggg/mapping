from __future__ import annotations

import sys
from pathlib import Path


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from ts_common import TEST  # noqa: E402
from validate_heldout_localization import (  # noqa: E402
    checks_pass,
    evaluate_results,
)


TEST_IDS = [video.seq for video in TEST]


def passing_result(sequence: str) -> dict:
    return {
        "video": sequence,
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


def evaluate_test_pair(first: dict, second: dict | None = None) -> dict:
    results = [first, second if second is not None else passing_result(TEST_IDS[1])]
    return evaluate_results(results, reverse_sequences={"REV"})


def test_two_green_test_videos_pass_without_reverse_fabrication() -> None:
    result = evaluate_results(
        [passing_result(TEST_IDS[0]), passing_result(TEST_IDS[1])],
        reverse_sequences={"REV"},
    )

    assert TEST_IDS == ["T01_P1230123", "T02_P1260126"]
    assert checks_pass(result["checks"])
    assert result["hard_status"] == "VALID"
    assert result["evidence_status"] == "PASS"
    assert result["status"] == "PASS"
    assert result["ok"] is True
    assert result["checks"]["G9.1"] is True
    assert result["checks"]["G9.3"]["status"] == "NOT_APPLICABLE"
    assert result["checks"]["G9.3"]["reason"]
    assert result["independent_session_count"] == 2
    assert result["identity_receipt"]["expected_ids"] == TEST_IDS
    assert result["identity_receipt"]["complete"] is True
    assert result["soft_checks"]["G9.1"]["observed"] == 0.97
    assert result["soft_checks"]["G9.1"]["required"] == 0.95
    assert result["soft_checks"]["G9.1"]["signed_margin"] == (
        result["soft_checks"]["G9.1"]["observed"]
        - result["soft_checks"]["G9.1"]["required"]
    )
    assert result["soft_checks"]["G9.3"]["status"] == "NOT_APPLICABLE"
    assert result["reverse_reference_diagnostic"]["status"] == "NOT_APPLICABLE"


def test_low_localization_rate_fails_acceptance() -> None:
    first = passing_result(TEST_IDS[0])
    first["localized"] = 940
    first["rate"] = 0.94

    result = evaluate_test_pair(first)

    assert result["hard_status"] == "VALID"
    assert result["evidence_status"] == "QUALITY_SHORTFALL"
    assert result["checks"]["G9.1"] is False
    assert result["checks"]["G9.3"]["status"] == "NOT_APPLICABLE"
    assert result["soft_checks"]["G9.1"]["observed"] == 0.94
    assert result["soft_checks"]["G9.1"]["signed_margin"] == (
        result["soft_checks"]["G9.1"]["observed"]
        - result["soft_checks"]["G9.1"]["required"]
    )
    assert checks_pass(result["checks"]) is False


def test_duplicate_test_identity_cannot_pass() -> None:
    result = evaluate_results(
        [passing_result(TEST_IDS[0]), passing_result(TEST_IDS[0])]
    )

    assert result["hard_status"] == "HARD_FAIL"
    assert result["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert result["identity_receipt"]["duplicates"] == [TEST_IDS[0]]
    assert result["independent_session_count"] == 1
    assert result["checks"]["G9.1"] is False
    assert checks_pass(result["checks"]) is False


def test_non_test_identity_cannot_pass() -> None:
    result = evaluate_results(
        [passing_result(TEST_IDS[0]), passing_result("FWD")]
    )

    assert result["hard_status"] == "HARD_FAIL"
    assert result["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert "FWD" in result["identity_receipt"]["unknown_ids"]
    assert TEST_IDS[1] in result["identity_receipt"]["missing_ids"]
    assert result["checks"]["G9.1"] is False
    assert checks_pass(result["checks"]) is False


def test_anonymous_result_cannot_pass() -> None:
    anonymous = passing_result(TEST_IDS[0])
    del anonymous["video"]

    result = evaluate_results([anonymous, passing_result(TEST_IDS[1])])

    assert result["hard_status"] == "HARD_FAIL"
    assert result["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert result["identity_receipt"]["complete"] is False
    assert checks_pass(result["checks"]) is False


def test_zero_frame_result_cannot_pass() -> None:
    first = passing_result(TEST_IDS[0])
    first["frames"] = 0
    first["localized"] = 0
    first["rate"] = 0.0

    result = evaluate_test_pair(first)

    assert result["hard_status"] == "HARD_FAIL"
    assert result["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert result["independent_session_count"] == 1
    assert result["checks"]["G9.1"] is False
    assert checks_pass(result["checks"]) is False


def test_inconsistent_rate_cannot_pass() -> None:
    first = passing_result(TEST_IDS[0])
    first["rate"] = 0.99

    result = evaluate_test_pair(first)

    assert first["localized"] / first["frames"] == 0.97
    assert result["hard_status"] == "HARD_FAIL"
    assert result["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert result["checks"]["G9.1"] is False
    assert checks_pass(result["checks"]) is False


def _heldout_contract(total_frames: int) -> list[dict[str, str | int]]:
    return [
        {
            "video_id": TEST_IDS[0],
            "video_sha256": "a" * 64,
            "total_frames": total_frames,
        },
        {
            "video_id": TEST_IDS[1],
            "video_sha256": "b" * 64,
            "total_frames": total_frames,
        },
    ]


def test_contract_frame_count_mismatch_is_hard_fail() -> None:
    first = passing_result(TEST_IDS[0])
    first["frames"] = 1
    first["localized"] = 1
    first["rate"] = 1.0

    result = evaluate_results(
        [first, passing_result(TEST_IDS[1])],
        heldout_contract=_heldout_contract(1000),
    )

    assert result["hard_status"] == "HARD_FAIL"
    assert result["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert result["status"] == "FAIL"
    bindings = result["identity_receipt"]["corpus_bindings"]
    assert bindings[TEST_IDS[0]]["bound"] is False
    assert bindings[TEST_IDS[0]]["result_frames"] == 1
    assert bindings[TEST_IDS[0]]["total_frames"] == 1000
    assert any("result frames" in item for item in result["hard_failures"])
    assert checks_pass(result["checks"]) is False
