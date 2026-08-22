from __future__ import annotations

import json
import sys
from pathlib import Path


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from ts_common import TEST  # noqa: E402
from validate_heldout_localization import (  # noqa: E402
    evaluate_results,
    heldout_contract_from_manifest,
)


HELDOUT_SEQ = TEST[0].seq
VIDEO_SHA256 = "d" * 64


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


def test_one_green_heldout_run_passes_without_reverse_support() -> None:
    result = evaluate_results([passing_result(HELDOUT_SEQ)])

    assert result["checks"]["G9.1"] is True
    assert result["checks"]["G9.2"] is True
    assert result["checks"]["G9.3"] == "NOT_APPLICABLE"
    assert result["checks"]["G9.4"] is True
    assert result["checks"]["G9.5"] is True
    assert result["heldout_video_id"] == HELDOUT_SEQ


def test_missing_heldout_result_fails_g91() -> None:
    result = evaluate_results([])

    assert result["checks"]["G9.1"] is False
    assert result["checks"]["G9.3"] == "NOT_APPLICABLE"


def test_low_localization_rate_fails_acceptance() -> None:
    first = passing_result(HELDOUT_SEQ)
    first["rate"] = 0.94

    result = evaluate_results([first])

    assert result["checks"]["G9.1"] is False
    assert result["checks"]["G9.3"] == "NOT_APPLICABLE"


def test_heldout_contract_binds_test_video_from_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "test": [
                    {
                        "seq": HELDOUT_SEQ,
                        "source_sha256": VIDEO_SHA256,
                        "probed": {"nb_frames": 1488},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert heldout_contract_from_manifest(manifest) == {
        "video_id": HELDOUT_SEQ,
        "video_sha256": VIDEO_SHA256,
        "total_frames": 1488,
    }
