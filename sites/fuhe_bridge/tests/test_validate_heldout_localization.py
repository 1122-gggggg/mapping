from __future__ import annotations

import sys
from pathlib import Path


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from validate_heldout_localization import (  # noqa: E402
    evaluate_results,
    heldout_contract_from_manifest,
)


VIDEO_SHA256 = "d" * 64
PROFILE_SHA256 = "a" * 64
PACKAGE_CONFIG_SHA256 = "b" * 64
EDM_BUNDLE_SHA256 = "c" * 64
TRACKING_BUNDLE_SHA256 = "e" * 64
BASELINE_SHA256 = (
    "39a817936c0ba314a739701411f974672a94f126830f9f5b9a7a4efdfee08117"
)
CAMERA_CONTRACT_SHA256 = "f" * 64


def passing_result(sequence: str, scope: str, frame_range: list[int]) -> dict:
    return {
        "video": "/queries/P1130113.MP4",
        "frames": frame_range[1] - frame_range[0],
        "localized": int(0.97 * (frame_range[1] - frame_range[0])),
        "rate": 0.97,
        "inliers_p05": 40.0,
        "step_median": 0.01,
        "step_p95": 0.05,
        "jumps_gt_10x_median": 0,
        "rejections": {"jump": 1},
        "reference_sequence_counts": {sequence: 100},
        "runtime": {
            "deployment_profile": "fuhe_bridge_balanced",
            "deployment_profile_sha256": PROFILE_SHA256,
            "package_config_sha256": PACKAGE_CONFIG_SHA256,
            "edm_bundle_sha256": EDM_BUNDLE_SHA256,
            "tracking_bundle_sha256": TRACKING_BUNDLE_SHA256,
            "anchor_density_baseline_sha256": BASELINE_SHA256,
            "anchor_density_baseline_median": 3811.5,
            "camera_contract": "benchmark_camera_contract",
            "benchmark_camera_contract_sha256": CAMERA_CONTRACT_SHA256,
        },
        "evaluation": {
            "scope": scope,
            "source_video_id": "P1130113",
            "source_video_sha256": VIDEO_SHA256,
            "source_total_frames": 1488,
            "source_frame_range": frame_range,
            "complete_scope": True,
            "diagnostic_segment": scope != "full",
            "independent_flight": scope == "full",
        },
    }


def passing_evidence() -> list[dict]:
    return [
        passing_result("REV", "full", [0, 1488]),
        passing_result("REV", "diagnostic_first_half", [0, 744]),
        passing_result("REV", "diagnostic_second_half", [744, 1488]),
    ]


def lineage_kwargs() -> dict:
    return {
        "expected_package_config_sha256": PACKAGE_CONFIG_SHA256,
        "expected_edm_bundle_sha256": EDM_BUNDLE_SHA256,
        "expected_tracking_bundle_sha256": TRACKING_BUNDLE_SHA256,
        "expected_baseline_sha256": BASELINE_SHA256,
        "expected_baseline_median": 3811.5,
        "expected_camera_contract_sha256": CAMERA_CONTRACT_SHA256,
    }


def test_one_full_video_and_two_diagnostic_halves_pass_without_claiming_two_flights() -> None:
    result = evaluate_results(
        passing_evidence(),
        reverse_sequences={"REV"},
        expected_video_id="P1130113",
        expected_video_sha256=VIDEO_SHA256,
        expected_total_frames=1488,
        expected_profile_sha256=PROFILE_SHA256,
        **lineage_kwargs(),
    )

    assert all(result["checks"].values())
    assert result["validation_semantics"] == {
        "independent_video_count": 1,
        "diagnostic_segment_count": 2,
        "diagnostic_segments_are_independent_flights": False,
        "source_video_id": "P1130113",
        "source_video_sha256": VIDEO_SHA256,
    }


def test_low_localization_rate_fails_acceptance() -> None:
    results = passing_evidence()
    results[1]["rate"] = 0.94

    result = evaluate_results(
        results,
        reverse_sequences={"REV"},
        expected_video_id="P1130113",
        expected_video_sha256=VIDEO_SHA256,
        expected_total_frames=1488,
        expected_profile_sha256=PROFILE_SHA256,
        **lineage_kwargs(),
    )

    assert result["checks"]["G9.1"] is False


def test_different_video_cannot_masquerade_as_second_independent_flight() -> None:
    results = passing_evidence()
    results[2]["evaluation"]["source_video_id"] = "P9999999"
    results[2]["evaluation"]["source_video_sha256"] = "e" * 64

    result = evaluate_results(
        results,
        reverse_sequences={"REV"},
        expected_video_id="P1130113",
        expected_video_sha256=VIDEO_SHA256,
        expected_total_frames=1488,
        expected_profile_sha256=PROFILE_SHA256,
        **lineage_kwargs(),
    )

    assert result["checks"]["G9.0"] is False
    assert result["validation_semantics"]["independent_video_count"] == 2


def test_stale_runtime_profile_hash_fails_semantic_gate() -> None:
    results = passing_evidence()
    results[1]["runtime"]["deployment_profile_sha256"] = "b" * 64

    result = evaluate_results(
        results,
        reverse_sequences={"REV"},
        expected_video_id="P1130113",
        expected_video_sha256=VIDEO_SHA256,
        expected_total_frames=1488,
        expected_profile_sha256=PROFILE_SHA256,
        **lineage_kwargs(),
    )

    assert result["checks"]["G9.0"] is False


def test_stale_edm_or_tracking_result_hash_fails_semantic_gate() -> None:
    results = passing_evidence()
    results[0]["runtime"]["edm_bundle_sha256"] = "0" * 64
    results[1]["runtime"]["tracking_bundle_sha256"] = "1" * 64

    result = evaluate_results(
        results,
        reverse_sequences={"REV"},
        expected_video_id="P1130113",
        expected_video_sha256=VIDEO_SHA256,
        expected_total_frames=1488,
        expected_profile_sha256=PROFILE_SHA256,
        **lineage_kwargs(),
    )

    assert result["checks"]["G9.0"] is False


def test_swapped_baseline_or_live_camera_claim_fails_semantic_gate() -> None:
    results = passing_evidence()
    results[0]["runtime"]["anchor_density_baseline_sha256"] = "2" * 64
    results[1]["runtime"]["camera_contract"] = "live_camera_contract"

    result = evaluate_results(
        results,
        reverse_sequences={"REV"},
        expected_video_id="P1130113",
        expected_video_sha256=VIDEO_SHA256,
        expected_total_frames=1488,
        expected_profile_sha256=PROFILE_SHA256,
        **lineage_kwargs(),
    )

    assert result["checks"]["G9.0"] is False


def test_manifest_contract_requires_exactly_one_hashed_independent_video(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "corpus_manifest.json"
    manifest.write_text(
        """{
          "test": [{
            "seq": "P1130113",
            "sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "probed": {"nb_frames": 1488}
          }]
        }""",
        encoding="utf-8",
    )

    assert heldout_contract_from_manifest(manifest) == {
        "video_id": "P1130113",
        "video_sha256": VIDEO_SHA256,
        "total_frames": 1488,
    }
