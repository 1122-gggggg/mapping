from __future__ import annotations

from update_quality_gates import bridge_quality_warnings, matched_warnings, parse_warning_set


def test_parse_warning_set_handles_commas_semicolons_and_duplicates():
    assert parse_warning_set("a,b; a,,c") == ["a", "b", "c"]


def test_matched_warnings_only_returns_configured_quarantine_warnings():
    assert matched_warnings(
        "retrieval_high_but_inliers_low,inliers_spatially_concentrated",
        "retrieval_high_but_inliers_low",
    ) == ["retrieval_high_but_inliers_low"]


def test_bridge_quality_warnings_include_geometry_hard_gate_inputs():
    warnings = bridge_quality_warnings(
        bridge_geometry=1,
        total_bridges=5,
        median_inlier_ratio=0.2,
        median_support_area=0.03,
        min_inlier_ratio=0.25,
        min_support_area=0.05,
        min_geometry=2,
        min_geometry_ratio=0.5,
    )

    assert warnings == [
        "low_bridge_inlier_ratio",
        "low_bridge_support_area",
        "low_bridge_geometry_count",
        "low_bridge_geometry_ratio",
    ]
