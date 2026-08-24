from __future__ import annotations

from update_quality_gates import (
    bridge_quality_checks,
    bridge_quality_warnings,
    matched_warnings,
    parse_warning_set,
)


def _bridge_kwargs(**overrides):
    values = dict(
        bridge_geometry=1,
        total_bridges=5,
        median_inlier_ratio=0.2,
        median_support_area=0.03,
        min_inlier_ratio=0.25,
        min_support_area=0.05,
        min_geometry=2,
        min_geometry_ratio=0.5,
    )
    values.update(overrides)
    return values


def test_parse_warning_set_handles_commas_semicolons_and_duplicates():
    assert parse_warning_set("a,b; a,,c") == ["a", "b", "c"]


def test_matched_warnings_only_returns_configured_quarantine_warnings():
    assert matched_warnings(
        "retrieval_high_but_inliers_low,inliers_spatially_concentrated",
        "retrieval_high_but_inliers_low",
    ) == ["retrieval_high_but_inliers_low"]


def test_bridge_quality_warnings_include_geometry_hard_gate_inputs():
    warnings = bridge_quality_warnings(**_bridge_kwargs())

    assert warnings == [
        "low_bridge_inlier_ratio",
        "low_bridge_support_area",
        "low_bridge_geometry_count",
        "low_bridge_geometry_ratio",
    ]


def test_bridge_quality_warnings_preserve_existing_order_from_checks():
    checks = bridge_quality_checks(**_bridge_kwargs())
    assert [check["reason"] for check in checks if check["reason"]] == [
        "low_bridge_inlier_ratio",
        "low_bridge_support_area",
        "low_bridge_geometry_count",
        "low_bridge_geometry_ratio",
    ]
    assert bridge_quality_warnings(**_bridge_kwargs()) == [
        check["reason"] for check in checks if check["reason"]
    ]


def test_bridge_quality_equality_does_not_warn():
    kwargs = _bridge_kwargs(
        bridge_geometry=2,
        total_bridges=4,
        median_inlier_ratio=0.25,
        median_support_area=0.05,
        min_geometry=2,
        min_geometry_ratio=0.5,
    )
    assert bridge_quality_warnings(**kwargs) == []
    for check in bridge_quality_checks(**kwargs):
        assert check["passed"] is True
        assert check["finite"] is True
        assert check["hard_status"] == "VALID"
        assert check["evidence_status"] == "PASS"
        assert check["signed_margin"] == 0.0


def test_disabled_geometry_gates_omit_valid_geometry_warnings():
    kwargs = _bridge_kwargs(min_geometry=0, min_geometry_ratio=0.0)
    warnings = bridge_quality_warnings(**kwargs)
    assert warnings == [
        "low_bridge_inlier_ratio",
        "low_bridge_support_area",
    ]
    checks = {check["name"]: check for check in bridge_quality_checks(**kwargs)}
    assert checks["bridge_geometry_count"]["enabled"] is False
    assert checks["bridge_geometry_count"]["passed"] is True
    assert checks["bridge_geometry_ratio"]["enabled"] is False
    assert checks["bridge_geometry_ratio"]["passed"] is True


def test_nonfinite_bridge_metric_is_hard_fail():
    nan_kwargs = _bridge_kwargs(
        median_inlier_ratio=float("nan"),
        median_support_area=float("inf"),
        bridge_geometry=4,
        total_bridges=4,
        min_geometry=2,
        min_geometry_ratio=0.0,
    )
    warnings = bridge_quality_warnings(**nan_kwargs)
    assert "invalid_bridge_inlier_ratio" in warnings
    assert "invalid_bridge_support_area" in warnings
    assert "low_bridge_inlier_ratio" not in warnings
    assert warnings != []

    checks = {check["name"]: check for check in bridge_quality_checks(**nan_kwargs)}
    assert checks["bridge_inlier_ratio"]["finite"] is False
    assert checks["bridge_inlier_ratio"]["passed"] is False
    assert checks["bridge_inlier_ratio"]["hard_status"] == "HARD_FAIL"
    assert checks["bridge_inlier_ratio"]["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert checks["bridge_support_area"]["hard_status"] == "HARD_FAIL"
    assert checks["bridge_support_area"]["reason"] == "invalid_bridge_support_area"


def test_negative_bridge_counts_are_invalid():
    kwargs = _bridge_kwargs(
        bridge_geometry=-1,
        total_bridges=-3,
        median_inlier_ratio=0.9,
        median_support_area=0.9,
    )
    warnings = bridge_quality_warnings(**kwargs)
    assert warnings == [
        "invalid_bridge_geometry_count",
        "invalid_bridge_geometry_ratio",
    ]
    checks = {check["name"]: check for check in bridge_quality_checks(**kwargs)}
    assert checks["bridge_geometry_count"]["hard_status"] == "HARD_FAIL"
    assert checks["bridge_geometry_ratio"]["hard_status"] == "HARD_FAIL"
    assert checks["bridge_geometry_count"]["passed"] is False
    assert checks["bridge_geometry_ratio"]["passed"] is False


def test_bool_counts_and_ratios_are_invalid():
    kwargs = _bridge_kwargs(
        bridge_geometry=True,
        total_bridges=False,
        median_inlier_ratio=True,
        median_support_area=False,
        min_geometry=True,
        min_geometry_ratio=False,
    )
    warnings = bridge_quality_warnings(**kwargs)
    assert warnings == [
        "invalid_bridge_inlier_ratio",
        "invalid_bridge_support_area",
        "invalid_bridge_geometry_count",
        "invalid_bridge_geometry_ratio",
    ]
    for check in bridge_quality_checks(**kwargs):
        assert check["finite"] is False
        assert check["passed"] is False
        assert check["hard_status"] == "HARD_FAIL"
        assert check["evidence_status"] == "INSUFFICIENT_EVIDENCE"


def test_disabled_gate_still_rejects_invalid_counts():
    kwargs = _bridge_kwargs(
        bridge_geometry=True,
        total_bridges=float("nan"),
        median_inlier_ratio=0.9,
        median_support_area=0.9,
        min_geometry=0,
        min_geometry_ratio=0.0,
    )
    warnings = bridge_quality_warnings(**kwargs)
    assert warnings == [
        "invalid_bridge_geometry_count",
        "invalid_bridge_geometry_ratio",
    ]


def test_bridge_receipt_reports_signed_margins():
    kwargs = _bridge_kwargs()
    checks = {check["name"]: check for check in bridge_quality_checks(**kwargs)}
    assert checks["bridge_inlier_ratio"]["direction"] == "gte"
    assert checks["bridge_inlier_ratio"]["signed_margin"] == 0.2 - 0.25
    assert checks["bridge_support_area"]["signed_margin"] == 0.03 - 0.05
    assert checks["bridge_geometry_count"]["signed_margin"] == 1 - 2
    assert checks["bridge_geometry_ratio"]["value"] == 1 / 5
    assert checks["bridge_geometry_ratio"]["signed_margin"] == (1 / 5) - 0.5
    assert checks["bridge_inlier_ratio"]["authority"] == "reporting/review"
    assert checks["bridge_inlier_ratio"]["independence_assumption"]
    assert checks["bridge_inlier_ratio"]["provenance_assumption"]


def test_zero_total_bridges_preserves_existing_ratio_denominator():
    kwargs = _bridge_kwargs(
        bridge_geometry=1,
        total_bridges=0,
        median_inlier_ratio=0.9,
        median_support_area=0.9,
        min_geometry=0,
        min_geometry_ratio=0.5,
    )
    checks = {check["name"]: check for check in bridge_quality_checks(**kwargs)}
    assert checks["bridge_geometry_ratio"]["value"] == 1.0
    assert checks["bridge_geometry_ratio"]["passed"] is True
    assert bridge_quality_warnings(**kwargs) == []
