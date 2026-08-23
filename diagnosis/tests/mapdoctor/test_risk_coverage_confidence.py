from __future__ import annotations

import pytest

from mapdoctor.benchmark import QueryLocalizationResult
from mapdoctor.config import LocalizationThresholds
from mapdoctor.diagnostics.risk_coverage import evaluate_risk_coverage


def query(name: str, failure: bool) -> QueryLocalizationResult:
    return QueryLocalizationResult(
        query=name,
        success=not failure,
        inliers=0 if failure else 60,
        inlier_ratio=0.0 if failure else 0.6,
        reproj_p90_px=None if failure else 1.5,
        hull_coverage=0.0 if failure else 0.4,
        grid4_occupancy=0 if failure else 10,
        positive_depth_ratio=0.0 if failure else 1.0,
        pose_consensus=0.0 if failure else 0.9,
    )


def test_existing_tie_invariant_behavior_is_preserved():
    results = [
        query("s1", False),
        query("s2", False),
        query("f1", True),
        query("f2", True),
    ]
    report = evaluate_risk_coverage(
        results,
        {result.query: 0.5 for result in results},
        LocalizationThresholds(),
        target_failure_rates=(0.25,),
    )
    assert report.aurc == pytest.approx(0.5)
    assert report.operating_points["0.25"] is None
    assert report.curve[-1].randomized_within_tie is False
    assert report.curve[-1].observed_failures == 2
    assert report.curve[0].simultaneous_failure_upper_bound is None


def test_safe_operating_point_uses_upper_bound_not_empirical_zero():
    results = [query(f"s{i}", False) for i in range(120)]
    risks = {result.query: 0.1 for result in results}
    report = evaluate_risk_coverage(
        results,
        risks,
        LocalizationThresholds(),
        target_failure_rates=(0.05,),
        confidence_level=0.95,
    )
    point = report.safe_operating_points["0.05"]
    assert point is not None
    assert point["accepted"] == 120
    assert 0.0 < point["failure_rate_upper_bound"] < 0.05


def test_bonferroni_correction_gets_more_conservative_with_more_thresholds():
    results = [query(f"s{i}", False) for i in range(100)]
    one_group = evaluate_risk_coverage(
        results,
        {row.query: 0.5 for row in results},
        LocalizationThresholds(),
    )
    many_groups = evaluate_risk_coverage(
        results,
        {row.query: i / 100 for i, row in enumerate(results)},
        LocalizationThresholds(),
    )
    assert (
        many_groups.curve[-1].simultaneous_failure_upper_bound
        > one_group.curve[-1].simultaneous_failure_upper_bound
    )


def test_equal_mass_calibration_does_not_split_ties():
    results = [query(f"q{i}", i >= 10) for i in range(20)]
    risks = {
        row.query: (0.2 if i < 7 else 0.5 if i < 15 else 0.9)
        for i, row in enumerate(results)
    }
    report = evaluate_risk_coverage(
        results,
        risks,
        LocalizationThresholds(),
        ece_bins=4,
        ece_binning="equal_mass",
    )
    assert report.calibration_binning == "equal_mass"
    assert sorted(bin.count for bin in report.calibration_bins) == [5, 7, 8]


def test_invalid_confidence_and_binning_fail_closed():
    results = [query("q", False)]
    with pytest.raises(ValueError, match="confidence_level"):
        evaluate_risk_coverage(
            results,
            {"q": 0.1},
            LocalizationThresholds(),
            confidence_level=1.0,
        )
    with pytest.raises(ValueError, match="ece_binning"):
        evaluate_risk_coverage(
            results,
            {"q": 0.1},
            LocalizationThresholds(),
            ece_binning="bad",
        )
