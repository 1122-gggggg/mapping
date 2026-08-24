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


def test_small_n_unique_scores_are_insufficient_for_tight_target():
    results = [query(f"s{i}", False) for i in range(20)]
    risks = {row.query: i / 19 for i, row in enumerate(results)}
    report = evaluate_risk_coverage(
        results,
        risks,
        LocalizationThresholds(),
        target_failure_rates=(0.01,),
        confidence_level=0.95,
    )
    assert report.operating_points["0.01"] is not None
    assert report.operating_points["0.01"]["accepted"] == 20
    assert report.safe_operating_points["0.01"] is None
    diagnostic = report.target_diagnostics["0.01"]
    assert diagnostic["empirical_status"] == "OPERATING_POINT_AVAILABLE"
    assert diagnostic["confidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert diagnostic["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert diagnostic["hard_status"] == "VALID"
    assert diagnostic["authority"] == "reporting"
    assert diagnostic["independence_verified"] is False
    assert diagnostic["complete_thresholds"] == 20
    assert diagnostic["largest_tie"] == 1
    assert diagnostic["queries_as_independent_units"] == 20
    assert diagnostic["bound_shortfall"] > 0.0
    assert diagnostic["accept_all_baseline"]["bound_shortfall"] > 0.0
    assert diagnostic["zero_failure_min_independent_units"] > 20


def test_one_tie_has_no_resolvable_selectivity():
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
    assert report.operating_points["0.25"] is None
    assert report.safe_operating_points["0.25"] is None
    diagnostic = report.target_diagnostics["0.25"]
    assert diagnostic["empirical_status"] == "NO_RESOLVABLE_SELECTIVITY"
    assert diagnostic["confidence_status"] == "NO_EMPIRICAL_FEASIBLE_POINT"
    assert diagnostic["complete_thresholds"] == 1
    assert diagnostic["largest_tie"] == 4
    assert diagnostic["best_empirical_point"] is None
    assert diagnostic["accept_all_baseline"]["accepted"] == 4
    assert diagnostic["accept_all_baseline"]["bound_shortfall"] > 0.0
    assert diagnostic["best_confidence_candidate"]["accepted"] == 4
    assert diagnostic["independence_verified"] is False


def test_distinct_scores_with_no_feasible_empirical_point():
    results = [
        query("f1", True),
        query("f2", True),
        query("s1", False),
        query("s2", False),
    ]
    report = evaluate_risk_coverage(
        results,
        {"f1": 0.1, "f2": 0.2, "s1": 0.8, "s2": 0.9},
        LocalizationThresholds(),
        target_failure_rates=(0.25,),
    )
    assert report.operating_points["0.25"] is None
    assert report.safe_operating_points["0.25"] is None
    diagnostic = report.target_diagnostics["0.25"]
    assert diagnostic["empirical_status"] == "NO_EMPIRICAL_FEASIBLE_POINT"
    assert diagnostic["confidence_status"] == "NO_EMPIRICAL_FEASIBLE_POINT"
    assert diagnostic["evidence_status"] == "QUALITY_SHORTFALL"
    assert diagnostic["complete_thresholds"] == 4
    assert diagnostic["best_empirical_point"] is None
    assert diagnostic["bound_shortfall"] > 0.0
    assert diagnostic["accept_all_baseline"]["observed_selective_risk"] == pytest.approx(
        0.5
    )


def test_interior_selective_success_is_not_accept_all():
    results = [query(f"s{i}", False) for i in range(3)] + [
        query(f"f{i}", True) for i in range(3)
    ]
    risks = {
        "s0": 0.05,
        "s1": 0.10,
        "s2": 0.15,
        "f0": 0.80,
        "f1": 0.85,
        "f2": 0.90,
    }
    report = evaluate_risk_coverage(
        results,
        risks,
        LocalizationThresholds(),
        target_failure_rates=(0.1,),
    )
    point = report.operating_points["0.1"]
    assert point is not None
    assert point["accepted"] == 3
    assert point["selective_risk"] == pytest.approx(0.0)
    diagnostic = report.target_diagnostics["0.1"]
    assert diagnostic["empirical_status"] == "OPERATING_POINT_AVAILABLE"
    assert diagnostic["best_empirical_point"]["accepted"] == 3
    assert diagnostic["accept_all_baseline"]["accepted"] == 6
    assert diagnostic["accept_all_baseline"]["observed_selective_risk"] == pytest.approx(
        0.5
    )
    assert diagnostic["accept_all_baseline"]["bound_shortfall"] > diagnostic[
        "bound_shortfall"
    ]
    assert diagnostic["independence_verified"] is False


def test_target_zero_and_one_diagnostics():
    results = [query(f"s{i}", False) for i in range(4)] + [
        query(f"f{i}", True) for i in range(2)
    ]
    risks = {
        "s0": 0.1,
        "s1": 0.2,
        "s2": 0.3,
        "s3": 0.4,
        "f0": 0.8,
        "f1": 0.9,
    }
    report = evaluate_risk_coverage(
        results,
        risks,
        LocalizationThresholds(),
        target_failure_rates=(0.0, 1.0),
    )
    assert report.operating_points["0"] is not None
    assert report.operating_points["0"]["accepted"] == 4
    assert report.operating_points["0"]["selective_risk"] == pytest.approx(0.0)
    assert report.safe_operating_points["0"] is None
    zero = report.target_diagnostics["0"]
    assert zero["empirical_status"] == "OPERATING_POINT_AVAILABLE"
    assert zero["confidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert zero["zero_failure_min_independent_units"] is None
    assert zero["bound_shortfall"] > 0.0

    assert report.operating_points["1"] is not None
    assert report.operating_points["1"]["accepted"] == 6
    assert report.safe_operating_points["1"] is not None
    one = report.target_diagnostics["1"]
    assert one["empirical_status"] == "OPERATING_POINT_AVAILABLE"
    assert one["confidence_status"] == "BOUND_AVAILABLE_ASSUMPTIONS_UNVERIFIED"
    assert one["evidence_status"] == "WARN"
    assert one["zero_failure_min_independent_units"] == 1
    assert one["bound_shortfall"] == pytest.approx(0.0)
    assert one["accept_all_baseline"]["bound_shortfall"] == pytest.approx(0.0)
    assert one["independence_verified"] is False


def test_risk_coverage_cli_prints_status_and_unverified_warning(tmp_path, capsys):
    from dataclasses import asdict
    import json

    from mapdoctor.cli import main

    results = [
        query("s1", False),
        query("s2", False),
        query("f1", True),
        query("f2", True),
    ]
    results_path = tmp_path / "results.json"
    risks_path = tmp_path / "risks.json"
    output_path = tmp_path / "risk.json"
    results_path.write_text(
        json.dumps([asdict(row) for row in results]),
        encoding="utf-8",
    )
    risks_path.write_text(
        json.dumps({row.query: 0.5 for row in results}),
        encoding="utf-8",
    )
    assert main(
        [
            "risk-coverage",
            str(results_path),
            str(risks_path),
            "--target-failure-rate",
            "0.25",
            "--output",
            str(output_path),
        ]
    ) == 0
    printed = capsys.readouterr().out
    assert "empirical=NO_RESOLVABLE_SELECTIVITY" in printed
    assert "confidence=NO_EMPIRICAL_FEASIBLE_POINT" in printed
    assert "bound_shortfall=" in printed
    assert "accept-all baseline shortfall=" in printed
    assert "Independence of query units is unverified" in printed
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["operating_points"]["0.25"] is None
    assert payload["target_diagnostics"]["0.25"]["independence_verified"] is False
