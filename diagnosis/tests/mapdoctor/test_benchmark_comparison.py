from pathlib import Path

import pytest

from mapdoctor.benchmark import QueryLocalizationResult, load_localization_results, summarize_benchmark
from mapdoctor.comparison import compare_results
from mapdoctor.config import ComparisonThresholds, LocalizationThresholds

FIXTURES = Path(__file__).parent / "fixtures"


def test_benchmark_strict_quality_gates():
    results = load_localization_results(FIXTURES / "localization_results.csv")
    summary = summarize_benchmark(results, LocalizationThresholds())
    assert summary.total_queries == 4
    assert summary.raw_success_rate == 0.75
    assert summary.strict_success_rate == 0.5
    assert {item["query"] for item in summary.failures} == {"q3.jpg", "q4.jpg"}
    assert summary.weak_regions


def test_comparison_detects_new_failure():
    base = load_localization_results(FIXTURES / "comparison_base.csv")
    candidate = load_localization_results(FIXTURES / "comparison_candidate.csv")
    result = compare_results(base, candidate, LocalizationThresholds(), ComparisonThresholds())
    assert result.status == "FAIL"
    assert result.newly_failed == ["q2.jpg"]
    assert result.newly_recovered == ["q4.jpg"]
    assert result.gate_failures


def test_result_rejects_invalid_probability():
    with pytest.raises(ValueError, match="inlier_ratio"):
        QueryLocalizationResult(
            query="bad.jpg",
            success=True,
            inliers=20,
            inlier_ratio=1.2,
            reproj_p90_px=1.0,
            hull_coverage=0.2,
            grid4_occupancy=8,
            positive_depth_ratio=1.0,
            pose_consensus=0.8,
        )


def test_result_rejects_invalid_grid_occupancy():
    with pytest.raises(ValueError, match="grid4_occupancy"):
        QueryLocalizationResult(
            query="bad.jpg",
            success=True,
            inliers=20,
            inlier_ratio=0.5,
            reproj_p90_px=1.0,
            hull_coverage=0.2,
            grid4_occupancy=17,
            positive_depth_ratio=1.0,
            pose_consensus=0.8,
        )


def test_benchmark_rejects_duplicate_query_names(tmp_path):
    path = tmp_path / "duplicate.csv"
    path.write_text(
        "query,success,inliers,inlier_ratio,reproj_p90_px,hull_coverage,grid4_occupancy,positive_depth_ratio,pose_consensus\n"
        "q.jpg,true,30,0.5,1.0,0.2,8,1.0,0.8\n"
        "q.jpg,true,35,0.5,1.0,0.2,8,1.0,0.8\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        load_localization_results(path)


def test_method_agnostic_results_require_only_query_and_success(tmp_path):
    path = tmp_path / "generic-localizer.json"
    path.write_text(
        '[{"query": "q1", "success": true, "localizer": "custom-method", '
        '"metrics": {"method_confidence": 0.8}}, '
        '{"query": "q2", "success": false, "localizer": "custom-method"}]',
        encoding="utf-8",
    )

    results = load_localization_results(path)
    summary = summarize_benchmark(results, LocalizationThresholds())

    assert summary.strict_success_rate == 0.5
    assert results[0].localizer == "custom-method"
    assert results[0].metrics == {"method_confidence": 0.8}
    assert results[0].failures(LocalizationThresholds()) == []
    assert results[1].failures(LocalizationThresholds()) == ["localization_failed"]


def test_config_can_require_selected_quality_evidence(tmp_path):
    path = tmp_path / "generic-localizer.csv"
    path.write_text("query,success\nq1,true\n", encoding="utf-8")
    result = load_localization_results(path)[0]

    thresholds = LocalizationThresholds(required_metrics=("inliers",))

    assert result.failures(thresholds) == ["missing_inliers"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("inliers", "20.7"),
        ("grid4_occupancy", "8.9"),
        ("inliers", "nan"),
        ("grid4_occupancy", "inf"),
    ],
)
def test_benchmark_rejects_fractional_or_nonfinite_integer_fields(tmp_path, field, value):
    row = {
        "query": "q.jpg",
        "success": "true",
        "inliers": "30",
        "inlier_ratio": "0.5",
        "reproj_p90_px": "1.0",
        "hull_coverage": "0.2",
        "grid4_occupancy": "8",
        "positive_depth_ratio": "1.0",
        "pose_consensus": "0.8",
    }
    row[field] = value
    columns = list(row)
    path = tmp_path / "malformed.csv"
    path.write_text(
        ",".join(columns) + "\n" + ",".join(row[column] for column in columns) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=field):
        load_localization_results(path)


def _q(name: str, success: bool) -> QueryLocalizationResult:
    if success:
        return QueryLocalizationResult(
            query=name,
            success=True,
            inliers=60,
            inlier_ratio=0.60,
            reproj_p90_px=1.2,
            hull_coverage=0.35,
            grid4_occupancy=10,
            positive_depth_ratio=1.0,
            pose_consensus=0.90,
        )
    return QueryLocalizationResult(
        query=name,
        success=False,
        inliers=0,
        inlier_ratio=0.0,
        reproj_p90_px=9.0,
        hull_coverage=0.0,
        grid4_occupancy=0,
        positive_depth_ratio=0.0,
        pose_consensus=0.0,
    )


def test_success_rate_gate_uses_exact_fraction():
    base = [_q(f"q{i:02d}.jpg", True) for i in range(50)]
    candidate_at_bound = [_q(f"q{i:02d}.jpg", i != 0) for i in range(50)]
    at_bound = compare_results(
        base,
        candidate_at_bound,
        LocalizationThresholds(),
        ComparisonThresholds(max_success_rate_drop=0.02, max_new_failure_rate=1.0),
    )
    assert "strict success-rate regression exceeds gate" not in at_bound.gate_failures

    candidate_over = [_q(f"q{i:02d}.jpg", i > 1) for i in range(50)]
    over = compare_results(
        base,
        candidate_over,
        LocalizationThresholds(),
        ComparisonThresholds(max_success_rate_drop=0.02, max_new_failure_rate=1.0),
    )
    assert "strict success-rate regression exceeds gate" in over.gate_failures


def test_new_failure_rate_gate_uses_exact_fraction():
    base = [_q(f"q{i:02d}.jpg", True) for i in range(50)]
    candidate_at_bound = [_q(f"q{i:02d}.jpg", i != 0) for i in range(50)]
    at_bound = compare_results(
        base,
        candidate_at_bound,
        LocalizationThresholds(),
        ComparisonThresholds(max_success_rate_drop=1.0, max_new_failure_rate=0.02),
    )
    assert "new-failure rate exceeds gate" not in at_bound.gate_failures

    candidate_over = [_q(f"q{i:02d}.jpg", i > 1) for i in range(50)]
    over = compare_results(
        base,
        candidate_over,
        LocalizationThresholds(),
        ComparisonThresholds(max_success_rate_drop=1.0, max_new_failure_rate=0.02),
    )
    assert "new-failure rate exceeds gate" in over.gate_failures


def _located(name: str, success: bool, x: float, y: float = 0.0, z: float = 0.0) -> QueryLocalizationResult:
    result = _q(name, success)
    return QueryLocalizationResult(
        query=result.query,
        success=result.success,
        inliers=result.inliers,
        inlier_ratio=result.inlier_ratio,
        reproj_p90_px=result.reproj_p90_px,
        hull_coverage=result.hull_coverage,
        grid4_occupancy=result.grid4_occupancy,
        positive_depth_ratio=result.positive_depth_ratio,
        pose_consensus=result.pose_consensus,
        x=x,
        y=y,
        z=z,
    )


def test_sparse_all_pass_is_not_evidentially_complete(tmp_path):
    path = tmp_path / "sparse.json"
    path.write_text(
        '[{"query": "q1", "success": true}, {"query": "q2", "success": true}]',
        encoding="utf-8",
    )
    results = load_localization_results(path)
    summary = summarize_benchmark(results, LocalizationThresholds())

    assert summary.strict_success_rate == 1.0
    assert summary.failures == []
    assert summary.interpretation == "DESCRIPTIVE_ONLY"
    assert summary.independent_units_verified is False
    assert set(summary.metric_evidence) == {
        "inliers",
        "inlier_ratio",
        "reproj_p90_px",
        "hull_coverage",
        "grid4_occupancy",
        "positive_depth_ratio",
        "pose_consensus",
    }
    for evidence in summary.metric_evidence.values():
        assert evidence["present"] == 0
        assert evidence["failed"] == 0
        assert evidence["fail_rate"] is None
    assert summary.failure_reason_counts == {}
    assert summary.leave_one_criterion_strict_success_rates == {
        name: 1.0 for name in summary.metric_evidence
    }


def test_complete_conjunction_reports_metric_evidence():
    results = load_localization_results(FIXTURES / "localization_results.csv")
    thresholds = LocalizationThresholds()
    summary = summarize_benchmark(results, thresholds)

    assert summary.strict_success_rate == 0.5
    assert {item["query"] for item in summary.failures} == {"q3.jpg", "q4.jpg"}
    assert [result.passes(thresholds) for result in results] == [True, True, False, False]
    assert summary.failure_reason_counts == {
        "high_reprojection_error": 1,
        "localization_failed": 1,
        "low_grid_occupancy": 1,
        "low_inlier_hull_coverage": 1,
        "low_inlier_ratio": 2,
        "low_inliers": 2,
        "low_pose_consensus": 1,
        "low_positive_depth_ratio": 1,
    }
    assert summary.metric_evidence["inliers"] == {"present": 4, "failed": 2, "fail_rate": 0.5}
    assert summary.metric_evidence["inlier_ratio"] == {
        "present": 4,
        "failed": 2,
        "fail_rate": 0.5,
    }
    assert summary.metric_evidence["reproj_p90_px"] == {
        "present": 4,
        "failed": 1,
        "fail_rate": 0.25,
    }
    assert summary.metric_evidence["hull_coverage"]["present"] == 4
    assert summary.metric_evidence["grid4_occupancy"]["failed"] == 1
    assert summary.leave_one_criterion_strict_success_rates["inliers"] == 0.5
    assert summary.leave_one_criterion_strict_success_rates["inlier_ratio"] == 0.5
    assert summary.interpretation == "DESCRIPTIVE_ONLY"
    assert summary.independent_units_verified is False


def test_missing_required_metric_ablation_does_not_change_labels():
    result = QueryLocalizationResult(query="q1", success=True)
    thresholds = LocalizationThresholds(required_metrics=("inliers",))

    assert result.failures(thresholds) == ["missing_inliers"]
    assert result.passes(thresholds) is False

    summary = summarize_benchmark([result], thresholds)

    assert summary.strict_success_rate == 0.0
    assert summary.failures[0]["query"] == "q1"
    assert summary.failures[0]["reasons"] == ["missing_inliers"]
    assert summary.failure_reason_counts == {"missing_inliers": 1}
    assert summary.metric_evidence["inliers"] == {
        "present": 0,
        "failed": 0,
        "fail_rate": None,
    }
    assert summary.leave_one_criterion_strict_success_rates["inliers"] == 1.0
    assert summary.leave_one_criterion_strict_success_rates["inlier_ratio"] == 0.0
    assert result.failures(thresholds) == ["missing_inliers"]
    assert result.passes(thresholds) is False


def test_one_query_cell_is_insufficient_evidence():
    results = [
        _located("lone.jpg", False, x=0.0),
        _located("pair_ok.jpg", True, x=10.0),
        _located("pair_fail.jpg", False, x=11.0),
    ]
    summary = summarize_benchmark(results, LocalizationThresholds())

    assert summary.strict_success_rate == 1 / 3
    by_cell = {tuple(row["cell"]): row for row in summary.weak_regions}
    lone = by_cell[(0, 0, 0)]
    assert lone["queries"] == 1
    assert lone["failed_queries"] == ["lone.jpg"]
    assert lone["strict_success_rate"] == 0.0
    assert lone["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert lone["authority"] == "DESCRIPTIVE_ONLY"
    assert lone["shortfall_amount"] == 1.0

    paired = by_cell[(2, 0, 0)]
    assert paired["queries"] == 2
    assert paired["failed_queries"] == ["pair_fail.jpg"]
    assert paired["strict_success_rate"] == 0.5
    assert paired["evidence_status"] == "QUALITY_SHORTFALL"
    assert paired["authority"] == "DESCRIPTIVE_ONLY"
    assert paired["shortfall_amount"] == 0.5
