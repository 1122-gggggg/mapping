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
