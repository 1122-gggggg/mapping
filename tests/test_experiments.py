from update_map.config import ValidationConfig
from update_map.experiments import aggregate_query_results, compare_to_baseline
from update_map.models import ExperimentResult, PoseQuality, QueryResult


def test_regression_detects_new_false_rejection() -> None:
    quality = PoseQuality(num_inliers=100, passed=True)
    baseline_results = [QueryResult("q", True, None, quality)]
    candidate_results = [QueryResult("q", False, None, PoseQuality())]
    baseline = ExperimentResult("E0", baseline_results, aggregate_query_results(baseline_results))
    candidate = ExperimentResult("E5", candidate_results, aggregate_query_results(candidate_results))
    report = compare_to_baseline(baseline, candidate, ValidationConfig())
    assert not report.passed
    assert report.new_false_rejections == ["q"]
