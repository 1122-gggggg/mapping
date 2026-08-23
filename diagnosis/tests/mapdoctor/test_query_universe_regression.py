from mapdoctor.benchmark import QueryLocalizationResult
from mapdoctor.comparison import compare_results
from mapdoctor.config import ComparisonThresholds, LocalizationThresholds


def good(name: str) -> QueryLocalizationResult:
    return QueryLocalizationResult(
        query=name,
        success=True,
        inliers=50,
        inlier_ratio=0.5,
        reproj_p90_px=1.5,
        hull_coverage=0.4,
        grid4_occupancy=10,
        positive_depth_ratio=1.0,
        pose_consensus=0.9,
    )


def test_candidate_omission_is_materialized_as_a_regression():
    result = compare_results(
        [good("q1"), good("q2")],
        [good("q1")],
        LocalizationThresholds(),
        ComparisonThresholds(),
    )
    assert result.status == "FAIL"
    assert result.compared_queries == 2
    assert result.missing_from_candidate == ["q2"]
    assert result.newly_failed == ["q2"]
    assert result.query_deltas[1]["candidate_present"] is False


def test_manifest_is_the_authoritative_query_universe():
    result = compare_results(
        [good("q1"), good("extra-base")],
        [good("q1"), good("extra-candidate")],
        LocalizationThresholds(),
        ComparisonThresholds(),
        required_queries=["q1", "q2"],
    )
    assert result.query_universe_source == "manifest"
    assert result.missing_from_base == ["q2"]
    assert result.missing_from_candidate == ["q2"]
    assert result.extra_in_base == ["extra-base"]
    assert result.extra_in_candidate == ["extra-candidate"]
    assert result.status == "FAIL"


def test_direct_api_rejects_duplicate_query_names():
    duplicate = good("q1")
    try:
        compare_results(
            [duplicate, duplicate],
            [good("q1")],
            LocalizationThresholds(),
            ComparisonThresholds(),
        )
    except ValueError as exc:
        assert "duplicate query" in str(exc)
    else:
        raise AssertionError("duplicate direct-API results must fail closed")
