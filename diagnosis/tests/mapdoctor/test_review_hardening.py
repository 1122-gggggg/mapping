from __future__ import annotations

import json

import pytest

from mapdoctor.benchmark import QueryLocalizationResult
from mapdoctor.comparison import compare_results
from mapdoctor.config import ComparisonThresholds, LocalizationThresholds
from mapdoctor.diagnostics.io import (
    load_query_manifest,
    load_region_assignments,
    load_risk_scores,
)
from mapdoctor.diagnostics.regions import RegionDiagnosisConfig, diagnose_regions
from mapdoctor.diagnostics.risk_coverage import evaluate_risk_coverage
from mapdoctor.diagnostics.statistics import empirical_bayes_prior


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
        x=0.0,
        y=0.0,
        z=0.0,
    )


def test_empirical_bayes_prior_has_declared_strength_and_center():
    alpha, beta = empirical_bayes_prior(2, 10, strength=8.0, pseudocount=0.5)
    expected_rate = 2.5 / 11.0
    assert alpha + beta == pytest.approx(8.0)
    assert alpha / (alpha + beta) == pytest.approx(expected_rate)


@pytest.mark.parametrize("rate", [-0.01, 1.01, float("nan"), True, "bad"])
def test_localization_aggregate_target_is_a_probability(rate):
    with pytest.raises(ValueError, match="min_strict_success_rate"):
        LocalizationThresholds(min_strict_success_rate=rate)


@pytest.mark.parametrize("manifest", ["q1", [123], [None]])
def test_direct_comparison_manifest_rejects_non_sequence_or_non_string_values(manifest):
    with pytest.raises(ValueError, match="sequence of strings|values must be strings"):
        compare_results(
            [good("q1")],
            [good("q1")],
            LocalizationThresholds(),
            ComparisonThresholds(),
            required_queries=manifest,
        )


def test_region_config_and_direct_assignments_are_strict():
    with pytest.raises(ValueError, match="integer"):
        RegionDiagnosisConfig(min_samples=8.5)
    with pytest.raises(ValueError, match="integer"):
        RegionDiagnosisConfig(min_failures_for_weak=True)
    with pytest.raises(ValueError, match="string query"):
        diagnose_regions(
            [good("q1")],
            LocalizationThresholds(),
            assignments={1: "A"},
        )
    with pytest.raises(ValueError, match="duplicate normalized"):
        diagnose_regions(
            [good("q1")],
            LocalizationThresholds(),
            assignments={"q1": "A", " q1 ": "B"},
        )


def test_risk_api_rejects_duplicate_queries_boolean_scores_and_bad_bins():
    query = good("q1")
    with pytest.raises(ValueError, match="unique"):
        evaluate_risk_coverage(
            [query, query],
            {"q1": 0.1},
            LocalizationThresholds(),
        )
    with pytest.raises(ValueError, match="numeric"):
        evaluate_risk_coverage(
            [query],
            {"q1": True},
            LocalizationThresholds(),
        )
    with pytest.raises(ValueError, match="integer"):
        evaluate_risk_coverage(
            [query],
            {"q1": 0.1},
            LocalizationThresholds(),
            ece_bins=2.5,
        )


def test_query_manifest_object_rejects_unknown_properties(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"queries": ["q1"], "typo": ["q2"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only 'queries'"):
        load_query_manifest(path)


def test_region_assignment_rows_match_the_schema_exactly(tmp_path):
    path = tmp_path / "regions.json"
    path.write_text(
        json.dumps([{"query": "q1", "region": "A", "unexpected": 1}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only 'query' and 'region'"):
        load_region_assignments(path)


def test_risk_inputs_reject_boolean_and_unknown_columns(tmp_path):
    boolean_path = tmp_path / "boolean.json"
    boolean_path.write_text(json.dumps({"q1": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="numeric"):
        load_risk_scores(boolean_path)

    extra_path = tmp_path / "extra.csv"
    extra_path.write_text("query,risk,note\nq1,0.1,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        load_risk_scores(extra_path)
