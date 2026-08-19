from __future__ import annotations

import pytest

from mapdoctor.comparison import ComparisonResult
from mapdoctor.recapture.counterfactual import (
    ExistingDataRepairTrial,
    summarize_existing_data_counterfactual,
)
from mapdoctor.recapture.types import Availability


def _comparison(base: float, candidate: float, *, status: str = "PASS") -> ComparisonResult:
    return ComparisonResult(
        status=status,
        compared_queries=20,
        base_strict_success_rate=base,
        candidate_strict_success_rate=candidate,
        newly_failed=[],
        newly_recovered=[],
        common_success_inlier_relative_change_median=0.0,
        common_success_reprojection_change_median_px=0.0,
        gate_failures=[] if status == "PASS" else ["regression"],
        query_deltas=[],
    )


def test_counterfactual_measures_safe_deficit_closure() -> None:
    summary = summarize_existing_data_counterfactual(
        [
            ExistingDataRepairTrial(
                "matching",
                _comparison(0.40, 0.80),
                _comparison(0.98, 0.98),
            ),
            ExistingDataRepairTrial(
                "geometry",
                _comparison(0.40, 0.55),
                _comparison(0.98, 0.98),
            ),
        ],
        required_stages=["matching", "geometry"],
        healthy_success_rate=0.95,
    )
    assert summary.complete
    assert summary.best_stage == "matching"
    assert summary.repairability == pytest.approx((0.80 - 0.40) / (0.95 - 0.40))
    metrics = summary.metric_values()
    assert metrics["existing_data_repairability"].status == Availability.DERIVED
    assert metrics["existing_data_counterfactual_complete"].value is True


def test_counterfactual_incomplete_stage_coverage_is_explicit() -> None:
    summary = summarize_existing_data_counterfactual(
        [
            ExistingDataRepairTrial(
                "matching",
                _comparison(0.20, 0.25),
                _comparison(0.98, 0.98),
            )
        ],
        required_stages=["matching", "geometry"],
    )
    assert not summary.complete
    assert summary.metric_values()["existing_data_counterfactual_complete"].value is False


def test_counterfactual_rejects_repair_that_regresses_stable_holdout() -> None:
    summary = summarize_existing_data_counterfactual(
        [
            ExistingDataRepairTrial(
                "geometry",
                _comparison(0.20, 0.90),
                _comparison(0.98, 0.90, status="FAIL"),
            )
        ],
        required_stages=["geometry"],
    )
    assert summary.complete
    assert summary.repairability == 0.0


def test_counterfactual_requires_predeclared_stages() -> None:
    with pytest.raises(ValueError, match="required_stages"):
        summarize_existing_data_counterfactual([], required_stages=[])
