from __future__ import annotations

import pytest

from sfm_qa.relative_quality import percentile_ranks, weighted_observed_score


def test_percentile_ranks_are_tie_aware_and_keep_missing_rows() -> None:
    ranks = percentile_ranks({"low": 1.0, "tie_a": 2.0, "tie_b": 2.0, "missing": None})

    assert ranks["low"] == pytest.approx(0.0)
    assert ranks["tie_a"] == pytest.approx(0.75)
    assert ranks["tie_b"] == pytest.approx(0.75)
    assert ranks["missing"] is None


def test_singleton_is_kept_as_best_available() -> None:
    assert percentile_ranks({"only": 3.0}) == {"only": 1.0}


def test_numerically_equivalent_floats_share_a_tie_rank() -> None:
    ranks = percentile_ranks({"a": 0.5098039215686274, "b": 0.5098039215686275})

    assert ranks == {"a": 0.5, "b": 0.5}


def test_weighted_score_renormalizes_observed_terms_and_reports_completeness() -> None:
    score, completeness = weighted_observed_score(
        {"observed": 0.8, "missing": None},
        {"observed": 1.0, "missing": 3.0},
    )

    assert score == pytest.approx(0.8)
    assert completeness == pytest.approx(0.25)
