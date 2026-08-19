from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

from mapdoctor.benchmark import QueryLocalizationResult
from mapdoctor.config import LocalizationThresholds
from mapdoctor.diagnostics.calibration import (
    apply_failure_calibrator,
    calibration_metrics,
    cross_fit_failure_calibration,
    failure_calibrator_from_dict,
    fit_beta_failure_calibrator,
    fit_isotonic_failure_calibrator,
    spatial_block_groups,
)


def result(
    name: str,
    failure: bool,
    *,
    x: float = 0.0,
) -> QueryLocalizationResult:
    return QueryLocalizationResult(
        query=name,
        success=not failure,
        inliers=0 if failure else 100,
        inlier_ratio=0.0 if failure else 0.8,
        reproj_p90_px=None if failure else 1.0,
        hull_coverage=0.0 if failure else 0.5,
        grid4_occupancy=0 if failure else 12,
        positive_depth_ratio=0.0 if failure else 1.0,
        pose_consensus=0.0 if failure else 0.9,
        x=x,
        y=0.0,
        z=0.0,
    )


def test_isotonic_is_monotone_and_pools_violations():
    scores = [0.1] * 10 + [0.4] * 10 + [0.7] * 10 + [0.9] * 10
    failures = (
        [0] * 9
        + [1]
        + [0] * 4
        + [1] * 6
        + [0] * 7
        + [1] * 3
        + [0]
        + [1] * 9
    )
    calibrator = fit_isotonic_failure_calibrator(
        scores,
        failures,
        min_samples=20,
    )
    predicted = calibrator.predict(np.linspace(0.0, 1.0, 101))
    assert np.all(np.diff(predicted) >= -1e-12)
    assert len(calibrator.block_counts) < 4
    assert all(0.0 < p < 1.0 for p in calibrator.block_failure_probability)


def test_beta_calibrator_preserves_risk_order_and_fails_closed():
    scores = np.linspace(0.01, 0.99, 80)
    failures = (scores > 0.6).astype(float)
    calibrator = fit_beta_failure_calibrator(
        scores,
        failures,
        min_samples=20,
    )
    predicted = calibrator.predict(scores)
    assert np.all(np.diff(predicted) >= -1e-12)
    assert predicted[0] < predicted[-1]
    assert isinstance(calibrator.predict(0.5), float)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calibrator.predict(1.1)


def test_serialized_calibrator_round_trip_and_apply():
    scores = np.linspace(0.05, 0.95, 40)
    failures = (scores > 0.55).astype(float)
    fitted = fit_beta_failure_calibrator(scores, failures, min_samples=20)
    restored = failure_calibrator_from_dict(fitted.to_dict())
    expected = fitted.predict([0.1, 0.5, 0.9])
    actual = restored.predict([0.1, 0.5, 0.9])
    assert actual == pytest.approx(expected)
    calibrated = apply_failure_calibrator(
        restored,
        {"future-a": 0.1, "future-b": 0.9},
    )
    assert calibrated["future-a"] < calibrated["future-b"]


def test_serialized_calibrator_rejects_malformed_parameters():
    with pytest.raises(ValueError, match="monotone"):
        failure_calibrator_from_dict(
            {
                "method": "isotonic",
                "block_lower_score": [0.0, 0.5],
                "block_upper_score": [0.4, 1.0],
                "block_failure_probability": [0.8, 0.2],
                "block_counts": [10, 10],
                "block_failures": [8, 2],
                "decision_thresholds": [0.45],
                "output_epsilon": 0.01,
            }
        )


def test_group_cross_fitting_never_splits_a_group():
    rows = []
    risks = {}
    groups = {}
    for session in range(6):
        for frame in range(8):
            name = f"s{session}-f{frame}"
            failure = session >= 3
            rows.append(result(name, failure))
            risks[name] = 0.8 if failure else 0.2
            groups[name] = f"session-{session}"
    report = cross_fit_failure_calibration(
        rows,
        risks,
        LocalizationThresholds(),
        groups=groups,
        folds=3,
        min_samples=20,
    )
    group_folds = {}
    for query, fold in report.fold_by_query.items():
        group = groups[query]
        group_folds.setdefault(group, set()).add(fold)
    assert all(len(folds) == 1 for folds in group_folds.values())
    assert report.grouping_mode == "explicit_group"
    assert report.calibrated_oof_brier <= report.raw_brier + 1e-8


def test_query_level_calibration_warns_about_video_leakage():
    rows = [result(f"q{i}", i >= 20) for i in range(40)]
    risks = {
        row.query: (0.1 if i < 20 else 0.9)
        for i, row in enumerate(rows)
    }
    report = cross_fit_failure_calibration(
        rows,
        risks,
        LocalizationThresholds(),
        folds=4,
        min_samples=20,
    )
    assert any(
        "adjacent video frames" in warning for warning in report.warnings
    )


def test_spatial_blocks_are_deterministic_and_require_positions():
    rows = [result("a", False, x=0.1), result("b", True, x=5.1)]
    groups = spatial_block_groups(rows, 5.0)
    assert groups == {
        "a": "spatial:0:0:0",
        "b": "spatial:1:0:0",
    }
    missing = result("missing", False)
    object.__setattr__(missing, "x", None)
    with pytest.raises(ValueError, match="required"):
        spatial_block_groups([missing], 5.0)


def test_cross_fit_rejects_too_few_groups():
    rows = [result(f"q{i}", i % 2 == 0) for i in range(30)]
    risks = {row.query: 0.5 for row in rows}
    groups = {row.query: "one-session" for row in rows}
    with pytest.raises(ValueError, match="exceeds the number"):
        cross_fit_failure_calibration(
            rows,
            risks,
            LocalizationThresholds(),
            groups=groups,
            folds=2,
            min_samples=10,
        )


def test_adaptive_calibration_metric_is_tie_order_invariant():
    probabilities = [0.2] * 7 + [0.5] * 8 + [0.9] * 5
    failures = [0] * 10 + [1] * 10
    forward = calibration_metrics(probabilities, failures, bins=4)
    reverse = calibration_metrics(
        list(reversed(probabilities)),
        list(reversed(failures)),
        bins=4,
    )
    assert forward == pytest.approx(reverse)


def test_calibrate_risk_cli_writes_oof_scores(tmp_path):
    from mapdoctor.cli import main

    rows = []
    risks = {}
    groups = {}
    for session in range(6):
        for frame in range(6):
            row = result(f"s{session}-f{frame}", session >= 3)
            rows.append(asdict(row))
            risks[row.query] = 0.2 if session < 3 else 0.8
            groups[row.query] = f"session-{session}"

    results_path = tmp_path / "results.json"
    risks_path = tmp_path / "risks.json"
    groups_path = tmp_path / "groups.json"
    report_path = tmp_path / "calibration.json"
    scores_path = tmp_path / "oof.json"
    results_path.write_text(json.dumps(rows), encoding="utf-8")
    risks_path.write_text(json.dumps(risks), encoding="utf-8")
    groups_path.write_text(json.dumps(groups), encoding="utf-8")

    assert main(
        [
            "calibrate-risk",
            str(results_path),
            str(risks_path),
            "--groups",
            str(groups_path),
            "--folds",
            "3",
            "--min-samples",
            "10",
            "--method",
            "identity",
            "--output",
            str(report_path),
            "--scores-output",
            str(scores_path),
        ]
    ) == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    assert payload["selected_method"] == "identity"
    assert payload["grouping_mode"] == "explicit_group"
    assert set(scores) == set(risks)

    future_path = tmp_path / "future.json"
    applied_path = tmp_path / "future-calibrated.json"
    future_path.write_text(json.dumps({"future": 0.37}), encoding="utf-8")
    assert main(
        [
            "apply-risk-calibrator",
            str(report_path),
            str(future_path),
            "--output",
            str(applied_path),
        ]
    ) == 0
    assert json.loads(applied_path.read_text(encoding="utf-8")) == {
        "future": pytest.approx(0.37)
    }
