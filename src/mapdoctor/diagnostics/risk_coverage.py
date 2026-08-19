from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from mapdoctor.benchmark import QueryLocalizationResult
from mapdoctor.config import LocalizationThresholds


@dataclass(frozen=True)
class RiskCoveragePoint:
    accepted: int
    coverage: float
    expected_failures: float
    selective_risk: float
    threshold: float
    tie_group_size: int
    tie_group_position: int
    randomized_within_tie: bool

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationBin:
    low: float
    high: float
    count: int
    mean_predicted_failure: float
    observed_failure_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class RiskCoverageReport:
    queries: int
    strict_failures: int
    aurc: float
    oracle_aurc: float
    excess_aurc: float
    failure_auroc: float | None
    brier_score: float
    expected_calibration_error: float
    curve: tuple[RiskCoveragePoint, ...]
    calibration_bins: tuple[CalibrationBin, ...]
    operating_points: dict[str, dict[str, float | int] | None]

    def to_dict(self) -> dict[str, object]:
        return {
            "queries": self.queries,
            "strict_failures": self.strict_failures,
            "aurc": self.aurc,
            "oracle_aurc": self.oracle_aurc,
            "excess_aurc": self.excess_aurc,
            "failure_auroc": self.failure_auroc,
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "curve": [point.to_dict() for point in self.curve],
            "calibration_bins": [
                calibration_bin.to_dict()
                for calibration_bin in self.calibration_bins
            ],
            "operating_points": self.operating_points,
        }


def _validate_risks(
    results: Sequence[QueryLocalizationResult],
    risks: Mapping[str, float],
) -> list[tuple[QueryLocalizationResult, float]]:
    result_name_list = [result.query for result in results]
    if len(result_name_list) != len(set(result_name_list)):
        raise ValueError("benchmark query names must be unique")
    non_string_keys = [key for key in risks if not isinstance(key, str)]
    if non_string_keys:
        raise ValueError("risk-score query IDs must be strings")

    result_names = set(result_name_list)
    missing = sorted(result_names - set(risks))
    extra = sorted(set(risks) - result_names)
    if missing:
        raise ValueError(
            "risk scores are missing required queries: " + ", ".join(missing)
        )
    if extra:
        raise ValueError(
            "risk scores contain queries outside the benchmark: " + ", ".join(extra)
        )

    output: list[tuple[QueryLocalizationResult, float]] = []
    for result in results:
        raw_risk = risks[result.query]
        if isinstance(raw_risk, bool):
            raise ValueError(f"{result.query}: risk must be numeric")
        try:
            risk = float(raw_risk)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{result.query}: risk must be numeric") from exc
        if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
            raise ValueError(f"{result.query}: risk must be finite and in [0, 1]")
        output.append((result, risk))
    return output


def _failure_auroc(risks: list[float], failures: list[int]) -> float | None:
    positives = sum(failures)
    negatives = len(failures) - positives
    if positives == 0 or negatives == 0:
        return None

    order = sorted(range(len(risks)), key=lambda index: risks[index])
    ranks = [0.0] * len(risks)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and risks[order[end]] == risks[order[position]]:
            end += 1
        average_rank = ((position + 1) + end) / 2.0
        for index in order[position:end]:
            ranks[index] = average_rank
        position = end

    positive_rank_sum = sum(
        rank for rank, failure in zip(ranks, failures) if failure
    )
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _calibration(
    risks: list[float],
    failures: list[int],
    bins: int,
) -> tuple[float, tuple[CalibrationBin, ...]]:
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("ece_bins must be an integer >= 1")

    rows: list[CalibrationBin] = []
    total = len(risks)
    ece = 0.0
    for bin_index in range(bins):
        low = bin_index / bins
        high = (bin_index + 1) / bins
        indices = [
            index
            for index, risk in enumerate(risks)
            if risk >= low and (risk <= high if bin_index == bins - 1 else risk < high)
        ]
        if not indices:
            continue
        predicted = sum(risks[index] for index in indices) / len(indices)
        observed = sum(failures[index] for index in indices) / len(indices)
        ece += len(indices) / total * abs(predicted - observed)
        rows.append(
            CalibrationBin(
                low=low,
                high=high,
                count=len(indices),
                mean_predicted_failure=predicted,
                observed_failure_rate=observed,
            )
        )
    return ece, tuple(rows)


def _oracle_aurc(failures: int, total: int) -> float:
    successes = total - failures
    cumulative = 0.0
    for accepted in range(1, total + 1):
        accepted_failures = max(0, accepted - successes)
        cumulative += accepted_failures / accepted
    return cumulative / total


def evaluate_risk_coverage(
    results: Sequence[QueryLocalizationResult],
    risks: Mapping[str, float],
    thresholds: LocalizationThresholds,
    *,
    ece_bins: int = 10,
    target_failure_rates: Sequence[float] = (0.01, 0.02, 0.05),
) -> RiskCoverageReport:
    """Evaluate selective localization with a tie-invariant empirical AURC.

    Lower predicted risk is accepted first. Within an equal-score group, each
    prefix uses the expected number of failures under a uniformly random tie
    order. This removes arbitrary dependence on query IDs or input ordering.
    """

    if not results:
        raise ValueError("risk-coverage evaluation requires at least one query")
    pairs = _validate_risks(results, risks)
    failures_by_query = {
        result.query: int(bool(result.failures(thresholds))) for result, _ in pairs
    }

    groups: dict[float, list[QueryLocalizationResult]] = {}
    for result, risk in pairs:
        groups.setdefault(risk, []).append(result)

    accepted_before = 0
    failures_before = 0
    points: list[RiskCoveragePoint] = []
    for threshold in sorted(groups):
        group = groups[threshold]
        group_size = len(group)
        group_failures = sum(failures_by_query[result.query] for result in group)
        for position in range(1, group_size + 1):
            accepted = accepted_before + position
            expected_failures = failures_before + position * group_failures / group_size
            points.append(
                RiskCoveragePoint(
                    accepted=accepted,
                    coverage=accepted / len(results),
                    expected_failures=expected_failures,
                    selective_risk=expected_failures / accepted,
                    threshold=threshold,
                    tie_group_size=group_size,
                    tie_group_position=position,
                    randomized_within_tie=position < group_size,
                )
            )
        accepted_before += group_size
        failures_before += group_failures

    aurc = sum(point.selective_risk for point in points) / len(points)
    failure_labels = [failures_by_query[result.query] for result, _ in pairs]
    risk_values = [risk for _, risk in pairs]
    oracle = _oracle_aurc(sum(failure_labels), len(failure_labels))
    brier = sum(
        (risk - failure) ** 2
        for risk, failure in zip(risk_values, failure_labels)
    ) / len(risk_values)
    ece, calibration_bins = _calibration(risk_values, failure_labels, ece_bins)
    auroc = _failure_auroc(risk_values, failure_labels)

    operating_points: dict[str, dict[str, float | int] | None] = {}
    for raw_target in target_failure_rates:
        if isinstance(raw_target, bool):
            raise ValueError("target failure rates must be finite and in [0, 1]")
        try:
            target = float(raw_target)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "target failure rates must be finite and in [0, 1]"
            ) from exc
        if not math.isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError("target failure rates must be finite and in [0, 1]")
        eligible = [
            point
            for point in points
            if not point.randomized_within_tie and point.selective_risk <= target
        ]
        best = max(eligible, key=lambda point: point.coverage) if eligible else None
        operating_points[f"{target:.6g}"] = (
            {
                "accepted": best.accepted,
                "coverage": best.coverage,
                "selective_risk": best.selective_risk,
                "threshold": best.threshold,
            }
            if best is not None
            else None
        )

    return RiskCoverageReport(
        queries=len(results),
        strict_failures=sum(failure_labels),
        aurc=aurc,
        oracle_aurc=oracle,
        excess_aurc=max(0.0, aurc - oracle),
        failure_auroc=auroc,
        brier_score=brier,
        expected_calibration_error=ece,
        curve=tuple(points),
        calibration_bins=calibration_bins,
        operating_points=operating_points,
    )
