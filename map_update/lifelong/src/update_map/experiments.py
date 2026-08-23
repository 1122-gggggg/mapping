from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from .config import ValidationConfig
from .models import ExperimentResult, QueryResult
from .states import ExperimentId


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: ExperimentId
    include_direct_historical: bool
    use_change_mask: bool
    include_bridged: bool
    utility_selection: bool
    current_first_fallback: bool
    source_aware_pnp: bool
    strict_bridge_gate: bool = True


EXPERIMENT_SPECS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(ExperimentId.E0_BASE_CURRENT_ONLY, False, False, False, False, False, False),
    ExperimentSpec(ExperimentId.E1_DIRECT_NO_CHANGE_MASK, True, False, False, False, False, False),
    ExperimentSpec(ExperimentId.E2_DIRECT_CHANGE_AWARE, True, True, False, False, False, True),
    ExperimentSpec(ExperimentId.E3_DIRECT_VERIFIED_BRIDGE, True, True, True, False, False, True),
    ExperimentSpec(ExperimentId.E4_SELECTED_AUGMENTED, True, True, True, True, False, True),
    ExperimentSpec(ExperimentId.E5_PRODUCTION_CANDIDATE, True, True, True, True, True, True),
)

ABLATIONS: dict[str, str] = {
    "A1": "no historical references",
    "A2": "direct historical references only",
    "A3": "direct + change mask",
    "A4": "direct + bridge without strict multi-anchor gate (offline only)",
    "A5": "direct + verified multi-anchor bridge",
    "A6": "all candidates versus utility-selected candidates",
    "A7": "pooled retrieval versus current-first fallback",
    "A8": "unweighted versus confidence/source-aware PnP",
    "A9": "no stable-mask filtering versus stable-mask filtering",
    "A10": "no FIM utility versus FIM + localizer front-end utility",
    "A11": "unmatched-decay versus conflict-only stability update",
}


class ExperimentEvaluator(Protocol):
    def evaluate(self, spec: ExperimentSpec) -> ExperimentResult: ...


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    clean = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.percentile(clean, percentile)) if clean else None


def max_consecutive_failures(results: Sequence[QueryResult]) -> int:
    maximum = current = 0
    for result in results:
        if result.success:
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def aggregate_query_results(results: Sequence[QueryResult]) -> dict[str, object]:
    query_count = len(results)
    success_count = sum(result.success for result in results)
    translations = [result.translation_error for result in results if result.translation_error is not None]
    rotations = [result.rotation_error_deg for result in results if result.rotation_error_deg is not None]
    total_latencies = [
        sum(result.latency_ms.values()) for result in results if result.latency_ms
    ]
    by_cell: dict[str, list[QueryResult]] = {}
    for result in results:
        if result.route_cell:
            by_cell.setdefault(result.route_cell, []).append(result)
    cell_success = {
        cell: sum(item.success for item in items) / len(items) for cell, items in by_cell.items()
    }
    success_values = sorted(cell_success.values())
    worst_decile = (
        float(np.mean(success_values[: max(1, int(np.ceil(0.1 * len(success_values))))]))
        if success_values
        else None
    )
    return {
        "query_count": query_count,
        "success_count": success_count,
        "success_rate": success_count / query_count if query_count else 0.0,
        "confident_wrong_pose_count": sum(item.confident_wrong_pose for item in results),
        "translation_median": _percentile(translations, 50),
        "translation_p90": _percentile(translations, 90),
        "translation_p95": _percentile(translations, 95),
        "rotation_median_deg": _percentile(rotations, 50),
        "rotation_p90_deg": _percentile(rotations, 90),
        "rotation_p95_deg": _percentile(rotations, 95),
        "latency_p50_ms": _percentile(total_latencies, 50),
        "latency_p95_ms": _percentile(total_latencies, 95),
        "latency_p99_ms": _percentile(total_latencies, 99),
        "max_consecutive_failures": max_consecutive_failures(results),
        "route_cell_success": cell_success,
        "failed_route_cell_count": sum(value == 0.0 for value in cell_success.values()),
        "worst_decile_route_cell_success": worst_decile,
    }


def finalize_experiment(result: ExperimentResult) -> ExperimentResult:
    result.aggregate = aggregate_query_results(result.query_results)
    return result


@dataclass
class RegressionReport:
    passed: bool
    new_false_rejections: list[str]
    new_confident_wrong_poses: list[str]
    common_success_inlier_drop_fraction: float
    baseline_success_rate: float
    candidate_success_rate: float
    baseline_max_failure_run: int
    candidate_max_failure_run: int
    p95_latency_increase_fraction: float | None
    failed_gates: list[str]
    details: dict[str, object]


def compare_to_baseline(
    baseline: ExperimentResult,
    candidate: ExperimentResult,
    config: ValidationConfig,
) -> RegressionReport:
    baseline_by_id = {item.query_id: item for item in baseline.query_results}
    candidate_by_id = {item.query_id: item for item in candidate.query_results}
    common_ids = sorted(set(baseline_by_id) & set(candidate_by_id))
    new_false_rejections = [
        query_id
        for query_id in common_ids
        if baseline_by_id[query_id].success and not candidate_by_id[query_id].success
    ]
    new_confident_wrong = [
        query_id
        for query_id in common_ids
        if not baseline_by_id[query_id].confident_wrong_pose
        and candidate_by_id[query_id].confident_wrong_pose
    ]
    baseline_inliers: list[float] = []
    candidate_inliers: list[float] = []
    for query_id in common_ids:
        base = baseline_by_id[query_id]
        cand = candidate_by_id[query_id]
        if base.success and cand.success:
            baseline_inliers.append(float(base.quality.num_inliers))
            candidate_inliers.append(float(cand.quality.num_inliers))
    base_sum = sum(baseline_inliers)
    inlier_drop = max(0.0, (base_sum - sum(candidate_inliers)) / base_sum) if base_sum > 0 else 0.0
    base_agg = baseline.aggregate or aggregate_query_results(baseline.query_results)
    cand_agg = candidate.aggregate or aggregate_query_results(candidate.query_results)
    base_latency = base_agg.get("latency_p95_ms")
    cand_latency = cand_agg.get("latency_p95_ms")
    latency_increase = None
    if isinstance(base_latency, (int, float)) and isinstance(cand_latency, (int, float)) and base_latency > 0:
        latency_increase = max(0.0, (cand_latency - base_latency) / base_latency)
    failed: list[str] = []
    if config.require_zero_new_false_rejections and new_false_rejections:
        failed.append("new_false_rejections")
    if config.require_zero_new_confident_wrong_poses and new_confident_wrong:
        failed.append("new_confident_wrong_poses")
    if inlier_drop > config.common_success_inlier_max_drop_fraction:
        failed.append("common_success_inlier_drop")
    if latency_increase is not None and latency_increase > config.max_p95_latency_increase_fraction:
        failed.append("p95_latency_budget")
    baseline_failure_run = int(base_agg.get("max_consecutive_failures", 0))
    candidate_failure_run = int(cand_agg.get("max_consecutive_failures", 0))
    required_run = baseline_failure_run - config.min_failure_run_reduction
    if candidate_failure_run > required_run:
        failed.append("max_failure_run_not_improved")
    baseline_worst = base_agg.get("worst_decile_route_cell_success")
    candidate_worst = cand_agg.get("worst_decile_route_cell_success")
    if isinstance(baseline_worst, (int, float)) and isinstance(candidate_worst, (int, float)):
        if candidate_worst - baseline_worst < config.min_weak_cell_success_gain:
            failed.append("weak_cell_success_not_improved")
    return RegressionReport(
        passed=not failed,
        new_false_rejections=new_false_rejections,
        new_confident_wrong_poses=new_confident_wrong,
        common_success_inlier_drop_fraction=inlier_drop,
        baseline_success_rate=float(base_agg.get("success_rate", 0.0)),
        candidate_success_rate=float(cand_agg.get("success_rate", 0.0)),
        baseline_max_failure_run=baseline_failure_run,
        candidate_max_failure_run=candidate_failure_run,
        p95_latency_increase_fraction=latency_increase,
        failed_gates=failed,
        details={
            "common_query_count": len(common_ids),
            "baseline_aggregate": base_agg,
            "candidate_aggregate": cand_agg,
        },
    )


def run_experiment_protocol(evaluator: ExperimentEvaluator) -> list[ExperimentResult]:
    return [finalize_experiment(evaluator.evaluate(spec)) for spec in EXPERIMENT_SPECS]


class CallableExperimentEvaluator:
    def __init__(self, function: Callable[[ExperimentSpec], ExperimentResult]):
        self.function = function

    def evaluate(self, spec: ExperimentSpec) -> ExperimentResult:
        return self.function(spec)
