from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from .config import RouteCellConfig, SelectionConfig
from .geometry import pose_distance, yaw_pitch_from_rotation
from .models import ReferenceCandidate, RouteCell, UtilityBreakdown
from .states import ReferenceProvenance


@dataclass
class SelectionResult:
    selected: list[ReferenceCandidate]
    rejected_redundant: list[ReferenceCandidate]
    rejected_budget: list[ReferenceCandidate]
    rejected_low_utility: list[ReferenceCandidate]
    coverage_before: dict[str, int]
    coverage_after: dict[str, int]
    uncovered_cells: dict[str, int]
    total_cost: float
    metadata: dict[str, object] = field(default_factory=dict)


def make_route_cell(
    route_segment: str,
    position_scalar: float,
    pose,
    config: RouteCellConfig,
    direction: str = "unknown",
    condition: str = "default",
) -> RouteCell:
    center = pose.camera_center
    yaw, pitch = yaw_pitch_from_rotation(pose)
    return RouteCell(
        route_segment=route_segment,
        position_bin=int(np.floor(position_scalar / config.position_bin_size)),
        height_bin=int(np.floor(center[2] / config.height_bin_size)),
        yaw_bin=int(np.floor((yaw + 180.0) / config.yaw_bin_deg)),
        pitch_bin=int(np.floor((pitch + 90.0) / config.pitch_bin_deg)),
        direction=direction,
        condition=condition,
    )


def compute_utility_total(breakdown: UtilityBreakdown, config: SelectionConfig) -> float:
    weights = config.weights
    total = (
        weights.viewpoint_gain * breakdown.viewpoint_gain
        + weights.localizer_success_gain * breakdown.localizer_success_gain
        + weights.pose_information_gain * breakdown.pose_information_gain
        + weights.stable_ratio * breakdown.stable_ratio
        - weights.redundancy_penalty * breakdown.redundancy_penalty
        - weights.runtime_cost * breakdown.runtime_cost
        - weights.risk_penalty * breakdown.risk_penalty
    )
    breakdown.total = float(total)
    return float(total)


def jaccard_similarity(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def cosine_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        return None
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 1e-12:
        return None
    return float(np.dot(a, b) / denominator)


def candidates_are_redundant(
    left: ReferenceCandidate,
    right: ReferenceCandidate,
    config: SelectionConfig,
) -> bool:
    rotation, translation = pose_distance(left.reference.pose, right.reference.pose)
    pose_redundant = (
        translation <= config.pose_translation_redundancy
        and rotation <= config.pose_rotation_redundancy_deg
    )
    landmark_redundant = (
        jaccard_similarity(left.visible_point3d_ids, right.visible_point3d_ids)
        >= config.landmark_jaccard_redundancy
    )
    descriptor_similarity = cosine_similarity(left.descriptor, right.descriptor)
    descriptor_redundant = (
        descriptor_similarity is not None
        and descriptor_similarity >= config.descriptor_cosine_redundancy
    )
    cell_redundant = left.supports_cells == right.supports_cells
    return pose_redundant and landmark_redundant and (descriptor_redundant or cell_redundant)


def candidate_risk(candidate: ReferenceCandidate) -> float:
    reference = candidate.reference
    risk = 0.0
    if reference.provenance == ReferenceProvenance.BRIDGED:
        risk += 0.20
        risk += min(reference.bridge_depth * 0.05, 0.25)
        if len(reference.anchor_ids) < 2:
            risk += 0.40
        if reference.bridge_path_count < 2:
            risk += 0.25
    risk += max(0.0, 0.7 - reference.stable_ratio)
    if reference.registration_quality is not None and not reference.registration_quality.passed:
        risk += 0.50
    return float(min(risk, 1.5))


def update_candidate_risk(candidate: ReferenceCandidate, config: SelectionConfig) -> float:
    candidate.utility.risk_penalty = candidate_risk(candidate)
    return compute_utility_total(candidate.utility, config)


def prune_redundant_candidates(
    candidates: Sequence[ReferenceCandidate], config: SelectionConfig
) -> tuple[list[ReferenceCandidate], list[ReferenceCandidate]]:
    ordered = sorted(candidates, key=lambda item: item.utility.total, reverse=True)
    kept: list[ReferenceCandidate] = []
    redundant: list[ReferenceCandidate] = []
    for candidate in ordered:
        if any(candidates_are_redundant(candidate, existing, config) for existing in kept):
            redundant.append(candidate)
        else:
            kept.append(candidate)
    return kept, redundant


def greedy_select_references(
    candidates: Sequence[ReferenceCandidate],
    config: SelectionConfig,
    current_coverage: Mapping[str, int] | None = None,
    cell_weights: Mapping[str, float] | None = None,
) -> SelectionResult:
    coverage_before = dict(current_coverage or {})
    coverage = dict(coverage_before)
    weights = dict(cell_weights or {})
    for candidate in candidates:
        compute_utility_total(candidate.utility, config)
    non_redundant, redundant = prune_redundant_candidates(candidates, config)
    selected: list[ReferenceCandidate] = []
    low_utility: list[ReferenceCandidate] = []
    remaining = list(non_redundant)
    total_cost = 0.0

    def marginal(candidate: ReferenceCandidate) -> float:
        cover_gain = 0.0
        for cell in candidate.supports_cells:
            deficit = max(config.min_k_cover - coverage.get(cell, 0), 0)
            if deficit > 0:
                cover_gain += weights.get(cell, 1.0) * (1.0 + deficit)
        return (candidate.utility.total + cover_gain) / max(candidate.cost, 1e-9)

    while remaining and len(selected) < config.budget:
        remaining.sort(key=marginal, reverse=True)
        candidate = remaining.pop(0)
        score = marginal(candidate)
        has_deficit_gain = any(
            coverage.get(cell, 0) < config.min_k_cover for cell in candidate.supports_cells
        )
        if candidate.utility.total < config.min_total_utility and not has_deficit_gain:
            low_utility.append(candidate)
            continue
        if score <= 0 and not has_deficit_gain:
            low_utility.append(candidate)
            continue
        selected.append(candidate)
        total_cost += candidate.cost
        for cell in candidate.supports_cells:
            coverage[cell] = coverage.get(cell, 0) + 1
    budget_rejected = list(remaining)
    all_cells = set(coverage)
    for candidate in candidates:
        all_cells.update(candidate.supports_cells)
    uncovered = {
        cell: max(config.min_k_cover - coverage.get(cell, 0), 0)
        for cell in sorted(all_cells)
        if coverage.get(cell, 0) < config.min_k_cover
    }
    return SelectionResult(
        selected=selected,
        rejected_redundant=redundant,
        rejected_budget=budget_rejected,
        rejected_low_utility=low_utility,
        coverage_before=coverage_before,
        coverage_after=coverage,
        uncovered_cells=uncovered,
        total_cost=total_cost,
        metadata={
            "budget": config.budget,
            "min_k_cover": config.min_k_cover,
            "candidate_count": len(candidates),
        },
    )


def route_cell_weights_from_baseline(
    success_rates: Mapping[str, float],
    query_counts: Mapping[str, int] | None = None,
) -> dict[str, float]:
    counts = query_counts or {}
    output: dict[str, float] = {}
    for cell, success in success_rates.items():
        weakness = 1.0 - float(np.clip(success, 0.0, 1.0))
        evidence = np.log1p(max(counts.get(cell, 1), 1))
        output[cell] = float(1.0 + 4.0 * weakness + 0.25 * evidence)
    return output


def estimate_viewpoint_gain(candidate_cells: Iterable[str], current_coverage: Mapping[str, int]) -> float:
    cells = set(candidate_cells)
    if not cells:
        return 0.0
    return float(np.mean([1.0 / (1.0 + current_coverage.get(cell, 0)) for cell in cells]))
