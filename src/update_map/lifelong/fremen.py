from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import LifelongConfig
from .models import FeatureCandidate, FeatureObservation, HarmonicComponent, TemporalScoreModel

def _candidate_periods(
    observations: Sequence[FeatureObservation],
    config: LifelongConfig,
) -> list[float]:
    periods = {float(item) for item in config.candidate_periods_days if float(item) > 0.0}
    if config.frequency_grid_size > 0 and len(observations) >= 2:
        minimum = max(float(config.min_period_days), 1e-9)
        maximum = max(float(config.max_period_days), minimum)
        periods.update(
            float(item)
            for item in np.geomspace(minimum, maximum, num=config.frequency_grid_size)
        )
    if len(observations) >= 2:
        times = np.asarray([item.timestamp_days for item in observations], dtype=np.float64)
        span = float(np.max(times) - np.min(times))
        if span > 0.0 and config.min_observed_cycles > 0.0:
            periods = {
                period
                for period in periods
                if span / period + 1e-12 >= config.min_observed_cycles
            }
    return sorted(periods)


def _ridge_fit(design: np.ndarray, values: np.ndarray, ridge: float) -> np.ndarray:
    regularizer = np.eye(design.shape[1], dtype=np.float64) * ridge
    regularizer[0, 0] = 0.0
    lhs = design.T @ design + regularizer
    rhs = design.T @ values
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(design, values, rcond=None)[0]


def fit_fremen_model(
    observations: Sequence[FeatureObservation],
    config: LifelongConfig,
) -> TemporalScoreModel:
    """Fit prominent harmonic components by irregular-time spectral regression."""

    if not observations:
        return TemporalScoreModel(
            time_origin_days=0.0,
            baseline=0.0,
            empirical_mean=0.0,
            sample_count=0,
            prediction_min=config.prediction_min,
            prediction_max=config.prediction_max,
        )
    ordered = sorted(observations, key=lambda item: item.timestamp_days)
    times = np.asarray([item.timestamp_days for item in ordered], dtype=np.float64)
    values = np.asarray([item.value for item in ordered], dtype=np.float64)
    origin = float(times[0])
    relative = times - origin
    empirical_mean = float(np.mean(values))
    fallback = TemporalScoreModel(
        time_origin_days=origin,
        baseline=empirical_mean,
        empirical_mean=empirical_mean,
        sample_count=len(values),
        prediction_min=config.prediction_min,
        prediction_max=config.prediction_max,
    )
    if len(values) < config.min_temporal_samples or config.max_harmonics <= 0:
        return fallback

    ranked: list[tuple[float, float]] = []
    for period in _candidate_periods(ordered, config):
        omega = 2.0 * np.pi / period
        design = np.column_stack(
            [np.ones_like(relative), np.cos(omega * relative), np.sin(omega * relative)]
        )
        coefficients = _ridge_fit(design, values, config.ridge)
        power = float(coefficients[1] ** 2 + coefficients[2] ** 2)
        if np.isfinite(power):
            ranked.append((power, period))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    selected_periods: list[float] = []
    for _power, period in ranked:
        if any(
            abs(np.log(period / existing)) < config.min_log_period_separation
            for existing in selected_periods
        ):
            continue
        selected_periods.append(period)
        if len(selected_periods) >= config.max_harmonics:
            break
    if not selected_periods:
        return fallback

    columns = [np.ones_like(relative)]
    for period in selected_periods:
        omega = 2.0 * np.pi / period
        columns.extend([np.cos(omega * relative), np.sin(omega * relative)])
    design = np.column_stack(columns)
    coefficients = _ridge_fit(design, values, config.ridge)
    components: list[HarmonicComponent] = []
    for index, period in enumerate(selected_periods):
        cosine = float(coefficients[1 + 2 * index])
        sine = float(coefficients[2 + 2 * index])
        amplitude = float(np.hypot(cosine, sine))
        components.append(
            HarmonicComponent(
                period_days=float(period),
                cosine_coefficient=cosine,
                sine_coefficient=sine,
                amplitude=amplitude,
                phase_rad=float(np.arctan2(-sine, cosine)),
                power=amplitude**2,
            )
        )
    components.sort(key=lambda item: (-item.power, item.period_days))
    return TemporalScoreModel(
        time_origin_days=origin,
        baseline=float(coefficients[0]),
        empirical_mean=empirical_mean,
        sample_count=len(values),
        components=components,
        prediction_min=config.prediction_min,
        prediction_max=config.prediction_max,
    )


def descriptor_distance(left: np.ndarray, right: np.ndarray, metric: str) -> float:
    first = np.asarray(left)
    second = np.asarray(right)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("descriptors must be one-dimensional and have equal shape")
    if metric == "l2":
        a = np.asarray(first, dtype=np.float64)
        b = np.asarray(second, dtype=np.float64)
        return float(np.linalg.norm(a - b))
    if metric == "cosine":
        a = np.asarray(first, dtype=np.float64)
        b = np.asarray(second, dtype=np.float64)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator <= 1e-12:
            return 0.0 if np.allclose(a, b) else 1.0
        similarity = float(np.dot(a, b) / denominator)
        return float(1.0 - np.clip(similarity, -1.0, 1.0))
    if metric == "hamming":
        if first.dtype == np.bool_ or second.dtype == np.bool_:
            return float(np.mean(first.astype(bool) != second.astype(bool)))
        a = np.unpackbits(np.asarray(first, dtype=np.uint8))
        b = np.unpackbits(np.asarray(second, dtype=np.uint8))
        return float(np.mean(a != b))
    raise ValueError(f"Unsupported descriptor metric: {metric}")


def descriptor_uniqueness(
    descriptor: np.ndarray | None,
    map_descriptors: Sequence[np.ndarray],
    metric: str,
) -> float:
    """Distance to the closest retained descriptor; larger means more unique."""

    if descriptor is None:
        return -1.0
    query = np.asarray(descriptor)
    compatible = [np.asarray(item) for item in map_descriptors if np.shape(item) == query.shape]
    if not compatible:
        return 1.0
    if metric == "l2":
        stacked = np.stack(compatible).astype(np.float64, copy=False)
        delta = stacked - np.asarray(query, dtype=np.float64)
        return float(np.min(np.linalg.norm(delta, axis=1)))
    if metric == "cosine":
        stacked = np.stack(compatible).astype(np.float64, copy=False)
        query_f = np.asarray(query, dtype=np.float64)
        query_norm = float(np.linalg.norm(query_f))
        ref_norms = np.linalg.norm(stacked, axis=1)
        denominators = query_norm * ref_norms
        dots = stacked @ query_f
        distances = np.empty(len(compatible), dtype=np.float64)
        tiny = denominators <= 1e-12
        if np.any(tiny):
            distances[tiny] = np.where(
                np.all(np.isclose(stacked[tiny], query_f), axis=1), 0.0, 1.0
            )
        ok = ~tiny
        if np.any(ok):
            similarity = np.clip(dots[ok] / denominators[ok], -1.0, 1.0)
            distances[ok] = 1.0 - similarity
        return float(np.min(distances))
    return float(min(descriptor_distance(query, item, metric) for item in compatible))


def rank_candidates_by_uniqueness(
    candidates: Sequence[FeatureCandidate],
    map_descriptors: Sequence[np.ndarray],
    metric: str,
) -> list[tuple[FeatureCandidate, float]]:
    """Greedy farthest-first ranking prevents mutually duplicate admissions."""

    remaining = {candidate.feature_id: candidate for candidate in candidates}
    references = [np.asarray(item) for item in map_descriptors]
    ranked: list[tuple[FeatureCandidate, float]] = []
    while remaining:
        scored = [
            (
                descriptor_uniqueness(candidate.descriptor, references, metric),
                candidate.feature_id,
                candidate,
            )
            for candidate in remaining.values()
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        uniqueness, feature_id, candidate = scored[0]
        ranked.append((candidate, float(uniqueness)))
        remaining.pop(feature_id)
        if candidate.descriptor is not None:
            references.append(candidate.descriptor)
    return ranked
