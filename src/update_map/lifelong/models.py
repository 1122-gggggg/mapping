from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..config import LifelongConfig


def _json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


class FeatureEvent(str, Enum):
    """Outcome of one eligible feature after a trusted localization."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNMATCHED = "unmatched"


class FeatureState(str, Enum):
    """Sidecar state; the frozen base-map geometry is never deleted."""

    ACTIVE = "active"
    RETIRED = "retired"
    QUARANTINED = "quarantined"


class MapManagementStrategy(str, Enum):
    """Single-map policies evaluated in the paper."""

    STATIC = "static"
    LATEST = "latest"
    AGGRESSIVE = "aggressive"
    STRICT = "strict"
    SUMMARY = "summary"
    SCORE = "score"
    FREMEN = "fremen"


@dataclass(frozen=True)
class FeatureObservation:
    timestamp_days: float
    event: FeatureEvent
    value: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp_days):
            raise ValueError("timestamp_days must be finite")
        if not np.isfinite(self.value):
            raise ValueError("observation value must be finite")


@dataclass(frozen=True)
class HarmonicComponent:
    period_days: float
    cosine_coefficient: float
    sine_coefficient: float
    amplitude: float
    phase_rad: float
    power: float


@dataclass
class TemporalScoreModel:
    """FreMEn-style harmonic approximation of time-dependent utility."""

    time_origin_days: float
    baseline: float
    empirical_mean: float
    sample_count: int
    components: list[HarmonicComponent] = field(default_factory=list)
    prediction_min: float = -1.0
    prediction_max: float = 1.0

    def predict(self, timestamp_days: float) -> float:
        if not np.isfinite(timestamp_days):
            raise ValueError("timestamp_days must be finite")
        relative = float(timestamp_days - self.time_origin_days)
        value = self.baseline
        for component in self.components:
            omega = 2.0 * np.pi / component.period_days
            value += component.cosine_coefficient * np.cos(omega * relative)
            value += component.sine_coefficient * np.sin(omega * relative)
        return float(np.clip(value, self.prediction_min, self.prediction_max))


@dataclass
class FeatureMemoryRecord:
    """Persistent score and temporal evidence for one map feature association."""

    feature_id: str
    point3d_id: int | None = None
    descriptor: np.ndarray | None = None
    score: float = 0.0
    state: FeatureState = FeatureState.ACTIVE
    verified_current_geometry: bool = True
    correct_count: int = 0
    incorrect_count: int = 0
    unmatched_count: int = 0
    observations: list[FeatureObservation] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.feature_id = str(self.feature_id)
        if self.descriptor is not None:
            descriptor = np.asarray(self.descriptor)
            if descriptor.ndim != 1:
                raise ValueError("descriptor must be one-dimensional")
            if not np.all(np.isfinite(descriptor.astype(np.float64, copy=False))):
                raise ValueError("descriptor contains non-finite values")
            self.descriptor = descriptor.copy()
        if not np.isfinite(self.score):
            raise ValueError("score must be finite")
        if self.state == FeatureState.ACTIVE and (
            not self.verified_current_geometry or self.point3d_id is None
        ):
            self.state = FeatureState.QUARANTINED

    @property
    def mean_event_score(self) -> float:
        if not self.observations:
            return 0.0
        return float(np.mean([item.value for item in self.observations]))

    def observe(
        self,
        event: FeatureEvent,
        timestamp_days: float,
        config: LifelongConfig,
    ) -> None:
        value = event_value(event, config)
        self.score = float(np.clip(self.score + value, config.score_min, config.score_max))
        if event == FeatureEvent.CORRECT:
            self.correct_count += 1
        elif event == FeatureEvent.INCORRECT:
            self.incorrect_count += 1
        elif event == FeatureEvent.UNMATCHED:
            self.unmatched_count += 1
        else:  # pragma: no cover
            raise ValueError(f"Unsupported feature event: {event}")
        self.observations.append(FeatureObservation(float(timestamp_days), event, value))
        if config.history_limit > 0 and len(self.observations) > config.history_limit:
            self.observations = self.observations[-config.history_limit :]

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_id": self.feature_id,
            "point3d_id": self.point3d_id,
            "descriptor": None if self.descriptor is None else self.descriptor.tolist(),
            "score": self.score,
            "state": self.state.value,
            "verified_current_geometry": self.verified_current_geometry,
            "correct_count": self.correct_count,
            "incorrect_count": self.incorrect_count,
            "unmatched_count": self.unmatched_count,
            "observations": [
                {
                    "timestamp_days": item.timestamp_days,
                    "event": item.event.value,
                    "value": item.value,
                }
                for item in self.observations
            ],
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FeatureMemoryRecord":
        raw_observations = payload.get("observations", [])
        if not isinstance(raw_observations, Sequence):
            raise ValueError("observations must be a sequence")
        observations: list[FeatureObservation] = []
        for item in raw_observations:
            if not isinstance(item, Mapping):
                raise ValueError("each observation must be an object")
            observations.append(
                FeatureObservation(
                    timestamp_days=float(item["timestamp_days"]),
                    event=FeatureEvent(str(item["event"])),
                    value=float(item["value"]),
                )
            )
        descriptor_payload = payload.get("descriptor")
        descriptor = None if descriptor_payload is None else np.asarray(descriptor_payload)
        point3d_payload = payload.get("point3d_id")
        raw_metadata = payload.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("metadata must be an object")
        return cls(
            feature_id=str(payload["feature_id"]),
            point3d_id=None if point3d_payload is None else int(point3d_payload),
            descriptor=descriptor,
            score=float(payload.get("score", 0.0)),
            state=FeatureState(str(payload.get("state", FeatureState.ACTIVE.value))),
            verified_current_geometry=bool(payload.get("verified_current_geometry", True)),
            correct_count=int(payload.get("correct_count", 0)),
            incorrect_count=int(payload.get("incorrect_count", 0)),
            unmatched_count=int(payload.get("unmatched_count", 0)),
            observations=observations,
            metadata=dict(raw_metadata),
        )


@dataclass(frozen=True)
class FeatureCandidate:
    """Candidate admitted only when linked to trusted current geometry."""

    feature_id: str
    point3d_id: int | None
    descriptor: np.ndarray | None
    verified_current_geometry: bool
    initial_score: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_id", str(self.feature_id))
        if self.descriptor is not None:
            descriptor = np.asarray(self.descriptor)
            if descriptor.ndim != 1:
                raise ValueError("candidate descriptor must be one-dimensional")
            if not np.all(np.isfinite(descriptor.astype(np.float64, copy=False))):
                raise ValueError("candidate descriptor contains non-finite values")
            object.__setattr__(self, "descriptor", descriptor.copy())
        if self.initial_score is not None and not np.isfinite(self.initial_score):
            raise ValueError("initial_score must be finite")


@dataclass(frozen=True)
class FeatureSelection:
    feature_id: str
    predicted_score: float
    scalar_score: float
    temporal_sample_count: int


@dataclass
class MapUpdatePlan:
    strategy: MapManagementStrategy
    applied: bool
    gate_passed: bool
    gate_reason: str = ""
    observed: list[str] = field(default_factory=list)
    activated: list[str] = field(default_factory=list)
    retired: list[str] = field(default_factory=list)
    quarantined: list[str] = field(default_factory=list)
    ignored_events: list[str] = field(default_factory=list)
    ignored_candidates: list[str] = field(default_factory=list)
    active_before: list[str] = field(default_factory=list)
    active_after: list[str] = field(default_factory=list)
    exchange_target: int = 0
    exchange_applied: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.observed or self.activated or self.retired or self.quarantined)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["strategy"] = self.strategy.value
        payload["changed"] = self.changed
        return payload


def event_value(event: FeatureEvent, config: LifelongConfig) -> float:
    if event == FeatureEvent.CORRECT:
        return float(config.correct_reward)
    if event == FeatureEvent.INCORRECT:
        return float(-config.incorrect_penalty)
    if event == FeatureEvent.UNMATCHED:
        return float(-config.unmatched_penalty)
    raise ValueError(f"Unsupported feature event: {event}")


def classify_feature_events(
    eligible_feature_ids: Iterable[str | int],
    matched_feature_ids: Sequence[str | int],
    inlier_mask: Sequence[bool] | np.ndarray,
) -> dict[str, FeatureEvent]:
    """Convert PnP/RANSAC outcomes to correct, incorrect and unmatched events.

    Eligibility must already account for FOV, occlusion, image quality and changed-region masks.
    For duplicate matches, any inlier wins over outlier matches for the same feature.
    """

    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if len(mask) != len(matched_feature_ids):
        raise ValueError("inlier_mask and matched_feature_ids must have equal length")
    feature_ids = {str(item) for item in eligible_feature_ids}
    feature_ids.update(str(item) for item in matched_feature_ids)
    outcomes: dict[str, bool] = {}
    for feature_id, is_inlier in zip(matched_feature_ids, mask, strict=True):
        key = str(feature_id)
        outcomes[key] = outcomes.get(key, False) or bool(is_inlier)
    events: dict[str, FeatureEvent] = {}
    for feature_id in sorted(feature_ids):
        if feature_id not in outcomes:
            events[feature_id] = FeatureEvent.UNMATCHED
        elif outcomes[feature_id]:
            events[feature_id] = FeatureEvent.CORRECT
        else:
            events[feature_id] = FeatureEvent.INCORRECT
    return events

