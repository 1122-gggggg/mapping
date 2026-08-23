from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import numpy as np

from .config import StabilityConfig
from .states import ReferenceState


class StabilityEvent(str, Enum):
    CURRENT_CONFIRMATION = "CURRENT_CONFIRMATION"
    STABLE_LOCALIZATION_SUPPORT = "STABLE_LOCALIZATION_SUPPORT"
    UNMATCHED = "UNMATCHED"
    GEOMETRIC_CONFLICT = "GEOMETRIC_CONFLICT"
    CHANGE_EVIDENCE = "CHANGE_EVIDENCE"
    CONFIDENT_FALSE_POSE = "CONFIDENT_FALSE_POSE"


@dataclass
class StabilityHistoryEntry:
    timestamp: str
    event: StabilityEvent
    geometry_currentness_before: float
    geometry_currentness_after: float
    historical_view_utility_before: float
    historical_view_utility_after: float
    note: str = ""


@dataclass
class StabilityRecord:
    reference_id: str
    geometry_currentness: float = 0.5
    historical_view_utility: float = 0.5
    state: ReferenceState = ReferenceState.HIST_CANDIDATE
    last_timestamp_days: float | None = None
    confirmations: int = 0
    conflicts: int = 0
    unmatched_count: int = 0
    history: list[StabilityHistoryEntry] = field(default_factory=list)

    def _clamp(self) -> None:
        self.geometry_currentness = float(np.clip(self.geometry_currentness, 0.0, 1.0))
        self.historical_view_utility = float(np.clip(self.historical_view_utility, 0.0, 1.0))

    def _transition(self, config: StabilityConfig) -> None:
        score = min(self.geometry_currentness, self.historical_view_utility)
        if score <= config.retire_threshold:
            self.state = ReferenceState.HIST_RETIRED
        elif score <= config.suspect_threshold:
            self.state = ReferenceState.HIST_SUSPECT
        elif score >= config.active_threshold and self.confirmations > 0:
            self.state = ReferenceState.HIST_ACTIVE
        elif self.geometry_currentness >= config.active_threshold:
            self.state = ReferenceState.HIST_STABLE
        else:
            self.state = ReferenceState.HIST_CANDIDATE

    def advance_time(self, timestamp_days: float, config: StabilityConfig) -> None:
        """Decay geometry-currentness evidence, never viewpoint utility solely because of age."""

        if self.last_timestamp_days is None:
            self.last_timestamp_days = timestamp_days
            return
        if timestamp_days <= self.last_timestamp_days:
            return
        delta = timestamp_days - self.last_timestamp_days
        self.geometry_currentness *= config.decay_per_day**delta
        self.last_timestamp_days = timestamp_days
        self._clamp()
        self._transition(config)

    def update(
        self,
        event: StabilityEvent,
        config: StabilityConfig,
        timestamp_days: float | None = None,
        note: str = "",
    ) -> None:
        if timestamp_days is not None:
            self.advance_time(timestamp_days, config)
        before_current = self.geometry_currentness
        before_utility = self.historical_view_utility
        if event == StabilityEvent.CURRENT_CONFIRMATION:
            self.geometry_currentness += config.confirmation_gain
            self.historical_view_utility += 0.5 * config.confirmation_gain
            self.confirmations += 1
        elif event == StabilityEvent.STABLE_LOCALIZATION_SUPPORT:
            self.historical_view_utility += config.confirmation_gain
            self.geometry_currentness += 0.25 * config.confirmation_gain
            self.confirmations += 1
        elif event == StabilityEvent.UNMATCHED:
            # Default is exactly zero: missing a match is not proof of a changed scene.
            self.historical_view_utility -= config.unmatched_penalty
            self.unmatched_count += 1
        elif event == StabilityEvent.GEOMETRIC_CONFLICT:
            self.geometry_currentness -= config.conflict_penalty
            self.historical_view_utility -= config.conflict_penalty
            self.conflicts += 1
        elif event == StabilityEvent.CHANGE_EVIDENCE:
            self.geometry_currentness -= config.change_penalty
            self.historical_view_utility -= 0.5 * config.change_penalty
            self.conflicts += 1
        elif event == StabilityEvent.CONFIDENT_FALSE_POSE:
            self.geometry_currentness -= config.false_pose_penalty
            self.historical_view_utility -= config.false_pose_penalty
            self.conflicts += 1
        else:
            raise ValueError(f"Unsupported stability event: {event}")
        self._clamp()
        self._transition(config)
        self.history.append(
            StabilityHistoryEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event=event,
                geometry_currentness_before=before_current,
                geometry_currentness_after=self.geometry_currentness,
                historical_view_utility_before=before_utility,
                historical_view_utility_after=self.historical_view_utility,
                note=note,
            )
        )
