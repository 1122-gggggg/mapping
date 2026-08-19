from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..config import LifelongConfig
from .fremen import fit_fremen_model, rank_candidates_by_uniqueness
from .models import (
    FeatureCandidate,
    FeatureEvent,
    FeatureMemoryRecord,
    FeatureSelection,
    FeatureState,
    MapManagementStrategy,
    MapUpdatePlan,
    TemporalScoreModel,
)


class PredictiveAdaptiveMapManager:
    """Geometry-safe score-based and FreMEn feature-map manager.

    Only a sidecar active set is modified. The manager never receives or mutates ``BaseMap``.
    Unverified or historical-only geometry is quarantined, and every learning operation is
    fail-closed behind ``gate_passed``.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        config: LifelongConfig | None = None,
        records: Mapping[str, FeatureMemoryRecord] | Sequence[FeatureMemoryRecord] | None = None,
    ) -> None:
        self.config = config or LifelongConfig()
        errors = self.config.validate()
        if errors:
            raise ValueError("; ".join(errors))
        if records is None:
            items: list[FeatureMemoryRecord] = []
        elif isinstance(records, Mapping):
            items = list(records.values())
        else:
            items = list(records)
        self.records = {item.feature_id: item for item in items}
        if len(self.records) != len(items):
            raise ValueError("feature_id values must be unique")

    @property
    def active_ids(self) -> list[str]:
        return sorted(
            feature_id
            for feature_id, record in self.records.items()
            if record.state == FeatureState.ACTIVE
        )

    @property
    def retired_ids(self) -> list[str]:
        return sorted(
            feature_id
            for feature_id, record in self.records.items()
            if record.state == FeatureState.RETIRED
        )

    @property
    def quarantined_ids(self) -> list[str]:
        return sorted(
            feature_id
            for feature_id, record in self.records.items()
            if record.state == FeatureState.QUARANTINED
        )

    def register_feature(
        self,
        feature_id: str | int,
        point3d_id: int | None,
        descriptor: np.ndarray | None = None,
        *,
        verified_current_geometry: bool = True,
        score: float | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> FeatureMemoryRecord:
        key = str(feature_id)
        if key in self.records:
            raise ValueError(f"feature already registered: {key}")
        if verified_current_geometry and point3d_id is None:
            raise ValueError("verified current geometry requires point3d_id")
        record = FeatureMemoryRecord(
            feature_id=key,
            point3d_id=point3d_id,
            descriptor=descriptor,
            score=self.config.initial_score if score is None else score,
            state=(
                FeatureState.ACTIVE
                if verified_current_geometry
                else FeatureState.QUARANTINED
            ),
            verified_current_geometry=verified_current_geometry,
            metadata=dict(metadata or {}),
        )
        self.records[key] = record
        return record

    def temporal_model(self, feature_id: str | int) -> TemporalScoreModel:
        return fit_fremen_model(self.records[str(feature_id)].observations, self.config)

    def select_features(
        self,
        timestamp_days: float,
        limit: int | None = None,
        strategy: MapManagementStrategy | str | None = None,
    ) -> list[FeatureSelection]:
        selected_strategy = self._strategy(strategy)
        selections: list[FeatureSelection] = []
        for feature_id in self.active_ids:
            record = self.records[feature_id]
            if selected_strategy == MapManagementStrategy.FREMEN:
                model = fit_fremen_model(record.observations, self.config)
                predicted = model.predict(timestamp_days)
                sample_count = model.sample_count
            else:
                predicted = record.score
                sample_count = len(record.observations)
            selections.append(
                FeatureSelection(
                    feature_id=feature_id,
                    predicted_score=float(predicted),
                    scalar_score=float(record.score),
                    temporal_sample_count=sample_count,
                )
            )
        selections.sort(
            key=lambda item: (-item.predicted_score, -item.scalar_score, item.feature_id)
        )
        maximum = self.config.query_budget if limit is None else limit
        if maximum < 0:
            raise ValueError("limit must be non-negative")
        return selections[:maximum]

    def update_session(
        self,
        *,
        events: Mapping[str | int, FeatureEvent | str],
        timestamp_days: float,
        candidates: Sequence[FeatureCandidate] = (),
        gate_passed: bool,
        gate_reason: str = "",
        strategy: MapManagementStrategy | str | None = None,
    ) -> MapUpdatePlan:
        selected_strategy = self._strategy(strategy)
        active_before = self.active_ids
        plan = MapUpdatePlan(
            strategy=selected_strategy,
            applied=False,
            gate_passed=gate_passed,
            gate_reason=gate_reason,
            active_before=active_before,
            active_after=list(active_before),
        )
        if not np.isfinite(timestamp_days):
            raise ValueError("timestamp_days must be finite")
        if not gate_passed:
            plan.ignored_events = sorted(str(item) for item in events)
            plan.ignored_candidates = sorted(candidate.feature_id for candidate in candidates)
            plan.metadata["reason"] = "registration_gate_failed"
            return plan
        if selected_strategy == MapManagementStrategy.STATIC:
            plan.applied = True
            plan.ignored_events = sorted(str(item) for item in events)
            plan.ignored_candidates = sorted(candidate.feature_id for candidate in candidates)
            plan.metadata["reason"] = "static_map"
            return plan

        normalized_events: dict[str, FeatureEvent] = {}
        for feature_id, event in events.items():
            normalized_events[str(feature_id)] = (
                event if isinstance(event, FeatureEvent) else FeatureEvent(str(event))
            )
        for feature_id, event in sorted(normalized_events.items()):
            record = self.records.get(feature_id)
            if record is None or record.state != FeatureState.ACTIVE:
                plan.ignored_events.append(feature_id)
                continue
            record.observe(event, timestamp_days, self.config)
            plan.observed.append(feature_id)

        admissible, quarantined = self._partition_candidates(candidates)
        plan.quarantined.extend(self._store_quarantined(quarantined))
        active_descriptors = [
            record.descriptor
            for record in self.records.values()
            if record.state == FeatureState.ACTIVE and record.descriptor is not None
        ]
        ranked_candidates = rank_candidates_by_uniqueness(
            admissible,
            active_descriptors,
            self.config.descriptor_metric,
        )
        candidate_ids = [item.feature_id for item, _score in ranked_candidates]
        plan.metadata["candidate_uniqueness"] = {
            item.feature_id: score for item, score in ranked_candidates
        }
        active_set = {
            feature_id
            for feature_id, record in self.records.items()
            if record.state == FeatureState.ACTIVE
        }

        if selected_strategy == MapManagementStrategy.LATEST:
            chosen = candidate_ids[: self.config.map_budget]
            if chosen:
                plan.retired.extend(self._retire(self.active_ids))
                plan.activated.extend(self._activate(chosen, admissible))
            else:
                plan.metadata["latest_noop"] = "no_admissible_candidates"
        elif selected_strategy in {
            MapManagementStrategy.AGGRESSIVE,
            MapManagementStrategy.STRICT,
        }:
            removable = {FeatureEvent.INCORRECT}
            if selected_strategy == MapManagementStrategy.AGGRESSIVE:
                removable.add(FeatureEvent.UNMATCHED)
            requested = sorted(
                feature_id
                for feature_id, event in normalized_events.items()
                if event in removable and feature_id in active_set
            )
            available = [item for item in candidate_ids if item not in active_set]
            replacement_count = min(len(requested), len(available))
            plan.retired.extend(self._retire(requested[:replacement_count]))
            capacity = max(self.config.map_budget - len(self.active_ids), 0)
            chosen = available[: min(replacement_count, capacity)]
            plan.activated.extend(self._activate(chosen, admissible))
        elif selected_strategy == MapManagementStrategy.SUMMARY:
            requested = [
                feature_id
                for feature_id, event in normalized_events.items()
                if event == FeatureEvent.INCORRECT and feature_id in active_set
            ]
            plan.retired.extend(self._retire(requested))
            chosen = [item for item in candidate_ids if item not in active_set]
            plan.activated.extend(self._activate(chosen, admissible))
        elif selected_strategy in {
            MapManagementStrategy.SCORE,
            MapManagementStrategy.FREMEN,
        }:
            available = [item for item in candidate_ids if item not in active_set]
            overflow = max(len(active_set) - self.config.map_budget, 0)
            if overflow > 0:
                plan.retired.extend(
                    self._retire(self._worst_active(overflow, selected_strategy))
                )
            capacity = max(self.config.map_budget - len(self.active_ids), 0)
            fill_count = min(capacity, len(available))
            if fill_count > 0:
                fill_ids = available[:fill_count]
                plan.activated.extend(self._activate(fill_ids, admissible))
                available = available[fill_count:]
                plan.metadata["capacity_fill"] = fill_count
            had_full_map = len(active_before) >= self.config.map_budget
            plan.exchange_target = self._exchange_target() if had_full_map else 0
            exchange_count = min(plan.exchange_target, len(available))
            if exchange_count > 0:
                worst = self._worst_active(exchange_count, selected_strategy)
                plan.retired.extend(self._retire(worst))
                exchange_ids = available[:exchange_count]
                plan.activated.extend(self._activate(exchange_ids, admissible))
                plan.exchange_applied = len(exchange_ids)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported strategy: {selected_strategy}")

        plan.applied = True
        plan.active_after = self.active_ids
        plan.metadata.update(
            {
                "active_count_before": len(active_before),
                "active_count_after": len(plan.active_after),
                "map_budget": self.config.map_budget,
            }
        )
        return plan

    def _strategy(
        self,
        strategy: MapManagementStrategy | str | None,
    ) -> MapManagementStrategy:
        value = self.config.strategy if strategy is None else strategy
        if isinstance(value, MapManagementStrategy):
            return value
        return MapManagementStrategy(str(value))

    @staticmethod
    def _partition_candidates(
        candidates: Sequence[FeatureCandidate],
    ) -> tuple[list[FeatureCandidate], list[FeatureCandidate]]:
        deduplicated: dict[str, FeatureCandidate] = {}
        for candidate in candidates:
            deduplicated.setdefault(candidate.feature_id, candidate)
        admissible: list[FeatureCandidate] = []
        quarantined: list[FeatureCandidate] = []
        for candidate in deduplicated.values():
            geometry_ok = candidate.verified_current_geometry and candidate.point3d_id is not None
            if geometry_ok:
                admissible.append(candidate)
            else:
                quarantined.append(candidate)
        return admissible, quarantined

    def _store_quarantined(self, candidates: Sequence[FeatureCandidate]) -> list[str]:
        stored: list[str] = []
        for candidate in sorted(candidates, key=lambda item: item.feature_id):
            existing = self.records.get(candidate.feature_id)
            if existing is not None and existing.state == FeatureState.ACTIVE:
                continue
            self.records[candidate.feature_id] = FeatureMemoryRecord(
                feature_id=candidate.feature_id,
                point3d_id=candidate.point3d_id,
                descriptor=candidate.descriptor,
                score=self.config.initial_score,
                state=FeatureState.QUARANTINED,
                verified_current_geometry=False,
                metadata=dict(candidate.metadata),
            )
            stored.append(candidate.feature_id)
        return stored

    def _activate(
        self,
        feature_ids: Sequence[str],
        candidates: Sequence[FeatureCandidate],
    ) -> list[str]:
        by_id = {item.feature_id: item for item in candidates}
        activated: list[str] = []
        for feature_id in feature_ids:
            candidate = by_id[feature_id]
            if not candidate.verified_current_geometry or candidate.point3d_id is None:
                continue
            existing = self.records.get(feature_id)
            if existing is None:
                self.records[feature_id] = FeatureMemoryRecord(
                    feature_id=feature_id,
                    point3d_id=candidate.point3d_id,
                    descriptor=candidate.descriptor,
                    score=(
                        self.config.initial_score
                        if candidate.initial_score is None
                        else candidate.initial_score
                    ),
                    state=FeatureState.ACTIVE,
                    verified_current_geometry=True,
                    metadata=dict(candidate.metadata),
                )
            else:
                existing.point3d_id = candidate.point3d_id
                existing.descriptor = (
                    None if candidate.descriptor is None else candidate.descriptor.copy()
                )
                existing.verified_current_geometry = True
                existing.state = FeatureState.ACTIVE
                existing.metadata.update(candidate.metadata)
            activated.append(feature_id)
        return activated

    def _retire(self, feature_ids: Iterable[str]) -> list[str]:
        retired: list[str] = []
        for feature_id in feature_ids:
            record = self.records.get(feature_id)
            if record is None or record.state != FeatureState.ACTIVE:
                continue
            record.state = FeatureState.RETIRED
            retired.append(feature_id)
        return retired

    def _exchange_target(self) -> int:
        active_count = len(self.active_ids)
        if active_count == 0 or self.config.exchange_fraction <= 0.0:
            return 0
        calculated = int(np.ceil(active_count * self.config.exchange_fraction))
        return min(active_count, max(self.config.min_exchange_count, calculated))

    def _worst_active(
        self,
        count: int,
        strategy: MapManagementStrategy,
    ) -> list[str]:
        if count <= 0:
            return []
        records = [
            record
            for record in self.records.values()
            if record.state == FeatureState.ACTIVE
        ]
        if strategy == MapManagementStrategy.FREMEN:
            records.sort(key=lambda item: (item.mean_event_score, item.score, item.feature_id))
        else:
            records.sort(key=lambda item: (item.score, item.feature_id))
        return [item.feature_id for item in records[:count]]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "records": [self.records[item].to_dict() for item in sorted(self.records)],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        config: LifelongConfig | None = None,
    ) -> "PredictiveAdaptiveMapManager":
        version = int(payload.get("schema_version", 0))
        if version != cls.SCHEMA_VERSION:
            raise ValueError(f"Unsupported lifelong-memory schema version: {version}")
        raw_records = payload.get("records", [])
        if not isinstance(raw_records, Sequence):
            raise ValueError("records must be a sequence")
        records: list[FeatureMemoryRecord] = []
        for item in raw_records:
            if not isinstance(item, Mapping):
                raise ValueError("each record must be an object")
            records.append(FeatureMemoryRecord.from_dict(item))
        return cls(config=config, records=records)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: LifelongConfig | None = None,
    ) -> "PredictiveAdaptiveMapManager":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("lifelong-memory JSON root must be an object")
        return cls.from_dict(payload, config=config)
