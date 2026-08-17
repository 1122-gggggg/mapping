from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .models import HistoricalReference, QueryResult
from .states import ReferenceState


class LocalizationPass(Protocol):
    def localize(
        self,
        query_id: str,
        query_path: Path,
        current_reference_ids: Sequence[str],
        historical_references: Sequence[HistoricalReference],
    ) -> QueryResult: ...


@dataclass
class OnlineDecision:
    result: QueryResult
    used_historical_fallback: bool
    historical_reference_ids: list[str]
    reason: str


class HistoricalReferenceIndex:
    def __init__(
        self,
        references: Sequence[HistoricalReference],
        route_cell_support: dict[str, set[str]] | None = None,
    ):
        self.references = {reference.reference_id: reference for reference in references}
        self.route_cell_support = route_cell_support or {}

    def active_for_cell(self, route_cell: str | None, limit: int = 20) -> list[HistoricalReference]:
        allowed_ids = self.route_cell_support.get(route_cell, set()) if route_cell else set()
        candidates = [
            reference
            for reference in self.references.values()
            if reference.state == ReferenceState.HIST_ACTIVE
            and (not allowed_ids or reference.reference_id in allowed_ids)
        ]
        candidates.sort(
            key=lambda reference: (
                reference.stable_ratio,
                len(reference.current_point3d_ids),
                -reference.bridge_depth,
            ),
            reverse=True,
        )
        return candidates[:limit]


class CurrentFirstLocalizer:
    """Production controller for current-first, historical-on-demand localization."""

    def __init__(
        self,
        backend: LocalizationPass,
        current_reference_ids: Sequence[str],
        historical_index: HistoricalReferenceIndex,
        max_historical_references: int = 20,
    ):
        self.backend = backend
        self.current_reference_ids = list(current_reference_ids)
        self.historical_index = historical_index
        self.max_historical_references = max_historical_references

    @staticmethod
    def is_strong(result: QueryResult) -> bool:
        return (
            result.success
            and result.quality.passed
            and result.quality.pose_mode_count <= 1
            and not result.confident_wrong_pose
        )

    def localize(
        self,
        query_id: str,
        query_path: str | Path,
        route_cell: str | None = None,
    ) -> OnlineDecision:
        query_path = Path(query_path)
        current = self.backend.localize(
            query_id,
            query_path,
            self.current_reference_ids,
            [],
        )
        if self.is_strong(current):
            return OnlineDecision(current, False, [], "current_pass_strong")
        historical = self.historical_index.active_for_cell(
            route_cell, self.max_historical_references
        )
        if not historical:
            return OnlineDecision(current, False, [], "no_active_historical_reference")
        fallback = self.backend.localize(
            query_id,
            query_path,
            self.current_reference_ids,
            historical,
        )
        if fallback.quality.pose_mode_count > 1:
            fallback.success = False
            fallback.failure_reason = "AMBIGUOUS_MULTIMODAL"
            return OnlineDecision(
                fallback,
                True,
                [reference.reference_id for reference in historical],
                "historical_fallback_ambiguous_fail_closed",
            )
        if self.is_strong(fallback):
            return OnlineDecision(
                fallback,
                True,
                [reference.reference_id for reference in historical],
                "historical_fallback_recovered",
            )
        return OnlineDecision(
            fallback,
            True,
            [reference.reference_id for reference in historical],
            "historical_fallback_failed",
        )
