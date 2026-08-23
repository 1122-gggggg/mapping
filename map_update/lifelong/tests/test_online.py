from pathlib import Path

import numpy as np

from update_map.models import HistoricalReference, Pose, PoseQuality, QueryResult
from update_map.online import CurrentFirstLocalizer, HistoricalReferenceIndex
from update_map.states import ReferenceProvenance, ReferenceState


def _result(success: bool, passed: bool, modes: int = 1) -> QueryResult:
    return QueryResult(
        query_id="q",
        success=success,
        pose=Pose.identity() if success else None,
        quality=PoseQuality(passed=passed, pose_mode_count=modes),
    )


def _reference(reference_id: str, stable_ratio: float = 0.9) -> HistoricalReference:
    return HistoricalReference(
        reference_id=reference_id,
        image_path=Path(f"{reference_id}.jpg"),
        pose=Pose(np.eye(3), np.zeros(3)),
        provenance=ReferenceProvenance.DIRECT,
        state=ReferenceState.HIST_ACTIVE,
        stable_ratio=stable_ratio,
        current_point3d_ids={1, 2, 3},
    )


class SequenceBackend:
    def __init__(self, results: list[QueryResult]):
        self.results = list(results)
        self.calls: list[list[str]] = []

    def localize(self, query_id, query_path, current_reference_ids, historical_references):
        self.calls.append([reference.reference_id for reference in historical_references])
        return self.results.pop(0)


def test_current_first_uses_historical_only_after_weak_current_pass(tmp_path: Path) -> None:
    backend = SequenceBackend([_result(False, False), _result(True, True)])
    reference = _reference("h1")
    index = HistoricalReferenceIndex([reference], {"cell": {"h1"}})
    localizer = CurrentFirstLocalizer(backend, ["current-1"], index)

    decision = localizer.localize("q", tmp_path / "q.jpg", "cell")

    assert decision.result.success
    assert decision.used_historical_fallback
    assert decision.historical_reference_ids == ["h1"]
    assert backend.calls == [[], ["h1"]]


def test_historical_multimodal_result_fails_closed(tmp_path: Path) -> None:
    backend = SequenceBackend([_result(False, False), _result(True, True, modes=2)])
    index = HistoricalReferenceIndex([_reference("h1")])
    localizer = CurrentFirstLocalizer(backend, ["current-1"], index)

    decision = localizer.localize("q", tmp_path / "q.jpg")

    assert not decision.result.success
    assert decision.result.failure_reason == "AMBIGUOUS_MULTIMODAL"
    assert decision.reason == "historical_fallback_ambiguous_fail_closed"
