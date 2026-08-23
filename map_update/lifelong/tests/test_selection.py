from pathlib import Path

import numpy as np

from update_map.config import SelectionConfig
from update_map.models import (
    HistoricalReference,
    Pose,
    ReferenceCandidate,
    UtilityBreakdown,
)
from update_map.selection import greedy_select_references
from update_map.states import ReferenceProvenance, ReferenceState


def candidate(reference_id: str, cells: set[str], utility: float, point_offset: int = 0):
    ref = HistoricalReference(
        reference_id=reference_id,
        image_path=Path(f"{reference_id}.jpg"),
        pose=Pose(np.eye(3), np.array([float(point_offset), 0.0, 0.0])),
        provenance=ReferenceProvenance.DIRECT,
        state=ReferenceState.HIST_STABLE,
        stable_ratio=0.9,
        current_point3d_ids=set(range(point_offset, point_offset + 20)),
    )
    return ReferenceCandidate(
        ref,
        cells,
        UtilityBreakdown(localizer_success_gain=utility, stable_ratio=0.9),
        visible_point3d_ids=ref.current_point3d_ids,
    )


def test_selection_prioritizes_uncovered_weak_cell() -> None:
    config = SelectionConfig(budget=1, min_k_cover=1)
    healthy = candidate("healthy", {"healthy"}, 0.9, 0)
    weak = candidate("weak", {"weak"}, 0.2, 100)
    result = greedy_select_references(
        [healthy, weak],
        config,
        current_coverage={"healthy": 2, "weak": 0},
        cell_weights={"weak": 10.0},
    )
    assert [item.reference.reference_id for item in result.selected] == ["weak"]
    assert result.coverage_after["weak"] == 1


def test_redundant_candidate_is_removed() -> None:
    config = SelectionConfig(budget=10, min_k_cover=1)
    first = candidate("a", {"cell"}, 0.5, 0)
    second = candidate("b", {"cell"}, 0.4, 0)
    first.descriptor = np.ones(8)
    second.descriptor = np.ones(8)
    result = greedy_select_references([first, second], config, current_coverage={"cell": 0})
    assert len(result.selected) == 1
    assert len(result.rejected_redundant) == 1
