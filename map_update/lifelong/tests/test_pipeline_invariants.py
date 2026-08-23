from pathlib import Path

import numpy as np
import pytest

from update_map.config import UpdateMapConfig
from update_map.models import HistoricalReference, Pose
from update_map.pipeline import HistoricalAssociation, HistoricalAugmentationPipeline
from update_map.states import ReferenceProvenance, ReferenceState
from update_map.synthetic import SyntheticMatcher, SyntheticRetriever, create_synthetic_base_map


def test_old_only_point_cannot_enter_sidecar(tmp_path: Path) -> None:
    base_map, _, _ = create_synthetic_base_map()
    pipeline = HistoricalAugmentationPipeline(
        base_map,
        UpdateMapConfig(),
        SyntheticRetriever([]),
        SyntheticMatcher({}),
        {},
    )
    reference = HistoricalReference(
        "old",
        tmp_path / "old.jpg",
        Pose.identity(),
        ReferenceProvenance.DIRECT,
        ReferenceState.HIST_STABLE,
    )
    invalid = HistoricalAssociation(
        "old",
        np.array([1.0, 2.0]),
        current_point3d_id=999999,
        confidence=1.0,
        supporting_references=2,
        provenance=ReferenceProvenance.DIRECT,
    )
    with pytest.raises(ValueError, match="forbidden point3D"):
        pipeline.export_sidecar([reference], [invalid], tmp_path / "sidecar")
