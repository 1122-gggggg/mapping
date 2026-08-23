"""Contract tests for Sim3 holdout residuals and independent bridges."""

from __future__ import annotations

import numpy as np
import pytest

from sfm_diagnosis.independent_bridges import (
    BridgeObservation,
    cluster_bridge_observations,
    decide_overlap_admission,
)
from sfm_diagnosis.sim3 import evaluate_holdout_similarity


def test_holdout_residual_split_fits_support_validates_holdout():
    support_source = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    support_target = 3.0 * support_source + np.array([2.0, -1.0, 4.0])
    holdout_source = np.array([[1.0, 1.0, 0.0]], dtype=np.float64)
    holdout_target = 3.0 * holdout_source + np.array([2.0, -1.0, 4.0])

    accepted = evaluate_holdout_similarity(
        support_source,
        support_target,
        holdout_source,
        holdout_target,
        max_relative_residual=0.05,
    )
    rejected = evaluate_holdout_similarity(
        support_source,
        support_target,
        holdout_source,
        holdout_target + np.array([1.0, 0.0, 0.0]),
        max_relative_residual=0.05,
    )

    assert accepted["consistent"] is True
    assert accepted["holdout_relative_residuals"] == pytest.approx([0.0])
    assert len(accepted["holdout_residuals"]) == 1
    assert len(accepted["holdout_relative_residuals"]) == 1
    assert rejected["consistent"] is False
    assert float(rejected["holdout_relative_residuals"][0]) > 0.05


def test_cluster_sequential_pairs_into_one_bridge_group():
    observations = [
        BridgeObservation(0, "ref/000.jpg", 0.90, (0.0, 0.0, 0.0)),
        BridgeObservation(5, "ref/001.jpg", 0.85, (0.04, 0.0, 0.0)),
        BridgeObservation(10, "ref/002.jpg", 0.80, (0.08, 0.0, 0.0)),
        BridgeObservation(400, "ref/far.jpg", 0.70, (8.0, 0.0, 0.0)),
    ]
    clusters = cluster_bridge_observations(
        observations,
        query_frame_separation=30,
        reference_center_separation=1.0,
    )
    grouped = [sorted(item.query_index for item in cluster) for cluster in clusters]
    grouped.sort(key=lambda ids: ids[0])
    assert [0, 5, 10] in grouped
    assert len(grouped) == 2
    far = [ids for ids in grouped if 400 in ids]
    assert far == [[400]]


def test_decide_overlap_admission_nogo_when_bridges_is_one():
    decision = decide_overlap_admission(
        independent_bridge_count=1,
        pnp_anchor_count=4,
        independent_source_model_available=True,
        sim3_consistent=True,
    )
    assert decision["status"] == "NO_GO"
    assert decision["map_fusion_authorized"] is False
    assert decision["independent_bridge_count"] == 1
