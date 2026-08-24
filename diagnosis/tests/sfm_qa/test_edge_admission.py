"""Behavioral tests: incomplete or shared-map evidence cannot be STRONG/USABLE."""

from __future__ import annotations

from sfm_qa.session_select import (
    classify_fusion_authorization,
    classify_session_edge,
    connection_is_admissible,
    usable_geometry_ready,
)
from sfm_qa.session_select.types import SessionEdgeQuality


def _complete_metrics() -> dict:
    return dict(
        rotation_consensus_deg=0.4,
        translation_direction_consensus_deg=1.0,
        scale_consensus=0.04,
        parallax_deg=3.0,
        edge_positive_depth_ratio=0.99,
        spatial_coverage=0.4,
        cross_session_reprojection_error=1.5,
        holdout_inlier_ratio=0.7,
        holdout_residual=2.0,
    )


def _independent_groups() -> dict:
    return {
        "g1": {
            "evidence_ids": ["f1", "f2"],
            "query_frame_indices": [0, 2],
            "reference_frame_indices": [100, 102],
            "query_image_ids": ["qa", "qb"],
            "reference_image_ids": ["ra", "rb"],
            "landmark_ids": ["l1", "l2"],
            "region_ids": ["north"],
            "time": ["t0"],
        },
        "g2": {
            "evidence_ids": ["h1", "h2"],
            "query_frame_indices": [80, 82],
            "reference_frame_indices": [200, 202],
            "query_image_ids": ["qc", "qd"],
            "reference_image_ids": ["rc", "rd"],
            "landmark_ids": ["l9", "l10"],
            "region_ids": ["south"],
            "time": ["t1"],
        },
    }


def test_shared_map_tracks_cannot_be_strong_or_usable():
    edge = classify_session_edge(
        "A",
        "B",
        num_verified_pairs=25066,
        independent_bridge_groups=3,
        num_cross_session_tracks=25066,
        trusted_geometry=True,
        source="shared_map",
        shared_map=True,
        **_complete_metrics(),
    )
    assert edge.status not in {"STRONG", "USABLE"}
    assert edge.independent_artifact is False
    assert edge.evidence_scope == "shared_map"


def test_vpr_only_cannot_be_strong_or_usable():
    edge = classify_session_edge(
        "A",
        "B",
        num_verified_pairs=0,
        independent_bridge_groups=0,
        num_candidate_pairs=275,
        evidence_scope="vpr",
        source="vpr",
    )
    assert edge.status not in {"STRONG", "USABLE"}
    assert edge.status == "REJECT"


def test_legacy_trusted_geometry_without_holdout_is_downgraded():
    edge = classify_session_edge(
        "A",
        "B",
        num_verified_pairs=800,
        independent_bridge_groups=3,
        trusted_geometry=True,
        rotation_consensus_deg=0.4,
        scale_consensus=0.05,
    )
    assert edge.status not in {"STRONG", "USABLE"}
    assert "legacy_or_incomplete_evidence_downgraded" in edge.reasons


def test_overlapping_fit_holdout_cannot_be_strong_or_usable():
    edge = classify_session_edge(
        "A",
        "B",
        num_verified_pairs=200,
        independent_bridge_groups=2,
        independent_artifact=True,
        evidence_scope="exact_pair",
        fit_evidence_ids=("f1", "h1"),
        holdout_evidence_ids=("h1", "h2"),
        bridge_groups=_independent_groups(),
        **_complete_metrics(),
    )
    assert edge.status not in {"STRONG", "USABLE"}
    assert edge.group_holdout_disjoint is False


def test_incomplete_geometry_cannot_be_strong_or_usable():
    edge = classify_session_edge(
        "A",
        "B",
        num_verified_pairs=200,
        independent_bridge_groups=2,
        independent_artifact=True,
        evidence_scope="exact_pair",
        fit_evidence_ids=("f1", "f2"),
        holdout_evidence_ids=("h1", "h2"),
        bridge_groups=_independent_groups(),
        rotation_consensus_deg=0.4,
        scale_consensus=0.04,
    )
    assert edge.status not in {"STRONG", "USABLE"}
    assert edge.geometry_complete is False


def test_complete_exact_pair_probe_can_be_strong():
    edge = classify_session_edge(
        "A",
        "B",
        num_verified_pairs=200,
        independent_bridge_groups=2,
        independent_artifact=True,
        evidence_scope="exact_pair",
        fit_evidence_ids=("f1", "f2"),
        holdout_evidence_ids=("h1", "h2"),
        bridge_groups=_independent_groups(),
        **_complete_metrics(),
    )
    assert edge.status == "STRONG"
    assert edge.independent_artifact is True
    assert edge.geometry_complete is True
    assert edge.group_holdout_disjoint is True


def test_weak_shared_map_edge_is_not_admissible_to_core():
    edge = SessionEdgeQuality(
        session_a="A",
        session_b="B",
        num_candidate_pairs=11,
        num_verified_pairs=25066,
        num_cross_session_tracks=25066,
        num_cross_session_observations=460588,
        independent_bridge_groups=3,
        inlier_count=25066,
        inlier_ratio=None,
        rotation_consensus_deg=None,
        translation_direction_consensus_deg=None,
        scale_consensus=None,
        cross_session_reprojection_error=None,
        spatial_coverage=None,
        cycle_support=None,
        cycle_error=None,
        edge_quality_score=0.35,
        is_bridge=False,
        is_critical_bridge=False,
        status="WEAK",
        reasons=("shared_reconstruction_not_independent_geometry",),
    )
    ok, reason = connection_is_admissible("B", {"A"}, [edge])
    assert ok is False
    assert reason in {"only_blocked_edges", "no_geometric_edge"}


def test_geometry_reinforcement_stays_local_without_complete_holdout():
    assert (
        classify_fusion_authorization(
            role="GEOMETRY_REINFORCEMENT",
            has_holdout=False,
            independent_bridge_groups=5,
            geometry_complete=False,
        )
        == "LOCAL_RELATION_ONLY"
    )
    assert (
        classify_fusion_authorization(
            role="GEOMETRY_REINFORCEMENT",
            has_holdout=True,
            independent_bridge_groups=2,
            geometry_complete=True,
            group_holdout_disjoint=True,
        )
        == "LOCAL_FUSION"
    )


def test_usable_geometry_ready_requires_exact_pair_contract():
    assert not usable_geometry_ready(
        independent_artifact=False,
        evidence_scope="shared_map",
        independent_groups=3,
        group_holdout_disjoint=False,
        geometry_complete=False,
    )
    assert usable_geometry_ready(
        independent_artifact=True,
        evidence_scope="exact_pair",
        independent_groups=2,
        group_holdout_disjoint=True,
        geometry_complete=True,
        fit_evidence_ids=("a",),
        holdout_evidence_ids=("b",),
    )
