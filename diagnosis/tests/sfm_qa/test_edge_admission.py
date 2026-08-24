"""Behavioral tests: incomplete or shared-map evidence cannot be STRONG/USABLE."""

from __future__ import annotations

from sfm_qa.session_select import (
    classify_fusion_authorization,
    classify_session_edge,
    connection_is_admissible,
    usable_geometry_ready,
)
from sfm_qa.session_select.admission import (
    evaluate_geometry_authority,
    incident_fusion_authorization,
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


def test_split_evidence_across_edges_is_not_geometry_authorized():
    groups_only = SessionEdgeQuality(
        session_a="A",
        session_b="B",
        num_verified_pairs=80,
        independent_bridge_groups=2,
        independent_artifact=True,
        evidence_scope="exact_pair",
        geometry_complete=False,
        group_holdout_disjoint=True,
        fit_evidence_ids=("f1", "f2"),
        holdout_evidence_ids=("h1", "h2"),
        status="WEAK",
    )
    complete_only = SessionEdgeQuality(
        session_a="A",
        session_b="C",
        num_verified_pairs=80,
        independent_bridge_groups=0,
        independent_artifact=True,
        evidence_scope="exact_pair",
        geometry_complete=True,
        group_holdout_disjoint=False,
        fit_evidence_ids=("f3",),
        holdout_evidence_ids=(),
        status="WEAK",
    )
    first = evaluate_geometry_authority(groups_only)
    second = evaluate_geometry_authority(complete_only)
    assert first.authorized is False
    assert second.authorized is False
    assert first.hard_status == "HARD_FAIL"
    assert second.hard_status == "HARD_FAIL"
    fusion, grant, receipts = incident_fusion_authorization(
        "GEOMETRY_REINFORCEMENT",
        [groups_only, complete_only],
    )
    assert grant is None
    assert fusion == "LOCAL_RELATION_ONLY"
    assert all(not item.authorized for item in receipts)
    core_fusion, _, _ = incident_fusion_authorization(
        "BASE_CORE",
        [groups_only, complete_only],
    )
    assert core_fusion == "GLOBAL_BA_PENDING_APPROVAL"


def test_role_alone_is_not_fusion_authority():
    fusion, grant, receipts = incident_fusion_authorization("BASE_CORE", [])
    assert grant is None
    assert receipts == ()
    assert fusion == "GLOBAL_BA_PENDING_APPROVAL"
    support, _, _ = incident_fusion_authorization("BASE_SUPPORT", [])
    assert support == "LOCAL_RELATION_ONLY"
    vpr = SessionEdgeQuality(
        session_a="A",
        session_b="B",
        num_candidate_pairs=275,
        num_verified_pairs=0,
        independent_bridge_groups=0,
        evidence_scope="vpr",
        independent_artifact=False,
        status="REJECT",
    )
    vpr_fusion, vpr_grant, vpr_receipts = incident_fusion_authorization("BASE_CORE", [vpr])
    assert vpr_grant is None
    assert vpr_fusion == "GLOBAL_BA_PENDING_APPROVAL"
    assert vpr_receipts[0].authorized is False
    assert vpr_receipts[0].hard_status == "HARD_FAIL"
    assert vpr_receipts[0].evidence_status == "INSUFFICIENT_EVIDENCE"


def test_complete_authorized_edge_grants_geometry_authority():
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
    receipt = evaluate_geometry_authority(edge)
    assert receipt.authorized is True
    assert receipt.hard_status == "VALID"
    assert receipt.evidence_status == "PASS"
    assert receipt.ready is True
    assert receipt.admit_why == "admissible"
    assert "reporting/review" in receipt.authority
    assert "single incident edge" in receipt.independence_assumptions
    fusion, grant, _ = incident_fusion_authorization("BASE_CORE", [edge])
    assert grant is not None
    assert grant.authorized is True
    assert fusion == "GLOBAL_BA"
    reinforce, _, _ = incident_fusion_authorization("GEOMETRY_REINFORCEMENT", [edge])
    assert reinforce == "LOCAL_FUSION"


def test_shared_map_and_ambiguous_edges_are_not_authoritative():
    shared = classify_session_edge(
        "A",
        "B",
        num_verified_pairs=25066,
        independent_bridge_groups=3,
        trusted_geometry=True,
        source="shared_map",
        shared_map=True,
        **_complete_metrics(),
    )
    receipt = evaluate_geometry_authority(shared)
    assert receipt.authorized is False
    assert receipt.hard_status == "HARD_FAIL"
    assert receipt.evidence_scope != "exact_pair" or receipt.independent_artifact is False
    fusion, grant, _ = incident_fusion_authorization("BASE_SUPPORT", [shared])
    assert grant is None
    assert fusion == "LOCAL_RELATION_ONLY"
    one_group = SessionEdgeQuality(
        session_a="A",
        session_b="D",
        num_verified_pairs=40,
        independent_bridge_groups=1,
        independent_artifact=True,
        evidence_scope="exact_pair",
        geometry_complete=True,
        group_holdout_disjoint=True,
        fit_evidence_ids=("f1",),
        holdout_evidence_ids=("h1",),
        status="USABLE",
    )
    local = evaluate_geometry_authority(one_group)
    assert local.ready is True
    assert local.authorized is False
    assert local.evidence_status == "WARN"
    assert local.hard_status == "HARD_FAIL"


def _typed_authority_payload(**overrides):
    payload = {
        "session_a": "A",
        "session_b": "B",
        "independent_artifact": True,
        "evidence_scope": "exact_pair",
        "geometry_complete": True,
        "group_holdout_disjoint": True,
        "independent_bridge_groups": 2,
        "fit_evidence_ids": ("f1", "f2"),
        "holdout_evidence_ids": ("h1", "h2"),
        "status": "STRONG",
        "is_critical_bridge": False,
    }
    payload.update(overrides)
    return payload


def test_string_false_flags_cannot_authorize_geometry():
    receipt = evaluate_geometry_authority(
        _typed_authority_payload(
            independent_artifact="false",
            geometry_complete="false",
            group_holdout_disjoint="false",
        )
    )
    assert receipt.authorized is False
    assert receipt.independent_artifact is False
    assert receipt.geometry_complete is False
    assert receipt.group_holdout_disjoint is False
    assert "independent_artifact_not_bool" in receipt.reasons
    assert "geometry_complete_not_bool" in receipt.reasons
    assert "group_holdout_disjoint_not_bool" in receipt.reasons
    fusion, grant, _ = incident_fusion_authorization(
        "BASE_CORE",
        [_typed_authority_payload(
            independent_artifact="false",
            geometry_complete="false",
            group_holdout_disjoint="false",
        )],
    )
    assert grant is None
    assert fusion != "GLOBAL_BA"


def test_truthy_integer_flags_cannot_authorize_geometry():
    receipt = evaluate_geometry_authority(
        _typed_authority_payload(
            independent_artifact=1,
            geometry_complete=1,
            group_holdout_disjoint=1,
        )
    )
    assert receipt.authorized is False
    assert "independent_artifact_not_bool" in receipt.reasons
    assert "geometry_complete_not_bool" in receipt.reasons
    assert "group_holdout_disjoint_not_bool" in receipt.reasons


def test_bool_group_count_cannot_authorize_geometry():
    receipt = evaluate_geometry_authority(
        _typed_authority_payload(independent_bridge_groups=True)
    )
    assert receipt.authorized is False
    assert receipt.independent_bridge_groups == 0
    assert "independent_bridge_groups_is_bool" in receipt.reasons
    assert "fewer_than_min_independent_bridge_groups" in receipt.reasons


def test_fractional_group_count_is_not_truncated():
    receipt = evaluate_geometry_authority(
        _typed_authority_payload(independent_bridge_groups=2.9)
    )
    assert receipt.authorized is False
    assert receipt.independent_bridge_groups == 0
    assert "independent_bridge_groups_not_integral" in receipt.reasons
    assert "fewer_than_min_independent_bridge_groups" in receipt.reasons


def test_nan_group_count_cannot_authorize_geometry():
    receipt = evaluate_geometry_authority(
        _typed_authority_payload(independent_bridge_groups=float("nan"))
    )
    assert receipt.authorized is False
    assert receipt.independent_bridge_groups == 0
    assert "independent_bridge_groups_not_finite" in receipt.reasons


def test_negative_group_count_cannot_authorize_geometry():
    receipt = evaluate_geometry_authority(
        _typed_authority_payload(independent_bridge_groups=-2)
    )
    assert receipt.authorized is False
    assert receipt.independent_bridge_groups == 0
    assert "independent_bridge_groups_negative" in receipt.reasons


def test_malformed_mapping_with_strong_status_cannot_authorize_fusion():
    payload = _typed_authority_payload(
        independent_artifact="false",
        geometry_complete="false",
        group_holdout_disjoint="false",
        independent_bridge_groups=2.9,
    )
    receipt = evaluate_geometry_authority(payload)
    assert receipt.authorized is False
    assert receipt.hard_status == "HARD_FAIL"
    assert receipt.independent_bridge_groups != 2
    fusion, grant, receipts = incident_fusion_authorization("BASE_CORE", [payload])
    assert grant is None
    assert fusion != "GLOBAL_BA"
    assert all(not item.authorized for item in receipts)
    reinforce, _, _ = incident_fusion_authorization("GEOMETRY_REINFORCEMENT", [payload])
    assert reinforce != "LOCAL_FUSION"
