from __future__ import annotations

import csv
from pathlib import Path

from sfm_qa.session_select.paper_graph import (
    harden_session_graph,
    planar_consistency,
    write_hardening_outputs,
)


def _edge(a: str, b: str, **updates):
    row = {
        "session_a": a,
        "session_b": b,
        "status": "STRONG",
        "num_candidate_pairs": 20,
        "num_verified_pairs": 120,
        "num_cross_session_tracks": 120,
        "independent_bridge_groups": 2,
        "inlier_ratio": 0.8,
        "holdout_inlier_ratio": 0.75,
        "rotation_consensus_deg": 1.0,
        "translation_direction_consensus_deg": 3.0,
        "scale_consensus": 0.03,
        "cross_session_reprojection_error": 1.5,
        "spatial_coverage": 0.45,
        "parallax_deg": 2.0,
        "edge_positive_depth_ratio": 0.97,
        "holdout_residual": 2.0,
        "edge_quality_score": 0.85,
        "independent_artifact": True,
        "evidence_scope": "exact_pair",
        "geometry_complete": True,
        "group_holdout_disjoint": True,
        "reasons": "verified_multi_bridge_geometry",
    }
    row.update(updates)
    return row


def test_retrieval_only_edge_never_enters_geometric_graph():
    rows = [
        _edge("a", "b"),
        {
            "session_a": "b",
            "session_b": "c",
            "status": "STRONG",
            "num_candidate_pairs": 100,
            "num_verified_pairs": 0,
            "independent_bridge_groups": 0,
            "evidence_scope": "vpr",
            "independent_artifact": False,
            "geometry_complete": False,
            "group_holdout_disjoint": False,
        },
    ]
    report = harden_session_graph(["a", "b", "c"], rows)
    assert report["counts"]["eligible_pairs"] == 1
    vpr = report["edge_rows"][1]
    assert vpr["paper_graph_eligible"] is False
    assert vpr["status"] == "STRONG"
    assert vpr["graph_pruned"] is False


def test_planarity_can_be_validated_instead_of_automatically_penalized():
    config = {
        "paper_graph": {
            "planar": {
                "minimum_observed_terms": 2,
                "validated_score": 0.65,
            }
        }
    }
    good = planar_consistency(
        {
            "homography_shared_inlier_ratio": 0.82,
            "plane_normal_similarity": 0.95,
            "homography_rotation_error_deg": 1.0,
        },
        harden_session_graph([], [], config)["config"],
    )
    bad = planar_consistency(
        {
            "homography_shared_inlier_ratio": 0.2,
            "plane_normal_similarity": 0.1,
            "homography_rotation_error_deg": 20.0,
        },
        harden_session_graph([], [], config)["config"],
    )
    assert good["validated"] is True
    assert good["score"] > 0.8
    assert bad["validated"] is False
    assert bad["score"] < 0.35


def test_minimum_range_backbone_downgrades_redundant_inconsistent_edge():
    rows = [
        _edge("a", "b"),
        _edge("b", "c", rotation_consensus_deg=1.2, edge_quality_score=0.86),
        _edge(
            "a",
            "c",
            rotation_consensus_deg=4.9,
            translation_direction_consensus_deg=14.5,
            scale_consensus=0.14,
            cross_session_reprojection_error=4.9,
            holdout_residual=7.9,
            inlier_ratio=0.46,
            holdout_inlier_ratio=0.46,
            edge_quality_score=0.25,
            homography_shared_inlier_ratio=0.15,
            plane_normal_similarity=0.1,
            homography_rotation_error_deg=15.0,
        ),
    ]
    config = {
        "paper_graph": {
            "embedding": {
                "prune_margin": 0.5,
                "relative_feature_anomaly_threshold": 1.0,
                "low_reliability_ratio": 0.95,
            }
        }
    }
    report = harden_session_graph(["a", "b", "c"], rows, config)
    by_pair = {
        tuple(sorted((row["session_a"], row["session_b"]))): row
        for row in report["edge_rows"]
    }
    assert by_pair[("a", "c")]["graph_pruned"] is True
    assert by_pair[("a", "c")]["status"] == "AMBIGUOUS"
    assert by_pair[("a", "b")]["graph_backbone"] is True
    assert by_pair[("b", "c")]["graph_backbone"] is True
    assert ["a", "c"] in report["pruned_pairs"]


def test_strong_disconnected_community_is_new_submap_not_quarantined():
    rows = [
        _edge("a", "b"),
        _edge("b", "c"),
        _edge("a", "c"),
        _edge("x", "y", edge_quality_score=0.95, inlier_ratio=0.9),
    ]
    report = harden_session_graph(["a", "b", "c", "x", "y"], rows)
    assert set(report["new_submap_candidates"]) == {"x", "y"}
    assert not report["quarantined_sessions"]


def test_weak_small_disconnected_community_is_quarantined():
    rows = [
        _edge("a", "b"),
        _edge("b", "c"),
        _edge("a", "c"),
        _edge(
            "x",
            "y",
            edge_quality_score=0.1,
            inlier_ratio=0.46,
            holdout_inlier_ratio=0.46,
            rotation_consensus_deg=4.8,
            translation_direction_consensus_deg=14.0,
            cross_session_reprojection_error=4.8,
        ),
    ]
    report = harden_session_graph(["a", "b", "c", "x", "y"], rows)
    assert set(report["quarantined_sessions"]) == {"x", "y"}
    xy = next(
        row
        for row in report["edge_rows"]
        if {row["session_a"], row["session_b"]} == {"x", "y"}
    )
    assert xy["status"] == "AMBIGUOUS"
    assert xy["graph_community_quarantined"] is True


def test_no_geometry_does_not_invent_a_base_anchor():
    report = harden_session_graph(
        ["a", "b"],
        [],
        protected_sessions=["frozen"],
    )
    assert set(report["sessions"]) == {"a", "b", "frozen"}
    assert set(report["quarantined_sessions"]) == {"a", "b"}
    roles = {row["sessions"][0]: row["role"] for row in report["communities"]}
    assert roles["frozen"] == "PROTECTED_KEEP"
    assert "BASE_CONNECTED" not in roles.values()


def test_schedule_and_output_contract(tmp_path: Path):
    report = harden_session_graph(
        ["a", "b", "c"],
        [_edge("a", "b"), _edge("b", "c"), _edge("a", "c")],
    )
    stages = [row["stage"] for row in report["optimization_schedule"]]
    assert "LOCAL" in stages
    assert "NEIGHBOR" in stages
    assert "GLOBAL_COMPONENT" in stages
    outputs = write_hardening_outputs(report, tmp_path)
    assert Path(outputs["report"]).is_file()
    assert Path(outputs["hardened_edges"]).is_file()
    with Path(outputs["hardened_edges"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert "graph_backbone" in rows[0]
