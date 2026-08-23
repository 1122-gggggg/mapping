from __future__ import annotations

from dataclasses import replace

import pytest

from sfm_qa.session_select import (
    SessionEdgeQuality,
    SessionQuality,
    camera_triplet_scores,
    compute_objective_terms,
    load_config,
    propose_prebuild_set,
)


def _session(session_id: str, **overrides) -> SessionQuality:
    payload = dict(
        session_id=session_id,
        timestamp="2026-01-01T00:00:00Z",
        num_frames=1000,
        num_keyframes=100,
        sharpness_median=150.0,
        sharpness_p10=100.0,
        underexposed_ratio=0.0,
        overexposed_ratio=0.0,
        near_duplicate_ratio=0.05,
        parallax_ratio=0.60,
        low_parallax_ratio=0.10,
        hover_ratio=0.05,
        pure_rotation_ratio=0.05,
        fast_motion_ratio=0.05,
        unproven_ratio=0.15,
        epipolar_outlier_ratio_median=0.10,
        essential_inlier_ratio_median=0.90,
        internal_quality_score=0.80,
        internal_status="USABLE",
    )
    payload.update(overrides)
    return SessionQuality(**payload)


def _candidate(a: str, b: str, count: int) -> SessionEdgeQuality:
    return SessionEdgeQuality(
        session_a=a,
        session_b=b,
        num_candidate_pairs=count,
        num_verified_pairs=0,
        independent_bridge_groups=0,
        status="REJECT",
        reasons=("vpr_candidate_only_not_geometry",),
    )


def test_camera_triplet_score_downweights_weak_triangle_edge() -> None:
    scores = camera_triplet_scores(
        [_candidate("A", "B", 100), _candidate("B", "C", 100), _candidate("A", "C", 10)]
    )
    assert scores[("A", "B")]["score"] == pytest.approx(1.0)
    assert scores[("B", "C")]["score"] == pytest.approx(1.0)
    assert scores[("A", "C")]["score"] == pytest.approx(0.1)
    assert scores[("A", "C")]["triplets"] == pytest.approx(1.0)


def test_prebuild_proposes_subset_and_reserves_validation_without_promoting_vpr() -> None:
    cfg = load_config()
    sessions = [_session(name) for name in ("A", "B", "C", "D")]
    sessions.append(_session("BAD", internal_status="REJECT", internal_quality_score=0.1))
    edges = [
        _candidate("A", "B", 80),
        _candidate("B", "C", 70),
        _candidate("A", "C", 60),
        _candidate("C", "D", 40),
    ]

    plan = propose_prebuild_set(sessions, edges, cfg)

    proposed = plan["proposed_base_sessions"]
    validation = plan["validation_candidates"]
    assert 2 <= len(proposed) <= 3
    assert len(validation) == 1
    assert set(proposed).isdisjoint(validation)
    assert "BAD" not in proposed
    assert plan["requires_geometric_verification"] is True
    assert plan["proposal_confidence"].endswith("PROPOSAL_ONLY")
    assert plan["verification_pairs"]
    assert all(row["requires_geometric_verification"] for row in plan["verification_pairs"])


def test_no_vpr_graph_still_proposes_small_geometry_probe_set() -> None:
    cfg = load_config()
    sessions = [_session(name) for name in ("A", "B", "C", "D", "E")]
    plan = propose_prebuild_set(sessions, [], cfg)

    assert 2 <= len(plan["proposed_base_sessions"]) <= cfg["prebuild"]["max_no_graph_sessions"]
    assert plan["proposal_confidence"] == "LOW"
    assert all(row["forced_probe"] for row in plan["verification_pairs"])


def test_objective_normalizes_grid_cell_count_and_ignores_timestamp_as_view_proxy() -> None:
    cfg = load_config()
    first = _session(
        "A",
        convex_hull_coverage=0.10,
        grid_occupancy_4x4=8.0,
        fim_logdet=5.0,
        num_tracks=1000,
        num_observations=5000,
    )
    second = _session(
        "B",
        timestamp="2035-12-31T23:59:59Z",
        convex_hull_coverage=0.10,
        grid_occupancy_4x4=8.0,
        fim_logdet=5.0,
        num_tracks=1000,
        num_observations=5000,
    )
    terms = compute_objective_terms([first, second], [], ["A", "B"], cfg)
    assert terms["coverage"] == pytest.approx(0.5)
    assert terms["view_diversity"] == pytest.approx(0.0)

    same_time = replace(second, timestamp=first.timestamp)
    same_terms = compute_objective_terms([first, same_time], [], ["A", "B"], cfg)
    assert same_terms["view_diversity"] == pytest.approx(terms["view_diversity"])
    assert same_terms["utility"] == pytest.approx(terms["utility"])
