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
from sfm_qa.session_select import critical_bridges


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


def test_relative_admission_keeps_best_probe_when_every_video_is_weak() -> None:
    cfg = load_config()
    cfg["prebuild"]["min_video_score"] = 0.99
    sessions = [
        _session(
            "LEAST_BAD",
            internal_status="WEAK",
            internal_quality_score=0.25,
            sharpness_p10=18.0,
            parallax_ratio=0.12,
            unproven_ratio=0.60,
        ),
        _session(
            "WEAKER",
            internal_status="REJECT",
            internal_quality_score=0.10,
            sharpness_p10=8.0,
            parallax_ratio=0.03,
            unproven_ratio=0.85,
        ),
        _session(
            "WEAKEST",
            internal_status="REJECT",
            internal_quality_score=0.05,
            sharpness_p10=3.0,
            parallax_ratio=0.01,
            unproven_ratio=0.95,
        ),
    ]

    plan = propose_prebuild_set(sessions, [], cfg)

    assert plan["proposed_base_sessions"]
    assert "LEAST_BAD" in plan["proposed_base_sessions"]
    assert plan["selection_mode"] == "RELATIVE_PORTFOLIO"
    assert plan["relative_fallback_used"] is True
    assert plan["best_available_not_release"] is True
    assert plan["requires_geometric_verification"] is True


def test_relative_portfolio_prefers_complementary_motion_over_duplicate() -> None:
    cfg = load_config()
    cfg["prebuild"]["min_base_sessions"] = 2
    cfg["prebuild"]["max_sessions"] = 2
    anchor = _session("ANCHOR", internal_quality_score=0.95)
    duplicate = _session("DUPLICATE", internal_quality_score=0.80)
    complement = _session(
        "COMPLEMENT",
        internal_quality_score=0.80,
        hover_ratio=0.13,
        pure_rotation_ratio=0.01,
        fast_motion_ratio=0.01,
    )
    edges = [
        _candidate("ANCHOR", "DUPLICATE", 80),
        _candidate("ANCHOR", "COMPLEMENT", 80),
    ]

    plan = propose_prebuild_set([anchor, duplicate, complement], edges, cfg)

    assert plan["proposed_base_sessions"] == ["ANCHOR", "COMPLEMENT"]
    step = plan["ranked_steps"][1]
    assert step["terms"]["motion_diversity"] > 0.0
    assert step["terms"]["redundancy"] < 1.0


def test_missing_measurements_are_not_counted_as_positive_evidence() -> None:
    cfg = load_config()
    missing = _session(
        "MISSING",
        internal_status="WEAK",
        internal_quality_score=0.2,
        sharpness_p10=None,
        parallax_ratio=None,
        low_parallax_ratio=None,
        unproven_ratio=None,
        near_duplicate_ratio=None,
        underexposed_ratio=None,
        overexposed_ratio=None,
        epipolar_outlier_ratio_median=None,
    )
    measured = _session(
        "MEASURED",
        internal_status="WEAK",
        internal_quality_score=0.2,
        sharpness_p10=40.0,
        parallax_ratio=0.2,
        low_parallax_ratio=0.1,
        unproven_ratio=0.5,
        near_duplicate_ratio=0.2,
        underexposed_ratio=0.1,
        overexposed_ratio=0.1,
        epipolar_outlier_ratio_median=0.4,
    )

    plan = propose_prebuild_set([missing, measured], [], cfg)

    scores = plan["session_scores"]
    assert scores["MISSING"]["evidence_completeness"] < scores["MEASURED"][
        "evidence_completeness"
    ]
    assert scores["MISSING"]["portfolio_score"] < scores["MEASURED"]["portfolio_score"]


def test_unreadable_video_is_not_ranked_or_reserved_for_validation() -> None:
    cfg = load_config()
    readable = [_session("A"), _session("B")]
    unreadable = _session(
        "BROKEN",
        internal_status="STRONG",
        internal_quality_score=1.0,
        reasons=("unreadable_video",),
    )

    plan = propose_prebuild_set(
        [*readable, unreadable],
        [_candidate("A", "BROKEN", 500)],
        cfg,
    )

    assert "BROKEN" not in plan["proposed_base_sessions"]
    assert "BROKEN" not in plan["validation_candidates"]
    assert plan["rejected"]["BROKEN"] == "unrecoverable_input=unreadable_video"
    assert plan["proposal_graph_available"] is False
    assert plan["session_scores"]["BROKEN"]["relative_quality_rank"] is None


def test_legacy_absolute_mode_does_not_reuse_rejected_video_as_validation() -> None:
    cfg = load_config()
    cfg["prebuild"]["relative_admission"] = False
    good = _session("GOOD")
    bad = _session("BAD", internal_status="REJECT", internal_quality_score=0.1)

    plan = propose_prebuild_set([good, bad], [], cfg)

    assert plan["proposed_base_sessions"] == ["GOOD"]
    assert plan["validation_candidates"] == []
    assert "BAD" in plan["rejected"]


def test_partial_exposure_measurement_reduces_evidence_completeness() -> None:
    cfg = load_config()
    complete = _session("COMPLETE")
    partial = _session("PARTIAL", overexposed_ratio=None)

    scores = propose_prebuild_set([complete, partial], [], cfg)["session_scores"]

    assert scores["PARTIAL"]["evidence_completeness"] < scores["COMPLETE"][
        "evidence_completeness"
    ]


def test_zero_relative_marginal_ratio_disables_ratio_stop() -> None:
    cfg = load_config()
    cfg["prebuild"]["relative_marginal_keep_ratio"] = 0.0
    cfg["prebuild"]["max_no_graph_sessions"] = 4
    cfg["prebuild"]["validation_candidates"] = 0
    sessions = [_session(name) for name in ("A", "B", "C", "D")]

    plan = propose_prebuild_set(sessions, [], cfg)

    assert plan["proposed_base_sessions"] == ["D", "C", "B", "A"]
    evidence = plan["stopping_evidence"]
    assert evidence["stop_reason"] == "candidates_exhausted"
    assert evidence["stop_reason"] != "relative_marginal_collapse"
    assert evidence["requires_geometric_verification"] is True
    assert evidence["hard_status"] == "VALID"
    assert evidence["evidence_status"] == "PASS"
    assert "margin" not in evidence
    assert plan["requires_geometric_verification"] is True


def _assert_reporting_stop(evidence: dict) -> None:
    assert evidence["requires_geometric_verification"] is True
    assert evidence["hard_status"] in {"VALID", "HARD_FAIL"}
    assert evidence["evidence_status"] in {
        "PASS",
        "WARN",
        "INSUFFICIENT_EVIDENCE",
        "QUALITY_SHORTFALL",
    }
    assert evidence["authority"] == "reporting_review_only"
    assert evidence["grants_selection_or_merge_authority"] is False
    assert evidence["independence_assumptions"]
    assert evidence["provenance_assumptions"]


def test_empty_input_stopping_evidence() -> None:
    plan = propose_prebuild_set([], [], load_config())

    assert plan["proposed_base_sessions"] == []
    assert plan["validation_candidates"] == []
    assert plan["verification_pairs"] == []
    evidence = plan["stopping_evidence"]
    assert evidence["stop_reason"] == "empty_input"
    assert evidence["hard_status"] == "HARD_FAIL"
    assert evidence["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    _assert_reporting_stop(evidence)
    assert plan["requires_geometric_verification"] is True


def test_no_readable_input_stopping_evidence() -> None:
    sessions = [
        _session("BROKEN", reasons=("unreadable_video",)),
        _session("GONE", reasons=("missing_video",)),
    ]

    plan = propose_prebuild_set(sessions, [_candidate("BROKEN", "GONE", 40)], load_config())

    assert plan["proposed_base_sessions"] == []
    assert plan["rejected"]["BROKEN"] == "unrecoverable_input=unreadable_video"
    assert plan["rejected"]["GONE"] == "unrecoverable_input=missing_video"
    evidence = plan["stopping_evidence"]
    assert evidence["stop_reason"] == "no_readable_input"
    assert evidence["hard_status"] == "HARD_FAIL"
    assert evidence["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    _assert_reporting_stop(evidence)
    assert plan["requires_geometric_verification"] is True


def test_no_legacy_eligible_input_stopping_evidence() -> None:
    cfg = load_config()
    cfg["prebuild"]["relative_admission"] = False
    bad = _session("BAD", internal_status="REJECT", internal_quality_score=0.1)

    plan = propose_prebuild_set([bad], [], cfg)

    assert plan["proposed_base_sessions"] == []
    assert plan["validation_candidates"] == []
    assert "BAD" in plan["rejected"]
    evidence = plan["stopping_evidence"]
    assert evidence["stop_reason"] == "no_legacy_eligible_input"
    assert evidence["hard_status"] == "HARD_FAIL"
    assert evidence["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    _assert_reporting_stop(evidence)
    assert plan["requires_geometric_verification"] is True


def test_relative_marginal_collapse_records_signed_margin_without_changing_selection() -> None:
    cfg = load_config()
    cfg["prebuild"]["validation_candidates"] = 0
    cfg["prebuild"]["max_sessions"] = 4
    cfg["prebuild"]["relative_marginal_keep_ratio"] = 0.80
    anchor = _session("ANCHOR", internal_quality_score=0.95)
    bridge = _session("BRIDGE", internal_quality_score=0.90)
    extra_one = _session("EXTRA1", internal_quality_score=0.90)
    extra_two = _session("EXTRA2", internal_quality_score=0.90)
    edges = [_candidate("ANCHOR", "BRIDGE", 200)]

    plan = propose_prebuild_set(
        [anchor, bridge, extra_one, extra_two],
        edges,
        cfg,
    )

    assert plan["proposed_base_sessions"] == ["ANCHOR", "BRIDGE"]
    assert [step["session_id"] for step in plan["ranked_steps"]] == ["ANCHOR", "BRIDGE"]
    evidence = plan["stopping_evidence"]
    assert evidence["stop_reason"] == "relative_marginal_collapse"
    assert evidence["next_candidate_id"] in {"EXTRA1", "EXTRA2"}
    assert evidence["keep_ratio"] == pytest.approx(0.80)
    assert evidence["keep_floor"] == pytest.approx(
        evidence["best_previous"] * evidence["keep_ratio"]
    )
    assert evidence["margin"] == pytest.approx(
        evidence["marginal"] - evidence["keep_floor"]
    )
    assert evidence["margin"] < 0.0
    assert evidence["hard_status"] == "VALID"
    assert evidence["evidence_status"] == "QUALITY_SHORTFALL"
    _assert_reporting_stop(evidence)
    assert plan["requires_geometric_verification"] is True
    pair = plan["verification_pairs"][0]
    assert pair["evidence_type"] == "retrieval_candidate"
    assert pair["retrieval_triangle_priority"] is False
    assert pair["count_field_provenance"] == "num_candidate_pairs"
    assert pair["reason"] == "retrieval_candidate_requires_geometry"


def test_complete_triangle_is_the_only_triangle_priority_evidence() -> None:
    cfg = load_config()
    cfg["prebuild"]["min_base_sessions"] = 3
    cfg["prebuild"]["max_sessions"] = 3
    cfg["prebuild"]["validation_candidates"] = 0
    sessions = [_session(name) for name in ("A", "B", "C")]
    edges = [
        _candidate("A", "B", 80),
        _candidate("B", "C", 70),
        _candidate("A", "C", 60),
    ]

    plan = propose_prebuild_set(sessions, edges, cfg)

    assert set(plan["proposed_base_sessions"]) == {"A", "B", "C"}
    assert plan["verification_pairs"]
    for pair in plan["verification_pairs"]:
        assert pair["evidence_type"] == "retrieval_triangle"
        assert pair["retrieval_triangle_priority"] is True
        assert pair["count_field_provenance"] == "num_candidate_pairs"
        assert pair["reason"] == "retrieval_candidate_requires_geometry"
        assert pair["requires_geometric_verification"] is True



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


def test_objective_computes_laplacian_spectrum_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = critical_bridges.np.linalg.eigvalsh

    def counting_eigvalsh(matrix):
        nonlocal calls
        calls += 1
        return original(matrix)

    monkeypatch.setattr(critical_bridges.np.linalg, "eigvalsh", counting_eigvalsh)

    compute_objective_terms(
        [_session("A"), _session("B")],
        [],
        ["A", "B"],
        load_config(),
    )

    assert calls == 1
