"""Contract tests for sfm_qa.session_select. Production lives elsewhere."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pytest

from sfm_qa.session_select import (
    ROLES,
    classify_leftover_vs_frozen_base,
    classify_remainder,
    classify_session_edge,
    efficiency_coverage,
    greedy_select_core,
    load_config,
    lookup,
    rotation_cycle_error_deg,
    seed_session,
    tag_suspicious_edges,
)
from sfm_qa.session_select import SessionEdgeQuality, SessionQuality
from sfm_qa.session_select.select_core import connecting_edges
from sfm_qa.session_select.export import (
    SESSION_ROLE_COLUMNS,
    build_role_rows,
    write_session_roles_csv,
)



ALLOWED_ROLES = frozenset(
    {
        "BASE_CORE",
        "BASE_SUPPORT",
        "APPEARANCE_REF",
        "GEOMETRY_REINFORCEMENT",
        "UPDATE_CANDIDATE",
        "NEW_SUBMAP",
        "QUARANTINE",
        "REJECT",
        "VALIDATION_ONLY",
    }
)


def _rodrigues(axis: Iterable[float], deg: float) -> np.ndarray:
    vec = np.asarray(list(axis), dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise ValueError("axis must be nonzero")
    axis_u = vec / norm
    theta = np.deg2rad(float(deg))
    kx, ky, kz = axis_u
    k = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def _relative_R(r_i: np.ndarray, r_j: np.ndarray) -> np.ndarray:
    return np.asarray(r_i, dtype=np.float64).T @ np.asarray(r_j, dtype=np.float64)


def _pair_key(left: str, right: str) -> frozenset[str]:
    return frozenset((str(left), str(right)))


def _record_endpoints(record: Any) -> frozenset[str] | None:
    if isinstance(record, Mapping):
        a = record.get("session_a", record.get("a"))
        b = record.get("session_b", record.get("b"))
        if a is not None and b is not None:
            return _pair_key(str(a), str(b))
        pair = record.get("edge") or record.get("pair")
        if isinstance(pair, (tuple, list)) and len(pair) >= 2:
            return _pair_key(str(pair[0]), str(pair[1]))
        return None
    a = getattr(record, "session_a", getattr(record, "a", None))
    b = getattr(record, "session_b", getattr(record, "b", None))
    if a is not None and b is not None:
        return _pair_key(str(a), str(b))
    return None


def _record_status(record: Any) -> str | None:
    if record is None:
        return None
    if isinstance(record, str):
        return record
    if isinstance(record, Mapping):
        for key in ("status", "tag", "label", "edge_status"):
            val = record.get(key)
            if val is not None:
                return str(val)
        return None
    for key in ("status", "tag", "label", "edge_status"):
        val = getattr(record, key, None)
        if val is not None:
            return str(val)
    return None


def _is_suspicious(record: Any) -> bool:
    status = _record_status(record)
    if status is None:
        return False
    token = status.upper()
    return token == "SUSPICIOUS_EDGE" or "SUSPICIOUS" in token


def _make_session(**overrides: Any) -> SessionQuality:
    fields = dict(
        session_id="S",
        timestamp=None,
        num_frames=200,
        num_keyframes=80,
        registered_ratio=0.96,
        sharpness_median=120.0,
        sharpness_p10=40.0,
        num_tracks=20_000,
        num_observations=120_000,
        median_track_length=4.0,
        long_track_ratio=0.35,
        reprojection_rmse=1.2,
        reprojection_p90=2.2,
        parallax_median_deg=4.0,
        parallax_p10_deg=2.0,
        positive_depth_ratio=0.99,
        convex_hull_coverage=0.45,
        grid_occupancy_4x4=0.50,
        fim_condition_number=80.0,
        fim_logdet=12.0,
        rotation_cycle_error=0.2,
        translation_consistency=0.1,
        connected_components=1,
        average_degree=4.0,
        num_bridges=0,
        num_articulation_points=0,
        fiedler_value=0.8,
        internal_quality_score=0.80,
        internal_status="STRONG",
        reasons=(),
    )
    fields.update(overrides)
    return SessionQuality(**fields)


def _make_edge(**overrides: Any) -> SessionEdgeQuality:
    fields = dict(
        session_a="A",
        session_b="B",
        num_candidate_pairs=40,
        num_verified_pairs=20,
        num_cross_session_tracks=800,
        num_cross_session_observations=4_000,
        independent_bridge_groups=3,
        inlier_count=200,
        inlier_ratio=0.7,
        rotation_consensus_deg=0.4,
        translation_direction_consensus_deg=1.0,
        scale_consensus=0.05,
        cross_session_reprojection_error=1.8,
        spatial_coverage=0.4,
        cycle_support=2,
        cycle_error=0.3,
        edge_quality_score=0.85,
        is_bridge=False,
        is_critical_bridge=False,
        status="STRONG",
        reasons=(),
    )
    fields.update(overrides)
    return SessionEdgeQuality(**fields)


def _as_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [str(key) for key, val in value.items() if val]
    return [str(item) for item in value]


def _greedy_groups(result: Any) -> dict[str, list[str]]:
    if not isinstance(result, Mapping):
        raise TypeError(f"greedy_select_core must return a mapping, got {type(result)!r}")
    selected = _as_id_list(result.get("selected"))
    core = _as_id_list(result.get("core"))
    support = _as_id_list(result.get("support"))
    if not selected:
        selected = list(dict.fromkeys([*core, *support]))
    return {"core": core, "support": support, "selected": selected}


def _roles_from_classify(result: Any) -> dict[str, str]:
    if not isinstance(result, Mapping):
        raise TypeError(f"classify_remainder must return a mapping, got {type(result)!r}")
    out: dict[str, str] = {}
    for key, val in result.items():
        role = val.role if hasattr(val, "role") else val
        out[str(key)] = str(role)
    return out


def test_identity_cycle_error_is_near_zero():
    r_ab = _rodrigues((0.0, 0.0, 1.0), 30.0)
    r_bc = _rodrigues((1.0, 0.0, 0.0), 40.0)
    r_ca = (r_ab @ r_bc).T
    err = float(rotation_cycle_error_deg(r_ab, r_bc, r_ca))
    assert err == pytest.approx(0.0, abs=1e-6)


def test_extra_rotation_cycle_error_is_twenty_degrees():
    r_ab = _rodrigues((0.0, 0.0, 1.0), 25.0)
    r_bc = _rodrigues((0.0, 1.0, 0.0), -15.0)
    r_ca = (r_ab @ r_bc).T
    r_ca_broken = _rodrigues((1.0, 0.0, 0.0), 20.0) @ r_ca
    err = float(rotation_cycle_error_deg(r_ab, r_bc, r_ca_broken))
    assert err == pytest.approx(20.0, abs=1e-4)


def test_tag_suspicious_edges_marks_high_error_edge():
    ra = _rodrigues((1.0, 0.0, 0.0), 10.0)
    rb = _rodrigues((0.0, 1.0, 0.0), 20.0)
    rc = _rodrigues((0.0, 0.0, 1.0), 30.0)
    rd = _rodrigues((1.0, 1.0, 0.0), 15.0)
    r_ab_true = _relative_R(ra, rb)
    r_ab_bad = _rodrigues((0.0, 1.0, 0.0), 20.0) @ r_ab_true
    poses = [
        ("A", "B", r_ab_bad),
        ("B", "C", _relative_R(rb, rc)),
        ("C", "A", _relative_R(rc, ra)),
        ("B", "D", _relative_R(rb, rd)),
        ("D", "A", _relative_R(rd, ra)),
        ("C", "D", _relative_R(rc, rd)),
    ]
    tagged = tag_suspicious_edges(poses, 5.0)
    assert tagged, "expected at least one tagged edge record"

    by_pair: dict[frozenset[str], Any] = {}
    for rec in tagged:
        ends = _record_endpoints(rec)
        if ends is not None and (_is_suspicious(rec) or ends not in by_pair):
            by_pair[ends] = rec

    ab = by_pair.get(_pair_key("A", "B"))
    assert ab is not None, "high-error A-B must appear in tagged edges"
    assert _is_suspicious(ab), "high-error A-B must be SUSPICIOUS_EDGE"

    cd = by_pair.get(_pair_key("C", "D"))
    if cd is not None:
        assert not _is_suspicious(cd), "consistent C-D must not be SUSPICIOUS_EDGE"


def test_seed_is_strong_high_coverage_not_earlier_timestamp():
    early_weak = _make_session(
        session_id="EARLY_WEAK",
        timestamp="2020-01-01T00:00:00+00:00",
        internal_status="WEAK",
        internal_quality_score=0.18,
        convex_hull_coverage=0.07,
        grid_occupancy_4x4=0.08,
        registered_ratio=0.58,
        fim_logdet=3.0,
        num_observations=20_000,
    )
    mid_strong = _make_session(
        session_id="MID_STRONG",
        timestamp="2022-06-15T12:00:00+00:00",
        internal_status="STRONG",
        internal_quality_score=0.93,
        convex_hull_coverage=0.72,
        grid_occupancy_4x4=0.70,
        registered_ratio=0.98,
        fim_logdet=14.0,
        parallax_p10_deg=3.2,
        positive_depth_ratio=0.995,
        num_observations=180_000,
    )
    late_weak = _make_session(
        session_id="LATE_WEAK",
        timestamp="2024-12-31T23:59:00+00:00",
        internal_status="WEAK",
        internal_quality_score=0.22,
        convex_hull_coverage=0.10,
        grid_occupancy_4x4=0.11,
        registered_ratio=0.60,
        fim_logdet=4.0,
        num_observations=25_000,
    )
    cfg = load_config()
    seed = seed_session([early_weak, mid_strong, late_weak], cfg)
    assert seed == "MID_STRONG"
    assert seed_session([early_weak, mid_strong, late_weak], cfg, exclude=("MID_STRONG",)) is None

    late_usable = _make_session(
        session_id="LATE_USABLE",
        timestamp="2025-01-01T00:00:00+00:00",
        internal_status="USABLE",
        internal_quality_score=0.70,
        convex_hull_coverage=0.40,
        grid_occupancy_4x4=0.38,
        registered_ratio=0.90,
        fim_logdet=8.0,
        num_observations=90_000,
    )
    assert (
        seed_session(
            [early_weak, mid_strong, late_weak, late_usable],
            cfg,
            exclude=("MID_STRONG",),
        )
        == "LATE_USABLE"
    )


def test_mapped_weak_sessions_use_relative_fallback_with_verified_geometry():
    first = _make_session(
        session_id="WEAK_A",
        internal_status="WEAK",
        internal_quality_score=0.42,
        registered_ratio=0.72,
        convex_hull_coverage=0.35,
    )
    second = _make_session(
        session_id="WEAK_B",
        internal_status="WEAK",
        internal_quality_score=0.38,
        registered_ratio=0.68,
        convex_hull_coverage=0.55,
    )
    edge = _make_edge(
        session_a="WEAK_A",
        session_b="WEAK_B",
        status="USABLE",
        independent_bridge_groups=2,
        num_verified_pairs=20,
    )

    result = greedy_select_core([first, second], [edge], load_config())

    assert result["selected"]
    assert result["seed"] == "WEAK_A"
    assert result["selection_mode"] == "RELATIVE_WEAK_FALLBACK"
    assert result["relative_fallback_used"] is True
    assert result["best_available_not_release"] is True


def test_mapped_weak_fallback_does_not_cross_zero_geometry_edge():
    sessions = [
        _make_session(
            session_id=sid,
            internal_status="WEAK",
            internal_quality_score=score,
        )
        for sid, score in (("WEAK_A", 0.42), ("WEAK_B", 0.38))
    ]
    empty_edge = _make_edge(
        session_a="WEAK_A",
        session_b="WEAK_B",
        status="WEAK",
        num_candidate_pairs=0,
        num_verified_pairs=0,
        num_cross_session_tracks=0,
        independent_bridge_groups=0,
        inlier_count=0,
    )

    result = greedy_select_core(sessions, [empty_edge], load_config())

    assert result["selected"] == ["WEAK_A"]


def test_zero_valued_map_metrics_do_not_create_a_weak_fallback_seed():
    zero = _make_session(
        session_id="ZERO",
        internal_status="WEAK",
        internal_quality_score=0.2,
        registered_ratio=0.0,
        num_tracks=0,
        num_observations=0,
        positive_depth_ratio=0.0,
        convex_hull_coverage=0.0,
        grid_occupancy_4x4=0.0,
        parallax_median_deg=0.0,
        fim_condition_number=0.0,
        fim_logdet=None,
        reprojection_rmse=None,
        reprojection_p90=None,
    )

    result = greedy_select_core([zero], [], load_config())

    assert result["selected"] == []
    assert result["selection_mode"] == "NO_MAPPED_CANDIDATE"


def test_finite_fim_is_enough_to_rank_a_mapped_weak_fallback() -> None:
    mapped = _make_session(
        session_id="FIM_ONLY",
        internal_status="WEAK",
        registered_ratio=None,
        num_tracks=None,
        num_observations=None,
        positive_depth_ratio=None,
        convex_hull_coverage=None,
        grid_occupancy_4x4=None,
        parallax_median_deg=None,
        fim_condition_number=None,
        fim_logdet=-3.0,
        reprojection_rmse=None,
        reprojection_p90=None,
    )

    result = greedy_select_core([mapped], [], load_config())

    assert result["selected"] == ["FIM_ONLY"]
    assert result["selection_mode"] == "RELATIVE_WEAK_FALLBACK"


def test_cycle_closer_cannot_use_zero_geometry_usable_edges():
    sessions = [_make_session(session_id=name) for name in ("A", "B", "C")]
    genuine = _make_edge(session_a="A", session_b="B", status="STRONG")
    empty_edges = [
        _make_edge(
            session_a=left,
            session_b="C",
            status="USABLE",
            num_candidate_pairs=0,
            num_verified_pairs=0,
            num_cross_session_tracks=0,
            independent_bridge_groups=0,
            inlier_count=0,
        )
        for left in ("A", "B")
    ]

    result = greedy_select_core(sessions, [genuine, *empty_edges], load_config())

    assert set(result["selected"]) == {"A", "B"}


@pytest.mark.parametrize(
    ("status", "groups"),
    [("UNKNOWN", 2), ("USABLE", None)],
)
def test_unknown_status_or_missing_bridge_count_fails_closed(status, groups):
    sessions = [_make_session(session_id=name) for name in ("A", "B")]
    edge = _make_edge(
        session_a="A",
        session_b="B",
        status=status,
        independent_bridge_groups=groups,
    )

    result = greedy_select_core(sessions, [edge], load_config())

    assert len(result["selected"]) == 1


def test_two_components_one_critical_ambiguous_edge_not_both_base_core():
    left = _make_session(
        session_id="L1",
        timestamp="2021-01-01T00:00:00+00:00",
        internal_quality_score=0.91,
        convex_hull_coverage=0.55,
        grid_occupancy_4x4=0.55,
    )
    left2 = _make_session(
        session_id="L2",
        timestamp="2021-01-02T00:00:00+00:00",
        internal_quality_score=0.88,
        convex_hull_coverage=0.52,
        grid_occupancy_4x4=0.50,
    )
    right = _make_session(
        session_id="R1",
        timestamp="2023-08-01T00:00:00+00:00",
        internal_quality_score=0.90,
        convex_hull_coverage=0.54,
        grid_occupancy_4x4=0.53,
    )
    right2 = _make_session(
        session_id="R2",
        timestamp="2023-08-02T00:00:00+00:00",
        internal_quality_score=0.87,
        convex_hull_coverage=0.50,
        grid_occupancy_4x4=0.49,
    )
    edges = [
        _make_edge(session_a="L1", session_b="L2", status="STRONG"),
        _make_edge(session_a="R1", session_b="R2", status="STRONG"),
        _make_edge(
            session_a="L2",
            session_b="R1",
            status="AMBIGUOUS",
            is_bridge=True,
            is_critical_bridge=True,
            independent_bridge_groups=1,
            num_verified_pairs=2,
            edge_quality_score=0.25,
            cycle_support=0,
        ),
    ]
    cfg = load_config()
    picked = _greedy_groups(greedy_select_core([left, left2, right, right2], edges, cfg))
    core = set(picked["core"]) or set(picked["selected"])
    selected = set(picked["selected"])
    left_ids, right_ids = {"L1", "L2"}, {"R1", "R2"}
    assert not (core & left_ids and core & right_ids), (
        "FAIL_CLOSED: both sides of a single CRITICAL/AMBIGUOUS bridge "
        "must not be BASE_CORE"
    )
    assert not (selected & left_ids and selected & right_ids), (
        "FAIL_CLOSED: do not force-merge two components across one "
        "CRITICAL/AMBIGUOUS edge"
    )

    roles = _roles_from_classify(
        classify_remainder(
            [left, left2, right, right2],
            edges,
            picked["core"],
            picked["support"],
            cfg,
        )
    )
    core_roles = {sid for sid, role in roles.items() if role == "BASE_CORE"}
    assert not (core_roles & left_ids and core_roles & right_ids)


def test_vpr_only_zero_verified_zero_bridges_not_strong():
    edge = classify_session_edge(
        "QUERY",
        "BASE",
        num_verified_pairs=0,
        independent_bridge_groups=0,
        num_candidate_pairs=275,
        num_cross_session_tracks=0,
    )
    status = edge.status if hasattr(edge, "status") else _record_status(edge)
    assert status is not None
    assert status != "STRONG"
    assert status != "USABLE"
    assert str(status).upper() not in {"STRONG", "USABLE"}

    isolated = _make_session(
        session_id="QUERY",
        timestamp="2024-01-01T00:00:00+00:00",
        internal_status="STRONG",
        internal_quality_score=0.86,
        convex_hull_coverage=0.60,
    )
    base = _make_session(
        session_id="BASE",
        timestamp="2021-01-01T00:00:00+00:00",
        internal_status="STRONG",
        internal_quality_score=0.92,
        convex_hull_coverage=0.65,
    )
    vpr_edge = _make_edge(
        session_a="QUERY",
        session_b="BASE",
        num_candidate_pairs=275,
        num_verified_pairs=0,
        independent_bridge_groups=0,
        num_cross_session_tracks=0,
        num_cross_session_observations=0,
        inlier_count=0,
        inlier_ratio=0.0,
        cycle_support=0,
        is_bridge=False,
        is_critical_bridge=False,
        status=str(status),
        edge_quality_score=0.05,
        reasons=("VPR_ONLY",),
    )
    cfg = load_config()
    picked = _greedy_groups(greedy_select_core([base, isolated], [vpr_edge], cfg))
    core = set(picked["core"]) or set(picked["selected"])
    assert not ({"QUERY", "BASE"} <= core), "VPR-only must not fuse both sessions into BASE_CORE"


def test_efficiency_prefers_coverage_gain_over_observation_cost():
    high = efficiency_coverage(0.18, 5e5)
    low = efficiency_coverage(0.02, 2e6)
    assert high is not None and low is not None
    assert float(high) > float(low)
    assert float(high) == pytest.approx(0.18 / 5e5)
    assert float(low) == pytest.approx(0.02 / 2e6)


def test_later_timestamp_without_change_score_is_not_update_candidate():
    core_s = _make_session(
        session_id="CORE",
        timestamp="2021-01-01T00:00:00+00:00",
        internal_quality_score=0.92,
        convex_hull_coverage=0.70,
    )
    later = _make_session(
        session_id="LATER",
        timestamp="2025-12-31T23:59:00+00:00",
        internal_status="STRONG",
        internal_quality_score=0.85,
        convex_hull_coverage=0.55,
        grid_occupancy_4x4=0.50,
    )
    edge = _make_edge(
        session_a="CORE",
        session_b="LATER",
        status="STRONG",
        independent_bridge_groups=3,
        num_verified_pairs=40,
    )
    cfg = load_config()
    extra = {
        "loc": {"LATER": {"registered_to_base": True, "loc_quality": "strong"}},
        "appearance_shift": {"LATER": True},
    }
    roles = _roles_from_classify(
        classify_remainder([core_s, later], [edge], ["CORE"], [], cfg, extra)
    )
    assert roles["LATER"] != "UPDATE_CANDIDATE"
    assert roles["LATER"] in ALLOWED_ROLES

    leftover = classify_leftover_vs_frozen_base(
        later,
        loc_strong=20,
        loc_queries=20,
        core_hit_fraction=0.9,
        independent_bridges=3,
        usable_edge=True,
        change_score=0.0,
        high_influence=False,
        geometry_already_covered=True,
    )
    leftover_role = leftover.role if hasattr(leftover, "role") else leftover
    assert leftover_role != "UPDATE_CANDIDATE"


def test_load_config_yaml_override_min_coverage_gain(tmp_path: Path):
    path = tmp_path / "override.yaml"
    path.write_text("selection:\n  min_coverage_gain: 0.18\n", encoding="utf-8")
    cfg = load_config(path)
    assert lookup(cfg, "selection.min_coverage_gain") == pytest.approx(0.18)
    if isinstance(cfg, Mapping) and isinstance(cfg.get("selection"), Mapping):
        assert cfg["selection"]["min_coverage_gain"] == pytest.approx(0.18)
    defaults = load_config()
    default_gain = lookup(defaults, "selection.min_coverage_gain")
    assert default_gain is not None
    assert default_gain != pytest.approx(0.18)
    assert lookup(defaults, "selection.min_information_gain") is not None


def test_emitted_roles_are_subset_of_allowed_set():
    assert set(ROLES) == ALLOWED_ROLES
    sessions = [
        _make_session(session_id="CORE", timestamp="2021-01-01T00:00:00+00:00"),
        _make_session(
            session_id="SUPPORT",
            timestamp="2021-01-02T00:00:00+00:00",
            convex_hull_coverage=0.20,
            internal_quality_score=0.75,
        ),
        _make_session(
            session_id="LONE",
            timestamp="2024-01-01T00:00:00+00:00",
            internal_status="STRONG",
        ),
        _make_session(
            session_id="LATE",
            timestamp="2024-06-01T00:00:00+00:00",
            internal_status="STRONG",
        ),
        _make_session(
            session_id="BAD",
            timestamp="2020-01-01T00:00:00+00:00",
            internal_status="REJECT",
            internal_quality_score=0.05,
            registered_ratio=0.2,
        ),
    ]
    edges = [
        _make_edge(session_a="CORE", session_b="SUPPORT", status="USABLE"),
        _make_edge(
            session_a="CORE",
            session_b="LATE",
            status="STRONG",
            independent_bridge_groups=3,
        ),
    ]
    extra = {
        "loc": {"LATE": {"registered_to_base": True, "loc_quality": "strong"}},
        "change_score": {"LATE": 0.0},
        "appearance_shift": {"LATE": True},
    }
    roles = _roles_from_classify(
        classify_remainder(sessions, edges, ["CORE"], ["SUPPORT"], load_config(), extra)
    )
    assert roles
    unknown = set(roles.values()) - ALLOWED_ROLES
    assert not unknown, f"unknown roles: {unknown}"
    assert set(roles.values()) <= ALLOWED_ROLES


def test_sparse_edges_both_directions_preserve_order_and_roles():
    core = _make_session(session_id="CORE", internal_quality_score=0.95)
    support = _make_session(
        session_id="SUPPORT",
        internal_quality_score=0.80,
        convex_hull_coverage=0.20,
    )
    left = _make_session(session_id="LEFT", internal_status="STRONG")
    right = _make_session(session_id="RIGHT", internal_status="STRONG")
    blocked = _make_session(session_id="BLOCKED", internal_status="STRONG")
    noise = _make_session(session_id="NOISE", internal_status="STRONG")
    isolated = _make_session(session_id="ISOLATED", internal_status="STRONG")

    noise_isolated = _make_edge(session_a="NOISE", session_b="ISOLATED", status="STRONG")
    left_to_core = _make_edge(
        session_a="LEFT",
        session_b="CORE",
        status="STRONG",
        independent_bridge_groups=3,
    )
    core_to_right = _make_edge(
        session_a="CORE",
        session_b="RIGHT",
        status="USABLE",
        independent_bridge_groups=3,
    )
    support_core = _make_edge(session_a="SUPPORT", session_b="CORE", status="USABLE")
    left_again = _make_edge(
        session_a="CORE",
        session_b="LEFT",
        status="USABLE",
        independent_bridge_groups=2,
    )
    noise_to_left = _make_edge(session_a="NOISE", session_b="LEFT", status="WEAK")
    blocked_to_core = _make_edge(
        session_a="BLOCKED",
        session_b="CORE",
        status="AMBIGUOUS",
        independent_bridge_groups=1,
    )
    edges = [
        noise_isolated,
        left_to_core,
        core_to_right,
        support_core,
        left_again,
        noise_to_left,
        blocked_to_core,
    ]
    selected = {"CORE", "SUPPORT"}

    assert connecting_edges("LEFT", selected, edges) == [left_to_core, left_again]
    assert connecting_edges("RIGHT", selected, edges) == [core_to_right]
    assert connecting_edges("BLOCKED", selected, edges) == [blocked_to_core]
    assert connecting_edges("ISOLATED", selected, edges) == []
    assert connecting_edges("NOISE", selected, edges) == []

    sessions = [core, support, left, right, blocked, noise, isolated]
    cfg = load_config()
    first = _roles_from_classify(
        classify_remainder(sessions, edges, ["CORE"], ["SUPPORT"], cfg)
    )
    second = _roles_from_classify(
        classify_remainder(sessions, edges, ["CORE"], ["SUPPORT"], cfg)
    )
    assert first == second
    assert first["CORE"] == "BASE_CORE"
    assert first["SUPPORT"] == "BASE_SUPPORT"
    assert first["LEFT"] == "GEOMETRY_REINFORCEMENT"
    assert first["RIGHT"] == "GEOMETRY_REINFORCEMENT"
    assert first["BLOCKED"] == "QUARANTINE"
    assert first["NOISE"] == "NEW_SUBMAP"
    assert first["ISOLATED"] == "NEW_SUBMAP"

    picked = greedy_select_core(sessions, edges, cfg)
    selected_ids = set(_as_id_list(picked.get("selected")))
    assert picked["seed"] == "CORE"
    assert selected_ids.isdisjoint({"NOISE", "ISOLATED", "BLOCKED"})


def test_build_role_rows_role_only_base_is_not_global_ba():
    rows = build_role_rows(
        [_make_session(session_id="CORE")],
        [],
        {"CORE": "BASE_CORE"},
        {},
    )
    assert len(rows) == 1
    assert rows[0]["role"] == "BASE_CORE"
    assert rows[0]["fusion_authorization"] == "GLOBAL_BA_PENDING_APPROVAL"
    assert rows[0]["authorized_edge"] == ""
    assert rows[0]["geometry_authority"] == []
    support = build_role_rows(
        [_make_session(session_id="SUP")],
        [],
        {"SUP": "BASE_SUPPORT"},
        {},
    )
    assert support[0]["fusion_authorization"] == "LOCAL_RELATION_ONLY"


def test_build_role_rows_split_evidence_does_not_compose_authority():
    sessions = [
        _make_session(session_id="CORE"),
        _make_session(session_id="B"),
        _make_session(session_id="C"),
    ]
    edges = [
        _make_edge(
            session_a="CORE",
            session_b="B",
            independent_bridge_groups=2,
            independent_artifact=True,
            evidence_scope="exact_pair",
            geometry_complete=False,
            group_holdout_disjoint=True,
            fit_evidence_ids=("f1", "f2"),
            holdout_evidence_ids=("h1", "h2"),
            status="WEAK",
        ),
        _make_edge(
            session_a="CORE",
            session_b="C",
            independent_bridge_groups=0,
            independent_artifact=True,
            evidence_scope="exact_pair",
            geometry_complete=True,
            group_holdout_disjoint=False,
            fit_evidence_ids=("f3",),
            holdout_evidence_ids=(),
            status="WEAK",
        ),
    ]
    rows = build_role_rows(
        sessions,
        edges,
        {
            "CORE": "BASE_CORE",
            "B": "GEOMETRY_REINFORCEMENT",
            "C": "GEOMETRY_REINFORCEMENT",
        },
        {},
    )
    by_id = {row["session_id"]: row for row in rows}
    assert by_id["CORE"]["fusion_authorization"] == "GLOBAL_BA_PENDING_APPROVAL"
    assert by_id["B"]["fusion_authorization"] == "LOCAL_RELATION_ONLY"
    assert by_id["C"]["fusion_authorization"] == "LOCAL_RELATION_ONLY"
    assert by_id["CORE"]["authorized_edge"] == ""
    assert all(not item["authorized"] for item in by_id["CORE"]["geometry_authority"])
    assert by_id["CORE"]["num_independent_bridge_groups"] == 2


def test_build_role_rows_authorized_edge_grants_global_ba():
    edge = _make_edge(
        session_a="CORE",
        session_b="SUP",
        independent_artifact=True,
        evidence_scope="exact_pair",
        geometry_complete=True,
        group_holdout_disjoint=True,
        independent_bridge_groups=2,
        fit_evidence_ids=("f1", "f2"),
        holdout_evidence_ids=("h1", "h2"),
        status="STRONG",
    )
    rows = build_role_rows(
        [_make_session(session_id="CORE"), _make_session(session_id="SUP")],
        [edge],
        {"CORE": "BASE_CORE", "SUP": "BASE_SUPPORT"},
        {},
    )
    by_id = {row["session_id"]: row for row in rows}
    assert by_id["CORE"]["fusion_authorization"] == "GLOBAL_BA"
    assert by_id["SUP"]["fusion_authorization"] == "GLOBAL_BA"
    assert by_id["CORE"]["authorized_edge"] == "CORE-SUP"
    assert any(item["authorized"] for item in by_id["CORE"]["geometry_authority"])
    assert any(item["hard_status"] == "VALID" for item in by_id["CORE"]["geometry_authority"])
    reinforce = build_role_rows(
        [_make_session(session_id="SIDE")],
        [
            _make_edge(
                session_a="SIDE",
                session_b="CORE",
                independent_artifact=True,
                evidence_scope="exact_pair",
                geometry_complete=True,
                group_holdout_disjoint=True,
                independent_bridge_groups=2,
                fit_evidence_ids=("f1", "f2"),
                holdout_evidence_ids=("h1", "h2"),
                status="STRONG",
            )
        ],
        {"SIDE": "GEOMETRY_REINFORCEMENT"},
        {},
    )
    assert reinforce[0]["fusion_authorization"] == "LOCAL_FUSION"


def test_build_role_rows_preserves_non_base_fusion_without_authorized_edge():
    rows = build_role_rows(
        [
            _make_session(session_id="U"),
            _make_session(session_id="A"),
            _make_session(session_id="N"),
            _make_session(session_id="V"),
        ],
        [],
        {
            "U": "UPDATE_CANDIDATE",
            "A": "APPEARANCE_REF",
            "N": "NEW_SUBMAP",
            "V": "VALIDATION_ONLY",
        },
        {},
    )
    by_id = {row["session_id"]: row for row in rows}
    assert by_id["U"]["fusion_authorization"] == "LOCAL_FUSION_PENDING_LOO"
    assert by_id["A"]["fusion_authorization"] == "LOCALIZATION_ONLY"
    assert by_id["N"]["fusion_authorization"] == "SUBMAP_ONLY"
    assert by_id["V"]["fusion_authorization"] == "EVALUATION_ONLY"


def test_write_session_roles_csv_persists_geometry_authority_receipt(tmp_path: Path):
    edge = _make_edge(
        session_a="CORE",
        session_b="SUP",
        independent_artifact=True,
        evidence_scope="exact_pair",
        geometry_complete=True,
        group_holdout_disjoint=True,
        independent_bridge_groups=2,
        fit_evidence_ids=("f1", "f2"),
        holdout_evidence_ids=("h1", "h2"),
        status="STRONG",
    )
    rows = build_role_rows(
        [_make_session(session_id="CORE"), _make_session(session_id="SUP")],
        [edge],
        {"CORE": "BASE_CORE", "SUP": "BASE_SUPPORT"},
        {},
    )
    path = tmp_path / "session_roles.csv"
    write_session_roles_csv(path, rows)
    with path.open(encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert written
    assert list(written[0].keys()) == SESSION_ROLE_COLUMNS
    assert "fusion_authorization" in written[0]
    assert "reason" in written[0]
    by_id = {row["session_id"]: row for row in written}
    for sid, expected in (("CORE", "GLOBAL_BA"), ("SUP", "GLOBAL_BA")):
        row = by_id[sid]
        assert row["fusion_authorization"] == expected
        assert row["authorized_edge"] == "CORE-SUP"
        receipts = json.loads(row["geometry_authority"])
        assert isinstance(receipts, list)
        assert any(item["authorized"] is True for item in receipts)
        grant = next(item for item in receipts if item["authorized"] is True)
        assert grant["session_a"] == "CORE"
        assert grant["session_b"] == "SUP"
        assert grant["hard_status"] == "VALID"
    reinforce_rows = build_role_rows(
        [_make_session(session_id="SIDE")],
        [
            _make_edge(
                session_a="SIDE",
                session_b="CORE",
                independent_artifact=True,
                evidence_scope="exact_pair",
                geometry_complete=True,
                group_holdout_disjoint=True,
                independent_bridge_groups=2,
                fit_evidence_ids=("f1", "f2"),
                holdout_evidence_ids=("h1", "h2"),
                status="STRONG",
            )
        ],
        {"SIDE": "GEOMETRY_REINFORCEMENT"},
        {},
    )
    reinforce_path = tmp_path / "reinforce_roles.csv"
    write_session_roles_csv(reinforce_path, reinforce_rows)
    with reinforce_path.open(encoding="utf-8", newline="") as handle:
        reinforce = list(csv.DictReader(handle))
    assert reinforce[0]["fusion_authorization"] == "LOCAL_FUSION"
    assert reinforce[0]["authorized_edge"] == "SIDE-CORE"
    local = json.loads(reinforce[0]["geometry_authority"])
    assert any(item["authorized"] is True for item in local)
