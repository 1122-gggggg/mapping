from __future__ import annotations

import pytest

from mapdoctor.recapture.audit import AuditStatus, audit_cells
from mapdoctor.recapture.planner import plan_regions
from mapdoctor.recapture.types import (
    Availability,
    Backend,
    CaptureMode,
    DecisionStatus,
    MetricValue,
    PoseDirectionCell,
)


def _metrics(
    *,
    weak: bool = False,
    snap: float | None = None,
    repairability: float | None = None,
) -> dict[str, MetricValue]:
    metrics = {
        "coordinate_scale_status": MetricValue("map_units", Availability.AVAILABLE),
        "camera_intrinsics_valid": MetricValue(True, Availability.AVAILABLE),
        "frame_transform_valid": MetricValue(True, Availability.AVAILABLE),
        "handedness_valid": MetricValue(True, Availability.AVAILABLE),
        "visible_landmark_count": MetricValue(10 if weak else 120, Availability.AVAILABLE),
        "inlier_convex_hull_coverage": MetricValue(0.03 if weak else 0.25, Availability.AVAILABLE),
        "grid_occupancy_count": MetricValue(2 if weak else 9, Availability.AVAILABLE),
        "positive_depth_ratio": MetricValue(1.0, Availability.AVAILABLE),
        "fim_rank": MetricValue(4 if weak else 6, Availability.AVAILABLE),
        "fim_lambda_min": MetricValue(1e-5 if weak else 0.1, Availability.AVAILABLE),
        "fim_condition_number": MetricValue(5000 if weak else 50, Availability.AVAILABLE),
        "triangulation_angle_p10_deg": MetricValue(0.2 if weak else 3.0, Availability.AVAILABLE),
        "view_direction_entropy": MetricValue(0.1 if weak else 0.7, Availability.AVAILABLE),
        "attempt_count": MetricValue(20, Availability.AVAILABLE),
        "localization_success_rate": MetricValue(0.2 if weak else 0.98, Availability.AVAILABLE),
        "holdout_query_coverage": MetricValue(0.9, Availability.AVAILABLE),
    }
    if snap is not None:
        metrics["dense_to_3d_snap_ratio"] = MetricValue(snap, Availability.AVAILABLE)
    if repairability is not None:
        metrics["existing_data_repairability"] = MetricValue(repairability, Availability.DERIVED)
        metrics["existing_data_counterfactual_complete"] = MetricValue(True, Availability.DERIVED)
    return metrics


def _cells(
    region: str,
    metrics: dict[str, MetricValue],
    health: tuple[float, float] = (0.2, 0.25),
    *,
    route_tangent: tuple[float, float, float] | None = (0.0, 1.0, 0.0),
    map_up_vector: tuple[float, float, float] | None = (0.0, 0.0, 1.0),
    producer: str = "gluemap",
) -> list[PoseDirectionCell]:
    return [
        PoseDirectionCell(
            f"{region}-a",
            region,
            (10.0, 20.0, 3.0),
            0.0,
            0.0,
            Backend.EDM,
            metrics,
            directional_health=health[0],
            route_tangent=route_tangent,
            map_up_vector=map_up_vector,
            map_producer=producer,
        ),
        PoseDirectionCell(
            f"{region}-b",
            region,
            (10.0, 20.0, 3.0),
            90.0,
            0.0,
            Backend.EDM,
            metrics,
            directional_health=health[1],
            route_tangent=route_tangent,
            map_up_vector=map_up_vector,
            map_producer=producer,
        ),
    ]


def test_estimated_hard_fim_cannot_authorize_recapture() -> None:
    metrics = _metrics(weak=True, repairability=0.1)
    metrics["fim_rank"] = MetricValue(6, Availability.ESTIMATED)
    report = audit_cells(_cells("r", metrics), Backend.EDM)
    assert report.item("fim_rank").status == AuditStatus.ESTIMATED_ONLY
    assert not report.authorization_ready


def test_orientation_sensitive_region_uses_navigation_policy() -> None:
    decisions, _ = plan_regions(
        _cells("orient", _metrics(repairability=0.1), health=(0.95, 0.2)),
        Backend.EDM,
    )
    assert decisions[0].status == DecisionStatus.NAVIGATION_POLICY_ONLY
    assert not decisions[0].recapture_required


def test_orientation_sensitive_region_requires_holdout_evidence() -> None:
    metrics = _metrics(repairability=0.1)
    del metrics["attempt_count"]
    del metrics["holdout_query_coverage"]
    del metrics["localization_success_rate"]
    decisions, _ = plan_regions(
        _cells("orient-no-holdout", metrics, health=(0.95, 0.2)),
        Backend.EDM,
    )
    assert decisions[0].status == DecisionStatus.EVIDENCE_CAPTURE_ONLY
    assert not decisions[0].recapture_required


def test_backend_association_failure_repairs_existing_data_first() -> None:
    decisions, _ = plan_regions(
        _cells("backend", _metrics(weak=True, snap=0.2)),
        Backend.EDM,
    )
    assert decisions[0].status == DecisionStatus.EXISTING_DATA_REPAIR_FIRST
    assert decisions[0].existing_data_repairability == pytest.approx(0.9)
    assert any("dense-to-track" in action for action in decisions[0].non_capture_actions)


def test_measured_low_repairability_is_not_overridden_by_stage_heuristics() -> None:
    decisions, _ = plan_regions(
        _cells("measured", _metrics(weak=True, snap=0.2, repairability=0.1)),
        Backend.EDM,
    )
    decision = decisions[0]
    assert decision.status == DecisionStatus.TARGETED_RECAPTURE_REQUIRED
    assert decision.recapture_required
    assert decision.existing_data_repairability == pytest.approx(0.1)


def test_unknown_repairability_never_authorizes_recapture() -> None:
    decisions, audits = plan_regions(
        _cells("unknown", _metrics(weak=True)),
        Backend.EDM,
    )
    decision = decisions[0]
    assert decision.status == DecisionStatus.EXISTING_DATA_REPAIR_FIRST
    assert not decision.recapture_required
    assert decision.existing_data_repairability is None
    assert "existing_data_repairability" in decision.blocked_by
    assert "existing_data_repairability" in audits["unknown"].blocking_metrics


def test_incomplete_counterfactual_cannot_authorize_recapture() -> None:
    metrics = _metrics(weak=True, repairability=0.1)
    metrics["existing_data_counterfactual_complete"] = MetricValue(False, Availability.DERIVED)
    decisions, audits = plan_regions(_cells("incomplete", metrics), Backend.EDM)
    assert not audits["incomplete"].authorization_ready
    assert decisions[0].status == DecisionStatus.EVIDENCE_CAPTURE_ONLY
    assert not decisions[0].recapture_required
    assert "existing_data_counterfactual_complete" in decisions[0].blocked_by


def test_intrinsic_weakness_generates_targeted_diverse_capture() -> None:
    decisions, audits = plan_regions(
        _cells("recap", _metrics(weak=True, repairability=0.1)),
        Backend.EDM,
    )
    decision = decisions[0]
    assert audits["recap"].authorization_ready
    assert decision.status == DecisionStatus.TARGETED_RECAPTURE_REQUIRED
    modes = {capture_pass.mode for capture_pass in decision.capture_passes}
    assert CaptureMode.ANCHOR_BRIDGE in modes
    assert CaptureMode.LATERAL_OBLIQUE_LEFT in modes
    assert CaptureMode.LATERAL_OBLIQUE_RIGHT in modes
    assert all(capture_pass.safety_status == Availability.UNAVAILABLE for capture_pass in decision.capture_passes)
    assert all(
        capture_pass.expected_gain["delta_fim_lambda_min"].status == Availability.UNAVAILABLE
        for capture_pass in decision.capture_passes
    )
    lateral = next(
        capture_pass
        for capture_pass in decision.capture_passes
        if capture_pass.mode == CaptureMode.LATERAL_OBLIQUE_LEFT
    )
    assert lateral.poses[0].yaw_deg is None
    assert lateral.poses[0].pitch_deg is None
    assert lateral.poses[0].look_at == pytest.approx((10.0, 20.0, 3.0))


def test_capture_offsets_follow_route_not_world_x_axis() -> None:
    decisions, _ = plan_regions(
        _cells("route", _metrics(weak=True, repairability=0.1), route_tangent=(0.0, 1.0, 0.0)),
        Backend.EDM,
    )
    anchor = next(
        capture_pass for capture_pass in decisions[0].capture_passes if capture_pass.mode == CaptureMode.ANCHOR_BRIDGE
    )
    entry, center, exit_pose = anchor.poses
    assert center.position == pytest.approx((10.0, 20.0, 3.0))
    assert entry.position[0] == pytest.approx(center.position[0])
    assert exit_pose.position[0] == pytest.approx(center.position[0])
    assert entry.position[1] < center.position[1] < exit_pose.position[1]

    lateral = next(
        capture_pass
        for capture_pass in decisions[0].capture_passes
        if capture_pass.mode == CaptureMode.LATERAL_OBLIQUE_LEFT
    )
    assert lateral.poses[0].position[1] == pytest.approx(center.position[1])
    assert lateral.poses[0].position[0] != pytest.approx(center.position[0])


def test_non_z_up_map_uses_explicit_up_vector() -> None:
    # Route +X and map-up +Y => lateral is -Z/+Z, not a hidden z-up assumption.
    decisions, _ = plan_regions(
        _cells(
            "custom-up",
            _metrics(weak=True, repairability=0.1),
            route_tangent=(1.0, 0.0, 0.0),
            map_up_vector=(0.0, 1.0, 0.0),
        ),
        Backend.EDM,
    )
    lateral = next(
        capture_pass
        for capture_pass in decisions[0].capture_passes
        if capture_pass.mode == CaptureMode.LATERAL_OBLIQUE_LEFT
    )
    assert lateral.poses[0].position[0] == pytest.approx(10.0)
    assert lateral.poses[0].position[1] == pytest.approx(20.0)
    assert lateral.poses[0].position[2] != pytest.approx(3.0)


def test_missing_route_frame_does_not_invent_capture_coordinates() -> None:
    for route_tangent, map_up_vector, expected_blocker in [
        (None, (0.0, 0.0, 1.0), "route_tangent"),
        ((0.0, 1.0, 0.0), None, "map_up_vector"),
    ]:
        decisions, _ = plan_regions(
            _cells(
                f"missing-{expected_blocker}",
                _metrics(weak=True, repairability=0.1),
                route_tangent=route_tangent,
                map_up_vector=map_up_vector,
            ),
            Backend.EDM,
        )
        decision = decisions[0]
        assert decision.status == DecisionStatus.TARGETED_RECAPTURE_REQUIRED
        assert decision.recapture_required
        assert decision.capture_passes == ()
        assert expected_blocker in decision.blocked_by


def test_non_gluemap_map_does_not_require_gluemap_diagnostics() -> None:
    report = audit_cells(
        _cells("colmap", _metrics(repairability=0.1), producer="colmap"),
        Backend.EDM,
    )
    assert report.item("gluemap_retrieval_degree_p50") is None


def test_gluemap_map_enables_gluemap_diagnostic_audit() -> None:
    report = audit_cells(
        _cells("gluemap", _metrics(repairability=0.1), producer="gluemap"),
        Backend.EDM,
    )
    item = report.item("gluemap_retrieval_degree_p50")
    assert item is not None
    assert item.status == AuditStatus.MISSING


def test_pose_cell_input_fails_closed_on_missing_geometry() -> None:
    with pytest.raises(ValueError, match="position"):
        PoseDirectionCell.from_dict({"region_id": "r", "yaw_deg": 0, "pitch_deg": 0})
    with pytest.raises(ValueError, match="yaw"):
        PoseDirectionCell.from_dict({"region_id": "r", "position": [0, 0, 0], "pitch_deg": 0})


def test_pose_cell_parses_false_operational_direction() -> None:
    cell = PoseDirectionCell.from_dict(
        {
            "region_id": "r",
            "position": [0, 0, 0],
            "yaw_deg": 0,
            "pitch_deg": 0,
            "operational_direction": "false",
        }
    )
    assert cell.operational_direction is False

