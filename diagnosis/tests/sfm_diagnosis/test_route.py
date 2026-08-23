from __future__ import annotations

import numpy as np
import pytest

from sfm_diagnosis.route import (
    RouteAuditConfig,
    audit_route,
    fit_monotonic_risk_calibrator,
    resample_route,
    save_route_audit,
)


def _heat_row(x, health, forward, *, primary="HEALTHY", weak=None):
    row = {
        "x": x,
        "y": 0.0,
        "z": 0.0,
        "health_score": health,
        "forward_x": forward[0],
        "forward_y": forward[1],
        "forward_z": forward[2],
        "primary": primary,
        "codes": primary,
        "fim_weakest_translation_fraction": 1.0,
        "fim_sigma_translation_worst_m": 1.0,
    }
    if weak is not None:
        row.update(
            {
                "fim_weakest_translation_world_x": weak[0],
                "fim_weakest_translation_world_y": weak[1],
                "fim_weakest_translation_world_z": weak[2],
            }
        )
    return row


def test_resample_route_is_uniform_and_keeps_endpoint():
    route = [{"x": 0, "y": 0, "z": 0}, {"x": 2.2, "y": 0, "z": 0}]
    samples = resample_route(route, spacing_m=0.5)
    assert samples.distance_m.tolist() == pytest.approx([0, 0.5, 1.0, 1.5, 2.0, 2.2])
    assert samples.positions[-1].tolist() == pytest.approx([2.2, 0, 0])
    assert np.allclose(samples.tangent_w, [1, 0, 0])


def test_isotonic_calibrator_is_monotonic_and_reports_risk_coverage():
    rows = []
    # Deliberately non-monotone empirical groups: PAV must pool violations.
    for health, successes in [(0.1, 1), (0.4, 4), (0.7, 3), (0.9, 9)]:
        rows.extend(
            {"health_score": health, "success": int(i < successes)} for i in range(10)
        )
    calibrator = fit_monotonic_risk_calibrator(rows, min_samples=20)
    health = np.linspace(0, 1, 101)
    risk = calibrator.predict_failure(health)
    assert np.all(np.diff(risk) <= 1e-12)
    assert calibrator.num_samples == 40
    assert len(calibrator.risk_coverage) == 10
    assert 0 <= calibrator.brier_score <= 1


def test_route_dynamic_program_prefers_smooth_sequence():
    # At x=1 the reverse-facing candidate is slightly healthier, but a smoothness
    # penalty should keep the globally coherent +X sequence.
    heat = [
        _heat_row(0, 0.80, (1, 0, 0)),
        _heat_row(0, 0.20, (-1, 0, 0)),
        _heat_row(1, 0.75, (1, 0, 0)),
        _heat_row(1, 0.82, (-1, 0, 0)),
        _heat_row(2, 0.80, (1, 0, 0)),
        _heat_row(2, 0.20, (-1, 0, 0)),
    ]
    route = [{"x": 0, "y": 0, "z": 0}, {"x": 2, "y": 0, "z": 0}]
    cfg = RouteAuditConfig(
        sample_spacing_m=1.0,
        max_heatmap_distance_m=0.1,
        smoothness_weight=1.0,
        task_forward_weight=0.0,
        weak_direction_weight=0.0,
        enter_risk=0.9,
        exit_risk=0.8,
    )
    result = audit_route(heat, route, config=cfg)
    assert [row["forward_x"] for row in result.samples] == pytest.approx([1, 1, 1])
    assert result.samples[1]["orientation_regret"] > 0


def test_route_marks_map_limited_orientation_limited_and_unsupported():
    heat = [
        _heat_row(0, 0.30, (1, 0, 0), primary="GEOMETRY_WEAK"),
        _heat_row(1, 0.90, (1, 0, 0)),
        _heat_row(1, 0.20, (-1, 0, 0)),
    ]
    route = [{"x": 0, "y": 0, "z": 0}, {"x": 3, "y": 0, "z": 0}]
    cfg = RouteAuditConfig(
        sample_spacing_m=1.0,
        max_heatmap_distance_m=0.2,
        smoothness_weight=0.0,
        task_forward_weight=0.0,
        weak_direction_weight=0.0,
        enter_risk=0.45,
        exit_risk=0.35,
        min_segment_length_m=0.0,
    )
    result = audit_route(heat, route, config=cfg)
    assert result.samples[0]["limitation"] == "MAP_LIMITED"
    assert result.samples[1]["limitation"] == "HEALTHY"
    assert result.samples[2]["limitation"] == "NO_HEATMAP_SUPPORT"
    assert result.samples[3]["limitation"] == "NO_HEATMAP_SUPPORT"
    assert result.summary["num_weak_segments"] >= 1
    assert result.summary["unsupported_length_m"] > 0


def test_observation_scale_weak_requests_standoff_change():
    heat = [
        _heat_row(0, 0.30, (1, 0, 0), primary="OBSERVATION_SCALE_WEAK"),
        _heat_row(1, 0.30, (1, 0, 0), primary="OBSERVATION_SCALE_WEAK"),
    ]
    route = [{"x": 0, "y": 0, "z": 0}, {"x": 1, "y": 0, "z": 0}]
    cfg = RouteAuditConfig(
        sample_spacing_m=1.0,
        max_heatmap_distance_m=0.5,
        smoothness_weight=0.0,
        task_forward_weight=0.0,
        weak_direction_weight=0.0,
        enter_risk=0.45,
        exit_risk=0.35,
        min_segment_length_m=0.0,
    )
    result = audit_route(heat, route, config=cfg)
    assert result.summary["num_weak_segments"] >= 1
    assert result.weak_segments[0]["repair_class"] == "STANDOFF_OR_ZOOM_CHANGE"

def test_weak_direction_penalty_increases_route_risk():
    heat = [
        _heat_row(0, 0.8, (1, 0, 0), weak=(1, 0, 0)),
        _heat_row(1, 0.8, (1, 0, 0), weak=(1, 0, 0)),
    ]
    route = [{"x": 0, "y": 0, "z": 0}, {"x": 1, "y": 0, "z": 0}]
    cfg = RouteAuditConfig(
        sample_spacing_m=1.0,
        max_heatmap_distance_m=0.1,
        weak_direction_weight=0.2,
        task_forward_weight=0.0,
        smoothness_weight=0.0,
        min_segment_length_m=0.0,
    )
    result = audit_route(heat, route, config=cfg)
    assert result.samples[0]["base_failure_risk"] == pytest.approx(0.2)
    assert result.samples[0]["directional_risk_penalty"] == pytest.approx(0.2)
    assert result.samples[0]["route_risk"] == pytest.approx(0.4)


def test_calibration_requires_enough_samples():
    with pytest.raises(ValueError, match="at least 20"):
        fit_monotonic_risk_calibrator(
            [{"health_score": 0.5, "success": 1}],
            min_samples=20,
        )


def test_calibrated_route_exposes_probability():
    heat = [
        _heat_row(0, 0.2, (1, 0, 0)),
        _heat_row(1, 0.8, (1, 0, 0)),
    ]
    route = [{"x": 0, "y": 0, "z": 0}, {"x": 1, "y": 0, "z": 0}]
    calibration = [
        {"health_score": i / 19, "success": int(i >= 10)} for i in range(20)
    ]
    cfg = RouteAuditConfig(
        sample_spacing_m=1.0,
        max_heatmap_distance_m=0.1,
        weak_direction_weight=0.0,
        task_forward_weight=0.0,
        smoothness_weight=0.0,
    )
    result = audit_route(heat, route, config=cfg, calibration=calibration)
    assert result.calibration is not None
    assert result.samples[0]["predicted_failure_probability"] is not None
    assert (
        result.samples[0]["predicted_failure_probability"]
        >= result.samples[1]["predicted_failure_probability"]
    )


def test_save_route_audit_writes_machine_readable_outputs(tmp_path):
    heat = [
        _heat_row(0, 0.8, (1, 0, 0)),
        _heat_row(1, 0.8, (1, 0, 0)),
    ]
    route = [{"x": 0, "y": 0, "z": 0}, {"x": 1, "y": 0, "z": 0}]
    result = audit_route(
        heat,
        route,
        config=RouteAuditConfig(
            sample_spacing_m=1.0,
            max_heatmap_distance_m=0.1,
            weak_direction_weight=0.0,
        ),
    )
    save_route_audit(tmp_path, result)
    assert (tmp_path / "route_samples.csv").exists()
    assert (tmp_path / "weak_segments.csv").exists()
    assert (tmp_path / "route_summary.json").exists()
