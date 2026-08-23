from __future__ import annotations

import csv
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

from mapdoctor.adapters import get_adapter
from sfm_qa.bridge import map_model_to_map_data
from sfm_qa.pipeline import (
    _overall_status,
    analyze,
    attribute_query,
    check,
    check_map,
    is_success_status,
)


def _generate_demo(tmp_path: Path):
    path = Path(__file__).resolve().parents[2] / "examples" / "reproducible_demo" / "generate_demo.py"
    spec = importlib.util.spec_from_file_location("generate_demo", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.generate(tmp_path)


def test_attribute_query_table():
    assert attribute_query(True, "DATA_SPARSE") == "OK"
    assert attribute_query(True, None) == "OK"
    assert attribute_query(False, None) == "UNATTRIBUTED"
    assert attribute_query(False, "DATA_SPARSE") == "MAP_LIMITED"
    assert attribute_query(False, "HEALTHY") == "LOCALIZER_LIMITED"
    assert attribute_query(False, "MATCHING_WEAK") == "LOCALIZER_LIMITED"
    assert attribute_query(False, "PERCEPTUAL_ALIASING_SUSPECTED") == "ALIAS_OR_PNP"


def test_check_demo_without_logs_writes_report(tmp_path):
    paths = _generate_demo(tmp_path / "demo")
    out = tmp_path / "out"
    report = check(paths["model"], backend="gluemap", output_dir=out)

    assert report["overall_status"] == "MAP_SCREENED_LOCALIZATION_UNCHECKED"
    assert report["localization"] is None
    readiness = report["map"]["readiness"]
    assert readiness["score"] == 100.0
    assert readiness["grade"] == "A"
    assert readiness["map_integrity_ok"] is True
    assert all(readiness["integrity_checks"].values())
    assert readiness["map_ok"] is True
    assert readiness["map_status"] == "PASS"
    assert all(item["pass"] for item in readiness["checks"].values())
    assert report["map"]["health"]["registered_images"] == 8
    assert report["map"]["health"]["points3d"] == 80
    assert report["map"]["graph"]["component_count"] == 1
    assert report["map"]["graph"]["largest_component_ratio"] == 1.0
    assert report["map"]["graph"]["articulation_images"] == []
    assert report["map"]["graph"]["bridge_edges"] == []
    assert report["map"]["reconstruction"]["num_weak_images"] == 0
    assert report["map"]["reconstruction"]["num_weak_regions"] == 0

    report_path = out / "report.json"
    assert report_path.exists()
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["overall_status"] == "MAP_SCREENED_LOCALIZATION_UNCHECKED"


def test_check_demo_base_csv_ready(tmp_path):
    paths = _generate_demo(tmp_path / "demo")
    report = check(paths["model"], backend="gluemap", logs_path=paths["base"])

    assert report["map"]["readiness"]["map_status"] == "PASS"
    assert report["overall_status"] == "READY"
    loc = report["localization"]
    assert loc["total"] == 20
    assert loc["strict_success_rate"] == 1.0
    assert loc["localization_ok"] is True
    assert loc["localization_status"] == "PASS"
    assert loc["attribution_counts"] == {"OK": 20}
    assert {row["attribution"] for row in loc["queries"]} == {"OK"}


def test_check_demo_candidate_csv_map_limited(tmp_path):
    paths = _generate_demo(tmp_path / "demo")
    report = check(paths["model"], backend="gluemap", logs_path=paths["candidate"])

    assert report["map"]["readiness"]["map_status"] == "PASS"
    assert report["overall_status"] == "LOCALIZATION_FAILED"
    loc = report["localization"]
    assert loc["total"] == 20
    assert loc["strict_success_rate"] == 0.9
    assert loc["localization_ok"] is False
    assert loc["localization_status"] == "FAIL"
    assert loc["attribution_counts"]["OK"] == 18
    assert loc["attribution_counts"]["MAP_LIMITED"] == 2

    by_name = {row["query"]: row for row in loc["queries"]}
    for name in ("query_007.jpg", "query_015.jpg"):
        row = by_name[name]
        assert row["strict_pass"] is False
        assert row["attribution"] == "MAP_LIMITED"
        assert row["diagnosis_primary"] == "DATA_SPARSE"
        assert "DATA_SPARSE" in row["diagnosis_codes"]
        assert row["attribution"] != "OK"

    relative = loc["relative_quality"]
    assert relative["selection_mode"] == "RELATIVE_RISK_COVERAGE"
    assert relative["coverage_curve"][0]["coverage"] > 0.0
    assert relative["coverage_curve"][-1]["coverage"] == 1.0
    best_score = max(row["relative_quality_score"] for row in loc["queries"])
    for name in ("query_007.jpg", "query_015.jpg"):
        assert by_name[name]["relative_quality_score"] < best_score
        assert by_name[name]["relative_risk_score"] > 0.0
        assert by_name[name]["relative_evidence_completeness"] < 1.0


def test_localization_accepts_target_rate_without_requiring_every_query(tmp_path):
    paths = _generate_demo(tmp_path / "demo")
    with paths["base"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0].update(
        {
            "success": "false",
            "inliers": "0",
            "inlier_ratio": "0.0",
            "reproj_p90_px": "9.0",
            "hull_coverage": "0.0",
            "grid4_occupancy": "0",
            "positive_depth_ratio": "0.0",
            "pose_consensus": "0.0",
        }
    )
    logs = tmp_path / "nineteen_of_twenty.json"
    logs.write_text(json.dumps(rows), encoding="utf-8")

    report = check(paths["model"], backend="gluemap", logs_path=logs)

    assert report["localization"]["strict_success_rate"] == 0.95
    assert report["localization"]["strict_successes"] == 19
    assert report["localization"]["required_strict_success_rate"] == 0.95
    assert report["localization"]["required_strict_successes"] == 19
    assert report["localization"]["localization_ok"] is True
    assert report["localization"]["heldout_provenance_verified"] is False
    assert report["overall_status"] == "READY"

    stricter = tmp_path / "stricter.json"
    stricter.write_text(
        json.dumps({"localization": {"min_strict_success_rate": 0.9500000000005}}),
        encoding="utf-8",
    )
    stricter_report = check(
        paths["model"],
        backend="gluemap",
        logs_path=logs,
        config_path=stricter,
    )
    assert stricter_report["localization"]["required_strict_successes"] == 20
    assert stricter_report["localization"]["localization_ok"] is False


def test_heldout_localization_can_override_advisory_map_metric_warning(tmp_path):
    paths = _generate_demo(tmp_path / "demo")
    config = tmp_path / "strict_map_screen.json"
    config.write_text(
        json.dumps({"health": {"min_observations_per_image": 1000}}),
        encoding="utf-8",
    )

    report = check(
        paths["model"],
        backend="gluemap",
        logs_path=paths["base"],
        config_path=config,
    )

    assert report["map"]["readiness"]["map_ok"] is False
    assert report["localization"]["localization_ok"] is True
    assert report["overall_status"] == "READY_WITH_MAP_WARNINGS"
    assert is_success_status(report["overall_status"]) is True


def test_invalid_map_integrity_cannot_be_overridden_by_logs() -> None:
    loc = {"localization_ok": True}

    assert (
        _overall_status(False, loc, map_integrity_ok=False)
        == "MAP_SCREENING_FAILED"
    )
    assert is_success_status("MAP_SCREENING_FAILED") is False


def test_nonfinite_map_pose_is_a_hard_integrity_failure(tmp_path) -> None:
    paths = _generate_demo(tmp_path / "demo")
    model = get_adapter("gluemap").load(paths["model"])
    valid_map_data = map_model_to_map_data(model)
    image_id = min(model.images)
    model.images[image_id] = replace(
        model.images[image_id],
        center=(float("nan"), 0.0, 0.0),
    )

    report = check_map(
        paths["model"],
        backend="gluemap",
        model=model,
        map_data=valid_map_data,
    )

    assert report["readiness"]["integrity_checks"]["image_poses_are_finite"] is False
    assert report["readiness"]["map_integrity_ok"] is False
    assert report["readiness"]["map_status"] == "INVALID"


def test_analyze_without_database_still_works(tmp_path):
    paths = _generate_demo(tmp_path / "demo")
    report = analyze(paths["model"], backend="gluemap", output_dir=tmp_path / "out")
    reconstruction = report["map"]["reconstruction"]
    assert "diagnostic_mode" in reconstruction or "num_weak_regions" in reconstruction
    assert (tmp_path / "out" / "map" / "report.json").exists()
    assert report["localization"] is None
