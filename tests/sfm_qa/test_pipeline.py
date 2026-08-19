from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from sfm_qa.pipeline import analyze, attribute_query, check


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


def test_analyze_without_database_still_works(tmp_path):
    paths = _generate_demo(tmp_path / "demo")
    report = analyze(paths["model"], backend="gluemap", output_dir=tmp_path / "out")
    reconstruction = report["map"]["reconstruction"]
    assert "diagnostic_mode" in reconstruction or "num_weak_regions" in reconstruction
    assert (tmp_path / "out" / "map" / "report.json").exists()
    assert report["localization"] is None
