import json

import pytest

from mapdoctor.diagnostics.io import (
    load_query_manifest,
    load_region_assignments,
    load_risk_scores,
)


def test_manifest_loaders(tmp_path):
    manifest = tmp_path / "queries.json"
    manifest.write_text(json.dumps({"queries": ["q1", "q2"]}), encoding="utf-8")
    assert load_query_manifest(manifest) == ["q1", "q2"]

    assignments = tmp_path / "regions.json"
    assignments.write_text(json.dumps({"q1": "A", "q2": "B"}), encoding="utf-8")
    assert load_region_assignments(assignments) == {"q1": "A", "q2": "B"}

    risks = tmp_path / "risk.csv"
    risks.write_text("query,risk\nq1,0.1\nq2,0.9\n", encoding="utf-8")
    assert load_risk_scores(risks) == {"q1": 0.1, "q2": 0.9}


def test_manifest_rejects_duplicates(tmp_path):
    path = tmp_path / "queries.txt"
    path.write_text("q1\nq1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_query_manifest(path)


@pytest.mark.parametrize(
    "payload",
    [
        {"queries": [None]},
        {"queries": [123]},
        [{"query": "q1"}],
    ],
)
def test_manifest_rejects_non_string_values(tmp_path, payload):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="string"):
        load_query_manifest(path)


def test_region_and_risk_loaders_reject_non_string_query_ids(tmp_path):
    assignments = tmp_path / "regions.json"
    assignments.write_text(json.dumps([{"query": None, "region": "A"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="string"):
        load_region_assignments(assignments)

    risks = tmp_path / "risk.json"
    risks.write_text(json.dumps([{"query": None, "risk": 0.5}]), encoding="utf-8")
    with pytest.raises(ValueError, match="string"):
        load_risk_scores(risks)
