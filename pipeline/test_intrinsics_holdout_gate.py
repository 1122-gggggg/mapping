from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from intrinsics_holdout_gate import build_gate, evaluate_eval_stream_json


def base_args(tmp_path: Path, candidate: str, eval_json: list[str] | None = None) -> Namespace:
    return Namespace(
        python="/usr/bin/python3.12",
        candidate=[candidate],
        main_candidate="no_undistort_official69",
        candidate_bundle=[],
        eval_json=eval_json or [],
        base_bundle=tmp_path / "base.pt",
        base_megaloc_cache=tmp_path / "cache.npz",
        holdout_dir=tmp_path / "holdout",
        stride=10,
        resize="1280x720",
        min_success=0.90,
        max_ok_to_fail=0,
        max_final_fail_run=30,
    )


def write_eval(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")


def write_eval_with_meta(path: Path, rows: list[dict], final: Path) -> None:
    path.write_text(json.dumps({"final": str(final), "rows": rows}), encoding="utf-8")


def test_eval_stream_json_passes_absolute_gate(tmp_path: Path):
    path = tmp_path / "eval.json"
    write_eval(path, [{
        "set": "P123",
        "base_success": 0.88,
        "final_success": 0.92,
        "ok_to_fail": 0,
        "final_max_fail_run": 12,
    }])

    ok, metrics, reasons = evaluate_eval_stream_json(path, 0.90, 0, 30)

    assert ok
    assert reasons == []
    assert metrics["min_final_success"] == 0.92


def test_build_gate_fails_when_holdout_eval_is_missing(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    report = build_gate(base_args(tmp_path, f"no_undistort_official69={run_dir}"))

    row = report["candidates"][0]
    assert not report["overall_ok"]
    assert row["status"] == "needs_holdout_eval"
    assert "holdout localization eval JSON missing" in row["reasons"]
    assert "--mode" in row["holdout_command"]


def test_build_gate_passes_when_main_candidate_eval_passes(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bundle = run_dir / "deploy" / "reloc_map_xfeat_tri.pt"
    bundle.parent.mkdir()
    bundle.write_bytes(b"bundle")
    eval_path = tmp_path / "eval.json"
    write_eval_with_meta(eval_path, [{
        "set": "P123",
        "base_success": 0.88,
        "final_success": 0.91,
        "ok_to_fail": 0,
        "final_max_fail_run": 10,
    }], final=bundle)

    report = build_gate(base_args(
        tmp_path,
        f"no_undistort_official69={run_dir}",
        eval_json=[f"no_undistort_official69={eval_path}"],
    ))

    assert report["overall_ok"]
    assert report["candidates"][0]["status"] == "pass"


def test_build_gate_rejects_eval_without_matching_candidate_bundle(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    eval_path = tmp_path / "eval.json"
    write_eval(eval_path, [{
        "set": "P123",
        "base_success": 0.88,
        "final_success": 0.99,
        "ok_to_fail": 0,
        "final_max_fail_run": 1,
    }])

    report = build_gate(base_args(
        tmp_path,
        f"no_undistort_official69={run_dir}",
        eval_json=[f"no_undistort_official69={eval_path}"],
    ))

    assert not report["overall_ok"]
    row = report["candidates"][0]
    assert row["status"] == "fail"
    assert "candidate deployment bundle missing" in row["reasons"]
    assert "eval JSON missing final bundle provenance" in row["reasons"]
