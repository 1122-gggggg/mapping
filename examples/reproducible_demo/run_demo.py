#!/usr/bin/env python3
"""Run MapDoctor's complete public demo and assert the expected behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_demo import NUM_QUERIES, generate
from mapdoctor.cli import main as mapdoctor_main


def run(output: Path) -> None:
    artifacts = generate(output / "data")
    model = artifacts["model"]

    for backend in ("colmap", "glomap", "gluemap"):
        report_dir = output / "map_health" / backend
        code = mapdoctor_main([backend, str(model), "--output", str(report_dir)])
        if code != 0:
            raise RuntimeError(f"{backend} map analysis returned {code}")
        payload = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
        if payload["metrics"]["source"] != backend:
            raise RuntimeError(f"{backend} adapter provenance was not preserved")
        if payload["metrics"]["registered_images"] != 8:
            raise RuntimeError(f"{backend} registered-image count was unexpected")

    base_dir = output / "benchmark" / "base"
    candidate_dir = output / "benchmark" / "candidate"
    if mapdoctor_main(["benchmark", str(artifacts["base"]), "--output", str(base_dir)]) != 0:
        raise RuntimeError("base benchmark failed to execute")
    if mapdoctor_main(["benchmark", str(artifacts["candidate"]), "--output", str(candidate_dir)]) != 0:
        raise RuntimeError("candidate benchmark failed to execute")

    base = json.loads((base_dir / "benchmark.json").read_text(encoding="utf-8"))["summary"]
    candidate = json.loads((candidate_dir / "benchmark.json").read_text(encoding="utf-8"))["summary"]
    if base["total_queries"] != NUM_QUERIES or base["strict_success_rate"] != 1.0:
        raise RuntimeError("base benchmark should pass every synthetic held-out query")
    expected_candidate_rate = (NUM_QUERIES - 2) / NUM_QUERIES
    if abs(candidate["strict_success_rate"] - expected_candidate_rate) > 1e-12:
        raise RuntimeError("candidate benchmark should contain exactly two strict failures")

    comparison_dir = output / "comparison"
    compare_code = mapdoctor_main(
        [
            "compare",
            str(artifacts["base"]),
            str(artifacts["candidate"]),
            "--output",
            str(comparison_dir),
        ]
    )
    if compare_code != 1:
        raise RuntimeError("deliberately regressed candidate should fail the regression gate")
    comparison = json.loads((comparison_dir / "comparison.json").read_text(encoding="utf-8"))
    if comparison["status"] != "FAIL":
        raise RuntimeError("comparison status should be FAIL")
    if comparison["newly_failed"] != ["query_007.jpg", "query_015.jpg"]:
        raise RuntimeError("comparison did not identify the two deliberately regressed queries")

    print("MapDoctor reproducible demo: PASS")
    print(f"Artifacts: {output}")
    print("Three adapters: COLMAP / GLOMAP / GLUEMAP")
    print(f"Base benchmark: {base['strict_success_rate']:.1%} strict success")
    print(f"Candidate benchmark: {candidate['strict_success_rate']:.1%} strict success")
    print(f"Regression gate: {comparison['status']} (expected)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("mapdoctor-demo-output"))
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
