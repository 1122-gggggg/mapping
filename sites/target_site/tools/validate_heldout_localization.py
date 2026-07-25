#!/usr/bin/env python3
"""Apply the S9 acceptance contract to held-out EDM video benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_results(results: list[dict], *, reverse_sequences: set[str]) -> dict:
    evidence = []
    for result in results:
        reference_counts = result.get("reference_sequence_counts", {})
        reference_total = sum(int(value) for value in reference_counts.values())
        reverse_total = sum(
            int(value)
            for sequence, value in reference_counts.items()
            if sequence in reverse_sequences
        )
        reverse_fraction = reverse_total / max(1, reference_total)
        frames = int(result.get("frames", 0))
        rejected_jumps = int(result.get("rejections", {}).get("jump", 0))
        median = result.get("step_median")
        p95 = result.get("step_p95")
        continuity_ok = (
            median is not None
            and p95 is not None
            and float(median) > 0
            and float(p95) <= 10.0 * float(median)
            and int(result.get("jumps_gt_10x_median", -1)) == 0
        )
        evidence.append(
            {
                "video": result.get("video"),
                "frames": frames,
                "localized": int(result.get("localized", 0)),
                "rate": float(result.get("rate", 0.0)),
                "inliers_p05": result.get("inliers_p05"),
                "continuity_ok": continuity_ok,
                "step_median": median,
                "step_p95": p95,
                "jumps_gt_10x_median": result.get("jumps_gt_10x_median"),
                "rejected_jumps": rejected_jumps,
                "rejected_jump_fraction": rejected_jumps / max(1, frames),
                "reverse_reference_fraction": reverse_fraction,
                "reference_sequence_counts": reference_counts,
            }
        )

    enough_runs = len(evidence) >= 2
    all_rates = enough_runs and all(item["rate"] >= 0.95 for item in evidence)
    all_continuous = enough_runs and all(item["continuity_ok"] for item in evidence)
    reverse_supported = all_rates and any(
        item["reverse_reference_fraction"] >= 0.50 for item in evidence
    )
    inliers_supported = enough_runs and all(
        item["inliers_p05"] is not None and float(item["inliers_p05"]) >= 30.0
        for item in evidence
    )
    no_ghost_teleports = enough_runs and all(
        int(item["jumps_gt_10x_median"] or 0) == 0
        and item["rejected_jump_fraction"] <= 0.005
        for item in evidence
    )
    return {
        "stage": "S9_heldout_localization",
        "checks": {
            "G9.1": all_rates,
            "G9.2": all_continuous,
            "G9.3": reverse_supported,
            "G9.4": inliers_supported,
            "G9.5": no_ghost_teleports,
        },
        "runs": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--forced-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.result]
    forced = json.loads(args.forced_manifest.read_text(encoding="utf-8"))
    report = evaluate_results(results, reverse_sequences=set(forced["rev"]))
    report["status"] = "PASS" if all(report["checks"].values()) else "FAIL"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if report["status"] != "PASS":
        raise SystemExit("S9 gate failed")


if __name__ == "__main__":
    main()
