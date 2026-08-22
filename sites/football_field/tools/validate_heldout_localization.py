#!/usr/bin/env python3
"""Apply the S9 acceptance contract to held-out EDM video benchmarks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ts_common import TEST


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def heldout_contract_from_manifest(path: Path) -> dict[str, str | int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    heldout = payload.get("test")
    if not isinstance(heldout, list) or len(heldout) != len(TEST):
        raise ValueError(
            f"held-out contract requires exactly {len(TEST)} independent video(s)"
        )
    item = heldout[0]
    digest = str(item.get("source_sha256") or item.get("sha256") or "")
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError("held-out video requires a lowercase SHA-256 digest")
    probed = item.get("probed", {})
    total_frames = int(probed.get("nb_frames", 0) or 0)
    video_id = str(item.get("seq", ""))
    expected = {video.seq for video in TEST}
    if video_id not in expected:
        raise ValueError(f"held-out video {video_id!r} is not in the declared TEST set")
    return {
        "video_id": video_id,
        "video_sha256": digest,
        "total_frames": total_frames,
    }


def _checks_pass(checks: dict) -> bool:
    return all(value is True or value == "NOT_APPLICABLE" for value in checks.values())


def evaluate_results(
    results: list[dict],
    *,
    expected_video_id: str | None = None,
) -> dict:
    expected_id = expected_video_id or TEST[0].seq
    evidence = []
    for result in results:
        reference_counts = result.get("reference_sequence_counts", {})
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
                "reference_sequence_counts": reference_counts,
            }
        )

    enough_runs = len(evidence) == len(TEST)
    bound = enough_runs and all(item.get("video") == expected_id for item in evidence)
    all_rates = bound and all(item["rate"] >= 0.95 for item in evidence)
    all_continuous = bound and all(item["continuity_ok"] for item in evidence)
    inliers_supported = bound and all(
        item["inliers_p05"] is not None and float(item["inliers_p05"]) >= 30.0
        for item in evidence
    )
    no_ghost_teleports = bound and all(
        int(item["jumps_gt_10x_median"] or 0) == 0
        and item["rejected_jump_fraction"] <= 0.005
        for item in evidence
    )
    return {
        "stage": "S9_heldout_localization",
        "checks": {
            "G9.1": all_rates,
            "G9.2": all_continuous,
            "G9.3": "NOT_APPLICABLE",
            "G9.4": inliers_supported,
            "G9.5": no_ghost_teleports,
        },
        "runs": evidence,
        "heldout_video_id": expected_id,
        "g9_3_reason": "no reverse pair; football BUILD directions are unknown",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--forced-manifest", type=Path, required=False)
    parser.add_argument("--corpus-manifest", type=Path, required=False)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.result]
    expected_video_id = TEST[0].seq
    if args.corpus_manifest is not None:
        expected_video_id = str(
            heldout_contract_from_manifest(args.corpus_manifest)["video_id"]
        )
    report = evaluate_results(results, expected_video_id=expected_video_id)
    report["status"] = "PASS" if _checks_pass(report["checks"]) else "FAIL"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if report["status"] != "PASS":
        raise SystemExit("S9 gate failed")


if __name__ == "__main__":
    main()
