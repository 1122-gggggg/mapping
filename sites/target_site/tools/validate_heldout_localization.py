#!/usr/bin/env python3
"""Apply the S9 acceptance contract to held-out EDM video benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ts_common import TEST, hash_artifact  # noqa: E402

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

# Existing G9 numeric policies. Do not change these operating points.
G9_1_MIN_RATE = 0.95
G9_2_P95_MEDIAN_FACTOR = 10.0
G9_4_MIN_INLIERS_P05 = 30.0
G9_5_MAX_REJECTED_JUMP_FRACTION = 0.005

G9_3_REASON = (
    "TEST video directions are unknown; reverse-direction evidence cannot be "
    "fabricated from forced-manifest rev labels"
)
G9_3_NOT_APPLICABLE: dict[str, str] = {
    "status": "NOT_APPLICABLE",
    "reason": G9_3_REASON,
    "authority": "reporting",
    "independence_assumption": (
        "ts_common.TEST directions are unknown and are not reverse-flight evidence"
    ),
    "provenance_assumption": (
        "forced-manifest rev labels are not an attested direction on held-out videos"
    ),
}

_REVIEW_AUTHORITY = "review"
_REPORTING_AUTHORITY = "reporting"
_IDENTITY_INDEPENDENCE = (
    "each ts_common.TEST video is an independent held-out session and never enters BUILD"
)
_IDENTITY_PROVENANCE = (
    "result identities bind only to declared ts_common.TEST seq/rel/stem values"
)


def _test_seq_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for video in TEST:
        aliases[video.seq] = video.seq
        aliases[video.rel] = video.seq
        aliases[Path(video.rel).as_posix()] = video.seq
        aliases[Path(video.rel).name] = video.seq
        aliases[Path(video.rel).stem] = video.seq
    return aliases


def expected_test_ids() -> list[str]:
    return [video.seq for video in TEST]


def _not_applicable(value: object) -> bool:
    if value == "NOT_APPLICABLE":
        return True
    if not isinstance(value, Mapping):
        return False
    status = value.get("status", value.get("state"))
    reason = value.get("reason")
    return (
        status == "NOT_APPLICABLE"
        and isinstance(reason, str)
        and bool(reason.strip())
    )


def checks_pass(checks: Mapping[str, object] | None) -> bool:
    """Pass only explicit True checks; typed NOT_APPLICABLE is skipped, not success-by-truthiness."""
    if not checks:
        return False
    for value in checks.values():
        if _not_applicable(value):
            continue
        if value is True:
            continue
        return False
    return True


def heldout_contract_from_manifest(path: Path) -> list[dict[str, str | int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    heldout = payload.get("test")
    expected = expected_test_ids()
    if not isinstance(heldout, list) or len(heldout) != len(expected):
        raise ValueError(
            f"held-out contract requires exactly {len(expected)} TEST video(s)"
        )
    aliases = _test_seq_aliases()
    contracts: list[dict[str, str | int]] = []
    seen: set[str] = set()
    for item in heldout:
        if not isinstance(item, Mapping):
            raise ValueError("held-out contract entries must be objects")
        raw_id = str(item.get("seq") or item.get("video_id") or "").strip()
        video_id = aliases.get(raw_id)
        if video_id is None:
            raise ValueError(f"held-out video {raw_id!r} is not in the declared TEST set")
        if video_id in seen:
            raise ValueError(f"duplicate held-out TEST identity: {video_id}")
        digest = str(item.get("source_sha256") or item.get("sha256") or "")
        if SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError("held-out video requires a lowercase SHA-256 digest")
        probed = item.get("probed", {})
        total_frames = int(probed.get("nb_frames", 0) or 0) if isinstance(probed, Mapping) else 0
        if total_frames <= 0:
            raise ValueError("held-out video requires a positive probed frame count")
        seen.add(video_id)
        contracts.append(
            {
                "video_id": video_id,
                "video_sha256": digest,
                "total_frames": total_frames,
            }
        )
    if seen != set(expected):
        raise ValueError("held-out contract is missing one or more TEST identities")
    order = {seq: index for index, seq in enumerate(expected)}
    contracts.sort(key=lambda item: order[str(item["video_id"])])
    return contracts


def resolve_result_identity(result: Mapping[str, Any]) -> str | None:
    aliases = _test_seq_aliases()
    evaluation = result.get("evaluation")
    candidates = (
        result.get("video_id"),
        result.get("seq"),
        result.get("source_video_id"),
        result.get("video"),
        evaluation.get("source_video_id") if isinstance(evaluation, Mapping) else None,
    )
    for raw in candidates:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        if text in aliases:
            return aliases[text]
        name = Path(text).name
        stem = Path(text).stem
        if name in aliases:
            return aliases[name]
        if stem in aliases:
            return aliases[stem]
    return None


def _raw_identity(result: Mapping[str, Any]) -> str | None:
    evaluation = result.get("evaluation")
    for raw in (
        result.get("video_id"),
        result.get("seq"),
        result.get("source_video_id"),
        result.get("video"),
        evaluation.get("source_video_id") if isinstance(evaluation, Mapping) else None,
    ):
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            return text
    return None


def _finite_nonneg_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0 or int(number) != number:
        return None
    return int(number)


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _signed_margin(
    observed: float | None,
    required: float | None,
    *,
    lower_is_better: bool = False,
) -> float | None:
    if observed is None or required is None:
        return None
    if not math.isfinite(observed) or not math.isfinite(required):
        return None
    return required - observed if lower_is_better else observed - required


def _soft_check(
    *,
    observed: float | None,
    required: float | None,
    passed: bool,
    authority: str,
    independence_assumption: str,
    provenance_assumption: str,
    lower_is_better: bool = False,
    status: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "observed": observed,
        "required": required,
        "signed_margin": _signed_margin(
            observed, required, lower_is_better=lower_is_better
        ),
        "status": status
        or ("PASS" if passed else "FAIL"),
        "authority": authority,
        "independence_assumption": independence_assumption,
        "provenance_assumption": provenance_assumption,
    }
    if reason is not None:
        record["reason"] = reason
    return record


def _parse_run(result: Mapping[str, Any], identity: str | None) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    frames = _finite_nonneg_int(result.get("frames"))
    localized = _finite_nonneg_int(result.get("localized"))
    rate = _finite_float(result.get("rate"))
    if frames is None:
        errors.append("frames must be a finite nonnegative integer")
    elif frames <= 0:
        errors.append("frames must be > 0")
    if localized is None:
        errors.append("localized must be a finite nonnegative integer")
    if frames is not None and localized is not None and localized > frames:
        errors.append("localized must be <= frames")
    recomputed_rate: float | None = None
    if frames is not None and frames > 0 and localized is not None and localized <= frames:
        recomputed_rate = localized / frames
    if rate is None:
        errors.append("rate must be a finite number")
    elif recomputed_rate is None or rate != recomputed_rate:
        errors.append("supplied rate must equal localized/frames")

    rejections = result.get("rejections")
    if isinstance(rejections, Mapping) and "jump" in rejections:
        rejected_jumps = _finite_nonneg_int(rejections.get("jump"))
    else:
        rejected_jumps = None
    jumps = result.get("jumps_gt_10x_median")
    jumps_int = _finite_nonneg_int(jumps) if jumps is not None else None
    median = _finite_float(result.get("step_median"))
    p95 = _finite_float(result.get("step_p95"))
    inliers = _finite_float(result.get("inliers_p05"))
    rejected_jump_fraction = (
        rejected_jumps / frames
        if rejected_jumps is not None and frames is not None and frames > 0
        else None
    )
    continuity_ok = bool(
        median is not None
        and median > 0
        and p95 is not None
        and p95 <= G9_2_P95_MEDIAN_FACTOR * median
        and jumps_int == 0
    )
    run = {
        "video": identity,
        "raw_video": _raw_identity(result),
        "frames": frames,
        "localized": localized,
        "rate": rate,
        "recomputed_rate": recomputed_rate,
        "inliers_p05": inliers,
        "continuity_ok": continuity_ok,
        "step_median": median,
        "step_p95": p95,
        "jumps_gt_10x_median": jumps_int,
        "rejected_jumps": rejected_jumps,
        "rejected_jump_fraction": rejected_jump_fraction,
        "reference_sequence_counts": result.get("reference_sequence_counts", {}),
        "counts_valid": not errors,
    }
    return run, errors


def _contract_by_id(
    heldout_contract: Iterable[Mapping[str, Any]] | None,
) -> dict[str, Mapping[str, Any]] | None:
    if heldout_contract is None:
        return None
    bound: dict[str, Mapping[str, Any]] = {}
    for item in heldout_contract:
        video_id = str(item.get("video_id") or "")
        if video_id:
            bound[video_id] = item
    return bound


def evaluate_results(
    results: list[dict],
    *,
    reverse_sequences: set[str] | None = None,
    heldout_contract: list[dict] | None = None,
) -> dict:
    expected_ids = expected_test_ids()
    evidence: list[dict[str, Any]] = []
    observed_ids: list[str] = []
    unknown_ids: list[str] = []
    hard_failures: list[str] = []
    geometry_ok_by_id: dict[str, bool] = {}

    for result in results:
        identity = resolve_result_identity(result)
        run, errors = _parse_run(result, identity)
        evidence.append(run)
        raw = run["raw_video"]
        if identity is None:
            unknown_ids.append(raw if isinstance(raw, str) else "<missing>")
            hard_failures.append("result is not bound to a declared TEST identity")
            continue
        observed_ids.append(identity)
        geometry_ok_by_id[identity] = geometry_ok_by_id.get(identity, True) and not errors
        hard_failures.extend(f"{identity}: {error}" for error in errors)

    counts = Counter(observed_ids)
    duplicates = sorted(seq for seq, count in counts.items() if count > 1)
    missing_ids = [seq for seq in expected_ids if counts[seq] == 0]
    if duplicates:
        hard_failures.append(f"duplicate TEST identities: {duplicates}")
    if missing_ids:
        hard_failures.append(f"missing TEST identities: {missing_ids}")
    if unknown_ids:
        hard_failures.append(f"non-TEST identities: {unknown_ids}")
    if len(results) != len(expected_ids):
        hard_failures.append(
            f"expected exactly one result per TEST video ({len(expected_ids)}), got {len(results)}"
        )

    contract_map = _contract_by_id(heldout_contract)
    corpus_bindings: dict[str, Any] = {}
    if heldout_contract is not None:
        if contract_map is None or set(contract_map) != set(expected_ids):
            hard_failures.append("held-out contract does not match declared TEST identities")
        else:
            runs_by_id: dict[str, list[dict[str, Any]]] = {}
            for run in evidence:
                seq = run["video"]
                if isinstance(seq, str) and seq:
                    runs_by_id.setdefault(seq, []).append(run)
            for seq in expected_ids:
                item = contract_map[seq]
                digest = str(item.get("video_sha256") or "")
                total_frames = _finite_nonneg_int(item.get("total_frames"))
                digest_ok = SHA256_PATTERN.fullmatch(digest) is not None
                frames_ok = total_frames is not None and total_frames > 0
                matching = runs_by_id.get(seq, [])
                observed_frames = matching[0]["frames"] if len(matching) == 1 else None
                frames_match = (
                    frames_ok
                    and observed_frames is not None
                    and observed_frames == total_frames
                )
                if not digest_ok:
                    hard_failures.append(f"{seq}: contract SHA-256 is not a lowercase digest")
                if not frames_ok:
                    hard_failures.append(f"{seq}: contract frame count must be > 0")
                elif not frames_match:
                    hard_failures.append(
                        f"{seq}: result frames {observed_frames} != contract frames {total_frames}"
                    )
                corpus_bindings[seq] = {
                    "video_sha256": digest if digest_ok else None,
                    "total_frames": total_frames,
                    "result_frames": observed_frames,
                    "bound": digest_ok and frames_match,
                    "authority": _REVIEW_AUTHORITY,
                }

    identity_complete = (
        not unknown_ids
        and not duplicates
        and not missing_ids
        and len(results) == len(expected_ids)
        and all(seq in expected_ids for seq in observed_ids)
    )
    geometry_valid = identity_complete and all(
        geometry_ok_by_id.get(seq, False) for seq in expected_ids
    )
    contract_valid = heldout_contract is None or (
        bool(corpus_bindings)
        and all(record.get("bound") for record in corpus_bindings.values())
    )
    hard_valid = geometry_valid and contract_valid
    independent_session_count = len(
        {
            seq
            for seq in expected_ids
            if geometry_ok_by_id.get(seq, False)
        }
    )

    rates = [item["rate"] for item in evidence if item["rate"] is not None]
    inliers = [item["inliers_p05"] for item in evidence if item["inliers_p05"] is not None]
    jump_fracs = [
        item["rejected_jump_fraction"]
        for item in evidence
        if item["rejected_jump_fraction"] is not None
    ]
    continuous = [item["continuity_ok"] for item in evidence]
    missing_soft_metrics = any(
        item["inliers_p05"] is None
        or item["step_median"] is None
        or item["step_p95"] is None
        or item["jumps_gt_10x_median"] is None
        or item["rejected_jumps"] is None
        or item["rejected_jump_fraction"] is None
        for item in evidence
    )

    all_rates = hard_valid and all(
        item["rate"] is not None and item["rate"] >= G9_1_MIN_RATE for item in evidence
    )
    all_continuous = hard_valid and all(item["continuity_ok"] for item in evidence)
    inliers_supported = hard_valid and all(
        item["inliers_p05"] is not None and item["inliers_p05"] >= G9_4_MIN_INLIERS_P05
        for item in evidence
    )
    no_ghost_teleports = hard_valid and all(
        item["jumps_gt_10x_median"] == 0
        and item["rejected_jump_fraction"] is not None
        and item["rejected_jump_fraction"] <= G9_5_MAX_REJECTED_JUMP_FRACTION
        for item in evidence
    )

    checks: dict[str, Any] = {
        "G9.1": all_rates,
        "G9.2": all_continuous,
        "G9.3": dict(G9_3_NOT_APPLICABLE),
        "G9.4": inliers_supported,
        "G9.5": no_ghost_teleports,
    }
    soft_checks = {
        "G9.1": _soft_check(
            observed=min(rates) if rates else None,
            required=G9_1_MIN_RATE,
            passed=all_rates,
            authority=_REVIEW_AUTHORITY,
            independence_assumption=_IDENTITY_INDEPENDENCE,
            provenance_assumption="localization rate is taken from each bound TEST result",
        ),
        "G9.2": _soft_check(
            observed=(sum(1 for item in continuous if item) / len(continuous))
            if continuous
            else None,
            required=1.0,
            passed=all_continuous,
            authority=_REVIEW_AUTHORITY,
            independence_assumption=_IDENTITY_INDEPENDENCE,
            provenance_assumption="continuity uses attested median, p95, and jump counts only",
        ),
        "G9.3": _soft_check(
            observed=None,
            required=None,
            passed=False,
            authority=_REPORTING_AUTHORITY,
            independence_assumption=G9_3_NOT_APPLICABLE["independence_assumption"],
            provenance_assumption=G9_3_NOT_APPLICABLE["provenance_assumption"],
            status="NOT_APPLICABLE",
            reason=G9_3_REASON,
        ),
        "G9.4": _soft_check(
            observed=min(inliers) if inliers else None,
            required=G9_4_MIN_INLIERS_P05,
            passed=inliers_supported,
            authority=_REVIEW_AUTHORITY,
            independence_assumption=_IDENTITY_INDEPENDENCE,
            provenance_assumption="inliers_p05 is consumed only from bound TEST results",
        ),
        "G9.5": _soft_check(
            observed=max(jump_fracs) if jump_fracs else None,
            required=G9_5_MAX_REJECTED_JUMP_FRACTION,
            passed=no_ghost_teleports,
            authority=_REVIEW_AUTHORITY,
            independence_assumption=_IDENTITY_INDEPENDENCE,
            provenance_assumption="jump counts are fail-closed; missing jumps are not treated as zero",
            lower_is_better=True,
        ),
    }

    if not hard_valid:
        evidence_status = "INSUFFICIENT_EVIDENCE"
    elif missing_soft_metrics:
        evidence_status = "INSUFFICIENT_EVIDENCE"
    elif not checks_pass(checks):
        evidence_status = "QUALITY_SHORTFALL"
    else:
        evidence_status = "PASS"

    identity_receipt = {
        "authority": _REVIEW_AUTHORITY,
        "source": "ts_common.TEST",
        "expected_ids": expected_ids,
        "observed_ids": observed_ids,
        "duplicates": duplicates,
        "unknown_ids": unknown_ids,
        "missing_ids": missing_ids,
        "complete": identity_complete,
        "independent_session_count": independent_session_count,
        "corpus_bindings": corpus_bindings,
        "independence_assumption": _IDENTITY_INDEPENDENCE,
        "provenance_assumption": _IDENTITY_PROVENANCE,
    }
    reverse_diagnostic = {
        "authority": _REPORTING_AUTHORITY,
        "status": "NOT_APPLICABLE",
        "reason": G9_3_REASON,
        "forced_reverse_sequences": sorted(reverse_sequences or ()),
        "independence_assumption": G9_3_NOT_APPLICABLE["independence_assumption"],
        "provenance_assumption": G9_3_NOT_APPLICABLE["provenance_assumption"],
    }
    status = "PASS" if hard_valid and checks_pass(checks) else "FAIL"
    return {
        "stage": "S9_heldout_localization",
        "status": status,
        "ok": status == "PASS",
        "hard_status": "VALID" if hard_valid else "HARD_FAIL",
        "evidence_status": evidence_status,
        "checks": checks,
        "soft_checks": soft_checks,
        "runs": evidence,
        "independent_session_count": independent_session_count,
        "identity_receipt": identity_receipt,
        "g9_3_reason": G9_3_REASON,
        "reverse_reference_diagnostic": reverse_diagnostic,
        "hard_failures": hard_failures,
    }


def _optional_artifact(label: str, path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    return {label: hash_artifact(path)}


def _lineage_input_artifacts(args: argparse.Namespace) -> dict[str, dict]:
    artifacts: dict[str, dict] = {
        **{
            f"benchmark_result_{index}": hash_artifact(path)
            for index, path in enumerate(args.result)
        },
        "forced_manifest": hash_artifact(args.forced_manifest),
        **_optional_artifact("corpus_manifest", args.corpus_manifest),
        **_optional_artifact("edm_bundle", args.edm_bundle),
        **_optional_artifact("tracking_bundle", args.tracking_bundle),
        **_optional_artifact("package_config", args.package_config),
        **_optional_artifact("package_bundle", args.package_bundle),
        **_optional_artifact("package_edm_bundle", args.package_bundle),
    }
    if args.package_bundle is None and args.edm_bundle is not None:
        edm_record = hash_artifact(args.edm_bundle)
        artifacts["package_edm_bundle"] = {
            **edm_record,
            "binding": "transitive_edm_identity",
        }
        artifacts["package_bundle"] = {
            **edm_record,
            "binding": "transitive_edm_identity",
        }
    return artifacts

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--forced-manifest", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=False)
    parser.add_argument("--edm-bundle", type=Path, required=False)
    parser.add_argument("--tracking-bundle", type=Path, required=False)
    parser.add_argument("--package-config", type=Path, required=False)
    parser.add_argument("--package-bundle", type=Path, required=False)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.result]
    forced = json.loads(args.forced_manifest.read_text(encoding="utf-8"))
    contract = (
        heldout_contract_from_manifest(args.corpus_manifest)
        if args.corpus_manifest is not None
        else None
    )
    report = evaluate_results(
        results,
        reverse_sequences=set(forced.get("rev") or []),
        heldout_contract=contract,
    )
    input_artifacts = _lineage_input_artifacts(args)
    bound = report["identity_receipt"]["observed_ids"]
    if len(bound) == len(args.result) and len(set(bound)) == len(bound):
        for seq, path in zip(bound, args.result):
            input_artifacts[f"benchmark_result/{seq}"] = hash_artifact(path)
    report["provenance"] = {
        "script": hash_artifact(Path(__file__)),
        "sources": {},
        "input_artifacts": input_artifacts,
        "predecessor_gates": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    if report["status"] != "PASS":
        raise SystemExit("S9 gate failed")


if __name__ == "__main__":
    main()
