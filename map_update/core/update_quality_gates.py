#!/usr/bin/env python3
"""Small, testable quality gates for incremental map updates."""
from __future__ import annotations

import math


_REVIEW_AUTHORITY = "reporting/review"
_INDEPENDENCE_ASSUMPTION = (
    "Each check is independent of the others and uses only the caller-supplied "
    "bridge summary; it does not attest independent bridge groups, exact-pair "
    "geometry, or held-out queries."
)
_PROVENANCE_ASSUMPTION = (
    "Values are trusted as already-derived bridge summary statistics; this "
    "module does not recompute correspondences, Sim3, or pair geometry."
)


def parse_warning_set(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            parts.extend(str(item).replace(";", ",").split(","))
    else:
        parts = str(value).replace(";", ",").split(",")
    out = []
    seen = set()
    for part in parts:
        text = part.strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def matched_warnings(classify_warnings, quarantine_warnings) -> list[str]:
    actual = set(parse_warning_set(classify_warnings))
    if not actual:
        return []
    return [warning for warning in parse_warning_set(quarantine_warnings) if warning in actual]


def _finite_number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _nonneg_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    number = _finite_number(value)
    if number is None:
        return None
    as_int = int(number)
    if number != as_int or as_int < 0:
        return None
    return as_int


def _check_record(
    *,
    name: str,
    value,
    threshold,
    finite: bool,
    enabled: bool,
    passed: bool,
    reason: str,
    evidence_status: str,
    hard_status: str,
) -> dict:
    signed_margin = None
    if finite and value is not None and threshold is not None:
        signed_margin = float(value) - float(threshold)
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "direction": "gte",
        "signed_margin": signed_margin,
        "finite": finite,
        "enabled": enabled,
        "passed": passed,
        "reason": reason,
        "hard_status": hard_status,
        "evidence_status": evidence_status,
        "authority": _REVIEW_AUTHORITY,
        "independence_assumption": _INDEPENDENCE_ASSUMPTION,
        "provenance_assumption": _PROVENANCE_ASSUMPTION,
    }


def _ratio_or_area_check(*, name: str, value, threshold, invalid_reason: str, shortfall_reason: str) -> dict:
    parsed_value = _finite_number(value)
    parsed_threshold = _finite_number(threshold)
    finite = parsed_value is not None and parsed_threshold is not None
    if not finite:
        return _check_record(
            name=name,
            value=parsed_value,
            threshold=parsed_threshold,
            finite=False,
            enabled=True,
            passed=False,
            reason=invalid_reason,
            evidence_status="INSUFFICIENT_EVIDENCE",
            hard_status="HARD_FAIL",
        )
    passed = not (parsed_value < parsed_threshold)
    if passed:
        evidence_status = "PASS"
        reason = ""
    else:
        evidence_status = "QUALITY_SHORTFALL"
        reason = shortfall_reason
    return _check_record(
        name=name,
        value=parsed_value,
        threshold=parsed_threshold,
        finite=True,
        enabled=True,
        passed=passed,
        reason=reason,
        evidence_status=evidence_status,
        hard_status="VALID",
    )


def _count_gate_check(
    *,
    name: str,
    value,
    threshold,
    invalid_reason: str,
    shortfall_reason: str,
) -> dict:
    parsed_value = _nonneg_int(value)
    parsed_threshold = _nonneg_int(threshold)
    finite = parsed_value is not None and parsed_threshold is not None
    enabled = parsed_threshold is None or parsed_threshold > 0
    if not finite:
        return _check_record(
            name=name,
            value=parsed_value,
            threshold=parsed_threshold,
            finite=False,
            enabled=enabled,
            passed=False,
            reason=invalid_reason,
            evidence_status="INSUFFICIENT_EVIDENCE",
            hard_status="HARD_FAIL",
        )
    if not enabled:
        return _check_record(
            name=name,
            value=parsed_value,
            threshold=parsed_threshold,
            finite=True,
            enabled=False,
            passed=True,
            reason="",
            evidence_status="PASS",
            hard_status="VALID",
        )
    passed = not (parsed_value < parsed_threshold)
    if passed:
        evidence_status = "PASS"
        reason = ""
    else:
        evidence_status = "QUALITY_SHORTFALL"
        reason = shortfall_reason
    return _check_record(
        name=name,
        value=parsed_value,
        threshold=parsed_threshold,
        finite=True,
        enabled=True,
        passed=passed,
        reason=reason,
        evidence_status=evidence_status,
        hard_status="VALID",
    )


def _geometry_ratio_check(
    *,
    bridge_geometry,
    total_bridges,
    min_geometry_ratio,
) -> dict:
    geometry = _nonneg_int(bridge_geometry)
    total = _nonneg_int(total_bridges)
    threshold = _finite_number(min_geometry_ratio)
    finite = geometry is not None and total is not None and threshold is not None
    enabled = threshold is None or threshold > 0
    value = None if geometry is None or total is None else float(geometry) / max(1.0, float(total))
    if not finite:
        return _check_record(
            name="bridge_geometry_ratio",
            value=value,
            threshold=threshold,
            finite=False,
            enabled=enabled,
            passed=False,
            reason="invalid_bridge_geometry_ratio",
            evidence_status="INSUFFICIENT_EVIDENCE",
            hard_status="HARD_FAIL",
        )
    if not enabled:
        return _check_record(
            name="bridge_geometry_ratio",
            value=value,
            threshold=threshold,
            finite=True,
            enabled=False,
            passed=True,
            reason="",
            evidence_status="PASS",
            hard_status="VALID",
        )
    passed = not (value < threshold)
    if passed:
        evidence_status = "PASS"
        reason = ""
    else:
        evidence_status = "QUALITY_SHORTFALL"
        reason = "low_bridge_geometry_ratio"
    return _check_record(
        name="bridge_geometry_ratio",
        value=value,
        threshold=threshold,
        finite=True,
        enabled=True,
        passed=passed,
        reason=reason,
        evidence_status=evidence_status,
        hard_status="VALID",
    )


def bridge_quality_checks(
    bridge_geometry: int,
    total_bridges: int,
    median_inlier_ratio: float,
    median_support_area: float,
    min_inlier_ratio: float,
    min_support_area: float,
    min_geometry: int = 0,
    min_geometry_ratio: float = 0.0,
) -> list[dict]:
    """Structured per-check receipts for bridge quality.

    Records are review diagnostics. Merge refuse/allow still uses
    ``bridge_quality_warnings``. Invalid or non-finite evidence is
    ``HARD_FAIL`` / ``INSUFFICIENT_EVIDENCE`` so it cannot silently pass.
    """
    return [
        _ratio_or_area_check(
            name="bridge_inlier_ratio",
            value=median_inlier_ratio,
            threshold=min_inlier_ratio,
            invalid_reason="invalid_bridge_inlier_ratio",
            shortfall_reason="low_bridge_inlier_ratio",
        ),
        _ratio_or_area_check(
            name="bridge_support_area",
            value=median_support_area,
            threshold=min_support_area,
            invalid_reason="invalid_bridge_support_area",
            shortfall_reason="low_bridge_support_area",
        ),
        _count_gate_check(
            name="bridge_geometry_count",
            value=bridge_geometry,
            threshold=min_geometry,
            invalid_reason="invalid_bridge_geometry_count",
            shortfall_reason="low_bridge_geometry_count",
        ),
        _geometry_ratio_check(
            bridge_geometry=bridge_geometry,
            total_bridges=total_bridges,
            min_geometry_ratio=min_geometry_ratio,
        ),
    ]


def bridge_quality_warnings(
    bridge_geometry: int,
    total_bridges: int,
    median_inlier_ratio: float,
    median_support_area: float,
    min_inlier_ratio: float,
    min_support_area: float,
    min_geometry: int = 0,
    min_geometry_ratio: float = 0.0,
) -> list[str]:
    return [
        check["reason"]
        for check in bridge_quality_checks(
            bridge_geometry=bridge_geometry,
            total_bridges=total_bridges,
            median_inlier_ratio=median_inlier_ratio,
            median_support_area=median_support_area,
            min_inlier_ratio=min_inlier_ratio,
            min_support_area=min_support_area,
            min_geometry=min_geometry,
            min_geometry_ratio=min_geometry_ratio,
        )
        if check["reason"]
    ]
