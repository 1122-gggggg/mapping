#!/usr/bin/env python3
"""Small, testable quality gates for incremental map updates."""
from __future__ import annotations


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
    warnings = []
    if float(median_inlier_ratio) < float(min_inlier_ratio):
        warnings.append("low_bridge_inlier_ratio")
    if float(median_support_area) < float(min_support_area):
        warnings.append("low_bridge_support_area")
    if int(min_geometry) > 0 and int(bridge_geometry) < int(min_geometry):
        warnings.append("low_bridge_geometry_count")
    if float(min_geometry_ratio) > 0:
        ratio = float(bridge_geometry) / max(1.0, float(total_bridges))
        if ratio < float(min_geometry_ratio):
            warnings.append("low_bridge_geometry_ratio")
    return warnings
