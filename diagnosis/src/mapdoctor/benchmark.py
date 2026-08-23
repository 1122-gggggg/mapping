from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mapdoctor.config import LocalizationThresholds


@dataclass(frozen=True)
class QueryLocalizationResult:
    query: str
    success: bool
    inliers: int
    inlier_ratio: float
    reproj_p90_px: float | None
    hull_coverage: float
    grid4_occupancy: int
    positive_depth_ratio: float
    pose_consensus: float
    x: float | None = None
    y: float | None = None
    z: float | None = None
    position_error_m: float | None = None
    rotation_error_deg: float | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if isinstance(self.inliers, bool) or not isinstance(self.inliers, int) or self.inliers < 0:
            raise ValueError(f"{self.query}: inliers must be a non-negative integer")
        for name, value in (
            ("inlier_ratio", self.inlier_ratio),
            ("hull_coverage", self.hull_coverage),
            ("positive_depth_ratio", self.positive_depth_ratio),
            ("pose_consensus", self.pose_consensus),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{self.query}: {name} must be finite and between 0 and 1")
        if self.reproj_p90_px is not None and (
            not math.isfinite(self.reproj_p90_px) or self.reproj_p90_px < 0
        ):
            raise ValueError(f"{self.query}: reproj_p90_px must be finite and >= 0 when present")
        if (
            isinstance(self.grid4_occupancy, bool)
            or not isinstance(self.grid4_occupancy, int)
            or not 0 <= self.grid4_occupancy <= 16
        ):
            raise ValueError(f"{self.query}: grid4_occupancy must be an integer between 0 and 16")
        if self.position_error_m is not None and (
            not math.isfinite(self.position_error_m) or self.position_error_m < 0
        ):
            raise ValueError(f"{self.query}: position_error_m must be finite and >= 0")
        if self.rotation_error_deg is not None and (
            not math.isfinite(self.rotation_error_deg) or self.rotation_error_deg < 0
        ):
            raise ValueError(f"{self.query}: rotation_error_deg must be finite and >= 0")
        for name, value in (("x", self.x), ("y", self.y), ("z", self.z)):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{self.query}: {name} must be finite")

    def failures(self, thresholds: LocalizationThresholds) -> list[str]:
        reasons: list[str] = []
        if not self.success:
            reasons.append("localization_failed")
        if self.inliers < thresholds.min_inliers:
            reasons.append("low_inliers")
        if self.inlier_ratio < thresholds.min_inlier_ratio:
            reasons.append("low_inlier_ratio")
        if self.reproj_p90_px is None:
            reasons.append("missing_reprojection_error")
        elif self.reproj_p90_px > thresholds.max_reprojection_p90_px:
            reasons.append("high_reprojection_error")
        if self.hull_coverage < thresholds.min_hull_coverage:
            reasons.append("low_inlier_hull_coverage")
        if self.grid4_occupancy < thresholds.min_grid4_occupancy:
            reasons.append("low_grid_occupancy")
        if self.positive_depth_ratio < thresholds.min_positive_depth_ratio:
            reasons.append("low_positive_depth_ratio")
        if self.pose_consensus < thresholds.min_pose_consensus:
            reasons.append("low_pose_consensus")
        return reasons

    def passes(self, thresholds: LocalizationThresholds) -> bool:
        return not self.failures(thresholds)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkSummary:
    total_queries: int
    raw_success_rate: float
    strict_success_rate: float
    median_inliers: float | None
    p10_inliers: float | None
    reprojection_p90_across_queries_px: float | None
    failures: list[dict[str, Any]]
    weak_regions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "pass", "success"}:
        return True
    if text in {"0", "false", "no", "fail", "failure"}:
        return False
    raise ValueError(f"Cannot parse boolean: {value!r}")


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Expected a finite number, got {value!r}")
    return number


def _required_float(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _required_int(value: Any, field: str) -> int:
    """Parse an integer field without silently truncating malformed exports."""
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    number = _required_float(value, field)
    if not number.is_integer():
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return int(number)


def _row(row: dict[str, Any]) -> QueryLocalizationResult:
    required = {
        "query",
        "success",
        "inliers",
        "inlier_ratio",
        "reproj_p90_px",
        "hull_coverage",
        "grid4_occupancy",
        "positive_depth_ratio",
        "pose_consensus",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"Missing localization fields: {', '.join(sorted(missing))}")
    return QueryLocalizationResult(
        query=str(row["query"]),
        success=_parse_bool(row["success"]),
        inliers=_required_int(row["inliers"], "inliers"),
        inlier_ratio=_required_float(row["inlier_ratio"], "inlier_ratio"),
        reproj_p90_px=_optional_float(row["reproj_p90_px"]),
        hull_coverage=_required_float(row["hull_coverage"], "hull_coverage"),
        grid4_occupancy=_required_int(row["grid4_occupancy"], "grid4_occupancy"),
        positive_depth_ratio=_required_float(row["positive_depth_ratio"], "positive_depth_ratio"),
        pose_consensus=_required_float(row["pose_consensus"], "pose_consensus"),
        x=_optional_float(row.get("x")),
        y=_optional_float(row.get("y")),
        z=_optional_float(row.get("z")),
        position_error_m=_optional_float(row.get("position_error_m")),
        rotation_error_deg=_optional_float(row.get("rotation_error_deg")),
    )


def load_localization_results(path: str | Path) -> list[QueryLocalizationResult]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif source.suffix.lower() == ".json":
        rows = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("Localization JSON must be a list")
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError("Every localization JSON item must be an object")
    else:
        raise ValueError("Localization results must be .csv or .json")
    results = [_row(row) for row in rows]
    names = [result.query for result in results]
    if len(names) != len(set(names)):
        raise ValueError("Localization query names must be unique")
    return results


def summarize_benchmark(
    results: list[QueryLocalizationResult],
    thresholds: LocalizationThresholds,
    cell_size: float = 5.0,
) -> BenchmarkSummary:
    if not results:
        raise ValueError("Localization benchmark contains no queries")
    if not math.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("region cell size must be finite and > 0")
    strict = [result.passes(thresholds) for result in results]
    failures = [
        {"query": result.query, "reasons": result.failures(thresholds), **result.to_dict()}
        for result in results
        if not result.passes(thresholds)
    ]
    cells: dict[tuple[int, int, int], list[QueryLocalizationResult]] = {}
    for result in results:
        if result.x is not None and result.y is not None and result.z is not None:
            key = (
                math.floor(result.x / cell_size),
                math.floor(result.y / cell_size),
                math.floor(result.z / cell_size),
            )
            cells.setdefault(key, []).append(result)
    weak_regions = []
    for key, members in cells.items():
        rate = sum(result.passes(thresholds) for result in members) / len(members)
        if rate < 1.0:
            weak_regions.append(
                {
                    "cell": list(key),
                    "queries": len(members),
                    "strict_success_rate": rate,
                    "failed_queries": [result.query for result in members if not result.passes(thresholds)],
                }
            )
    inliers = [float(result.inliers) for result in results]
    reprojection = [result.reproj_p90_px for result in results if result.reproj_p90_px is not None]
    return BenchmarkSummary(
        total_queries=len(results),
        raw_success_rate=sum(result.success for result in results) / len(results),
        strict_success_rate=sum(strict) / len(results),
        median_inliers=_percentile(inliers, 0.5),
        p10_inliers=_percentile(inliers, 0.1),
        reprojection_p90_across_queries_px=_percentile(reprojection, 0.9),
        failures=failures,
        weak_regions=sorted(weak_regions, key=lambda item: item["strict_success_rate"]),
    )
