from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .models import ExperimentResult, ImageRecord, PoseEstimate, to_jsonable


def write_json(data: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(to_jsonable(data), indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )


def write_records_table(records: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame([to_jsonable(dict(item)) for item in records])
    if destination.suffix.lower() == ".parquet":
        try:
            dataframe.to_parquet(destination, index=False)
            return
        except (ImportError, ValueError):
            destination = destination.with_suffix(".csv")
    dataframe.to_csv(destination, index=False)


def pose_estimate_row(estimate: PoseEstimate) -> dict[str, Any]:
    quality = estimate.quality
    row: dict[str, Any] = {
        "query_id": estimate.query_id,
        "status": estimate.status.value,
        "supporting_references": ";".join(estimate.supporting_references),
        "num_raw_matches": quality.num_raw_matches,
        "num_lifted_matches": quality.num_lifted_matches,
        "num_unique_point3d": quality.num_unique_point3d,
        "num_inliers": quality.num_inliers,
        "inlier_ratio": quality.inlier_ratio,
        "reprojection_rmse": quality.reprojection_rmse,
        "reprojection_p90": quality.reprojection_p90,
        "convex_hull_ratio": quality.convex_hull_ratio,
        "grid_occupancy": quality.grid_occupancy,
        "positive_depth_ratio": quality.positive_depth_ratio,
        "independent_reference_support": quality.independent_reference_support,
        "pose_mode_count": quality.pose_mode_count,
        "passed": quality.passed,
        "failed_gates": ";".join(quality.failed_gates),
    }
    if estimate.pose is not None:
        row["camera_center_x"], row["camera_center_y"], row["camera_center_z"] = (
            estimate.pose.camera_center.tolist()
        )
        row["R_cw"] = json.dumps(estimate.pose.R_cw.tolist())
        row["t_cw"] = json.dumps(estimate.pose.t_cw.tolist())
    if quality.fim is not None:
        row.update(
            {
                "fim_condition_number": quality.fim.condition_number,
                "fim_logdet": quality.fim.logdet,
                "fim_trace_covariance": quality.fim.trace_covariance,
                "fim_eigenvalues": json.dumps(quality.fim.eigenvalues.tolist()),
                "pose_marginal_std": json.dumps(quality.fim.marginal_std.tolist()),
            }
        )
    return row


def generate_data_audit_markdown(
    records: Iterable[ImageRecord],
    base_map_summary: Mapping[str, Any],
    validation_grade: str,
) -> str:
    records_list = list(records)
    by_source: dict[str, int] = {}
    by_quality: dict[str, int] = {}
    sessions: set[str] = set()
    for record in records_list:
        by_source[record.source.value] = by_source.get(record.source.value, 0) + 1
        quality = record.quality_status.value if record.quality_status else "NOT_EVALUATED"
        by_quality[quality] = by_quality.get(quality, 0) + 1
        sessions.add(record.session_id)
    lines = [
        "# Data audit",
        "",
        f"- Validation grade: `{validation_grade}`",
        f"- Base-map cameras: {base_map_summary.get('cameras', 0)}",
        f"- Base-map images: {base_map_summary.get('images', 0)}",
        f"- Base-map points3D: {base_map_summary.get('points3d', 0)}",
        f"- Input image records: {len(records_list)}",
        f"- Sessions: {len(sessions)}",
        "",
        "## Source counts",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(by_source.items()))
    lines.extend(["", "## Quality counts", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(by_quality.items()))
    lines.extend(
        [
            "",
            "## Leakage warning",
            "",
            "Validation images must be separated by flight/session. Adjacent frames may not be randomly split across map construction and evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_experiment_summary(result: ExperimentResult) -> str:
    aggregate = result.aggregate
    lines = [
        f"# {result.experiment_id}",
        "",
        f"- Queries: {aggregate.get('query_count', len(result.query_results))}",
        f"- Success rate: {aggregate.get('success_rate', 0.0):.4f}",
        f"- Confident wrong poses: {aggregate.get('confident_wrong_pose_count', 0)}",
        f"- Median translation error: {aggregate.get('translation_median', 'N/A')}",
        f"- Median rotation error: {aggregate.get('rotation_median_deg', 'N/A')}",
        f"- p95 total latency: {aggregate.get('latency_p95_ms', 'N/A')}",
        f"- Active references: {len(result.references_used)}",
        "",
    ]
    return "\n".join(lines)
