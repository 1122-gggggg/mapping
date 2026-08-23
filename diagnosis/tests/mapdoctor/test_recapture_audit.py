from __future__ import annotations

from mapdoctor.recapture.audit import AuditStatus, audit_cells
from mapdoctor.recapture.types import Availability, MetricValue, PoseDirectionCell


LOCALIZER = "arbitrary-localizer"


def _cell(metric_name: str) -> PoseDirectionCell:
    return PoseDirectionCell(
        cell_id="r:0",
        region_id="r",
        position=(0.0, 0.0, 0.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        localizer=LOCALIZER,
        metrics={metric_name: MetricValue(float("nan"), Availability.AVAILABLE)},
        map_producer="colmap",
    )


def test_nonfinite_diagnostic_metric_is_invalid() -> None:
    report = audit_cells((_cell("retrieval_recall_at_k"),), LOCALIZER)
    assert report.item("retrieval_recall_at_k").status == AuditStatus.INVALID


def test_nonfinite_authorization_metric_is_invalid() -> None:
    report = audit_cells((_cell("fim_lambda_min"),), LOCALIZER)
    assert report.item("fim_lambda_min").status == AuditStatus.INVALID
    assert not report.authorization_ready
