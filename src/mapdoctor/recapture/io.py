from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from mapdoctor.report import write_recapture_bundle

from .audit import MetricAuditReport
from .types import Backend, PoseDirectionCell, RecaptureDecision


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_pose_cells(
    path: str | Path,
    *,
    backend: Backend | str = Backend.GENERIC,
) -> tuple[PoseDirectionCell, ...]:
    backend = backend if isinstance(backend, Backend) else Backend.coerce(backend)
    data = _read_json(path)
    raw = data.get("pose_cells", data) if isinstance(data, Mapping) else data
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("pose-cell input must be a list or an object containing pose_cells")
    if not all(isinstance(item, Mapping) for item in raw):
        raise ValueError("every pose-cell item must be an object")
    return tuple(PoseDirectionCell.from_dict(item, default_backend=backend) for item in raw)


def write_plan(
    decisions: Sequence[RecaptureDecision],
    output_dir: str | Path,
    *,
    audits: Mapping[str, MetricAuditReport] | None = None,
) -> dict[str, Path]:
    return write_recapture_bundle(decisions, audits or {}, output_dir)
