from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from ..models import ImageRecord
from ..states import ImageSource, QualityStatus

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
FRAME_PATTERN = re.compile(r"(\d+)(?=\.[^.]+$)")


def _stable_image_id(source: ImageSource, root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    digest = hashlib.sha1(f"{source.value}:{relative}".encode("utf-8")).hexdigest()[:16]
    return f"{source.value.lower()}:{digest}"


def _infer_frame_index(path: Path) -> int | None:
    match = FRAME_PATTERN.search(path.name)
    return int(match.group(1)) if match else None


def scan_image_root(root: str | Path, source: ImageSource) -> list[ImageRecord]:
    root_path = Path(root).resolve()
    if not root_path.exists():
        return []
    records: list[ImageRecord] = []
    for path in sorted(item for item in root_path.rglob("*") if item.is_file()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(root_path)
        session_id = relative.parts[0] if len(relative.parts) > 1 else root_path.name
        sequence_id = relative.parent.as_posix() if relative.parent.as_posix() != "." else session_id
        records.append(
            ImageRecord(
                image_id=_stable_image_id(source, root_path, path),
                path=path,
                source=source,
                session_id=session_id,
                sequence_id=sequence_id,
                frame_index=_infer_frame_index(path),
            )
        )
    return records


def build_image_manifest(
    historical_root: str | Path,
    validation_root: str | Path | None = None,
    current_root: str | Path | None = None,
) -> list[ImageRecord]:
    records = scan_image_root(historical_root, ImageSource.HISTORICAL_UPDATE)
    if validation_root:
        records.extend(scan_image_root(validation_root, ImageSource.CURRENT_VALIDATION))
    if current_root:
        records.extend(scan_image_root(current_root, ImageSource.CURRENT_MAP))
    return records


def _record_to_row(record: ImageRecord) -> dict[str, str | int | float | None]:
    return {
        "image_id": record.image_id,
        "path": str(record.path),
        "source": record.source.value,
        "session_id": record.session_id,
        "sequence_id": record.sequence_id,
        "frame_index": record.frame_index,
        "timestamp": record.timestamp,
        "camera_id": record.camera_id,
        "width": record.width,
        "height": record.height,
        "quality_status": record.quality_status.value if record.quality_status else None,
        "quality_metrics": json.dumps(record.quality_metrics, sort_keys=True),
        "metadata": json.dumps(record.metadata, sort_keys=True),
    }


def write_manifest(records: Iterable[ImageRecord], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [_record_to_row(record) for record in records]
    fields = list(rows[0]) if rows else [
        "image_id",
        "path",
        "source",
        "session_id",
        "sequence_id",
        "frame_index",
        "timestamp",
        "camera_id",
        "width",
        "height",
        "quality_status",
        "quality_metrics",
        "metadata",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: str | Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            quality = row.get("quality_status")
            records.append(
                ImageRecord(
                    image_id=row["image_id"],
                    path=Path(row["path"]),
                    source=ImageSource(row["source"]),
                    session_id=row["session_id"],
                    sequence_id=row.get("sequence_id") or None,
                    frame_index=int(row["frame_index"]) if row.get("frame_index") else None,
                    timestamp=float(row["timestamp"]) if row.get("timestamp") else None,
                    camera_id=row.get("camera_id") or None,
                    width=int(row["width"]) if row.get("width") else None,
                    height=int(row["height"]) if row.get("height") else None,
                    quality_status=QualityStatus(quality) if quality else None,
                    quality_metrics=json.loads(row.get("quality_metrics") or "{}"),
                    metadata=json.loads(row.get("metadata") or "{}"),
                )
            )
    return records
