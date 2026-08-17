from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .io.hashing import sha256_file
from .models import ImageRecord
from .states import ImageSource, ValidationGrade


@dataclass
class SplitAudit:
    validation_grade: ValidationGrade
    critical_leakage: bool
    exact_path_overlaps: list[str] = field(default_factory=list)
    duplicate_content_overlaps: list[dict[str, str]] = field(default_factory=list)
    current_validation_session_name_overlaps: list[str] = field(default_factory=list)
    current_validation_sequence_name_overlaps: list[str] = field(default_factory=list)
    historical_validation_path_overlaps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _paths(records: Iterable[ImageRecord], source: ImageSource) -> set[Path]:
    return {record.path.resolve() for record in records if record.source == source}


def _names(records: Iterable[ImageRecord], source: ImageSource, field_name: str) -> set[str]:
    values: set[str] = set()
    for record in records:
        if record.source != source:
            continue
        value = getattr(record, field_name)
        if value:
            values.add(str(value))
    return values


def _content_duplicates(
    current_records: list[ImageRecord], validation_records: list[ImageRecord]
) -> list[dict[str, str]]:
    by_hash: dict[str, Path] = {}
    for record in current_records:
        if record.path.is_file():
            by_hash.setdefault(sha256_file(record.path), record.path.resolve())
    duplicates: list[dict[str, str]] = []
    for record in validation_records:
        if not record.path.is_file():
            continue
        digest = sha256_file(record.path)
        current = by_hash.get(digest)
        if current is not None:
            duplicates.append(
                {
                    "sha256": digest,
                    "current_path": str(current),
                    "validation_path": str(record.path.resolve()),
                }
            )
    return duplicates


def audit_dataset_splits(
    records: Iterable[ImageRecord],
    check_content_hashes: bool = False,
) -> SplitAudit:
    """Audit map/update/validation separation without assuming folder names prove independence."""

    records_list = list(records)
    current = [record for record in records_list if record.source == ImageSource.CURRENT_MAP]
    validation = [
        record for record in records_list if record.source == ImageSource.CURRENT_VALIDATION
    ]
    historical = [
        record for record in records_list if record.source == ImageSource.HISTORICAL_UPDATE
    ]

    current_paths = _paths(current, ImageSource.CURRENT_MAP)
    validation_paths = _paths(validation, ImageSource.CURRENT_VALIDATION)
    historical_paths = _paths(historical, ImageSource.HISTORICAL_UPDATE)
    exact = sorted(str(path) for path in current_paths & validation_paths)
    historical_validation = sorted(str(path) for path in historical_paths & validation_paths)

    current_sessions = _names(current, ImageSource.CURRENT_MAP, "session_id")
    validation_sessions = _names(validation, ImageSource.CURRENT_VALIDATION, "session_id")
    current_sequences = _names(current, ImageSource.CURRENT_MAP, "sequence_id")
    validation_sequences = _names(validation, ImageSource.CURRENT_VALIDATION, "sequence_id")
    session_overlap = sorted(current_sessions & validation_sessions)
    sequence_overlap = sorted(current_sequences & validation_sequences)

    duplicate_content = _content_duplicates(current, validation) if check_content_hashes else []
    critical = bool(exact or duplicate_content or historical_validation)
    warnings: list[str] = []
    if not validation:
        grade = ValidationGrade.NO_VALIDATION
        warnings.append("No current validation images were supplied.")
    elif not current:
        grade = ValidationGrade.PROVISIONAL_PROXY_VALIDATION
        warnings.append(
            "Current map source images were not supplied, so session independence cannot be verified."
        )
    elif critical or session_overlap or sequence_overlap:
        grade = ValidationGrade.PROVISIONAL_PROXY_VALIDATION
        if session_overlap:
            warnings.append(
                "Current-map and validation roots share session names; verify that these are not the same flight."
            )
        if sequence_overlap:
            warnings.append(
                "Current-map and validation roots share sequence names; adjacent-frame leakage is possible."
            )
    else:
        grade = ValidationGrade.INDEPENDENT_CURRENT_SESSION

    if not check_content_hashes and current and validation:
        warnings.append(
            "Content-hash duplicate detection was not requested; exact copies under different paths may remain undetected."
        )

    return SplitAudit(
        validation_grade=grade,
        critical_leakage=critical,
        exact_path_overlaps=exact,
        duplicate_content_overlaps=duplicate_content,
        current_validation_session_name_overlaps=session_overlap,
        current_validation_sequence_name_overlaps=sequence_overlap,
        historical_validation_path_overlaps=historical_validation,
        warnings=warnings,
    )


def assert_no_critical_leakage(audit: SplitAudit) -> None:
    if audit.critical_leakage:
        raise ValueError(
            "Critical dataset leakage detected: exact path, duplicate content, or historical/validation overlap"
        )
