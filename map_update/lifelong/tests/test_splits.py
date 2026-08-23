from pathlib import Path

from update_map.models import ImageRecord
from update_map.splits import audit_dataset_splits
from update_map.states import ImageSource, ValidationGrade


def _record(path: Path, source: ImageSource, session: str) -> ImageRecord:
    return ImageRecord(
        image_id=f"{source.value}:{path.name}",
        path=path,
        source=source,
        session_id=session,
        sequence_id=session,
    )


def test_independent_sessions_are_accepted(tmp_path: Path) -> None:
    current = tmp_path / "current.jpg"
    validation = tmp_path / "validation.jpg"
    current.write_bytes(b"current")
    validation.write_bytes(b"validation")
    audit = audit_dataset_splits(
        [
            _record(current, ImageSource.CURRENT_MAP, "flight_map"),
            _record(validation, ImageSource.CURRENT_VALIDATION, "flight_validation"),
        ],
        check_content_hashes=True,
    )
    assert audit.validation_grade == ValidationGrade.INDEPENDENT_CURRENT_SESSION
    assert not audit.critical_leakage


def test_duplicate_content_downgrades_validation(tmp_path: Path) -> None:
    current = tmp_path / "current.jpg"
    validation = tmp_path / "copied.jpg"
    current.write_bytes(b"same-frame")
    validation.write_bytes(b"same-frame")
    audit = audit_dataset_splits(
        [
            _record(current, ImageSource.CURRENT_MAP, "flight_map"),
            _record(validation, ImageSource.CURRENT_VALIDATION, "flight_validation"),
        ],
        check_content_hashes=True,
    )
    assert audit.validation_grade == ValidationGrade.PROVISIONAL_PROXY_VALIDATION
    assert audit.critical_leakage
    assert audit.duplicate_content_overlaps
