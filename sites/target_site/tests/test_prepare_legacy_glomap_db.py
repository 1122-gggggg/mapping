from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

TARGET = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET / "tools"))
from prepare_legacy_glomap_db import (  # noqa: E402
    LEGACY,
    LegacyMigrationError,
    prepare_legacy_clone,
    publish_receipt,
    sha,
)

MODERN = """CREATE TABLE pose_priors (pose_prior_id INTEGER PRIMARY KEY, corr_data_id INTEGER, corr_sensor_id INTEGER, corr_sensor_type INTEGER, position BLOB, position_covariance BLOB, gravity BLOB, coordinate_system INTEGER);"""


def db(path: Path, *, row: bool = False) -> None:
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT)")
    c.execute("INSERT INTO images VALUES (1,'a.jpg')")
    c.execute("CREATE TABLE matches (pair_id INTEGER PRIMARY KEY)")
    c.execute("INSERT INTO matches VALUES (3)")
    c.execute("CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY)")
    c.execute("INSERT INTO two_view_geometries VALUES (3)")
    c.execute(MODERN)
    if row:
        c.execute("INSERT INTO pose_priors VALUES (1,2,3,4,NULL,NULL,NULL,5)")
    c.commit()
    c.close()


def test_lossless_empty_modern_migration_preserves_source_and_core(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    clone = tmp_path / "clone.db"
    db(source)
    shutil.copyfile(source, clone)
    source_before = sha(source)
    receipt = prepare_legacy_clone(source, clone, expected_source_sha256=sha(source))
    assert (
        receipt["source_unchanged"]
        and receipt["core_sha3_before"] == receipt["core_sha3_after"]
    )
    assert sha(source) == source_before == receipt["source_sha256"]
    c = sqlite3.connect(clone)
    assert [r[1] for r in c.execute("pragma table_info(pose_priors)")] == [
        "image_id",
        "position",
        "coordinate_system",
        "position_covariance",
    ]
    assert c.execute("select count(*) from pose_priors").fetchone()[0] == 0
    assert (
        c.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'pose_priors'"
        ).fetchone()[0]
        == LEGACY
    )
    c.close()


def test_rejects_nonempty_modern_pose_priors(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    clone = tmp_path / "clone.db"
    db(source, row=True)
    shutil.copyfile(source, clone)
    with pytest.raises(LegacyMigrationError, match="non-empty"):
        prepare_legacy_clone(source, clone, expected_source_sha256=sha(source))


def test_rejects_mismatched_clone_before_writable_open(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    clone = tmp_path / "clone.db"
    db(source)
    db(clone, row=True)
    before = clone.read_bytes()
    with pytest.raises(LegacyMigrationError, match="clone pre-SHA"):
        prepare_legacy_clone(source, clone, expected_source_sha256=sha(source))
    assert clone.read_bytes() == before


def test_rejects_nonempty_pre_wal_without_touching_clone(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    clone = tmp_path / "clone.db"
    db(source)
    shutil.copyfile(source, clone)
    clone.with_name("clone.db-wal").write_bytes(b"journal")
    before = clone.read_bytes()
    with pytest.raises(LegacyMigrationError, match="non-empty WAL"):
        prepare_legacy_clone(source, clone, expected_source_sha256=sha(source))
    assert clone.read_bytes() == before


def test_clone_must_match_explicit_pinned_source_sha_before_sqlite_open(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    clone = tmp_path / "clone.db"
    db(source)
    shutil.copyfile(source, clone)
    before = clone.read_bytes()

    with pytest.raises(LegacyMigrationError, match="pinned source SHA"):
        prepare_legacy_clone(source, clone, expected_source_sha256="0" * 64)

    assert clone.read_bytes() == before


def test_success_receipt_records_sidecars_and_schema_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    clone = tmp_path / "clone.db"
    db(source)
    shutil.copyfile(source, clone)

    receipt = prepare_legacy_clone(source, clone, expected_source_sha256=sha(source))

    assert receipt["source_evidence"]["sidecars"]["wal"]["size_bytes"] == 0
    assert receipt["clone_evidence_after"]["sidecars"]["wal"]["size_bytes"] == 0
    assert receipt["clone_evidence_after"]["pose_priors"]["count"] == 0
    assert receipt["clone_evidence_after"]["pose_priors"]["ddl"] == LEGACY
    assert receipt["clone_evidence_after"]["quick_check"] == "ok"
    assert receipt["clone_evidence_after"]["integrity_check"] == "ok"


def test_receipt_publish_is_exclusive_and_never_overwrites(tmp_path: Path) -> None:
    out = tmp_path / "receipt.json"
    publish_receipt(out, {"status": "PASS"})
    before = out.read_bytes()

    with pytest.raises(FileExistsError):
        publish_receipt(out, {"status": "FAIL"})

    assert out.read_bytes() == before


def test_cli_existing_receipt_fails_before_any_clone_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    clone = tmp_path / "clone.db"
    out = tmp_path / "receipt.json"
    db(source)
    shutil.copyfile(source, clone)
    out.write_text('{"status":"PASS"}\n', encoding="utf-8")
    before = clone.read_bytes()
    script = TARGET / "tools" / "prepare_legacy_glomap_db.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source",
            str(source),
            "--clone",
            str(clone),
            "--expected-source-sha256",
            sha(source),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "receipt exists" in result.stderr
    assert clone.read_bytes() == before


def test_cli_error_never_publishes_a_complete_receipt(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    clone = tmp_path / "clone.db"
    out = tmp_path / "receipt.json"
    db(source, row=True)
    shutil.copyfile(source, clone)
    script = TARGET / "tools" / "prepare_legacy_glomap_db.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source",
            str(source),
            "--clone",
            str(clone),
            "--expected-source-sha256",
            sha(source),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert not out.exists()
