"""Read-only, fail-closed SQLite provenance evidence for GLOMAP candidates."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote


CORE_TABLES = (
    "rigs",
    "rig_sensors",
    "frames",
    "frame_data",
    "cameras",
    "images",
    "keypoints",
    "descriptors",
    "matches",
    "two_view_geometries",
    "sqlite_sequence",
)
DATABASE_TABLES = ("images", "matches", "two_view_geometries")
HEADER_MUTABLE_RANGES = ((24, 28), (92, 100))


class SQLiteEvidenceError(ValueError):
    """The on-disk SQLite state cannot be accepted as immutable evidence."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_hashes(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SQLiteEvidenceError(f"database is absent: {path}")
    raw = hashlib.sha256()
    normalized = hashlib.sha256()
    offset = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            raw.update(chunk)
            mutable = bytearray(chunk)
            for start, end in HEADER_MUTABLE_RANGES:
                left = max(start - offset, 0)
                right = min(end - offset, len(mutable))
                if left < right:
                    mutable[left:right] = b"\0" * (right - left)
            normalized.update(mutable)
            offset += len(chunk)
    return {
        "sha256": raw.hexdigest(),
        "normalized_header_sha256": normalized.hexdigest(),
        "size_bytes": offset,
        "zeroed_header_ranges": [list(value) for value in HEADER_MUTABLE_RANGES],
    }


def sidecar_snapshot(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, suffix in (("wal", "-wal"), ("shm", "-shm")):
        sidecar = path.with_name(path.name + suffix)
        exists = sidecar.is_file()
        result[name] = {
            "exists": exists,
            "size_bytes": sidecar.stat().st_size if exists else 0,
            "sha256": sha256_file(sidecar) if exists else None,
        }
    return result


def reject_nonempty_wal(sidecars: dict[str, dict[str, Any]], path: Path) -> None:
    if sidecars["wal"]["size_bytes"] > 0:
        raise SQLiteEvidenceError(
            f"non-empty SQLite WAL blocks immutable audit: {path}"
        )


def _immutable_uri(path: Path) -> str:
    return f"file:{quote(str(path.resolve()), safe='/')}?mode=ro&immutable=1"


def _has_table(connection: sqlite3.Connection, name: str) -> bool:
    return bool(
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
    )


def _core_sha3(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha3_256()
    for table in CORE_TABLES:
        if not _has_table(connection, table):
            continue
        digest.update(table.encode("utf-8"))
        columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
        digest.update(repr(columns).encode("utf-8"))
        for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid"):
            digest.update(repr(row).encode("utf-8"))
    return digest.hexdigest()


def _semantic_sha3sum(path: Path) -> tuple[str, dict[str, Any]]:
    executable = shutil.which("sqlite3")
    if executable is None:
        raise SQLiteEvidenceError("sqlite3 CLI with .sha3sum is required")
    resolved = Path(executable).resolve()
    version = subprocess.run(
        [str(resolved), "--version"], check=False, capture_output=True, text=True
    )
    if version.returncode:
        raise SQLiteEvidenceError(
            f"sqlite3 --version failed with exit status {version.returncode}"
        )
    argv = [
        str(resolved),
        _immutable_uri(path),
        ".sha3sum --sha3-256 --schema",
    ]
    result = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    if result.returncode or len(output.splitlines()) != 1 or len(output) != 64:
        detail = result.stderr.strip() or output or f"exit status {result.returncode}"
        raise SQLiteEvidenceError(
            f"sqlite .sha3sum --sha3-256 --schema failed: {detail}"
        )
    try:
        int(output, 16)
    except ValueError as error:
        raise SQLiteEvidenceError(
            "sqlite .sha3sum returned a non-hex digest"
        ) from error
    return output, {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "version": version.stdout.strip(),
        "argv": argv,
    }


def database_evidence(path: Path) -> dict[str, Any]:
    """Return SQLite evidence without writable opens or sidecar creation."""
    sidecars = sidecar_snapshot(path)
    reject_nonempty_wal(sidecars, path)
    raw = _raw_hashes(path)
    semantic, semantic_tool = _semantic_sha3sum(path)
    try:
        connection = sqlite3.connect(_immutable_uri(path), uri=True)
    except sqlite3.DatabaseError as error:
        raise SQLiteEvidenceError(
            f"cannot read SQLite database {path}: {error}"
        ) from error
    try:
        quick = [row[0] for row in connection.execute("PRAGMA quick_check")]
        integrity = [row[0] for row in connection.execute("PRAGMA integrity_check")]
        if quick != ["ok"]:
            raise SQLiteEvidenceError(f"SQLite quick_check failed for {path}: {quick}")
        if integrity != ["ok"]:
            raise SQLiteEvidenceError(
                f"SQLite integrity_check failed for {path}: {integrity}"
            )
        counts = {
            table: int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            for table in DATABASE_TABLES
        }
        pose_exists = _has_table(connection, "pose_priors")
        pose_count = (
            int(connection.execute("SELECT COUNT(*) FROM pose_priors").fetchone()[0])
            if pose_exists
            else 0
        )
        pose_ddl = (
            connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'pose_priors'"
            ).fetchone()[0]
            if pose_exists
            else None
        )
        return {
            "path": str(path.resolve()),
            "raw": raw,
            "sha256": raw["sha256"],
            "semantic_sha3_256_schema": semantic,
            "semantic_sha3sum_tool": semantic_tool,
            "core_sha3_256": _core_sha3(connection),
            "counts": counts,
            "quick_check": "ok",
            "integrity_check": "ok",
            "pose_priors": {
                "exists": pose_exists,
                "count": pose_count,
                "ddl": pose_ddl,
            },
            "sidecars": sidecars,
        }
    except sqlite3.DatabaseError as error:
        raise SQLiteEvidenceError(
            f"cannot read SQLite database {path}: {error}"
        ) from error
    finally:
        connection.close()
