#!/usr/bin/env python3
"""Prepare a fresh GLOMAP-compatible clone without changing matching semantics."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path

from sqlite_db_evidence import SQLiteEvidenceError, database_evidence

CORE = (
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
LEGACY = "CREATE TABLE pose_priors (image_id INTEGER PRIMARY KEY NOT NULL, position BLOB, coordinate_system INTEGER NOT NULL, position_covariance BLOB, FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"


class LegacyMigrationError(ValueError):
    pass


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def cols(c):
    return [str(r[1]) for r in c.execute("pragma table_info(pose_priors)")]


def core(c):
    h = hashlib.sha3_256()
    for t in CORE:
        if c.execute(
            "select count(*) from sqlite_master where type='table' and name=?", (t,)
        ).fetchone()[0]:
            h.update(t.encode())
            for r in c.execute(f"select * from {t} order by rowid"):
                h.update(repr(r).encode())
    return h.hexdigest()


def publish_receipt(path: Path, payload: dict) -> None:
    """Exclusively publish a completed receipt after every migration gate passed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_legacy_clone(
    source: Path, clone: Path, *, expected_source_sha256: str
) -> dict:
    before_source = sha(source)
    if before_source != expected_source_sha256:
        raise LegacyMigrationError("source pre-SHA differs from pinned source SHA")
    before = sha(clone)
    if before != before_source:
        raise LegacyMigrationError("clone pre-SHA differs from source")
    wal = clone.with_name(clone.name + "-wal")
    shm = clone.with_name(clone.name + "-shm")
    if wal.is_file() and wal.stat().st_size:
        raise LegacyMigrationError("non-empty WAL blocks migration")
    sidecars_before = {
        "wal": {
            "exists": wal.exists(),
            "size": wal.stat().st_size if wal.exists() else 0,
        },
        "shm": {
            "exists": shm.exists(),
            "size": shm.stat().st_size if shm.exists() else 0,
        },
    }
    try:
        source_evidence = database_evidence(source)
        clone_evidence_before = database_evidence(clone)
    except SQLiteEvidenceError as error:
        raise LegacyMigrationError(str(error)) from error
    c = sqlite3.connect(clone)
    try:
        modern = [
            "pose_prior_id",
            "corr_data_id",
            "corr_sensor_id",
            "corr_sensor_type",
            "position",
            "position_covariance",
            "gravity",
            "coordinate_system",
        ]
        if cols(c) != modern:
            raise LegacyMigrationError("unexpected pose_priors schema")
        if c.execute("select count(*) from pose_priors").fetchone()[0]:
            raise LegacyMigrationError("cannot migrate non-empty pose_priors")
        cb = core(c)
        c.execute("BEGIN IMMEDIATE")
        try:
            c.execute("drop table pose_priors")
            c.execute(LEGACY)
            c.commit()
        except Exception:
            c.rollback()
            raise
        ca = core(c)
        qc = c.execute("pragma quick_check").fetchone()[0]
        ic = c.execute("pragma integrity_check").fetchone()[0]
    finally:
        c.close()
    if sha(source) != before_source:
        raise LegacyMigrationError("source changed")
    if cb != ca or qc != "ok" or ic != "ok":
        raise LegacyMigrationError("semantic migration gate failed")
    try:
        clone_evidence_after = database_evidence(clone)
    except SQLiteEvidenceError as error:
        raise LegacyMigrationError(str(error)) from error
    if clone_evidence_after["pose_priors"]["ddl"] != LEGACY:
        raise LegacyMigrationError("legacy pose_priors DDL differs from contract")
    if clone_evidence_after["pose_priors"]["count"] != 0:
        raise LegacyMigrationError("legacy pose_priors must remain empty")
    return {
        "source_sha256": before_source,
        "clone_sha256_before": before,
        "clone_sha256_after": sha(clone),
        "source_unchanged": True,
        "core_sha3_before": cb,
        "core_sha3_after": ca,
        "pose_priors_before": "modern_empty",
        "pose_priors_after": "legacy_empty",
        "quick_check": qc,
        "integrity_check": ic,
        "sidecars_before": sidecars_before,
        "sidecars_after": {
            "wal": {
                "exists": wal.exists(),
                "size": wal.stat().st_size if wal.exists() else 0,
            },
            "shm": {
                "exists": shm.exists(),
                "size": shm.stat().st_size if shm.exists() else 0,
            },
        },
        "pose_priors_columns_after": [
            "image_id",
            "position",
            "coordinate_system",
            "position_covariance",
        ],
        "source_evidence": source_evidence,
        "clone_evidence_before": clone_evidence_before,
        "clone_evidence_after": clone_evidence_after,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--clone", type=Path, required=True)
    p.add_argument("--expected-source-sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        print("receipt exists", file=os.sys.stderr, flush=True)
        return 2
    try:
        r = prepare_legacy_clone(
            a.source, a.clone, expected_source_sha256=a.expected_source_sha256
        )
        r["status"] = "PASS"
    except Exception as e:
        print(json.dumps({"status": "FAIL", "error": str(e)}, indent=2), flush=True)
        return 2
    publish_receipt(a.out, r)
    print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
