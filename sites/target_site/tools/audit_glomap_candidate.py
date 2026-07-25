#!/usr/bin/env python3
"""Fail-closed provenance and geometry audit for a DB-reuse GLOMAP candidate.

The tool has two deliberately separate operations:

* ``snapshot-db`` records the immutable pre-mapper source and reflink-clone state.
* ``audit`` verifies that neither database changed and that the raw (and optional
  finalized) COLMAP models remain fixed-intrinsics, finite, and localizable.

Both operations write a fresh JSON file atomically.  A validation failure is
reported in that JSON and exits with status 2; an existing output is never
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from sqlite_db_evidence import (
    SQLiteEvidenceError,
    database_evidence,
)


DATABASE_TABLES = ("images", "matches", "two_view_geometries")


class AuditFailure(ValueError):
    """A candidate violated a scientific or provenance invariant."""

    def __init__(self, message: str, report: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report = report


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 without modifying the artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sidecar_snapshot(path: Path) -> dict[str, dict[str, Any]]:
    """Record SQLite sidecars without treating an empty WAL or SHM as a mutation."""
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


def _reject_nonempty_wal(sidecars: dict[str, dict[str, Any]], path: Path) -> None:
    if sidecars["wal"]["size_bytes"] > 0:
        raise AuditFailure(f"non-empty SQLite WAL blocks immutable audit: {path}")


def _sqlite_counts_immutable(path: Path) -> dict[str, int]:
    """Read mapping-defining counts without SQLite creating WAL/SHM sidecars."""
    if not path.is_file():
        raise AuditFailure(f"database is absent: {path}")
    uri = f"file:{quote(str(path.resolve()))}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()
        if quick_check != ("ok",):
            raise AuditFailure(f"SQLite quick_check failed for {path}: {quick_check}")
        return {
            table: int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            for table in DATABASE_TABLES
        }
    except sqlite3.DatabaseError as error:
        raise AuditFailure(f"cannot read SQLite database {path}: {error}") from error
    finally:
        connection.close()


def sqlite_counts(path: Path) -> dict[str, int]:
    """Reject a nonempty WAL before opening SQLite through its immutable URI."""
    _reject_nonempty_wal(_sidecar_snapshot(path), path)
    return _sqlite_counts_immutable(path)


def database_snapshot(path: Path) -> dict[str, Any]:
    """Capture the hash and table counts needed to prove a DB remained immutable."""
    try:
        return database_evidence(path)
    except SQLiteEvidenceError as error:
        raise AuditFailure(
            str(error),
            {
                "path": str(path.resolve()),
                "sidecars": _sidecar_snapshot(path),
            },
        ) from error


def _same_snapshot(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left["sha256"] == right["sha256"] and left["counts"] == right["counts"]


def validate_database_contract(
    *,
    source: Path,
    candidate: Path,
    source_pre: dict[str, Any],
    candidate_pre: dict[str, Any],
    expected_source_sha256: str,
    expected_counts: dict[str, int],
) -> dict[str, Any]:
    """Verify source and clone are byte-identical to their pre-mapper states."""
    try:
        source_post = database_snapshot(source)
    except AuditFailure as error:
        raise AuditFailure(
            str(error),
            {
                "source": {"pre": source_pre, "post": error.report},
                "candidate": {"pre": candidate_pre, "post": None},
            },
        ) from error
    try:
        candidate_post = database_snapshot(candidate)
    except AuditFailure as error:
        raise AuditFailure(
            str(error),
            {
                "source": {"pre": source_pre, "post": source_post},
                "candidate": {"pre": candidate_pre, "post": error.report},
            },
        ) from error
    checks = {
        "source_pre_expected": (
            source_pre["sha256"] == expected_source_sha256
            and source_pre["counts"] == expected_counts
        ),
        "candidate_pre_matches_source": _same_snapshot(candidate_pre, source_pre),
        "source_unchanged": _same_snapshot(source_pre, source_post),
        "candidate_unchanged": _same_snapshot(candidate_pre, candidate_post),
        "source_post_expected": (
            source_post["sha256"] == expected_source_sha256
            and source_post["counts"] == expected_counts
        ),
    }
    report = {
        "source": {"pre": source_pre, "post": source_post},
        "candidate": {"pre": candidate_pre, "post": candidate_post},
        "expected": {"sha256": expected_source_sha256, "counts": expected_counts},
        "checks": checks,
    }
    errors: list[str] = []
    if not checks["source_pre_expected"]:
        errors.append("source database pre-state differs from the pinned contract")
    if not checks["candidate_pre_matches_source"]:
        errors.append("candidate database pre-state differs from source")
    if not checks["source_unchanged"]:
        errors.append("source database changed")
    if not checks["candidate_unchanged"]:
        errors.append("candidate database changed")
    if not checks["source_post_expected"]:
        errors.append("source database post-state differs from the pinned contract")
    if errors:
        raise AuditFailure("; ".join(errors), report)
    return report


def validate_mapper_post_contract(
    *,
    source: Path,
    candidate: Path,
    source_pre: dict[str, Any],
    candidate_pre: dict[str, Any],
    expected_source_sha256: str,
    expected_counts: dict[str, int],
) -> dict[str, Any]:
    """Prove that mapper execution left its legacy-ready database semantically fixed.

    SQLite is allowed to update only its page-zero change/version metadata at byte
    ranges 24:28 and 92:100.  The normalized raw digest enforces that exception;
    semantic and core-table SHA3 values independently reject logical mutation.
    """
    try:
        source_post = database_snapshot(source)
    except AuditFailure as error:
        raise AuditFailure(
            str(error),
            {
                "source": {"pre": source_pre, "post": error.report},
                "candidate": {"pre": candidate_pre, "post": None},
            },
        ) from error
    try:
        candidate_post = database_snapshot(candidate)
    except AuditFailure as error:
        raise AuditFailure(
            str(error),
            {
                "source": {"pre": source_pre, "post": source_post},
                "candidate": {"pre": candidate_pre, "post": error.report},
            },
        ) from error
    checks = {
        "source_pre_expected": (
            source_pre["sha256"] == expected_source_sha256
            and source_pre["counts"] == expected_counts
        ),
        "source_raw_exact": source_pre["raw"] == source_post["raw"],
        "source_semantic_sha3_exact": (
            source_pre["semantic_sha3_256_schema"]
            == source_post["semantic_sha3_256_schema"]
        ),
        "candidate_semantic_sha3_exact": (
            candidate_pre["semantic_sha3_256_schema"]
            == candidate_post["semantic_sha3_256_schema"]
        ),
        "candidate_core_sha3_exact": (
            candidate_pre["core_sha3_256"] == candidate_post["core_sha3_256"]
        ),
        "candidate_counts_exact": candidate_pre["counts"] == candidate_post["counts"],
        "candidate_pose_priors_empty": (
            candidate_post["pose_priors"]["exists"]
            and candidate_post["pose_priors"]["count"] == 0
        ),
        "candidate_quick_check": candidate_post["quick_check"] == "ok",
        "candidate_integrity_check": candidate_post["integrity_check"] == "ok",
        "candidate_wal_absent_or_zero": (
            candidate_post["sidecars"]["wal"]["size_bytes"] == 0
        ),
        "candidate_raw_header_only": (
            candidate_pre["raw"]["size_bytes"] == candidate_post["raw"]["size_bytes"]
            and candidate_pre["raw"]["normalized_header_sha256"]
            == candidate_post["raw"]["normalized_header_sha256"]
        ),
    }
    report = {
        "source": {"pre": source_pre, "post": source_post},
        "candidate": {"pre": candidate_pre, "post": candidate_post},
        "expected": {"sha256": expected_source_sha256, "counts": expected_counts},
        "checks": checks,
    }
    errors: list[str] = []
    labels = {
        "source_pre_expected": "source database pre-state differs from pinned contract",
        "source_raw_exact": "source raw database changed",
        "source_semantic_sha3_exact": "source semantic SHA3 changed",
        "candidate_semantic_sha3_exact": "candidate semantic SHA3 changed",
        "candidate_core_sha3_exact": "candidate core-table SHA3 changed",
        "candidate_counts_exact": "candidate mapping table counts changed",
        "candidate_pose_priors_empty": "candidate pose_priors is non-empty",
        "candidate_quick_check": "candidate SQLite quick_check failed",
        "candidate_integrity_check": "candidate SQLite integrity_check failed",
        "candidate_wal_absent_or_zero": "candidate has non-empty WAL",
        "candidate_raw_header_only": "candidate raw database changed outside SQLite header metadata",
    }
    for name, passed in checks.items():
        if not passed:
            errors.append(labels[name])
    if errors:
        raise AuditFailure("; ".join(errors), report)
    return report


def validate_migration_receipt(args: argparse.Namespace) -> dict[str, Any]:
    """Require a successful, path-pinned migration receipt before post-map audit."""
    receipt = _read_json(args.migration_receipt)
    try:
        source_evidence = receipt["source_evidence"]
        candidate_evidence = receipt["clone_evidence_after"]
        checks = {
            "receipt_status_pass": receipt["status"] == "PASS",
            "source_path_matches_cli": (
                Path(source_evidence["path"]).resolve() == args.source_db.resolve()
            ),
            "candidate_path_matches_cli": (
                Path(candidate_evidence["path"]).resolve()
                == args.candidate_db.resolve()
            ),
            "receipt_source_sha_matches_cli": (
                receipt["source_sha256"] == args.expected_source_sha256
            ),
            "source_evidence_sha_matches_cli": (
                source_evidence["sha256"] == args.expected_source_sha256
            ),
            "source_evidence_counts_match_cli": (
                source_evidence["counts"] == args.expected_counts
            ),
            "receipt_clone_sha_matches_evidence": (
                receipt["clone_sha256_after"] == candidate_evidence["sha256"]
            ),
            "candidate_counts_match_source": (
                candidate_evidence["counts"] == source_evidence["counts"]
            ),
        }
    except (KeyError, TypeError, OSError) as error:
        raise AuditFailure(
            f"migration receipt is incomplete: {error}",
            {"migration_receipt": str(args.migration_receipt.resolve())},
        ) from error
    messages = {
        "receipt_status_pass": "migration receipt status is not PASS",
        "source_path_matches_cli": "migration receipt source path differs from CLI --source-db",
        "candidate_path_matches_cli": "migration receipt candidate path differs from CLI --candidate-db",
        "receipt_source_sha_matches_cli": "migration receipt source SHA differs from CLI",
        "source_evidence_sha_matches_cli": "migration receipt source evidence SHA differs from CLI",
        "source_evidence_counts_match_cli": "migration receipt source counts differ from CLI",
        "receipt_clone_sha_matches_evidence": "migration receipt clone SHA differs from evidence",
        "candidate_counts_match_source": "migration receipt candidate counts differ from source",
    }
    failures = [messages[name] for name, passed in checks.items() if not passed]
    if failures:
        raise AuditFailure(
            "; ".join(failures),
            {
                "migration_receipt": str(args.migration_receipt.resolve()),
                "receipt": receipt,
                "checks": checks,
            },
        )
    return receipt


def _mapper_post_report(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    try:
        receipt = validate_migration_receipt(args)
        report = validate_mapper_post_contract(
            source=args.source_db,
            candidate=args.candidate_db,
            source_pre=receipt["source_evidence"],
            candidate_pre=receipt["clone_evidence_after"],
            expected_source_sha256=args.expected_source_sha256,
            expected_counts=args.expected_counts,
        )
    except (AuditFailure, KeyError) as error:
        report = error.report if isinstance(error, AuditFailure) else None
        return {
            "schema": "glomap-db-reuse-mapper-post-audit/v1",
            "status": "FAIL",
            "migration_receipt": str(args.migration_receipt.resolve()),
            "database": report,
            "errors": [str(error)],
        }, False
    return {
        "schema": "glomap-db-reuse-mapper-post-audit/v1",
        "status": "PASS",
        "migration_receipt": str(args.migration_receipt.resolve()),
        "database": report,
        "errors": [],
    }, True


def _camera_key(camera: Any) -> tuple[str, int, int]:
    return (str(camera.model.name), int(camera.width), int(camera.height))


def _camera_table(reconstruction: Any) -> dict[tuple[str, int, int], np.ndarray]:
    table: dict[tuple[str, int, int], np.ndarray] = {}
    for camera in reconstruction.cameras.values():
        key = _camera_key(camera)
        params = np.asarray(camera.params, dtype=np.float64)
        if key in table and not np.array_equal(table[key], params):
            raise AuditFailure(f"ambiguous camera signature: {key}")
        table[key] = params
    return table


def _camera_records(reconstruction: Any) -> list[dict[str, Any]]:
    return [
        {
            "camera_id": int(camera.camera_id),
            "model": str(camera.model.name),
            "width": int(camera.width),
            "height": int(camera.height),
            "params": np.asarray(camera.params, dtype=np.float64).tolist(),
        }
        for camera in sorted(
            reconstruction.cameras.values(), key=lambda item: int(item.camera_id)
        )
    ]


def _finite_values(values: Any) -> bool:
    return bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _model_summary(reconstruction: Any) -> dict[str, Any]:
    """Return raw geometry facts without suppressing invalid observations."""
    camera_records = _camera_records(reconstruction)
    registered_images = [
        image for image in reconstruction.images.values() if bool(image.has_pose)
    ]
    finite_cameras = all(_finite_values(record["params"]) for record in camera_records)
    finite_points = True
    finite_poses = True
    finite_reprojections = True
    invalid_depth_observations = 0
    missing_point_observations = 0
    observations = 0
    reprojection_errors: list[float] = []
    for image in registered_images:
        center = np.asarray(image.projection_center(), dtype=np.float64)
        finite_poses &= bool(np.isfinite(center).all())
        for point2d in image.points2D:
            if not point2d.has_point3D():
                continue
            observations += 1
            point_id = int(point2d.point3D_id)
            point = (
                reconstruction.points3D[point_id]
                if point_id in reconstruction.points3D
                else None
            )
            if point is None:
                missing_point_observations += 1
                finite_reprojections = False
                continue
            xyz = np.asarray(point.xyz, dtype=np.float64)
            finite_points &= bool(np.isfinite(xyz).all())
            if not np.isfinite(xyz).all():
                finite_reprojections = False
                continue
            camera_point = np.asarray(image.cam_from_world() * xyz, dtype=np.float64)
            if not np.isfinite(camera_point).all() or camera_point[2] <= 1e-12:
                invalid_depth_observations += 1
                finite_reprojections = False
                continue
            projection = image.project_point(xyz)
            if projection is None:
                finite_reprojections = False
                continue
            error = float(
                np.linalg.norm(
                    np.asarray(projection, dtype=np.float64)
                    - np.asarray(point2d.xy, dtype=np.float64)
                )
            )
            if not math.isfinite(error):
                finite_reprojections = False
                continue
            reprojection_errors.append(error)
    track_lengths = [
        int(point.track.length())
        if hasattr(point.track, "length")
        else len(point.track.elements)
        for point in reconstruction.points3D.values()
    ]
    track_elements = int(sum(track_lengths))
    finite_points &= all(
        _finite_values(point.xyz) for point in reconstruction.points3D.values()
    )
    observation_track_consistent = (
        missing_point_observations == 0 and observations == track_elements
    )
    camera_table = _camera_table(reconstruction)
    return {
        "cameras": camera_records,
        "camera_count": int(reconstruction.num_cameras()),
        "camera_signature_count": len(camera_table),
        "distinct_camera_model_count": len({key[0] for key in camera_table}),
        "images": int(reconstruction.num_images()),
        "registered_images": int(reconstruction.num_reg_images()),
        "points3D": int(reconstruction.num_points3D()),
        "observations": observations,
        "track_elements": track_elements,
        "track_length": {
            "min": min(track_lengths) if track_lengths else None,
            "median": _percentile([float(value) for value in track_lengths], 50),
            "max": max(track_lengths) if track_lengths else None,
        },
        "reprojection_error_px": {
            "count": len(reprojection_errors),
            "mean": float(np.mean(reprojection_errors))
            if reprojection_errors
            else None,
            "median": _percentile(reprojection_errors, 50),
            "p95": _percentile(reprojection_errors, 95),
            "max": max(reprojection_errors) if reprojection_errors else None,
            "invalid_depth_observations": invalid_depth_observations,
            "missing_point_observations": missing_point_observations,
        },
        "finite": {
            "cameras": finite_cameras,
            "points": finite_points,
            "poses": finite_poses,
            "reprojections": finite_reprojections,
        },
        "observation_track_consistent": observation_track_consistent,
    }


def _intrinsics_comparison(
    candidate: Any, seed_table: dict[tuple[str, int, int], np.ndarray], tolerance: float
) -> dict[str, Any]:
    candidate_table = _camera_table(candidate)
    keys_match = set(candidate_table) == set(seed_table)
    maximum_delta = float("inf")
    if keys_match:
        maximum_delta = max(
            float(np.max(np.abs(candidate_table[key] - seed_table[key])))
            for key in sorted(seed_table)
        )
    return {
        "keys_match_seed": keys_match,
        "maximum_abs_delta": maximum_delta,
        "tolerance": tolerance,
        "matches_seed": keys_match and maximum_delta <= tolerance,
    }


def audit_models(
    *,
    seed: Any,
    raw: Any,
    final: Any | None,
    expected_camera_count: int,
    minimum_registered: int,
    raw_intrinsics_tolerance: float,
    final_intrinsics_tolerance: float,
) -> dict[str, Any]:
    """Validate raw/final models and raise with their JSON evidence on failure."""
    if expected_camera_count <= 0 or minimum_registered <= 0:
        raise ValueError("expected camera count and registered floor must be positive")
    seed_table = _camera_table(seed)
    seed_summary = _model_summary(seed)
    raw_summary = _model_summary(raw)
    raw_intrinsics = _intrinsics_comparison(raw, seed_table, raw_intrinsics_tolerance)
    checks: dict[str, bool] = {
        "raw_camera_count": raw_summary["camera_count"] == expected_camera_count,
        "raw_camera_signatures_match_seed": raw_intrinsics["keys_match_seed"],
        "raw_distinct_camera_models_match_seed": (
            raw_summary["distinct_camera_model_count"]
            == seed_summary["distinct_camera_model_count"]
        ),
        "raw_registered_floor": raw_summary["registered_images"] >= minimum_registered,
        "raw_intrinsics_match_seed": raw_intrinsics["matches_seed"],
        "raw_finite": all(raw_summary["finite"].values()),
        "raw_observation_track_consistent": raw_summary["observation_track_consistent"],
    }
    report: dict[str, Any] = {
        "seed": seed_summary,
        "raw": raw_summary,
        "raw_intrinsics": raw_intrinsics,
        "checks": checks,
    }
    if final is not None:
        final_summary = _model_summary(final)
        final_intrinsics = _intrinsics_comparison(
            final, seed_table, final_intrinsics_tolerance
        )
        checks.update(
            {
                "final_camera_count": final_summary["camera_count"]
                == expected_camera_count,
                "final_camera_signatures_match_seed": final_intrinsics[
                    "keys_match_seed"
                ],
                "final_distinct_camera_models_match_seed": (
                    final_summary["distinct_camera_model_count"]
                    == seed_summary["distinct_camera_model_count"]
                ),
                "final_registered_floor": final_summary["registered_images"]
                >= minimum_registered,
                "final_intrinsics_match_seed": final_intrinsics["matches_seed"],
                "final_finite": all(final_summary["finite"].values()),
                "final_observation_track_consistent": final_summary[
                    "observation_track_consistent"
                ],
            }
        )
        report.update({"final": final_summary, "final_intrinsics": final_intrinsics})
    errors: list[str] = []
    for label, passed in checks.items():
        if passed:
            continue
        if "registered" in label:
            errors.append(f"registered image floor failed: {label}")
        elif "finite" in label:
            errors.append(f"non-finite geometry detected: {label}")
        elif "camera_count" in label:
            errors.append(f"unexpected camera count: {label}")
        elif "intrinsics" in label:
            errors.append(f"intrinsics do not match seed: {label}")
        else:
            errors.append(f"observation/track consistency failed: {label}")
    if errors:
        raise AuditFailure("; ".join(errors), report)
    return report


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically create a JSON artifact without overwriting an existing audit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _json_counts(value: str) -> dict[str, int]:
    try:
        decoded = json.loads(value)
        result = {table: int(decoded[table]) for table in DATABASE_TABLES}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise argparse.ArgumentTypeError(
            f"expected JSON counts for {DATABASE_TABLES}: {error}"
        ) from error
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditFailure(f"cannot read JSON {path}: {error}") from error


def _snapshot_report(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    source = database_snapshot(args.source_db)
    candidate = database_snapshot(args.candidate_db)
    checks = {
        "source_matches_expected": (
            source["sha256"] == args.expected_source_sha256
            and source["counts"] == args.expected_counts
        ),
        "candidate_matches_source": _same_snapshot(candidate, source),
        "candidate_pre_sidecars_absent": not any(
            item["exists"] for item in candidate["sidecars"].values()
        ),
    }
    report = {
        "schema": "glomap-db-reuse-snapshot/v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "source": source,
        "candidate": candidate,
        "expected": {
            "sha256": args.expected_source_sha256,
            "counts": args.expected_counts,
        },
        "checks": checks,
    }
    return report, bool(all(checks.values()))


def _audit_report(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    errors: list[str] = []
    database: dict[str, Any] | None = None
    models: dict[str, Any] | None = None
    try:
        if args.migration_receipt is not None:
            receipt = validate_migration_receipt(args)
            database = validate_mapper_post_contract(
                source=args.source_db,
                candidate=args.candidate_db,
                source_pre=receipt["source_evidence"],
                candidate_pre=receipt["clone_evidence_after"],
                expected_source_sha256=args.expected_source_sha256,
                expected_counts=args.expected_counts,
            )
            database_contract = "migration_receipt_mapper_post"
        else:
            snapshot = _read_json(args.db_snapshot)
            database = validate_database_contract(
                source=args.source_db,
                candidate=args.candidate_db,
                source_pre=snapshot["source"],
                candidate_pre=snapshot["candidate"],
                expected_source_sha256=args.expected_source_sha256,
                expected_counts=args.expected_counts,
            )
            database_contract = "raw_db_snapshot"
    except (AuditFailure, KeyError) as error:
        errors.append(str(error))
        if isinstance(error, AuditFailure):
            database = error.report
    try:
        import pycolmap

        seed = pycolmap.Reconstruction(str(args.intrinsics_seed))
        raw = pycolmap.Reconstruction(str(args.raw_model))
        final = (
            pycolmap.Reconstruction(str(args.final_model))
            if args.final_model is not None
            else None
        )
        models = audit_models(
            seed=seed,
            raw=raw,
            final=final,
            expected_camera_count=args.expected_camera_count,
            minimum_registered=args.minimum_registered,
            raw_intrinsics_tolerance=args.raw_intrinsics_tolerance,
            final_intrinsics_tolerance=args.final_intrinsics_tolerance,
        )
    except (AuditFailure, OSError, RuntimeError) as error:
        errors.append(str(error))
        if isinstance(error, AuditFailure):
            models = error.report
    report = {
        "schema": "glomap-db-reuse-candidate-audit/v1",
        "status": "PASS" if not errors else "FAIL",
        "database_contract": (
            database_contract
            if "database_contract" in locals()
            else "migration_receipt_mapper_post"
            if args.migration_receipt is not None
            else "raw_db_snapshot"
        ),
        "inputs": {
            "raw_model": str(args.raw_model.resolve()),
            "final_model": str(args.final_model.resolve())
            if args.final_model
            else None,
            "intrinsics_seed": str(args.intrinsics_seed.resolve()),
            "db_snapshot": (
                str(args.db_snapshot.resolve())
                if args.db_snapshot is not None
                else None
            ),
            "migration_receipt": (
                str(args.migration_receipt.resolve())
                if args.migration_receipt is not None
                else None
            ),
        },
        "database": database,
        "models": models,
        "errors": errors,
    }
    return report, not errors


def _add_database_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--candidate-db", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-counts", type=_json_counts, required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot_parser = subparsers.add_parser("snapshot-db")
    _add_database_contract_arguments(snapshot_parser)
    snapshot_parser.add_argument("--out", type=Path, required=True)
    audit_parser = subparsers.add_parser("audit")
    _add_database_contract_arguments(audit_parser)
    audit_proof = audit_parser.add_mutually_exclusive_group(required=True)
    audit_proof.add_argument("--db-snapshot", type=Path)
    audit_proof.add_argument("--migration-receipt", type=Path)
    audit_parser.add_argument("--raw-model", type=Path, required=True)
    audit_parser.add_argument("--final-model", type=Path)
    audit_parser.add_argument("--intrinsics-seed", type=Path, required=True)
    audit_parser.add_argument("--expected-camera-count", type=int, default=3)
    audit_parser.add_argument("--minimum-registered", type=int, default=1390)
    audit_parser.add_argument("--raw-intrinsics-tolerance", type=float, default=1e-3)
    audit_parser.add_argument("--final-intrinsics-tolerance", type=float, default=1e-6)
    audit_parser.add_argument("--out", type=Path, required=True)
    mapper_post_parser = subparsers.add_parser("mapper-post-db")
    _add_database_contract_arguments(mapper_post_parser)
    mapper_post_parser.add_argument("--migration-receipt", type=Path, required=True)
    mapper_post_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "snapshot-db":
            report, passed = _snapshot_report(args)
        elif args.command == "audit":
            report, passed = _audit_report(args)
        else:
            report, passed = _mapper_post_report(args)
        atomic_json(args.out, report)
    except (AuditFailure, FileExistsError, OSError, sqlite3.DatabaseError) as error:
        print(str(error), flush=True)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
