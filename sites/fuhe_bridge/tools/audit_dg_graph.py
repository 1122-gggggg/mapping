#!/usr/bin/env python3
"""Recompute the Fuhe Bridge S4 Doppelgangers graph acceptance gate."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import struct
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence
from urllib.parse import quote

import numpy as np
import yaml

from ts_intrinsics import FUHE_CX, FUHE_CY, FUHE_FX, FUHE_FY
from ts_common import Gate, assert_gate_chain_fresh, required_check_ids, sha256
from run_gluemap_memory_safe import (
    SIFT_COORDINATE_LIMIT_FACTOR,
    SIFT_DESCRIPTOR_COLUMNS,
    SIFT_KEYPOINT_COLUMNS,
    SIFT_RUNTIME_MARKER_KEYS,
    SIFT_RUNTIME_MARKER_PREFIX,
    sift_tables_sha256,
)


LOFTR_SEQUENCE_ALLOWLIST = frozenset(
    {
        tuple(sorted(("P1100110_005", "P1110111"))),
        tuple(sorted(("P1090109_002", "P1110111"))),
    }
)
FIXED_CAMERA_CONTRACT = {
    "camera_count": 1,
    "model": "PINHOLE",
    "width": 1920,
    "height": 1080,
    "params": [FUHE_FX, FUHE_FY, FUHE_CX, FUHE_CY],
    "maximum_drift": 1e-6,
}
SIFT_DB_IMAGE_COUNT = 240
SIFT_MAX_NUM_FEATURES = 2048
SIFT_MAX_NUM_ORIENTATIONS = 1
SIFT_ROW_CAP = 2048
def _valid_relative_image_name(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and path.as_posix() == value
    )


def _row_summary(values: list[int]) -> tuple[int | None, float | None, int | None]:
    if not values:
        return None, None, None
    return min(values), float(sum(values) / len(values)), max(values)


def _table_columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(
        str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
    )


def _invalid_keypoint_rows(
    rows: int,
    cols: int,
    data: bytes,
    *,
    width: int,
    height: int,
) -> int:
    coordinate_limit = SIFT_COORDINATE_LIMIT_FACTOR * max(width, height)
    invalid = 0
    consumed = 0
    for values in struct.iter_unpack(f"<{cols}f", data):
        consumed += 1
        scale = (
            values[2]
            if cols == 4
            else (
                math.hypot(values[2], values[4])
                + math.hypot(values[3], values[5])
            )
            / 2.0
        )
        if (
            not all(math.isfinite(value) for value in values)
            or not math.isfinite(scale)
            or scale <= 0
            or abs(values[0]) > coordinate_limit
            or abs(values[1]) > coordinate_limit
        ):
            invalid += 1
    if consumed != rows:
        raise ValueError(f"keypoint BLOB decoded {consumed} rows, expected {rows}")
    return invalid


def _runtime_markers(guard_text: str, violations: list[str]) -> list[dict]:
    markers: list[dict] = []
    for line in guard_text.splitlines():
        if line.startswith("sift_row_cap=PASS"):
            violations.append("legacy prewritten SIFT row-cap PASS line is forbidden")
        if not line.startswith(SIFT_RUNTIME_MARKER_PREFIX):
            continue
        try:
            payload = json.loads(line.removeprefix(SIFT_RUNTIME_MARKER_PREFIX))
        except (json.JSONDecodeError, TypeError) as error:
            violations.append(f"SIFT runtime marker is invalid JSON: {error}")
            continue
        if not isinstance(payload, dict) or payload.keys() != SIFT_RUNTIME_MARKER_KEYS:
            violations.append("SIFT runtime marker payload schema is not exact")
            continue
        markers.append(payload)
    return markers


def audit_sift_database(
    database: Path | str,
    *,
    config_path: Path | str,
    guard_log_path: Path | str,
    frame_manifest_path: Path | str,
) -> dict:
    """Validate the post-GlueMap SIFT database without mutating or locking it."""
    db_path = Path(database).expanduser().resolve()
    config_file = Path(config_path).expanduser().resolve()
    guard_file = Path(guard_log_path).expanduser().resolve()
    manifest_file = Path(frame_manifest_path).expanduser().resolve()
    wal_path = Path(f"{db_path}-wal")
    wal_size = wal_path.stat().st_size if wal_path.is_file() else 0
    violations: list[str] = []
    database_sha256: str | None = None
    database_opened = False
    database_stable = False
    database_schema_exact = False
    blob_layout_valid = False
    keypoints_finite_valid = False
    invalid_keypoint_row_count = 0
    manifest_names: list[str] = []
    database_names: list[str] = []
    keypoint_rows_by_id: dict[int, int] = {}
    descriptor_rows_by_id: dict[int, int] = {}
    image_name_by_id: dict[int, str] = {}
    image_records: dict[int, tuple[str, int, int]] = {}
    keypoint_records_by_id: dict[int, tuple[int, int, bytes]] = {}
    descriptor_records_by_id: dict[int, tuple[int, int, int, bytes]] = {}
    config: dict = {}
    runtime_marker_exact = False
    runtime_marker: dict | None = None
    guard_text = ""
    marker_candidates: list[dict] = []
    sift_tables_digest: str | None = None

    try:
        database_sha256 = sha256(db_path)
    except OSError as error:
        violations.append(f"database is unreadable: {error}")

    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        frames = payload.get("frames")
        if not isinstance(frames, list):
            raise ValueError("frame_manifest.frames must be a list")
        manifest_names = [frame.get("name") for frame in frames]
        if not all(_valid_relative_image_name(name) for name in manifest_names):
            violations.append("frame manifest contains invalid relative image names")
        if payload.get("n_frames") != len(manifest_names):
            violations.append("frame manifest n_frames differs from frames")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        violations.append(f"frame manifest is invalid: {error}")

    try:
        loaded_config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        if not isinstance(loaded_config, dict):
            raise ValueError("config must be a YAML mapping")
        config = loaded_config
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        violations.append(f"GlueMap config is invalid: {error}")

    try:
        guard_text = guard_file.read_text(encoding="utf-8")
        marker_candidates = _runtime_markers(guard_text, violations)
    except OSError as error:
        violations.append(f"resource guard log is unreadable: {error}")

    max_num_features = config.get("sift_max_num_features")
    max_num_orientations = config.get("sift_max_num_orientations")
    derived_cap = (
        max_num_features * max_num_orientations
        if type(max_num_features) is int
        and type(max_num_orientations) is int
        else None
    )
    config_policy_ok = bool(
        type(max_num_features) is int
        and max_num_features == SIFT_MAX_NUM_FEATURES
        and type(max_num_orientations) is int
        and max_num_orientations == SIFT_MAX_NUM_ORIENTATIONS
        and derived_cap == SIFT_ROW_CAP
    )
    if not config_policy_ok:
        violations.append(
            "config must declare 2048 SIFT locations, literal one orientation, "
            "and derived 2048 rows/image"
        )
    manifest_names_valid = bool(
        len(manifest_names) == SIFT_DB_IMAGE_COUNT
        and len(set(manifest_names)) == SIFT_DB_IMAGE_COUNT
        and all(_valid_relative_image_name(name) for name in manifest_names)
    )
    if not manifest_names_valid:
        violations.append("frame manifest must contain exactly 240 unique relative names")

    if wal_size > 0:
        violations.append(
            f"non-empty SQLite WAL is unsafe for immutable reads: {wal_path} "
            f"({wal_size} bytes)"
        )
    elif database_sha256 is not None:
        uri = (
            f"file:{quote(db_path.as_posix(), safe='/')}?mode=ro&immutable=1"
        )
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(uri, uri=True)
            database_opened = True
            expected_schemas = {
                "images": ("image_id", "name", "camera_id"),
                "keypoints": ("image_id", "rows", "cols", "data"),
                "descriptors": ("image_id", "type", "rows", "cols", "data"),
            }
            schema_drifts = {
                table: _table_columns(connection, table)
                for table in expected_schemas
                if _table_columns(connection, table) != expected_schemas[table]
            }
            camera_columns = set(_table_columns(connection, "cameras"))
            if not {"camera_id", "width", "height"}.issubset(camera_columns):
                schema_drifts["cameras"] = tuple(sorted(camera_columns))
            database_schema_exact = not schema_drifts
            if schema_drifts:
                violations.append(f"SIFT database schema drift: {schema_drifts}")

            image_rows = connection.execute(
                """
                SELECT i.image_id, i.name, c.width, c.height,
                       typeof(i.image_id), typeof(i.name),
                       typeof(c.width), typeof(c.height)
                FROM images AS i
                LEFT JOIN cameras AS c ON c.camera_id=i.camera_id
                ORDER BY i.image_id
                """
            ).fetchall()
            keypoint_records = connection.execute(
                """
                SELECT image_id, rows, cols, data,
                       typeof(image_id), typeof(rows), typeof(cols), typeof(data)
                FROM keypoints ORDER BY image_id
                """
            ).fetchall()
            descriptor_records = connection.execute(
                """
                SELECT image_id, type, rows, cols, data,
                       typeof(image_id), typeof(type), typeof(rows),
                       typeof(cols), typeof(data)
                FROM descriptors ORDER BY image_id
                """
            ).fetchall()
            image_names_seen: set[str] = set()
            for image_id, name, width, height, *types in image_rows:
                if (
                    types != ["integer", "text", "integer", "integer"]
                    or type(image_id) is not int
                    or image_id <= 0
                    or not isinstance(name, str)
                    or not name
                    or type(width) is not int
                    or width <= 0
                    or type(height) is not int
                    or height <= 0
                ):
                    violations.append("images table contains invalid types or values")
                    continue
                if image_id in image_records or name in image_names_seen:
                    violations.append("images table contains duplicate ids or names")
                    continue
                image_records[image_id] = (name, width, height)
                image_names_seen.add(name)
                image_name_by_id[image_id] = name
                database_names.append(name)

            layout_errors = 0
            for image_id, rows, cols, data, *types in keypoint_records:
                valid = bool(
                    types == ["integer", "integer", "integer", "blob"]
                    and type(image_id) is int
                    and image_id > 0
                    and type(rows) is int
                    and rows >= 0
                    and type(cols) is int
                    and cols in SIFT_KEYPOINT_COLUMNS
                    and isinstance(data, bytes)
                    and len(data) == rows * cols * 4
                    and image_id not in keypoint_records_by_id
                )
                if not valid:
                    layout_errors += 1
                    violations.append(
                        f"keypoints image_id={image_id!r} has invalid "
                        "types/cols/BLOB length"
                    )
                    continue
                keypoint_rows_by_id[image_id] = rows
                keypoint_records_by_id[image_id] = (rows, cols, data)

            for (
                image_id,
                descriptor_type,
                rows,
                cols,
                data,
                *types,
            ) in descriptor_records:
                descriptor_type_valid = bool(
                    types[1] == "integer"
                    and type(descriptor_type) is int
                    and descriptor_type == 0
                )
                valid = bool(
                    types
                    == ["integer", "integer", "integer", "integer", "blob"]
                    and descriptor_type_valid
                    and type(image_id) is int
                    and image_id > 0
                    and type(rows) is int
                    and rows >= 0
                    and type(cols) is int
                    and cols == SIFT_DESCRIPTOR_COLUMNS
                    and isinstance(data, bytes)
                    and len(data) == rows * cols
                    and image_id not in descriptor_records_by_id
                )
                if not valid:
                    layout_errors += 1
                    if not descriptor_type_valid:
                        violations.append(
                            "descriptors descriptor type must be SQLite integer 0 "
                            f"for image_id={image_id!r}"
                        )
                    else:
                        violations.append(
                            f"descriptors image_id={image_id!r} has invalid "
                            "types/cols/BLOB length"
                        )
                    continue
                descriptor_rows_by_id[image_id] = rows
                descriptor_records_by_id[image_id] = (
                    descriptor_type,
                    rows,
                    cols,
                    data,
                )
            blob_layout_valid = layout_errors == 0 and bool(
                keypoint_records_by_id and descriptor_records_by_id
            )

            if blob_layout_valid:
                for image_id, (rows, cols, data) in keypoint_records_by_id.items():
                    image = image_records.get(image_id)
                    if image is None:
                        continue
                    _name, width, height = image
                    invalid_keypoint_row_count += _invalid_keypoint_rows(
                        rows, cols, data, width=width, height=height
                    )
                keypoints_finite_valid = invalid_keypoint_row_count == 0
                if not keypoints_finite_valid:
                    violations.append(
                        f"keypoints contain {invalid_keypoint_row_count} invalid rows"
                    )
        except sqlite3.Error as error:
            violations.append(f"immutable SQLite audit failed: {error}")
        except (RuntimeError, ValueError, struct.error) as error:
            violations.append(f"immutable SIFT content audit failed: {error}")
        finally:
            if connection is not None:
                connection.close()
        try:
            database_stable = sha256(db_path) == database_sha256
        except OSError as error:
            violations.append(f"database rehash failed: {error}")
        if not database_stable:
            violations.append("database bytes changed during the immutable audit")

    expected_names = set(manifest_names)
    database_name_set = set(database_names)
    database_names_ok = bool(
        manifest_names_valid
        and len(database_names) == SIFT_DB_IMAGE_COUNT
        and len(database_name_set) == SIFT_DB_IMAGE_COUNT
        and database_name_set == expected_names
        and all(_valid_relative_image_name(name) for name in database_names)
    )
    if database_opened and not database_names_ok:
        violations.append("images table names differ from the exact frame manifest")

    image_ids = set(image_name_by_id)
    keypoint_ids = set(keypoint_rows_by_id)
    descriptor_ids = set(descriptor_rows_by_id)
    row_tables_complete = bool(
        database_names_ok
        and database_schema_exact
        and blob_layout_valid
        and len(keypoint_rows_by_id) == SIFT_DB_IMAGE_COUNT
        and len(descriptor_rows_by_id) == SIFT_DB_IMAGE_COUNT
        and keypoint_ids == image_ids
        and descriptor_ids == image_ids
    )
    if database_opened and not row_tables_complete:
        violations.append(
            "images/keypoints/descriptors must each cover the same exact 240 images"
        )

    row_mismatch_ids = sorted(
        image_id
        for image_id in image_ids & keypoint_ids & descriptor_ids
        if keypoint_rows_by_id[image_id] != descriptor_rows_by_id[image_id]
    )
    if row_mismatch_ids:
        violations.append(
            f"keypoint/descriptor row mismatch for {len(row_mismatch_ids)} images"
        )
    row_cap_ids = sorted(
        image_id
        for image_id in image_ids & keypoint_ids & descriptor_ids
        if keypoint_rows_by_id[image_id] > SIFT_ROW_CAP
        or descriptor_rows_by_id[image_id] > SIFT_ROW_CAP
    )
    for image_id in row_cap_ids:
        violations.append(
            f"{image_name_by_id[image_id]} exceeds 2048 rows: "
            f"keypoints={keypoint_rows_by_id[image_id]} "
            f"descriptors={descriptor_rows_by_id[image_id]}"
        )

    keypoint_values = list(keypoint_rows_by_id.values())
    descriptor_values = list(descriptor_rows_by_id.values())
    key_min, key_avg, key_max = _row_summary(keypoint_values)
    descriptor_min, descriptor_avg, descriptor_max = _row_summary(
        descriptor_values
    )

    if row_tables_complete and keypoints_finite_valid:
        sift_tables_digest = sift_tables_sha256(
            image_records, keypoint_records_by_id, descriptor_records_by_id
        )
    matching_markers = [
        marker
        for marker in marker_candidates
        if marker.get("database_path") == str(db_path)
    ]
    if matching_markers:
        runtime_marker = matching_markers[-1]
        integer_fields = (
            "image_count",
            "clamped_image_count",
            "modified_image_count",
            "removed_invalid_rows",
            "invalid_rows_after",
            "max_rows_before",
            "max_rows_after",
            "max_keypoint_descriptor_rows_per_image",
            "sift_max_num_features",
            "sift_max_num_orientations",
            "wal_size_bytes",
        )
        marker_types_valid = all(
            type(runtime_marker.get(field)) is int for field in integer_fields
        ) and type(runtime_marker.get("immutable_verified")) is bool
        marker_values_valid = bool(
            marker_types_valid
            and runtime_marker.get("schema_version") == "fuhe-sift-runtime-cap-v1"
            and runtime_marker.get("status") == "PASS"
            and runtime_marker.get("sift_max_num_features") == SIFT_MAX_NUM_FEATURES
            and runtime_marker.get("sift_max_num_orientations")
            == SIFT_MAX_NUM_ORIENTATIONS
            and runtime_marker.get("max_keypoint_descriptor_rows_per_image")
            == SIFT_ROW_CAP
            and runtime_marker.get("image_count") == len(image_records)
            and runtime_marker.get("max_rows_after") == key_max
            and runtime_marker.get("invalid_rows_after") == 0
            and runtime_marker.get("removed_invalid_rows", -1) >= 0
            and runtime_marker.get("clamped_image_count", -1) >= 0
            and runtime_marker.get("modified_image_count", -1) >= 0
            and runtime_marker.get("max_rows_before", -1)
            >= runtime_marker.get("max_rows_after", 0)
            and runtime_marker.get("wal_size_bytes") == 0
            and runtime_marker.get("immutable_verified") is True
            and runtime_marker.get("sift_tables_sha256") == sift_tables_digest
        )
        runtime_marker_exact = marker_values_valid
    if not runtime_marker_exact:
        violations.append(
            "resource guard log lacks an exact durable SIFT runtime marker "
            "matching the final feature tables"
        )

    checks = {
        "wal_empty": wal_size == 0,
        "manifest_exact_240": manifest_names_valid,
        "database_schema_exact": database_schema_exact,
        "database_names_exact": database_names_ok,
        "blob_layout_valid": blob_layout_valid,
        "keypoints_finite_valid": keypoints_finite_valid,
        "row_tables_complete": row_tables_complete,
        "keypoint_descriptor_rows_equal": not row_mismatch_ids
        and row_tables_complete,
        "rows_at_most_2048": not row_cap_ids and row_tables_complete,
        "config_row_cap_policy": config_policy_ok,
        "runtime_marker_exact": runtime_marker_exact,
        "database_stable": database_stable,
    }
    return {
        "schema_version": "fuhe-sift-db-cap-v2",
        "ok": all(checks.values()) and not violations,
        "checks": checks,
        "database_path": str(db_path),
        "database_sha256": database_sha256,
        "database_opened": database_opened,
        "sqlite_open_mode": "mode=ro&immutable=1",
        "wal_path": str(wal_path),
        "wal_size_bytes": wal_size,
        "manifest_image_count": len(manifest_names),
        "database_image_count": len(database_names),
        "keypoint_rows_min": key_min,
        "keypoint_rows_avg": key_avg,
        "keypoint_rows_max": key_max,
        "descriptor_rows_min": descriptor_min,
        "descriptor_rows_avg": descriptor_avg,
        "descriptor_rows_max": descriptor_max,
        "row_mismatch_count": len(row_mismatch_ids),
        "row_cap_violation_count": len(row_cap_ids),
        "invalid_keypoint_row_count": invalid_keypoint_row_count,
        "configured_sift_max_num_features": max_num_features,
        "configured_sift_max_num_orientations": max_num_orientations,
        "derived_max_keypoint_descriptor_rows_per_image": derived_cap,
        "sift_tables_sha256": sift_tables_digest,
        "runtime_marker": runtime_marker,
        "violations": violations,
    }


def _sequence_edge(value: tuple[str, str] | list[str] | str) -> tuple[str, str]:
    if isinstance(value, str):
        fields = value.split("|")
    else:
        fields = list(value)
    if len(fields) != 2 or not all(fields) or fields[0] == fields[1]:
        raise ValueError(f"invalid sequence edge: {value!r}")
    return tuple(sorted((str(fields[0]), str(fields[1]))))


def loftr_trigger_contract(
    edge: tuple[str, str] | list[str] | str,
    pure_checks: dict[str, bool],
    *,
    ghost_check_id: str,
    blocking_edges: Iterable[tuple[str, str] | list[str] | str],
) -> dict:
    """Authorize LoFTR only when it can cover every blocking ghost edge."""
    normalized = _sequence_edge(edge)
    normalized_blockers = {_sequence_edge(item) for item in blocking_edges}
    nonallowlisted_blockers = normalized_blockers - LOFTR_SEQUENCE_ALLOWLIST
    ghost_present = ghost_check_id in pure_checks
    ghost_failed = ghost_present and pure_checks[ghost_check_id] is False
    other_checks = {
        name: value for name, value in pure_checks.items() if name != ghost_check_id
    }
    non_ghost_pass = bool(other_checks) and all(value is True for value in other_checks.values())
    allowlisted = normalized in LOFTR_SEQUENCE_ALLOWLIST
    blocking_set_complete = bool(normalized_blockers) and normalized in normalized_blockers
    all_blockers_allowlisted = blocking_set_complete and not nonallowlisted_blockers
    authorized = (
        allowlisted
        and ghost_failed
        and non_ghost_pass
        and all_blockers_allowlisted
    )
    return {
        "schema_version": "fuhe-targeted-loftr-v2",
        "edge": list(normalized),
        "allowlisted": allowlisted,
        "blocking_edges": [list(item) for item in sorted(normalized_blockers)],
        "blocking_set_complete": blocking_set_complete,
        "all_blocking_edges_allowlisted": all_blockers_allowlisted,
        "nonallowlisted_blocking_edges": [
            list(item) for item in sorted(nonallowlisted_blockers)
        ],
        "ghost_check_id": ghost_check_id,
        "ghost_is_only_failure": ghost_failed and non_ghost_pass,
        "authorized": authorized,
        "database_copy_required": True,
        "source_database_read_only": True,
        "nonallowlisted_pairs_immutable": True,
        "full_glomap_rerun": False,
        "pgo": False,
        "model_execution_requested": False,
    }


def loftr_database_isolation(
    *,
    source_path: Path,
    branch_path: Path,
    source_before_sha256: str,
    source_after_sha256: str,
    branch_base_sha256: str,
    pair_digests_before: dict[str, str],
    pair_digests_after: dict[str, str],
) -> dict:
    """Prove DB-copy isolation and immutability outside the two allowed edges."""
    all_keys = set(pair_digests_before) | set(pair_digests_after)
    changed = sorted(
        key
        for key in all_keys
        if pair_digests_before.get(key) != pair_digests_after.get(key)
    )
    malformed: list[str] = []
    changed_edges: set[tuple[str, str]] = set()
    for key in changed:
        try:
            changed_edges.add(_sequence_edge(key))
        except ValueError:
            malformed.append(key)
    hashes = (source_before_sha256, source_after_sha256, branch_base_sha256)
    hash_evidence_valid = all(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
        for value in hashes
    )
    checks = {
        "database_paths_are_distinct": source_path.resolve()
        != branch_path.resolve(),
        "hash_evidence_valid": hash_evidence_valid,
        "source_database_unchanged": source_before_sha256 == source_after_sha256,
        "branch_started_from_source_copy": branch_base_sha256
        == source_before_sha256,
        "only_allowlisted_sequence_pairs_changed": not malformed
        and changed_edges <= LOFTR_SEQUENCE_ALLOWLIST,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "changed_sequence_pairs": changed,
        "malformed_pair_keys": malformed,
        "allowlist": [list(edge) for edge in sorted(LOFTR_SEQUENCE_ALLOWLIST)],
        "source_database": str(source_path.resolve()),
        "branch_database": str(branch_path.resolve()),
    }


def evaluate_loftr_promotion(baseline: dict, candidate: dict) -> dict:
    """Evaluate the bounded LoFTR promotion policy; lower scores are better."""
    baseline_failed = float(baseline["failed_edge_score"])
    candidate_failed = float(candidate["failed_edge_score"])
    baseline_unaffected = float(baseline["unaffected_score"])
    candidate_unaffected = float(candidate["unaffected_score"])
    baseline_registration = float(baseline["registration_fraction"])
    candidate_registration = float(candidate["registration_fraction"])
    values = (
        baseline_failed,
        candidate_failed,
        baseline_unaffected,
        candidate_unaffected,
        baseline_registration,
        candidate_registration,
    )
    if (
        not all(math.isfinite(value) for value in values)
        or baseline_failed <= 0
        or candidate_failed < 0
        or baseline_unaffected < 0
        or candidate_unaffected < 0
        or not 0 <= baseline_registration <= 1
        or not 0 <= candidate_registration <= 1
    ):
        raise ValueError("promotion metrics must be finite and within valid ranges")
    failed_improvement = (baseline_failed - candidate_failed) / baseline_failed
    unaffected_worsening = (
        (candidate_unaffected - baseline_unaffected) / baseline_unaffected
        if baseline_unaffected > 0
        else (0.0 if candidate_unaffected == 0 else math.inf)
    )
    registration_drop = baseline_registration - candidate_registration
    checks = {
        "failed_edge_improves_at_least_10pct": failed_improvement >= 0.10 - 1e-12,
        "unaffected_worsens_at_most_10pct": unaffected_worsening <= 0.10 + 1e-12,
        "registration_drops_at_most_1pp": registration_drop <= 0.01 + 1e-12,
        "no_loftr_only_two_view": int(
            candidate.get(
                "loftr_only_two_view_count",
                candidate.get("loftr_only_two_view_points", -1),
            )
        ) == 0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "failed_edge_improvement_fraction": failed_improvement,
        "unaffected_worsening_fraction": unaffected_worsening,
        "registration_drop_fraction": registration_drop,
        "full_glomap_rerun": False,
        "pgo": False,
    }


def targeted_loftr_branch_gate(
    trigger: dict,
    database_isolation: dict,
    promotion: dict,
) -> dict:
    """Compose the non-executing targeted LoFTR branch promotion gate."""
    checks = {
        "trigger_authorized": trigger.get("authorized") is True,
        "database_isolated": database_isolation.get("ok") is True,
        "promotion_safe": promotion.get("status") == "PASS",
        "no_second_full_mapper_or_pgo": trigger.get("full_glomap_rerun") is False
        and trigger.get("pgo") is False
        and promotion.get("full_glomap_rerun") is False
        and promotion.get("pgo") is False,
    }
    return {
        "stage": "S6_targeted_loftr_branch",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "trigger": trigger,
        "database_isolation": database_isolation,
        "promotion": promotion,
    }

def largest_component_fraction(
    node_count: int, edges: Iterable[tuple[int, int]]
) -> tuple[float, int]:
    if node_count <= 0:
        return 0.0, 0
    parent = list(range(node_count))
    size = [1] * node_count

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        a, b = find(int(left)), find(int(right))
        if a == b:
            continue
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
    largest = max(size[find(node)] for node in range(node_count))
    return largest / node_count, largest


def independent_bridge_count(
    normalized_pairs: list[tuple[float, float]], *, minimum_separation: float
) -> int:
    """Return two only when bridge support separates on both traversals."""
    if not normalized_pairs:
        return 0
    for index, (left_a, right_a) in enumerate(normalized_pairs):
        for left_b, right_b in normalized_pairs[index + 1 :]:
            if (
                abs(left_a - left_b) >= minimum_separation
                and abs(right_a - right_b) >= minimum_separation
            ):
                return 2
    return 1


def robust_sequence_component_fraction(
    directions: dict[str, str],
    sequence_edges: set[tuple[str, str]],
    bridge_counts: dict[tuple[str, str], int],
) -> tuple[float, set[tuple[str, str]]]:
    """Drop weak cross-direction hinges and score the remaining backbone."""
    names = sorted(directions)
    index = {name: position for position, name in enumerate(names)}
    retained = set()
    for raw_left, raw_right in sequence_edges:
        left, right = sorted((raw_left, raw_right))
        if (
            directions[left] == directions[right]
            or bridge_counts.get((left, right), 0) >= 2
        ):
            retained.add((left, right))
    fraction, _ = largest_component_fraction(
        len(names), ((index[left], index[right]) for left, right in retained)
    )
    return fraction, retained


def load_forced_pairs(path: Path) -> set[tuple[str, str]]:
    pairs = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"invalid forced-pair line: {line!r}")
        pairs.add(tuple(sorted((fields[0], fields[1]))))
    return pairs


def classify_accepted_cross_pairs(
    names: list[str],
    pairs: Iterable[tuple[int, int]],
    scores: Iterable[float],
    *,
    threshold: float,
    directions: dict[str, str],
    conditional_pairs: set[tuple[str, str]],
) -> dict[tuple[str, str], list[dict[str, object]]]:
    """Return every DG-accepted cross-direction pair with its retrieval source.

    The conditional-pair file is provenance/classification evidence only.  It
    never filters DG-accepted evidence: naturally retrieved pairs are equally
    eligible to establish separated bridge support.
    """
    normalized_conditional = {
        tuple(sorted((str(left), str(right))))
        for left, right in conditional_pairs
    }
    result: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    pair_rows = list(pairs)
    score_rows = list(scores)
    if len(pair_rows) != len(score_rows):
        raise ValueError("two-view score/pair length mismatch")
    for pair, raw_score in zip(pair_rows, score_rows, strict=True):
        score = float(raw_score)
        if not math.isfinite(score) or score <= threshold:
            continue
        left_index, right_index = (int(value) for value in pair)
        if not 0 <= left_index < len(names) or not 0 <= right_index < len(names):
            raise IndexError("two-view pair index is outside the image corpus")
        left, right = names[left_index], names[right_index]
        left_sequence = left.split("/", 1)[0]
        right_sequence = right.split("/", 1)[0]
        if left_sequence == right_sequence:
            continue
        if left_sequence not in directions or right_sequence not in directions:
            raise KeyError("accepted pair references a sequence without direction")
        if directions[left_sequence] == directions[right_sequence]:
            continue
        edge = tuple(sorted((left_sequence, right_sequence)))
        if left_sequence != edge[0]:
            left, right = right, left
        source = (
            "conditional"
            if tuple(sorted((left, right))) in normalized_conditional
            else "natural"
        )
        result[edge].append(
            {"left": left, "right": right, "score": score, "source": source}
        )
    return dict(result)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twoview", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--forced-pairs", type=Path, required=True)
    parser.add_argument("--forced-manifest", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--guard-log", type=Path, required=True)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--minimum-separation", type=float, default=0.25)
    return parser.parse_args(argv)


def stage_gate(run_dir: Path, args: argparse.Namespace) -> Gate:
    gate = Gate(
        "S4_doppelgangers",
        required_check_ids("S4_doppelgangers"),
        script_path=__file__,
        source_files=[
            Path(__file__).with_name("ts_common.py"),
            Path(__file__).with_name("ts_intrinsics.py"),
        ],
        input_artifacts={
            "two_view_scores": args.twoview,
            "image_root": args.image_root,
            "conditional_pairs": args.forced_pairs,
            "conditional_pair_manifest": args.forced_manifest,
            "sift_database": args.database,
            "gluemap_config": args.config,
            "resource_guard_log": args.guard_log,
            "frame_manifest": args.frame_manifest,
        },
    )
    gate.record_predecessor_gate(
        "S3_pairs",
        run_dir / "gates" / "S3_pairs.json",
        expected_stage="S3_pairs",
    )
    return gate


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    run_dir = args.image_root.resolve().parent
    predecessor_gate = run_dir / "gates" / "S3_pairs.json"
    assert_gate_chain_fresh(predecessor_gate)

    gate = stage_gate(run_dir, args)
    sift_db_cap = audit_sift_database(
        args.database,
        config_path=args.config,
        guard_log_path=args.guard_log,
        frame_manifest_path=args.frame_manifest,
    )
    if not sift_db_cap["ok"]:
        gate.check(
            "G4.0",
            False,
            "SIFT_DB_CAP failed before Doppelgangers graph loading",
            stage_metrics=sift_db_cap,
        )
        for gid in ("G4.1", "G4.2", "G4.3", "G4.4"):
            gate.not_run(
                gid,
                "Doppelgangers graph audit is blocked by failed SIFT_DB_CAP",
                sift_db_cap_ok=False,
            )
        gate.write(
            run_dir,
            output_path=args.out,
            stage_metrics={"sift_db_cap": sift_db_cap},
        )
        return

    import torch

    names = sorted(
        path.relative_to(args.image_root).as_posix()
        for path in args.image_root.glob("*/*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    payload = torch.load(args.twoview, map_location="cpu", weights_only=False)
    pairs = np.asarray(payload["pairs"], dtype=np.int64)
    scores = np.asarray(payload["scores"], dtype=np.float64)
    if len(scores) != len(pairs):
        raise SystemExit("two-view score/pair length mismatch")
    valid_mask = scores > args.threshold
    valid_pairs = pairs[valid_mask]
    component_fraction, component_size = largest_component_fraction(
        len(names), ((int(a), int(b)) for a, b in valid_pairs)
    )
    rejection_rate = float(np.mean(~valid_mask))

    forced = load_forced_pairs(args.forced_pairs)
    manifest = json.loads(args.forced_manifest.read_text(encoding="utf-8"))
    forward = set(manifest["fwd"])
    reverse = set(manifest["rev"])
    directions = {
        **{sequence: "fwd" for sequence in forward},
        **{sequence: "rev" for sequence in reverse},
    }
    local_position: dict[str, float] = {}
    by_sequence: dict[str, list[str]] = defaultdict(list)
    for name in names:
        by_sequence[name.split("/", 1)[0]].append(name)
    for sequence_names in by_sequence.values():
        denominator = max(1, len(sequence_names) - 1)
        for index, name in enumerate(sequence_names):
            local_position[name] = index / denominator

    sequence_edges: set[tuple[str, str]] = set()
    for left_index, right_index in valid_pairs:
        left, right = names[int(left_index)], names[int(right_index)]
        left_sequence, right_sequence = left.split("/", 1)[0], right.split("/", 1)[0]
        if left_sequence == right_sequence:
            continue
        sequence_edges.add(tuple(sorted((left_sequence, right_sequence))))
    cross_valid = classify_accepted_cross_pairs(
        names,
        valid_pairs,
        scores[valid_mask],
        threshold=args.threshold,
        directions=directions,
        conditional_pairs=forced,
    )

    bridge_evidence = {}
    bridge_counts: dict[tuple[str, str], int] = {}
    for sequence_pair, accepted in sorted(cross_valid.items()):
        normalized = [
            (local_position[str(row["left"])], local_position[str(row["right"])])
            for row in accepted
        ]
        count = independent_bridge_count(
            normalized, minimum_separation=args.minimum_separation
        )
        bridge_counts[tuple(sorted(sequence_pair))] = count
        source_counts = {
            source: sum(row["source"] == source for row in accepted)
            for source in ("natural", "conditional")
        }
        bridge_evidence["|".join(sequence_pair)] = {
            "accepted_cross_edges": len(accepted),
            "source_counts": source_counts,
            "accepted_pairs": accepted,
            "independent_bridge_count": count,
        }

    robust_fraction, retained_sequence_edges = robust_sequence_component_fraction(
        directions, sequence_edges, bridge_counts
    )
    robust_cross_edges = [
        edge
        for edge in retained_sequence_edges
        if directions[edge[0]] != directions[edge[1]]
    ]

    pre_dg_database_sha256 = sift_db_cap["database_sha256"]
    sift_db_cap = audit_sift_database(
        args.database,
        config_path=args.config,
        guard_log_path=args.guard_log,
        frame_manifest_path=args.frame_manifest,
    )
    sift_db_cap["pre_dg_database_sha256"] = pre_dg_database_sha256
    sift_db_cap["revalidated_after_dg"] = True
    if sift_db_cap["database_sha256"] != pre_dg_database_sha256:
        sift_db_cap["ok"] = False
        sift_db_cap["violations"].append(
            "SIFT database SHA-256 changed after the pre-DG validation"
        )
    gate.check(
        "G4.0",
        sift_db_cap["ok"],
        "SIFT_DB_CAP proves exact 240-image coverage and at most 2048 "
        "keypoint/descriptor rows per image",
        stage_metrics=sift_db_cap,
    )

    graph_checks = {
        "G4.1": bool(
            len(scores)
            and float(np.min(scores)) < args.threshold < float(np.max(scores))
            and float(np.std(scores)) > 1e-6
        ),
        "G4.2": 0.02 <= rejection_rate <= 0.40,
        "G4.3": robust_fraction == 1.0 and bool(robust_cross_edges),
        "G4.4": component_fraction >= 0.90,
    }
    checks = {"G4.0": bool(sift_db_cap["ok"]), **graph_checks}
    result = {
        "stage": "S4_doppelgangers",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "sift_db_cap": sift_db_cap,
        "threshold": args.threshold,
        "candidate_pairs": len(scores),
        "accepted_pairs": int(np.sum(valid_mask)),
        "rejected_pairs": int(np.sum(~valid_mask)),
        "rejection_rate": rejection_rate,
        "largest_component_images": component_size,
        "largest_component_fraction": component_fraction,
        "robust_sequence_component_fraction": robust_fraction,
        "retained_sequence_edges": [list(edge) for edge in sorted(retained_sequence_edges)],
        "robust_cross_direction_edges": [list(edge) for edge in sorted(robust_cross_edges)],
        "bridge_evidence": bridge_evidence,
        "conditional_pair_provenance": {
            "role": "source classification only; never an acceptance filter",
            "pair_file": {
                "path": str(args.forced_pairs.resolve()),
                "sha256": sha256(args.forced_pairs),
            },
            "manifest": {
                "path": str(args.forced_manifest.resolve()),
                "sha256": sha256(args.forced_manifest),
            },
        },
        "fixed_camera_contract": FIXED_CAMERA_CONTRACT,
        "targeted_loftr_branch_contract": {
            "schema_version": "fuhe-targeted-loftr-v2",
            "sequence_pair_allowlist": [
                list(edge) for edge in sorted(LOFTR_SEQUENCE_ALLOWLIST)
            ],
            "trigger": (
                "all pure gates except ghost PASS; ghost alone FAIL; every "
                "blocking applicable sequence edge is allowlisted"
            ),
            "database_copy_required": True,
            "source_and_nonallowlisted_immutable": True,
            "promotion": {
                "failed_edge_minimum_improvement_fraction": 0.10,
                "unaffected_maximum_worsening_fraction": 0.10,
                "maximum_registration_drop_fraction": 0.01,
                "loftr_only_two_view_points": 0,
            },
            "full_glomap_rerun": False,
            "pgo": False,
        },
    }
    for gid, passed in graph_checks.items():
        gate.check(
            gid,
            passed,
            "Doppelgangers graph predicate recomputed from complete accepted evidence",
            stage_metrics=result,
        )
    payload = gate.write(run_dir, output_path=args.out, stage_metrics=result)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
