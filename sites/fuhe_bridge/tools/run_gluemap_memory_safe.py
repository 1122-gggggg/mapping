#!/usr/bin/env python3
"""Launch GlueMap with a fail-closed per-image SIFT feature ceiling."""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import sqlite3
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import quote

import yaml

from resource_guard import (
    contract_from_config,
    exclusive_resource_lock,
    startup_preflight,
)


SIFT_RUNTIME_MARKER_PREFIX = "sift_row_cap_runtime_v1="
SIFT_COORDINATE_LIMIT_FACTOR = 4.0
SIFT_KEYPOINT_COLUMNS = frozenset({4, 6})
SIFT_DESCRIPTOR_COLUMNS = 128
SIFT_RUNTIME_MARKER_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "database_path",
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
        "sift_tables_sha256",
        "wal_size_bytes",
        "immutable_verified",
    }
)


def _config_path(argv: Sequence[str]) -> Path:
    for index, argument in enumerate(argv):
        if argument == "--config" and index + 1 < len(argv):
            return Path(argv[index + 1]).expanduser().resolve()
        if argument.startswith("--config="):
            return Path(argument.split("=", 1)[1]).expanduser().resolve()
    raise ValueError("the memory-safe launcher requires --config")


def _config_payload(argv: Sequence[str]) -> tuple[Path, dict[str, Any]]:
    path = _config_path(argv)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")
    return path, payload


def sift_policy_from_config(argv: Sequence[str]) -> tuple[int, int]:
    """Load the fail-closed SIFT row-cap policy from the GlueMap config."""
    path, payload = _config_payload(argv)
    max_num_features = payload.get("sift_max_num_features")
    if type(max_num_features) is not int or max_num_features <= 0:
        raise ValueError(
            f"{path}: sift_max_num_features must be a positive integer"
        )
    max_num_orientations = payload.get("sift_max_num_orientations")
    if type(max_num_orientations) is not int or max_num_orientations != 1:
        raise ValueError(
            f"{path}: sift_max_num_orientations must be literal integer 1 "
            "for the required extraction policy; the database adapter "
            "enforces the strict descriptor-row cap"
        )
    return max_num_features, max_num_orientations


def sift_cap_from_config(argv: Sequence[str]) -> int:
    """Compatibility accessor; validates the complete row-cap policy."""
    return sift_policy_from_config(argv)[0]


def ba_limits_from_config(
    argv: Sequence[str],
) -> tuple[int | None, int | None]:
    """Read optional recovery-only BA limits from the target-site config."""
    path, payload = _config_payload(argv)
    values: list[int | None] = []
    for key in ("ba_max_num_iterations", "ba_max_filter_iterations"):
        value = payload.get(key)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{path}: {key} must be a positive integer")
        values.append(value)
    return values[0], values[1]


def assert_clean_gluemap_paths(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Reject stale output/temp contents before importing the heavy runtime."""
    evidence: dict[str, dict[str, str]] = {}
    for key in ("write_path", "temp_path"):
        raw_path = config.get(key)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"{key} must be a non-empty path")
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            state = "ABSENT"
        elif not path.is_dir():
            raise RuntimeError(f"{key} exists and is not a directory: {path}")
        elif any(path.iterdir()):
            raise RuntimeError(f"{key} is non-empty; refusing stale resume: {path}")
        else:
            state = "EMPTY"
        evidence[key] = {"path": str(path), "state": state}
    return evidence


def _exact_table_schema(
    connection: sqlite3.Connection,
    table: str,
    expected_columns: tuple[str, ...],
) -> None:
    columns = tuple(
        str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
    )
    if columns != expected_columns:
        raise RuntimeError(
            f"SIFT database {table} schema must be exactly "
            f"{expected_columns}, got {columns}"
        )


def _read_sift_images(
    connection: sqlite3.Connection,
) -> dict[int, tuple[str, int, int]]:
    _exact_table_schema(connection, "images", ("image_id", "name", "camera_id"))
    camera_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(cameras)")
    }
    required_camera_columns = {"camera_id", "width", "height"}
    if not required_camera_columns.issubset(camera_columns):
        raise RuntimeError("SIFT database cameras schema lacks dimensions")

    images: dict[int, tuple[str, int, int]] = {}
    names: set[str] = set()
    query = """
        SELECT i.image_id, i.name, c.width, c.height,
               typeof(i.image_id), typeof(i.name),
               typeof(c.width), typeof(c.height)
        FROM images AS i
        LEFT JOIN cameras AS c ON c.camera_id = i.camera_id
        ORDER BY i.image_id
    """
    for image_id, name, width, height, *types in connection.execute(query):
        if types != ["integer", "text", "integer", "integer"]:
            raise RuntimeError("SIFT images table contains invalid SQLite types")
        if (
            type(image_id) is not int
            or image_id <= 0
            or not isinstance(name, str)
            or not name
            or type(width) is not int
            or width <= 0
            or type(height) is not int
            or height <= 0
        ):
            raise RuntimeError("SIFT images table contains invalid values")
        if image_id in images or name in names:
            raise RuntimeError("SIFT images table must contain unique ids and names")
        images[image_id] = (name, width, height)
        names.add(name)
    if not images:
        raise RuntimeError("SIFT images table must be non-empty")
    return images


def _read_sift_keypoints(
    connection: sqlite3.Connection,
) -> dict[int, tuple[int, int, bytes]]:
    table = "keypoints"
    _exact_table_schema(connection, table, ("image_id", "rows", "cols", "data"))
    records: dict[int, tuple[int, int, bytes]] = {}
    query = """
        SELECT image_id, rows, cols, data,
               typeof(image_id), typeof(rows), typeof(cols), typeof(data)
        FROM keypoints ORDER BY image_id
    """
    for image_id, rows, cols, data, *types in connection.execute(query):
        if types != ["integer", "integer", "integer", "blob"]:
            raise RuntimeError(f"{table} table contains invalid SQLite types")
        if type(image_id) is not int or image_id <= 0:
            raise RuntimeError(f"{table} contains an invalid image_id: {image_id!r}")
        if image_id in records:
            raise RuntimeError(f"{table} contains duplicate image_id={image_id}")
        if type(rows) is not int or rows < 0:
            raise RuntimeError(
                f"{table} image_id={image_id} has invalid rows: {rows!r}"
            )
        if type(cols) is not int or cols not in SIFT_KEYPOINT_COLUMNS:
            raise RuntimeError(
                f"SIFT keypoint cols are invalid for image_id={image_id}: {cols!r}"
            )
        if not isinstance(data, bytes):
            raise RuntimeError(f"{table} image_id={image_id} has invalid BLOB data")
        expected_bytes = rows * cols * 4
        if len(data) != expected_bytes:
            raise RuntimeError(
                f"{table} image_id={image_id} BLOB length {len(data)} "
                f"does not match rows={rows}, cols={cols}, "
                f"element_bytes=4 ({expected_bytes})"
            )
        records[image_id] = (rows, cols, data)
    if not records:
        raise RuntimeError(f"SIFT {table} table must be non-empty")
    return records


def _read_sift_descriptors(
    connection: sqlite3.Connection,
) -> dict[int, tuple[int, int, int, bytes]]:
    table = "descriptors"
    _exact_table_schema(
        connection, table, ("image_id", "type", "rows", "cols", "data")
    )
    records: dict[int, tuple[int, int, int, bytes]] = {}
    query = """
        SELECT image_id, type, rows, cols, data,
               typeof(image_id), typeof(type), typeof(rows),
               typeof(cols), typeof(data)
        FROM descriptors ORDER BY image_id
    """
    for image_id, descriptor_type, rows, cols, data, *types in connection.execute(
        query
    ):
        if types[1] != "integer" or type(descriptor_type) is not int or descriptor_type != 0:
            raise RuntimeError(
                "SIFT descriptor type must be SQLite integer 0 "
                f"for image_id={image_id}: {descriptor_type!r}"
            )
        if types != ["integer", "integer", "integer", "integer", "blob"]:
            raise RuntimeError("descriptors table contains invalid SQLite types")
        if type(image_id) is not int or image_id <= 0:
            raise RuntimeError(
                f"descriptors contains an invalid image_id: {image_id!r}"
            )
        if image_id in records:
            raise RuntimeError(f"descriptors contains duplicate image_id={image_id}")
        if type(rows) is not int or rows < 0:
            raise RuntimeError(
                f"descriptors image_id={image_id} has invalid rows: {rows!r}"
            )
        if type(cols) is not int or cols != SIFT_DESCRIPTOR_COLUMNS:
            raise RuntimeError(
                f"SIFT descriptor cols are invalid for image_id={image_id}: {cols!r}"
            )
        if not isinstance(data, bytes):
            raise RuntimeError(
                f"descriptors image_id={image_id} has invalid BLOB data"
            )
        expected_bytes = rows * cols
        if len(data) != expected_bytes:
            raise RuntimeError(
                f"descriptors image_id={image_id} BLOB length {len(data)} "
                f"does not match rows={rows}, cols={cols}, "
                f"element_bytes=1 ({expected_bytes})"
            )
        records[image_id] = (descriptor_type, rows, cols, data)
    if not records:
        raise RuntimeError("SIFT descriptors table must be non-empty")
    return records


def _validated_sift_rows(
    connection: sqlite3.Connection,
) -> tuple[
    dict[int, tuple[str, int, int]],
    dict[int, tuple[int, int, bytes]],
    dict[int, tuple[int, int, int, bytes]],
]:
    images = _read_sift_images(connection)
    keypoints = _read_sift_keypoints(connection)
    descriptors = _read_sift_descriptors(connection)
    if images.keys() != keypoints.keys() or images.keys() != descriptors.keys():
        raise RuntimeError(
            "SIFT images/keypoints/descriptors image sets do not match exactly"
        )
    for image_id in images:
        if keypoints[image_id][0] != descriptors[image_id][1]:
            raise RuntimeError(
                f"SIFT image_id={image_id} has inconsistent keypoint/descriptor rows: "
                f"{keypoints[image_id][0]} != {descriptors[image_id][1]}"
            )
    return images, keypoints, descriptors


def _keypoint_scale(values: tuple[float, ...]) -> float:
    if len(values) == 4:
        return values[2]
    return (
        math.hypot(values[2], values[4])
        + math.hypot(values[3], values[5])
    ) / 2.0


def _selected_sift_indices(
    record: tuple[int, int, bytes],
    *,
    width: int,
    height: int,
    max_rows: int,
) -> tuple[list[int], int, bool]:
    rows, cols, data = record
    coordinate_limit = SIFT_COORDINATE_LIMIT_FACTOR * max(width, height)
    valid: list[tuple[int, float]] = []
    removed_invalid = 0
    for index, values in enumerate(struct.iter_unpack(f"<{cols}f", data)):
        scale = _keypoint_scale(values)
        invalid = bool(
            not all(math.isfinite(value) for value in values)
            or not math.isfinite(scale)
            or scale <= 0
            or abs(values[0]) > coordinate_limit
            or abs(values[1]) > coordinate_limit
        )
        if invalid:
            removed_invalid += 1
        else:
            valid.append((index, scale))
    if len(valid) + removed_invalid != rows:
        raise RuntimeError("SIFT keypoint parser did not consume every declared row")
    ranked = sorted(valid, key=lambda item: (-item[1], item[0]))[:max_rows]
    selected = sorted(index for index, _scale in ranked)
    return selected, removed_invalid, len(valid) > max_rows


def _select_blob_rows(data: bytes, row_bytes: int, indices: list[int]) -> bytes:
    return b"".join(
        data[index * row_bytes : (index + 1) * row_bytes] for index in indices
    )


def sift_tables_sha256(
    images: dict[int, tuple[str, int, int]],
    keypoints: dict[int, tuple[int, int, bytes]],
    descriptors: dict[int, tuple[int, int, int, bytes]],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"fuhe-sift-tables-v2\0")
    for image_id in sorted(images):
        name, width, height = images[image_id]
        name_bytes = name.encode("utf-8")
        digest.update(struct.pack("<QIII", image_id, width, height, len(name_bytes)))
        digest.update(name_bytes)
        keypoint_rows, keypoint_cols, keypoint_data = keypoints[image_id]
        digest.update(b"keypoints\0")
        digest.update(
            struct.pack(
                "<III", keypoint_rows, keypoint_cols, len(keypoint_data)
            )
        )
        digest.update(keypoint_data)
        descriptor_type, descriptor_rows, descriptor_cols, descriptor_data = (
            descriptors[image_id]
        )
        digest.update(b"descriptors\0")
        digest.update(
            struct.pack(
                "<iIII",
                descriptor_type,
                descriptor_rows,
                descriptor_cols,
                len(descriptor_data),
            )
        )
        digest.update(descriptor_data)
    return digest.hexdigest()


def _immutable_sift_verification(path: Path, max_rows: int) -> dict[str, Any]:
    uri = f"file:{quote(path.resolve().as_posix(), safe='/')}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        images, keypoints, descriptors = _validated_sift_rows(connection)
        invalid_rows = 0
        for image_id, (_name, width, height) in images.items():
            selected, removed, over_cap = _selected_sift_indices(
                keypoints[image_id], width=width, height=height, max_rows=max_rows
            )
            invalid_rows += removed
            if over_cap or len(selected) != keypoints[image_id][0]:
                raise RuntimeError(
                    f"SIFT immutable verification failed for image_id={image_id}"
                )
        max_rows_after = max(record[0] for record in keypoints.values())
        return {
            "image_count": len(images),
            "invalid_rows_after": invalid_rows,
            "max_rows_after": max_rows_after,
            "sift_tables_sha256": sift_tables_sha256(
                images, keypoints, descriptors
            ),
        }
    finally:
        connection.close()


def clamp_sift_database_rows(
    database_path: os.PathLike[str] | str | bytes,
    *,
    max_rows: int,
) -> dict[str, Any]:
    """Filter invalid SIFT rows and keep the strongest paired rows transactionally."""
    if type(max_rows) is not int or max_rows <= 0:
        raise ValueError("SIFT database max_rows must be a positive integer")
    path = Path(os.fsdecode(database_path)).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"SIFT database was not created by extraction: {path}")

    connection = sqlite3.connect(path)
    clamped_image_count = 0
    modified_image_count = 0
    removed_invalid_rows = 0
    max_rows_before = 0
    try:
        connection.execute("BEGIN IMMEDIATE")
        images, keypoints, descriptors = _validated_sift_rows(connection)
        max_rows_before = max(record[0] for record in keypoints.values())
        for image_id, (_name, width, height) in images.items():
            keypoint_rows, keypoint_cols, keypoint_data = keypoints[image_id]
            (
                _descriptor_type,
                _descriptor_rows,
                descriptor_cols,
                descriptor_data,
            ) = descriptors[image_id]
            selected, removed, over_cap = _selected_sift_indices(
                keypoints[image_id], width=width, height=height, max_rows=max_rows
            )
            removed_invalid_rows += removed
            clamped_image_count += int(over_cap)
            if selected == list(range(keypoint_rows)):
                continue
            modified_image_count += 1
            selected_keypoints = _select_blob_rows(
                keypoint_data, keypoint_cols * 4, selected
            )
            selected_descriptors = _select_blob_rows(
                descriptor_data, descriptor_cols, selected
            )
            keypoint_cursor = connection.execute(
                "UPDATE keypoints SET rows = ?, data = ? WHERE image_id = ?",
                (len(selected), selected_keypoints, image_id),
            )
            if keypoint_cursor.rowcount != 1:
                raise RuntimeError(
                    f"SIFT keypoints image_id={image_id} updated "
                    f"{keypoint_cursor.rowcount} rows instead of 1"
                )
            descriptor_cursor = connection.execute(
                "UPDATE descriptors SET rows = ?, data = ? WHERE image_id = ?",
                (len(selected), selected_descriptors, image_id),
            )
            if descriptor_cursor.rowcount != 1:
                raise RuntimeError(
                    f"SIFT descriptors image_id={image_id} updated "
                    f"{descriptor_cursor.rowcount} rows instead of 1"
                )
        _validated_sift_rows(connection)
        connection.commit()
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is None or checkpoint[0] != 0:
            raise RuntimeError(f"SIFT WAL checkpoint failed: {checkpoint!r}")
    except sqlite3.Error as error:
        connection.rollback()
        raise RuntimeError(f"SIFT database transaction failed: {error}") from error
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    wal_path = Path(f"{path}-wal")
    wal_size = wal_path.stat().st_size if wal_path.is_file() else 0
    if wal_size != 0:
        raise RuntimeError(f"SIFT WAL remains non-empty after checkpoint: {wal_path}")
    verified = _immutable_sift_verification(path, max_rows)
    if verified["invalid_rows_after"] != 0:
        raise RuntimeError("SIFT immutable verification found invalid rows")
    return {
        "database_path": str(path),
        "image_count": verified["image_count"],
        "clamped_image_count": clamped_image_count,
        "modified_image_count": modified_image_count,
        "removed_invalid_rows": removed_invalid_rows,
        "invalid_rows_after": verified["invalid_rows_after"],
        "max_rows_before": max_rows_before,
        "max_rows_after": verified["max_rows_after"],
        "max_keypoint_descriptor_rows_per_image": max_rows,
        "sift_tables_sha256": verified["sift_tables_sha256"],
        "wal_size_bytes": wal_size,
        "immutable_verified": True,
    }


def write_sift_runtime_marker(
    marker_path: os.PathLike[str] | str,
    clamp_result: dict[str, Any],
    *,
    max_num_features: int,
    max_num_orientations: int,
) -> dict[str, Any]:
    payload = {
        "schema_version": "fuhe-sift-runtime-cap-v1",
        "status": "PASS",
        **clamp_result,
        "sift_max_num_features": max_num_features,
        "sift_max_num_orientations": max_num_orientations,
    }
    if payload.keys() != SIFT_RUNTIME_MARKER_KEYS:
        raise RuntimeError("SIFT runtime marker payload schema is not exact")
    line = (
        SIFT_RUNTIME_MARKER_PREFIX
        + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    path = Path(marker_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        written = os.write(descriptor, line)
        if written != len(line):
            raise RuntimeError(
                f"short SIFT runtime marker write: {written} != {len(line)}"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def install_sift_feature_cap(
    pycolmap_module: Any,
    max_num_features: int,
    max_num_orientations: int,
    *,
    runtime_marker_path: os.PathLike[str] | str,
) -> Callable[..., Any]:
    """Apply SIFT extraction limits, then enforce the database row ceiling.

    COLMAP can retain a complete boundary scale level and exceed
    ``max_num_features`` even with one orientation.  The runtime options reduce the
    extraction result; the paired database clamp makes the configured limit strict.
    """
    if type(max_num_features) is not int or max_num_features <= 0:
        raise ValueError("SIFT max_num_features must be a positive integer")
    if type(max_num_orientations) is not int or max_num_orientations != 1:
        raise ValueError("SIFT max_num_orientations must be literal integer 1")

    original = pycolmap_module.extract_features

    @functools.wraps(original)
    def capped_extract_features(*args: Any, **kwargs: Any) -> Any:
        positional = list(args)
        options = kwargs.get("extraction_options")
        if options is None and len(positional) >= 6:
            options = positional[5]
        if options is None:
            options = pycolmap_module.FeatureExtractionOptions()
            kwargs["extraction_options"] = options
        if not hasattr(options, "sift"):
            raise RuntimeError("pycolmap feature options expose no SIFT settings")
        for field in ("max_num_features", "max_num_orientations"):
            if not hasattr(options.sift, field):
                raise RuntimeError(f"pycolmap SIFT options expose no {field}")
        options.sift.max_num_features = max_num_features
        options.sift.max_num_orientations = max_num_orientations
        if (
            options.sift.max_num_features != max_num_features
            or options.sift.max_num_orientations != max_num_orientations
        ):
            raise RuntimeError("pycolmap rejected the configured SIFT row-cap policy")
        result = original(*positional, **kwargs)
        database_path = positional[0] if positional else kwargs.get("database_path")
        if database_path is None:
            raise RuntimeError("pycolmap extract_features received no database_path")
        clamp_result = clamp_sift_database_rows(
            database_path, max_rows=max_num_features
        )
        write_sift_runtime_marker(
            runtime_marker_path,
            clamp_result,
            max_num_features=max_num_features,
            max_num_orientations=max_num_orientations,
        )
        return result

    pycolmap_module.extract_features = capped_extract_features
    return original


def install_ba_limits(
    global_refinement_module: Any,
    *,
    max_num_iterations: int | None,
    max_filter_iterations: int | None,
) -> Callable[..., Any]:
    """Override GlueMap's hardcoded BA limits without editing its repository."""
    for label, value in (
        ("max_num_iterations", max_num_iterations),
        ("max_filter_iterations", max_filter_iterations),
    ):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{label} must be a positive integer")

    original = global_refinement_module.IterativeBAOptions

    @functools.wraps(original)
    def limited_options(*args: Any, **kwargs: Any) -> Any:
        positional = list(args)
        if max_num_iterations is not None:
            if positional:
                positional[0] = max_num_iterations
            else:
                kwargs["max_ba_iterations"] = max_num_iterations
        if max_filter_iterations is not None:
            if len(positional) >= 2:
                positional[1] = max_filter_iterations
            else:
                kwargs["max_filter_iterations"] = max_filter_iterations
        return original(*positional, **kwargs)

    global_refinement_module.IterativeBAOptions = limited_options
    return original


def main(argv: Sequence[str] | None = None) -> Any:
    arguments = list(sys.argv[1:] if argv is None else argv)
    config_path, config = _config_payload(arguments)
    contract = contract_from_config(config, config_path.parent)
    cap, max_num_orientations = sift_policy_from_config(arguments)
    ba_max_iterations, ba_max_filter_iterations = ba_limits_from_config(arguments)
    if argv is not None:
        sys.argv = [sys.argv[0], *arguments]
    guard = contract["guard"]
    guard_log = Path(guard["log_path"])
    guard_log.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_resource_lock(Path(contract["lock"]["path"])):
        path_preflight = assert_clean_gluemap_paths(config)
        preflight = startup_preflight(Path(contract["run_dir"]))
        print(
            "[memory-safe-launcher] startup preflight PASS: "
            f"MemAvailable={preflight['mem_available_gib']:.3f} GiB, "
            f"VRAM free={preflight['vram_free_gib']:.3f} GiB, "
            f"swap free={preflight['swap_free_gib']:.3f} GiB, "
            f"disk free={preflight['disk_free_gib']:.3f} GiB",
            flush=True,
        )
        print(
            "[memory-safe-launcher] output paths PASS: "
            + ", ".join(
                f"{name}={record['state']}"
                for name, record in sorted(path_preflight.items())
            ),
            flush=True,
        )

        import pycolmap

        install_sift_feature_cap(
            pycolmap,
            cap,
            max_num_orientations,
            runtime_marker_path=guard_log,
        )
        print(
            "[memory-safe-launcher] SIFT row-cap policy armed: "
            f"max_num_features={cap}, "
            f"max_num_orientations={max_num_orientations}, "
            f"max_keypoint_descriptor_rows_per_image={cap}",
            flush=True,
        )

        if ba_max_iterations is not None or ba_max_filter_iterations is not None:
            from gluemap.controllers import global_refinement

            install_ba_limits(
                global_refinement,
                max_num_iterations=ba_max_iterations,
                max_filter_iterations=ba_max_filter_iterations,
            )
            print(
                "[memory-safe-launcher] BA limits: "
                f"iterations={ba_max_iterations or 'GlueMap default'}, "
                f"filter_rounds={ba_max_filter_iterations or 'GlueMap default'}",
                flush=True,
            )

        from gluemap.cli import demo_main

        with guard_log.open("a", encoding="utf-8") as stream:
            stream.write(
                "startup_preflight=PASS "
                f"mem_available_gib={preflight['mem_available_gib']:.3f} "
                f"vram_free_gib={preflight['vram_free_gib']:.3f} "
                f"swap_free_gib={preflight['swap_free_gib']:.3f} "
                f"disk_free_gib={preflight['disk_free_gib']:.3f}\n"
            )
            stream.flush()
            monitor = subprocess.Popen(
                [sys.executable, guard["script"], str(os.getpid())],
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                return demo_main()
            finally:
                monitor.terminate()
                try:
                    monitor.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    monitor.kill()
                    monitor.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
