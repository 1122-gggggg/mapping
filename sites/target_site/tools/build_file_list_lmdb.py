#!/usr/bin/env python3
"""Build a fail-closed SCRStudio-compatible LMDB from an explicit image file list.

SCRStudio's ImageFolderReader discovers files with an unsorted glob, so its image
order is not a valid companion for packed pose and calibration arrays.  This tool
treats ``<split>/file_list.txt`` as the single source of ordering truth.  It never
overwrites an existing ``rgb_lmdb`` target and publishes a fully verified LMDB only
after an atomic sibling-directory rename.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

import lmdb
import numpy as np
from PIL import Image


SCHEMA_VERSION = 1
LMDB_DB_NAME = b"images"
LMDB_TARGET_NAME = "rgb_lmdb"


def sha256_bytes(value: bytes) -> str:
    """Return a stable SHA-256 for raw file content."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a stable SHA-256 without loading an entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_file_list(path: Path) -> tuple[bytes, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"file list is unavailable: {path}")
    raw = path.read_bytes()
    try:
        names = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"file list is not UTF-8: {path}") from error
    if not names:
        raise ValueError(f"file list is empty: {path}")
    return raw, names


def _safe_relative_path(name: str) -> PurePosixPath:
    if not name:
        raise ValueError("file list contains an empty row")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.name in ("", ".", "..")
        or path.as_posix() != name
    ):
        raise ValueError(f"unsafe image path in file list: {name!r}")
    return path


def _validate_names(names: list[str], rgb_root: Path) -> list[Path]:
    seen: set[str] = set()
    sources: list[Path] = []
    for name in names:
        relative = _safe_relative_path(name)
        if name in seen:
            raise ValueError(f"duplicate image path in file list: {name}")
        seen.add(name)
        source = rgb_root / Path(relative)
        if not source.is_file():
            raise FileNotFoundError(f"missing image referenced by file list: {source}")
        sources.append(source)
    return sources


def _load_array_with_rows(path: Path, expected_rows: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"required packed array is unavailable: {path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.ndim < 1 or int(values.shape[0]) != expected_rows:
        actual_rows = int(values.shape[0]) if values.ndim else 0
        raise ValueError(
            f"{path.name} row count {actual_rows} does not match file list rows {expected_rows}"
        )
    return values


def _load_split_metadata(
    split_dir: Path, expected_rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    poses = _load_array_with_rows(split_dir / "poses.npy", expected_rows)
    calibration = _load_array_with_rows(split_dir / "calibration.npy", expected_rows)
    image_shapes = _load_array_with_rows(split_dir / "image_shapes.npy", expected_rows)
    if image_shapes.ndim < 2 or int(image_shapes.shape[1]) < 2:
        raise ValueError("image_shapes.npy must have at least height and width columns")
    return poses, calibration, image_shapes


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_synced(path: Path, value: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _decode_shape(value: bytes, *, name: str) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(value)) as image:
            image.load()
            width, height = image.size
    except Exception as error:
        raise ValueError(f"failed to decode image bytes for {name}") from error
    return int(height), int(width)


def _map_size(sources: list[Path]) -> int:
    total_bytes = sum(path.stat().st_size for path in sources)
    return max(64 * 1024 * 1024, total_bytes * 2 + 64 * 1024 * 1024)


def _write_lmdb(target: Path, sources: list[Path]) -> None:
    target.mkdir()
    env = lmdb.open(
        str(target),
        subdir=True,
        map_size=_map_size(sources),
        readonly=False,
        meminit=False,
        max_dbs=1,
    )
    try:
        db = env.open_db(LMDB_DB_NAME, integerkey=True)
        with env.begin(write=True, db=db) as transaction:
            for index, source in enumerate(sources):
                transaction.put(
                    int(index).to_bytes(4, sys.byteorder),
                    source.read_bytes(),
                    db=db,
                )
        env.sync()
    finally:
        env.close()


def verify_lmdb_from_file_list(
    split_dir: Path,
    lmdb_dir: Path,
    *,
    report_lmdb_dir: Path | None = None,
) -> dict[str, Any]:
    """Return a complete order, byte, decode, and packed-array verification report."""
    split_dir = split_dir.resolve(strict=True)
    lmdb_dir = lmdb_dir.resolve(strict=True)
    displayed_lmdb_dir = (
        report_lmdb_dir.resolve(strict=False)
        if report_lmdb_dir is not None
        else lmdb_dir
    )
    source_list, source_names = _read_file_list(split_dir / "file_list.txt")
    generated_list, generated_names = _read_file_list(lmdb_dir / "file_list.txt")
    if source_names != generated_names:
        raise ValueError(
            "generated LMDB file_list.txt differs from source file_list.txt"
        )
    sources = _validate_names(source_names, split_dir / "rgb")
    poses, calibration, image_shapes = _load_split_metadata(
        split_dir, len(source_names)
    )

    env = lmdb.open(
        str(lmdb_dir),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_dbs=1,
    )
    try:
        db = env.open_db(LMDB_DB_NAME, integerkey=True)
        with env.begin(write=False, db=db) as transaction:
            entries = int(transaction.stat(db=db)["entries"])
            if entries != len(source_names):
                raise ValueError(
                    f"LMDB entry count {entries} does not match file list rows {len(source_names)}"
                )
            per_key: list[dict[str, Any]] = []
            total_bytes = 0
            for index, (name, source) in enumerate(
                zip(source_names, sources, strict=True)
            ):
                key = int(index).to_bytes(4, sys.byteorder)
                value = transaction.get(key, db=db)
                if value is None:
                    raise ValueError(f"LMDB is missing key {index} for {name}")
                source_bytes = source.read_bytes()
                if value != source_bytes:
                    raise ValueError(
                        f"LMDB raw bytes differ from source for key {index}: {name}"
                    )
                decoded_shape = _decode_shape(value, name=name)
                expected_shape = (
                    int(image_shapes[index][0]),
                    int(image_shapes[index][1]),
                )
                if decoded_shape != expected_shape:
                    raise ValueError(
                        f"decoded shape {decoded_shape} differs from image_shapes.npy "
                        f"{expected_shape} for key {index}: {name}"
                    )
                total_bytes += len(value)
                per_key.append(
                    {
                        "index": index,
                        "path": name,
                        "bytes": len(value),
                        "value_sha256": sha256_bytes(value),
                        "decoded_shape_hw": list(decoded_shape),
                        "expected_shape_hw": list(expected_shape),
                    }
                )
    finally:
        env.close()

    return {
        "schema_version": SCHEMA_VERSION,
        "format": "scrstudio_lmdb_file_list_order/v1",
        "split_dir": str(split_dir),
        "lmdb_dir": str(displayed_lmdb_dir),
        "source_file_list_sha256": sha256_bytes(source_list),
        "generated_file_list_sha256": sha256_bytes(generated_list),
        "file_list_sha256_equal": sha256_bytes(source_list)
        == sha256_bytes(generated_list),
        "rows": len(source_names),
        "entries": len(per_key),
        "bytes": total_bytes,
        "missing": 0,
        "duplicate": 0,
        "extra": 0,
        "pose_rows": int(poses.shape[0]),
        "calibration_rows": int(calibration.shape[0]),
        "image_shape_rows": int(image_shapes.shape[0]),
        "reader_integration_tested": False,
        "training_ready": False,
        "per_key": per_key,
    }


def _write_verification_json_atomically(lmdb_dir: Path, report: dict[str, Any]) -> None:
    destination = lmdb_dir / "verification.json"
    temporary = lmdb_dir / f".verification.json.tmp-{uuid.uuid4().hex}"
    try:
        _write_bytes_synced(
            temporary,
            (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        os.replace(temporary, destination)
        _fsync_directory(lmdb_dir)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_existing_lmdb_from_file_list(
    split_dir: Path,
    *,
    target_name: str = LMDB_TARGET_NAME,
) -> dict[str, Any]:
    """Re-verify a published LMDB and atomically replace only its receipt JSON."""
    split_dir = split_dir.resolve(strict=True)
    target = split_dir / target_name
    if not target.is_dir():
        raise FileNotFoundError(f"published LMDB target is unavailable: {target}")
    report = verify_lmdb_from_file_list(split_dir, target)
    _write_verification_json_atomically(target, report)
    return report


def build_lmdb_from_file_list(
    split_dir: Path,
    *,
    target_name: str = LMDB_TARGET_NAME,
) -> dict[str, Any]:
    """Build and atomically publish a verified LMDB for one packed SCRStudio split."""
    split_dir = split_dir.resolve(strict=True)
    if not split_dir.is_dir():
        raise NotADirectoryError(f"split directory is not a directory: {split_dir}")
    if not target_name or Path(target_name).name != target_name:
        raise ValueError(f"unsafe LMDB target name: {target_name!r}")
    target = split_dir / target_name
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing LMDB target: {target}")

    source_list, names = _read_file_list(split_dir / "file_list.txt")
    sources = _validate_names(names, split_dir / "rgb")
    _load_split_metadata(split_dir, len(names))

    temporary = split_dir / f".{target_name}.tmp-{uuid.uuid4().hex}"
    try:
        _write_lmdb(temporary, sources)
        _write_bytes_synced(temporary / "file_list.txt", source_list)
        report = verify_lmdb_from_file_list(
            split_dir,
            temporary,
            report_lmdb_dir=target,
        )
        _write_bytes_synced(
            temporary / "verification.json",
            (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        _fsync_directory(temporary)
        if target.exists():
            raise FileExistsError(
                f"refusing to overwrite existing LMDB target: {target}"
            )
        os.rename(temporary, target)
        _fsync_directory(split_dir)
        return report
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-dir",
        type=Path,
        action="append",
        required=True,
        help="Packed SCRStudio split containing rgb/, file_list.txt, and packed arrays; repeatable.",
    )
    parser.add_argument("--target-name", default=LMDB_TARGET_NAME)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Re-verify a published LMDB and replace only verification.json.",
    )
    args = parser.parse_args()
    for split_dir in args.split_dir:
        resolved_split = split_dir.expanduser().resolve(strict=True)
        if args.verify_existing:
            report = verify_existing_lmdb_from_file_list(
                resolved_split,
                target_name=args.target_name,
            )
        else:
            report = build_lmdb_from_file_list(
                resolved_split, target_name=args.target_name
            )
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
