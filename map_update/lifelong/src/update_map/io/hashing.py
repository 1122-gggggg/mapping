from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root: Path, ignored_names: set[str] | None = None) -> Iterable[Path]:
    ignored = ignored_names or {".DS_Store", "Thumbs.db"}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in ignored:
            yield path


def create_map_snapshot(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    if not root_path.exists() or not root_path.is_dir():
        raise FileNotFoundError(f"Map directory not found: {root_path}")
    files: dict[str, dict[str, Any]] = {}
    aggregate = hashlib.sha256()
    for path in _iter_files(root_path):
        relative = path.relative_to(root_path).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        files[relative] = {"sha256": digest, "size": size}
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\n")
    return {
        "schema_version": 1,
        "root": str(root_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": files,
    }


def save_snapshot(snapshot: dict[str, Any], output: str | Path) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")


def load_snapshot(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_map_snapshot(root: str | Path, expected: dict[str, Any]) -> dict[str, Any]:
    actual = create_map_snapshot(root)
    expected_files = expected.get("files", {})
    actual_files = actual.get("files", {})
    missing = sorted(set(expected_files) - set(actual_files))
    added = sorted(set(actual_files) - set(expected_files))
    changed = sorted(
        path
        for path in set(expected_files).intersection(actual_files)
        if expected_files[path] != actual_files[path]
    )
    return {
        "ok": not missing and not added and not changed,
        "aggregate_matches": expected.get("aggregate_sha256") == actual.get("aggregate_sha256"),
        "missing": missing,
        "added": added,
        "changed": changed,
        "expected_aggregate_sha256": expected.get("aggregate_sha256"),
        "actual_aggregate_sha256": actual.get("aggregate_sha256"),
    }
