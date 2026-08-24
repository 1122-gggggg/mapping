"""Attach existing S0–S9 gate JSON. Never recompute gates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_ATTACHMENT_GLOBS = (
    "gates/*.json",
    "**/verify_final_release*.json",
    "**/validate_heldout*.json",
    "**/audit_map_geometry*.json",
)

_OK_STATUSES = frozenset({"PASS", "OK", "TRUE", "YES"})


def load_run_attachments(run_dir: str | Path | None) -> dict[str, Any]:
    """Read already-written gate JSON under ``run_dir`` into advisory findings.

    Missing directories and unreadable/corrupt files yield empty or partial
    findings. Nothing here is a release decision.
    """
    findings: list[dict[str, Any]] = []
    if run_dir is None:
        return {"findings": findings}
    root = Path(run_dir)
    if not root.is_dir():
        return {"findings": findings}
    for path in _iter_attachment_files(root):
        payload = _read_json(path)
        if payload is None:
            continue
        source = _source_label(root, path)
        findings.extend(_normalize_checks(payload, source))
    return {"findings": findings}


def _iter_attachment_files(root: Path) -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in _ATTACHMENT_GLOBS:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            files.append(path)
    files.sort(key=lambda item: item.as_posix().lower())
    return files


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        return None


def _source_label(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _extract_checks(payload: Any) -> Any:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping) and "checks" in payload:
        return payload["checks"]
    return None


def _status_ok(value: str) -> bool:
    return value.strip().upper() in _OK_STATUSES


def _as_ok(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        for key in ("ok", "pass", "passed"):
            if key in value:
                return _as_ok(value[key])
        for key in ("state", "status"):
            if key in value:
                return _as_ok(value[key])
        return False
    if isinstance(value, str):
        return _status_ok(value)
    if value is None:
        return False
    return bool(value)


def _finding(check_id: Any, value: Any, source: str) -> dict[str, Any]:
    return {"id": str(check_id), "ok": _as_ok(value), "source": source}


def _normalize_checks(payload: Any, source: str) -> list[dict[str, Any]]:
    checks = _extract_checks(payload)
    if isinstance(checks, Mapping):
        return [_finding(check_id, value, source) for check_id, value in checks.items()]
    if isinstance(checks, list):
        findings: list[dict[str, Any]] = []
        for index, item in enumerate(checks):
            if isinstance(item, Mapping):
                check_id = item.get("id") or item.get("name") or item.get("check_id")
                if check_id is None or check_id == "":
                    check_id = str(index)
                findings.append(_finding(check_id, item, source))
            else:
                findings.append(_finding(index, item, source))
        return findings
    return []
