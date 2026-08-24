"""Shared fail-closed parsing and serialization helpers."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _eligible(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[bool, str]:
    if str(row.get("status") or "").upper() not in set(cfg["geometric_statuses"]):
        return False, "status_not_geometrically_usable"
    verified = _first_number(row, "num_verified_pairs", "num_cross_session_tracks", "inlier_count")
    if not verified or verified <= 0:
        return False, "retrieval_only_or_missing_verified_geometry"
    if (_number(row.get("independent_bridge_groups")) or 0) <= 0:
        return False, "missing_independent_bridge_group"
    required = (
        (str(row.get("evidence_scope") or "") == "exact_pair", "not_exact_pair_evidence"),
        (_truthy(row.get("independent_artifact")), "not_independent_artifact"),
        (_truthy(row.get("geometry_complete")), "incomplete_geometry"),
        (_truthy(row.get("group_holdout_disjoint")), "fit_holdout_not_group_disjoint"),
    )
    for valid, reason in required:
        if not valid:
            return False, reason
    return True, "eligible_exact_pair_geometry"


def _row_dict(row: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    if is_dataclass(row):
        payload = asdict(row)
        payload.update(getattr(row, "__dict__", {}))
        return payload
    if isinstance(getattr(row, "__dict__", None), dict):
        return dict(row.__dict__)
    raise TypeError(f"Unsupported edge row: {type(row)!r}")


def _artifact_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    pair = row.get("pair")
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        row.setdefault("session_a", pair[0])
        row.setdefault("session_b", pair[1])
    if isinstance(row.get("metrics"), Mapping):
        row.update(row["metrics"])
    return row


def _pair(a: Any, b: Any) -> tuple[str, str] | None:
    if a in (None, "") or b in (None, "") or str(a) == str(b):
        return None
    left, right = str(a), str(b)
    return (left, right) if left <= right else (right, left)


def _number(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _first_number(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _number(row.get(key))
        if value is not None:
            return value
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _clip(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _gaussian(value: float, scale: float) -> float:
    return float(math.exp(-0.5 * (abs(value) / max(abs(float(scale)), 1e-9)) ** 2))


def _json_value(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip().startswith(("[", "{")):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _items(value: Any) -> set[str]:
    value = _json_value(value)
    if value in (None, ""):
        return set()
    if isinstance(value, str):
        return {item for item in value.replace("|", " ").replace(",", " ").split() if item}
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        return {str(item) for item in value}
    return {str(value)}


def _append_reason(value: Any, reason: str) -> str:
    return "|".join(sorted(_items(value) | {reason}))


def _prune_reason(
    pair: tuple[str, str],
    spectral: set[tuple[str, str]],
    community: set[tuple[str, str]],
) -> str:
    reasons = []
    if pair in spectral:
        reasons.append("spectral_minimum_range_outlier")
    if pair in community:
        reasons.append("separated_weak_community")
    return "+".join(reasons)


def _standardize(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return (values - median) / max(1.4826 * mad, float(np.std(values)), 1e-9)


def _deep_update(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set, np.ndarray)):
        return json.dumps(value, sort_keys=True, default=_json_default)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON type: {type(value).__name__}")


def _write_pairs(path: Path, pairs: Iterable[Sequence[str]]) -> None:
    _write_lines(path, (f"{pair[0]} {pair[1]}" for pair in pairs))


def _write_lines(path: Path, values: Iterable[Any]) -> None:
    lines = [str(value) for value in values]
    path.write_text(
        ("\n".join(lines) + "\n") if lines else "# none\n", encoding="utf-8"
    )


class _UnionFind:
    def __init__(self, nodes: Iterable[str]) -> None:
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}
        self.count = len(self.parent)

    def find(self, node: str) -> str:
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1
        self.count -= 1
