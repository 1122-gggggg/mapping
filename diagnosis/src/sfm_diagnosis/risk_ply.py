"""Colored risk-sphere PLY export from MapData plus canonical diagnosis JSON.

This module does not compute a second Fisher information matrix. Heatmap and
weak-region artifacts are consumed as already-diagnosed evidence.

Held-out success is the conjunction of outer ``DIRECT_STRONG`` and nested
``decision.status == ACCEPT``. Nested ``REJECT*`` always fails. Outer
``GEOMETRY_WEAK`` / ``PROVISIONAL`` plus nested ``ACCEPT`` remains a marker.
If either richer status is present, boolean ``success`` cannot override;
missing one side is not strict. Boolean ``success`` is used only when both
richer statuses are absent.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .io import write_json
from .models import MapData


ISSUE_COLORS: dict[str, tuple[int, int, int]] = {
    "coverage_hole": (255, 0, 0),
    "fim_rank_deficient": (255, 0, 255),
    "fim_weak": (255, 140, 0),
    "direction_sensitive": (255, 215, 0),
    "unverified_bridge_pose": (148, 0, 211),
    "zero_triangulation": (0, 110, 70),
    "weak_region": (255, 64, 64),
    "heldout_geometry_weak": (180, 0, 0),
    "heldout_provisional": (0, 180, 200),
    "failure_retrieval_proxy": (30, 90, 255),
    "actloc_shadow": (160, 160, 160),
}

ISSUE_LEGEND: dict[str, str] = {
    "coverage_hole": "Heatmap pose has sparse visible support or occupancy below the diagnostic floor.",
    "fim_rank_deficient": "Consumed heatmap FIM rank is below 6; at least one pose direction is unobservable.",
    "fim_weak": "Consumed heatmap marks weak FIM isotropy/condition. This is observability, not success probability.",
    "direction_sensitive": "Consumed heatmap health changes sharply across orientations at one position.",
    "unverified_bridge_pose": "Registered camera has zero 3D observations; pose is not localization evidence.",
    "zero_triangulation": (
        "Registered triangulation camera has zero 3D observations; the pose is unsupported by map tracks."
    ),
    "weak_region": "Weak-region centroid from sfm-diagnosis analyze. Not a calibrated failure location.",
    "heldout_geometry_weak": (
        "Optional localization log has a pose but failed the strict conjunction "
        "(outer DIRECT_STRONG and nested decision.status ACCEPT). Nested REJECT* "
        "always fails. Outer GEOMETRY_WEAK/PROVISIONAL plus nested ACCEPT stays a marker."
    ),
    "heldout_provisional": (
        "Optional localization log is outer PROVISIONAL (even with nested ACCEPT); "
        "not a deployable success."
    ),
    "failure_retrieval_proxy": "Failed query has no pose; marker is a retrieval reference, not ground truth.",
    "actloc_shadow": "StructuralLocalizabilityProxy shadow marker only. Not an authorized ActLoc network.",
}

CAVEATS = (
    "FIM observability is not a calibrated localization success probability.",
    "No-pose failure retrieval proxies are not ground-truth query poses.",
    "StructuralLocalizabilityProxy and ExternalPredictorAdapter are ActLoc-style "
    "shadow diagnostics only; they are not an authorized ActLoc network and are "
    "not held-out calibrated.",
    "Held-out success is outer DIRECT_STRONG AND nested decision.status ACCEPT; "
    "nested REJECT always wins. Outer GEOMETRY_WEAK/PROVISIONAL plus nested "
    "ACCEPT remains a marker. If either richer status exists, boolean success "
    "cannot override; it is used only when both statuses are absent.",
)

_JSONL_SUFFIXES = {".jsonl", ".ndjson"}
_HEATMAP_DIR_NAMES = (
    "pose_health.csv",
    "position_health.csv",
    "weak_regions.json",
    "summary.json",
)
_JSON_ROW_KEYS = (
    "images",
    "regions",
    "rows",
    "queries",
    "virtual_camera_rows",
    "frames",
)
_BRIDGE_ROLE_TOKENS = {
    "bridge",
    "bridge_only",
    "pure_rotation",
    "unverified_bridge_pose",
}
_TRIANGULATION_ROLE_TOKENS = {
    "core",
    "parallax",
    "triangulate",
    "triangulation",
    "zero_triangulation",
}
_ZERO_OBS_ISSUES = {"unverified_bridge_pose", "zero_triangulation"}


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    if number is None:
        return None
    return int(number)


def _rgb(name: str) -> tuple[int, int, int]:
    return ISSUE_COLORS[name]


def sphere_points(center: Sequence[float], radius: float, count: int = 48) -> np.ndarray:
    """Deterministic spherical shell. Count is the number of sample points."""

    origin = np.asarray(center, dtype=float).reshape(3)
    n = max(8, int(count))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    out = np.zeros((n, 3), dtype=float)
    for index in range(n):
        y = 1.0 - (index / max(n - 1, 1)) * 2.0
        radial = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * index
        out[index] = (
            origin
            + radius
            * np.array([math.cos(theta) * radial, y, math.sin(theta) * radial], dtype=float)
        )
    return out


def _mapping_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in _JSON_ROW_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, Mapping)]
        return [dict(payload)]
    return []


def load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    """Parse one JSON object per non-blank line. Non-objects fail closed."""

    target = Path(path)
    rows: list[dict[str, Any]] = []
    with target.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{target}:{lineno} is not valid JSON") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"{target}:{lineno} is not a JSON object "
                    f"(got {type(payload).__name__})"
                )
            rows.append(dict(payload))
    return rows


def load_rows(path: str | Path | None) -> list[dict[str, Any]]:
    """Load diagnosis or localization rows from JSON/JSONL/CSV or a directory."""

    if path is None:
        return []
    target = Path(path)
    if target.is_dir():
        for name in _HEATMAP_DIR_NAMES:
            candidate = target / name
            if candidate.is_file():
                target = candidate
                break
        else:
            jsonl_files = sorted(
                [
                    *target.glob("*.jsonl"),
                    *target.glob("*.ndjson"),
                ]
            )
            rows: list[dict[str, Any]] = []
            for candidate in jsonl_files:
                rows.extend(load_jsonl_rows(candidate))
            return rows
    if not target.is_file():
        return []
    suffix = target.suffix.lower()
    if suffix in _JSONL_SUFFIXES:
        return load_jsonl_rows(target)
    if suffix == ".json":
        return _mapping_rows(json.loads(target.read_text(encoding="utf-8")))
    with target.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_rows(path: str | Path | None) -> list[dict[str, Any]]:
    return load_rows(path)


def _is_row_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _coerce_rows(
    value: str | Path | Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if _is_row_sequence(value):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    return load_rows(value)


def _image_name(row: Mapping[str, Any]) -> str | None:
    for key in ("image_name", "output_name", "name"):
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    return None


def _image_keys(row: Mapping[str, Any]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    name = _image_name(row)
    if name is not None:
        keys.add(("name", name))
        keys.add(("name", Path(name).name))
    image_id = _as_int(row.get("image_id"))
    if image_id is not None:
        keys.add(("id", str(image_id)))
    return keys


def _normalize_role(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _zero_obs_issue(role: Any) -> str:
    token = _normalize_role(role)
    if token in _TRIANGULATION_ROLE_TOKENS:
        return "zero_triangulation"
    if token in _BRIDGE_ROLE_TOKENS or token == "":
        return "unverified_bridge_pose"
    return "zero_triangulation"


def normalize_image_roles(
    value: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> dict[str, str]:
    """Accept a name→role map, row sequence, or JSON/JSONL/CSV path."""

    if value is None:
        return {}
    if isinstance(value, (str, bytes, Path)):
        value = load_rows(value)
    if isinstance(value, Mapping):
        if isinstance(value.get("frames"), list):
            value = value["frames"]
        elif any(key in value for key in ("role", "image_role", "motion_role")):
            value = [value]
        elif all(
            not isinstance(role, (Mapping, list, tuple)) for role in value.values()
        ):
            return {
                str(key): str(role)
                for key, role in value.items()
                if role is not None
            }
        else:
            return {}
    if not _is_row_sequence(value):
        return {}
    roles: dict[str, str] = {}
    for row in value:
        if not isinstance(row, Mapping):
            continue
        name = _image_name(row)
        role = row.get("role", row.get("image_role", row.get("motion_role")))
        if name is None or role is None:
            continue
        roles[name] = str(role)
        base = Path(name).name
        roles.setdefault(base, str(role))
    return roles


def _role_for_image(
    name: str,
    image_id: int,
    roles: Mapping[str, str],
) -> str | None:
    if name in roles:
        return roles[name]
    base = Path(name).name
    if base in roles:
        return roles[base]
    if str(image_id) in roles:
        return roles[str(image_id)]
    return None


def _observation_counts(map_data: MapData) -> np.ndarray:
    counts = np.zeros(map_data.num_images, dtype=int)
    lookup = map_data.image_index()
    for track in map_data.track_image_ids:
        for image_id in np.asarray(track).reshape(-1):
            index = lookup.get(int(image_id))
            if index is not None:
                counts[index] += 1
    return counts


def markers_from_map(
    map_data: MapData,
    image_roles: Mapping[str, str] | None = None,
    skip_image_keys: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    counts = _observation_counts(map_data)
    roles = dict(image_roles or {})
    skip = skip_image_keys or set()
    markers: list[dict[str, Any]] = []
    for index, count in enumerate(counts.tolist()):
        if count > 0:
            continue
        name = str(map_data.image_names[index])
        image_id = int(map_data.image_ids[index])
        keys = {("name", name), ("name", Path(name).name), ("id", str(image_id))}
        if keys & skip:
            continue
        role = _role_for_image(name, image_id, roles)
        issue = _zero_obs_issue(role)
        center = map_data.image_centers[index]
        markers.append(
            {
                "issue_class": issue,
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
                "source": "map_zero_observations",
                "image_name": name,
                "image_id": image_id,
                "image_role": role,
            }
        )
    return markers


def markers_from_heatmap(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_position: dict[tuple[float, float, float], list[Mapping[str, Any]]] = {}
    markers: list[dict[str, Any]] = []
    for row in rows:
        x, y, z = _as_float(row.get("x")), _as_float(row.get("y")), _as_float(row.get("z"))
        if x is None or y is None or z is None:
            continue
        by_position.setdefault((round(x, 5), round(y, 5), round(z, 5)), []).append(row)
        codes = str(row.get("codes") or row.get("best_codes") or "")
        primary = str(row.get("primary") or row.get("best_primary") or "")
        visible = _as_int(row.get("visible_points"))
        occupancy = _as_float(row.get("grid_occupancy"))
        rank_proxy = _as_float(row.get("fim_lambda_min"))
        condition = _as_float(row.get("fim_condition"))
        issues: list[str] = []
        if "DATA_SPARSE" in codes or primary == "DATA_SPARSE" or (
            visible is not None and visible < 40
        ) or (occupancy is not None and occupancy < 6):
            issues.append("coverage_hole")
        if rank_proxy is not None and rank_proxy <= 1e-12:
            issues.append("fim_rank_deficient")
        if "GEOMETRY_WEAK" in codes or primary == "GEOMETRY_WEAK" or (
            condition is not None and condition > 1e6
        ):
            issues.append("fim_weak")
        for issue in issues:
            markers.append(
                {
                    "issue_class": issue,
                    "x": x,
                    "y": y,
                    "z": z,
                    "source": "heatmap",
                    "primary": primary,
                    "codes": codes,
                }
            )
    for (x, y, z), group in by_position.items():
        scores = [_as_float(row.get("health_score") or row.get("best_health")) for row in group]
        finite = [value for value in scores if value is not None]
        if len(finite) < 2:
            continue
        if max(finite) - min(finite) >= 0.25 and max(finite) >= 0.45:
            markers.append(
                {
                    "issue_class": "direction_sensitive",
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "source": "heatmap_orientation_spread",
                }
            )
    return markers


def markers_from_weak_regions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    for row in rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else row
        x = _as_float(metrics.get("center_x") or metrics.get("x") or row.get("x"))
        y = _as_float(metrics.get("center_y") or metrics.get("y") or row.get("y"))
        z = _as_float(metrics.get("center_z") or metrics.get("z") or row.get("z"))
        if x is None or y is None or z is None:
            continue
        if row.get("weak") is False:
            continue
        markers.append(
            {
                "issue_class": "weak_region",
                "x": x,
                "y": y,
                "z": z,
                "source": "weak_regions",
                "region_id": row.get("region_id") or row.get("id"),
            }
        )
    return markers


def _pose_center(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        x = _as_float(value.get("x"))
        y = _as_float(value.get("y"))
        z = _as_float(value.get("z"))
        if x is not None and y is not None and z is not None:
            return x, y, z
        value = value.get("pose") or value.get("T_cw") or value.get("matrix")
        if value is None:
            return None
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.shape == (4, 4):
        center = -matrix[:3, :3].T @ matrix[:3, 3]
        return float(center[0]), float(center[1]), float(center[2])
    if matrix.shape == (3,):
        return float(matrix[0]), float(matrix[1]), float(matrix[2])
    return None


def _nested_decision(row: Mapping[str, Any]) -> tuple[str, bool, bool]:
    decision = row.get("decision")
    if not isinstance(decision, Mapping):
        return "", False, False
    nested = str(decision.get("status") or "").strip().upper()
    accept_flag = decision.get("accept")
    if accept_flag is None:
        accept_flag = decision.get("accepted")
    accepted = nested == "ACCEPT" or accept_flag in {1, True, "1", "true", "True"}
    present = bool(nested) or accept_flag is not None
    return nested, accepted, present


def markers_from_localization(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Emit held-out failure and provisional markers.

    Precedence:
    1. Compute ``outer_strong`` from the outer ``status`` first. Only the
       exact token ``DIRECT_STRONG`` counts; ``STRONG`` and substring
       matches do not.
    2. When either richer status is present, success is
       ``outer_token == DIRECT_STRONG and nested_status == ACCEPT``.
       Missing nested status is failure. Any nested ``REJECT*`` is
       failure. Outer ``GEOMETRY_WEAK`` or ``PROVISIONAL`` plus nested
       ``ACCEPT`` stays a weak/provisional marker and is never hidden.
    3. Explicit boolean ``success`` is used only when the row lacks both
       richer outer and nested statuses. If either richer field exists,
       boolean cannot override the conjunction.
    """
    markers: list[dict[str, Any]] = []
    for row in rows:
        nested_status, _nested_accepted, has_nested = _nested_decision(row)
        outer_status = str(row.get("status") or "").strip()
        outer_token = outer_status.upper()
        outer_strong = outer_token == "DIRECT_STRONG"
        if has_nested or outer_status:
            strong = outer_strong and nested_status == "ACCEPT"
        else:
            success = row.get("success")
            strong = success in {1, True, "1", "true", "True"}
        provisional = bool(
            row.get("provisional")
            or "PROVISIONAL" in outer_token
            or outer_token in {"HELD_OUT_PROVISIONAL"}
        )
        if strong and not provisional:
            continue
        center = _pose_center(row) or _pose_center(row.get("pose"))
        if center is None:
            ref = row.get("retrieval_reference") or row.get("top1_reference")
            if isinstance(ref, Mapping):
                center = _pose_center(ref)
            if center is None:
                continue
            markers.append(
                {
                    "issue_class": "failure_retrieval_proxy",
                    "x": center[0],
                    "y": center[1],
                    "z": center[2],
                    "source": "localization_retrieval_proxy",
                    "query": row.get("query") or row.get("query_name"),
                    "status": outer_status or nested_status,
                    "nested_decision": nested_status or None,
                }
            )
            continue
        issue = "heldout_provisional" if provisional else "heldout_geometry_weak"
        markers.append(
            {
                "issue_class": issue,
                "x": center[0],
                "y": center[1],
                "z": center[2],
                "source": "localization_log",
                "query": row.get("query") or row.get("query_name"),
                "status": outer_status or nested_status,
                "nested_decision": nested_status or None,
            }
        )
    return markers


def write_ascii_ply(
    path: str | Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
) -> Path:
    points = np.asarray(xyz, dtype=float).reshape(-1, 3)
    colors = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError("xyz and rgb must have the same length")
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    lines = [header]
    for (x, y, z), (r, g, b) in zip(points, colors):
        lines.append(f"{x:.9g} {y:.9g} {z:.9g} {int(r)} {int(g)} {int(b)}\n")
    dest.write_text("".join(lines), encoding="utf-8")
    return dest


def write_risk_ply(
    map_data: MapData,
    output_dir: str | Path,
    *,
    heatmap: str | Path | Sequence[Mapping[str, Any]] | None = None,
    weak_regions: str | Path | Sequence[Mapping[str, Any]] | None = None,
    localization: str | Path | Sequence[Mapping[str, Any]] | None = None,
    extra_markers: Sequence[Mapping[str, Any]] | None = None,
    image_roles: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    sphere_radius: float | None = None,
    sphere_samples: int = 48,
    include_actloc_shadow: bool = False,
    filename: str = "localization_risk_spheres.ply",
) -> dict[str, Any]:
    """Write map RGB vertices plus colored issue-class sphere markers."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    heatmap_rows = _coerce_rows(heatmap)
    weak_rows = _coerce_rows(weak_regions)
    loc_rows = _coerce_rows(localization)
    roles = normalize_image_roles(image_roles)
    skip_keys: set[tuple[str, str]] = set()
    accepted_extra: list[dict[str, Any]] = []
    for row in extra_markers or ():
        if not isinstance(row, Mapping):
            continue
        issue = str(row.get("issue_class") or "")
        if issue in _ZERO_OBS_ISSUES:
            skip_keys |= _image_keys(row)
        accepted_extra.append(dict(row))
    markers = markers_from_map(map_data, image_roles=roles, skip_image_keys=skip_keys)
    markers.extend(markers_from_heatmap(heatmap_rows))
    markers.extend(markers_from_weak_regions(weak_rows))
    markers.extend(markers_from_localization(loc_rows))
    for row in accepted_extra:
        issue = str(row.get("issue_class") or "")
        if issue not in ISSUE_COLORS:
            continue
        if issue == "actloc_shadow" and not include_actloc_shadow:
            continue
        x, y, z = _as_float(row.get("x")), _as_float(row.get("y")), _as_float(row.get("z"))
        if x is None or y is None or z is None:
            continue
        markers.append(dict(row))

    if sphere_radius is None:
        if map_data.num_images >= 2:
            diffs = np.linalg.norm(
                map_data.image_centers[1:] - map_data.image_centers[:-1], axis=1
            )
            positive = diffs[diffs > 0]
            sphere_radius = float(np.median(positive) * 0.08) if len(positive) else 0.05
        else:
            sphere_radius = 0.05
    sphere_radius = max(float(sphere_radius), 1e-4)

    map_xyz = np.asarray(map_data.points_xyz, dtype=float).reshape(-1, 3)
    map_rgb = np.asarray(map_data.point_rgb, dtype=np.uint8).reshape(-1, 3)
    marker_xyz: list[np.ndarray] = []
    marker_rgb: list[np.ndarray] = []
    counts: dict[str, int] = {name: 0 for name in ISSUE_COLORS}
    for index, row in enumerate(markers):
        issue = str(row["issue_class"])
        counts[issue] = counts.get(issue, 0) + 1
        color = np.asarray(_rgb(issue), dtype=np.uint8)
        shell = sphere_points((row["x"], row["y"], row["z"]), sphere_radius, sphere_samples)
        marker_xyz.append(shell)
        marker_rgb.append(np.repeat(color.reshape(1, 3), len(shell), axis=0))
        row["sphere_index"] = index

    if marker_xyz:
        xyz = np.vstack([map_xyz, *marker_xyz]) if len(map_xyz) else np.vstack(marker_xyz)
        rgb = np.vstack([map_rgb, *marker_rgb]) if len(map_rgb) else np.vstack(marker_rgb)
    else:
        xyz = map_xyz
        rgb = map_rgb
    ply_path = write_ascii_ply(out / filename, xyz, rgb)
    receipt = {
        "schema_version": 1,
        "artifact_type": "SFM_DIAGNOSIS_RISK_PLY",
        "ply": str(ply_path),
        "map_vertices": int(len(map_xyz)),
        "marker_spheres": int(len(markers)),
        "sphere_samples": int(sphere_samples),
        "sphere_radius": float(sphere_radius),
        "vertex_count": int(len(xyz)),
        "counts": {key: value for key, value in counts.items() if value},
        "colors_rgb": {key: list(value) for key, value in ISSUE_COLORS.items()},
        "legend": ISSUE_LEGEND,
        "caveats": list(CAVEATS),
        "actloc": (
            "SHADOW_ONLY_IF_EXPLICITLY_INCLUDED"
            if include_actloc_shadow
            else "NOT_RUN_UNLICENSED_UNCALIBRATED_SHADOW_ONLY"
        ),
        "fim_recomputed": False,
        "inputs": {
            "heatmap_rows": len(heatmap_rows),
            "weak_region_rows": len(weak_rows),
            "localization_rows": len(loc_rows),
            "image_role_rows": len(roles),
        },
        "markers": markers,
    }
    write_json(out / "legend.json", {key: ISSUE_LEGEND[key] for key in ISSUE_COLORS})
    write_json(out / "risk_ply_receipt.json", receipt)
    return receipt


__all__ = [
    "CAVEATS",
    "ISSUE_COLORS",
    "ISSUE_LEGEND",
    "load_jsonl_rows",
    "load_rows",
    "markers_from_heatmap",
    "markers_from_localization",
    "markers_from_map",
    "markers_from_weak_regions",
    "normalize_image_roles",
    "sphere_points",
    "write_ascii_ply",
    "write_risk_ply",
]
