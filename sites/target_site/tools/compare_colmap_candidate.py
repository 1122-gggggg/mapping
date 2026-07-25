#!/usr/bin/env python3
"""Compare an SfM refinement candidate with its exact source reconstruction.

This is a structural and internal-consistency gate, not an absolute-accuracy test:
the target-site corpus has no independent metric pose ground truth.  The candidate
must preserve cameras, image assignments, point IDs, and every track observation.
Only then are pose/point motion and reprojection-error changes reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def _plain(value: Any) -> Any:
    return value() if callable(value) else value


def _registered(image: Any) -> bool:
    for name in ("has_pose", "registered"):
        if hasattr(image, name):
            return bool(_plain(getattr(image, name)))
    return True


def _digest_row(digest: Any, row: Any) -> None:
    digest.update(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"\n")


def topology_signature(reconstruction: Any) -> dict[str, Any]:
    """Hash registered images, point IDs, and exact observation topology."""
    digest = hashlib.sha256()
    registered = []
    for image_id, image in sorted(reconstruction.images.items()):
        if not _registered(image):
            continue
        row = ["image", int(image_id), str(image.name), int(image.camera_id)]
        registered.append(row)
        _digest_row(digest, row)

    observations = 0
    for point_id, point in sorted(reconstruction.points3D.items()):
        elements = getattr(point.track, "elements", point.track)
        track = sorted(
            (int(element.image_id), int(element.point2D_idx)) for element in elements
        )
        observations += len(track)
        _digest_row(digest, ["point3D", int(point_id), track])

    return {
        "registered_images": len(registered),
        "points3D": len(reconstruction.points3D),
        "observations": observations,
        "sha256": digest.hexdigest(),
    }


def distribution_summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty distribution")
    if not np.isfinite(array).all():
        raise ValueError("distribution contains non-finite values")
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def rotation_delta_degrees(before: np.ndarray, after: np.ndarray) -> float:
    relative = np.asarray(after, dtype=np.float64) @ np.asarray(
        before, dtype=np.float64
    ).T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def reprojection_metrics(reconstruction: Any) -> dict[str, float | int]:
    """Recompute pixel residuals from current poses/points instead of stored errors."""
    residuals: list[float] = []
    observations = 0
    invalid_or_behind = 0
    for _, image in sorted(reconstruction.images.items()):
        if not _registered(image):
            continue
        observed = [
            point2D
            for point2D in image.points2D
            if bool(_plain(point2D.has_point3D))
        ]
        if not observed:
            continue
        observations += len(observed)
        xyz = np.asarray(
            [
                reconstruction.points3D[int(point2D.point3D_id)].xyz
                for point2D in observed
            ],
            dtype=np.float64,
        )
        expected = np.asarray([point2D.xy for point2D in observed], dtype=np.float64)
        camera_xyz = np.asarray(image.cam_from_world() * xyz, dtype=np.float64)
        valid = np.isfinite(camera_xyz).all(axis=1) & (camera_xyz[:, 2] > 0.0)
        invalid_or_behind += int((~valid).sum())
        if not valid.any():
            continue
        projected = np.asarray(
            reconstruction.cameras[int(image.camera_id)].img_from_cam(camera_xyz[valid]),
            dtype=np.float64,
        )
        finite_projection = np.isfinite(projected).all(axis=1)
        invalid_or_behind += int((~finite_projection).sum())
        if finite_projection.any():
            residuals.extend(
                np.linalg.norm(
                    projected[finite_projection] - expected[valid][finite_projection],
                    axis=1,
                ).tolist()
            )

    summary = distribution_summary(residuals)
    return {
        "observations": observations,
        "valid_observations": len(residuals),
        "invalid_or_behind": invalid_or_behind,
        "mean_px": float(np.mean(np.asarray(residuals, dtype=np.float64))),
        "median_px": summary["median"],
        "p95_px": summary["p95"],
        "max_px": summary["max"],
    }


def _rotation_matrix(image: Any) -> np.ndarray:
    pose = image.cam_from_world()
    rotation = pose.rotation
    matrix = rotation.matrix() if callable(rotation.matrix) else rotation.matrix
    return np.asarray(matrix, dtype=np.float64)


def _projection_center(image: Any) -> np.ndarray:
    center = image.projection_center()
    return np.asarray(center, dtype=np.float64)


def _camera_signature(reconstruction: Any) -> dict[str, Any]:
    cameras = {}
    for camera_id, camera in sorted(reconstruction.cameras.items()):
        model = getattr(camera, "model_name", getattr(camera, "model", "unknown"))
        model = _plain(model)
        model = getattr(model, "name", model)
        cameras[str(int(camera_id))] = {
            "model": str(model),
            "width": int(camera.width),
            "height": int(camera.height),
            "params": [
                float(value) for value in np.asarray(camera.params).reshape(-1)
            ],
        }
    return cameras


def compare_reconstructions(source: Any, candidate: Any) -> dict[str, Any]:
    source_topology = topology_signature(source)
    candidate_topology = topology_signature(candidate)
    topology_exact = source_topology == candidate_topology
    intrinsics_exact = _camera_signature(source) == _camera_signature(candidate)

    registered_source = {
        int(image_id): image
        for image_id, image in source.images.items()
        if _registered(image)
    }
    registered_candidate = {
        int(image_id): image
        for image_id, image in candidate.images.items()
        if _registered(image)
    }
    image_ids_exact = set(registered_source) == set(registered_candidate)
    point_ids_exact = set(source.points3D) == set(candidate.points3D)

    geometry: dict[str, Any] | None = None
    finite_geometry = False
    if topology_exact and image_ids_exact and point_ids_exact:
        center_deltas = []
        rotation_deltas = []
        for image_id in sorted(registered_source):
            before = registered_source[image_id]
            after = registered_candidate[image_id]
            center_deltas.append(
                float(np.linalg.norm(_projection_center(after) - _projection_center(before)))
            )
            rotation_deltas.append(
                rotation_delta_degrees(
                    _rotation_matrix(before), _rotation_matrix(after)
                )
            )
        point_deltas = [
            float(
                np.linalg.norm(
                    np.asarray(candidate.points3D[point_id].xyz, dtype=np.float64)
                    - np.asarray(source.points3D[point_id].xyz, dtype=np.float64)
                )
            )
            for point_id in sorted(source.points3D)
        ]
        geometry = {
            "camera_center_delta_map_units": distribution_summary(center_deltas),
            "camera_rotation_delta_degrees": distribution_summary(rotation_deltas),
            "point3D_delta_map_units": distribution_summary(point_deltas),
        }
        finite_geometry = all(
            np.isfinite(value)
            for group in geometry.values()
            for value in group.values()
        )

    source_reprojection = reprojection_metrics(source)
    candidate_reprojection = reprojection_metrics(candidate)
    before_error = float(source_reprojection["mean_px"])
    after_error = float(candidate_reprojection["mean_px"])
    finite_reprojection = bool(np.isfinite([before_error, after_error]).all())
    reprojection_tolerance = max(1e-9, abs(before_error) * 1e-9)
    reprojection_nonworse = (
        finite_reprojection and after_error <= before_error + reprojection_tolerance
    )
    observation_validity_preserved = (
        source_reprojection["observations"] == candidate_reprojection["observations"]
        and source_reprojection["invalid_or_behind"]
        == candidate_reprojection["invalid_or_behind"]
    )
    gates = {
        "topology_exact": topology_exact,
        "registered_image_ids_exact": image_ids_exact,
        "point3D_ids_exact": point_ids_exact,
        "intrinsics_exact": intrinsics_exact,
        "finite_geometry": finite_geometry,
        "finite_reprojection": finite_reprojection,
        "observation_validity_preserved": observation_validity_preserved,
        "mean_reprojection_nonworse": reprojection_nonworse,
    }
    return {
        "schema": "colmap-refinement-comparison/v1",
        "interpretation": (
            "internal consistency only; no independent absolute pose ground truth"
        ),
        "source_topology": source_topology,
        "candidate_topology": candidate_topology,
        "reprojection_error_px": {
            "source": source_reprojection,
            "candidate": candidate_reprojection,
            "relative_change": (
                (after_error - before_error) / before_error
                if before_error > 0.0
                else None
            ),
            "nonworse_absolute_tolerance_px": reprojection_tolerance,
        },
        "geometry_delta": geometry,
        "gates": gates,
        "structurally_eligible": all(gates.values()),
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    import pycolmap

    args = parse_args(argv)
    source_path = args.source_model.expanduser().resolve()
    candidate_path = args.candidate_model.expanduser().resolve()
    report = compare_reconstructions(
        pycolmap.Reconstruction(source_path),
        pycolmap.Reconstruction(candidate_path),
    )
    report["source_model"] = str(source_path)
    report["candidate_model"] = str(candidate_path)
    _atomic_json(args.out.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["structurally_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
