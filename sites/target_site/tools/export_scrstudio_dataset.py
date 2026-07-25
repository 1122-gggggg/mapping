#!/usr/bin/env python3
"""Export the frozen GlueMap model to scrstudio/R-SCoRe scene format.

The export uses only registered reference images from the sealed EDM bundle.  Images
are linked rather than copied, poses remain camera-to-world in COLMAP/OpenCV
coordinates, and P123/P126 held-out videos are never decoded or added to training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pycolmap
import torch


SCHEMA_VERSION = 1
HELDOUT_TOKENS = ("P1230123", "P1260126")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def camera_intrinsics(camera) -> np.ndarray:
    if str(camera.model_name) != "PINHOLE":
        raise ValueError(
            f"scrstudio export requires fixed PINHOLE cameras, got {camera.model_name}"
        )
    fx, fy, cx, cy = np.asarray(camera.params, dtype=float)
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def camera_to_world_matrix(image) -> np.ndarray:
    transform = image.cam_from_world
    if callable(transform):
        transform = transform()
    matrix = np.asarray(transform.inverse().matrix(), dtype=float)
    if matrix.shape not in ((3, 4), (4, 4)):
        raise ValueError(f"unexpected camera-to-world shape: {matrix.shape}")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :4] = matrix[:3, :4]
    return pose


def select_reference_images(
    images: Mapping[int, object],
    *,
    required_names: set[str],
) -> list[object]:
    by_name = {
        str(image.name): image
        for image in images.values()
        if bool(image.has_pose)
    }
    missing = sorted(required_names - set(by_name))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} required reference images are missing: {preview}")
    return [by_name[name] for name in sorted(required_names)]


def _safe_relative_image_path(name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.name == "":
        raise ValueError(f"unsafe reference image path: {name!r}")
    return relative


def _hash_model(model: Path) -> dict[str, object]:
    rows = []
    digest = hashlib.sha256()
    for name in ("cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin"):
        path = model / name
        if not path.is_file():
            continue
        file_hash = sha256_file(path)
        rows.append({"name": name, "bytes": path.stat().st_size, "sha256": file_hash})
        digest.update(name.encode("utf-8") + b"\0" + file_hash.encode("ascii"))
    return {"sha256": digest.hexdigest(), "files": rows}


def _write_split(
    root: Path,
    records: Iterable[tuple[str, Path, np.ndarray, np.ndarray, tuple[int, int]]],
) -> dict[str, object]:
    ordered = list(records)
    rgb = root / "rgb"
    rgb.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    poses: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    shapes: list[tuple[int, int]] = []
    for name, source, pose, calibration, shape in ordered:
        relative = _safe_relative_image_path(name)
        destination = rgb / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source.resolve())
        names.append(relative.as_posix())
        poses.append(pose)
        intrinsics.append(calibration)
        shapes.append(shape)
    np.save(root / "poses.npy", np.stack(poses))
    np.save(root / "calibration.npy", np.stack(intrinsics))
    np.save(root / "image_shapes.npy", np.asarray(shapes, dtype=np.int32))
    (root / "file_list.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return {
        "images": len(names),
        "file_list_sha256": sha256_file(root / "file_list.txt"),
        "poses_sha256": sha256_file(root / "poses.npy"),
        "calibration_sha256": sha256_file(root / "calibration.npy"),
        "image_shapes_sha256": sha256_file(root / "image_shapes.npy"),
    }


def export_dataset(
    *,
    sfm_model: Path,
    image_root: Path,
    reference_bundle: Path,
    output: Path,
    diagnostic_val_stride: int = 20,
) -> dict[str, object]:
    if diagnostic_val_stride <= 0:
        raise ValueError("diagnostic_val_stride must be positive")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing export: {output}")
    bundle = torch.load(reference_bundle, map_location="cpu", weights_only=False)
    if str(bundle.get("meta", {}).get("feature", "")).lower() != "edm":
        raise ValueError("reference bundle must be the original sealed EDM map")
    required_names = set(str(name) for name in bundle["ref_names"])
    heldout_hits = sorted(
        name for name in required_names if any(token in name for token in HELDOUT_TOKENS)
    )
    if heldout_hits:
        raise ValueError(f"held-out inputs leaked into reference bundle: {heldout_hits[:3]}")

    reconstruction = pycolmap.Reconstruction(str(sfm_model))
    images = select_reference_images(
        reconstruction.images,
        required_names=required_names,
    )
    records = []
    for image in images:
        source = image_root / _safe_relative_image_path(str(image.name))
        if not source.is_file():
            raise FileNotFoundError(f"reference image is unavailable: {source}")
        camera = reconstruction.cameras[image.camera_id]
        records.append(
            (
                str(image.name),
                source,
                camera_to_world_matrix(image),
                camera_intrinsics(camera),
                (int(camera.height), int(camera.width)),
            )
        )

    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary export exists: {temporary}")
    temporary.mkdir(parents=True)
    train_stats = _write_split(temporary / "train", records)
    validation_records = records[::diagnostic_val_stride]
    val_stats = _write_split(temporary / "val", validation_records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "format": "scrstudio_dsac_packed_arrays_with_image_folder",
        "scene": "target_site",
        "source": {
            "sfm_model": str(sfm_model.resolve()),
            "sfm_model_hash": _hash_model(sfm_model),
            "reference_bundle": str(reference_bundle.resolve()),
            "reference_bundle_sha256": sha256_file(reference_bundle),
            "image_root": str(image_root.resolve()),
        },
        "coordinate_convention": "COLMAP/OpenCV +X right +Y down +Z forward",
        "splits": {
            "train": train_stats,
            "val": {
                **val_stats,
                "diagnostic_only": True,
                "is_subset_of_train": True,
                "stride": diagnostic_val_stride,
            },
        },
        "heldout_policy": {
            "excluded_tokens": list(HELDOUT_TOKENS),
            "heldout_videos_decoded": False,
            "absolute_pose_accuracy_claim_allowed": False,
        },
        "reader_override_required": (
            "Use ImageFolderReaderConfig(data='rgb') until an isolated scrstudio "
            "environment builds rgb_lmdb."
        ),
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.rename(output)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfm-model", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--reference-bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--diagnostic-val-stride", type=int, default=20)
    args = parser.parse_args()
    manifest = export_dataset(
        sfm_model=args.sfm_model.expanduser().resolve(strict=True),
        image_root=args.image_root.expanduser().resolve(strict=True),
        reference_bundle=args.reference_bundle.expanduser().resolve(strict=True),
        output=args.output.expanduser().resolve(),
        diagnostic_val_stride=args.diagnostic_val_stride,
    )
    print(json.dumps(manifest["splits"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
