#!/usr/bin/env python3
"""Build the minimal MegaLoc seed bundle consumed by the EDM map builder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


SFMSYSTEM = Path("/media/cihcilab/新增磁碟區/sfm_system")
MEGALOC_SCRIPTS = SFMSYSTEM / "定位/source/sfm_glomap/scripts"


def active_ref_names(
    reconstruction: Any, image_root: Path, excluded_names: set[str]
) -> list[str]:
    """Return sorted, registered, on-disk reference images safe for EDM."""
    names = {
        image.name
        for image in reconstruction.images.values()
        if image.has_pose
        and image.name not in excluded_names
        and (image_root / image.name).is_file()
    }
    return sorted(names)


def validate_descriptors(
    descriptors: np.ndarray, *, expected_rows: int
) -> np.ndarray:
    """Fail closed if MegaLoc did not emit one finite unit vector per ref."""
    result = np.asarray(descriptors, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] != expected_rows:
        raise ValueError(
            f"MegaLoc row count mismatch: expected {expected_rows}, got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError("MegaLoc descriptors contain non-finite values")
    norms = np.linalg.norm(result, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4, rtol=1e-4):
        raise ValueError("MegaLoc descriptors are not L2-normalized")
    return result


def bridge_only_names(frame_manifest: Path) -> set[str]:
    payload = json.loads(frame_manifest.read_text(encoding="utf-8"))
    return {
        str(frame["name"])
        for frame in payload["frames"]
        if frame.get("role") == "bridge_only"
        or frame.get("motion_class") == "pure_rotation"
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import pycolmap
    import torch

    if str(MEGALOC_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(MEGALOC_SCRIPTS))
    import megaloc_lib  # noqa: PLC0415

    rec = pycolmap.Reconstruction(str(args.model))
    excluded = bridge_only_names(args.frame_manifest)
    names = active_ref_names(rec, args.image_root, excluded)
    if not names:
        raise SystemExit("no active references remain after bridge-only exclusion")

    descriptors = validate_descriptors(
        megaloc_lib.extract(names, args.image_root, args.device),
        expected_rows=len(names),
    )
    bundle = {
        "ref_names": names,
        "ref_global": descriptors,
        "meta": {
            "bundle_vpr": "megaloc",
            "vpr_input": 322,
            "descriptor_dim": int(descriptors.shape[1]),
            "excluded_bridge_only": len(excluded),
            "source_model": str(args.model.resolve()),
            "image_root": str(args.image_root.resolve()),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp.pt")
    torch.save(bundle, tmp)
    tmp.replace(args.out)
    print(
        f"saved {args.out}: refs={len(names)} dim={descriptors.shape[1]} "
        f"excluded_bridge_only={len(excluded)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
