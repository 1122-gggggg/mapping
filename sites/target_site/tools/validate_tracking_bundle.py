#!/usr/bin/env python3
"""Validate the MegaLoc/covis/yaw tracking bundle before EDM triangulation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def circular_bimodality(yaws: np.ndarray) -> dict:
    angles = np.asarray(yaws, dtype=np.float64).reshape(-1)
    if len(angles) < 4 or not np.isfinite(angles).all():
        return {"passed": False, "reason": "fewer than four finite yaws"}
    vectors = np.column_stack((np.cos(angles), np.sin(angles)))
    dot = vectors @ vectors.T
    first, second = np.unravel_index(int(np.argmin(dot)), dot.shape)
    centers = np.asarray([vectors[first], vectors[second]], dtype=np.float64)
    for _ in range(50):
        labels = np.argmax(vectors @ centers.T, axis=1)
        updated = []
        for index in (0, 1):
            members = vectors[labels == index]
            if not len(members):
                return {"passed": False, "reason": "empty yaw cluster"}
            center = members.mean(axis=0)
            norm = np.linalg.norm(center)
            if norm <= 1e-12:
                return {"passed": False, "reason": "undefined circular mean"}
            updated.append(center / norm)
        updated_array = np.asarray(updated)
        if np.allclose(updated_array, centers, atol=1e-10):
            centers = updated_array
            break
        centers = updated_array
    labels = np.argmax(vectors @ centers.T, axis=1)
    counts = [int(np.sum(labels == index)) for index in (0, 1)]
    concentrations = [
        float(np.linalg.norm(vectors[labels == index].mean(axis=0))) for index in (0, 1)
    ]
    separation = math.degrees(
        math.acos(float(np.clip(centers[0] @ centers[1], -1.0, 1.0)))
    )
    minimum_fraction = min(counts) / len(angles)
    passed = (
        separation >= 90.0
        and minimum_fraction >= 0.10
        and min(concentrations) >= 0.50
    )
    return {
        "passed": passed,
        "separation_deg": separation,
        "counts": counts,
        "minimum_cluster_fraction": minimum_fraction,
        "concentrations": concentrations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import pycolmap
    import torch

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    names = list(bundle.get("ref_names", []))
    descriptors = np.asarray(bundle.get("ref_global"), dtype=np.float32)
    rec = pycolmap.Reconstruction(str(args.model))
    model_names = {image.name for image in rec.images.values() if image.has_pose}
    manifest = json.loads(args.frame_manifest.read_text(encoding="utf-8"))
    forbidden = {
        frame["name"]
        for frame in manifest["frames"]
        if frame.get("role") == "bridge_only"
        or frame.get("motion_class") == "pure_rotation"
    }

    norms = np.linalg.norm(descriptors, axis=1) if descriptors.ndim == 2 else np.asarray([])
    descriptor_ok = bool(
        bool(names)
        and names == sorted(names)
        and len(names) == len(set(names))
        and descriptors.shape == (len(names), 8448)
        and np.isfinite(descriptors).all()
        and np.allclose(norms, 1.0, atol=1e-4, rtol=1e-4)
        and all(name in model_names and (args.image_root / name).is_file() for name in names)
        and not (set(names) & forbidden)
    )

    covis = bundle.get("covis")
    centers = np.asarray(bundle.get("ref_centers"), dtype=np.float32)
    yaws = np.asarray(bundle.get("ref_yaws"), dtype=np.float32)
    covis_ok = bool(
        isinstance(covis, dict)
        and set(covis) == set(names)
        and all(
            isinstance(index, (int, np.integer)) and 0 <= int(index) < len(names)
            for values in covis.values()
            for index in values
        )
        and centers.shape == (len(names), 3)
        and np.isfinite(centers).all()
        and yaws.shape == (len(names),)
        and np.isfinite(yaws).all()
    )
    yaw_result = circular_bimodality(yaws)
    checks = {
        "G7.1": descriptor_ok,
        "G7.2": covis_ok,
        "G7.3": bool(yaw_result.get("passed")),
    }
    result = {
        "stage": "S7_tracking_bundle",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "references": len(names),
        "descriptor_shape": list(descriptors.shape),
        "forbidden_reference_intersection": sorted(set(names) & forbidden),
        "yaw_bimodality": yaw_result,
        "median_covis_degree": (
            float(np.median([len(values) for values in covis.values()]))
            if isinstance(covis, dict) and covis
            else 0.0
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if result["status"] != "PASS":
        raise SystemExit("S7 gate failed")


if __name__ == "__main__":
    main()
