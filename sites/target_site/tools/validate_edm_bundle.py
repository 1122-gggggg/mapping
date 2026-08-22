#!/usr/bin/env python3
"""Validate the final EDM bundle against its model and river baseline."""

from __future__ import annotations
import sys

import argparse
import gc
import json
from pathlib import Path

import numpy as np


_CORE = Path(__file__).resolve().parents[3] / "map_update" / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))
from edm_cells import cell_identity_ok  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ts_common import hash_artifact  # noqa: E402

def anchored_counts(refs: dict) -> dict[str, int]:
    return {
        name: int(np.isfinite(np.asarray(payload["xyz_by_cell"])[:, 0]).sum())
        for name, payload in refs.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--tracking-bundle", type=Path, required=True)
    parser.add_argument("--baseline-bundle", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import pycolmap
    import torch

    baseline = torch.load(args.baseline_bundle, map_location="cpu", weights_only=False)
    baseline_median = float(baseline["meta"]["median_3d_anchored_per_ref"])
    del baseline
    gc.collect()

    tracking = torch.load(args.tracking_bundle, map_location="cpu", weights_only=False)
    tracking_names = list(tracking["ref_names"])
    del tracking
    gc.collect()

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    names = list(bundle.get("ref_names", []))
    refs = bundle.get("refs", {})
    meta = bundle.get("meta", {})
    rec = pycolmap.Reconstruction(str(args.model))
    model_names = {image.name for image in rec.images.values() if image.has_pose}
    resolutions = sorted(
        {f"{camera.width}x{camera.height}" for camera in rec.cameras.values()}
    )
    cameras_ok = bool(rec.cameras) and all(
        abs(camera.width / camera.height - 16 / 9) <= 1e-12
        for camera in rec.cameras.values()
    )
    edm_shape_ok = (
        int(meta.get("edm_input_w", 0)) == 1024
        and int(meta.get("edm_input_h", 0)) == 576
    )

    counts = anchored_counts(refs) if isinstance(refs, dict) else {}
    recomputed_total = int(sum(counts.values()))
    target_median = float(np.median(list(counts.values()))) if counts else 0.0
    reference_payload_ok = bool(
        names == tracking_names
        and names == sorted(names)
        and len(names) == len(set(names))
        and set(refs) == set(names)
        and set(names) <= model_names
        and all(count > 0 for count in counts.values())
        and all(
            np.asarray(refs[name].get("image_jpg", [])).size > 0 for name in names
        )
        and int(meta.get("refs", -1)) == len(names)
        and int(meta.get("total_3d_anchored_cells", -1)) == recomputed_total
        and np.isclose(
            float(meta.get("median_3d_anchored_per_ref", -1)), target_median
        )
    )
    checks = {
        "G8.1": cameras_ok and edm_shape_ok,
        "G8.2": target_median >= baseline_median,
        "G8.3": reference_payload_ok,
        "G8.R": cell_identity_ok(refs) if isinstance(refs, dict) else False,
    }
    result = {
        "stage": "S8_edm_bundle",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "camera_count": len(rec.cameras),
        "camera_resolutions": resolutions,
        "edm_input": [meta.get("edm_input_w"), meta.get("edm_input_h")],
        "references": len(names),
        "tracking_references": len(tracking_names),
        "recomputed_total_3d_anchored_cells": recomputed_total,
        "target_median_3d_anchored_per_ref": target_median,
        "river_baseline_median_3d_anchored_per_ref": baseline_median,
        "minimum_3d_anchored_cells_per_ref": min(counts.values()) if counts else 0,
        "provenance": {
            "script": hash_artifact(Path(__file__)),
            "sources": {},
            "input_artifacts": {
                "edm_bundle": hash_artifact(args.bundle),
                "tracking_bundle": hash_artifact(args.tracking_bundle),
                "baseline_bundle": hash_artifact(args.baseline_bundle),
                "final_model": hash_artifact(args.model),
            },
            "predecessor_gates": {},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if result["status"] != "PASS":
        raise SystemExit("S8 gate failed")


if __name__ == "__main__":
    main()
