#!/usr/bin/env python3
"""Validate the final EDM bundle against its model and river baseline."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from edm_gate_contract import require_fresh_v2_gate
from ts_common import Gate


EXPECTED_PAIR_TOPK = 30
EXPECTED_ANCHOR_BATCH_SIZE = 64
EXPECTED_BASELINE_PATH = (
    Path(__file__).resolve().parents[3]
    / "EDM定位測試"
    / "outputs"
    / "river_site_reloc_map_edm.pt"
)
EXPECTED_BASELINE_SHA256 = (
    "39a817936c0ba314a739701411f974672a94f126830f9f5b9a7a4efdfee08117"
)
EXPECTED_BASELINE_MEDIAN = 3811.5
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COLMAP_MODEL_STEMS = ("cameras", "images", "points3D", "rigs", "frames")
COLMAP_REQUIRED_STEMS = frozenset(("cameras", "images", "points3D"))


def anchored_counts(refs: dict) -> dict[str, int]:
    return {
        name: int(np.isfinite(np.asarray(payload["xyz_by_cell"])[:, 0]).sum())
        for name, payload in refs.items()
    }


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    source = path.expanduser().resolve(strict=True)
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = source.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if tuple(getattr(before, field) for field in fields) != tuple(
        getattr(after, field) for field in fields
    ):
        raise RuntimeError(f"file changed while hashing: {source}")
    return digest.hexdigest()


def source_model_sha256(model: Path) -> str:
    """Recompute the exact semantic model digest stored by the EDM builder."""
    root = model.expanduser().resolve(strict=True)
    files: list[dict[str, str]] = []
    found: set[str] = set()
    for stem in COLMAP_MODEL_STEMS:
        for suffix in (".bin", ".txt"):
            path = root / f"{stem}{suffix}"
            if path.is_file():
                files.append({"name": path.name, "sha256": _file_sha256(path)})
                found.add(stem)
    missing = sorted(COLMAP_REQUIRED_STEMS - found)
    if missing:
        raise FileNotFoundError(f"COLMAP model {root} lacks components: {missing}")
    return _canonical_json_sha256({"schema_version": 1, "files": files})


def build_contract_checks(
    meta: dict,
    *,
    current_model_sha256: str,
    current_tracking_bundle_sha256: str,
) -> dict[str, bool]:
    source_digest = str(meta.get("source_model_sha256", ""))
    return {
        "pair_topk_30": int(meta.get("triangulation_pair_topk", -1))
        == EXPECTED_PAIR_TOPK,
        "anchor_batch_size_64": int(
            meta.get("triangulation_anchor_batch_size", -1)
        )
        == EXPECTED_ANCHOR_BATCH_SIZE,
        "source_model_fresh": bool(
            SHA256_PATTERN.fullmatch(source_digest)
            and source_digest == current_model_sha256
        ),
        "source_tracking_bundle_fresh": bool(
            SHA256_PATTERN.fullmatch(
                str(meta.get("source_tracking_bundle_sha256", ""))
            )
            and meta.get("source_tracking_bundle_sha256")
            == current_tracking_bundle_sha256
        ),
        "pair_provenance_hashed": bool(
            SHA256_PATTERN.fullmatch(str(meta.get("pair_provenance_sha256", "")))
        ),
        "pair_artifact_hashed": bool(
            SHA256_PATTERN.fullmatch(str(meta.get("pair_artifact_sha256", "")))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--tracking-bundle", type=Path, required=True)
    parser.add_argument(
        "--baseline-bundle", type=Path, default=EXPECTED_BASELINE_PATH
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--predecessor-gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    gate = Gate(
        "S8_edm_bundle",
        {"G8.0", "G8.1", "G8.2", "G8.3", "G8.4"},
        script_path=__file__,
        source_files=[Path(__file__).with_name("edm_gate_contract.py")],
        input_artifacts={
            "edm_bundle": args.bundle,
            "tracking_bundle": args.tracking_bundle,
            "anchor_density_baseline": args.baseline_bundle,
            "final_model": args.model,
        },
    )
    gate.record_predecessor_gate(
        "S7_tracking_bundle",
        args.predecessor_gate,
        expected_stage="S7_tracking_bundle",
    )
    predecessor_error = None
    try:
        require_fresh_v2_gate(
            args.predecessor_gate, expected_stage="S7_tracking_bundle"
        )
    except (OSError, ValueError, RuntimeError) as error:
        predecessor_error = str(error)
    gate.check(
        "G8.0",
        predecessor_error is None,
        "S7 predecessor is a fresh sfm-gate-v2 PASS",
        error=predecessor_error,
    )

    import pycolmap
    import torch

    baseline_path = args.baseline_bundle.expanduser().resolve(strict=True)
    baseline_sha256 = _file_sha256(baseline_path)
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    baseline_median = float(baseline["meta"]["median_3d_anchored_per_ref"])
    baseline_contract_ok = bool(
        baseline_path == EXPECTED_BASELINE_PATH.resolve(strict=True)
        and baseline_sha256 == EXPECTED_BASELINE_SHA256
        and baseline_median == EXPECTED_BASELINE_MEDIAN
    )
    del baseline
    gc.collect()

    tracking = torch.load(args.tracking_bundle, map_location="cpu", weights_only=False)
    tracking_names = list(tracking["ref_names"])
    tracking_digest = _file_sha256(args.tracking_bundle)
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
    current_model_digest = source_model_sha256(args.model)
    build_checks = build_contract_checks(
        meta,
        current_model_sha256=current_model_digest,
        current_tracking_bundle_sha256=tracking_digest,
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
    gate.check(
        "G8.1",
        all(build_checks.values()),
        "EDM build knobs and model/tracking source hashes are current",
        build_contract=build_checks,
        current_source_model_sha256=current_model_digest,
        current_tracking_bundle_sha256=tracking_digest,
        bundle_source_model_sha256=meta.get("source_model_sha256"),
        bundle_source_tracking_bundle_sha256=meta.get(
            "source_tracking_bundle_sha256"
        ),
        pair_provenance_sha256=meta.get("pair_provenance_sha256"),
        pair_artifact_sha256=meta.get("pair_artifact_sha256"),
    )
    gate.check(
        "G8.2",
        cameras_ok and edm_shape_ok,
        "model cameras and EDM tensor geometry match the Fuhe contract",
        camera_count=len(rec.cameras),
        camera_resolutions=resolutions,
        edm_input=[meta.get("edm_input_w"), meta.get("edm_input_h")],
    )
    gate.check(
        "G8.3",
        bool(baseline_contract_ok and target_median >= baseline_median),
        "anchor density meets the immutable predeclared river baseline",
        target_median_3d_anchored_per_ref=target_median,
        baseline_path=str(baseline_path),
        baseline_sha256=baseline_sha256,
        baseline_median_3d_anchored_per_ref=baseline_median,
        baseline_contract_ok=baseline_contract_ok,
    )
    gate.check(
        "G8.4",
        reference_payload_ok,
        "EDM reference payload is complete and aligned to the tracking bundle",
        references=len(names),
        tracking_references=len(tracking_names),
        recomputed_total_3d_anchored_cells=recomputed_total,
        minimum_3d_anchored_cells_per_ref=(
            min(counts.values()) if counts else 0
        ),
    )
    gate.write(args.out.parent.parent, output_path=args.out)


if __name__ == "__main__":
    main()
