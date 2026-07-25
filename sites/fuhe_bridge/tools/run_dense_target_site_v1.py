#!/usr/bin/env python3
"""Create an RGB dense PLY from the frozen target_site_v1 COLMAP model.

This script only writes inside the supplied workspace.  A completed PLY is
never overwritten; pass --resume to inspect an already-completed workspace.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DensePlan:
    workspace: Path
    fused_ply: Path
    patch_marker: Path
    fusion_marker: Path
    needs_undistortion: bool
    needs_patch_match: bool
    needs_fusion: bool


def build_plan(workspace: Path, *, allow_completed: bool = False) -> DensePlan:
    """Return the phase plan without changing the workspace."""
    workspace = workspace.resolve()
    fused_ply = workspace / "fused.ply"
    if fused_ply.exists() and not allow_completed:
        raise FileExistsError(
            f"Refusing to overwrite completed dense map: {fused_ply}. "
            "Use a new workspace."
        )
    patch_marker = workspace / "patch_match.complete.json"
    fusion_marker = workspace / "fusion.complete.json"
    undistorted = (
        (workspace / "images").is_dir()
        and (workspace / "sparse").is_dir()
        and (workspace / "stereo" / "patch-match.cfg").is_file()
    )
    return DensePlan(
        workspace=workspace,
        fused_ply=fused_ply,
        patch_marker=patch_marker,
        fusion_marker=fusion_marker,
        needs_undistortion=not undistorted,
        needs_patch_match=not patch_marker.is_file(),
        needs_fusion=not fusion_marker.is_file() or not fused_ply.is_file(),
    )


def validate_input_layout(model_dir: Path, image_dir: Path) -> None:
    """Fail early if the immutable sparse model or its RGB images are absent."""
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        candidate = model_dir / name
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing sparse-model file: {candidate}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing RGB image directory: {image_dir}")


def write_marker(path: Path, **data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def count_depth_maps(workspace: Path) -> int:
    return len(list((workspace / "stereo" / "depth_maps").glob("*.geometric.bin")))


def fusion_kwargs() -> dict[str, str]:
    """Keep the fusion output a portable PLY, not a COLMAP model directory."""
    return {"input_type": "geometric", "output_type": "ply"}


def run(args: argparse.Namespace) -> int:
    import pycolmap

    model_dir = args.model.resolve()
    image_dir = args.images.resolve()
    validate_input_layout(model_dir, image_dir)
    if not pycolmap.has_cuda:
        raise RuntimeError("PyCOLMAP was built without CUDA; refusing CPU dense reconstruction.")

    reconstruction = pycolmap.Reconstruction(model_dir)
    image_names = sorted(image.name for image in reconstruction.images.values())
    missing = [name for name in image_names if not (image_dir / name).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} registered RGB images are missing (e.g. {preview})")

    plan = build_plan(args.workspace, allow_completed=args.resume)
    if plan.fused_ply.is_file() and args.resume:
        print(f"Dense RGB PLY already complete: {plan.fused_ply}")
        return 0
    plan.workspace.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(
        f"CUDA dense reconstruction: {len(image_names)} registered RGB images, "
        f"max_image_size={args.max_image_size}, workspace={plan.workspace}",
        flush=True,
    )

    if plan.needs_undistortion:
        print("[1/3] Undistorting images", flush=True)
        pycolmap.undistort_images(
            output_path=plan.workspace,
            input_path=model_dir,
            image_path=image_dir,
            image_names=image_names,
            output_type="COLMAP",
            num_patch_match_src_images=args.num_source_images,
            jpeg_quality=95,
            num_threads=args.num_threads,
        )

    if plan.needs_patch_match:
        print("[2/3] CUDA PatchMatch Stereo", flush=True)
        options = pycolmap.PatchMatchOptions()
        options.max_image_size = args.max_image_size
        options.gpu_index = str(args.gpu_index)
        options.geom_consistency = True
        options.filter = True
        options.cache_size = args.cache_size_gb
        options.num_threads = args.num_threads
        pycolmap.patch_match_stereo(plan.workspace, options=options)
        write_marker(
            plan.patch_marker,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            depth_maps=count_depth_maps(plan.workspace),
            max_image_size=args.max_image_size,
        )

    if plan.needs_fusion:
        print("[3/3] Fusing geometric depth maps into RGB PLY", flush=True)
        options = pycolmap.StereoFusionOptions()
        options.max_image_size = args.max_image_size
        options.min_num_pixels = args.min_num_pixels
        options.num_threads = args.num_threads
        options.cache_size = args.cache_size_gb
        pycolmap.stereo_fusion(plan.fused_ply, plan.workspace, options=options, **fusion_kwargs())
        if not plan.fused_ply.is_file() or plan.fused_ply.stat().st_size == 0:
            raise RuntimeError(f"Stereo fusion did not produce a PLY: {plan.fused_ply}")
        write_marker(
            plan.fusion_marker,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            fused_ply=str(plan.fused_ply),
            bytes=plan.fused_ply.stat().st_size,
            elapsed_seconds=round(time.time() - started, 1),
        )
    print(f"Complete: {plan.fused_ply}", flush=True)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    run_root = Path(__file__).parents[1] / "runs" / "target_site_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=run_root / "final_model")
    parser.add_argument("--images", type=Path, default=run_root / "images")
    parser.add_argument("--workspace", type=Path, default=run_root / "dense_mvs_20260719")
    parser.add_argument("--max-image-size", type=int, default=2000)
    parser.add_argument("--num-source-images", type=int, default=20)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--cache-size-gb", type=float, default=24.0)
    parser.add_argument("--num-threads", type=int, default=12)
    parser.add_argument("--min-num-pixels", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(sys.argv[1:] if argv is None else argv))
    except (FileExistsError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
