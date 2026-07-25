#!/usr/bin/env python3
"""Run PixSfM feature-metric BA as an isolated, fixed-intrinsics candidate.

This runner deliberately starts from an existing COLMAP reconstruction.  It does not
read a match database, create pairs, rematch images, triangulate tracks, or run PGO.
PixSfM may refine camera extrinsics and 3D point positions only.  The result is first
written to ``model.partial`` and is promoted to ``model`` only after exact camera and
image-to-camera assignment gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve()
SYSTEM_ROOT = HERE.parents[2]
TARGET_RUN = HERE.parents[1] / "runs" / "target_site_v1"
DEFAULT_PIXSFM_PYTHON = Path("/home/cihcilab/micromamba/envs/pixsfm-fuhe/bin/python")
MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin")


def fixed_intrinsics_config(
    *, max_edge: int = 1600, max_iterations: int = 30, num_threads: int = 4
) -> dict[str, Any]:
    """Return the only accepted PixSfM mapping configuration for this study."""
    if max_edge <= 0 or max_iterations <= 0 or num_threads <= 0:
        raise ValueError("max_edge, max_iterations, and num_threads must be positive")
    return {
        "mapping": {
            "dense_features": {
                "model": {"name": "s2dnet", "num_layers": 1},
                "device": "cuda",
                # PixSfM 1.0's HighFive bulk loader cannot construct an f16
                # FeatureView with the HDF5 2.1 runtime in the pinned
                # environment (single-patch reads work, bulk reads fail with
                # create_and_check_datatype).  FP32 is the stable execution
                # path and can safely convert an existing FP16 H5 cache while
                # loading, so feature extraction does not need to be repeated.
                "dtype": "float",
                "sparse": True,
                "max_edge": int(max_edge),
                "patch_size": 8,
                "use_cache": True,
                "overwrite_cache": False,
                "load_cache_on_init": False,
                "cache_format": "chunked",
            },
            # refine_reconstruction runs BA only, but keep this explicit so the
            # candidate contract cannot be mistaken for a KA + BA remapping job.
            "KA": {"apply": False},
            "BA": {
                "apply": True,
                "strategy": "feature_reference",
                "repeats": 1,
                "max_tracks_per_problem": 10,
                "num_threads": int(num_threads),
                "optimizer": {
                    "solver": {
                        "max_num_iterations": int(max_iterations),
                        "num_threads": int(num_threads),
                    },
                    "print_summary": True,
                    "refine_focal_length": False,
                    "refine_principal_point": False,
                    "refine_extra_params": False,
                    "refine_extrinsics": True,
                },
                "references": {"num_threads": int(num_threads)},
            },
        }
    }


def _plain(value: Any) -> Any:
    return value() if callable(value) else value


def _camera_model_name(camera: Any) -> str:
    name = getattr(camera, "model_name", None)
    if name is not None:
        return str(_plain(name))
    model = getattr(camera, "model", None)
    return str(getattr(model, "name", model))


def _image_is_registered(image: Any) -> bool:
    for name in ("has_pose", "registered"):
        if hasattr(image, name):
            return bool(_plain(getattr(image, name)))
    return True


def intrinsics_signature(reconstruction: Any) -> dict[str, Any]:
    """Capture cameras plus shared-camera assignments using version-neutral access."""
    cameras = {}
    for camera_id, camera in sorted(reconstruction.cameras.items()):
        cameras[str(int(camera_id))] = {
            "model": _camera_model_name(camera),
            "width": int(camera.width),
            "height": int(camera.height),
            "params": [float(value) for value in np.asarray(camera.params).reshape(-1)],
        }
    assignments = {
        str(image.name): int(image.camera_id)
        for _, image in sorted(reconstruction.images.items())
        if _image_is_registered(image)
    }
    return {"cameras": cameras, "assignments": dict(sorted(assignments.items()))}


def assert_intrinsics_unchanged(before: dict[str, Any], after: dict[str, Any]) -> None:
    before_cameras = before["cameras"]
    after_cameras = after["cameras"]
    if set(before_cameras) != set(after_cameras):
        raise RuntimeError("camera IDs changed")
    for camera_id in sorted(before_cameras):
        lhs, rhs = before_cameras[camera_id], after_cameras[camera_id]
        if (lhs["model"], lhs["width"], lhs["height"]) != (
            rhs["model"],
            rhs["width"],
            rhs["height"],
        ):
            raise RuntimeError(
                f"camera model or dimensions changed for camera {camera_id}"
            )
        if lhs["params"] != rhs["params"]:
            raise RuntimeError(f"camera parameters changed for camera {camera_id}")
    if before["assignments"] != after["assignments"]:
        raise RuntimeError("registered image camera assignments changed")


def _overlaps(path: Path, protected: Path) -> bool:
    return path == protected or protected in path.parents or path in protected.parents


def validate_isolated_output(
    input_model: Path, output_model: Path, protected_paths: Iterable[Path]
) -> Path:
    """Reject output paths that contain, or are contained by, protected artifacts."""
    source = Path(input_model).resolve()
    output = Path(output_model).resolve()
    protected = {source, *(Path(path).resolve() for path in protected_paths)}
    collision = next((path for path in protected if _overlaps(output, path)), None)
    if collision is not None:
        raise ValueError(f"output overlaps protected path: {collision}")
    return output


def select_feature_cache(
    output_root: Path, reuse_feature_cache: Path | None
) -> tuple[Path, bool]:
    """Select a cache path without copying a potentially multi-gigabyte artifact."""
    if reuse_feature_cache is None:
        return (Path(output_root) / "s2dnet_featuremaps_sparse.h5").resolve(), False
    cache = Path(reuse_feature_cache).expanduser().resolve()
    if not cache.is_file():
        raise FileNotFoundError(f"reused PixSfM feature cache does not exist: {cache}")
    return cache, True


def build_pixsfm_command(
    *,
    python: Path,
    input_model: Path,
    output_model: Path,
    image_root: Path,
    config_path: Path,
    cache_path: Path,
) -> list[str]:
    """Build an FBA-only command; no database or match artifact is accepted."""
    return [
        str(Path(python)),
        "-m",
        "pixsfm.refine_colmap",
        "ba",
        "--input_path",
        str(Path(input_model).resolve()),
        "--output_path",
        str(Path(output_model).resolve()),
        "--image_dir",
        str(Path(image_root).resolve()),
        "--config",
        str(Path(config_path).resolve()),
        "--cache_path",
        str(Path(cache_path).resolve()),
    ]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(model_dir: Path) -> dict[str, Any]:
    root = Path(model_dir).resolve()
    files = []
    combined = hashlib.sha256()
    for name in MODEL_FILES:
        path = root / name
        if not path.is_file():
            continue
        digest = sha256_file(path)
        row = {"name": name, "bytes": path.stat().st_size, "sha256": digest}
        files.append(row)
        combined.update(json.dumps(row, sort_keys=True).encode("utf-8"))
    required = {"cameras.bin", "images.bin", "points3D.bin"}
    if not required.issubset({row["name"] for row in files}):
        raise FileNotFoundError(f"incomplete COLMAP model: {root}")
    return {"path": str(root), "sha256": combined.hexdigest(), "files": files}


def model_summary(reconstruction: Any) -> dict[str, Any]:
    def number(name: str) -> int:
        return int(_plain(getattr(reconstruction, name)))

    reprojection = None
    method = getattr(reconstruction, "compute_mean_reprojection_error", None)
    if method is not None and number("num_points3D"):
        reprojection = float(method())
    return {
        "registered_images": number("num_reg_images"),
        "cameras": len(reconstruction.cameras),
        "points3D": number("num_points3D"),
        "mean_reprojection_error": reprojection,
    }


def assert_structure_preserved(before: dict[str, Any], after: dict[str, Any]) -> None:
    if before["registered_images"] != after["registered_images"]:
        raise RuntimeError("registered image count changed")
    if before["points3D"] != after["points3D"]:
        raise RuntimeError("3D point count changed")
    if before["cameras"] != after["cameras"]:
        raise RuntimeError("camera count changed")


def memory_snapshot() -> dict[str, float]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0])
    return {
        "available_gib": values.get("MemAvailable", 0) / 1024**2,
        "swap_free_gib": values.get("SwapFree", 0) / 1024**2,
    }


def _atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _write_config(path: Path, config: dict[str, Any]) -> None:
    # JSON is valid YAML and avoids adding a serializer dependency to the controller.
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _preflight_pixsfm(python: Path) -> dict[str, str]:
    if not python.is_file():
        raise FileNotFoundError(f"PixSfM Python does not exist: {python}")
    probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import importlib.metadata as m, pixsfm, pycolmap; "
                "print(m.version('pixsfm')); print(pycolmap.__version__); "
                "print(pixsfm.__file__)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    lines = [line for line in probe.stdout.splitlines() if line]
    return {"pixsfm": lines[0], "pycolmap": lines[1], "module": lines[2]}


def compare_reconstructions(source: Any, candidate: Any) -> dict[str, Any]:
    """Load the shared geometry comparator only for a completed candidate."""
    import sys

    tools_dir = str(HERE.parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from compare_colmap_candidate import compare_reconstructions as compare

    return compare(source, candidate)


def comparison_reject_reasons(comparison: dict[str, Any]) -> list[str]:
    """Explain every failed comparator gate while treating malformed output as unsafe."""
    error = comparison.get("error")
    if error:
        return [f"comparison failed: {error}"]
    gates = comparison.get("gates")
    if not isinstance(gates, dict):
        return ["comparison result lacks gates"]
    reasons = [
        f"comparison gate failed: {name}"
        for name, passed in sorted(gates.items())
        if passed is not True
    ]
    if comparison.get("structurally_eligible") is not True and not reasons:
        reasons.append("comparison did not mark candidate structurally eligible")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", type=Path, default=TARGET_RUN / "final_model")
    parser.add_argument("--image-root", type=Path, default=TARGET_RUN / "images")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pixsfm-python", type=Path, default=DEFAULT_PIXSFM_PYTHON)
    parser.add_argument("--max-edge", type=int, default=1600)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--minimum-available-ram-gib", type=float, default=8.0)
    parser.add_argument(
        "--reuse-feature-cache",
        type=Path,
        help="read an existing PixSfM H5 cache in place; never copies or modifies it",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_model = args.input_model.resolve()
    image_root = args.image_root.resolve()
    output_root = args.output_root.resolve()
    output_model = output_root / "model"
    partial_model = output_root / "model.partial"
    protected = [
        TARGET_RUN / "final_model",
        TARGET_RUN / "edm",
        SYSTEM_ROOT / "EDM定位測試" / "transfer" / "release",
    ]
    validate_isolated_output(input_model, output_model, protected)
    validate_isolated_output(input_model, partial_model, protected)
    if not image_root.is_dir():
        raise FileNotFoundError(f"image root does not exist: {image_root}")
    if output_model.exists() or partial_model.exists():
        raise FileExistsError(
            f"candidate output already exists; choose a new --output-root: {output_root}"
        )

    memory = memory_snapshot()
    if memory["available_gib"] < args.minimum_available_ram_gib:
        raise RuntimeError(
            f"only {memory['available_gib']:.2f} GiB RAM available; "
            f"requires {args.minimum_available_ram_gib:.2f} GiB"
        )

    source_fingerprint = model_fingerprint(input_model)
    pixsfm_runtime = _preflight_pixsfm(args.pixsfm_python.resolve())

    output_root.mkdir(parents=True, exist_ok=False)
    config_path = output_root / "pixsfm_fixed_intrinsics.yaml"
    cache_path, cache_reused = select_feature_cache(
        output_root, args.reuse_feature_cache
    )
    config = fixed_intrinsics_config(
        max_edge=args.max_edge,
        max_iterations=args.max_iterations,
        num_threads=args.num_threads,
    )
    _write_config(config_path, config)
    command = build_pixsfm_command(
        python=args.pixsfm_python.resolve(),
        input_model=input_model,
        output_model=partial_model,
        image_root=image_root,
        config_path=config_path,
        cache_path=cache_path,
    )
    report: dict[str, Any] = {
        "schema": "pixsfm-fixed-intrinsics-candidate/v1",
        "status": "planned" if args.dry_run else "running",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": "PixSfM feature-metric bundle adjustment",
        "architecture": {
            "starts_from_existing_model": True,
            "rematching": False,
            "track_rebuild": False,
            "triangulation": False,
            "pose_graph_optimization": False,
        },
        "source_model": source_fingerprint,
        "image_root": str(image_root),
        "pixsfm_runtime": pixsfm_runtime,
        "memory_preflight": memory,
        "geometry_preflight": (
            "deferred_until_execution" if args.dry_run else "pending"
        ),
        "config": config,
        "command": command,
        "feature_cache": {
            "path": str(cache_path),
            "reused": cache_reused,
            **(
                {
                    "bytes": cache_path.stat().st_size,
                    "sha256": sha256_file(cache_path),
                }
                if cache_reused
                else {}
            ),
        },
    }
    report_path = output_root / "candidate_report.json"
    _atomic_json(report_path, report)
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    import pycolmap

    source = pycolmap.Reconstruction(str(input_model))
    before_intrinsics = intrinsics_signature(source)
    before_summary = model_summary(source)
    report.update(
        {
            "geometry_preflight": "completed",
            "before": before_summary,
            "intrinsics_before": before_intrinsics,
        }
    )
    _atomic_json(report_path, report)

    started = time.perf_counter()
    log_path = output_root / "pixsfm.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            check=False,
        )
    report["elapsed_seconds"] = time.perf_counter() - started
    report["returncode"] = process.returncode
    if cache_path.is_file() and not cache_reused:
        report["feature_cache"].update(
            {"bytes": cache_path.stat().st_size, "sha256": sha256_file(cache_path)}
        )
    if process.returncode != 0:
        report["status"] = "failed_runtime"
        _atomic_json(report_path, report)
        raise SystemExit(
            f"PixSfM failed with exit {process.returncode}; see {log_path}"
        )

    candidate = pycolmap.Reconstruction(str(partial_model))
    after_intrinsics = intrinsics_signature(candidate)
    after_summary = model_summary(candidate)
    try:
        comparison = compare_reconstructions(source, candidate)
    except Exception as exc:
        comparison = {
            "schema": "colmap-refinement-comparison/v1",
            "structurally_eligible": False,
            "gates": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    reject_reasons = comparison_reject_reasons(comparison)
    legacy_gate_error = None
    try:
        assert_intrinsics_unchanged(before_intrinsics, after_intrinsics)
        assert_structure_preserved(before_summary, after_summary)
    except RuntimeError as exc:
        legacy_gate_error = str(exc)
        reject_reasons.append(f"fixed-structure gate failed: {exc}")

    if (
        comparison.get("structurally_eligible") is not True
        or reject_reasons
        or legacy_gate_error
    ):
        report.update(
            {
                "status": "rejected",
                "after": after_summary,
                "intrinsics_after": after_intrinsics,
                "candidate_model": model_fingerprint(partial_model),
                "comparison": comparison,
                "reject_reasons": reject_reasons,
                **(
                    {"gate_error": legacy_gate_error}
                    if legacy_gate_error is not None
                    else {}
                ),
            }
        )
        _atomic_json(report_path, report)
        raise SystemExit(f"PixSfM candidate rejected; see {report_path}")

    partial_model.rename(output_model)
    report.update(
        {
            "status": "passed",
            "after": after_summary,
            "intrinsics_after": after_intrinsics,
            "output_model": model_fingerprint(output_model),
            "comparison": comparison,
            "reject_reasons": [],
            "gates": {
                "fixed_intrinsics_exact": True,
                "camera_assignments_exact": True,
                "registered_images_preserved": True,
                "points3D_preserved": True,
                "recomputed_geometry_structurally_eligible": True,
                "production_artifacts_untouched": True,
            },
        }
    )
    _atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
