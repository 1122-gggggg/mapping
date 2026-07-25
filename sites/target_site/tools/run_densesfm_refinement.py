#!/usr/bin/env python3
"""Run Dense-SfM post-optimization on an existing, frozen COLMAP model.

This controller never invokes Dense-SfM's coarse matching or SfM construction.
It validates exact image/keypoint identity against an existing COLMAP database,
runs post-optimization in an isolated candidate directory, and promotes the
candidate only after the shared geometry gates and fixed-intrinsics contract pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


HERE = Path(__file__).resolve()
SYSTEM_ROOT = HERE.parents[2]
TARGET_RUN = HERE.parents[1] / "runs" / "target_site_v1"
DEFAULT_DENSE_REPO = Path("/home/cihcilab/.cache/sfm_system/DenseSfM-Refine")
DEFAULT_DENSE_PYTHON = Path(
    "/home/cihcilab/.cache/sfm_system/dense-sfm-py312/bin/python"
)
DEFAULT_CUDA_ROOT = Path("/home/cihcilab/micromamba/envs/cudatk")
MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin")
MAX_IMAGE_ID = 2_147_483_647
REQUIRED_DENSE_PATCH_FILES = (
    "src/utils/colmap/read_write_model.py",
    "src/utils/colmap/database.py",
)


@dataclass(frozen=True)
class ModelImageRecord:
    image_id: int
    name: str
    points2d: np.ndarray


def _overlaps(path: Path, protected: Path) -> bool:
    return path == protected or protected in path.parents or path in protected.parents


def validate_isolated_output(
    input_model: Path, output_model: Path, protected_paths: Iterable[Path]
) -> Path:
    source = Path(input_model).resolve()
    output = Path(output_model).resolve()
    protected = {source, *(Path(path).resolve() for path in protected_paths)}
    collision = next((path for path in protected if _overlaps(output, path)), None)
    if collision is not None:
        raise ValueError(f"output overlaps protected path: {collision}")
    return output


def dense_call_contract(
    *,
    image_paths: Sequence[Path | str],
    input_model: Path,
    output_model: Path,
    database_path: Path,
    dense_repo: Path,
    mode: str,
    chunk_size: int,
    iterations: int,
    img_resize: int = 1200,
    num_threads: int = 4,
) -> dict[str, Any]:
    """Build the only accepted Dense-SfM call for this target-site study."""
    if mode not in {"points_only", "poses_and_points"}:
        raise ValueError(f"unsupported Dense-SfM mode: {mode}")
    if min(chunk_size, iterations, img_resize, num_threads) <= 0:
        raise ValueError("chunk size, iterations, image size, and threads must be positive")
    repo = Path(dense_repo).resolve()
    return {
        "image_lists": [str(path) for path in image_paths],
        "covis_pairs_pth": None,
        "colmap_coarse_dir": str(Path(input_model).resolve()),
        "refined_model_save_dir": str(Path(output_model).resolve()),
        "match_out_pth": None,
        "chunk_size": int(chunk_size),
        "matcher_model_path": str((repo / "weight" / "mv_refinement.ckpt").resolve()),
        "matcher_cfg_path": str(
            (
                repo
                / "hydra_training_configs"
                / "experiment"
                / "multiview_refinement_matching.yaml"
            ).resolve()
        ),
        "img_resize": int(img_resize),
        "img_preload": False,
        "fine_match_use_ray": False,
        "ray_cfg": None,
        "colmap_configs": {
            "use_pba": False,
            "no_refine_intrinsics": True,
            "n_threads": int(num_threads),
        },
        "only_basename_in_colmap": False,
        "refine_iter_n_times": int(iterations),
        "incremental_refiner_filter_thresholds": [4, 3, 2],
        "model_refiner_no_filter_pts": True,
        "refine_3D_pts_only": mode == "points_only",
        "verbose": True,
        "database_path": str(Path(database_path).resolve()),
        "use_pycolmap": True,
    }


def build_worker_command(
    *, python: Path, runner: Path, contract_path: Path, dense_repo: Path
) -> list[str]:
    return [
        str(Path(python)),
        str(Path(runner).resolve()),
        "--worker-contract",
        str(Path(contract_path).resolve()),
        "--dense-repo",
        str(Path(dense_repo).resolve()),
    ]


def validate_database_contract(
    records: Sequence[ModelImageRecord], database_path: Path, *, atol_px: float = 1e-6
) -> dict[str, Any]:
    """Fail closed unless model IDs, names and all keypoint XY values match the DB."""
    database = Path(database_path).resolve()
    if not database.is_file():
        raise FileNotFoundError(f"COLMAP database does not exist: {database}")
    max_error = 0.0
    keypoints_checked = 0
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        for record in records:
            image_row = db.execute(
                "SELECT name FROM images WHERE image_id = ?", (int(record.image_id),)
            ).fetchone()
            if image_row is None:
                raise RuntimeError(f"database lacks model image ID {record.image_id}")
            if image_row[0] != record.name:
                raise RuntimeError(
                    f"database image name mismatch for ID {record.image_id}: "
                    f"{image_row[0]!r} != {record.name!r}"
                )
            keypoint_row = db.execute(
                "SELECT rows, cols, data FROM keypoints WHERE image_id = ?",
                (int(record.image_id),),
            ).fetchone()
            if keypoint_row is None:
                raise RuntimeError(f"database lacks keypoints for image ID {record.image_id}")
            rows, cols, blob = keypoint_row
            keypoints = np.frombuffer(blob, dtype=np.float32)
            if keypoints.size != int(rows) * int(cols):
                raise RuntimeError(f"malformed keypoint blob for image ID {record.image_id}")
            keypoints = keypoints.reshape(int(rows), int(cols))
            model_xy = np.asarray(record.points2d, dtype=np.float64).reshape(-1, 2)
            if int(cols) < 2 or keypoints.shape[0] != model_xy.shape[0]:
                raise RuntimeError(
                    f"keypoint count mismatch for image ID {record.image_id}: "
                    f"database={keypoints.shape[0]} model={model_xy.shape[0]}"
                )
            if model_xy.size:
                error = np.linalg.norm(
                    keypoints[:, :2].astype(np.float64) - model_xy, axis=1
                )
                local_max = float(np.max(error))
                max_error = max(max_error, local_max)
                if not np.all(error <= atol_px):
                    raise RuntimeError(
                        f"keypoint mismatch for image ID {record.image_id}; "
                        f"max XY error is {local_max:.9g} px"
                    )
            keypoints_checked += model_xy.shape[0]
    return {
        "database": str(database),
        "images_checked": len(records),
        "keypoints_checked": int(keypoints_checked),
        "max_xy_error_px": float(max_error),
        "atol_px": float(atol_px),
    }


COLMAP311_SCHEMA = """
PRAGMA foreign_keys = OFF;
CREATE TABLE cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL
);
CREATE TABLE images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 AND image_id < 2147483647),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
);
CREATE UNIQUE INDEX index_name ON images(name);
CREATE TABLE pose_priors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    position BLOB,
    coordinate_system INTEGER NOT NULL,
    position_covariance BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
);
CREATE TABLE keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
);
CREATE TABLE descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
);
CREATE TABLE matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB
);
CREATE TABLE two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB
);
"""


def build_colmap311_compat_database(
    records: Sequence[ModelImageRecord], source_database: Path, output_database: Path
) -> dict[str, Any]:
    """Transcode selected existing records to the schema expected by pycolmap 3.11."""
    source = Path(source_database).resolve()
    output = Path(output_database).resolve()
    if output.exists():
        raise FileExistsError(f"compatibility database already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_ids = sorted({int(record.image_id) for record in records})
    if not selected_ids:
        raise ValueError("at least one selected model image is required")

    with sqlite3.connect(output) as database:
        database.executescript(COLMAP311_SCHEMA)
        database.execute("ATTACH DATABASE ? AS source", (f"file:{source}?mode=ro",))
        database.execute("CREATE TEMP TABLE selected_ids(image_id INTEGER PRIMARY KEY)")
        database.executemany(
            "INSERT INTO selected_ids(image_id) VALUES(?)",
            [(image_id,) for image_id in selected_ids],
        )
        database.execute(
            """
            INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length)
            SELECT DISTINCT c.camera_id, c.model, c.width, c.height, c.params, c.prior_focal_length
            FROM source.cameras AS c
            JOIN source.images AS i ON i.camera_id = c.camera_id
            JOIN selected_ids AS selected ON selected.image_id = i.image_id
            """
        )
        database.execute(
            """
            INSERT INTO images(image_id, name, camera_id)
            SELECT i.image_id, i.name, i.camera_id
            FROM source.images AS i
            JOIN selected_ids AS selected ON selected.image_id = i.image_id
            """
        )
        database.execute(
            """
            INSERT INTO keypoints(image_id, rows, cols, data)
            SELECT k.image_id, k.rows, k.cols, k.data
            FROM source.keypoints AS k
            JOIN selected_ids AS selected ON selected.image_id = k.image_id
            """
        )
        pair_filter = f"""
            JOIN selected_ids AS first_image
              ON first_image.image_id = CAST(
                   (pairs.pair_id - (pairs.pair_id % {MAX_IMAGE_ID})) / {MAX_IMAGE_ID}
                 AS INTEGER)
            JOIN selected_ids AS second_image
              ON second_image.image_id = (pairs.pair_id % {MAX_IMAGE_ID})
        """
        database.execute(
            f"""
            INSERT INTO matches(pair_id, rows, cols, data)
            SELECT pairs.pair_id, pairs.rows, pairs.cols, pairs.data
            FROM source.matches AS pairs
            {pair_filter}
            """
        )
        database.execute(
            f"""
            INSERT INTO two_view_geometries(
                pair_id, rows, cols, data, config, F, E, H, qvec, tvec
            )
            SELECT pairs.pair_id, pairs.rows, pairs.cols, pairs.data, pairs.config,
                   pairs.F, pairs.E, pairs.H, pairs.qvec, pairs.tvec
            FROM source.two_view_geometries AS pairs
            {pair_filter}
            """
        )
        counts = {
            table: int(database.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("cameras", "images", "keypoints", "matches", "two_view_geometries")
        }
        integrity = database.execute("PRAGMA integrity_check").fetchone()[0]
        database.commit()
    if counts["images"] != len(selected_ids) or counts["keypoints"] != len(selected_ids):
        raise RuntimeError(f"compatibility database lost selected images: {counts}")
    if integrity != "ok":
        raise RuntimeError(f"compatibility database integrity failed: {integrity}")
    return {
        "path": str(output),
        **counts,
        "integrity_check": integrity,
        "source_schema_user_version": _sqlite_user_version(source),
        "target_schema": "COLMAP/pycolmap 3.11",
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
    }


def _sqlite_user_version(database_path: Path) -> int:
    with sqlite3.connect(f"file:{Path(database_path).resolve()}?mode=ro", uri=True) as db:
        return int(db.execute("PRAGMA user_version").fetchone()[0])


def _plain(value: Any) -> Any:
    return value() if callable(value) else value


def _image_is_registered(reconstruction: Any, image_id: int, image: Any) -> bool:
    registered_ids = getattr(reconstruction, "reg_image_ids", None)
    if registered_ids is not None:
        return int(image_id) in {int(value) for value in _plain(registered_ids)}
    for attribute in ("has_pose", "registered"):
        if hasattr(image, attribute):
            return bool(_plain(getattr(image, attribute)))
    return True


def reconstruction_records(reconstruction: Any) -> list[ModelImageRecord]:
    records = []
    for image_id, image in sorted(reconstruction.images.items()):
        if not _image_is_registered(reconstruction, int(image_id), image):
            continue
        points = []
        for point in image.points2D:
            points.append(np.asarray(point.xy, dtype=np.float64).reshape(2))
        records.append(
            ModelImageRecord(
                image_id=int(image_id),
                name=str(image.name),
                points2d=np.asarray(points, dtype=np.float64).reshape(-1, 2),
            )
        )
    if not records:
        raise RuntimeError("input reconstruction has no registered images")
    return records


def intrinsics_signature(reconstruction: Any) -> dict[str, Any]:
    cameras = {}
    for camera_id, camera in sorted(reconstruction.cameras.items()):
        model_name = getattr(camera, "model_name", None)
        if model_name is None:
            model = getattr(camera, "model", None)
            model_name = getattr(model, "name", model)
        cameras[str(int(camera_id))] = {
            "model": str(_plain(model_name)),
            "width": int(camera.width),
            "height": int(camera.height),
            "params": [float(value) for value in np.asarray(camera.params).reshape(-1)],
        }
    assignments = {
        str(image.name): int(image.camera_id)
        for image_id, image in sorted(reconstruction.images.items())
        if _image_is_registered(reconstruction, int(image_id), image)
    }
    return {"cameras": cameras, "assignments": dict(sorted(assignments.items()))}


def assert_intrinsics_unchanged(before: dict[str, Any], after: dict[str, Any]) -> None:
    if before != after:
        raise RuntimeError("camera intrinsics or image-to-camera assignments changed")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(model_dir: Path) -> dict[str, Any]:
    root = Path(model_dir).resolve()
    rows = []
    combined = hashlib.sha256()
    for name in MODEL_FILES:
        path = root / name
        if not path.is_file():
            continue
        row = {"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        rows.append(row)
        combined.update(json.dumps(row, sort_keys=True).encode("utf-8"))
    required = {"cameras.bin", "images.bin", "points3D.bin"}
    if not required.issubset({row["name"] for row in rows}):
        raise FileNotFoundError(f"incomplete COLMAP model: {root}")
    return {"path": str(root), "sha256": combined.hexdigest(), "files": rows}


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def memory_snapshot() -> dict[str, float]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0])
    return {
        "available_gib": values.get("MemAvailable", 0) / 1024**2,
        "swap_free_gib": values.get("SwapFree", 0) / 1024**2,
    }


def dense_source_provenance(repo: Path) -> dict[str, Any]:
    root = Path(repo).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    compatibility = {}
    for relative in REQUIRED_DENSE_PATCH_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Dense-SfM compatibility file is missing: {path}")
        compatibility[relative] = sha256_file(path)
    return {
        "repository": str(root),
        "commit": commit,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "compatibility_files": compatibility,
        "license_file_present": any((root / name).is_file() for name in ("LICENSE", "LICENSE.md")),
    }


def preflight_dense_runtime(python: Path, repo: Path) -> dict[str, Any]:
    executable = Path(python).expanduser()
    if not executable.is_absolute():
        executable = Path.cwd() / executable
    if not executable.is_file():
        raise FileNotFoundError(f"Dense-SfM Python does not exist: {executable}")
    code = (
        "import json, einops, kornia, pycolmap, torch; import roi_align; "
        "print(json.dumps({'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
        "'pycolmap':pycolmap.__version__,'kornia':kornia.__version__,"
        "'einops':einops.__version__,'gpu':torch.cuda.get_device_name(0)}))"
    )
    result = subprocess.run(
        [str(executable), "-c", code],
        cwd=Path(repo).resolve(),
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return json.loads(result.stdout.splitlines()[-1])


def compare_reconstructions(source: Any, candidate: Any) -> dict[str, Any]:
    tools_dir = str(HERE.parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from compare_colmap_candidate import compare_reconstructions as compare

    return compare(source, candidate)


def comparison_reject_reasons(comparison: dict[str, Any]) -> list[str]:
    if comparison.get("error"):
        return [f"comparison failed: {comparison['error']}"]
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


def run_worker(contract_path: Path, dense_repo: Path) -> None:
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    repo = Path(dense_repo).resolve()
    image_root = Path(contract.pop("image_root")).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.chdir(image_root)
    from src.post_optimization.post_optimization import post_optimization

    state = post_optimization(**contract)
    if not state:
        raise RuntimeError("Dense-SfM post_optimization returned a false state")


def controller(args: argparse.Namespace) -> None:
    import pycolmap

    input_model = args.input_model.resolve()
    image_root = args.image_root.resolve()
    database = args.database.resolve()
    output_root = args.output_root.resolve()
    partial_model = output_root / "model.partial"
    output_model = output_root / "model"
    protected = [
        TARGET_RUN / "final_model",
        TARGET_RUN / "edm",
        SYSTEM_ROOT / "EDM定位測試" / "transfer" / "release",
    ]
    validate_isolated_output(input_model, partial_model, protected)
    validate_isolated_output(input_model, output_model, protected)
    if not image_root.is_dir():
        raise FileNotFoundError(f"image root does not exist: {image_root}")
    if output_root.exists():
        raise FileExistsError(f"candidate output already exists: {output_root}")

    memory = memory_snapshot()
    if memory["available_gib"] < args.minimum_available_ram_gib:
        raise RuntimeError(
            f"only {memory['available_gib']:.2f} GiB RAM available; "
            f"requires {args.minimum_available_ram_gib:.2f} GiB"
        )

    source = pycolmap.Reconstruction(str(input_model))
    records = reconstruction_records(source)
    missing_images = [record.name for record in records if not (image_root / record.name).is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} registered images are missing; first={missing_images[0]}"
        )
    database_report = validate_database_contract(records, database)
    source_fingerprint = model_fingerprint(input_model)
    source_intrinsics = intrinsics_signature(source)
    runtime = preflight_dense_runtime(args.dense_python, args.dense_repo)
    provenance = dense_source_provenance(args.dense_repo)

    output_root.mkdir(parents=True, exist_ok=False)
    compatibility_database = output_root / "database_colmap311.db"
    compatibility_report = build_colmap311_compat_database(
        records, database, compatibility_database
    )
    contract_path = output_root / "dense_call_contract.json"
    report_path = output_root / "candidate_report.json"
    log_path = output_root / "dense_sfm.log"
    contract = dense_call_contract(
        image_paths=[record.name for record in records],
        input_model=input_model,
        output_model=partial_model,
        database_path=compatibility_database,
        dense_repo=args.dense_repo,
        mode=args.mode,
        chunk_size=args.chunk_size,
        iterations=args.iterations,
        img_resize=args.img_resize,
        num_threads=args.num_threads,
    )
    contract["image_root"] = str(image_root)
    _atomic_json(contract_path, contract)
    command = build_worker_command(
        python=args.dense_python,
        runner=HERE,
        contract_path=contract_path,
        dense_repo=args.dense_repo,
    )
    report: dict[str, Any] = {
        "schema": "densesfm-post-glomap-candidate/v1",
        "status": "planned" if args.dry_run else "running",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": "Dense-SfM CVPR 2025 post-optimization",
        "mode": args.mode,
        "architecture": {
            "starts_from_existing_colmap_model": True,
            "reuses_existing_database": True,
            "database_schema_transcode_only": True,
            "rematching": False,
            "sfm_rebuild": False,
            "pose_graph_rebuild": False,
            "intrinsics_optimization": False,
            "pose_optimization": args.mode == "poses_and_points",
            "point_optimization": True,
            "topology_preserved": True,
        },
        "source_model": source_fingerprint,
        "database": {
            **database_report,
            "bytes": database.stat().st_size,
            "sha256": sha256_file(database),
            "compatibility_copy": compatibility_report,
        },
        "image_root": str(image_root),
        "registered_image_names": [record.name for record in records],
        "runtime": runtime,
        "dense_source": provenance,
        "memory_preflight": memory,
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "command": command,
        "geometry_preflight": "pending",
    }
    _atomic_json(report_path, report)
    if args.dry_run:
        return

    environment = os.environ.copy()
    cuda_root = args.cuda_root.resolve()
    prior_library_path = environment.get("LD_LIBRARY_PATH", "")
    environment["CUDA_HOME"] = str(cuda_root)
    environment["LD_LIBRARY_PATH"] = ":".join(
        value
        for value in (str(cuda_root / "lib"), str(cuda_root / "lib64"), prior_library_path)
        if value
    )
    started = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=args.dense_repo.resolve(),
                env=environment,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if not partial_model.is_dir():
            raise RuntimeError("Dense-SfM did not produce model.partial")
        if model_fingerprint(input_model) != source_fingerprint:
            raise RuntimeError("protected source model changed during refinement")

        candidate = pycolmap.Reconstruction(str(partial_model))
        assert_intrinsics_unchanged(source_intrinsics, intrinsics_signature(candidate))
        comparison = compare_reconstructions(source, candidate)
        reasons = comparison_reject_reasons(comparison)
        report["comparison"] = comparison
        report["geometry_preflight"] = "passed" if not reasons else "rejected"
        report["reject_reasons"] = reasons
        report["partial_model"] = model_fingerprint(partial_model)
        if reasons:
            report["status"] = "rejected"
        else:
            partial_model.replace(output_model)
            report["status"] = "accepted"
            report["output_model"] = model_fingerprint(output_model)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["log"] = str(log_path)
        if log_path.is_file():
            report["log_sha256"] = sha256_file(log_path)
        _atomic_json(report_path, report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-contract", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--dense-repo", type=Path, default=DEFAULT_DENSE_REPO)
    parser.add_argument("--input-model", type=Path)
    parser.add_argument("--image-root", type=Path, default=TARGET_RUN / "images")
    parser.add_argument(
        "--database", type=Path, default=TARGET_RUN / "gluemap" / "database_merged.db"
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dense-python", type=Path, default=DEFAULT_DENSE_PYTHON)
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA_ROOT)
    parser.add_argument(
        "--mode", choices=("points_only", "poses_and_points"), default="points_only"
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--img-resize", type=int, default=1200)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--minimum-available-ram-gib", type=float, default=8.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.worker_contract is None and (args.input_model is None or args.output_root is None):
        parser.error("--input-model and --output-root are required outside worker mode")
    return args


def main() -> None:
    args = parse_args()
    if args.worker_contract is not None:
        run_worker(args.worker_contract, args.dense_repo)
    else:
        controller(args)


if __name__ == "__main__":
    main()
