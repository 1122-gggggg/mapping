#!/usr/bin/env python3
"""S2b -- fixed-camera applicability gate for the Fuhe v2 adapter.

P0 fixes one calibrated 1920x1080 PINHOLE camera, so seed convergence and
cross-resolution agreement are explicitly NOT_APPLICABLE. The CLI writes the
typed policy/gate and returns before any runtime preflight or solve. Legacy
diagnostic helpers remain importable only for historical v1 artifact inspection.
"""
from __future__ import annotations

import argparse
import copy
import contextlib
import fcntl
import functools
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ts_common import (  # noqa: E402
    BUILD,
    GLUEMAP_DEMO,
    GLUEMAP_ENV,
    GLUEMAP_REPO,
    RUNS,
    RUN_ID,
    WORKING_HEIGHT,
    WORKING_WIDTH,
    Gate,
    hash_artifact,
    log,
    read_json,
    required_check_ids,
    sha256,
    stage_material_artifacts,
    source_sha256,
    write_json,
)
from ts_env import verify_pycolmap_runtime  # noqa: E402
from ts_intrinsics import (  # noqa: E402
    CANDIDATES,
    Camera,
    FUHE_CX,
    FUHE_CY,
    FUHE_FX,
    FUHE_FY,
    write_intrinsics_seed,
)


TS_COMMON = Path(__file__).with_name("ts_common.py")
TS_ENV = Path(__file__).with_name("ts_env.py")
TS_INTRINSICS = Path(__file__).with_name("ts_intrinsics.py")
EXTERNAL_CAMERA_RECORD = Path(
    "/media/cihcilab/新增磁碟區/福和橋場域/fuhe_submaps/"
    "FUHE_BRIDGE_PROJECT_COMPLETE_RECORD.md"
).resolve()
EXTERNAL_CAMERA_RECORD_SHA256 = (
    "65b1b50dff22935711263ab9b546cbbe1dc0f2c3443782e83c3be7e4def03903"
)

# The cleanest video in each resolution group (highest parallax fraction, per S1).
PROBE_VIDEO = {(WORKING_WIDTH, WORKING_HEIGHT): BUILD[0].seq}
SEEDS = ("official69", "charuco")
REQUIRED_SHAPES = tuple(f"{w}x{h}" for w, h in PROBE_VIDEO)
REQUIRED_RESULT_KEYS = frozenset(
    (f"{w}x{h}", seed) for (w, h) in PROBE_VIDEO for seed in SEEDS
)
S2B_REQUIRED_IDS = required_check_ids("S2b_intrinsics")

AGREE_TOL = 0.015
CROSS_RESOLUTION_TOL = 0.02
MODEL_FILENAMES = ("cameras.bin", "images.bin", "points3D.bin")
CHECKPOINTS = {
    "pi3": GLUEMAP_REPO / "checkpoints" / "pi3.safetensors",
    "retrieval": GLUEMAP_REPO / "checkpoints" / "dino_salad.ckpt",
    "tracker": GLUEMAP_REPO / "checkpoints" / "vggsfm_v2_0_0_track_predictor.bin",
    "doppelgangers": GLUEMAP_REPO / "checkpoints" / "checkpoint-dg+visym.pth",
}


def external_camera_policy() -> dict[str, Any]:
    """Build the hash-bound official69 frozen-camera decision record."""
    actual_sha256 = sha256(EXTERNAL_CAMERA_RECORD)
    source_matches = actual_sha256 == EXTERNAL_CAMERA_RECORD_SHA256
    camera = {
        "model": "PINHOLE",
        "width": WORKING_WIDTH,
        "height": WORKING_HEIGHT,
        "params": [FUHE_FX, FUHE_FY, FUHE_CX, FUHE_CY],
    }
    return {
        "schema_version": "fuhe-intrinsics-policy-v2",
        "camera_policy": {
            "state": "PASS" if source_matches else "FAIL",
            "applicable": True,
            "decision": "official69_fixed_pinhole",
            "external_record": {
                "path": str(EXTERNAL_CAMERA_RECORD),
                "sha256": actual_sha256,
            },
            "locked_external_record_sha256": EXTERNAL_CAMERA_RECORD_SHA256,
            "camera": camera,
            "official_hfov_deg": 69.0,
            "resize": "raw INTER_AREA to 1920x1080",
            "undistort": False,
            "fixed_intrinsics_ba": True,
        },
        "diagnostics": {
            "two_seed": {
                "state": "NOT_APPLICABLE",
                "applicable": False,
                "reason": "camera is externally frozen; seed convergence is not estimated",
            },
            "cross_resolution": {
                "state": "NOT_APPLICABLE",
                "applicable": False,
                "reason": "adapter has one 1920x1080 working resolution",
            },
        },
    }


def _camera_policy_valid(policy: dict[str, Any]) -> bool:
    camera_policy = policy.get("camera_policy", {})
    return (
        policy.get("schema_version") == "fuhe-intrinsics-policy-v2"
        and camera_policy.get("state") == "PASS"
        and camera_policy.get("applicable") is True
        and camera_policy.get("decision") == "official69_fixed_pinhole"
        and camera_policy.get("external_record")
        == {
            "path": str(EXTERNAL_CAMERA_RECORD),
            "sha256": EXTERNAL_CAMERA_RECORD_SHA256,
        }
        and camera_policy.get("locked_external_record_sha256")
        == EXTERNAL_CAMERA_RECORD_SHA256
        and camera_policy.get("camera")
        == {
            "model": "PINHOLE",
            "width": 1920,
            "height": 1080,
            "params": [FUHE_FX, FUHE_FY, FUHE_CX, FUHE_CY],
        }
        and camera_policy.get("official_hfov_deg") == 69.0
        and camera_policy.get("resize") == "raw INTER_AREA to 1920x1080"
        and camera_policy.get("undistort") is False
        and camera_policy.get("fixed_intrinsics_ba") is True
    )


def emit_external_camera_policy_checks(gate: Gate, policy: dict[str, Any]) -> None:
    """Emit one substantive camera-policy PASS and two typed diagnostic N/As."""
    camera_policy = policy.get("camera_policy", {})
    gate.check(
        "G2.7/results_complete",
        _camera_policy_valid(policy),
        "external official69 camera decision, exact K, resize, and no-undistort bind",
        policy=policy,
    )
    two_seed = policy.get("diagnostics", {}).get("two_seed", {})
    gate.not_applicable(
        "G2.7/1920x1080",
        str(two_seed.get("reason", "two-seed diagnostic is not applicable")),
        diagnostic="two_seed",
        resolution="1920x1080",
        external_record_sha256=camera_policy.get("external_record", {}).get("sha256"),
    )
    cross_resolution = policy.get("diagnostics", {}).get("cross_resolution", {})
    gate.not_applicable(
        "G2.8",
        str(
            cross_resolution.get(
                "reason", "cross-resolution diagnostic is not applicable"
            )
        ),
        diagnostic="cross_resolution",
        working_resolutions=["1920x1080"],
        external_record_sha256=camera_policy.get("external_record", {}).get("sha256"),
    )


def keystr(key: tuple[str, str]) -> str:
    return f"{key[0]}/{key[1]}"


def parse_key(value: str) -> tuple[str, str]:
    parts = value.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"invalid logical key {value!r}")
    return parts[0], parts[1]


def shape_tuple(shape: str) -> tuple[int, int]:
    try:
        width, height = (int(value) for value in shape.split("x"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid shape {shape!r}") from exc
    return width, height


def _canonical_sha256(payload: Any) -> str:
    data = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@functools.lru_cache(maxsize=1)
def _generating_bundle() -> dict[str, Any]:
    """Hash the immutable generating source/checkpoint bundle once per process."""
    source_paths = sorted((GLUEMAP_REPO / "gluemap").rglob("*.py")) + [
        GLUEMAP_REPO / "pyproject.toml",
        Path(__file__).resolve(),
        TS_COMMON,
        TS_ENV,
        TS_INTRINSICS,
    ]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=GLUEMAP_REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=GLUEMAP_REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return {
        "sources": {
            str(path.resolve()): hash_artifact(path) for path in source_paths
        },
        "checkpoints": {
            name: hash_artifact(path) for name, path in CHECKPOINTS.items()
        },
        "gluemap_git": {"commit": commit, "porcelain_v1": status},
    }


def generating_provenance(
    runtime: dict[str, Any], scientific_config: dict[str, Any]
) -> dict[str, Any]:
    bundle = copy.deepcopy(_generating_bundle())
    bundle.update(
        {
            "mode": "fresh_solve",
            "runtime_fingerprint": runtime,
            "scientific_config": scientific_config,
        }
    )
    return bundle


def jpeg_shape(path: Path) -> tuple[int, int]:
    """Read JPEG dimensions from SOF without adding a runtime dependency."""
    sof_markers = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }
    with path.open("rb") as stream:
        if stream.read(2) != b"\xff\xd8":
            raise ValueError(f"{path}: not a JPEG")
        while True:
            byte = stream.read(1)
            if not byte:
                break
            if byte != b"\xff":
                continue
            while byte == b"\xff":
                byte = stream.read(1)
            marker = byte[0]
            if marker in {0xD8, 0xD9}:
                continue
            length_bytes = stream.read(2)
            if len(length_bytes) != 2:
                break
            length = int.from_bytes(length_bytes, "big")
            if length < 2:
                break
            if marker in sof_markers:
                payload = stream.read(5)
                if len(payload) != 5:
                    break
                height = int.from_bytes(payload[1:3], "big")
                width = int.from_bytes(payload[3:5], "big")
                return width, height
            stream.seek(length - 2, os.SEEK_CUR)
    raise ValueError(f"{path}: JPEG SOF dimensions not found")


def probe_input(run_dir: Path, shape: tuple[int, int]) -> dict[str, Any]:
    """Hash ordered persisted probe inputs and verify their real dimensions."""
    seq = PROBE_VIDEO[shape]
    manifest = read_json(run_dir / "frame_manifest.json")
    frames = sorted(
        (frame for frame in manifest["frames"] if frame["seq"] == seq),
        key=lambda frame: frame["name"],
    )
    if not frames:
        raise ValueError(f"{seq}: no probe frames in frame_manifest.json")

    rows: list[dict[str, Any]] = []
    for frame in frames:
        image_path = run_dir / "images" / frame["name"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        actual_shape = jpeg_shape(image_path)
        if actual_shape != shape or (frame["width"], frame["height"]) != shape:
            raise ValueError(
                f"{frame['name']}: disk/manifest shape {actual_shape}/"
                f"{(frame['width'], frame['height'])} != {shape}"
            )
        image_hash = sha256(image_path)
        recorded = frame.get("image_sha256")
        if recorded is not None and recorded != image_hash:
            raise ValueError(f"{frame['name']}: frame-manifest SHA-256 drift")
        rows.append(
            {
                "name": frame["name"],
                "width": shape[0],
                "height": shape[1],
                "image_sha256": image_hash,
            }
        )
    return {
        "shape": f"{shape[0]}x{shape[1]}",
        "seq": seq,
        "n_frames": len(rows),
        "frames": rows,
        "probe_input_sha256": _canonical_sha256(rows),
    }


def build_probe_dir(
    run_dir: Path, probe: dict[str, Any], work: Path
) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """Symlink one sequence's locked frames into an isolated image root."""
    images = work / "images"
    names: list[str] = []
    shape_of: dict[str, tuple[int, int]] = {}
    for row in probe["frames"]:
        name = row["name"]
        destination = images / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() and destination.resolve() != (
            run_dir / "images" / name
        ).resolve():
            destination.unlink()
        if not destination.exists():
            destination.symlink_to(run_dir / "images" / name)
        names.append(name)
        shape_of[name] = (row["width"], row["height"])
    return names, shape_of


def run_gluemap(
    work: Path,
    images: Path,
    seed: Path,
    refine: bool,
    *,
    maxkp: int,
    num_workers: int,
    rerun_from: str | None,
) -> Path:
    """Run one solve with scientific and operational settings kept distinct."""
    if not 1 <= maxkp <= 1024:
        raise ValueError("maxkp must be in [1, 1024]")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    out = work / "gluemap"
    cfg = {
        "chosen_model": "pi3",
        "path_feedforward": str(GLUEMAP_REPO / "checkpoints" / "pi3.safetensors"),
        "path_retrieval": str(GLUEMAP_REPO / "checkpoints" / "dino_salad.ckpt"),
        "path_tracker": str(GLUEMAP_REPO / "checkpoints" / "vggsfm_v2_0_0_track_predictor.bin"),
        "path_dg": str(GLUEMAP_REPO / "checkpoints" / "checkpoint-dg+visym.pth"),
        "images_path": str(images),
        "write_path": str(out),
        "temp_path": str(work / "tmp"),
        "chosen_output": "gluemap_aba",
        "camera_model": "PINHOLE",
        "intrinsics_mode": "SHARED",
        "use_gt_intrinsics": True,
        "gt_intrinsics_path": str(seed),
        "refine_intrinsics": refine,
        "min_track_length": 3,
        "num_track_per_img": maxkp,
        "max_num_tracks": None,
        "num_neighbors": 50,
        "num_neighbors_sequential": 20,
        "valid_pose_threshold": 0.05,
        "valid_dg_threshold": 0.8,
        "skip_doppelgangers": True,
        "is_sequential": True,
        "sample_frequency": 1,
        "is_multi_sequence": True,
        "subfolder_regex": "^S[0-9]+",
        "batch_size": 8,
        "retrieval_batch_size": 30,
        "num_workers": num_workers,
        "force_load": True,
        "coarse_only": False,
        "use_dummy_tracks": False,
    }
    if rerun_from:
        cfg["rerun_from"] = rerun_from
    cfg_path = work / "gluemap_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{GLUEMAP_ENV}/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["PATH"] = f"{GLUEMAP_ENV}/bin:" + env.get("PATH", "")
    log_path = work / "gluemap.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            [str(GLUEMAP_DEMO), "--config", str(cfg_path)],
            cwd=GLUEMAP_REPO,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="ignore").splitlines()[-40:])
        raise RuntimeError(f"gluemap failed ({process.returncode}):\n{tail}")
    return out / "gluemap_aba"


def _required_model_files(model: Path) -> dict[str, Path]:
    files = {name: model / name for name in MODEL_FILENAMES}
    if all(path.is_file() for path in files.values()):
        return files
    text_files = {
        name.replace(".bin", ".txt"): model / name.replace(".bin", ".txt")
        for name in MODEL_FILENAMES
    }
    if all(path.is_file() for path in text_files.values()):
        return text_files
    raise ValueError(f"{model}: required COLMAP model files are incomplete")


def inspect_model(
    model: Path,
    *,
    shape: tuple[int, int],
    seed_name: str,
    probe: dict[str, Any],
    cam0: Camera,
    runtime_fingerprint: dict[str, Any],
    seconds: int,
    attempt_id: str,
    work: Path,
) -> dict[str, Any]:
    """Read and validate the real model, returning a self-authenticating result."""
    import pycolmap

    model_files = _required_model_files(model)
    reconstruction = pycolmap.Reconstruction(str(model))
    if len(reconstruction.cameras) != 1:
        raise ValueError(f"{model}: expected one camera, got {len(reconstruction.cameras)}")
    camera = next(iter(reconstruction.cameras.values()))
    params = [float(value) for value in camera.params]
    if camera.model_name != "PINHOLE":
        raise ValueError(f"{model}: camera model {camera.model_name} is not PINHOLE")
    if (camera.width, camera.height) != shape:
        raise ValueError(
            f"{model}: camera shape {(camera.width, camera.height)} != {shape}"
        )
    if len(params) != 4 or not all(math.isfinite(value) for value in params):
        raise ValueError(f"{model}: camera parameters are not finite PINHOLE params")
    if params[0] <= 0 or params[1] <= 0:
        raise ValueError(f"{model}: focal lengths must be positive")

    registered_names = {image.name for image in reconstruction.images.values()}
    expected_names = {row["name"] for row in probe["frames"]}
    if registered_names != expected_names or reconstruction.num_reg_images() != len(expected_names):
        raise ValueError(
            f"{model}: registered names do not exactly cover all probe frames"
        )
    mean_reproj = float(reconstruction.compute_mean_reprojection_error())
    if reconstruction.num_points3D() <= 0 or not math.isfinite(mean_reproj):
        raise ValueError(f"{model}: reprojection error is not finite")

    fx, fy, cx, cy = params
    got = fx / camera.width
    config_path = work / "gluemap_config.yaml"
    seed_dir = work / "seed"
    if not config_path.is_file() or not seed_dir.is_dir():
        raise ValueError(f"{work}: config or seed model is absent")
    persisted_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        persisted_config.get("camera_model") != "PINHOLE"
        or persisted_config.get("refine_intrinsics") is not True
        or persisted_config.get("max_num_tracks", "missing") is not None
    ):
        raise ValueError(f"{work}: scientific bake-off config is not the required policy")
    seed_reconstruction = pycolmap.Reconstruction(str(seed_dir))
    if len(seed_reconstruction.cameras) != 1:
        raise ValueError(f"{seed_dir}: expected exactly one seed camera")
    persisted_seed = next(iter(seed_reconstruction.cameras.values()))
    expected_seed_params = np.asarray(cam0.params, dtype=float)
    if (
        persisted_seed.model_name != "PINHOLE"
        or (persisted_seed.width, persisted_seed.height) != shape
        or not np.allclose(persisted_seed.params, expected_seed_params, rtol=0, atol=1e-8)
    ):
        raise ValueError(f"{seed_dir}: seed camera does not match {seed_name}")
    scientific_config = {
        "camera_model": persisted_config["camera_model"],
        "refine_intrinsics": persisted_config["refine_intrinsics"],
        "num_track_per_img": persisted_config.get("num_track_per_img"),
        "max_num_tracks": persisted_config.get("max_num_tracks"),
    }
    return {
        "schema_version": 1,
        "logical_key": f"{shape[0]}x{shape[1]}/{seed_name}",
        "attempt_id": attempt_id,
        "shape": f"{shape[0]}x{shape[1]}",
        "seq": probe["seq"],
        "seed": seed_name,
        "n_frames": len(expected_names),
        "fx0": cam0.fx,
        "fx_over_w_0": cam0.fx_over_w,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "fx_over_w": got,
        "moved_pct": (got - cam0.fx_over_w) / cam0.fx_over_w * 100,
        "camera_model": camera.model_name,
        "camera_count": len(reconstruction.cameras),
        "camera_width": int(camera.width),
        "camera_height": int(camera.height),
        "registered": int(reconstruction.num_reg_images()),
        "points3D": int(reconstruction.num_points3D()),
        "mean_reproj": mean_reproj,
        "seconds": int(seconds),
        "model_path": str(model.resolve()),
        "model_files_sha256": {
            name: sha256(path) for name, path in sorted(model_files.items())
        },
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "scientific_config": scientific_config,
        "resource_config": {"num_workers": persisted_config.get("num_workers")},
        "seed_model": hash_artifact(seed_dir),
        "probe_input_sha256": probe["probe_input_sha256"],
        "runtime_fingerprint": runtime_fingerprint,
        "generating_provenance": generating_provenance(
            runtime_fingerprint, scientific_config
        ),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }


def validate_result_payload(payload: dict, *, expected_probe_hash: str) -> None:
    """Reject incomplete/nonfinite/wrong-camera marker measurements."""
    if payload.get("camera_model") != "PINHOLE":
        raise ValueError("result camera model must be PINHOLE")
    if payload.get("camera_count") != 1:
        raise ValueError("result must contain exactly one camera")
    finite_fields = ("fx_over_w", "mean_reproj")
    if any(
        not isinstance(payload.get(field), (int, float))
        or not math.isfinite(float(payload[field]))
        for field in finite_fields
    ):
        raise ValueError("result focal and reprojection measurements must be finite")
    if float(payload["fx_over_w"]) <= 0 or float(payload["mean_reproj"]) < 0:
        raise ValueError("result focal/reprojection measurements are outside range")
    if payload.get("registered") != payload.get("n_frames"):
        raise ValueError("result does not register all probe frames")
    if payload.get("probe_input_sha256") != expected_probe_hash:
        raise ValueError("result probe hash does not match the locked probe input")
    if "imported_from" in payload:
        raise ValueError("imported_from laundering is forbidden for COMPLETE results")
    provenance = payload.get("generating_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("fresh generating provenance is required")
    if provenance.get("mode") != "fresh_solve":
        raise ValueError("generating provenance mode must be fresh_solve")
    for name in ("runtime_fingerprint", "scientific_config", "sources", "checkpoints"):
        if not isinstance(provenance.get(name), dict) or not provenance[name]:
            raise ValueError(f"generating provenance {name} must be non-empty")


def _revalidate_result_files(payload: dict) -> None:
    model = Path(payload["model_path"])
    for name, expected in payload.get("model_files_sha256", {}).items():
        path = model / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"model-file SHA-256 drift: {path}")
    config_path = Path(payload["config_path"])
    if not config_path.is_file() or sha256(config_path) != payload.get("config_sha256"):
        raise ValueError(f"config SHA-256 drift: {config_path}")
    seed_record = payload.get("seed_model", {})
    actual_seed = hash_artifact(seed_record.get("path", ""))
    if actual_seed.get("sha256") != seed_record.get("sha256"):
        raise ValueError(f"seed-model SHA-256 drift: {seed_record.get('path')}")


def _complete_status(status: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Bind the COMPLETE marker itself to every required result artifact."""
    result_path = Path(result["model_path"]).parents[1] / "result.json"
    status.update(
        {
            "result_sha256": sha256(result_path) if result_path.is_file() else None,
            "model_path": result["model_path"],
            "model_files_sha256": result["model_files_sha256"],
            "config_path": result["config_path"],
            "config_sha256": result["config_sha256"],
            "seed_model": result["seed_model"],
            "probe_input_sha256": result["probe_input_sha256"],
            "runtime_fingerprint": result["runtime_fingerprint"],
            "generating_provenance": result["generating_provenance"],
        }
    )
    return status


def load_result_markers(
    out_root: Path,
    *,
    validate: bool = True,
    expected_probe_hashes: dict[str, str] | None = None,
) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str], dict]]:
    """Load marker state; only the current matching COMPLETE attempt can count."""
    results: dict[tuple[str, str], dict] = {}
    statuses: dict[tuple[str, str], dict] = {}
    if not out_root.is_dir():
        return results, statuses
    for status_path in sorted(out_root.glob("*/status.json")):
        status = read_json(status_path)
        key = parse_key(status.get("logical_key", ""))
        if key in statuses:
            raise ValueError(
                f"duplicate logical key {keystr(key)} in marker directories"
            )
        statuses[key] = status
        if status.get("state") != "COMPLETE":
            continue
        result_path = status_path.with_name("result.json")
        if not result_path.is_file():
            raise ValueError(f"{keystr(key)}: COMPLETE status has no result.json")
        result = read_json(result_path)
        if result.get("logical_key") != keystr(key):
            raise ValueError(f"{keystr(key)}: result logical key mismatch")
        if result.get("attempt_id") != status.get("attempt_id"):
            raise ValueError(f"{keystr(key)}: status/result attempt mismatch")
        if validate:
            if expected_probe_hashes is None or key[0] not in expected_probe_hashes:
                raise ValueError(f"{keystr(key)}: expected probe hash is unavailable")
            validate_result_payload(
                result, expected_probe_hash=expected_probe_hashes[key[0]]
            )
            _revalidate_result_files(result)
            for field in (
                "model_files_sha256",
                "config_sha256",
                "seed_model",
                "probe_input_sha256",
                "runtime_fingerprint",
                "generating_provenance",
            ):
                if status.get(field) != result.get(field):
                    raise ValueError(
                        f"{keystr(key)}: COMPLETE marker/result {field} mismatch"
                    )
            if status.get("result_sha256") != sha256(result_path):
                raise ValueError(f"{keystr(key)}: result.json SHA-256 drift")
        results[key] = result
    return results, statuses


def emit_aggregate_checks(
    gate: Gate,
    results: dict[tuple[str, str], dict],
    statuses: dict[tuple[str, str], dict],
    *,
    runtime_fingerprint: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Emit the full exact S2b check set for any current marker population."""
    actual_keys = set(results)
    key_metrics = {
        "required_result_keys": sorted(keystr(key) for key in REQUIRED_RESULT_KEYS),
        "actual_result_keys": sorted(keystr(key) for key in actual_keys),
        "missing_result_keys": sorted(
            keystr(key) for key in REQUIRED_RESULT_KEYS - actual_keys
        ),
        "unexpected_result_keys": sorted(
            keystr(key) for key in actual_keys - REQUIRED_RESULT_KEYS
        ),
        "marker_states": {
            keystr(key): status.get("state") for key, status in sorted(statuses.items())
        },
        "runtime_fingerprint": runtime_fingerprint or {},
    }
    results_complete = (
        actual_keys == REQUIRED_RESULT_KEYS
        and set(statuses) == REQUIRED_RESULT_KEYS
        and all(status.get("state") == "COMPLETE" for status in statuses.values())
    )
    if results_complete:
        gate.check(
            "G2.7/results_complete",
            results_complete,
            "6/6 required solves are COMPLETE",
            **key_metrics,
        )
    else:
        gate.incomplete(
            "G2.7/results_complete",
            f"{len(actual_keys & REQUIRED_RESULT_KEYS)}/6 required solves are COMPLETE",
            **key_metrics,
        )

    consensus: dict[str, float] = {}
    for shape in REQUIRED_SHAPES:
        shape_runs = [results.get((shape, seed)) for seed in SEEDS]
        gid = f"G2.7/{shape}"
        if any(run is None for run in shape_runs):
            gate.incomplete(
                gid,
                "both seed solves are not COMPLETE",
                present_seeds=[
                    seed for seed, run in zip(SEEDS, shape_runs, strict=True) if run
                ],
                missing_seeds=[
                    seed for seed, run in zip(SEEDS, shape_runs, strict=True) if not run
                ],
            )
            continue
        a, b = shape_runs
        assert a is not None and b is not None
        spread = abs(a["fx_over_w"] - b["fx_over_w"]) / max(
            min(a["fx_over_w"], b["fx_over_w"]), 1e-12
        )
        identifiable = spread <= AGREE_TOL
        if identifiable:
            consensus[shape] = float(np.mean([a["fx_over_w"], b["fx_over_w"]]))
        gate.check(
            gid,
            identifiable,
            (
                f"two seeds converge within {spread * 100:.3f}%; "
                f"consensus fx/W={consensus.get(shape, float('nan')):.9f}"
            ),
            spread_pct=spread * 100,
            agree_tolerance_pct=AGREE_TOL * 100,
            official69_result=a["fx_over_w"],
            charuco_result=b["fx_over_w"],
        )

    if len(consensus) != len(REQUIRED_SHAPES):
        gate.incomplete(
            "G2.8",
            "all configured resolution consensuses are not yet available",
            consensus=consensus,
            missing_shapes=sorted(set(REQUIRED_SHAPES) - set(consensus)),
        )
    else:
        values = list(consensus.values())
        spread = (max(values) - min(values)) / min(values)
        gate.check(
            "G2.8",
            spread <= CROSS_RESOLUTION_TOL,
            f"configured resolution consensuses spread by {spread * 100:.3f}%",
            consensus=consensus,
            spread_pct=spread * 100,
            tolerance_pct=CROSS_RESOLUTION_TOL * 100,
        )
    return consensus


def stage_gate(run_dir: Path) -> Gate:
    gate = Gate(
        "S2b_intrinsics",
        S2B_REQUIRED_IDS,
        script_path=Path(__file__),
        input_artifacts=stage_material_artifacts("S2b_intrinsics", run_dir),
        source_files=[TS_COMMON, TS_ENV, TS_INTRINSICS, EXTERNAL_CAMERA_RECORD],
    )
    gate.record_predecessor_gate(
        "S2_extract",
        run_dir / "gates" / "S2_extract.json",
        expected_stage="S2_extract",
    )
    return gate


def rebuild_aggregate(
    run_dir: Path, runtime_fingerprint: dict[str, Any]
) -> dict[str, Any]:
    """Rebuild aggregate and gate solely from durable current marker state."""
    probes = {shape: probe_input(run_dir, shape_tuple(shape)) for shape in REQUIRED_SHAPES}
    probe_hashes = {
        shape: probe["probe_input_sha256"] for shape, probe in probes.items()
    }
    out_root = run_dir / "intrinsics_bakeoff"
    results, statuses = load_result_markers(
        out_root, validate=True, expected_probe_hashes=probe_hashes
    )
    gate = stage_gate(run_dir)
    consensus = emit_aggregate_checks(
        gate,
        results,
        statuses,
        runtime_fingerprint=runtime_fingerprint,
    )
    write_json(
        run_dir / "intrinsics_bakeoff.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "required_result_keys": sorted(keystr(key) for key in REQUIRED_RESULT_KEYS),
            "agree_tolerance": AGREE_TOL,
            "cross_resolution_tolerance": CROSS_RESOLUTION_TOL,
            "runs": [results[key] for key in sorted(results)],
            "statuses": {
                keystr(key): statuses[key] for key in sorted(statuses)
            },
            "probe_inputs": probes,
            "consensus_fx_over_w": consensus,
            "runtime_fingerprint": runtime_fingerprint,
        },
    )
    return gate.write(run_dir, fail_hard=False)


def _status_payload(
    key: tuple[str, str],
    state: str,
    attempt_id: str,
    *,
    probe: dict[str, Any],
    runtime: dict[str, Any],
    maxkp: int,
    num_workers: int,
    rerun_from: str | None,
    detail: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "logical_key": keystr(key),
        "state": state,
        "attempt_id": attempt_id,
        "detail": detail,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "scientific_config": {
            "camera_model": "PINHOLE",
            "refine_intrinsics": True,
            "num_track_per_img": maxkp,
            "max_num_tracks": None,
        },
        "resource_config": {
            "num_workers": num_workers,
            "rerun_from": rerun_from,
        },
        "script_sha256": source_sha256(Path(__file__)),
        "source_sha256": {
            "ts_common.py": source_sha256(TS_COMMON),
            "ts_env.py": source_sha256(TS_ENV),
            "ts_intrinsics.py": source_sha256(TS_INTRINSICS),
        },
        "probe_input_sha256": probe["probe_input_sha256"],
        "runtime_fingerprint": runtime,
        "generating_provenance": generating_provenance(
            runtime,
            {
                "camera_model": "PINHOLE",
                "refine_intrinsics": True,
                "num_track_per_img": maxkp,
                "max_num_tracks": None,
            },
        ),
    }


def _memory_metrics() -> dict[str, float]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, rest = line.split(":", 1)
        values[name] = int(rest.split()[0]) * 1024
    metrics = {
        "mem_available_gib": values.get("MemAvailable", 0) / 2**30,
        "swap_free_gib": values.get("SwapFree", 0) / 2**30,
        "vram_free_gib": 0.0,
    }
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode == 0:
        free_mib = [float(line) for line in process.stdout.splitlines() if line.strip()]
        metrics["vram_free_gib"] = max(free_mib, default=0.0) / 1024
    return metrics


def require_4k_resources() -> dict[str, float]:
    metrics = _memory_metrics()
    required = {
        "mem_available_gib": 24.0,
        "swap_free_gib": 6.0,
        "vram_free_gib": 24.0,
    }
    short = {
        name: {"actual": metrics[name], "required": threshold}
        for name, threshold in required.items()
        if metrics[name] < threshold
    }
    if short:
        raise RuntimeError(f"4K resource preflight failed: {short}")
    return metrics


@contextlib.contextmanager
def exclusive_solve_lock(run_dir: Path) -> Iterator[None]:
    """Prevent concurrent S2b solves that could recreate the audited host OOM."""
    lock_path = run_dir / "intrinsics_bakeoff" / ".solve.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another S2b solve is already running") from exc
        yield


def run_one(
    run_dir: Path,
    key: tuple[str, str],
    runtime: dict[str, Any],
    *,
    maxkp: int,
    num_workers: int,
    rerun_from: str | None,
) -> None:
    shape = shape_tuple(key[0])
    seed_name = key[1]
    probe = probe_input(run_dir, shape)
    work = run_dir / "intrinsics_bakeoff" / f"{key[0]}__{seed_name}"
    work.mkdir(parents=True, exist_ok=True)
    names, shape_of = build_probe_dir(run_dir, probe, work)
    cam0: Camera = CANDIDATES[seed_name](*shape)
    write_intrinsics_seed(work / "seed", names, shape_of, {shape: cam0})
    attempt_id = uuid.uuid4().hex
    status_args = {
        "probe": probe,
        "runtime": runtime,
        "maxkp": maxkp,
        "num_workers": num_workers,
        "rerun_from": rerun_from,
    }
    write_json(
        work / "status.json",
        _status_payload(
            key, "RUNNING", attempt_id, detail="child solve launched", **status_args
        ),
    )
    rebuild_aggregate(run_dir, runtime)
    started = time.perf_counter()
    try:
        resources = require_4k_resources() if shape == (3840, 2160) else _memory_metrics()
        with exclusive_solve_lock(run_dir):
            model = run_gluemap(
                work,
                work / "images",
                work / "seed",
                True,
                maxkp=maxkp,
                num_workers=num_workers,
                rerun_from=rerun_from,
            )
        result = inspect_model(
            model,
            shape=shape,
            seed_name=seed_name,
            probe=probe,
            cam0=cam0,
            runtime_fingerprint=runtime,
            seconds=round(time.perf_counter() - started),
            attempt_id=attempt_id,
            work=work,
        )
        result["resource_preflight"] = resources
        write_json(work / "result.json", result)
        complete = _status_payload(
            key, "COMPLETE", attempt_id, detail="model revalidated and committed", **status_args
        )
        complete = _complete_status(complete, result)
        write_json(work / "status.json", complete)
    except BaseException as exc:
        write_json(
            work / "status.json",
            _status_payload(
                key,
                "FAILED",
                attempt_id,
                detail=f"{type(exc).__name__}: {exc}",
                **status_args,
            ),
        )
        rebuild_aggregate(run_dir, runtime)
        raise
    rebuild_aggregate(run_dir, runtime)


def _copy_import_payload(source_work: Path, target_work: Path) -> None:
    if source_work.resolve() == target_work.resolve():
        return
    target_work.mkdir(parents=True, exist_ok=True)
    for dirname in ("seed",):
        source = source_work / dirname
        if source.is_dir():
            shutil.copytree(source, target_work / dirname, dirs_exist_ok=True)
    source_model = source_work / "gluemap" / "gluemap_aba"
    if source_model.is_dir():
        shutil.copytree(
            source_model,
            target_work / "gluemap" / "gluemap_aba",
            dirs_exist_ok=True,
        )
    for filename in ("gluemap_config.yaml", "gluemap.log"):
        source = source_work / filename
        if source.is_file():
            shutil.copy2(source, target_work / filename)


def _copy_salad_descriptor_cache(
    source_work: Path, target_work: Path
) -> dict[str, Any] | None:
    """Copy and re-hash the one sequence-level SALAD cache from a failed solve."""
    candidates = sorted((source_work / "gluemap").glob("*/salad_descriptors.pt"))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(
            f"expected one SALAD descriptor cache under {source_work}, got {candidates}"
        )
    source = candidates[0]
    relative = source.relative_to(source_work)
    target = target_work / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_record = hash_artifact(source)
    target_record = hash_artifact(target)
    if source_record["sha256"] != target_record["sha256"]:
        raise ValueError(f"SALAD descriptor cache copy failed hash validation: {target}")
    return target_record


def collect_existing(
    run_dir: Path, source_run: Path, runtime: dict[str, Any], *, maxkp: int, num_workers: int
) -> None:
    """Preserve a failed cache only; Phase 5 forbids completed-model relabeling."""
    for key in sorted(REQUIRED_RESULT_KEYS):
        shape = shape_tuple(key[0])
        source_probe = probe_input(source_run, shape)
        target_probe = probe_input(run_dir, shape)
        source_work = source_run / "intrinsics_bakeoff" / f"{key[0]}__{key[1]}"
        target_work = run_dir / "intrinsics_bakeoff" / f"{key[0]}__{key[1]}"
        model = source_work / "gluemap" / "gluemap_aba"
        attempt_id = f"import-{uuid.uuid4().hex}"
        status_args = {
            "probe": target_probe,
            "runtime": runtime,
            "maxkp": maxkp,
            "num_workers": num_workers,
            "rerun_from": None,
        }
        if not model.is_dir():
            if source_probe["probe_input_sha256"] != target_probe["probe_input_sha256"]:
                raise ValueError(f"{keystr(key)}: archived/current probe hash mismatch")
            target_work.mkdir(parents=True, exist_ok=True)
            cache_record = _copy_salad_descriptor_cache(source_work, target_work)
            if key == ("3840x2160", "charuco") and cache_record is None:
                raise ValueError(
                    "3840x2160/charuco: archived SALAD descriptor cache is missing"
                )
            status = _status_payload(
                key,
                "NOT_RUN",
                attempt_id,
                detail="no completed gluemap_aba model exists; Phase 3 must solve this slot",
                **status_args,
            )
            status["descriptor_cache"] = cache_record
            write_json(
                target_work / "status.json",
                status,
            )
            continue
        raise RuntimeError(
            f"{keystr(key)}: archived completed-model import is disabled; "
            "rerun this solve under the locked Phase 5 runtime"
        )
    rebuild_aggregate(run_dir, runtime)


def _parse_selection(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    selected = list(allowed) if value == "all" else value.split(",")
    unknown = set(selected) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {label}: {sorted(unknown)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=RUN_ID)
    parser.add_argument("--shapes", default="all")
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--maxkp", type=int, default=1024)
    parser.add_argument("--rerun-from")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--import-results-from", type=Path)
    args = parser.parse_args()

    # Fuhe v2 has one externally fixed working camera. Running a two-seed or
    # cross-resolution bake-off would be both scientifically inapplicable and a
    # needless heavy solve, so persist that fact as typed evidence.
    run_dir = RUNS / args.run_name
    policy_path = run_dir / "intrinsics_policy.json"
    policy = external_camera_policy()
    write_json(policy_path, policy)
    gate = stage_gate(run_dir)
    emit_external_camera_policy_checks(gate, policy)
    gate.write(run_dir)
    log("S2b PASS -- external official69 camera policy bound; no heavy solve launched")
    return

    # Mandatory startup preflight precedes pycolmap import/model work.
    runtime = verify_pycolmap_runtime()
    run_dir = RUNS / args.run_name
    out_root = run_dir / "intrinsics_bakeoff"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.collect_only or args.import_results_from:
        source_run = (args.import_results_from or run_dir).expanduser().resolve()
        collect_existing(
            run_dir, source_run, runtime, maxkp=args.maxkp, num_workers=args.num_workers
        )
    else:
        selected_shapes = _parse_selection(args.shapes, REQUIRED_SHAPES, "shapes")
        selected_seeds = _parse_selection(args.seeds, SEEDS, "seeds")
        probe_hashes = {
            shape: probe_input(run_dir, shape_tuple(shape))["probe_input_sha256"]
            for shape in REQUIRED_SHAPES
        }
        current, _ = load_result_markers(
            out_root, validate=True, expected_probe_hashes=probe_hashes
        )
        for shape in selected_shapes:
            for seed in selected_seeds:
                key = (shape, seed)
                if key in current and not args.rerun_existing:
                    log(f"skip COMPLETE {keystr(key)} (use --rerun-existing to replace)")
                    continue
                run_one(
                    run_dir,
                    key,
                    runtime,
                    maxkp=args.maxkp,
                    num_workers=args.num_workers,
                    rerun_from=args.rerun_from,
                )

    gate_result = rebuild_aggregate(run_dir, runtime)
    if not gate_result["ok"]:
        raise SystemExit(2)
    log("S2b historical diagnostics complete")


if __name__ == "__main__":
    main()
