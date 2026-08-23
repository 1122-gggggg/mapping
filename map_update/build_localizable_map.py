#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a localizable SfM map from field videos.

This is a one-file orchestration wrapper for the pipeline that was used around
sfm_glomap / sfm_reshot25:

  videos -> extracted frames -> MegaLoc retrieval pairs -> MV-RoMa dense matches
  -> hloc dense aggregation -> legacy COLMAP DB -> GLOMAP sparse model
  -> RGB point cloud + XFeat localization bundle.

The file intentionally keeps all important knobs in this CLI so a new site can
reuse the same experience without hunting through several ad-hoc scripts. Heavy
dependencies are resolved from the portable sfm_system layout by default.

Example:

  /usr/bin/python3 build_localizable_map.py \
    --videos /path/P001.mp4 /path/P002.mp4 \
    --work-dir /media/cihcilab/新增磁碟區/my_new_site_map \
    --site-name my_new_site

Resume only selected stages:

  /usr/bin/python3 build_localizable_map.py --config my_new_site_map/build_config.json \
    --stages glomap,color,snap,tracking,triangulate,report
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import operator
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable


def _mvroma_hash_loaded_source_fd(fd: int) -> dict[str, Any]:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError("loaded MV-RoMa source inode is not a regular file")
    digest = hashlib.sha256()
    size = 0
    while True:
        block = os.pread(fd, 4 * 1024 * 1024, size)
        if not block:
            break
        digest.update(block)
        size += len(block)
    after = os.fstat(fd)
    before_signature = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
    )
    after_signature = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    )
    if before_signature != after_signature or size != int(after.st_size):
        raise RuntimeError("loaded MV-RoMa source inode changed while hashing")
    return {"size": size, "sha256": digest.hexdigest()}


def _capture_mvroma_loaded_source(path: str | Path) -> SimpleNamespace:
    resolved = Path(path).resolve(strict=True)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise RuntimeError("O_NOFOLLOW is required for loaded source capture")
    fd = os.open(
        resolved,
        os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        value = os.fstat(fd)
        identity = _mvroma_hash_loaded_source_fd(fd)
        return SimpleNamespace(
            fd=fd,
            path=resolved,
            device=int(value.st_dev),
            inode=int(value.st_ino),
            size=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            ctime_ns=int(value.st_ctime_ns),
            identity=identity,
        )
    except BaseException:
        os.close(fd)
        raise


def attest_mvroma_loaded_source(
    capture: SimpleNamespace | None = None,
) -> dict[str, Any]:
    selected = capture or _MVROMA_LOADED_SOURCE_CAPTURE
    opened = os.fstat(selected.fd)
    signature = (
        int(opened.st_dev),
        int(opened.st_ino),
        int(opened.st_size),
        int(opened.st_mtime_ns),
        int(opened.st_ctime_ns),
    )
    expected_signature = (
        selected.device,
        selected.inode,
        selected.size,
        selected.mtime_ns,
        selected.ctime_ns,
    )
    if signature != expected_signature:
        raise RuntimeError("loaded MV-RoMa source inode changed")
    current = selected.path.lstat()
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or (int(current.st_dev), int(current.st_ino))
        != (selected.device, selected.inode)
    ):
        raise RuntimeError("loaded source path changed after module import")
    identity = _mvroma_hash_loaded_source_fd(selected.fd)
    if identity != selected.identity:
        raise RuntimeError("loaded MV-RoMa source content changed")
    return dict(identity)


_MVROMA_LOADED_SOURCE_CAPTURE = _capture_mvroma_loaded_source(Path(__file__))


def find_system_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if p.name == "sfm_system":
            return p
    return Path("/media/cihcilab/新增磁碟區/sfm_system")


DEFAULT_SYSTEM_ROOT = find_system_root(Path(__file__).resolve())
DEFAULT_TEMPLATE_REPO = str(DEFAULT_SYSTEM_ROOT / "定位" / "source" / "sfm_glomap")
DEFAULT_MVROMA_ROOT = str(DEFAULT_SYSTEM_ROOT / "建圖" / "external_tools" / "MV-RoMa")
DEFAULT_UFM_ROOT = str(DEFAULT_SYSTEM_ROOT / "建圖" / "external_tools" / "UFM")
DEFAULT_UFM_ROOT = str(DEFAULT_SYSTEM_ROOT / "建圖" / "external_tools" / "UFM")
DEFAULT_DG_ROOT = str(DEFAULT_SYSTEM_ROOT / "建圖" / "external_tools" / "doppelgangers-plusplus")
DEFAULT_DG_CHECKPOINT = str(Path(DEFAULT_DG_ROOT) / "checkpoints" / "checkpoint-dg+visym.pth")
DEFAULT_LFOE_GLOMAP = str(DEFAULT_SYSTEM_ROOT / "建圖" / "external_tools" / "LFOE-GlobalSfM" / "build" / "glomap_filter")
DEFAULT_MPSFM_REPO = str(DEFAULT_SYSTEM_ROOT / "建圖" / "external_tools" / "mpsfm")
DEFAULT_GLOMAP = "/home/cihcilab/micromamba/envs/sfm/bin/glomap"
DEFAULT_PY_SFM = "/usr/bin/python3.12"
DEFAULT_PY_SFMDB = "/home/cihcilab/micromamba/envs/sfmdb/bin/python"
DEFAULT_PY_MVROMA = "/home/cihcilab/micromamba/envs/mvroma/bin/python"
DEFAULT_PY_MPSFM = "/home/cihcilab/micromamba/envs/mpsfm/bin/python"

MVROMA_UFM_REPO_ID = "infinity1096/UFM-Refine"
MVROMA_UFM_REVISION = "89cc70107b483170ebfc908630e6f40ca1b77315"
MVROMA_UFM_EXPECTED_FILES = {
    "config.json": {
        "size": 3318,
        "sha256": "c4e4fdd231fe7c6fff6b9f290518fc1f44e2e2b8996c95f1be845c8868e4bc69",
    },
    "model.safetensors": {
        "size": 1911127408,
        "sha256": "ccddd553551a6dc8298bef0f2e9227a70d9ebd4b1d3537bc8b132dd673c625d5",
    },
}
MVROMA_DINOV2_SOURCE_EXPECTED = {
    "file_count": 203,
    "sha256sum_sha256": "949c3df7acb4b4dd097b6d4c7de228d058f1fded32584d397d0848fb2ab815e2",
}
MVROMA_DINOV2_WEIGHTS_EXPECTED = {
    "size": 1217586395,
    "sha256": "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428",
}
MVROMA_CHECKPOINT_EXPECTED = {
    "size": 3525840900,
    "sha256": "0533e67ca2071f510b0335d277936679d1cdca1ea006c87484e7e2a28003c9e3",
}
MVROMA_POST_MODEL_BASE_EXPECTED_SHA256 = (
    "ef67cfa6df768073b5f9cbf4381fa709454d73568bb6b5d25a0ab62cd22b723c"
)
MVROMA_POST_MODEL_EXPECTED_REF = {
    "schema": "o101-post-model-contract-ref/v1",
    "contract_schema": "o101-post-model-contract/v3",
    "sha256": "bd9b9175c9e32ea83adcea412c77dc4e0e928ce76fc26e4b65067b33fd7d3afa",
    "base_sha256": MVROMA_POST_MODEL_BASE_EXPECTED_SHA256,
}
_MVROMA_UNUSED_CONV_BACKBONE_SUFFIXES = (
    "0.0.bias",
    "0.0.weight",
    "0.1.bias",
    "0.1.num_batches_tracked",
    "0.1.running_mean",
    "0.1.running_var",
    "0.1.weight",
    "1.bn1.bias",
    "1.bn1.num_batches_tracked",
    "1.bn1.running_mean",
    "1.bn1.running_var",
    "1.bn1.weight",
    "1.bn2.bias",
    "1.bn2.num_batches_tracked",
    "1.bn2.running_mean",
    "1.bn2.running_var",
    "1.bn2.weight",
    "1.conv1.bias",
    "1.conv1.weight",
    "1.conv2.bias",
    "1.conv2.weight",
)
MVROMA_STATE_LOAD_EXPECTED = {
    "schema": "mvroma-state-load/v1",
    "strict": False,
    "missing_keys": [],
    "unexpected_keys": sorted(
        [
            f"conv_backbones.{index}.{suffix}"
            for index in (1, 2, 4, 8)
            for suffix in _MVROMA_UNUSED_CONV_BACKBONE_SUFFIXES
        ]
        + [
            f"tracktention_modules.{index}.{role}.attn.rope.periods"
            for index in range(12)
            for role in ("sampler", "splatter")
        ]
    ),
}
MVROMA_UFM_STATE_LOAD_EXPECTED = {
    "schema": "ufm-safetensors-state-load/v1",
    "strict": False,
    "missing_keys": [],
    "unexpected_keys": [],
}
MVROMA_NONPERSISTENT_BUFFER_MAX_BYTES = 1024 * 1024
MVROMA_PYTHON_SOURCE_EXPECTED = {
    "mvroma_git_head": "acb09efb0212129ac191031f6e56f150524b304f",
    "mvroma_file_count": 61,
    "mvroma_tree_sha256": "7a848b2efb63d8636f2bdfa285cf2b90a96f90a1ebe7f1288bc25f7d2f2edb49",
    "ufm_file_count": 95,
    "ufm_tree_sha256": "d7ce29c2a172262b1956286e5ba46e9f720abeead955e167c5bcabd7715a9670",
}

MVROMA_TORCH_FLAG_KEYS = (
    "default_dtype",
    "cuda_matmul_allow_tf32",
    "cudnn_allow_tf32",
    "cudnn_benchmark",
    "cudnn_deterministic",
    "deterministic_algorithms",
    "float32_matmul_precision",
    "flash_sdp_enabled",
    "mem_efficient_sdp_enabled",
    "math_sdp_enabled",
    "cudnn_sdp_enabled",
    "fp16_reduced_precision_reduction",
    "bf16_reduced_precision_reduction",
    "fp16_bf16_reduction_math_sdp_allowed",
    "cuda_matmul_allow_fp16_accumulation",
    "cudnn_benchmark_limit",
)
MVROMA_ENVIRONMENT_KEYS = (
    "PYTHONHASHSEED",
    "CUBLAS_WORKSPACE_CONFIG",
    "NVIDIA_TF32_OVERRIDE",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "CUDA_LAUNCH_BLOCKING",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
    "UNICEPTION_FUSED_ATTN",
    "XFORMERS_DISABLED",
)
MVROMA_ATTENTION_BACKEND_KEYS = (
    "uniception_has_fused_attn",
    "uniception_use_fused_attn_raw",
    "uniception_use_fused_attn",
    "mvroma_attention_xformers_available",
    "mvroma_block_xformers_available",
    "dino_attention_xformers_enabled",
    "dino_attention_xformers_available",
    "dino_block_xformers_enabled",
    "dino_block_xformers_available",
    "dino_swiglu_xformers_enabled",
    "dino_swiglu_xformers_available",
)
MVROMA_DINO_MATERIALIZED_BACKEND_KEYS = (
    "dino_attention_xformers_enabled",
    "dino_attention_xformers_available",
    "dino_block_xformers_enabled",
    "dino_block_xformers_available",
    "dino_swiglu_xformers_enabled",
    "dino_swiglu_xformers_available",
)
MVROMA_PATH_ENVIRONMENT_KEYS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
)

ALL_STAGES = [
    "preflight",
    "extract",
    "manifest",
    "pairs",
    "doppelgangers",
    "mvroma",
    "verify_pairs",
    "aggregate",
    "db",
    "glomap",
    "color",
    "snap",
    "tracking",
    "triangulate",
    "report",
]

PY_STAGE_ENV = {
    "pairs": "python_sfm",
    "doppelgangers": "python_sfm",
    "mvroma": "python_mvroma",
    "verify_pairs": "python_sfm",
    "aggregate": "python_sfm",
    "db": "python_sfmdb",
    "color": "python_sfm",
    "snap": "python_sfm",
    "tracking": "python_sfm",
    "triangulate": "python_sfm",
}


STRICT_PROFILES: dict[str, dict[str, Any]] = {
    "football_field_1920": {
        "expected_video_count": 3,
        "video_width": 2688,
        "video_height": 1512,
        "fps_min": 23.9,
        "fps_max": 24.1,
        "min_duration_seconds": 10.0,
        "target_width": 1920,
        "target_height": 1080,
        "min_free_gb": 50.0,
        "camera_model": "FULL_OPENCV",
        "camera_params": [
            1440.7279649640707,
            1437.2942620721813,
            1006.2251477118223,
            538.0787720175211,
            -0.016359355216362784,
            0.256336300878371,
            -0.006099082030819077,
            0.019509803298460405,
            -0.1198628127364991,
            0.0,
            0.0,
            0.0,
        ],
        "camera_param_abs_tol": 1e-6,
        "extract_expected_frame_tolerance": 0.15,
        "extract_min_kept_abs": 20,
        "extract_min_kept_ratio": 0.15,
        "extract_min_parallax_abs": 15,
        "extract_min_parallax_ratio": 0.10,
        "extract_max_pure_rotation_kept_ratio": 0.70,
        "pairs_min_abs": 500,
        "pairs_per_frame": 4.0,
        "pairs_min_largest_component_ratio": 0.90,
        "pairs_min_parallax_component_ratio": 0.80,
        "doppelgangers_min_retention_ratio": 0.30,
        "mvroma_min_dense_ratio": 0.85,
        "verify_min_retention_ratio": 0.50,
        "verify_max_missing_dense_ratio": 0.01,
        "aggregate_min_feature_ratio": 0.85,
        "aggregate_min_pair_retention_ratio": 0.45,
        "aggregate_min_assigned_ratio": 0.90,
        "aggregate_min_mean_matches_per_pair": 100.0,
        "db_min_bytes": 10 * 1024 * 1024,
        "glomap_min_registered_ratio": 0.80,
        "glomap_min_registered_images": 60,
        "glomap_min_points_per_registered_image": 1000,
        "glomap_max_mean_reprojection_error": 2.0,
        "color_min_bytes": 128 * 1024,
        "color_min_vertex_ratio": 0.30,
        "snap_min_ref_registered_ratio": 0.80,
        "tracking_min_ref_snap_ratio": 0.80,
        "tri_min_ref_tracking_ratio": 0.80,
        "tri_min_mean_anchored_per_ref": 50.0,
    },
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def shell_join(cmd: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([str(x)]) for x in cmd)


def run_cmd(cmd: list[str], cwd: str | Path | None = None, env: dict[str, str] | None = None,
            dry_run: bool = False) -> None:
    log("$ " + shell_join([str(x) for x in cmd]))
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], cwd=str(cwd) if cwd else None, env=env, check=True)


def unresolved_shared_libs(binary: str | Path) -> list[str]:
    ldd = shutil.which("ldd")
    if not ldd:
        return []
    proc = subprocess.run(
        [ldd, str(binary)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return [line.strip() for line in proc.stdout.splitlines() if "not found" in line]


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def stage_time_record(stage: str, started: float, ended: float, status: str,
                      error: str = "") -> dict[str, Any]:
    return {
        "stage": stage,
        "status": status,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "ended_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ended)),
        "duration_seconds": round(float(ended - started), 3),
        "error": error,
    }


def read_stage_times(cfg: SimpleNamespace) -> dict[str, Any]:
    path = cfg_paths(cfg).stage_times
    if not path.exists():
        return {"stages": [], "total_seconds": 0.0}
    try:
        data = read_json(path)
        if "stages" not in data:
            data["stages"] = []
        return data
    except Exception as exc:
        return {"stages": [], "total_seconds": 0.0, "read_error": str(exc)}


def append_stage_time(cfg: SimpleNamespace, record: dict[str, Any]) -> None:
    data = read_stage_times(cfg)
    data.setdefault("stages", []).append(record)
    data["total_seconds"] = round(sum(float(x.get("duration_seconds", 0.0)) for x in data["stages"]), 3)
    data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(cfg_paths(cfg).stage_times, data)


def cfg_paths(cfg: SimpleNamespace) -> SimpleNamespace:
    work = as_path(cfg.work_dir)
    paths = {
        "work": work,
        "images": work / "images",
        "logs": work / "logs",
        "megaloc": work / "work" / "megaloc",
        "mvroma": work / "work" / "mvroma",
        "tmp": work / "work" / "tmp",
        "deploy": work / "deploy",
        "reloc_tri": work / "reloc_xfeat_tri",
        "model": work / "glomap",
        "model0": work / "glomap" / "0",
        "manifest": work / "frame_manifest.json",
        "intrinsics": work / "map_intrinsics.json",
        "config": work / "build_config.json",
        "preflight_report": work / "preflight_report.json",
        "report_json": work / "build_report.json",
        "report_md": work / "BUILD_LOCALIZABLE_MAP_REPORT.md",
        "stage_times": work / "stage_times.json",
        "global_desc": work / "work" / "megaloc" / "global-feats-megaloc.h5",
        "pairs": work / "work" / "megaloc" / "pairs_megaloc_mvroma_forced.txt",
        "pairs_uncapped": work / "work" / "megaloc" / "pairs_megaloc_uncapped.txt",
        "pairs_before_dg": work / "work" / "megaloc" / "pairs_before_doppelgangers.txt",
        "pairs_before_verification": work / "work" / "megaloc" / "pairs_before_pair_verification.txt",
        "pairs_verified": work / "work" / "megaloc" / "pairs_verified_dms.txt",
        "pair_verification_report": work / "work" / "megaloc" / "pair_verification_report.json",
        "dense_matches": work / "work" / "mvroma" / "matches-mvroma-dense.h5",
        "features": work / "work" / "mvroma" / "feats-mvroma.h5",
        "aggregate_pairs": work / "work" / "mvroma" / "pairs_aggregate_used.txt",
        "database": work / "work" / "mvroma" / "database_mvroma_forced.db",
        "tmp_database": work / "work" / "tmp" / "database_mvroma_forced_tmp.db",
        "model_standard": work / "glomap_standard",
        "model_standard0": work / "glomap_standard" / "0",
        "model_lfoe": work / "glomap_lfoe",
        "model_lfoe0": work / "glomap_lfoe" / "0",
        "model_dense": work / "glomap_dense_mvroma",
        "model_dense0": work / "glomap_dense_mvroma" / "0",
        "model_mpsfm": work / "mpsfm_model",
        "model_mpsfm0": work / "mpsfm_model" / "0",
        "mpsfm_data": work / "work" / "mpsfm",
        "mpsfm_cache": work / "work" / "mpsfm" / "cache",
        "mpsfm_images": work / "work" / "mpsfm" / "images",
        "mpsfm_image_map": work / "work" / "mpsfm" / "image_name_map.json",
        "mpsfm_intrinsics": work / "work" / "mpsfm" / "intrinsics.yaml",
        "mpsfm_custom_conf": work / "work" / "mpsfm" / "configs" / "roma_m3dv2_large.yaml",
        "mapper_diagnostics": work / "mapper_diagnostics.json",
        "backend_comparison": work / "backend_comparison.json",
        "pair_graph_diagnostics": work / "pair_graph_diagnostics.json",
        "rotation_bridge_report": work / "rotation_bridge_report.json",
        "rgb_ply": work / "deploy" / "map_rgb.ply",
        "snap_bundle": work / "deploy" / "reloc_map_xfeat_snap.pt",
        "tracking_bundle": work / "deploy" / "reloc_map_xfeat_tracking.pt",
        "tri_bundle": work / "deploy" / "reloc_map_xfeat_tri.pt",
    }
    return SimpleNamespace(**paths)


def ensure_dirs(cfg: SimpleNamespace) -> None:
    p = cfg_paths(cfg)
    for d in (p.work, p.images, p.logs, p.megloc if False else p.megaloc,
              p.mvroma, p.tmp, p.deploy, p.reloc_tri, p.model):
        d.mkdir(parents=True, exist_ok=True)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for line in f if line.strip())


def h5_group_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        import h5py
        with h5py.File(path, "r") as fd:
            count = 0

            def visit(_name: str, obj: Any) -> None:
                nonlocal count
                if isinstance(obj, h5py.Group) and any(isinstance(v, h5py.Dataset) for v in obj.values()):
                    count += 1

            fd.visititems(visit)
            return count if count else len(fd.keys())
    except Exception:
        return -1


def open_pycolmap_database(pycolmap_module: Any, path: str | Path) -> Any:
    """Open a COLMAP database across pycolmap Database API variants."""
    db_cls = pycolmap_module.Database
    try:
        return db_cls(str(path))
    except TypeError:
        return db_cls.open(str(path))


COLMAP_CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}


def colmap_db_summary_sqlite(path: str | Path) -> dict[str, Any]:
    uri = f"file:{Path(path).resolve()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        cur = con.cursor()
        image_count = int(cur.execute("select count(*) from images").fetchone()[0])
        cameras = cur.execute("select camera_id, model, width, height from cameras order by camera_id").fetchall()
        camera_models = [
            COLMAP_CAMERA_MODEL_NAMES.get(int(model), str(model))
            for _camera_id, model, _width, _height in cameras
        ]
        camera_resolutions = [
            {"camera_id": int(camera_id), "width": int(width), "height": int(height)}
            for camera_id, _model, width, height in cameras
        ]
        keypointless = [
            str(row[0])
            for row in cur.execute(
                """
                select images.name
                from images
                left join keypoints on images.image_id = keypoints.image_id
                where keypoints.image_id is null or keypoints.rows <= 0
                order by images.image_id
                """
            ).fetchall()
        ]
        return {
            "db_images": image_count,
            "camera_models": camera_models,
            "camera_resolutions": camera_resolutions,
            "images_without_keypoints": keypointless,
        }
    finally:
        con.close()


def strict_enabled(cfg: SimpleNamespace) -> bool:
    return bool(getattr(cfg, "strict_gates", False))


def strict_thresholds(cfg: SimpleNamespace) -> dict[str, Any]:
    if not strict_enabled(cfg):
        return {}
    profile = str(getattr(cfg, "strict_profile", "") or "football_field_1920")
    return dict(STRICT_PROFILES.get(profile, STRICT_PROFILES["football_field_1920"]))


def ratio(num: float, den: float) -> float:
    return float(num) / max(1.0, float(den))


def read_gate_metrics(cfg: SimpleNamespace, stage: str) -> dict[str, Any]:
    path = cfg_paths(cfg).work / "gates" / f"{stage}.json"
    if not path.exists():
        return {}
    data = read_json(path)
    return dict(data.get("metrics", {}))


def parse_rate(value: str | None) -> float:
    text = str(value or "0")
    if "/" in text:
        a, b = text.split("/", 1)
        return float(a) / max(1.0, float(b))
    return float(text)


def scaled_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    if width >= height:
        out_w = min(int(max_side), int(width))
        out_h = int(round(height * out_w / max(1, width)))
    else:
        out_h = min(int(max_side), int(height))
        out_w = int(round(width * out_h / max(1, height)))
    if out_w % 2:
        out_w += 1
    if out_h % 2:
        out_h += 1
    return out_w, out_h


def expected_camera_for_resolution(cfg: SimpleNamespace, width: int, height: int) -> dict[str, Any] | None:
    key = f"{width}x{height}"
    camera_init = getattr(cfg, "camera_init", {}) or {}
    if key in camera_init:
        item = camera_init[key]
        return {"model": str(item["model"]), "params": [float(x) for x in item["params"]]}
    thresholds = strict_thresholds(cfg)
    if int(thresholds.get("target_width", 0)) == width and int(thresholds.get("target_height", 0)) == height:
        return {
            "model": str(thresholds.get("camera_model", "")),
            "params": [float(x) for x in thresholds.get("camera_params", [])],
        }
    return None


def camera_params_close(actual: Iterable[float], expected: Iterable[float], tol: float) -> bool:
    a = [float(x) for x in actual]
    e = [float(x) for x in expected]
    return len(a) == len(e) and all(abs(x - y) <= tol for x, y in zip(a, e))


def count_pair_relations(cfg: SimpleNamespace, pairs: list[tuple[str, str]]) -> Counter[str]:
    _, groups = list_images(cfg_paths(cfg).images) if cfg_paths(cfg).images.exists() else ([], {})
    directions = sequence_directions(cfg, groups)
    folder_of = {name: folder for folder, rel in groups.items() for name in rel}
    counts: Counter[str] = Counter()
    for a, b in pairs:
        rel = pair_relation(a, b, folder_of, directions)
        if rel["cross_direction"]:
            counts["cross_direction"] += 1
        elif rel["cross_video"]:
            counts["cross_video"] += 1
        elif rel["same_video"]:
            counts["same_video"] += 1
    return counts


def ply_vertex_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("rb") as f:
            for raw in f:
                line = raw.decode("ascii", errors="ignore").strip()
                if line.startswith("element vertex "):
                    return int(line.split()[-1])
                if line == "end_header":
                    break
    except Exception:
        return 0
    return 0


def bundle_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "refs": 0}
    try:
        import torch
        data = torch.load(path, map_location="cpu", weights_only=False)
        names = list(data.get("ref_names", []))
        rg = data.get("ref_global")
        shape = list(getattr(rg, "shape", [])) if rg is not None else None
        meta = data.get("meta", {}) if isinstance(data.get("meta", {}), dict) else {}
        return {
            "exists": True,
            "refs": len(names),
            "unique_refs": len(set(names)),
            "ref_global_shape": shape,
            "meta": meta,
            "total_3d_anchored_kp": int(meta.get("total_3d_anchored_kp", 0) or 0),
            "mean_3d_anchored_per_ref": float(meta.get("mean_3d_anchored_per_ref", 0.0) or 0.0),
        }
    except Exception as exc:
        return {"exists": True, "refs": 0, "error": str(exc)}


def glomap_summary(model0: Path) -> dict[str, Any]:
    required = ["cameras.bin", "images.bin", "points3D.bin"]
    present = {name: (model0 / name).exists() for name in required}
    out: dict[str, Any] = {"exists": all(present.values()), "required": present}
    if not out["exists"]:
        return out
    try:
        import pycolmap
        rec = pycolmap.Reconstruction(str(model0))
        out.update({
            "registered_images": int(rec.num_reg_images()),
            "points3D": int(rec.num_points3D()),
            "mean_reprojection_error": float(rec.compute_mean_reprojection_error()) if rec.num_points3D() else None,
        })
    except Exception as exc:
        out["pycolmap_error"] = str(exc)
    return out


def ffprobe_video(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {"path": str(path), "exists": path.exists(), "error": "ffprobe not found"}
    proc = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,duration,nb_frames",
            "-of", "json",
            str(path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return {"path": str(path), "exists": path.exists(), "error": proc.stderr.strip()}
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    if not streams:
        return {"path": str(path), "exists": path.exists(), "error": "no video stream"}
    stream = streams[0]
    fps = parse_rate(stream.get("r_frame_rate"))
    duration = float(stream.get("duration") or 0.0)
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    return {
        "path": str(path),
        "exists": path.exists(),
        "codec_name": stream.get("codec_name", ""),
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "nb_frames": int(stream.get("nb_frames") or 0),
    }


def stage_preflight(cfg: SimpleNamespace) -> None:
    p = cfg_paths(cfg)
    thresholds = strict_thresholds(cfg)
    videos = [as_path(v) for v in getattr(cfg, "videos", [])]
    video_reports: list[dict[str, Any]] = []
    for video in videos:
        report = ffprobe_video(video)
        if report.get("width") and report.get("height"):
            tw, th = scaled_size(int(report["width"]), int(report["height"]), int(cfg.max_side))
            report["target_width"] = tw
            report["target_height"] = th
            report["sanitized_stem"] = sanitize_stem(video)
            report["expected_extracted_frames"] = int(round(float(report.get("duration", 0.0)) * float(cfg.fps)))
        video_reports.append(report)

    target_w = int(thresholds.get("target_width", 0) or 0)
    target_h = int(thresholds.get("target_height", 0) or 0)
    camera = expected_camera_for_resolution(cfg, target_w, target_h) if target_w and target_h else None
    tool_paths = {
        "ffmpeg": shutil.which(str(cfg.ffmpeg)) or "",
        "ffprobe": shutil.which("ffprobe") or "",
        "glomap": str(cfg.glomap_command),
        "python_sfm": str(getattr(cfg, "python_sfm", "")),
        "python_sfmdb": str(getattr(cfg, "python_sfmdb", "")),
        "python_mvroma": str(getattr(cfg, "python_mvroma", "")),
        "template_repo": str(getattr(cfg, "template_repo", "")),
        "mvroma_weights": str(getattr(cfg, "mvroma_weights", "")),
        "doppelgangers_checkpoint": str(getattr(cfg, "doppelgangers_checkpoint", "")),
    }
    tool_exists = {
        name: bool(value) and (Path(value).exists() if "/" in value else bool(shutil.which(value)))
        for name, value in tool_paths.items()
    }
    disk = shutil.disk_usage(p.work.parent)
    gpu_report: dict[str, Any] = {"required": str(getattr(cfg, "device", "")).startswith("cuda")}
    if gpu_report["required"]:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        gpu_report.update({
            "ok": proc.returncode == 0,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        })

    write_json(p.preflight_report, {
        "strict_gates": strict_enabled(cfg),
        "strict_profile": str(getattr(cfg, "strict_profile", "")),
        "videos": video_reports,
        "fps": float(cfg.fps),
        "max_side": int(cfg.max_side),
        "target_resolution": {"width": target_w, "height": target_h},
        "camera": camera,
        "camera_init_json": str(getattr(cfg, "camera_init_json", "")),
        "optimize_intrinsics": int(getattr(cfg, "optimize_intrinsics", 0)),
        "tool_paths": tool_paths,
        "tool_exists": tool_exists,
        "disk": {
            "path": str(p.work.parent),
            "free_bytes": int(disk.free),
            "free_gb": float(disk.free / (1024 ** 3)),
        },
        "gpu": gpu_report,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    log(f"preflight -> {p.preflight_report}")


def validate_stage_gate(stage: str, cfg: SimpleNamespace) -> None:
    if getattr(cfg, "disable_stage_gates", False) or getattr(cfg, "dry_run", False):
        return
    p = cfg_paths(cfg)
    gate_dir = p.work / "gates"
    gate_dir.mkdir(parents=True, exist_ok=True)
    strict = strict_enabled(cfg)
    thresholds = strict_thresholds(cfg)
    ok = True
    reasons: list[str] = []
    metrics: dict[str, Any] = {"stage": stage}

    def require(cond: bool, msg: str) -> None:
        nonlocal ok
        if not cond:
            ok = False
            reasons.append(msg)

    if stage == "preflight":
        require(p.preflight_report.exists(), f"missing {p.preflight_report}")
        data = read_json(p.preflight_report) if p.preflight_report.exists() else {}
        videos = data.get("videos", [])
        metrics.update({
            "videos": len(videos),
            "target_resolution": data.get("target_resolution"),
            "disk_free_gb": data.get("disk", {}).get("free_gb"),
            "tool_exists": data.get("tool_exists", {}),
            "gpu": data.get("gpu", {}),
            "camera": data.get("camera"),
        })
        if strict:
            require(len(videos) == int(thresholds["expected_video_count"]), f"video_count={len(videos)}")
            for item in videos:
                label = Path(str(item.get("path", ""))).name
                require(bool(item.get("exists")), f"{label}: missing video")
                require(int(item.get("width", 0)) == int(thresholds["video_width"]), f"{label}: width={item.get('width')}")
                require(int(item.get("height", 0)) == int(thresholds["video_height"]), f"{label}: height={item.get('height')}")
                fps = float(item.get("fps", 0.0))
                require(float(thresholds["fps_min"]) <= fps <= float(thresholds["fps_max"]), f"{label}: fps={fps:.3f}")
                require(float(item.get("duration", 0.0)) >= float(thresholds["min_duration_seconds"]), f"{label}: duration={item.get('duration')}")
                require(int(item.get("target_width", 0)) == int(thresholds["target_width"]), f"{label}: target_width={item.get('target_width')}")
                require(int(item.get("target_height", 0)) == int(thresholds["target_height"]), f"{label}: target_height={item.get('target_height')}")
            camera = data.get("camera") or {}
            require(camera.get("model") == thresholds["camera_model"], f"camera_model={camera.get('model')}")
            require(
                camera_params_close(camera.get("params", []), thresholds["camera_params"], float(thresholds["camera_param_abs_tol"])),
                "camera_params_mismatch",
            )
            require(int(data.get("optimize_intrinsics", 1)) == 0, f"optimize_intrinsics={data.get('optimize_intrinsics')}")
            tools = data.get("tool_exists", {})
            for name in ("ffmpeg", "ffprobe", "glomap", "python_sfm", "python_sfmdb", "python_mvroma", "template_repo", "mvroma_weights", "doppelgangers_checkpoint"):
                require(bool(tools.get(name)), f"missing_tool_or_path={name}")
            require(float(data.get("disk", {}).get("free_gb", 0.0)) >= float(thresholds["min_free_gb"]), f"disk_free_gb={data.get('disk', {}).get('free_gb')}")
            gpu = data.get("gpu", {})
            if bool(gpu.get("required")):
                require(bool(gpu.get("ok")), f"gpu_unavailable={gpu.get('stderr', '')}")
    elif stage == "extract":
        frames = sorted(p.images.glob("*/*.jpg")) or sorted(p.images.glob("*.jpg"))
        metrics["frames"] = len(frames)
        mq = p.work / "motion_quality.json"
        if mq.exists():
            motion = read_json(mq)
            metrics["motion_gate"] = {
                name: {
                    "total_before": item.get("total_before", 0),
                    "kept": item.get("kept", 0),
                    "rejected": item.get("rejected", 0),
                    "motion_classes": item.get("motion_classes", {}),
                }
                for name, item in motion.get("sequences", {}).items()
            }
            if strict:
                preflight = read_json(p.preflight_report) if p.preflight_report.exists() else {}
                expected_by_stem = {
                    str(v.get("sanitized_stem")): int(v.get("expected_extracted_frames", 0))
                    for v in preflight.get("videos", [])
                }
                for name, item in motion.get("sequences", {}).items():
                    total = int(item.get("total_before", 0))
                    kept = int(item.get("kept", 0))
                    classes = item.get("motion_classes", {})
                    parallax = int(classes.get("parallax", 0)) + int(classes.get("seed", 0))
                    pure_kept = sum(1 for r in item.get("records", []) if r.get("kept") and normalize_motion_class(r.get("motion_class")) == "pure_rotation")
                    expected = expected_by_stem.get(name, 0)
                    if expected:
                        tol = float(thresholds["extract_expected_frame_tolerance"])
                        require(abs(total - expected) <= max(2, int(round(expected * tol))), f"{name}: extracted_frames={total} expected={expected}")
                    require(kept >= max(int(thresholds["extract_min_kept_abs"]), int(math.ceil(total * float(thresholds["extract_min_kept_ratio"])))), f"{name}: kept={kept}/{total}")
                    require(parallax >= max(int(thresholds["extract_min_parallax_abs"]), int(math.ceil(total * float(thresholds["extract_min_parallax_ratio"])))), f"{name}: parallax={parallax}/{total}")
                    require(ratio(pure_kept, kept) <= float(thresholds["extract_max_pure_rotation_kept_ratio"]), f"{name}: pure_rotation_kept_ratio={ratio(pure_kept, kept):.3f}")
        if strict:
            for frame in frames:
                w, h = image_size(frame)
                require(w == int(thresholds["target_width"]) and h == int(thresholds["target_height"]), f"{frame.name}: size={w}x{h}")
        require(len(frames) >= int(cfg.gate_min_frames), f"only {len(frames)} extracted frames")
    elif stage == "manifest":
        require(p.manifest.exists(), f"missing {p.manifest}")
        require(p.intrinsics.exists(), f"missing {p.intrinsics}")
        if p.manifest.exists():
            data = read_json(p.manifest)
            total = int(data.get("total_frames", 0))
            metrics["total_frames"] = total
            if strict:
                image_count = len(sorted(p.images.glob("*/*.jpg")) or sorted(p.images.glob("*.jpg")))
                metrics["image_count"] = image_count
                require(total == image_count, f"manifest_total_frames={total} image_count={image_count}")
                for frame in data.get("frames", []):
                    name = str(frame.get("name", ""))
                    require(int(frame.get("width", 0)) == int(thresholds["target_width"]), f"{name}: manifest_width={frame.get('width')}")
                    require(int(frame.get("height", 0)) == int(thresholds["target_height"]), f"{name}: manifest_height={frame.get('height')}")
                    require(bool(frame.get("motion_class")), f"{name}: missing motion_class")
                    require(bool(frame.get("motion_role")), f"{name}: missing motion_role")
            require(total >= int(cfg.gate_min_frames), f"manifest total_frames={total}")
        if strict and p.intrinsics.exists():
            intr = read_json(p.intrinsics)
            by_res = intr.get("intrinsics_by_resolution", {})
            key = f"{int(thresholds['target_width'])}x{int(thresholds['target_height'])}"
            cam = by_res.get(key, {})
            metrics["camera_resolution_key"] = key
            metrics["camera_model"] = cam.get("model")
            require(cam.get("model") == thresholds["camera_model"], f"map_intrinsics_model={cam.get('model')}")
            require(
                camera_params_close(cam.get("params", []), thresholds["camera_params"], float(thresholds["camera_param_abs_tol"])),
                "map_intrinsics_params_mismatch",
            )
    elif stage == "pairs":
        pairs = count_lines(p.pairs)
        desc_groups = h5_group_count(p.global_desc)
        metrics.update({"pairs": pairs, "global_descriptor_groups": desc_groups})
        if strict:
            manifest_total = int(read_json(p.manifest).get("total_frames", 0)) if p.manifest.exists() else 0
            summary = read_json(p.megaloc / "pairs_summary.json") if (p.megaloc / "pairs_summary.json").exists() else {}
            graph = read_json(p.pair_graph_diagnostics) if p.pair_graph_diagnostics.exists() else {}
            rotation = read_json(p.rotation_bridge_report) if p.rotation_bridge_report.exists() else {}
            roles = load_motion_roles(cfg)
            parallax_nodes = sum(1 for role in roles.values() if is_parallax_role(role))
            largest_ratio = ratio(int(graph.get("largest_component", 0)), int(graph.get("total_frames", manifest_total)))
            parallax_ratio = ratio(int(graph.get("largest_parallax_component_without_bridges", 0)), parallax_nodes)
            rel = graph.get("relations", {})
            cross = int(rel.get("cross_video", 0)) + int(rel.get("cross_direction", 0))
            min_pairs = max(int(thresholds["pairs_min_abs"]), int(math.ceil(manifest_total * float(thresholds["pairs_per_frame"]))))
            metrics.update({
                "manifest_total_frames": manifest_total,
                "pair_kinds": graph.get("pair_kinds", summary.get("pair_kinds", {})),
                "relations": rel,
                "connected_components": graph.get("connected_components", 0),
                "largest_component": graph.get("largest_component", 0),
                "largest_component_ratio": largest_ratio,
                "parallax_components_without_bridges": graph.get("parallax_components_without_bridges", 0),
                "largest_parallax_component_without_bridges": graph.get("largest_parallax_component_without_bridges", 0),
                "parallax_nodes": parallax_nodes,
                "parallax_component_ratio": parallax_ratio,
                "cross_video_or_direction_pairs": cross,
                "required_bridge_frames": rotation.get("required_bridge_frames", 0),
                "min_pairs_required": min_pairs,
            })
            require(desc_groups == manifest_total, f"MegaLoc descriptors={desc_groups} manifest_frames={manifest_total}")
            require(pairs >= min_pairs, f"MegaLoc pairs={pairs} min_required={min_pairs}")
            require(cross > 0, "MegaLoc cross-video pairs=0")
            require(largest_ratio >= float(thresholds["pairs_min_largest_component_ratio"]), f"largest_component_ratio={largest_ratio:.3f}")
            if parallax_ratio < float(thresholds["pairs_min_parallax_component_ratio"]):
                require(int(rotation.get("required_bridge_frames", 0)) > 0, f"parallax_component_ratio={parallax_ratio:.3f} required_bridge_frames=0")
        require(desc_groups >= int(cfg.gate_min_frames), f"MegaLoc descriptors={desc_groups}")
        require(pairs >= int(cfg.gate_min_pairs), f"MegaLoc pairs={pairs}")
    elif stage == "doppelgangers":
        pairs = count_lines(p.pairs)
        before = count_lines(p.pairs_before_dg)
        metrics.update({"pairs_after": pairs, "pairs_before": before})
        if strict:
            retention = ratio(pairs, before)
            after_rel = count_pair_relations(cfg, read_pairs(p.pairs))
            before_rel = count_pair_relations(cfg, read_pairs(p.pairs_before_dg))
            metrics.update({
                "doppelgangers_retention_ratio": retention,
                "relations_after": dict(after_rel),
                "relations_before": dict(before_rel),
            })
            require(retention >= float(thresholds["doppelgangers_min_retention_ratio"]), f"doppelgangers_retention_ratio={retention:.3f}")
            if int(before_rel.get("cross_video", 0)) + int(before_rel.get("cross_direction", 0)) > 0:
                require(int(after_rel.get("cross_video", 0)) + int(after_rel.get("cross_direction", 0)) > 0, "Doppelgangers++ removed all cross-video pairs")
        require(pairs > 0, "Doppelgangers++ left zero pairs")
        if before:
            require(pairs <= before, "Doppelgangers++ pair count grew unexpectedly")
    elif stage == "mvroma":
        groups = h5_group_count(p.dense_matches)
        size = p.dense_matches.stat().st_size if p.dense_matches.exists() else 0
        input_pairs = count_lines(p.pairs)
        metrics.update({"dense_match_groups": groups, "dense_match_bytes": size, "input_pairs": input_pairs})
        if strict and input_pairs:
            dense_ratio = ratio(groups, input_pairs)
            metrics["dense_match_ratio"] = dense_ratio
            require(dense_ratio >= float(thresholds["mvroma_min_dense_ratio"]), f"dense_match_ratio={dense_ratio:.3f}")
        require(groups > 0, "MV-RoMa dense match h5 has no readable groups")
        require(size > 0, "MV-RoMa dense match h5 is empty")
    elif stage == "verify_pairs":
        if getattr(cfg, "pair_verification", "dms") == "off":
            metrics["pair_verification"] = "off"
        else:
            report = read_json(p.pair_verification_report) if p.pair_verification_report.exists() else {}
            before = int(report.get("pairs_before", count_lines(p.pairs_before_verification)))
            after = int(report.get("pairs_after", count_lines(p.pairs_verified) or count_lines(p.pairs)))
            missing = int(report.get("missing_dense_match_groups", 0))
            metrics.update({"pairs_before": before, "pairs_after": after, "mode": report.get("mode"), "missing_dense_match_groups": missing})
            if strict:
                retention = ratio(after, before)
                missing_ratio = ratio(missing, before)
                after_rel = count_pair_relations(cfg, read_pairs(p.pairs))
                before_rel = count_pair_relations(cfg, read_pairs(p.pairs_before_verification))
                metrics.update({
                    "verify_retention_ratio": retention,
                    "missing_dense_match_ratio": missing_ratio,
                    "relations_after": dict(after_rel),
                    "relations_before": dict(before_rel),
                })
                require(retention >= float(thresholds["verify_min_retention_ratio"]), f"verify_retention_ratio={retention:.3f}")
                require(missing_ratio <= float(thresholds["verify_max_missing_dense_ratio"]), f"missing_dense_match_ratio={missing_ratio:.3f}")
                if int(before_rel.get("cross_video", 0)) + int(before_rel.get("cross_direction", 0)) > 0:
                    require(int(after_rel.get("cross_video", 0)) + int(after_rel.get("cross_direction", 0)) > 0, "pair verification removed all cross-video pairs")
            require(p.pair_verification_report.exists(), f"missing {p.pair_verification_report}")
            require(after > 0, "pair verification left zero pairs")
            if before:
                require(after <= before, "pair verification count grew unexpectedly")
    elif stage == "aggregate":
        feat_groups = h5_group_count(p.features)
        agg_pairs = count_lines(p.aggregate_pairs)
        metrics.update({"feature_groups": feat_groups, "aggregate_pairs": agg_pairs})
        if strict:
            manifest_total = int(read_json(p.manifest).get("total_frames", 0)) if p.manifest.exists() else 0
            verified_pairs = count_lines(p.pairs)
            summary = read_json(p.mvroma / "assign_matches_cached_summary.json") if (p.mvroma / "assign_matches_cached_summary.json").exists() else {}
            assigned = int(summary.get("assigned_pairs", 0))
            matches = int(summary.get("matches", 0))
            feature_ratio = ratio(feat_groups, manifest_total)
            pair_retention = ratio(agg_pairs, verified_pairs)
            assigned_ratio = ratio(assigned, agg_pairs)
            mean_matches = ratio(matches, max(1, assigned))
            metrics.update({
                "manifest_total_frames": manifest_total,
                "verified_pairs": verified_pairs,
                "assigned_pairs": assigned,
                "matches": matches,
                "feature_ratio": feature_ratio,
                "aggregate_pair_retention_ratio": pair_retention,
                "assigned_pair_ratio": assigned_ratio,
                "mean_matches_per_assigned_pair": mean_matches,
            })
            require(feature_ratio >= float(thresholds["aggregate_min_feature_ratio"]), f"aggregate_feature_ratio={feature_ratio:.3f}")
            require(pair_retention >= float(thresholds["aggregate_min_pair_retention_ratio"]), f"aggregate_pair_retention_ratio={pair_retention:.3f}")
            require(assigned_ratio >= float(thresholds["aggregate_min_assigned_ratio"]), f"assigned_pair_ratio={assigned_ratio:.3f}")
            require(mean_matches >= float(thresholds["aggregate_min_mean_matches_per_pair"]), f"mean_matches_per_assigned_pair={mean_matches:.1f}")
        require(feat_groups >= int(cfg.gate_min_frames), f"aggregate features={feat_groups}")
        require(agg_pairs > 0, "aggregate used zero pairs")
    elif stage == "db":
        size = p.database.stat().st_size if p.database.exists() else 0
        metrics["database_bytes"] = size
        if strict:
            require(size >= int(thresholds["db_min_bytes"]), f"database_bytes={size}")
            try:
                db_summary = colmap_db_summary_sqlite(p.database)
                image_count = int(db_summary["db_images"])
                feature_groups = h5_group_count(p.features)
                camera_models = list(db_summary["camera_models"])
                keypointless = list(db_summary["images_without_keypoints"])
                metrics.update({
                    "db_images": image_count,
                    "feature_groups": feature_groups,
                    "camera_models": camera_models,
                    "camera_resolutions": db_summary["camera_resolutions"],
                    "images_without_keypoints": keypointless[:20],
                })
                require(image_count == feature_groups, f"db_images={image_count} feature_groups={feature_groups}")
                require(not keypointless, f"images_without_keypoints={len(keypointless)}")
                for model in camera_models:
                    require(str(thresholds["camera_model"]) in model, f"db_camera_model={model}")
            except Exception as exc:
                require(False, f"db_strict_check_error={exc}")
        require(size > 10000, f"COLMAP database too small: {size} bytes")
    elif stage == "glomap":
        metrics.update(glomap_summary(p.model0))
        require(bool(metrics.get("exists")), f"missing GLOMAP model binaries in {p.model0}")
        if "registered_images" in metrics:
            manifest_total = read_json(p.manifest).get("total_frames", 0) if p.manifest.exists() else 0
            reg = int(metrics.get("registered_images", 0))
            registered_ratio = ratio(reg, int(manifest_total))
            metrics["registered_ratio"] = registered_ratio
            if strict:
                points = int(metrics.get("points3D", 0) or 0)
                mean_error = float(metrics.get("mean_reprojection_error") or 999999.0)
                points_per_reg = ratio(points, reg)
                metrics["points_per_registered_image"] = points_per_reg
                require(reg >= int(thresholds["glomap_min_registered_images"]), f"registered_images={reg}")
                require(registered_ratio >= float(thresholds["glomap_min_registered_ratio"]), f"registered_ratio={registered_ratio:.3f}")
                require(points_per_reg >= float(thresholds["glomap_min_points_per_registered_image"]), f"points_per_registered_image={points_per_reg:.1f}")
                require(mean_error <= float(thresholds["glomap_max_mean_reprojection_error"]), f"mean_reprojection_error={mean_error:.3f}")
            require(reg >= int(cfg.gate_min_registered_images), f"registered_images={reg}")
            require(registered_ratio >= float(cfg.gate_min_registered_ratio), f"registered_ratio={registered_ratio:.3f}")
    elif stage == "color":
        size = p.rgb_ply.stat().st_size if p.rgb_ply.exists() else 0
        vertex_count = ply_vertex_count(p.rgb_ply)
        metrics.update({"rgb_ply_bytes": size, "ply_vertices": vertex_count})
        if strict:
            glomap_metrics = read_gate_metrics(cfg, "glomap")
            points = int(glomap_metrics.get("points3D", 0) or 0)
            vertex_ratio = ratio(vertex_count, points)
            metrics["ply_vertex_ratio_vs_points3D"] = vertex_ratio
            require(size >= int(thresholds["color_min_bytes"]), f"rgb_ply_bytes={size}")
            require(vertex_ratio >= float(thresholds["color_min_vertex_ratio"]), f"ply_vertex_ratio={vertex_ratio:.3f}")
        require(size > 1000, f"RGB PLY too small: {size} bytes")
    elif stage == "snap":
        metrics.update(bundle_summary(p.snap_bundle))
        if strict:
            glomap_metrics = read_gate_metrics(cfg, "glomap")
            registered = int(glomap_metrics.get("registered_images", 0) or 0)
            ref_ratio = ratio(int(metrics.get("refs", 0)), registered)
            metrics["ref_registered_ratio"] = ref_ratio
            require(int(metrics.get("refs", 0)) == int(metrics.get("unique_refs", 0)), "snap bundle has duplicate refs")
            require(ref_ratio >= float(thresholds["snap_min_ref_registered_ratio"]), f"snap_ref_registered_ratio={ref_ratio:.3f}")
            shape = metrics.get("ref_global_shape")
            require(bool(shape) and len(shape) == 2 and int(shape[1]) == 8448, f"snap ref_global shape={shape}")
        require(metrics.get("refs", 0) > 0, "snap bundle has no refs")
    elif stage == "tracking":
        metrics.update(bundle_summary(p.tracking_bundle))
        if strict:
            snap_metrics = read_gate_metrics(cfg, "snap")
            snap_refs = int(snap_metrics.get("refs", 0) or 0)
            ref_ratio = ratio(int(metrics.get("refs", 0)), snap_refs)
            metrics["ref_snap_ratio"] = ref_ratio
            require(int(metrics.get("refs", 0)) == int(metrics.get("unique_refs", 0)), "tracking bundle has duplicate refs")
            require(ref_ratio >= float(thresholds["tracking_min_ref_snap_ratio"]), f"tracking_ref_snap_ratio={ref_ratio:.3f}")
        require(metrics.get("refs", 0) > 0, "tracking bundle has no refs")
    elif stage == "triangulate":
        metrics.update(bundle_summary(p.tri_bundle))
        if strict:
            tracking_metrics = read_gate_metrics(cfg, "tracking")
            tracking_refs = int(tracking_metrics.get("refs", 0) or 0)
            ref_ratio = ratio(int(metrics.get("refs", 0)), tracking_refs)
            metrics["ref_tracking_ratio"] = ref_ratio
            require(int(metrics.get("refs", 0)) == int(metrics.get("unique_refs", 0)), "triangulated bundle has duplicate refs")
            require(ref_ratio >= float(thresholds["tri_min_ref_tracking_ratio"]), f"tri_ref_tracking_ratio={ref_ratio:.3f}")
            require(int(metrics.get("total_3d_anchored_kp", 0)) > 0, "total_3d_anchored_kp=0")
            require(float(metrics.get("mean_3d_anchored_per_ref", 0.0)) >= float(thresholds["tri_min_mean_anchored_per_ref"]), f"mean_3d_anchored_per_ref={metrics.get('mean_3d_anchored_per_ref')}")
        require(metrics.get("refs", 0) > 0, "triangulated deployment bundle has no refs")
        shape = metrics.get("ref_global_shape")
        if shape:
            require(len(shape) == 2 and int(shape[1]) == 8448, f"ref_global is not MegaLoc 8448-d: {shape}")
    elif stage == "report":
        require(p.report_json.exists(), f"missing {p.report_json}")
        require(p.report_md.exists(), f"missing {p.report_md}")
        if strict:
            failed_gates = []
            passed_gates = set()
            for gate_path in sorted((p.work / "gates").glob("*.json")):
                if gate_path.name == "report.json":
                    continue
                gate = read_json(gate_path)
                if not bool(gate.get("ok")):
                    failed_gates.append(gate_path.stem)
                else:
                    passed_gates.add(gate_path.stem)
            timing = read_stage_times(cfg)
            latest: dict[str, str] = {}
            for item in timing.get("stages", []):
                latest[str(item.get("stage"))] = str(item.get("status"))
            failed_latest = {
                stage_name: status
                for stage_name, status in latest.items()
                if status == "failed" and stage_name != "report" and stage_name not in passed_gates
            }
            metrics.update({"failed_gates": failed_gates, "latest_failed_stages": failed_latest})
            require(not failed_gates, f"failed_gates={failed_gates}")
            require(not failed_latest, f"latest_failed_stages={failed_latest}")

    result = {
        "stage": stage,
        "ok": ok,
        "reasons": reasons,
        "metrics": metrics,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(gate_dir / f"{stage}.json", result)
    if ok:
        log(f"gate {stage}: PASS {metrics}")
        return
    log(f"gate {stage}: FAIL {reasons}")
    raise SystemExit(f"stage gate failed after {stage}: {'; '.join(reasons)}")


def parse_stages(value: str | None) -> list[str]:
    if not value or value == "all":
        return list(ALL_STAGES)
    stages = [x.strip() for x in value.split(",") if x.strip()]
    bad = [s for s in stages if s not in ALL_STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s): {bad}; valid={ALL_STAGES}")
    if "doppelgangers" in stages:
        return stages
    return [s for s in stages if s != "doppelgangers"]


def sanitize_stem(path: Path) -> str:
    stem = path.stem.strip()
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return stem or "video"


def list_images(image_root: Path) -> tuple[list[str], dict[str, list[str]]]:
    names: list[str] = []
    groups: dict[str, list[str]] = {}
    for folder in sorted([p for p in image_root.iterdir() if p.is_dir()]):
        rel = [f"{folder.name}/{p.name}" for p in sorted(folder.glob("*.jpg"))]
        if not rel:
            rel = [f"{folder.name}/{p.name}" for p in sorted(folder.glob("*.png"))]
        if rel:
            groups[folder.name] = rel
            names.extend(rel)
    return names, groups


def pair_name(n0: str, n1: str) -> str:
    return "/".join((n0.replace("/", "-"), n1.replace("/", "-")))


def build_mvroma_source_jobs(
    pair_lines: Iterable[str], limit_src: int, chunk_size: int
) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    pairs_by_src: dict[str, set[str]] = defaultdict(set)
    group_owners: dict[str, tuple[str, str]] = {}
    for line in pair_lines:
        parts = line.split()
        if len(parts) != 2 or parts[0] == parts[1]:
            continue
        src, target = sorted(parts)
        group = pair_name(src, target)
        owner = group_owners.setdefault(group, (src, target))
        if owner != (src, target):
            raise ValueError(
                f"pair_name collision for {group!r}: {owner!r} and {(src, target)!r}"
            )
        pairs_by_src[src].add(target)

    sources = sorted(pairs_by_src)
    if limit_src:
        sources = sources[:limit_src]

    jobs: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources):
        targets = sorted(pairs_by_src[source])
        chunks = [targets[start:start + chunk_size] for start in range(0, len(targets), chunk_size)]
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        jobs.append(
            {
                "source_index": source_index,
                "source": source,
                "targets": targets,
                "chunks": chunks,
                "shard_name": f"{source_index:06d}-{source_hash}.h5",
            }
        )
    return jobs


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def mvroma_file_content_identity(path: str | Path) -> dict[str, Any]:
    """Hash one stable regular-file inode while keeping paths out of identity."""
    requested = Path(path)
    resolved = requested.resolve(strict=True)
    before_path = resolved.lstat()
    if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
        raise ValueError(f"MV-RoMa identity requires a regular file: {requested}")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise RuntimeError("O_NOFOLLOW is required for MV-RoMa content identity")
    fd = os.open(
        resolved,
        os.O_RDONLY
        | nofollow
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    digest = hashlib.sha256()
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"MV-RoMa identity requires a regular file: {requested}")
        if (opened.st_dev, opened.st_ino) != (
            before_path.st_dev,
            before_path.st_ino,
        ):
            raise ValueError(f"MV-RoMa identity path changed while opening: {requested}")
        while True:
            block = os.read(fd, 4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after_fd = os.fstat(fd)
        opened_tuple = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        after_tuple = (
            after_fd.st_dev,
            after_fd.st_ino,
            after_fd.st_size,
            after_fd.st_mtime_ns,
            after_fd.st_ctime_ns,
        )
        if after_tuple != opened_tuple:
            raise ValueError(f"MV-RoMa identity file changed while hashing: {requested}")
    finally:
        os.close(fd)

    after_path = resolved.lstat()
    if stat.S_ISLNK(after_path.st_mode) or (
        after_path.st_dev,
        after_path.st_ino,
        after_path.st_size,
        after_path.st_mtime_ns,
        after_path.st_ctime_ns,
    ) != opened_tuple:
        raise ValueError(f"MV-RoMa identity pathname changed while hashing: {requested}")
    if requested.resolve(strict=True) != resolved:
        raise ValueError(f"MV-RoMa identity symlink target changed: {requested}")
    return {"size": int(opened.st_size), "sha256": digest.hexdigest()}


def mvroma_tree_content_identity(
    root: str | Path, *, python_only: bool = False
) -> dict[str, Any]:
    """Return a relative-name/content tree identity, excluding generated caches."""
    root_path = Path(root).resolve(strict=True)
    if not root_path.is_dir():
        raise ValueError(f"MV-RoMa identity tree is not a directory: {root}")
    files: list[list[Any]] = []
    for current, directory_names, file_names in os.walk(root_path, followlinks=False):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            child = current_path / name
            if name in {"__pycache__", ".git"}:
                continue
            if child.is_symlink():
                raise ValueError(f"symlink directory in MV-RoMa identity tree: {child}")
            kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            if name.endswith(".pyc") or (python_only and not name.endswith(".py")):
                continue
            child = current_path / name
            if child.is_symlink():
                raise ValueError(f"symlink file in MV-RoMa identity tree: {child}")
            relative = child.relative_to(root_path).as_posix()
            identity = mvroma_file_content_identity(child)
            files.append([relative, identity["size"], identity["sha256"]])
    files.sort(key=lambda item: item[0])
    payload = {"schema": "mvroma-content-tree/v1", "files": files}
    sha256sum_rows = "".join(
        f"{item[2]}  ./{item[0]}\n" for item in files
    ).encode("utf-8")
    return {
        **payload,
        "file_count": len(files),
        "tree_sha256": hashlib.sha256(
            b"mvroma-content-tree-v1\0" + _canonical_json_bytes(payload)
        ).hexdigest(),
        "sha256sum_sha256": hashlib.sha256(sha256sum_rows).hexdigest(),
    }


def _mvroma_selected_source_path(root: Path, relative_name: str) -> Path:
    if not relative_name or "\\" in relative_name:
        raise ValueError(f"invalid MV-RoMa source path: {relative_name!r}")
    relative = Path(relative_name)
    if (
        relative.is_absolute()
        or relative.as_posix() != relative_name
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"unsafe MV-RoMa source path: {relative_name!r}")
    current = root
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            raise
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"symlink MV-RoMa source is not allowed: {relative_name}")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"MV-RoMa source escapes root: {relative_name}") from exc
    if not resolved.is_file():
        raise ValueError(f"MV-RoMa source is not a regular file: {relative_name}")
    return resolved


def _mvroma_frozen_source_digest(files: list[list[Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"o101-source-tree-v1\0")
    for relative, _size, leaf_sha256 in files:
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(leaf_sha256).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_mvroma_frozen_source_tree(
    root: str | Path, relative_paths: Iterable[str]
) -> dict[str, Any]:
    root_path = Path(root).resolve(strict=True)
    if not root_path.is_dir():
        raise ValueError(f"MV-RoMa source root is not a directory: {root}")
    raw_paths = [str(value) for value in relative_paths]
    if len(raw_paths) != len(set(raw_paths)):
        raise ValueError("duplicate MV-RoMa source path")
    files: list[list[Any]] = []
    for relative in sorted(raw_paths):
        source = _mvroma_selected_source_path(root_path, relative)
        identity = mvroma_file_content_identity(source)
        files.append([relative, identity["size"], identity["sha256"]])
    return {
        "schema": "o101-source-tree/v1",
        "algorithm": "sha256(domain_nul + sorted(relative_nul_leafhex_lf))",
        "file_count": len(files),
        "tree_sha256": _mvroma_frozen_source_digest(files),
        "files": files,
    }


def _validate_mvroma_frozen_source_identity(identity: dict[str, Any]) -> None:
    expected_keys = {
        "schema",
        "algorithm",
        "file_count",
        "tree_sha256",
        "files",
    }
    if set(identity) != expected_keys or identity.get("schema") != "o101-source-tree/v1":
        raise ValueError("invalid frozen MV-RoMa source identity schema")
    files = identity.get("files")
    if not isinstance(files, list):
        raise ValueError("invalid frozen MV-RoMa source files")
    names: list[str] = []
    for row in files:
        if (
            not isinstance(row, list)
            or len(row) != 3
            or not isinstance(row[0], str)
            or not isinstance(row[1], int)
            or row[1] < 0
            or not isinstance(row[2], str)
            or re.fullmatch(r"[0-9a-f]{64}", row[2]) is None
        ):
            raise ValueError("invalid frozen MV-RoMa source row")
        names.append(row[0])
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("frozen MV-RoMa source paths must be sorted and unique")
    if identity.get("file_count") != len(files):
        raise ValueError("frozen MV-RoMa source file count mismatch")
    if identity.get("tree_sha256") != _mvroma_frozen_source_digest(files):
        raise ValueError("frozen MV-RoMa source tree digest mismatch")


@contextmanager
def private_attested_mvroma_source_tree(
    source_root: str | Path, identity: dict[str, Any]
) -> Iterable[Path]:
    _validate_mvroma_frozen_source_identity(identity)
    root = Path(source_root).resolve(strict=True)
    names = [str(row[0]) for row in identity["files"]]
    if build_mvroma_frozen_source_tree(root, names) != identity:
        raise RuntimeError("MV-RoMa source root does not match frozen identity")
    with tempfile.TemporaryDirectory(prefix=".o101-source-tree-") as raw:
        private = Path(raw)
        private.chmod(0o700)
        for relative, size, sha256_value in identity["files"]:
            source = _mvroma_selected_source_path(root, str(relative))
            target = private / str(relative)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            current = target.parent
            while current != private:
                current.chmod(0o700)
                current = current.parent
            expected = {"size": int(size), "sha256": str(sha256_value)}
            with open_attested_mvroma_file(source, expected=expected) as handle:
                flags = (
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                output_fd = os.open(target, flags, 0o400)
                with os.fdopen(output_fd, "wb", closefd=True) as output:
                    handle.file.seek(0)
                    shutil.copyfileobj(handle.file, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
            target.chmod(0o400)
        if build_mvroma_frozen_source_tree(private, names) != identity:
            raise RuntimeError("private MV-RoMa source copy identity mismatch")
        try:
            yield private
        finally:
            if build_mvroma_frozen_source_tree(private, names) != identity:
                raise RuntimeError("private MV-RoMa source copy changed while in use")


def attest_mvroma_python_source_roots(
    mvroma_root: str | Path,
    ufm_root: str | Path,
    *,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mvroma = Path(mvroma_root).resolve(strict=True)
    ufm = Path(ufm_root).resolve(strict=True)
    tracked = subprocess.check_output(
        ["git", "ls-files", "*.py"], cwd=mvroma, text=True
    ).splitlines()
    ufm_python = sorted(
        path.relative_to(ufm).as_posix()
        for path in ufm.rglob("*.py")
        if path.is_file()
    )
    mvroma_identity = build_mvroma_frozen_source_tree(mvroma, tracked)
    ufm_identity = build_mvroma_frozen_source_tree(ufm, ufm_python)
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=mvroma, text=True
    ).strip()
    if re.fullmatch(r"[0-9a-f]{40}", git_head) is None:
        raise RuntimeError(f"invalid MV-RoMa git HEAD: {git_head!r}")
    if expected is not None:
        expected_keys = {
            "mvroma_git_head",
            "mvroma_file_count",
            "mvroma_tree_sha256",
            "ufm_file_count",
            "ufm_tree_sha256",
        }
        if set(expected) != expected_keys:
            raise RuntimeError("invalid frozen MV-RoMa/UFM source expectation")
        actual = {
            "mvroma_git_head": git_head,
            "mvroma_file_count": mvroma_identity["file_count"],
            "mvroma_tree_sha256": mvroma_identity["tree_sha256"],
            "ufm_file_count": ufm_identity["file_count"],
            "ufm_tree_sha256": ufm_identity["tree_sha256"],
        }
        if actual != expected:
            raise RuntimeError(
                f"frozen MV-RoMa/UFM source identity mismatch: {actual} != {expected}"
            )
    return {
        "identity": {
            "schema": "o101-python-source-roots/v1",
            "mvroma": mvroma_identity,
            "ufm": ufm_identity,
        },
        "provenance": {
            "mvroma_root": str(mvroma),
            "ufm_root": str(ufm),
            "mvroma_git_head": git_head,
        },
    }


def _is_mvroma_target_module(name: str) -> bool:
    return any(
        name == prefix or name.startswith(f"{prefix}.")
        for prefix in ("src", "uniflowmatch", "uniception", "dinov2")
    )


@contextmanager
def private_mvroma_import_environment(
    mvroma_root: str | Path, ufm_root: str | Path
) -> Iterable[None]:
    import importlib

    preloaded = sorted(name for name in sys.modules if _is_mvroma_target_module(name))
    if preloaded:
        raise RuntimeError(f"preloaded MV-RoMa target modules are forbidden: {preloaded}")
    mvroma = Path(mvroma_root).resolve(strict=True)
    ufm = Path(ufm_root).resolve(strict=True)
    uniception = ufm / "UniCeption"
    if not mvroma.is_dir() or not ufm.is_dir() or not uniception.is_dir():
        raise RuntimeError("private MV-RoMa/UFM/UniCeption source roots are incomplete")
    saved_path = list(sys.path)
    saved_cwd = Path.cwd()
    saved_dont_write = sys.dont_write_bytecode
    saved_pycache_prefix = sys.pycache_prefix
    with tempfile.TemporaryDirectory(prefix=".o101-pycache-") as raw_cache:
        cache = Path(raw_cache)
        cache.chmod(0o700)
        try:
            sys.path[:] = [str(mvroma), str(ufm), str(uniception), *saved_path]
            os.chdir(mvroma)
            sys.dont_write_bytecode = True
            sys.pycache_prefix = str(cache)
            importlib.invalidate_caches()
            yield
        finally:
            for name in list(sys.modules):
                if _is_mvroma_target_module(name):
                    sys.modules.pop(name, None)
            sys.path[:] = saved_path
            os.chdir(saved_cwd)
            sys.dont_write_bytecode = saved_dont_write
            sys.pycache_prefix = saved_pycache_prefix
            importlib.invalidate_caches()


def _mvroma_safe_image_path(image_root: Path, name: str) -> Path:
    if not name or "\\" in name:
        raise ValueError(f"invalid MV-RoMa image name: {name!r}")
    relative = Path(name)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"unsafe MV-RoMa image name: {name!r}")
    candidate = image_root / relative
    current = image_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink MV-RoMa image is not allowed: {name}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(image_root)
    except ValueError as exc:
        raise ValueError(f"MV-RoMa image escapes image root: {name}") from exc
    if not resolved.is_file():
        raise ValueError(f"MV-RoMa image is not a regular file: {name}")
    return resolved


def build_mvroma_image_sha256_tree(
    image_root: str | Path, jobs: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    root = Path(image_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"MV-RoMa image root is not a directory: {image_root}")
    names: set[str] = set()
    for job in jobs:
        names.add(str(job["source"]))
        names.update(str(name) for name in job["targets"])
    files: list[list[Any]] = []
    by_name: dict[str, str] = {}
    for name in sorted(names):
        identity = mvroma_file_content_identity(_mvroma_safe_image_path(root, name))
        files.append([name, identity["size"], identity["sha256"]])
        by_name[name] = str(identity["sha256"])
    payload = {"schema": "mvroma-image-tree/v1", "files": files}
    return {
        **payload,
        "file_count": len(files),
        "tree_sha256": hashlib.sha256(
            b"mvroma-image-tree-v1\0" + _canonical_json_bytes(payload)
        ).hexdigest(),
        "by_name": by_name,
    }


def snapshot_mvroma_image_sha256_tree(
    image_tree: dict[str, Any], jobs: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    required = {"schema", "files", "file_count", "tree_sha256", "by_name"}
    if not isinstance(image_tree, dict) or set(image_tree) != required:
        raise ValueError("invalid MV-RoMa image tree keys")
    if image_tree["schema"] != "mvroma-image-tree/v1":
        raise ValueError("invalid MV-RoMa image tree schema")
    rows = image_tree["files"]
    if not isinstance(rows, list):
        raise ValueError("invalid MV-RoMa image tree files")
    normalized_rows: list[list[Any]] = []
    names: list[str] = []
    for row in rows:
        if (
            not isinstance(row, list)
            or len(row) != 3
            or not isinstance(row[0], str)
            or not row[0]
            or "\\" in row[0]
            or Path(row[0]).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(row[0]).parts)
            or not isinstance(row[1], int)
            or isinstance(row[1], bool)
            or row[1] < 0
            or not isinstance(row[2], str)
            or re.fullmatch(r"[0-9a-f]{64}", row[2]) is None
        ):
            raise ValueError("invalid MV-RoMa image tree row")
        names.append(row[0])
        normalized_rows.append([row[0], row[1], row[2]])
    if names != sorted(set(names)):
        raise ValueError("invalid MV-RoMa image tree name order")
    if image_tree["file_count"] != len(normalized_rows):
        raise ValueError("invalid MV-RoMa image tree file count")
    by_name = image_tree["by_name"]
    expected_by_name = {row[0]: row[2] for row in normalized_rows}
    if not isinstance(by_name, dict) or by_name != expected_by_name:
        raise ValueError("invalid MV-RoMa image tree by-name identity")
    payload = {"schema": "mvroma-image-tree/v1", "files": normalized_rows}
    expected_tree_sha256 = hashlib.sha256(
        b"mvroma-image-tree-v1\0" + _canonical_json_bytes(payload)
    ).hexdigest()
    if image_tree["tree_sha256"] != expected_tree_sha256:
        raise ValueError("invalid MV-RoMa image tree digest")
    referenced_names: set[str] = set()
    for job in jobs:
        referenced_names.add(str(job["source"]))
        referenced_names.update(str(value) for value in job["targets"])
    if referenced_names != set(names):
        raise ValueError("invalid MV-RoMa image tree job coverage")
    return {
        **payload,
        "file_count": len(normalized_rows),
        "tree_sha256": expected_tree_sha256,
        "by_name": expected_by_name,
    }


def _canonical_mvroma_post_model_expectation_ref(
    value: dict[str, Any],
) -> dict[str, str]:
    expected_ref_keys = {
        "schema",
        "contract_schema",
        "sha256",
        "base_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_ref_keys
        or value.get("schema") != "o101-post-model-contract-ref/v1"
        or value.get("contract_schema") != "o101-post-model-contract/v3"
        or any(
            not isinstance(value.get(key), str)
            or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None
            for key in ("sha256", "base_sha256")
        )
    ):
        raise ValueError("invalid MV-RoMa post-model expectation reference")
    return {key: str(value[key]) for key in sorted(expected_ref_keys)}


def build_mvroma_stage_contract(
    *,
    implementation: dict[str, Any],
    models: dict[str, Any],
    inference: dict[str, Any],
    runtime: dict[str, Any],
    post_model_expectation_ref: dict[str, Any],
) -> dict[str, Any]:
    if runtime.get("phase") != "post_import_pre_model":
        raise ValueError("MV-RoMa runtime identity must be post_import_pre_model")
    forbidden_runtime_keys = {"provenance", "environment_paths"} & set(runtime)
    if forbidden_runtime_keys:
        raise ValueError(
            "MV-RoMa runtime provenance is forbidden in the stage contract: "
            f"{sorted(forbidden_runtime_keys)}"
        )
    canonical_post_model_ref = _canonical_mvroma_post_model_expectation_ref(
        post_model_expectation_ref
    )
    candidate = {
        "schema": "mvroma-stage-contract/v2",
        "implementation": implementation,
        "models": models,
        "inference": inference,
        "runtime": runtime,
        "post_model_expectation_ref": canonical_post_model_ref,
    }
    try:
        encoded = json.dumps(
            candidate,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("MV-RoMa stage contract must contain finite JSON values") from exc
    payload = json.loads(encoded.decode("utf-8"))
    return {"payload": payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def _reject_mvroma_absolute_identity_paths(value: Any, trail: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_mvroma_absolute_identity_paths(child, f"{trail}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _reject_mvroma_absolute_identity_paths(child, f"{trail}[{index}]")
        return
    if isinstance(value, str) and (
        value.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:[\\/]", value) is not None
    ):
        raise ValueError(f"absolute path in MV-RoMa identity at {trail}")


def snapshot_mvroma_stage_contract(
    stage_contract: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(stage_contract, dict) or set(stage_contract) != {
        "payload",
        "sha256",
    }:
        raise ValueError("invalid MV-RoMa prepared stage contract")
    payload = stage_contract["payload"]
    required_payload = {
        "schema",
        "implementation",
        "models",
        "inference",
        "runtime",
        "post_model_expectation_ref",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required_payload
        or payload.get("schema") != "mvroma-stage-contract/v2"
        or not all(
            isinstance(payload.get(key), dict)
            for key in ("implementation", "models", "inference", "runtime")
        )
        or payload["runtime"].get("phase") != "post_import_pre_model"
    ):
        raise ValueError("invalid MV-RoMa stage contract payload")
    canonical_ref = _canonical_mvroma_post_model_expectation_ref(
        payload["post_model_expectation_ref"]
    )
    if payload["post_model_expectation_ref"] != canonical_ref:
        raise ValueError("invalid MV-RoMa stage contract payload reference")
    claimed_sha256 = stage_contract["sha256"]
    if (
        not isinstance(claimed_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", claimed_sha256) is None
    ):
        raise ValueError("invalid MV-RoMa prepared stage contract digest")
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("MV-RoMa stage contract must contain finite JSON values") from exc
    actual_sha256 = hashlib.sha256(encoded).hexdigest()
    if claimed_sha256 != actual_sha256:
        raise RuntimeError(
            "MV-RoMa stage contract hash mismatch: "
            f"{claimed_sha256} != {actual_sha256}"
        )
    snapshot_payload = json.loads(encoded.decode("utf-8"))
    _reject_mvroma_absolute_identity_paths(snapshot_payload)
    rebuilt = build_mvroma_stage_contract(
        implementation=snapshot_payload["implementation"],
        models=snapshot_payload["models"],
        inference=snapshot_payload["inference"],
        runtime=snapshot_payload["runtime"],
        post_model_expectation_ref=snapshot_payload[
            "post_model_expectation_ref"
        ],
    )
    snapshot = {"payload": snapshot_payload, "sha256": claimed_sha256}
    if rebuilt != snapshot:
        raise RuntimeError("MV-RoMa prepared stage contract rebuild mismatch")
    return snapshot


def _validate_mvroma_class_identity(value: Any, *, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"module", "qualname"}:
        raise ValueError(f"invalid MV-RoMa {label} class identity")
    if not all(isinstance(value[key], str) and value[key] for key in value):
        raise ValueError(f"invalid MV-RoMa {label} class identity")


def _validate_mvroma_object_identity(value: Any, *, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"type", "mro"}:
        raise ValueError(f"invalid MV-RoMa {label} object identity")
    _validate_mvroma_class_identity(value["type"], label=f"{label} type")
    if not isinstance(value["mro"], list) or not value["mro"]:
        raise ValueError(f"invalid MV-RoMa {label} MRO identity")
    for index, row in enumerate(value["mro"]):
        _validate_mvroma_class_identity(row, label=f"{label} MRO[{index}]")


def _validate_mvroma_load_key_identity(
    value: Any, *, schema: str, label: str
) -> None:
    required = {"schema", "strict", "missing_keys", "unexpected_keys"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"invalid MV-RoMa {label} load-key identity")
    if value["schema"] != schema or value["strict"] is not False:
        raise ValueError(f"invalid MV-RoMa {label} load-key identity")
    for key in ("missing_keys", "unexpected_keys"):
        rows = value[key]
        if (
            not isinstance(rows, list)
            or not all(isinstance(row, str) for row in rows)
            or rows != sorted(set(rows))
        ):
            raise ValueError(f"invalid MV-RoMa {label} load-key identity")


def build_mvroma_post_model_contract(post_model: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "mvroma",
        "mvroma_state_load",
        "ufm_runner_class",
        "ufm_runner_is_vendored_class",
        "ufm",
        "ufm_state_load",
        "encoder",
        "dinov2_model",
        "fused_attention",
        "dinov2_blocks",
        "module_identity",
        "model_state_identity",
        "runtime_identity",
        "post_model_assets",
    }
    if set(post_model) != required:
        raise ValueError(
            "MV-RoMa post-model identity keys mismatch: "
            f"{sorted(set(post_model) ^ required)}"
        )
    if post_model["schema"] != "o101-post-model-identity/v1":
        raise ValueError("invalid MV-RoMa post-model identity schema")
    for key in ("mvroma", "ufm", "encoder", "dinov2_model"):
        _validate_mvroma_object_identity(post_model[key], label=key)
    _validate_mvroma_class_identity(
        post_model["ufm_runner_class"], label="UFM runner"
    )
    if post_model["ufm_runner_is_vendored_class"] is not True:
        raise ValueError("MV-RoMa UFM runner is not the vendored class")
    _validate_mvroma_load_key_identity(
        post_model["mvroma_state_load"],
        schema="mvroma-state-load/v1",
        label="MV-RoMa",
    )
    _validate_mvroma_load_key_identity(
        post_model["ufm_state_load"],
        schema="ufm-safetensors-state-load/v1",
        label="UFM",
    )
    fused_attention = post_model["fused_attention"]
    if not isinstance(fused_attention, list) or not fused_attention:
        raise ValueError("MV-RoMa post-model fused-attention identity is empty")
    fused_names: list[str] = []
    for index, row in enumerate(fused_attention):
        if not isinstance(row, dict) or set(row) != {
            "name",
            "identity",
            "fused_attn",
        }:
            raise ValueError("invalid MV-RoMa fused-attention row")
        if not isinstance(row["name"], str) or not row["name"]:
            raise ValueError("invalid MV-RoMa fused-attention name")
        if not isinstance(row["fused_attn"], bool):
            raise ValueError("invalid MV-RoMa fused-attention value")
        _validate_mvroma_object_identity(
            row["identity"], label=f"fused-attention[{index}]"
        )
        fused_names.append(row["name"])
    if len(fused_names) != len(set(fused_names)):
        raise ValueError("duplicate MV-RoMa fused-attention name")
    dino_blocks = post_model["dinov2_blocks"]
    if not isinstance(dino_blocks, list) or not dino_blocks:
        raise ValueError("MV-RoMa post-model DINO block identity is empty")
    for index, row in enumerate(dino_blocks):
        if (
            not isinstance(row, dict)
            or set(row) != {"index", "block", "attention"}
            or row["index"] != index
        ):
            raise ValueError("invalid MV-RoMa DINO block coverage")
        _validate_mvroma_object_identity(row["block"], label=f"DINO block[{index}]")
        _validate_mvroma_object_identity(
            row["attention"], label=f"DINO attention[{index}]"
        )
    runtime = post_model["runtime_identity"]
    runtime_keys = {
        "phase",
        "versions",
        "torch_flags",
        "gpu",
        "environment",
        "attention_backend",
    }
    if not isinstance(runtime, dict) or set(runtime) != runtime_keys:
        raise ValueError("MV-RoMa post-model runtime identity keys mismatch")
    if runtime["phase"] != "post_model_pre_publish":
        raise ValueError("MV-RoMa post-model runtime phase mismatch")
    if set(runtime["attention_backend"]) != set(MVROMA_ATTENTION_BACKEND_KEYS):
        raise ValueError("MV-RoMa post-model attention backend keys mismatch")
    module_identity = post_model["module_identity"]
    if not isinstance(module_identity, dict) or set(module_identity) != {
        "schema",
        "modules",
        "sha256",
    }:
        raise ValueError("MV-RoMa post-model module identity keys mismatch")
    if module_identity["schema"] != "mvroma-module-origins/v1" or not isinstance(
        module_identity["modules"], list
    ):
        raise ValueError("invalid MV-RoMa post-model module identity schema")
    module_payload = {
        "schema": module_identity["schema"],
        "modules": module_identity["modules"],
    }
    expected_module_hash = hashlib.sha256(
        b"mvroma-module-origins-v1\0" + _canonical_json_bytes(module_payload)
    ).hexdigest()
    if module_identity["sha256"] != expected_module_hash:
        raise ValueError("MV-RoMa post-model module identity hash mismatch")
    model_state_identity = post_model["model_state_identity"]
    if not isinstance(model_state_identity, dict) or set(model_state_identity) != {
        "schema",
        "nonpersistent_content_encoding",
        "maximum_nonpersistent_content_bytes_per_model",
        "models",
        "sha256",
    }:
        raise ValueError("MV-RoMa post-model state identity keys mismatch")
    if (
        model_state_identity["schema"] != "o101-model-state-identity/v2"
        or model_state_identity["nonpersistent_content_encoding"]
        != "numpy-c-order-bytes/v1"
        or model_state_identity["maximum_nonpersistent_content_bytes_per_model"]
        != MVROMA_NONPERSISTENT_BUFFER_MAX_BYTES
        or not isinstance(model_state_identity["models"], dict)
        or set(model_state_identity["models"]) != {"mvroma", "ufm"}
    ):
        raise ValueError("invalid MV-RoMa post-model state identity schema")
    model_state_payload = {
        "schema": model_state_identity["schema"],
        "nonpersistent_content_encoding": model_state_identity[
            "nonpersistent_content_encoding"
        ],
        "maximum_nonpersistent_content_bytes_per_model": model_state_identity[
            "maximum_nonpersistent_content_bytes_per_model"
        ],
        "models": model_state_identity["models"],
    }
    expected_model_state_hash = hashlib.sha256(
        b"o101-model-state-identity-v2\0"
        + _canonical_json_bytes(model_state_payload)
    ).hexdigest()
    if model_state_identity["sha256"] != expected_model_state_hash:
        raise ValueError("MV-RoMa post-model state identity hash mismatch")
    assets = post_model["post_model_assets"]
    if not isinstance(assets, dict) or set(assets) != {
        "schema",
        "files",
        "dinov2_source",
    }:
        raise ValueError("MV-RoMa post-model asset identity keys mismatch")
    base_candidate = {
        "schema": "o101-post-model-contract/v1",
        "mvroma": post_model["mvroma"],
        "mvroma_state_load": post_model["mvroma_state_load"],
        "ufm_runner_class": post_model["ufm_runner_class"],
        "ufm_runner_is_vendored_class": post_model[
            "ufm_runner_is_vendored_class"
        ],
        "ufm": post_model["ufm"],
        "ufm_state_load": post_model["ufm_state_load"],
        "encoder": post_model["encoder"],
        "dinov2_model": post_model["dinov2_model"],
        "fused_attention": post_model["fused_attention"],
        "dinov2_blocks": post_model["dinov2_blocks"],
        "module_identity": module_identity,
        "runtime": {
            "phase": runtime["phase"],
            "attention_backend": runtime["attention_backend"],
        },
    }
    candidate = {
        **base_candidate,
        "schema": "o101-post-model-contract/v3",
        "model_state_identity": model_state_identity,
    }
    try:
        base_encoded = json.dumps(
            base_candidate,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        encoded = json.dumps(
            candidate,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "MV-RoMa post-model contract must contain finite JSON values"
        ) from exc
    payload = json.loads(encoded.decode("utf-8"))
    base_payload = json.loads(base_encoded.decode("utf-8"))
    _reject_mvroma_absolute_identity_paths(payload)
    _reject_mvroma_absolute_identity_paths(base_payload)
    return {
        "payload": payload,
        "sha256": hashlib.sha256(
            b"o101-post-model-contract-v3\0" + encoded
        ).hexdigest(),
        "base_sha256": hashlib.sha256(
            b"o101-post-model-contract-v1\0" + base_encoded
        ).hexdigest(),
    }


def validate_mvroma_post_model_runtime(
    pre_model: dict[str, Any], post_model: dict[str, Any]
) -> None:
    required = {
        "phase",
        "versions",
        "torch_flags",
        "gpu",
        "environment",
        "attention_backend",
    }
    if set(pre_model) != required or set(post_model) != required:
        raise ValueError("MV-RoMa pre/post runtime keys mismatch")
    if pre_model["phase"] != "post_import_pre_model":
        raise ValueError("MV-RoMa pre-model runtime phase mismatch")
    if post_model["phase"] != "post_model_pre_publish":
        raise ValueError("MV-RoMa post-model runtime phase mismatch")
    for key in ("versions", "torch_flags", "gpu", "environment"):
        if post_model[key] != pre_model[key]:
            raise RuntimeError(f"MV-RoMa post-model runtime {key} drift")
    pre_backend = pre_model["attention_backend"]
    post_backend = post_model["attention_backend"]
    if set(pre_backend) != set(MVROMA_ATTENTION_BACKEND_KEYS) or set(
        post_backend
    ) != set(MVROMA_ATTENTION_BACKEND_KEYS):
        raise ValueError("MV-RoMa pre/post attention backend keys mismatch")
    materialized = set(MVROMA_DINO_MATERIALIZED_BACKEND_KEYS)
    for key in MVROMA_ATTENTION_BACKEND_KEYS:
        if key in materialized:
            if pre_backend[key] is not None or not isinstance(
                post_backend[key], bool
            ):
                raise RuntimeError(
                    f"MV-RoMa DINO backend transition mismatch at {key}"
                )
        elif post_backend[key] != pre_backend[key]:
            raise RuntimeError(f"MV-RoMa attention backend drift at {key}")


def verify_mvroma_post_model_expectation(
    post_model: dict[str, Any],
    *,
    pre_model_runtime: dict[str, Any],
    expected_ref: dict[str, Any],
) -> dict[str, Any]:
    canonical_expected = _canonical_mvroma_post_model_expectation_ref(
        expected_ref
    )
    validate_mvroma_post_model_runtime(
        pre_model_runtime, post_model.get("runtime_identity", {})
    )
    contract = build_mvroma_post_model_contract(post_model)
    actual_ref = {
        "base_sha256": contract["base_sha256"],
        "contract_schema": contract["payload"]["schema"],
        "schema": "o101-post-model-contract-ref/v1",
        "sha256": contract["sha256"],
    }
    if actual_ref != canonical_expected:
        raise RuntimeError(
            "MV-RoMa post-model expectation mismatch: "
            f"{actual_ref} != {canonical_expected}"
        )
    return contract


def collect_mvroma_model_state_identity(
    models: dict[str, Any], *, expected_device: str
) -> dict[str, Any]:
    if set(models) != {"mvroma", "ufm"}:
        raise ValueError(
            "MV-RoMa model-state roles mismatch: "
            f"{sorted(set(models) ^ {'mvroma', 'ufm'})}"
        )
    identity_models: dict[str, Any] = {}
    provenance_models: dict[str, Any] = {}
    for role in sorted(models):
        model = models[role]
        named_modules = list(model.named_modules())
        module_names = [str(name) for name, _module in named_modules]
        if len(module_names) != len(set(module_names)):
            raise RuntimeError(f"duplicate {role} named-module identity")
        training_modules = sorted(
            str(name)
            for name, module in named_modules
            if bool(getattr(module, "training", True))
        )
        if training_modules:
            raise RuntimeError(f"{role} has training modules: {training_modules}")

        try:
            named_parameters = list(model.named_parameters(remove_duplicate=False))
            named_buffers = list(model.named_buffers(remove_duplicate=False))
        except TypeError as exc:
            raise RuntimeError(
                "MV-RoMa model-state identity requires remove_duplicate support"
            ) from exc
        if not named_parameters:
            raise RuntimeError(f"{role} model has no parameters")

        buffer_persistence: dict[str, bool] = {}
        for module_name, module in named_modules:
            buffers = getattr(module, "_buffers", {})
            non_persistent = set(
                getattr(module, "_non_persistent_buffers_set", set())
            )
            for local_name, tensor in buffers.items():
                if tensor is None:
                    continue
                full_name = (
                    str(local_name)
                    if not module_name
                    else f"{module_name}.{local_name}"
                )
                buffer_persistence[full_name] = local_name not in non_persistent

        parameter_rows: list[dict[str, Any]] = []
        parameter_aliases: dict[int, int] = {}
        parameter_devices: Counter[str] = Counter()
        for name, tensor in sorted(named_parameters, key=lambda item: str(item[0])):
            name = str(name)
            alias_group = parameter_aliases.setdefault(
                id(tensor), len(parameter_aliases)
            )
            device = str(tensor.device)
            parameter_devices[device] += 1
            if device != expected_device:
                raise RuntimeError(
                    f"{role} parameter device mismatch at {name}: "
                    f"{device} != {expected_device}"
                )
            parameter_rows.append(
                {
                    "name": name,
                    "shape": [int(size) for size in tensor.shape],
                    "dtype": str(tensor.dtype),
                    "requires_grad": bool(tensor.requires_grad),
                    "alias_group": alias_group,
                }
            )
        if len(parameter_rows) != len({row["name"] for row in parameter_rows}):
            raise RuntimeError(f"duplicate {role} parameter name")

        buffer_rows: list[dict[str, Any]] = []
        buffer_aliases: dict[int, int] = {}
        buffer_devices: Counter[str] = Counter()
        nonpersistent_content_bytes = 0
        for name, tensor in sorted(named_buffers, key=lambda item: str(item[0])):
            name = str(name)
            if name not in buffer_persistence:
                raise RuntimeError(f"missing {role} buffer persistence at {name}")
            alias_group = buffer_aliases.setdefault(id(tensor), len(buffer_aliases))
            device = str(tensor.device)
            buffer_devices[device] += 1
            if device != expected_device:
                raise RuntimeError(
                    f"{role} buffer device mismatch at {name}: "
                    f"{device} != {expected_device}"
                )
            persistent = buffer_persistence[name]
            if persistent:
                content_nbytes = None
                content_sha256 = None
            else:
                try:
                    tensor_numel = operator.index(tensor.numel())
                    tensor_element_size = operator.index(tensor.element_size())
                except (AttributeError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"cannot size {role} non-persistent buffer {name}"
                    ) from exc
                if tensor_numel < 0 or tensor_element_size <= 0:
                    raise RuntimeError(
                        f"invalid size for {role} non-persistent buffer {name}"
                    )
                expected_content_nbytes = tensor_numel * tensor_element_size
                if expected_content_nbytes > (
                    MVROMA_NONPERSISTENT_BUFFER_MAX_BYTES
                    - nonpersistent_content_bytes
                ):
                    raise RuntimeError(
                        f"{role} non-persistent buffer content cap exceeded"
                    )
                try:
                    content = (
                        tensor.detach()
                        .cpu()
                        .contiguous()
                        .numpy()
                        .tobytes(order="C")
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"cannot encode {role} non-persistent buffer {name}"
                    ) from exc
                content_nbytes = len(content)
                if content_nbytes != expected_content_nbytes:
                    raise RuntimeError(
                        f"{role} non-persistent buffer byte count mismatch at {name}"
                    )
                nonpersistent_content_bytes += content_nbytes
                content_sha256 = hashlib.sha256(content).hexdigest()
            buffer_rows.append(
                {
                    "name": name,
                    "shape": [int(size) for size in tensor.shape],
                    "dtype": str(tensor.dtype),
                    "persistent": persistent,
                    "alias_group": alias_group,
                    "content_nbytes": content_nbytes,
                    "content_sha256": content_sha256,
                }
            )
        if len(buffer_rows) != len({row["name"] for row in buffer_rows}):
            raise RuntimeError(f"duplicate {role} buffer name")
        if set(buffer_persistence) != {row["name"] for row in buffer_rows}:
            raise RuntimeError(f"{role} named-buffer coverage mismatch")

        identity_models[role] = {
            "module_count": len(named_modules),
            "training_modules": training_modules,
            "parameters": parameter_rows,
            "buffers": buffer_rows,
            "nonpersistent_content_bytes": nonpersistent_content_bytes,
        }
        provenance_models[role] = {
            "parameter_devices": dict(sorted(parameter_devices.items())),
            "buffer_devices": dict(sorted(buffer_devices.items())),
        }
    payload = {
        "schema": "o101-model-state-identity/v2",
        "nonpersistent_content_encoding": "numpy-c-order-bytes/v1",
        "maximum_nonpersistent_content_bytes_per_model": (
            MVROMA_NONPERSISTENT_BUFFER_MAX_BYTES
        ),
        "models": identity_models,
    }
    _reject_mvroma_absolute_identity_paths(payload)
    return {
        "identity": {
            **payload,
            "sha256": hashlib.sha256(
                b"o101-model-state-identity-v2\0" + _canonical_json_bytes(payload)
            ).hexdigest(),
        },
        "provenance": {
            "expected_device": expected_device,
            "models": provenance_models,
        },
    }


_MVROMA_REQUIRED_MODULE_PATHS = {
    "src.build_model": "src/build_model.py",
    "src.mvroma": "src/mvroma/__init__.py",
    "src.run_model": "src/run_model.py",
    "src.matchers": "src/matchers/__init__.py",
    "src.matchers.run_matcher_path": "src/matchers/run_matcher_path.py",
    "src.matchers.uniflowmatch": "src/matchers/uniflowmatch/__init__.py",
    "src.matchers.uniflowmatch.models": "src/matchers/uniflowmatch/models/__init__.py",
    "src.matchers.uniflowmatch.models.ufm": "src/matchers/uniflowmatch/models/ufm.py",
    "src.matchers.uniflowmatch.models.base": "src/matchers/uniflowmatch/models/base.py",
    "src.mvroma.models.pipeline": "src/mvroma/models/pipeline.py",
    "src.mvroma.utils.grids": "src/mvroma/utils/grids.py",
    "src.track_cluster": "src/track_cluster.py",
    "uniflowmatch": "uniflowmatch/__init__.py",
    "uniflowmatch.models": "uniflowmatch/models/__init__.py",
    "uniflowmatch.models.base": "uniflowmatch/models/base.py",
    "uniflowmatch.models.ufm": "uniflowmatch/models/ufm.py",
    "uniflowmatch.models.unet_encoder": "uniflowmatch/models/unet_encoder.py",
    "uniflowmatch.models.utils": "uniflowmatch/models/utils.py",
    "uniflowmatch.utils": "uniflowmatch/utils/__init__.py",
    "uniflowmatch.utils.flow_resizing": "uniflowmatch/utils/flow_resizing.py",
    "uniflowmatch.utils.geometry": "uniflowmatch/utils/geometry.py",
    "uniception": "UniCeption/uniception/__init__.py",
    "uniception.models": "UniCeption/uniception/models/__init__.py",
    "uniception.models.encoders": "UniCeption/uniception/models/encoders/__init__.py",
    "uniception.models.encoders.base": "UniCeption/uniception/models/encoders/base.py",
    "uniception.models.encoders.dinov2": "UniCeption/uniception/models/encoders/dinov2.py",
    "uniception.models.encoders.image_normalizations": "UniCeption/uniception/models/encoders/image_normalizations.py",
    "uniception.models.info_sharing": "UniCeption/uniception/models/info_sharing/__init__.py",
    "uniception.models.info_sharing.global_attention_transformer": "UniCeption/uniception/models/info_sharing/global_attention_transformer.py",
    "uniception.models.prediction_heads": "UniCeption/uniception/models/prediction_heads/__init__.py",
    "uniception.models.prediction_heads.adaptors": "UniCeption/uniception/models/prediction_heads/adaptors.py",
    "uniception.models.prediction_heads.base": "UniCeption/uniception/models/prediction_heads/base.py",
    "uniception.models.prediction_heads.dpt": "UniCeption/uniception/models/prediction_heads/dpt.py",
    "uniception.models.prediction_heads.mlp_feature": "UniCeption/uniception/models/prediction_heads/mlp_feature.py",
    "uniception.models.prediction_heads.moge_conv": "UniCeption/uniception/models/prediction_heads/moge_conv.py",
    "uniception.models.utils.config": "UniCeption/uniception/models/utils/config.py",
    "uniception.models.utils.intermediate_feature_return": "UniCeption/uniception/models/utils/intermediate_feature_return.py",
    "uniception.models.utils.transformer_blocks": "UniCeption/uniception/models/utils/transformer_blocks.py",
}


_MVROMA_REQUIRED_DINOV2_MODULE_PATHS = {
    "dinov2": "dinov2/__init__.py",
    "dinov2.hub": "dinov2/hub/__init__.py",
    "dinov2.hub.backbones": "dinov2/hub/backbones.py",
    "dinov2.hub.utils": "dinov2/hub/utils.py",
    "dinov2.models": "dinov2/models/__init__.py",
    "dinov2.models.vision_transformer": "dinov2/models/vision_transformer.py",
    "dinov2.layers": "dinov2/layers/__init__.py",
    "dinov2.layers.attention": "dinov2/layers/attention.py",
    "dinov2.layers.block": "dinov2/layers/block.py",
    "dinov2.layers.drop_path": "dinov2/layers/drop_path.py",
    "dinov2.layers.layer_scale": "dinov2/layers/layer_scale.py",
    "dinov2.layers.mlp": "dinov2/layers/mlp.py",
    "dinov2.layers.patch_embed": "dinov2/layers/patch_embed.py",
    "dinov2.layers.swiglu_ffn": "dinov2/layers/swiglu_ffn.py",
}


_MVROMA_ALLOWED_DINOV2_NAMESPACE_SENTINELS = {
    "dinov2.hub.cell_dino": "dinov2/hub/cell_dino/backbones.py",
    "dinov2.hub.xray_dino": "dinov2/hub/xray_dino/backbones.py",
}


def collect_mvroma_module_identity(
    modules: dict[str, Any],
    *,
    mvroma_root: str | Path,
    ufm_root: str | Path,
    dinov2_root: str | Path | None = None,
    allowed_relative_paths: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    loaded_dinov2 = any(
        name == "dinov2" or name.startswith("dinov2.") for name in modules
    )
    if loaded_dinov2 and dinov2_root is None:
        raise RuntimeError("loaded DINOv2 modules require an attested DINOv2 root")
    required_paths = dict(_MVROMA_REQUIRED_MODULE_PATHS)
    if dinov2_root is not None:
        required_paths.update(_MVROMA_REQUIRED_DINOV2_MODULE_PATHS)
    missing = sorted(set(required_paths) - set(modules))
    if missing:
        raise RuntimeError(f"missing required MV-RoMa runtime modules: {missing}")
    mvroma_path = Path(mvroma_root).resolve(strict=True)
    ufm_path = Path(ufm_root).resolve(strict=True)
    dinov2_path = (
        None if dinov2_root is None else Path(dinov2_root).resolve(strict=True)
    )
    identity_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, str]] = []
    for name in sorted(modules):
        if name.startswith("src."):
            role, expected_root = "mvroma_vendored", mvroma_path
        elif name == "uniflowmatch" or name.startswith("uniflowmatch."):
            role, expected_root = "external_ufm", ufm_path
        elif name == "uniception" or name.startswith("uniception."):
            role, expected_root = "external_uniception", ufm_path
        elif name == "dinov2" or name.startswith("dinov2."):
            if dinov2_path is None:
                raise RuntimeError("loaded DINOv2 module has no attested DINOv2 root")
            role, expected_root = "dinov2", dinov2_path
        else:
            continue
        raw_path = getattr(modules[name], "__file__", None)
        if not raw_path:
            sentinel_relative = _MVROMA_ALLOWED_DINOV2_NAMESPACE_SENTINELS.get(name)
            if role != "dinov2" or sentinel_relative is None:
                raise RuntimeError(f"MV-RoMa module has no origin: {name}")
            namespace_relative = sentinel_relative.rsplit("/", 1)[0]
            namespace_path = (expected_root / namespace_relative).resolve(strict=True)
            loader = getattr(modules[name], "__loader__", None)
            loader_type = type(loader)
            if (
                loader_type.__module__ != "_frozen_importlib_external"
                or loader_type.__qualname__ != "_NamespaceLoader"
            ):
                raise RuntimeError(f"MV-RoMa namespace has wrong loader: {name}")
            spec = getattr(modules[name], "__spec__", None)
            search_locations = list(
                getattr(spec, "submodule_search_locations", None) or []
            )
            module_paths = list(getattr(modules[name], "__path__", []))
            expected_namespace = [str(namespace_path)]
            if (
                getattr(spec, "origin", None) is not None
                or [str(Path(path).resolve(strict=True)) for path in search_locations]
                != expected_namespace
                or [str(Path(path).resolve(strict=True)) for path in module_paths]
                != expected_namespace
            ):
                raise RuntimeError(f"MV-RoMa namespace origin mismatch: {name}")
            if (
                allowed_relative_paths is not None
                and sentinel_relative
                not in allowed_relative_paths.get("dinov2", set())
            ):
                raise RuntimeError(
                    "MV-RoMa namespace sentinel is outside frozen source manifest: "
                    f"{name} -> {sentinel_relative}"
                )
            sentinel = mvroma_file_content_identity(expected_root / sentinel_relative)
            identity_rows.append(
                {
                    "module": name,
                    "role": role,
                    "relative_path": namespace_relative,
                    "loader": "NamespaceLoader",
                    "namespace": True,
                    "sentinel": {
                        "relative_path": sentinel_relative,
                        **sentinel,
                    },
                }
            )
            provenance_rows.append({"module": name, "path": str(namespace_path)})
            continue
        module_path = Path(raw_path)
        if module_path.suffix in {".pyc", ".pyo"}:
            raise RuntimeError(f"MV-RoMa bytecode module origin is forbidden: {name}")
        resolved = module_path.resolve(strict=True)
        try:
            relative = resolved.relative_to(expected_root).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"MV-RoMa module origin is outside expected {role} root: {name} -> {resolved}"
            ) from exc
        if name.startswith("uniception.") and not relative.startswith(
            "UniCeption/uniception/"
        ):
            raise RuntimeError(f"MV-RoMa module origin has wrong UniCeption role: {name}")
        expected_relative = required_paths.get(name)
        if expected_relative is not None and relative != expected_relative:
            raise RuntimeError(
                f"MV-RoMa module has wrong relative origin: "
                f"{name} -> {relative}, expected {expected_relative}"
            )
        from importlib.machinery import SourceFileLoader

        if not isinstance(getattr(modules[name], "__loader__", None), SourceFileLoader):
            raise RuntimeError(f"MV-RoMa module does not use SourceFileLoader: {name}")
        if allowed_relative_paths is not None:
            allowed_role = (
                "external_ufm" if role == "external_uniception" else role
            )
            if relative not in allowed_relative_paths.get(allowed_role, set()):
                raise RuntimeError(
                    f"MV-RoMa module origin is outside frozen source manifest: "
                    f"{name} -> {relative}"
                )
        content = mvroma_file_content_identity(resolved)
        identity_rows.append(
            {
                "module": name,
                "role": role,
                "relative_path": relative,
                "loader": "SourceFileLoader",
                **content,
            }
        )
        provenance_rows.append({"module": name, "path": str(resolved)})
    identity_payload = {
        "schema": "mvroma-module-origins/v1",
        "modules": identity_rows,
    }
    return {
        "identity": {
            **identity_payload,
            "sha256": hashlib.sha256(
                b"mvroma-module-origins-v1\0" + _canonical_json_bytes(identity_payload)
            ).hexdigest(),
        },
        "provenance": provenance_rows,
    }


def resolve_local_ufm_snapshot(
    hf_hub_cache: str | Path,
    *,
    revision: str = MVROMA_UFM_REVISION,
    expected_files: dict[str, dict[str, Any]] = MVROMA_UFM_EXPECTED_FILES,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"invalid pinned UFM revision: {revision}")
    cache_root = Path(hf_hub_cache).resolve(strict=True)
    snapshot = (
        cache_root
        / "models--infinity1096--UFM-Refine"
        / "snapshots"
        / revision
    ).resolve(strict=True)
    if not snapshot.is_dir():
        raise RuntimeError(f"pinned UFM snapshot is not a directory: {snapshot}")
    actual: dict[str, dict[str, Any]] = {}
    for name, expected in sorted(expected_files.items()):
        identity = mvroma_file_content_identity(snapshot / name)
        actual[name] = identity
        if identity != expected:
            raise RuntimeError(
                f"pinned UFM {name} identity mismatch: {identity} != {expected}"
            )
    identity = {
        "repo_id": MVROMA_UFM_REPO_ID,
        "revision": revision,
        "files": actual,
    }
    return {
        "snapshot_path": str(snapshot),
        "revision": revision,
        "identity": identity,
    }


def _mvroma_runtime_versions() -> dict[str, str | None]:
    import importlib.metadata
    import platform

    def distribution(*names: str) -> str | None:
        for name in names:
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                continue
        return None

    return {
        "python_implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "torch": distribution("torch"),
        "cuda": None,
        "cudnn": None,
        "numpy": distribution("numpy"),
        "h5py": distribution("h5py"),
        "opencv": distribution("opencv-python", "opencv-python-headless"),
        "pillow": distribution("Pillow"),
        "torchvision": distribution("torchvision"),
        "transformers": distribution("transformers"),
        "huggingface_hub": distribution("huggingface-hub"),
        "safetensors": distribution("safetensors"),
        "timm": distribution("timm"),
        "einops": distribution("einops"),
        "scipy": distribution("scipy"),
    }


def _mvroma_cuda_gpu_identity(torch_module: Any, device: str) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        raise RuntimeError(f"CUDA device selected but CUDA is unavailable: {device}")
    parsed = torch_module.device(device)
    index = (
        int(parsed.index)
        if parsed.index is not None
        else int(torch_module.cuda.current_device())
    )
    properties = torch_module.cuda.get_device_properties(index)
    raw_uuid = str(getattr(properties, "uuid", ""))
    if not raw_uuid:
        raise RuntimeError(f"CUDA device UUID is unavailable: {device}")
    normalized_uuid = raw_uuid if raw_uuid.startswith("GPU-") else f"GPU-{raw_uuid}"
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    driver = None
    for line in query.stdout.splitlines():
        values = [value.strip() for value in line.split(",", 1)]
        if len(values) == 2 and values[0].lower() == normalized_uuid.lower():
            driver = values[1]
            break
    if not driver:
        raise RuntimeError(f"cannot resolve NVIDIA driver for CUDA UUID {normalized_uuid}")
    return {
        "selected_device": device,
        "name": str(properties.name),
        "uuid": normalized_uuid,
        "compute_capability": f"{int(properties.major)}.{int(properties.minor)}",
        "driver": driver,
    }


def _mvroma_optional_runtime_value(owner: Any, name: str) -> Any:
    if owner is None or not hasattr(owner, name):
        return None
    value = getattr(owner, name)
    return value() if callable(value) else value


def _mvroma_attention_backend_identity(modules: dict[str, Any]) -> dict[str, Any]:
    config = modules.get("uniception.models.utils.config")
    use_fused = (
        None
        if config is None or not callable(getattr(config, "use_fused_attn", None))
        else bool(config.use_fused_attn())
    )

    def module_flag(module_name: str, flag: str) -> bool | None:
        module = modules.get(module_name)
        value = None if module is None else getattr(module, flag, None)
        return None if value is None else bool(value)

    return {
        "uniception_has_fused_attn": (
            None if config is None else bool(getattr(config, "_HAS_FUSED_ATTN", False))
        ),
        "uniception_use_fused_attn_raw": (
            None if config is None else int(getattr(config, "_USE_FUSED_ATTN"))
        ),
        "uniception_use_fused_attn": use_fused,
        "mvroma_attention_xformers_available": module_flag(
            "src.mvroma.romatch.models.transformer.layers.attention",
            "XFORMERS_AVAILABLE",
        ),
        "mvroma_block_xformers_available": module_flag(
            "src.mvroma.romatch.models.transformer.layers.block",
            "XFORMERS_AVAILABLE",
        ),
        "dino_attention_xformers_enabled": module_flag(
            "dinov2.layers.attention", "XFORMERS_ENABLED"
        ),
        "dino_attention_xformers_available": module_flag(
            "dinov2.layers.attention", "XFORMERS_AVAILABLE"
        ),
        "dino_block_xformers_enabled": module_flag(
            "dinov2.layers.block", "XFORMERS_ENABLED"
        ),
        "dino_block_xformers_available": module_flag(
            "dinov2.layers.block", "XFORMERS_AVAILABLE"
        ),
        "dino_swiglu_xformers_enabled": module_flag(
            "dinov2.layers.swiglu_ffn", "XFORMERS_ENABLED"
        ),
        "dino_swiglu_xformers_available": module_flag(
            "dinov2.layers.swiglu_ffn", "XFORMERS_AVAILABLE"
        ),
    }


def probe_mvroma_effective_runtime(
    torch_module: Any,
    *,
    device: str,
    phase: str = "post_import_pre_model",
    versions: Any = _mvroma_runtime_versions,
    gpu_query: Any = None,
    environ: Any = os.environ,
    modules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if phase not in {"post_import_pre_model", "post_model_pre_publish"}:
        raise ValueError(f"invalid MV-RoMa runtime phase: {phase}")
    version_values = dict(versions())
    required_versions = {
        "python_implementation",
        "python",
        "torch",
        "cuda",
        "cudnn",
        "numpy",
        "h5py",
        "opencv",
        "pillow",
        "torchvision",
        "transformers",
        "huggingface_hub",
        "safetensors",
        "timm",
        "einops",
        "scipy",
    }
    if set(version_values) != required_versions:
        raise RuntimeError(
            "MV-RoMa runtime version keys mismatch: "
            f"{sorted(set(version_values) ^ required_versions)}"
        )
    version_values["torch"] = str(torch_module.__version__)
    version_values["cuda"] = getattr(torch_module.version, "cuda", None)
    cudnn_version = torch_module.backends.cudnn.version()
    version_values["cudnn"] = None if cudnn_version is None else str(cudnn_version)
    cuda_backend = torch_module.backends.cuda
    cuda_matmul = cuda_backend.matmul
    math_reduction = _mvroma_optional_runtime_value(
        cuda_backend, "fp16_bf16_reduction_math_sdp_allowed"
    )
    torch_flags = {
        "default_dtype": str(torch_module.get_default_dtype()),
        "cuda_matmul_allow_tf32": bool(cuda_matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch_module.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch_module.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch_module.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(
            torch_module.are_deterministic_algorithms_enabled()
        ),
        "float32_matmul_precision": str(
            torch_module.get_float32_matmul_precision()
        ),
        "flash_sdp_enabled": bool(cuda_backend.flash_sdp_enabled()),
        "mem_efficient_sdp_enabled": bool(
            cuda_backend.mem_efficient_sdp_enabled()
        ),
        "math_sdp_enabled": bool(cuda_backend.math_sdp_enabled()),
        "cudnn_sdp_enabled": bool(cuda_backend.cudnn_sdp_enabled()),
        "fp16_reduced_precision_reduction": bool(
            cuda_matmul.allow_fp16_reduced_precision_reduction
        ),
        "bf16_reduced_precision_reduction": bool(
            cuda_matmul.allow_bf16_reduced_precision_reduction
        ),
        "fp16_bf16_reduction_math_sdp_allowed": (
            None if math_reduction is None else bool(math_reduction)
        ),
        "cuda_matmul_allow_fp16_accumulation": (
            None
            if not hasattr(cuda_matmul, "allow_fp16_accumulation")
            else bool(cuda_matmul.allow_fp16_accumulation)
        ),
        "cudnn_benchmark_limit": (
            None
            if not hasattr(torch_module.backends.cudnn, "benchmark_limit")
            else int(torch_module.backends.cudnn.benchmark_limit)
        ),
    }
    if tuple(torch_flags) != MVROMA_TORCH_FLAG_KEYS:
        raise RuntimeError("MV-RoMa Torch flag schema drifted")
    if str(device).startswith("cuda"):
        gpu = (gpu_query or _mvroma_cuda_gpu_identity)(torch_module, str(device))
        if any(gpu.get(key) is None for key in ("name", "uuid", "compute_capability", "driver")):
            raise RuntimeError("CUDA MV-RoMa runtime identity is incomplete")
    else:
        gpu = {
            "selected_device": str(device),
            "name": None,
            "uuid": None,
            "compute_capability": None,
            "driver": None,
        }
    return {
        "identity": {
            "phase": phase,
            "versions": version_values,
            "torch_flags": torch_flags,
            "gpu": gpu,
            "environment": {
                key: environ.get(key) for key in MVROMA_ENVIRONMENT_KEYS
            },
            "attention_backend": _mvroma_attention_backend_identity(modules or {}),
        },
        "provenance": {
            "environment_paths": {
                key: environ.get(key) for key in MVROMA_PATH_ENVIRONMENT_KEYS
            }
        },
    }


def attest_mvroma_runtime_assets(
    paths: dict[str, str | Path],
    *,
    phase: str,
    expected: dict[str, Any] | None = None,
    require_frozen_dinov2: bool = False,
) -> dict[str, Any]:
    required_files = {
        "mvroma_checkpoint",
        "ufm_config",
        "ufm_weights",
        "dinov2_weights",
    }
    required = required_files | {"dinov2_source"}
    if set(paths) != required:
        raise RuntimeError(
            f"MV-RoMa asset path keys mismatch at {phase}: {sorted(set(paths) ^ required)}"
        )
    files = {
        key: mvroma_file_content_identity(paths[key]) for key in sorted(required_files)
    }
    dinov2_source = mvroma_tree_content_identity(
        paths["dinov2_source"], python_only=False
    )
    if require_frozen_dinov2:
        for key, value in MVROMA_DINOV2_SOURCE_EXPECTED.items():
            if dinov2_source.get(key) != value:
                raise RuntimeError(
                    f"dinov2_source {key} mismatch at {phase}: "
                    f"{dinov2_source.get(key)} != {value}"
                )
        if files["dinov2_weights"] != MVROMA_DINOV2_WEIGHTS_EXPECTED:
            raise RuntimeError(
                f"dinov2_weights identity mismatch at {phase}: "
                f"{files['dinov2_weights']} != {MVROMA_DINOV2_WEIGHTS_EXPECTED}"
            )
    result = {
        "schema": "mvroma-runtime-assets/v1",
        "files": files,
        "dinov2_source": dinov2_source,
    }
    if expected is not None:
        if set(expected.get("files", {})) != required_files:
            raise RuntimeError(f"invalid expected MV-RoMa asset schema at {phase}")
        for key in sorted(required_files):
            if files[key] != expected["files"][key]:
                raise RuntimeError(f"{key} identity changed at {phase}")
        if dinov2_source != expected.get("dinov2_source"):
            raise RuntimeError(f"dinov2_source identity changed at {phase}")
    return result


def _mvroma_open_file_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _mvroma_hash_open_file(file_obj: Any) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_obj.seek(0)
    size = 0
    while True:
        block = file_obj.read(4 * 1024 * 1024)
        if not block:
            break
        size += len(block)
        digest.update(block)
    file_obj.seek(0)
    return {"size": size, "sha256": digest.hexdigest()}


@contextmanager
def open_attested_mvroma_file(
    path: str | Path, *, expected: dict[str, Any] | None = None
) -> Iterable[Any]:
    """Keep one verified regular-file inode open through its consumer."""
    requested = Path(path)
    resolved = requested.resolve(strict=True)
    before_path = resolved.lstat()
    if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
        raise ValueError(f"MV-RoMa held asset must be a regular file: {requested}")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    if not nofollow or not nonblock:
        raise RuntimeError("O_NOFOLLOW and O_NONBLOCK are required for held assets")
    fd = os.open(
        resolved,
        os.O_RDONLY | nofollow | nonblock | getattr(os, "O_CLOEXEC", 0),
    )
    file_obj = os.fdopen(fd, "rb", closefd=True)
    try:
        opened = os.fstat(file_obj.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"MV-RoMa held asset must be a regular file: {requested}")
        if (opened.st_dev, opened.st_ino) != (before_path.st_dev, before_path.st_ino):
            raise RuntimeError(f"MV-RoMa held asset changed while opening: {requested}")
        identity = _mvroma_hash_open_file(file_obj)
        if identity["size"] != int(opened.st_size):
            raise RuntimeError(f"MV-RoMa held asset size changed while hashing: {requested}")
        if expected is not None and identity != expected:
            raise RuntimeError(
                f"MV-RoMa held asset identity mismatch: {requested}: "
                f"{identity} != {expected}"
            )
        proc_path = Path("/proc/self/fd") / str(file_obj.fileno())
        try:
            proc_stat = proc_path.stat()
        except OSError as exc:
            raise RuntimeError("/proc/self/fd is required for MV-RoMa held assets") from exc
        if (proc_stat.st_dev, proc_stat.st_ino) != (opened.st_dev, opened.st_ino):
            raise RuntimeError(f"MV-RoMa /proc fd alias changed: {requested}")
        handle = SimpleNamespace(
            file=file_obj,
            identity=identity,
            requested_path=requested,
            resolved_path=resolved,
            proc_path=proc_path,
        )
        try:
            yield handle
        finally:
            if file_obj.closed:
                raise RuntimeError(f"MV-RoMa consumer closed held asset: {requested}")
            after = os.fstat(file_obj.fileno())
            after_identity = _mvroma_hash_open_file(file_obj)
            if (
                _mvroma_open_file_signature(after)
                != _mvroma_open_file_signature(opened)
                or after_identity != identity
            ):
                raise RuntimeError(f"MV-RoMa held file changed while in use: {requested}")
            try:
                after_path = resolved.lstat()
                after_requested = requested.resolve(strict=True)
            except OSError as exc:
                raise RuntimeError(
                    f"MV-RoMa held asset pathname changed while in use: {requested}"
                ) from exc
            if (
                stat.S_ISLNK(after_path.st_mode)
                or _mvroma_open_file_signature(after_path)
                != _mvroma_open_file_signature(opened)
                or after_requested != resolved
            ):
                raise RuntimeError(
                    f"MV-RoMa held asset pathname changed while in use: {requested}"
                )
    finally:
        if not file_obj.closed:
            file_obj.close()


@contextmanager
def open_attested_mvroma_job_images(
    image_root: str | Path,
    job: dict[str, Any],
    expected_tree: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    source_index = int(job["source_index"])
    stack = ExitStack()
    try:
        if expected_tree.get("schema") != "mvroma-image-tree/v1":
            raise ValueError("invalid frozen MV-RoMa image tree schema")
        expected_files: dict[str, dict[str, Any]] = {}
        for row in expected_tree.get("files", []):
            if (
                not isinstance(row, list)
                or len(row) != 3
                or not isinstance(row[0], str)
                or not isinstance(row[1], int)
                or not isinstance(row[2], str)
            ):
                raise ValueError("invalid frozen MV-RoMa image row")
            if row[0] in expected_files:
                raise ValueError("duplicate frozen MV-RoMa image name")
            expected_files[row[0]] = {"size": row[1], "sha256": row[2]}
        root = Path(image_root).resolve(strict=True)
        names = [str(job["source"]), *[str(value) for value in job["targets"]]]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate image in MV-RoMa source {source_index}")
        missing = sorted(set(names) - set(expected_files))
        if missing:
            raise ValueError(
                f"missing frozen images for MV-RoMa source {source_index}: {missing}"
            )

        raw = stack.enter_context(
            tempfile.TemporaryDirectory(prefix=f".o101-images-{source_index:06d}-")
        )
        private = Path(raw)
        private.chmod(0o700)
        aliases: dict[str, str] = {}
        for name in names:
            source = _mvroma_safe_image_path(root, name)
            handle = stack.enter_context(
                open_attested_mvroma_file(source, expected=expected_files[name])
            )
            alias = private / name
            alias.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            current = alias.parent
            while current != private:
                current.chmod(0o700)
                current = current.parent
            alias.symlink_to(handle.proc_path)
            aliases[name] = str(alias)
        bound_job = dict(job)
        bound_job["bound_source_path"] = aliases[str(job["source"])]
        bound_job["bound_target_paths"] = [
            aliases[str(name)] for name in job["targets"]
        ]
    except BaseException as setup_error:
        try:
            stack.close()
        except BaseException as cleanup_error:
            _attach_mvroma_cleanup_error(setup_error, cleanup_error)
        if isinstance(setup_error, Exception):
            raise RuntimeError(
                f"MV-RoMa image guard failed for source {source_index}: "
                f"{setup_error}"
            ) from setup_error
        raise

    try:
        yield bound_job
    except BaseException as primary_error:
        try:
            stack.__exit__(
                type(primary_error), primary_error, primary_error.__traceback__
            )
        except BaseException as cleanup_error:
            _attach_mvroma_cleanup_error(primary_error, cleanup_error)
        raise
    else:
        try:
            stack.close()
        except BaseException as cleanup_error:
            if isinstance(cleanup_error, Exception):
                raise RuntimeError(
                    f"MV-RoMa image guard failed for source {source_index}: "
                    f"{cleanup_error}"
                ) from cleanup_error
            raise


def _attach_mvroma_cleanup_error(
    primary_error: BaseException, cleanup_error: BaseException
) -> None:
    def graph_nodes(root: BaseException) -> list[BaseException]:
        ordered: list[BaseException] = []
        pending = [root]
        visited: set[int] = set()
        while pending:
            error = pending.pop()
            identity = id(error)
            if identity in visited:
                continue
            visited.add(identity)
            ordered.append(error)
            if error.__cause__ is not None:
                pending.append(error.__cause__)
            if error.__context__ is not None:
                pending.append(error.__context__)
        return ordered

    def prune_cycles(
        root: BaseException, *, forbidden: set[int] | None = None
    ) -> None:
        blocked = forbidden or set()
        attributes = ("__cause__", "__context__")
        states: dict[int, int] = {id(root): 1}
        stack: list[tuple[BaseException, int]] = [(root, 0)]
        while stack:
            error, index = stack[-1]
            if index >= len(attributes):
                states[id(error)] = 2
                stack.pop()
                continue
            attribute = attributes[index]
            stack[-1] = (error, index + 1)
            linked = getattr(error, attribute)
            if linked is None:
                continue
            linked_identity = id(linked)
            linked_state = states.get(linked_identity, 0)
            if linked_identity in blocked or linked_state == 1:
                setattr(error, attribute, None)
                continue
            if linked_state == 0:
                states[linked_identity] = 1
                stack.append((linked, 0))

    def visible_chain(root: BaseException) -> list[BaseException]:
        result: list[BaseException] = []
        visited: set[int] = set()
        current: BaseException | None = root
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            result.append(current)
            if current.__cause__ is not None:
                current = current.__cause__
            elif current.__context__ is not None and not current.__suppress_context__:
                current = current.__context__
            else:
                current = None
        return result

    prune_cycles(primary_error)
    existing = graph_nodes(primary_error)
    existing_ids = {id(error) for error in existing}
    if id(cleanup_error) not in existing_ids:
        prune_cycles(cleanup_error, forbidden=existing_ids)

    visible = visible_chain(primary_error)
    if any(error is cleanup_error for error in visible):
        return
    tail = visible[-1]
    tail.__cause__ = cleanup_error
    tail.__suppress_context__ = True
    prune_cycles(primary_error)


@contextmanager
def _mvroma_context_preserving_primary(manager: Any) -> Iterable[Any]:
    value = manager.__enter__()
    try:
        yield value
    except BaseException as primary_error:
        try:
            manager.__exit__(
                type(primary_error), primary_error, primary_error.__traceback__
            )
        except BaseException as cleanup_error:
            _attach_mvroma_cleanup_error(primary_error, cleanup_error)
        raise
    else:
        manager.__exit__(None, None, None)


def _mvroma_key_diff_identity(
    missing: Any, unexpected: Any, *, schema: str
) -> dict[str, Any]:
    accepted = (list, tuple, set, frozenset)
    if not isinstance(missing, accepted) or not isinstance(unexpected, accepted):
        raise RuntimeError("strict=False model load did not return key differences")
    if not all(isinstance(value, str) for value in [*missing, *unexpected]):
        raise RuntimeError("strict=False model load returned a non-string key")
    if len(missing) != len(set(missing)) or len(unexpected) != len(set(unexpected)):
        raise RuntimeError("strict=False model load returned duplicate keys")
    return {
        "schema": schema,
        "strict": False,
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
    }


def _mvroma_state_load_identity(
    result: Any, *, schema: str
) -> dict[str, Any]:
    return _mvroma_key_diff_identity(
        getattr(result, "missing_keys", None),
        getattr(result, "unexpected_keys", None),
        schema=schema,
    )


def load_pinned_mvroma_model(
    runtime_objects: SimpleNamespace,
    *,
    checkpoint: str | Path,
    expected_checkpoint: dict[str, Any],
    device: str,
    expected_load_identity: dict[str, Any] | None = None,
) -> SimpleNamespace:
    args = runtime_objects.Namespace(
        use_dinov2=True,
        train_until_16x=False,
        train_refiner=False,
        train_all_model=False,
        num_cluster=512,
    )
    model_config = runtime_objects.ModelConfig()
    model_config.num_cluster = 512
    model, model_config = runtime_objects.build_our_model(
        args, model_config, use_dinov2=True
    )
    with open_attested_mvroma_file(
        checkpoint, expected=expected_checkpoint
    ) as checkpoint_handle:
        checkpoint_handle.file.seek(0)
        state = runtime_objects.torch.load(
            checkpoint_handle.file,
            map_location="cpu",
            weights_only=True,
        )
    load_result = model.load_state_dict(state, strict=False)
    load_identity = _mvroma_state_load_identity(
        load_result, schema="mvroma-state-load/v1"
    )
    if expected_load_identity is not None and load_identity != expected_load_identity:
        raise RuntimeError(
            "MV-RoMa state key identity mismatch: "
            f"{load_identity} != {expected_load_identity}"
        )
    evaluated = model.eval()
    if evaluated is not model:
        raise RuntimeError("MV-RoMa eval() replaced the model instance")
    moved = model.to(device)
    if moved is not model:
        raise RuntimeError("MV-RoMa to() replaced the model instance")
    return SimpleNamespace(
        model=model,
        model_config=model_config,
        load_identity=load_identity,
    )


@contextmanager
def pinned_dinov2_torch_hub(
    torch_module: Any,
    *,
    source_root: str | Path,
    checkpoint: str | Path,
    expected_checkpoint: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    source = Path(source_root).resolve(strict=True)
    weights = Path(checkpoint).resolve(strict=True)
    if not source.is_dir() or not weights.is_file():
        raise RuntimeError("pinned DINOv2 source or checkpoint is unavailable")
    expected_uri = weights.as_uri()
    original_load = torch_module.hub.load
    original_state_loader = torch_module.hub.load_state_dict_from_url
    guard_state: dict[str, Any] = {
        "hub_loads": 0,
        "state_loads": 0,
        "violation": None,
    }

    def reject(message: str) -> None:
        guard_state["violation"] = message
        raise RuntimeError(message)

    def guarded_load(
        repo_or_dir: str, entrypoint: str, *args: Any, **kwargs: Any
    ) -> Any:
        if guard_state["violation"]:
            raise RuntimeError(str(guard_state["violation"]))
        if args:
            reject("pinned DINOv2 repository load rejects positional arguments")
        if repo_or_dir != "facebookresearch/dinov2":
            reject(f"unexpected DINOv2 repository: {repo_or_dir}")
        if entrypoint != "dinov2_vitl14":
            reject(f"unexpected DINOv2 entrypoint: {entrypoint}")
        unknown = set(kwargs) - {"force_reload"}
        if unknown:
            reject(f"unexpected DINOv2 repository load options: {sorted(unknown)}")
        if kwargs.get("force_reload", False):
            reject("pinned DINOv2 repository forbids force_reload")
        preexisting = sorted(
            name for name in sys.modules if name == "dinov2" or name.startswith("dinov2.")
        )
        if preexisting:
            reject(f"preexisting dinov2 modules are forbidden: {preexisting}")
        if guard_state["hub_loads"] != 0:
            reject("pinned loader requires exactly one DINOv2 Hub load")
        guard_state["hub_loads"] += 1
        return original_load(
            str(source),
            "dinov2_vitl14",
            source="local",
            pretrained=True,
            weights=str(weights),
        )

    def guarded_state_load(url: str, *args: Any, **kwargs: Any) -> Any:
        if guard_state["violation"]:
            raise RuntimeError(str(guard_state["violation"]))
        if guard_state["state_loads"] != 0:
            reject("pinned loader requires exactly one DINOv2 state load")
        if args:
            reject("pinned DINOv2 state load rejects positional arguments")
        if str(url) != expected_uri:
            reject(f"unexpected DINOv2 state URI: {url}")
        unknown = set(kwargs) - {"map_location", "check_hash", "weights_only"}
        if unknown:
            reject(f"unexpected DINOv2 state load options: {sorted(unknown)}")
        map_location = kwargs.get("map_location", "cpu")
        if str(map_location) != "cpu":
            reject(f"pinned DINOv2 state load requires CPU map_location: {map_location}")
        if kwargs.get("check_hash", False):
            reject("pinned DINOv2 state load rejects basename hash inference")
        if kwargs.get("weights_only", True) is not True:
            reject("pinned DINOv2 state load requires weights_only=True")
        guard_state["state_loads"] += 1
        checkpoint_handle.file.seek(0)
        return torch_module.load(
            checkpoint_handle.file, map_location="cpu", weights_only=True
        )

    with open_attested_mvroma_file(
        weights, expected=expected_checkpoint
    ) as checkpoint_handle:
        torch_module.hub.load = guarded_load
        torch_module.hub.load_state_dict_from_url = guarded_state_load
        try:
            yield guard_state
        finally:
            torch_module.hub.load = original_load
            torch_module.hub.load_state_dict_from_url = original_state_loader


def load_pinned_ufm_model(
    ufm_class: Any,
    *,
    snapshot_path: str | Path,
    device: str,
    torch_module: Any,
    dinov2_source: str | Path,
    dinov2_weights: str | Path,
    expected_assets: dict[str, dict[str, Any]],
    pre_device_validator: Any = None,
) -> list[Any]:
    snapshot = Path(snapshot_path).resolve(strict=True)
    required_assets = {"ufm_config", "ufm_weights", "dinov2_weights"}
    if set(expected_assets) != required_assets:
        raise RuntimeError(
            "pinned UFM expected asset keys mismatch: "
            f"{sorted(set(expected_assets) ^ required_assets)}"
        )
    saved_environment = {
        key: os.environ.get(key)
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")
    }
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    config_path = snapshot / "config.json"
    weights_path = snapshot / "model.safetensors"
    try:
        with open_attested_mvroma_file(
            config_path, expected=expected_assets["ufm_config"]
        ) as config_handle:
            with open_attested_mvroma_file(
                weights_path, expected=expected_assets["ufm_weights"]
            ) as weights_handle:
                with tempfile.TemporaryDirectory(prefix=".o101-ufm-snapshot-") as raw:
                    private_snapshot = Path(raw)
                    private_snapshot.chmod(0o700)
                    (private_snapshot / "config.json").symlink_to(
                        config_handle.proc_path
                    )
                    (private_snapshot / "model.safetensors").symlink_to(
                        weights_handle.proc_path
                    )
                    with pinned_dinov2_torch_hub(
                        torch_module,
                        source_root=dinov2_source,
                        checkpoint=dinov2_weights,
                        expected_checkpoint=expected_assets["dinov2_weights"],
                    ) as guard_state:
                        model = ufm_class.from_pretrained(
                            str(private_snapshot),
                            local_files_only=True,
                            map_location="cpu",
                            strict=False,
                        )
                        if (
                            guard_state["violation"] is not None
                            or guard_state["hub_loads"] != 1
                            or guard_state["state_loads"] != 1
                        ):
                            raise RuntimeError(
                                "pinned loader requires exactly one DINOv2 Hub load "
                                "and exactly one DINOv2 state load"
                            )
    finally:
        for key, value in saved_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if pre_device_validator is not None:
        pre_device_validator(model)
    model.eval()
    model = model.to(device)
    return [None, model]


def load_pinned_ufm_model_with_identity(
    ufm_class: Any,
    *,
    snapshot_path: str | Path,
    device: str,
    torch_module: Any,
    dinov2_source: str | Path,
    dinov2_weights: str | Path,
    expected_assets: dict[str, dict[str, Any]],
    expected_load_identity: dict[str, Any] | None = None,
    safetensors_torch: Any = None,
) -> SimpleNamespace:
    if safetensors_torch is None:
        import safetensors.torch as safetensors_torch

    original_load_model = safetensors_torch.load_model
    observations: list[dict[str, Any]] = []
    load_identity: dict[str, Any] | None = None

    def tracked_load_model(
        model: Any,
        filename: str,
        *,
        strict: bool = True,
        device: Any = "cpu",
        **kwargs: Any,
    ) -> Any:
        if observations:
            raise RuntimeError("pinned UFM loader requires exactly one safetensors load")
        if strict is not False or str(device) != "cpu" or kwargs:
            raise RuntimeError(
                "pinned UFM safetensors load requires strict=False, device=cpu, "
                "and no extra options"
            )
        result = original_load_model(
            model,
            filename,
            strict=False,
            device="cpu",
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("pinned UFM safetensors load returned no key differences")
        observations.append(
            _mvroma_key_diff_identity(
                result[0],
                result[1],
                schema="ufm-safetensors-state-load/v1",
            )
        )
        return result

    def validate_before_device_move(model: Any) -> None:
        nonlocal load_identity
        if len(observations) != 1:
            raise RuntimeError("pinned UFM loader requires exactly one safetensors load")
        load_identity = observations[0]
        if expected_load_identity is not None and load_identity != expected_load_identity:
            raise RuntimeError(
                f"UFM state key identity mismatch: {load_identity} != {expected_load_identity}"
            )

    safetensors_torch.load_model = tracked_load_model
    try:
        prematch = load_pinned_ufm_model(
            ufm_class,
            snapshot_path=snapshot_path,
            device=device,
            torch_module=torch_module,
            dinov2_source=dinov2_source,
            dinov2_weights=dinov2_weights,
            expected_assets=expected_assets,
            pre_device_validator=validate_before_device_move,
        )
    finally:
        safetensors_torch.load_model = original_load_model
    if load_identity is None:
        raise RuntimeError("pinned UFM loader did not validate a safetensors load")
    return SimpleNamespace(prematch=prematch, load_identity=load_identity)


def _mvroma_contract_class_identity(value: type[Any]) -> dict[str, str]:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not module:
        raise RuntimeError("MV-RoMa contract class has no module identity")
    if not isinstance(qualname, str) or not qualname:
        raise RuntimeError("MV-RoMa contract class has no qualname identity")
    return {"module": module, "qualname": qualname}


def _mvroma_contract_object_identity(value: Any) -> dict[str, Any]:
    return {
        "type": _mvroma_contract_class_identity(type(value)),
        "mro": [
            _mvroma_contract_class_identity(parent)
            for parent in type(value).__mro__
        ],
    }


def collect_prepared_mvroma_post_model_identity(
    cfg: SimpleNamespace,
    prepared: SimpleNamespace,
    *,
    mvroma_loaded: SimpleNamespace,
    ufm_loaded: SimpleNamespace,
    post_model_assets: dict[str, Any],
) -> SimpleNamespace:
    runtime = prepared.runtime_objects
    if runtime.runner_ufm_class is not runtime.vendored_ufm_class:
        raise RuntimeError("MV-RoMa runner UFM class is not the vendored class")
    if (
        not isinstance(ufm_loaded.prematch, (list, tuple))
        or len(ufm_loaded.prematch) != 2
    ):
        raise RuntimeError("MV-RoMa UFM loader returned an invalid prematch bundle")
    prematch_model = ufm_loaded.prematch[1]
    if type(prematch_model) is not runtime.runner_ufm_class:
        raise RuntimeError("MV-RoMa constructed UFM type is not the runner class")
    encoder = getattr(prematch_model, "encoder", None)
    dinov2_model = getattr(encoder, "model", None)
    if dinov2_model is None:
        raise RuntimeError("MV-RoMa constructed UFM has no DINOv2 model")

    modules = {
        name: module
        for name, module in sys.modules.items()
        if _is_mvroma_target_module(name) and name != "src"
    }
    allowed_paths = {
        "mvroma_vendored": {
            str(row[0])
            for row in prepared.source_roots["identity"]["mvroma"]["files"]
        },
        "external_ufm": {
            str(row[0])
            for row in prepared.source_roots["identity"]["ufm"]["files"]
        },
        "dinov2": {
            str(row[0]) for row in prepared.dino_source_identity["files"]
        },
    }
    module_identity = collect_mvroma_module_identity(
        modules,
        mvroma_root=prepared.private_mvroma_root,
        ufm_root=prepared.private_ufm_root,
        dinov2_root=prepared.private_dinov2_root,
        allowed_relative_paths=allowed_paths,
    )

    fused_attention: list[dict[str, Any]] = []
    for name, module in prematch_model.named_modules():
        if hasattr(module, "fused_attn"):
            fused_value = module.fused_attn
            if not isinstance(fused_value, bool):
                raise RuntimeError(
                    f"MV-RoMa UFM fused_attn is not bool at module {name}"
                )
            fused_attention.append(
                {
                    "name": str(name),
                    "identity": _mvroma_contract_object_identity(module),
                    "fused_attn": fused_value,
                }
            )
    if not fused_attention:
        raise RuntimeError("MV-RoMa constructed UFM has no fused-attention identity")

    try:
        blocks = list(dinov2_model.blocks)
    except (AttributeError, TypeError) as exc:
        raise RuntimeError("MV-RoMa constructed DINOv2 has no block sequence") from exc
    dinov2_blocks = [
        {
            "index": index,
            "block": _mvroma_contract_object_identity(block),
            "attention": _mvroma_contract_object_identity(block.attn),
        }
        for index, block in enumerate(blocks)
    ]
    model_state = collect_mvroma_model_state_identity(
        {
            "mvroma": mvroma_loaded.model,
            "ufm": prematch_model,
        },
        expected_device=str(cfg.device),
    )
    runtime_probe = probe_mvroma_effective_runtime(
        runtime.torch,
        device=str(cfg.device),
        phase="post_model_pre_publish",
        modules=modules,
    )
    identity = {
        "schema": "o101-post-model-identity/v1",
        "mvroma": _mvroma_contract_object_identity(mvroma_loaded.model),
        "mvroma_state_load": mvroma_loaded.load_identity,
        "ufm_runner_class": _mvroma_contract_class_identity(
            runtime.runner_ufm_class
        ),
        "ufm_runner_is_vendored_class": True,
        "ufm": _mvroma_contract_object_identity(prematch_model),
        "ufm_state_load": ufm_loaded.load_identity,
        "encoder": _mvroma_contract_object_identity(encoder),
        "dinov2_model": _mvroma_contract_object_identity(dinov2_model),
        "fused_attention": fused_attention,
        "dinov2_blocks": dinov2_blocks,
        "module_identity": module_identity["identity"],
        "model_state_identity": model_state["identity"],
        "runtime_identity": runtime_probe["identity"],
        "post_model_assets": post_model_assets,
    }
    return SimpleNamespace(
        identity=identity,
        provenance={
            "module_paths": module_identity["provenance"],
            "model_state": model_state["provenance"],
            "runtime": runtime_probe["provenance"],
        },
    )


def mvroma_source_fingerprint(
    job: dict[str, Any], stage_contract_sha256: str, image_sha256: dict[str, str]
) -> str:
    names = [str(job["source"]), *[str(name) for name in job["targets"]]]
    missing = [name for name in names if name not in image_sha256]
    if missing:
        raise ValueError(f"missing image SHA-256 for {missing}")
    payload = {
        "schema": "mvroma-source-fingerprint/v1",
        "stage_contract_sha256": stage_contract_sha256,
        "source_index": int(job["source_index"]),
        "source": str(job["source"]),
        "targets": list(job["targets"]),
        "chunks": [list(chunk) for chunk in job["chunks"]],
        "images": [[name, image_sha256[name]] for name in names],
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


_MVROMA_RAW_DATASETS = ("keypoints0", "keypoints1", "scores")
_MVROMA_LEAF_DOMAIN = b"o101-dense-leaf-v1\0"
_MVROMA_ROOT_DOMAIN = b"o101-dense-raw-root-v1\0"


def _mvroma_update_raw_dataset(digest: Any, key: str, value: Any) -> None:
    import numpy as np

    array = np.ascontiguousarray(value)
    metadata = _canonical_json_bytes(
        {"key": key, "dtype": array.dtype.str, "shape": list(array.shape)}
    )
    digest.update(len(metadata).to_bytes(8, byteorder="little", signed=False))
    digest.update(metadata)
    digest.update(array.tobytes(order="C"))


def _mvroma_raw_leaf_digest(path: str, group: Any) -> str:
    digest = hashlib.sha256()
    digest.update(_MVROMA_LEAF_DOMAIN)
    digest.update(path.encode("utf-8"))
    digest.update(b"\0")
    for key in _MVROMA_RAW_DATASETS:
        _mvroma_update_raw_dataset(digest, key, group[key][...])
    return digest.hexdigest()


def mvroma_raw_semantic_digest(path: str | Path) -> str:
    import h5py

    rows: list[tuple[str, str]] = []
    with h5py.File(str(path), "r") as h5:
        leaf_paths: list[str] = []

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Group) and all(key in obj for key in _MVROMA_RAW_DATASETS):
                leaf_paths.append(name)

        h5.visititems(visit)
        for leaf_path in sorted(leaf_paths):
            rows.append((leaf_path, _mvroma_raw_leaf_digest(leaf_path, h5[leaf_path])))

    digest = hashlib.sha256()
    digest.update(_MVROMA_ROOT_DOMAIN)
    for leaf_path, leaf_digest in rows:
        digest.update(leaf_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(leaf_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


_MVROMA_SHARD_SCHEMA = "mvroma-source-shard/v1"
_MVROMA_MANIFEST_ATTR = "o101_manifest_json"


def build_mvroma_shard_metadata(
    job: dict[str, Any],
    stage_contract_sha256: str,
    image_sha256: dict[str, str],
    *,
    max_correspondences: int,
) -> dict[str, Any]:
    if max_correspondences < 16:
        raise ValueError(
            f"max_correspondences must be at least 16, got {max_correspondences}"
        )
    return {
        "schema": _MVROMA_SHARD_SCHEMA,
        "stage_contract_sha256": stage_contract_sha256,
        "source_fingerprint": mvroma_source_fingerprint(
            job, stage_contract_sha256, image_sha256
        ),
        "source_index": int(job["source_index"]),
        "source": str(job["source"]),
        "targets": list(job["targets"]),
        "chunks": [list(chunk) for chunk in job["chunks"]],
        "processed_target_count": len(job["targets"]),
        "max_correspondences": int(max_correspondences),
    }


def _validate_mvroma_raw_arrays(
    values: Any, max_correspondences: int | None = None
) -> tuple[Any, Any, Any]:
    import numpy as np

    if not isinstance(values, (tuple, list)) or len(values) != 3:
        raise ValueError("raw match value must be (keypoints0, keypoints1, scores)")
    keypoints0, keypoints1, scores = (np.asarray(value) for value in values)
    if keypoints0.dtype != np.dtype(np.float32):
        raise ValueError(f"keypoints0 dtype must be float32, got {keypoints0.dtype}")
    if keypoints1.dtype != np.dtype(np.float32):
        raise ValueError(f"keypoints1 dtype must be float32, got {keypoints1.dtype}")
    if scores.dtype != np.dtype(np.float32):
        raise ValueError(f"scores dtype must be float32, got {scores.dtype}")
    count = int(scores.shape[0]) if scores.ndim == 1 else -1
    if count < 16:
        raise ValueError(f"raw match count must be at least 16, got {count}")
    if max_correspondences is not None and count > max_correspondences:
        raise ValueError(
            f"raw match count exceeds maximum {max_correspondences}: {count}"
        )
    if keypoints0.shape != (count, 2) or keypoints1.shape != (count, 2):
        raise ValueError(
            f"raw match shape mismatch: {keypoints0.shape}, {keypoints1.shape}, {scores.shape}"
        )
    if not all(np.isfinite(value).all() for value in (keypoints0, keypoints1, scores)):
        raise ValueError("raw match values must be finite")
    return keypoints0, keypoints1, scores


def _mvroma_json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _mvroma_h5_object_token(obj: Any) -> int:
    import h5py

    return int(h5py.h5o.get_info(obj.id).addr)


def _mvroma_stream_raw_leaf_digest(
    leaf_path: str, group: Any, max_correspondences: int
) -> tuple[str, int]:
    import numpy as np

    datasets = {key: group[key] for key in _MVROMA_RAW_DATASETS}
    for key, dataset in datasets.items():
        if dataset.external:
            raise ValueError(f"external storage dataset at {leaf_path}/{key}")
        if dataset.is_virtual:
            raise ValueError(f"virtual dataset at {leaf_path}/{key}")
        if np.dtype(dataset.dtype) != np.dtype(np.float32):
            raise ValueError(
                f"{leaf_path}/{key} dtype must be float32, got {dataset.dtype}"
            )

    scores = datasets["scores"]
    count = int(scores.shape[0]) if scores.ndim == 1 else -1
    if count < 16:
        raise ValueError(f"raw match count must be at least 16, got {count}")
    if count > max_correspondences:
        raise ValueError(
            f"raw match count exceeds maximum {max_correspondences}: {count}"
        )
    if datasets["keypoints0"].shape != (count, 2) or datasets["keypoints1"].shape != (
        count,
        2,
    ):
        raise ValueError(f"raw match shape mismatch at {leaf_path}")

    digest = hashlib.sha256()
    digest.update(_MVROMA_LEAF_DOMAIN)
    digest.update(leaf_path.encode("utf-8"))
    digest.update(b"\0")
    rows_per_block = max(1, (1024 * 1024) // (2 * np.dtype(np.float32).itemsize))
    for key in _MVROMA_RAW_DATASETS:
        dataset = datasets[key]
        metadata = _canonical_json_bytes(
            {"key": key, "dtype": np.dtype(dataset.dtype).str, "shape": list(dataset.shape)}
        )
        digest.update(len(metadata).to_bytes(8, byteorder="little", signed=False))
        digest.update(metadata)
        for start in range(0, count, rows_per_block):
            block = np.ascontiguousarray(dataset[start : start + rows_per_block])
            if not np.isfinite(block).all():
                raise ValueError(f"non-finite raw value at {leaf_path}/{key}")
            digest.update(block.tobytes(order="C"))
    return digest.hexdigest(), count


def _strict_mvroma_shard_inspection(
    path: Path, expected_metadata: dict[str, Any], consumer: Any = None
) -> dict[str, Any]:
    import h5py

    expected_metadata_keys = {
        "schema",
        "stage_contract_sha256",
        "source_fingerprint",
        "source_index",
        "source",
        "targets",
        "chunks",
        "processed_target_count",
        "max_correspondences",
    }
    if set(expected_metadata) != expected_metadata_keys:
        raise ValueError(
            "expected shard metadata keys mismatch: "
            f"{sorted(set(expected_metadata) ^ expected_metadata_keys)}"
        )
    max_correspondences = int(expected_metadata["max_correspondences"])
    if max_correspondences < 16:
        raise ValueError("expected max_correspondences must be at least 16")

    try:
        before_path = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError("missing shard") from exc
    if stat.S_ISLNK(before_path.st_mode):
        raise ValueError("symlink shard is not allowed")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise RuntimeError("O_NOFOLLOW is required for shard validation")
    fd = os.open(path, os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0))
    file_obj = os.fdopen(fd, "rb", closefd=True)
    h5 = None
    opened = os.fstat(file_obj.fileno())
    if not stat.S_ISREG(opened.st_mode):
        file_obj.close()
        raise ValueError("shard must be a regular file")
    if (opened.st_dev, opened.st_ino) != (before_path.st_dev, before_path.st_ino):
        file_obj.close()
        raise ValueError("shard pathname changed while opening")

    expected_keys = set(expected_metadata) | {
        "complete",
        "produced_groups",
        "content_sha256",
    }
    raw_root_digest = hashlib.sha256()
    raw_root_digest.update(_MVROMA_ROOT_DOMAIN)
    raw_correspondences = 0
    leaf_digests: dict[str, str] = {}
    try:
        h5 = h5py.File(file_obj, "r")
        if set(h5.attrs) != {_MVROMA_MANIFEST_ATTR}:
            raise ValueError(f"unexpected root attributes: {sorted(h5.attrs)}")
        manifest_attr = h5.attrs.get_id(_MVROMA_MANIFEST_ATTR)
        if manifest_attr.get_storage_size() > 1024 * 1024:
            raise ValueError("shard manifest attribute is too large")
        raw_manifest = h5.attrs[_MVROMA_MANIFEST_ATTR]
        if isinstance(raw_manifest, bytes):
            raw_manifest = raw_manifest.decode("utf-8")
        manifest = json.loads(
            str(raw_manifest), object_pairs_hook=_mvroma_json_object_without_duplicates
        )
        if not isinstance(manifest, dict):
            raise ValueError("shard manifest must be a JSON object")
        if set(manifest) != expected_keys:
            raise ValueError(
                f"manifest keys mismatch: {sorted(set(manifest) ^ expected_keys)}"
            )
        for key, expected in expected_metadata.items():
            if manifest.get(key) != expected:
                raise ValueError(
                    f"metadata {key} mismatch: {manifest.get(key)!r} != {expected!r}"
                )
        if manifest.get("complete") is not True:
            raise ValueError("shard manifest is not complete")

        produced_groups = manifest.get("produced_groups")
        if (
            not isinstance(produced_groups, list)
            or not all(isinstance(value, str) for value in produced_groups)
            or produced_groups != sorted(produced_groups)
        ):
            raise ValueError("produced_groups must be a sorted list")
        if len(produced_groups) != len(set(produced_groups)):
            raise ValueError("duplicate produced group")
        allowed_groups = {
            pair_name(str(expected_metadata["source"]), str(target))
            for target in expected_metadata["targets"]
        }
        if not set(produced_groups).issubset(allowed_groups):
            raise ValueError("produced group is outside the frozen source plan")

        expected_groups: set[str] = set()
        expected_datasets: set[str] = set()
        for leaf_path in produced_groups:
            parts = leaf_path.split("/")
            expected_groups.update(
                "/".join(parts[:index]) for index in range(1, len(parts) + 1)
            )
            expected_datasets.update(
                f"{leaf_path}/{key}" for key in _MVROMA_RAW_DATASETS
            )
        expected_objects = expected_groups | expected_datasets
        actual_groups: set[str] = set()
        actual_datasets: set[str] = set()
        seen_objects = {_mvroma_h5_object_token(h5)}
        maximum_objects = len(expected_objects) + 1

        def walk(group: Any, prefix: str = "") -> None:
            if prefix and group.attrs:
                raise ValueError(f"unexpected attributes on group {prefix}")
            for name in group:
                child_path = f"{prefix}/{name}" if prefix else name
                if child_path not in expected_objects:
                    raise ValueError(f"unexpected HDF5 object at {child_path}")
                link = group.get(name, getlink=True)
                if not isinstance(link, h5py.HardLink):
                    raise ValueError(f"non-hard link at {child_path}")
                child = group.get(name)
                token = _mvroma_h5_object_token(child)
                if token in seen_objects:
                    raise ValueError(f"hard-link alias or cycle at {child_path}")
                seen_objects.add(token)
                if len(seen_objects) > maximum_objects:
                    raise ValueError("HDF5 object count exceeds the frozen schema")
                if isinstance(child, h5py.Group):
                    actual_groups.add(child_path)
                    walk(child, child_path)
                elif isinstance(child, h5py.Dataset):
                    if child.attrs:
                        raise ValueError(f"unexpected attributes on dataset {child_path}")
                    actual_datasets.add(child_path)
                else:
                    raise ValueError(f"unsupported HDF5 object at {child_path}")

        walk(h5)
        if actual_groups != expected_groups:
            raise ValueError(
                f"group set mismatch: missing={sorted(expected_groups - actual_groups)} "
                f"extra={sorted(actual_groups - expected_groups)}"
            )
        if actual_datasets != expected_datasets:
            raise ValueError(
                f"dataset set mismatch: missing={sorted(expected_datasets - actual_datasets)} "
                f"extra={sorted(actual_datasets - expected_datasets)}"
            )
        for leaf_path in produced_groups:
            leaf_digest, count = _mvroma_stream_raw_leaf_digest(
                leaf_path, h5[leaf_path], max_correspondences
            )
            leaf_digests[leaf_path] = leaf_digest
            raw_correspondences += count
            raw_root_digest.update(leaf_path.encode("utf-8"))
            raw_root_digest.update(b"\0")
            raw_root_digest.update(leaf_digest.encode("ascii"))
            raw_root_digest.update(b"\n")

        content_sha256 = raw_root_digest.hexdigest()
        if manifest.get("content_sha256") != content_sha256:
            raise ValueError("raw content digest mismatch")
        inspection = {
            "valid": True,
            "reason": "",
            "path": str(path),
            "processed_target_count": int(manifest["processed_target_count"]),
            "produced_groups": list(manifest["produced_groups"]),
            "content_sha256": content_sha256,
            "source_fingerprint": str(manifest["source_fingerprint"]),
            "raw_correspondences": raw_correspondences,
            "leaf_digests": leaf_digests,
            "file_identity": {
                "device": int(opened.st_dev),
                "inode": int(opened.st_ino),
                "size": int(opened.st_size),
                "mtime_ns": int(opened.st_mtime_ns),
            },
        }
        if consumer is not None:
            consumer(h5, inspection)
        after_fd = os.fstat(file_obj.fileno())
        if (after_fd.st_size, after_fd.st_mtime_ns) != (
            opened.st_size,
            opened.st_mtime_ns,
        ):
            raise ValueError("shard changed during validation")
    finally:
        if h5 is not None:
            h5.close()
        file_obj.close()

    after_path = path.lstat()
    if stat.S_ISLNK(after_path.st_mode) or (
        after_path.st_dev,
        after_path.st_ino,
        after_path.st_size,
        after_path.st_mtime_ns,
    ) != (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
    ):
        raise ValueError("shard pathname changed during validation")
    return inspection


def validate_mvroma_source_shard(
    path: str | Path, expected_metadata: dict[str, Any]
) -> dict[str, Any]:
    shard_path = Path(path)
    try:
        return _strict_mvroma_shard_inspection(shard_path, expected_metadata)
    except Exception as exc:
        return {"valid": False, "reason": str(exc), "path": str(shard_path)}


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _mvroma_private_candidate_path(destination: Path) -> Iterable[Path]:
    work_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-", dir=str(destination.parent)
        )
    )
    candidate: Path | None = None
    try:
        work_dir.chmod(0o700)
        work_stat = work_dir.lstat()
        if (
            not stat.S_ISDIR(work_stat.st_mode)
            or stat.S_IMODE(work_stat.st_mode) != 0o700
            or work_stat.st_uid != os.geteuid()
        ):
            raise RuntimeError(f"MV-RoMa temp directory is not private: {work_dir}")
        fd, temp_name = tempfile.mkstemp(prefix="candidate-", dir=str(work_dir))
        os.close(fd)
        candidate = Path(temp_name)
        yield candidate
    finally:
        if candidate is not None:
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
            except IsADirectoryError:
                try:
                    candidate.rmdir()
                except OSError:
                    pass
        try:
            work_dir.rmdir()
        except OSError:
            pass


def _mvroma_committed_identity(
    path: Path, expected: dict[str, int]
) -> None:
    expected_tuple = (
        int(expected["device"]),
        int(expected["inode"]),
        int(expected["size"]),
        int(expected["mtime_ns"]),
    )
    try:
        committed = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError("committed candidate identity is missing") from exc
    committed_tuple = (
        int(committed.st_dev),
        int(committed.st_ino),
        int(committed.st_size),
        int(committed.st_mtime_ns),
    )
    if stat.S_ISREG(committed.st_mode) and committed_tuple == expected_tuple:
        return

    # Never leave a known-wrong replacement at the public destination.
    try:
        current = path.lstat()
        current_tuple = (
            int(current.st_dev),
            int(current.st_ino),
            int(current.st_size),
            int(current.st_mtime_ns),
        )
        if current_tuple == committed_tuple and not stat.S_ISDIR(current.st_mode):
            path.unlink()
            _fsync_directory(path.parent)
    except FileNotFoundError:
        pass
    raise ValueError(
        "committed candidate identity differs from the prevalidated temp file"
    )


def _unlink_mvroma_exact_commit(
    path: Path, expected: dict[str, int]
) -> bool:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return False
    if (
        not stat.S_ISREG(current.st_mode)
        or (int(current.st_dev), int(current.st_ino))
        != (int(expected["device"]), int(expected["inode"]))
    ):
        return False
    path.unlink()
    _fsync_directory(path.parent)
    return True


def _guard_mvroma_exact_commit(
    path: Path,
    expected: dict[str, int],
    commit_guard: Any,
) -> None:
    if commit_guard is None:
        return
    try:
        commit_guard()
    except BaseException as guard_error:
        try:
            _unlink_mvroma_exact_commit(path, expected)
        except BaseException as cleanup_error:
            _attach_mvroma_cleanup_error(guard_error, cleanup_error)
        raise


def publish_mvroma_source_shard_atomic(
    path: str | Path,
    metadata: dict[str, Any],
    matches_by_target: dict[str, Any],
) -> dict[str, Any]:
    import h5py

    shard_path = Path(path)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    targets = list(metadata["targets"])
    extra_targets = sorted(set(matches_by_target) - set(targets))
    if extra_targets:
        raise ValueError(f"matches contain unplanned targets: {extra_targets}")

    with _mvroma_private_candidate_path(shard_path) as temp_path:
        produced_groups: list[str] = []
        with h5py.File(str(temp_path), "w") as h5:
            for target in targets:
                if target not in matches_by_target:
                    continue
                keypoints0, keypoints1, scores = _validate_mvroma_raw_arrays(
                    matches_by_target[target], int(metadata["max_correspondences"])
                )
                leaf_path = pair_name(str(metadata["source"]), target)
                group = h5.create_group(leaf_path)
                group.create_dataset("keypoints0", data=keypoints0)
                group.create_dataset("keypoints1", data=keypoints1)
                group.create_dataset("scores", data=scores)
                produced_groups.append(leaf_path)

        manifest = {
            **metadata,
            "complete": True,
            "produced_groups": sorted(produced_groups),
            "content_sha256": mvroma_raw_semantic_digest(temp_path),
        }
        with h5py.File(str(temp_path), "a") as h5:
            h5.attrs[_MVROMA_MANIFEST_ATTR] = _canonical_json_bytes(manifest).decode("utf-8")

        _fsync_file(temp_path)
        inspection = validate_mvroma_source_shard(temp_path, metadata)
        if not inspection["valid"]:
            raise ValueError(f"candidate shard validation failed: {inspection['reason']}")
        candidate_stat = temp_path.lstat()
        candidate_identity = inspection["file_identity"]
        if stat.S_ISLNK(candidate_stat.st_mode) or (
            candidate_stat.st_dev,
            candidate_stat.st_ino,
            candidate_stat.st_size,
            candidate_stat.st_mtime_ns,
        ) != (
            candidate_identity["device"],
            candidate_identity["inode"],
            candidate_identity["size"],
            candidate_identity["mtime_ns"],
        ):
            raise ValueError("candidate shard changed after validation")
        os.replace(temp_path, shard_path)
        _mvroma_committed_identity(shard_path, candidate_identity)
        _fsync_directory(shard_path.parent)
        inspection["path"] = str(shard_path)
        return inspection


def _strict_mvroma_final_inspection(
    path: Path,
    expected_leaf_digests: dict[str, str],
    max_correspondences: int,
) -> dict[str, Any]:
    import h5py

    before_path = path.lstat()
    if stat.S_ISLNK(before_path.st_mode):
        raise ValueError("symlink final is not allowed")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise RuntimeError("O_NOFOLLOW is required for final validation")
    fd = os.open(path, os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0))
    file_obj = os.fdopen(fd, "rb", closefd=True)
    h5 = None
    opened = os.fstat(file_obj.fileno())
    if not stat.S_ISREG(opened.st_mode):
        file_obj.close()
        raise ValueError("final must be a regular file")
    if (opened.st_dev, opened.st_ino) != (before_path.st_dev, before_path.st_ino):
        file_obj.close()
        raise ValueError("final pathname changed while opening")

    leaf_paths = sorted(expected_leaf_digests)
    expected_groups: set[str] = set()
    expected_datasets: set[str] = set()
    for leaf_path in leaf_paths:
        parts = leaf_path.split("/")
        expected_groups.update(
            "/".join(parts[:index]) for index in range(1, len(parts) + 1)
        )
        expected_datasets.update(
            f"{leaf_path}/{key}" for key in _MVROMA_RAW_DATASETS
        )
    expected_objects = expected_groups | expected_datasets
    root_digest = hashlib.sha256()
    root_digest.update(_MVROMA_ROOT_DOMAIN)
    raw_correspondences = 0
    try:
        h5 = h5py.File(file_obj, "r")
        if h5.attrs:
            raise ValueError(f"unexpected final root attributes: {sorted(h5.attrs)}")
        actual_groups: set[str] = set()
        actual_datasets: set[str] = set()
        seen_objects = {_mvroma_h5_object_token(h5)}

        def walk(group: Any, prefix: str = "") -> None:
            if prefix and group.attrs:
                raise ValueError(f"unexpected attributes on final group {prefix}")
            for name in group:
                child_path = f"{prefix}/{name}" if prefix else name
                if child_path not in expected_objects:
                    raise ValueError(f"unexpected final HDF5 object at {child_path}")
                link = group.get(name, getlink=True)
                if not isinstance(link, h5py.HardLink):
                    raise ValueError(f"non-hard final link at {child_path}")
                child = group.get(name)
                token = _mvroma_h5_object_token(child)
                if token in seen_objects:
                    raise ValueError(f"final hard-link alias or cycle at {child_path}")
                seen_objects.add(token)
                if isinstance(child, h5py.Group):
                    actual_groups.add(child_path)
                    walk(child, child_path)
                elif isinstance(child, h5py.Dataset):
                    if child.attrs:
                        raise ValueError(
                            f"unexpected attributes on final dataset {child_path}"
                        )
                    actual_datasets.add(child_path)
                else:
                    raise ValueError(f"unsupported final HDF5 object at {child_path}")

        walk(h5)
        if actual_groups != expected_groups:
            raise ValueError("final group set mismatch")
        if actual_datasets != expected_datasets:
            raise ValueError("final dataset set mismatch")
        for leaf_path in leaf_paths:
            leaf_digest, count = _mvroma_stream_raw_leaf_digest(
                leaf_path, h5[leaf_path], max_correspondences
            )
            if leaf_digest != expected_leaf_digests[leaf_path]:
                raise ValueError(f"final leaf digest mismatch at {leaf_path}")
            raw_correspondences += count
            root_digest.update(leaf_path.encode("utf-8"))
            root_digest.update(b"\0")
            root_digest.update(leaf_digest.encode("ascii"))
            root_digest.update(b"\n")
        after_fd = os.fstat(file_obj.fileno())
        if (after_fd.st_size, after_fd.st_mtime_ns) != (
            opened.st_size,
            opened.st_mtime_ns,
        ):
            raise ValueError("final changed during validation")
    finally:
        if h5 is not None:
            h5.close()
        file_obj.close()

    after_path = path.lstat()
    if stat.S_ISLNK(after_path.st_mode) or (
        after_path.st_dev,
        after_path.st_ino,
        after_path.st_size,
        after_path.st_mtime_ns,
    ) != (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
    ):
        raise ValueError("final pathname changed during validation")
    return {
        "valid": True,
        "content_sha256": root_digest.hexdigest(),
        "raw_correspondences": raw_correspondences,
        "file_identity": {
            "device": int(opened.st_dev),
            "inode": int(opened.st_ino),
            "size": int(opened.st_size),
            "mtime_ns": int(opened.st_mtime_ns),
        },
    }


def _mvroma_canonical_path(path: str | Path) -> str:
    try:
        return os.path.normcase(str(Path(path).resolve(strict=False)))
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot resolve MV-RoMa path {path}: {exc}") from exc


def _mvroma_existing_inode(path: str | Path) -> tuple[int, int] | None:
    try:
        value = Path(path).stat()
    except FileNotFoundError:
        return None
    return int(value.st_dev), int(value.st_ino)


def merge_mvroma_shards_atomic(
    final_path: str | Path,
    planned_shards: Iterable[tuple[str | Path, dict[str, Any]]],
    *,
    expected_source_count: int,
    commit_guard: Any = None,
) -> dict[str, Any]:
    import h5py

    output_path = Path(final_path)
    entries = [(Path(path), metadata) for path, metadata in planned_shards]
    canonical_paths = [_mvroma_canonical_path(path) for path, _metadata in entries]
    if len(canonical_paths) != len(set(canonical_paths)):
        raise ValueError("duplicate planned shard path")
    shard_inodes = [
        inode
        for path, _metadata in entries
        if (inode := _mvroma_existing_inode(path)) is not None
    ]
    if len(shard_inodes) != len(set(shard_inodes)):
        raise ValueError("duplicate planned shard inode")
    output_canonical = _mvroma_canonical_path(output_path)
    output_inode = _mvroma_existing_inode(output_path)
    if output_canonical in set(canonical_paths):
        raise ValueError("final path collides with planned shard path")
    if output_inode is not None and output_inode in set(shard_inodes):
        raise ValueError("final inode collides with planned shard inode")

    inspections: list[dict[str, Any]] = []
    expected_leaf_digests: dict[str, str] = {}
    stage_contracts = {
        str(metadata["stage_contract_sha256"]) for _path, metadata in entries
    }
    maxima = {int(metadata["max_correspondences"]) for _path, metadata in entries}
    source_fingerprints = [
        str(metadata["source_fingerprint"]) for _path, metadata in entries
    ]
    if len(source_fingerprints) != len(set(source_fingerprints)):
        raise ValueError("duplicate source fingerprint")
    source_indices = [int(metadata["source_index"]) for _path, metadata in entries]
    if expected_source_count < 0 or len(entries) != expected_source_count:
        raise ValueError("planned shards do not contain the complete source plan")
    if source_indices != list(range(expected_source_count)):
        raise ValueError("planned shards do not match the complete source plan")
    if len(stage_contracts) > 1:
        raise ValueError("planned shards have different stage contracts")
    if len(maxima) > 1:
        raise ValueError("planned shards have different correspondence maxima")
    max_correspondences = next(iter(maxima), 16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _mvroma_private_candidate_path(output_path) as temp_path:
        with h5py.File(str(temp_path), "w") as output_h5:
            for shard_path, metadata in entries:
                def copy_validated_shard(
                    shard_h5: Any, inspection: dict[str, Any]
                ) -> None:
                    for leaf_path, leaf_digest in inspection["leaf_digests"].items():
                        if leaf_path in expected_leaf_digests:
                            raise ValueError(f"duplicate produced group: {leaf_path}")
                        expected_leaf_digests[leaf_path] = leaf_digest
                    for leaf_path in inspection["produced_groups"]:
                        values = tuple(
                            shard_h5[leaf_path][key][...] for key in _MVROMA_RAW_DATASETS
                        )
                        keypoints0, keypoints1, scores = _validate_mvroma_raw_arrays(
                            values, max_correspondences
                        )
                        group = output_h5.create_group(leaf_path)
                        group.create_dataset("keypoints0", data=keypoints0)
                        group.create_dataset("keypoints1", data=keypoints1)
                        group.create_dataset("scores", data=scores)

                try:
                    inspection = _strict_mvroma_shard_inspection(
                        shard_path, metadata, consumer=copy_validated_shard
                    )
                except Exception as exc:
                    raise ValueError(
                        f"invalid planned shard {shard_path}: {exc}"
                    ) from exc
                inspections.append(inspection)

        _fsync_file(temp_path)
        final_inspection = _strict_mvroma_final_inspection(
            temp_path, expected_leaf_digests, max_correspondences
        )
        candidate = temp_path.lstat()
        identity = final_inspection["file_identity"]
        if stat.S_ISLNK(candidate.st_mode) or (
            candidate.st_dev,
            candidate.st_ino,
            candidate.st_size,
            candidate.st_mtime_ns,
        ) != (
            identity["device"],
            identity["inode"],
            identity["size"],
            identity["mtime_ns"],
        ):
            raise ValueError("candidate final changed after validation")
        if commit_guard is not None:
            commit_guard()
        os.replace(temp_path, output_path)
        _mvroma_committed_identity(output_path, identity)
        _guard_mvroma_exact_commit(output_path, identity, commit_guard)
        _fsync_directory(output_path.parent)
        _guard_mvroma_exact_commit(output_path, identity, commit_guard)
        return {
            **final_inspection,
            "path": str(output_path),
            "shards": len(inspections),
            "groups": len(expected_leaf_digests),
        }


@contextmanager
def open_attested_mvroma_shard_root(
    shard_dir: str | Path,
) -> Iterable[tuple[Path, Any]]:
    shard_root = Path(shard_dir)
    try:
        initial_shard_root = shard_root.lstat()
    except FileNotFoundError:
        initial_shard_root = None
    if initial_shard_root is not None and stat.S_ISLNK(initial_shard_root.st_mode):
        raise ValueError("MV-RoMa shard root must not be a symlink")
    if initial_shard_root is not None and not stat.S_ISDIR(
        initial_shard_root.st_mode
    ):
        raise ValueError("MV-RoMa shard root is not a directory")
    shard_root.mkdir(parents=True, exist_ok=True)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory_flag:
        raise RuntimeError("O_NOFOLLOW and O_DIRECTORY are required for shards")
    try:
        fd = os.open(
            shard_root,
            os.O_RDONLY
            | nofollow
            | directory_flag
            | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as exc:
        raise ValueError(f"cannot open MV-RoMa shard root: {exc}") from exc
    path_guard = None
    primary_error: BaseException | None = None
    primary_traceback = None
    try:
        opened = os.fstat(fd)
        current = shard_root.lstat()
        identity = (int(opened.st_dev), int(opened.st_ino))
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or (int(current.st_dev), int(current.st_ino)) != identity
        ):
            raise RuntimeError("MV-RoMa shard root changed while opening")

        def guard_path() -> None:
            try:
                value = shard_root.lstat()
            except FileNotFoundError as exc:
                raise RuntimeError("MV-RoMa shard root changed: missing") from exc
            if (
                stat.S_ISLNK(value.st_mode)
                or not stat.S_ISDIR(value.st_mode)
                or (int(value.st_dev), int(value.st_ino)) != identity
            ):
                raise RuntimeError("MV-RoMa shard root changed during execution")

        path_guard = guard_path
        yield Path(f"/proc/self/fd/{fd}"), path_guard
    except BaseException as exc:
        primary_error = exc
        primary_traceback = exc.__traceback__
    finally:
        cleanup_error: BaseException | None = None
        if path_guard is not None:
            try:
                path_guard()
            except BaseException as exc:
                cleanup_error = exc
        try:
            os.close(fd)
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
            else:
                exc.__context__ = None
                cleanup_error.__cause__ = exc
        if primary_error is not None:
            if cleanup_error is not None:
                _attach_mvroma_cleanup_error(primary_error, cleanup_error)
            raise primary_error.with_traceback(primary_traceback)
        if cleanup_error is not None:
            raise cleanup_error


def _execute_mvroma_resume_held_root(
    job_list: list[dict[str, Any]],
    shard_root: Path,
    final_path: str | Path,
    stage_contract_sha256: str,
    image_sha256: dict[str, str],
    *,
    max_correspondences: int,
    mvroma_resume: bool,
    overwrite: bool,
    runner_factory: Any,
    after_runner_factory: Any,
    job_context_factory: Any,
    before_merge: Any,
    shard_root_guard: Any,
) -> dict[str, Any]:
    planned_paths: list[Path] = []
    for job in job_list:
        shard_name = str(job["shard_name"])
        if Path(shard_name).name != shard_name:
            raise ValueError(f"invalid shard filename: {shard_name}")
        planned_paths.append(shard_root / shard_name)
    held_root_canonical = Path(_mvroma_canonical_path(shard_root))
    final_canonical = Path(_mvroma_canonical_path(final_path))
    planned_canonical = [_mvroma_canonical_path(path) for path in planned_paths]
    if len(planned_canonical) != len(set(planned_canonical)):
        raise ValueError("duplicate planned shard path")
    final_inode = _mvroma_existing_inode(final_path)
    planned_inode_list = [
        inode
        for path in planned_paths
        if (inode := _mvroma_existing_inode(path)) is not None
    ]
    if len(planned_inode_list) != len(set(planned_inode_list)):
        raise ValueError("duplicate planned shard inode")
    planned_inodes = set(planned_inode_list)
    if str(final_canonical) in set(planned_canonical) or (
        final_inode is not None and final_inode in planned_inodes
    ):
        raise ValueError("final path collides with planned shard path")
    try:
        final_canonical.relative_to(held_root_canonical)
    except ValueError:
        pass
    else:
        raise ValueError("MV-RoMa final path is inside shard root")

    entries: list[tuple[Path, dict[str, Any]]] = []
    pending: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []
    invalid_reasons: dict[str, str] = {}
    reuse_allowed = bool(mvroma_resume) and not bool(overwrite)
    reused_sources = 0
    for job, shard_path in zip(job_list, planned_paths):
        shard_name = str(job["shard_name"])
        metadata = build_mvroma_shard_metadata(
            job,
            stage_contract_sha256,
            image_sha256,
            max_correspondences=max_correspondences,
        )
        entries.append((shard_path, metadata))
        if reuse_allowed:
            inspection = validate_mvroma_source_shard(shard_path, metadata)
            if inspection["valid"]:
                reused_sources += 1
                continue
            invalid_reasons[shard_name] = str(inspection["reason"])
        pending.append((job, shard_path, metadata))

    model_builds = 0
    recomputed_sources = 0
    if pending:
        runner = runner_factory()
        model_builds = 1
        try:
            if not callable(runner):
                raise TypeError("MV-RoMa runner factory must return a callable")
            if after_runner_factory is not None:
                after_runner_factory()
            for job, shard_path, metadata in pending:
                if job_context_factory is None:
                    matches_by_target = runner(job)
                else:
                    with _mvroma_context_preserving_primary(
                        job_context_factory(job)
                    ) as bound_job:
                        matches_by_target = runner(bound_job)
                if not isinstance(matches_by_target, dict):
                    raise TypeError(
                        "MV-RoMa source runner must return a target mapping"
                    )
                publish_mvroma_source_shard_atomic(
                    shard_path, metadata, matches_by_target
                )
                recomputed_sources += 1
        finally:
            runner = None

    shard_root_guard()
    if before_merge is not None:
        before_merge()
    shard_root_guard()
    merged = merge_mvroma_shards_atomic(
        final_path,
        entries,
        expected_source_count=len(job_list),
        commit_guard=shard_root_guard,
    )
    _guard_mvroma_exact_commit(
        Path(final_path), merged["file_identity"], shard_root_guard
    )
    return {
        **merged,
        "planned_sources": len(job_list),
        "reused_sources": reused_sources,
        "recomputed_sources": recomputed_sources,
        "model_builds": model_builds,
        "invalid_reasons": invalid_reasons,
    }


def execute_mvroma_resume(
    jobs: Iterable[dict[str, Any]],
    shard_dir: str | Path,
    final_path: str | Path,
    stage_contract_sha256: str,
    image_sha256: dict[str, str],
    *,
    max_correspondences: int,
    mvroma_resume: bool,
    overwrite: bool,
    runner_factory: Any,
    after_runner_factory: Any = None,
    job_context_factory: Any = None,
    before_merge: Any = None,
) -> dict[str, Any]:
    job_list = list(jobs)
    source_indices = [int(job["source_index"]) for job in job_list]
    if source_indices != list(range(len(job_list))):
        raise ValueError("MV-RoMa jobs do not match the complete source plan")
    shard_names: list[str] = []
    for job in job_list:
        shard_name = str(job["shard_name"])
        if Path(shard_name).name != shard_name:
            raise ValueError(f"invalid shard filename: {shard_name}")
        shard_names.append(shard_name)
    if len(shard_names) != len(set(shard_names)):
        raise ValueError("duplicate planned shard path")
    result: dict[str, Any] | None = None
    try:
        with open_attested_mvroma_shard_root(shard_dir) as (
            held_root,
            shard_root_guard,
        ):
            result = _execute_mvroma_resume_held_root(
                job_list,
                held_root,
                final_path,
                stage_contract_sha256,
                image_sha256,
                max_correspondences=max_correspondences,
                mvroma_resume=mvroma_resume,
                overwrite=overwrite,
                runner_factory=runner_factory,
                after_runner_factory=after_runner_factory,
                job_context_factory=job_context_factory,
                before_merge=before_merge,
                shard_root_guard=shard_root_guard,
            )
    except BaseException as execution_error:
        if result is not None and isinstance(result.get("file_identity"), dict):
            try:
                _unlink_mvroma_exact_commit(
                    Path(final_path), result["file_identity"]
                )
            except BaseException as cleanup_error:
                _attach_mvroma_cleanup_error(execution_error, cleanup_error)
        raise
    if result is None:
        raise RuntimeError("MV-RoMa resume completed without a result")
    return result


def execute_attested_mvroma_resume(
    jobs: Iterable[dict[str, Any]],
    shard_dir: str | Path,
    final_path: str | Path,
    stage_contract_sha256: str,
    image_root: str | Path,
    image_tree: dict[str, Any],
    asset_paths: dict[str, str | Path],
    initial_assets: dict[str, Any],
    *,
    max_correspondences: int,
    mvroma_resume: bool,
    overwrite: bool,
    runner_loader: Any,
    full_attestor: Any = attest_mvroma_runtime_assets,
    job_image_guard: Any = open_attested_mvroma_job_images,
    global_image_attestor: Any = build_mvroma_image_sha256_tree,
    pre_merge_runtime_finalizer: Any = None,
    post_model_verifier: Any = None,
) -> dict[str, Any]:
    job_list = list(jobs)
    if pre_merge_runtime_finalizer is not None and not callable(
        pre_merge_runtime_finalizer
    ):
        raise TypeError("MV-RoMa pre-merge runtime finalizer must be callable")
    if post_model_verifier is not None and not callable(post_model_verifier):
        raise TypeError("MV-RoMa post-model verifier must be callable")
    image_tree_snapshot = snapshot_mvroma_image_sha256_tree(image_tree, job_list)
    image_sha256 = image_tree_snapshot["by_name"]

    def after_runner_factory() -> None:
        post_model_assets = full_attestor(
            asset_paths,
            phase="post_model_pre_publish",
            expected=initial_assets,
        )
        if post_model_verifier is not None:
            post_model_verifier(post_model_assets)

    def job_context_factory(job: dict[str, Any]) -> Any:
        return job_image_guard(image_root, job, image_tree_snapshot)

    def before_merge() -> None:
        full_attestor(asset_paths, phase="pre_merge", expected=initial_assets)
        current_images = global_image_attestor(Path(image_root), job_list)
        if current_images != image_tree_snapshot:
            raise RuntimeError("MV-RoMa global image tree changed at pre_merge")
        if pre_merge_runtime_finalizer is not None:
            pre_merge_runtime_finalizer()

    return execute_mvroma_resume(
        job_list,
        shard_dir,
        final_path,
        stage_contract_sha256,
        image_sha256,
        max_correspondences=max_correspondences,
        mvroma_resume=mvroma_resume,
        overwrite=overwrite,
        runner_factory=runner_loader,
        after_runner_factory=after_runner_factory,
        job_context_factory=job_context_factory,
        before_merge=before_merge,
    )


def mvroma_lock_path(dense_matches: str | Path) -> Path:
    return Path(f"{dense_matches}.lock")


_MVROMA_KERNEL_STAGE_LOCK_NAME = b"\0sfm_system.mvroma_stage.v1"
_MVROMA_ACTIVE_KERNEL_STAGE_LOCKS: set[Any] = set()
_MVROMA_ACTIVE_STAGE_LOCK_FDS: set[int] = set()
_MVROMA_STAGE_LOCK_REGISTRY_GUARD = threading.RLock()
_MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS: set[object] = set()
_MVROMA_FORK_CHILD_FAIL_CLOSED_EXIT_CODE = getattr(os, "EX_SOFTWARE", 70)
_MVROMA_FORK_PREPARE_STATE = threading.local()
_MVROMA_FORK_PREPARE_FAILED = 0
_MVROMA_FORK_PREPARE_ACQUIRED = 1
_MVROMA_FORK_PREPARE_PREOWNED = 2


def _begin_mvroma_stage_lock_resource_transition() -> object:
    transition = object()
    with _MVROMA_STAGE_LOCK_REGISTRY_GUARD:
        _MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS.add(transition)
    return transition


def _finish_mvroma_stage_lock_resource_transition(transition: object) -> None:
    with _MVROMA_STAGE_LOCK_REGISTRY_GUARD:
        _MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS.discard(transition)


def _close_registered_mvroma_stage_fd(fd: int) -> None:
    transition = _begin_mvroma_stage_lock_resource_transition()
    try:
        os.close(fd)
    finally:
        with _MVROMA_STAGE_LOCK_REGISTRY_GUARD:
            # Linux releases the descriptor number before reporting late close
            # errors. Retrying a retained integer could close an unrelated fd.
            _MVROMA_ACTIVE_STAGE_LOCK_FDS.discard(fd)
            _MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS.discard(transition)


def _close_registered_mvroma_kernel_lock(kernel_lock: Any) -> None:
    transition = _begin_mvroma_stage_lock_resource_transition()
    try:
        kernel_lock.close()
    finally:
        with _MVROMA_STAGE_LOCK_REGISTRY_GUARD:
            _MVROMA_ACTIVE_KERNEL_STAGE_LOCKS.discard(kernel_lock)
            _MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS.discard(transition)


def _close_inherited_mvroma_stage_locks() -> bool:
    with _MVROMA_STAGE_LOCK_REGISTRY_GUARD:
        inherited_sockets = tuple(_MVROMA_ACTIVE_KERNEL_STAGE_LOCKS)
        inherited_fds = tuple(_MVROMA_ACTIVE_STAGE_LOCK_FDS)
        _MVROMA_ACTIVE_KERNEL_STAGE_LOCKS.clear()
        _MVROMA_ACTIVE_STAGE_LOCK_FDS.clear()
        transition = object()
        _MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS.add(transition)
    cleanup_failed = False
    try:
        for kernel_lock in inherited_sockets:
            try:
                kernel_lock.close()
            except BaseException:
                cleanup_failed = True
        for fd in inherited_fds:
            try:
                os.close(fd)
            except BaseException:
                cleanup_failed = True
    finally:
        _finish_mvroma_stage_lock_resource_transition(transition)
    return cleanup_failed


def _mvroma_fork_prepare_stack() -> list[int]:
    stack = getattr(_MVROMA_FORK_PREPARE_STATE, "stack", None)
    if stack is None:
        stack = []
        _MVROMA_FORK_PREPARE_STATE.stack = stack
    return stack


def _pop_mvroma_fork_prepare_state() -> int:
    try:
        stack = _mvroma_fork_prepare_stack()
        return int(stack.pop()) if stack else _MVROMA_FORK_PREPARE_FAILED
    except BaseException:
        return _MVROMA_FORK_PREPARE_FAILED


def _before_mvroma_stage_lock_fork() -> None:
    try:
        stack = _mvroma_fork_prepare_stack()
        stack.append(_MVROMA_FORK_PREPARE_FAILED)
    except BaseException:
        return
    try:
        if _MVROMA_STAGE_LOCK_REGISTRY_GUARD._is_owned():
            stack[-1] = _MVROMA_FORK_PREPARE_PREOWNED
            return
    except BaseException:
        return
    try:
        _MVROMA_STAGE_LOCK_REGISTRY_GUARD.acquire()
    except BaseException:
        try:
            if _MVROMA_STAGE_LOCK_REGISTRY_GUARD._is_owned():
                try:
                    _MVROMA_STAGE_LOCK_REGISTRY_GUARD.release()
                except BaseException:
                    if _MVROMA_STAGE_LOCK_REGISTRY_GUARD._is_owned():
                        stack[-1] = _MVROMA_FORK_PREPARE_ACQUIRED
        except BaseException:
            pass
        return
    stack[-1] = _MVROMA_FORK_PREPARE_ACQUIRED


def _after_mvroma_stage_lock_fork_parent() -> None:
    prepare_state = _pop_mvroma_fork_prepare_state()
    if prepare_state != _MVROMA_FORK_PREPARE_ACQUIRED:
        return
    try:
        _MVROMA_STAGE_LOCK_REGISTRY_GUARD.release()
    except BaseException:
        pass


def _after_mvroma_stage_lock_fork_child() -> None:
    prepare_state = _pop_mvroma_fork_prepare_state()
    if prepare_state == _MVROMA_FORK_PREPARE_FAILED:
        os._exit(_MVROMA_FORK_CHILD_FAIL_CLOSED_EXIT_CODE)
    if _MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS:
        os._exit(_MVROMA_FORK_CHILD_FAIL_CLOSED_EXIT_CODE)
    cleanup_failed = True
    try:
        cleanup_failed = _close_inherited_mvroma_stage_locks()
    except BaseException:
        cleanup_failed = True
    if prepare_state == _MVROMA_FORK_PREPARE_ACQUIRED:
        try:
            _MVROMA_STAGE_LOCK_REGISTRY_GUARD.release()
        except BaseException:
            cleanup_failed = True
    if cleanup_failed:
        os._exit(_MVROMA_FORK_CHILD_FAIL_CLOSED_EXIT_CODE)


_MVROMA_AT_FORK_LOCK_CLEANUP = hasattr(os, "register_at_fork")
if _MVROMA_AT_FORK_LOCK_CLEANUP:
    os.register_at_fork(
        before=_before_mvroma_stage_lock_fork,
        after_in_parent=_after_mvroma_stage_lock_fork_parent,
        after_in_child=_after_mvroma_stage_lock_fork_child,
    )


def _acquire_mvroma_kernel_stage_lock(lock_path: Path) -> Any:
    import errno
    import socket

    if not sys.platform.startswith("linux") or not _MVROMA_AT_FORK_LOCK_CLEANUP:
        raise RuntimeError("Linux abstract sockets are required for the MV-RoMa lock")
    transition = _begin_mvroma_stage_lock_resource_transition()
    try:
        kernel_lock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    except BaseException:
        _finish_mvroma_stage_lock_resource_transition(transition)
        raise
    try:
        with _MVROMA_STAGE_LOCK_REGISTRY_GUARD:
            _MVROMA_ACTIVE_KERNEL_STAGE_LOCKS.add(kernel_lock)
            _MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS.discard(transition)
    except BaseException as primary_error:
        try:
            kernel_lock.close()
        except BaseException as cleanup_error:
            _attach_mvroma_cleanup_error(primary_error, cleanup_error)
        finally:
            with _MVROMA_STAGE_LOCK_REGISTRY_GUARD:
                _MVROMA_ACTIVE_KERNEL_STAGE_LOCKS.discard(kernel_lock)
                _MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS.discard(transition)
        raise
    try:
        kernel_lock.set_inheritable(False)
        kernel_lock.bind(_MVROMA_KERNEL_STAGE_LOCK_NAME)
    except BaseException as exc:
        try:
            _close_registered_mvroma_kernel_lock(kernel_lock)
        except BaseException as cleanup_error:
            _attach_mvroma_cleanup_error(exc, cleanup_error)
        if isinstance(exc, OSError) and exc.errno == errno.EADDRINUSE:
            raise RuntimeError(
                f"MV-RoMa stage is already locked: {lock_path}"
            ) from exc
        if isinstance(exc, OSError):
            raise RuntimeError(
                f"cannot acquire MV-RoMa kernel stage lock: {exc}"
            ) from exc
        raise
    return kernel_lock


def _mvroma_guard_lock_path(lock_path: Path) -> Path:
    return lock_path.with_name(f".{lock_path.name}.guard")


def _open_mvroma_lock_inode(path: Path) -> tuple[int, tuple[int, int]]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise RuntimeError("O_NOFOLLOW is required for the MV-RoMa lock")
    transition = _begin_mvroma_stage_lock_resource_transition()
    try:
        try:
            fd = os.open(
                path,
                os.O_RDWR
                | os.O_CREAT
                | nofollow
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
        except BaseException:
            _finish_mvroma_stage_lock_resource_transition(transition)
            raise
        try:
            with _MVROMA_STAGE_LOCK_REGISTRY_GUARD:
                _MVROMA_ACTIVE_STAGE_LOCK_FDS.add(fd)
                _MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS.discard(transition)
        except BaseException as primary_error:
            try:
                os.close(fd)
            except BaseException as cleanup_error:
                _attach_mvroma_cleanup_error(primary_error, cleanup_error)
            finally:
                with _MVROMA_STAGE_LOCK_REGISTRY_GUARD:
                    _MVROMA_ACTIVE_STAGE_LOCK_FDS.discard(fd)
                    _MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS.discard(transition)
            raise
    except OSError as exc:
        raise RuntimeError(f"cannot open MV-RoMa lock inode {path}: {exc}") from exc
    try:
        opened = os.fstat(fd)
    except BaseException as primary_error:
        try:
            _close_registered_mvroma_stage_fd(fd)
        except BaseException as cleanup_error:
            _attach_mvroma_cleanup_error(primary_error, cleanup_error)
        raise
    if not stat.S_ISREG(opened.st_mode):
        primary_error = RuntimeError(f"MV-RoMa lock is not a regular file: {path}")
        try:
            _close_registered_mvroma_stage_fd(fd)
        except BaseException as cleanup_error:
            _attach_mvroma_cleanup_error(primary_error, cleanup_error)
        raise primary_error
    return fd, (int(opened.st_dev), int(opened.st_ino))


def _validate_mvroma_lock_inode(
    path: Path, identity: tuple[int, int]
) -> None:
    try:
        value = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"MV-RoMa lock path changed: missing {path}") from exc
    if (
        stat.S_ISLNK(value.st_mode)
        or not stat.S_ISREG(value.st_mode)
        or (int(value.st_dev), int(value.st_ino)) != identity
    ):
        raise RuntimeError(f"MV-RoMa lock path changed: {path}")


@contextmanager
def mvroma_stage_lock(lock_path: str | Path) -> Iterable[None]:
    import fcntl

    path = Path(lock_path)
    kernel_lock = _acquire_mvroma_kernel_stage_lock(path)
    owner_pid = os.getpid()
    locked: list[tuple[Path, int, tuple[int, int]]] = []
    primary_error: BaseException | None = None
    primary_traceback = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        guard_path = _mvroma_guard_lock_path(path)
        for current_path in (guard_path, path):
            fd, identity = _open_mvroma_lock_inode(current_path)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException as exc:
                try:
                    _close_registered_mvroma_stage_fd(fd)
                except BaseException as cleanup_error:
                    _attach_mvroma_cleanup_error(exc, cleanup_error)
                if isinstance(exc, BlockingIOError):
                    raise RuntimeError(
                        f"MV-RoMa stage is already locked: {path}"
                    ) from exc
                raise
            locked.append((current_path, fd, identity))
            _validate_mvroma_lock_inode(current_path, identity)

        yield
        if os.getpid() != owner_pid:
            raise RuntimeError(
                "MV-RoMa stage cannot continue in a fork child"
            )
    except BaseException as exc:
        primary_error = exc
        primary_traceback = exc.__traceback__

    cleanup_error: BaseException | None = None

    def remember_cleanup_error(error: BaseException) -> None:
        nonlocal cleanup_error
        if cleanup_error is None:
            cleanup_error = error
            return
        _attach_mvroma_cleanup_error(cleanup_error, error)

    if os.getpid() == owner_pid:
        for current_path, _fd, identity in locked:
            try:
                _validate_mvroma_lock_inode(current_path, identity)
            except BaseException as exc:
                remember_cleanup_error(exc)
        for _current_path, fd, _identity in reversed(locked):
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except BaseException as exc:
                remember_cleanup_error(exc)
            try:
                _close_registered_mvroma_stage_fd(fd)
            except BaseException as exc:
                remember_cleanup_error(exc)
        try:
            _close_registered_mvroma_kernel_lock(kernel_lock)
        except BaseException as exc:
            remember_cleanup_error(exc)

    if primary_error is not None:
        if cleanup_error is not None:
            _attach_mvroma_cleanup_error(primary_error, cleanup_error)
        raise primary_error.with_traceback(primary_traceback)
    if cleanup_error is not None:
        raise cleanup_error


def read_pairs(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] != parts[1]:
            pairs.append((parts[0], parts[1]))
    return pairs


def write_pairs(path: Path, pairs: Iterable[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [f"{a} {b}" for a, b in pairs]
    path.write_text(("\n".join(rows) + "\n") if rows else "", encoding="utf-8")


def parse_grid_spec(value: str | None, default_h: int, default_w: int) -> tuple[int, int]:
    if not value:
        return default_h, default_w
    text = str(value).lower().replace(" ", "")
    if "x" not in text:
        raise ValueError(f"grid must be like 8x12, got {value!r}")
    a, b = text.split("x", 1)
    return max(1, int(a)), max(1, int(b))


def infer_sequence_direction(folder: str) -> str:
    text = folder.strip()
    lower = text.lower().replace("_", "-")
    compact = re.sub(r"[^a-z0-9]+", "", lower)
    if lower.startswith("a-b") or compact.startswith("ab") or "forward" in lower or "正向" in text:
        return "forward"
    if lower.startswith("b-a") or compact.startswith("ba") or "reverse" in lower or "backward" in lower or "反向" in text:
        return "reverse"
    return "unknown"


def load_direction_overrides(cfg: SimpleNamespace) -> dict[str, str]:
    override_path = getattr(cfg, "direction_overrides_json", "") or ""
    candidates = [Path(override_path)] if override_path else []
    candidates.append(cfg_paths(cfg).work / "sequence_directions.json")
    for path in candidates:
        if path and path.exists():
            data = read_json(path)
            if "directions" in data and isinstance(data["directions"], dict):
                data = data["directions"]
            return {str(k): str(v).lower() for k, v in data.items()}
    return {}


def sequence_directions(cfg: SimpleNamespace, groups: dict[str, list[str]]) -> dict[str, str]:
    overrides = load_direction_overrides(cfg)
    directions: dict[str, str] = {}
    for folder in groups:
        value = overrides.get(folder, infer_sequence_direction(folder))
        if value in {"fwd", "forward", "a-b", "ab", "正向"}:
            value = "forward"
        elif value in {"rev", "reverse", "backward", "b-a", "ba", "反向"}:
            value = "reverse"
        else:
            value = "unknown"
        directions[folder] = value
    return directions


def connected_components(nodes: Iterable[str], pairs: Iterable[tuple[str, str]]) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {n: set() for n in nodes}
    for a, b in pairs:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    seen: set[str] = set()
    components: list[list[str]] = []
    for node in sorted(adjacency):
        if node in seen:
            continue
        stack = [node]
        comp: list[str] = []
        seen.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in sorted(adjacency.get(cur, ())):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(sorted(comp))
    return components


def write_pair_graph_reports(
    cfg: SimpleNamespace,
    pairs: list[tuple[str, str]],
    uncapped_pairs: list[tuple[str, str]],
    groups: dict[str, list[str]],
    directions: dict[str, str],
    pair_meta: dict[tuple[str, str], dict[str, Any]],
) -> None:
    p = cfg_paths(cfg)
    names = [name for rel in groups.values() for name in rel]
    folder_of = {name: folder for folder, rel in groups.items() for name in rel}
    roles = load_motion_roles(cfg)
    relation_counts: Counter[str] = Counter()
    motion_pair_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    for a, b in pairs:
        rel = pair_relation(a, b, folder_of, directions)
        if rel["cross_direction"]:
            relation_counts["cross_direction"] += 1
        elif rel["cross_video"]:
            relation_counts["cross_video"] += 1
        elif rel["same_video"]:
            relation_counts["same_video"] += 1
        ca = normalize_motion_class(str(roles.get(a, {}).get("motion_class", "parallax")))
        cb = normalize_motion_class(str(roles.get(b, {}).get("motion_class", "parallax")))
        motion_pair_counts["+".join(sorted((ca, cb)))] += 1
        pair = tuple(sorted((a, b)))
        kind_counts[str(pair_meta.get(pair, {}).get("kind", "unknown"))] += 1
    parallax_pairs, filter_stats = filter_pairs_by_motion_roles(pairs, roles, exclude_non_parallax=True)
    all_components = connected_components(names, pairs)
    parallax_nodes = [n for n in names if is_parallax_role(roles.get(n))]
    parallax_components = connected_components(parallax_nodes, parallax_pairs)
    write_json(p.pair_graph_diagnostics, {
        "total_frames": len(names),
        "pairs": len(pairs),
        "uncapped_pairs": len(uncapped_pairs),
        "pair_kinds": dict(kind_counts),
        "relations": dict(relation_counts),
        "motion_pair_classes": dict(motion_pair_counts),
        "connected_components": len(all_components),
        "largest_component": max((len(c) for c in all_components), default=0),
        "parallax_components_without_bridges": len(parallax_components),
        "largest_parallax_component_without_bridges": max((len(c) for c in parallax_components), default=0),
        "parallax_pair_filter": filter_stats,
    })

    parallax_component_id: dict[str, int] = {}
    for cid, comp in enumerate(parallax_components):
        for name in comp:
            parallax_component_id[name] = cid
    bridge_frames: list[dict[str, Any]] = []
    for name in names:
        role = roles.get(name, {})
        if normalize_motion_class(str(role.get("motion_class", ""))) != "pure_rotation":
            continue
        linked_components: set[int] = set()
        linked_frames: list[str] = []
        cross_direction_edges = 0
        for a, b in pairs:
            other = ""
            if a == name:
                other = b
            elif b == name:
                other = a
            if not other:
                continue
            if other in parallax_component_id:
                linked_components.add(parallax_component_id[other])
                linked_frames.append(other)
            if pair_relation(name, other, folder_of, directions).get("cross_direction"):
                cross_direction_edges += 1
        bridge_frames.append({
            "frame": name,
            "linked_parallax_components": sorted(linked_components),
            "linked_parallax_component_count": len(linked_components),
            "linked_parallax_frames": sorted(linked_frames)[:20],
            "cross_direction_edges": cross_direction_edges,
            "required_bridge": len(linked_components) >= 2 or cross_direction_edges > 0,
        })
    write_json(p.rotation_bridge_report, {
        "pure_rotation_frames": len(bridge_frames),
        "required_bridge_frames": sum(1 for item in bridge_frames if item["required_bridge"]),
        "parallax_components_without_bridges": len(parallax_components),
        "all_components_with_bridges": len(all_components),
        "frames": bridge_frames,
    })


def pair_relation(a: str, b: str, folder_of: dict[str, str], directions: dict[str, str]) -> dict[str, Any]:
    fa, fb = folder_of.get(a, ""), folder_of.get(b, "")
    da, db = directions.get(fa, "unknown"), directions.get(fb, "unknown")
    known_opposite = da in {"forward", "reverse"} and db in {"forward", "reverse"} and da != db
    return {
        "folder0": fa,
        "folder1": fb,
        "direction0": da,
        "direction1": db,
        "same_video": fa == fb,
        "cross_video": fa != fb,
        "cross_direction": known_opposite,
        "same_direction": da == db and da in {"forward", "reverse"},
    }


def score_grid_select_indices(xs: Any, ys: Any, scores: Any, max_count: int,
                              grid_h: int, grid_w: int, image_h: int | None = None,
                              image_w: int | None = None) -> Any:
    import numpy as np

    n = int(len(xs))
    if max_count <= 0 or n <= max_count:
        return np.arange(n, dtype=np.int64)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    scores = np.asarray(scores, dtype=np.float32)
    image_h = int(image_h or (ys.max() + 1 if n else 1))
    image_w = int(image_w or (xs.max() + 1 if n else 1))
    grid_h = max(1, int(grid_h))
    grid_w = max(1, int(grid_w))
    cy = np.clip((ys.astype(np.float64) / max(1, image_h) * grid_h).astype(np.int64), 0, grid_h - 1)
    cx = np.clip((xs.astype(np.float64) / max(1, image_w) * grid_w).astype(np.int64), 0, grid_w - 1)
    cells = cy * grid_w + cx
    per_cell = max(1, max_count // max(1, grid_h * grid_w))
    selected: list[int] = []
    selected_set: set[int] = set()
    for cell in range(grid_h * grid_w):
        idx = np.flatnonzero(cells == cell)
        if len(idx) == 0:
            continue
        take = idx[np.argsort(-scores[idx])[:per_cell]]
        for i in take.tolist():
            if i not in selected_set:
                selected.append(int(i))
                selected_set.add(int(i))
                if len(selected) >= max_count:
                    return np.asarray(selected, dtype=np.int64)
    for i in np.argsort(-scores).tolist():
        if i not in selected_set:
            selected.append(int(i))
            selected_set.add(int(i))
            if len(selected) >= max_count:
                break
    return np.asarray(selected, dtype=np.int64)


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def load_gray_small(path: Path, width: int = 640):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
    return img


def two_view_motion_metrics(prev_path: Path, cur_path: Path, cfg: SimpleNamespace) -> dict[str, Any]:
    import cv2
    import numpy as np

    prev = load_gray_small(prev_path)
    cur = load_gray_small(cur_path)
    pts0 = cv2.goodFeaturesToTrack(prev, maxCorners=1000, qualityLevel=0.01, minDistance=8)
    if pts0 is None:
        return {"tracks": 0, "median_flow_px": 0.0, "f_inliers": 0, "h_inliers": 0, "h_over_f": 0.0}
    pts1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts0, None)
    if pts1 is None or st is None:
        return {"tracks": 0, "median_flow_px": 0.0, "f_inliers": 0, "h_inliers": 0, "h_over_f": 0.0}
    ok = st.reshape(-1).astype(bool)
    p0 = pts0.reshape(-1, 2)[ok]
    p1 = pts1.reshape(-1, 2)[ok]
    if len(p0) < 8:
        return {"tracks": int(len(p0)), "median_flow_px": 0.0, "f_inliers": 0, "h_inliers": 0, "h_over_f": 0.0}
    flow = np.linalg.norm(p1 - p0, axis=1)
    f_inl = 0
    h_inl = 0
    try:
        _F, fm = cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC, 1.5, 0.99)
        if fm is not None:
            f_inl = int(fm.reshape(-1).astype(bool).sum())
    except Exception:
        f_inl = 0
    try:
        _H, hm = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
        if hm is not None:
            h_inl = int(hm.reshape(-1).astype(bool).sum())
    except Exception:
        h_inl = 0
    return {
        "tracks": int(len(p0)),
        "median_flow_px": float(np.median(flow)) if len(flow) else 0.0,
        "f_inliers": f_inl,
        "h_inliers": h_inl,
        "h_over_f": float(h_inl / max(1, f_inl)),
    }


def classify_motion_metrics(metrics: dict[str, Any], cfg: SimpleNamespace) -> tuple[str, str]:
    tracks = int(metrics.get("tracks", 0))
    flow = float(metrics.get("median_flow_px", 0.0))
    h_inliers = int(metrics.get("h_inliers", 0))
    h_over_f = float(metrics.get("h_over_f", 0.0))
    if tracks < int(cfg.motion_min_tracks) or flow < float(cfg.motion_min_flow_px):
        return "hover", "hover_or_low_motion"
    if h_inliers >= int(cfg.motion_rotation_min_inliers) and h_over_f >= float(cfg.motion_rotation_h_over_f):
        return "pure_rotation", "rotation_dominant"
    return "parallax", "geometry"


def normalize_motion_class(value: str | None, reason: str | None = None) -> str:
    text = (value or "").strip().lower()
    if text in {"parallax", "pure_rotation", "hover", "seed"}:
        return text
    reason = (reason or "").strip().lower()
    if reason in {"geometry", "parallax"}:
        return "parallax"
    if reason in {"rotation_dominant", "pure_rotation"}:
        return "pure_rotation"
    if reason in {"hover_or_low_motion", "low_motion", "hover"}:
        return "hover"
    return "parallax"


def motion_role_for_class(motion_class: str, use_rotation_bridges: bool = False) -> str:
    motion_class = normalize_motion_class(motion_class)
    if motion_class in {"seed", "parallax"}:
        return "triangulation"
    if motion_class == "pure_rotation" and use_rotation_bridges:
        return "bridge_only"
    return "non_parallax"


def load_motion_roles(cfg: SimpleNamespace) -> dict[str, dict[str, Any]]:
    p = cfg_paths(cfg)
    roles: dict[str, dict[str, Any]] = {}
    names, groups = list_images(p.images) if p.images.exists() else ([], {})
    for name in names:
        roles[name] = {"motion_class": "parallax", "motion_role": "triangulation", "kept": True}
    report_path = p.work / "motion_quality.json"
    if not report_path.exists():
        return roles
    report = read_json(report_path)
    use_rotation_bridges = bool(getattr(cfg, "use_rotation_bridges", False))
    for folder, item in report.get("sequences", {}).items():
        rel_names = groups.get(folder, [])
        if rel_names:
            first = rel_names[0]
            roles[first] = {"motion_class": "seed", "motion_role": "triangulation", "kept": True}
        for record in item.get("records", []):
            frame = str(record.get("frame", ""))
            if not frame:
                continue
            name = f"{folder}/{frame}"
            motion_class = normalize_motion_class(record.get("motion_class"), record.get("reason"))
            roles[name] = {
                "motion_class": motion_class,
                "motion_role": motion_role_for_class(motion_class, use_rotation_bridges),
                "kept": bool(record.get("kept", True)),
                "reason": record.get("reason", ""),
            }
    return roles


def is_parallax_role(role: dict[str, Any] | None) -> bool:
    motion_class = normalize_motion_class(str((role or {}).get("motion_class", "parallax")))
    return motion_class in {"parallax", "seed"}


def filter_pairs_by_motion_roles(
    pairs: list[tuple[str, str]],
    roles: dict[str, dict[str, Any]],
    exclude_non_parallax: bool = False,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    stats: dict[str, Any] = {
        "input_pairs": len(pairs),
        "removed_non_parallax_pairs": 0,
        "kept_pairs": len(pairs),
        "exclude_non_parallax": bool(exclude_non_parallax),
    }
    if not exclude_non_parallax:
        return pairs, stats
    kept: list[tuple[str, str]] = []
    removed_by_class: Counter[str] = Counter()
    for a, b in pairs:
        ra = roles.get(a, {"motion_class": "parallax"})
        rb = roles.get(b, {"motion_class": "parallax"})
        if is_parallax_role(ra) and is_parallax_role(rb):
            kept.append((a, b))
            continue
        stats["removed_non_parallax_pairs"] += 1
        removed_by_class[str(ra.get("motion_class", "parallax"))] += 1
        removed_by_class[str(rb.get("motion_class", "parallax"))] += 1
    stats["kept_pairs"] = len(kept)
    stats["removed_endpoint_classes"] = dict(removed_by_class)
    return kept, stats


def motion_gate_frames(cfg: SimpleNamespace) -> None:
    if not getattr(cfg, "motion_gate", True):
        return
    p = cfg_paths(cfg)
    report_path = p.work / "motion_quality.json"
    if report_path.exists() and getattr(cfg, "resume", False) and not getattr(cfg, "overwrite", False):
        log(f"resume: reuse motion gate report {report_path}")
        return
    rejected_root = p.work / "rejected_frames"
    report = {
        "mode": cfg.motion_action,
        "min_flow_px": cfg.motion_min_flow_px,
        "rotation_h_over_f": cfg.motion_rotation_h_over_f,
        "rotation_min_inliers": cfg.motion_rotation_min_inliers,
        "use_rotation_bridges": bool(getattr(cfg, "use_rotation_bridges", False)),
        "classes": ["parallax", "pure_rotation", "hover"],
        "sequences": {},
    }
    for folder in sorted([x for x in p.images.iterdir() if x.is_dir()]):
        frames = sorted(folder.glob("*.jpg"))
        if len(frames) < 2:
            continue
        seq_records = []
        kept = [frames[0]]
        rejected = []
        rotation_seen = 0
        prev_kept = frames[0]
        class_counts: Counter[str] = Counter({"seed": 1})
        for frame in frames[1:]:
            m = two_view_motion_metrics(prev_kept, frame, cfg)
            motion_class, reason = classify_motion_metrics(m, cfg)
            class_counts[motion_class] += 1
            keep = motion_class == "parallax"
            if motion_class == "pure_rotation" and bool(getattr(cfg, "use_rotation_bridges", False)):
                keep = True
            if motion_class == "pure_rotation" and int(cfg.motion_keep_rotation_every) > 0:
                rotation_seen += 1
                keep = (rotation_seen % int(cfg.motion_keep_rotation_every)) == 0
            if keep:
                kept.append(frame)
                prev_kept = frame
            else:
                rejected.append(frame)
            seq_records.append({"frame": frame.name, "reason": reason, "motion_class": motion_class, "kept": keep, **m})
        if cfg.motion_action == "filter" and rejected:
            dst = rejected_root / folder.name
            dst.mkdir(parents=True, exist_ok=True)
            for frame in rejected:
                shutil.move(str(frame), str(dst / frame.name))
        report["sequences"][folder.name] = {
            "total_before": len(frames),
            "kept": len(kept),
            "rejected": len(rejected),
            "motion_classes": dict(class_counts),
            "records": seq_records,
        }
        log(f"motion gate {folder.name}: kept={len(kept)} rejected={len(rejected)} action={cfg.motion_action}")
    write_json(report_path, report)


def focal_for_resolution(width: int, height: int, cfg: SimpleNamespace) -> tuple[str, list[float]]:
    key = f"{width}x{height}"
    if key in cfg.camera_init:
        item = cfg.camera_init[key]
        return item["model"], list(item["params"])
    if cfg.focal_px > 0:
        focal = float(cfg.focal_px)
    else:
        focal = float(width) * float(cfg.focal_ratio)
    return cfg.camera_model, [focal, width / 2.0, height / 2.0, float(cfg.k1)]


def stage_extract(cfg: SimpleNamespace) -> None:
    p = cfg_paths(cfg)
    p.images.mkdir(parents=True, exist_ok=True)
    if cfg.image_root:
        src = as_path(cfg.image_root)
        if src != p.images:
            if p.images.exists() and any(p.images.iterdir()) and not cfg.overwrite:
                log(f"reuse existing images dir {p.images}")
            else:
                if p.images.exists():
                    shutil.rmtree(p.images)
                shutil.copytree(src, p.images)
        if not cfg.dry_run:
            motion_gate_frames(cfg)
        return
    if not cfg.videos:
        raise SystemExit("provide --videos or --image-root")
    for video_s in cfg.videos:
        video = as_path(video_s)
        if not video.exists():
            raise FileNotFoundError(video)
        out_dir = p.images / sanitize_stem(video)
        if out_dir.exists() and list(out_dir.glob("*.jpg")) and cfg.resume and not cfg.overwrite:
            log(f"resume: frames already exist for {video.name}: {out_dir}")
            continue
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        scale = f"scale='if(gt(iw,ih),min({cfg.max_side},iw),-2)':'if(gt(iw,ih),-2,min({cfg.max_side},ih))'"
        vf = f"fps={cfg.fps},{scale}"
        cmd = [
            cfg.ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(video),
            "-vf", vf,
            "-q:v", str(cfg.jpeg_quality),
            str(out_dir / "%06d.jpg"),
        ]
        run_cmd(cmd, dry_run=cfg.dry_run)
    if not cfg.dry_run:
        motion_gate_frames(cfg)


def stage_manifest(cfg: SimpleNamespace) -> None:
    p = cfg_paths(cfg)
    names, groups = list_images(p.images)
    if not names:
        raise SystemExit(f"no extracted frames under {p.images}")
    sizes: dict[str, dict[str, int]] = {}
    for name in names:
        w, h = image_size(p.images / name)
        sizes[name] = {"width": w, "height": h}
    intrinsics_by_resolution: dict[str, dict[str, Any]] = {}
    for s in sizes.values():
        w, h = s["width"], s["height"]
        model, params = focal_for_resolution(w, h, cfg)
        intrinsics_by_resolution[f"{w}x{h}"] = {"model": model, "params": params}
    motion_roles = load_motion_roles(cfg)
    manifest = {
        "site_name": cfg.site_name,
        "image_root": str(p.images),
        "total_frames": len(names),
        "groups": {k: len(v) for k, v in groups.items()},
        "frames": [
            {
                "name": n,
                **sizes[n],
                "motion_class": motion_roles.get(n, {}).get("motion_class", "parallax"),
                "motion_role": motion_roles.get(n, {}).get("motion_role", "triangulation"),
            }
            for n in names
        ],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(p.manifest, manifest)
    write_json(p.intrinsics, {
        "camera_mode": "PER_FOLDER",
        "intrinsics_by_resolution": intrinsics_by_resolution,
        "notes": "Principal point is image center. Override --camera-init-json for calibrated cameras.",
    })
    log(f"manifest: {len(names)} frames across {len(groups)} videos -> {p.manifest}")


def stage_pairs(cfg: SimpleNamespace) -> None:
    import h5py
    import numpy as np
    import torch

    p = cfg_paths(cfg)
    sys.path.insert(0, str(as_path(cfg.template_repo) / "scripts"))
    import megaloc_lib  # type: ignore

    p.megaloc.mkdir(parents=True, exist_ok=True)
    names, groups = list_images(p.images)
    if len(names) < 2:
        raise SystemExit("need at least two frames")
    device = "cuda" if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"
    log(f"MegaLoc descriptors: frames={len(names)} groups={', '.join(f'{k}:{len(v)}' for k, v in groups.items())}")
    desc = megaloc_lib.extract(names, p.images, device).astype(np.float32)
    desc /= np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12
    with h5py.File(str(p.global_desc), "w") as fd:
        for name, d in zip(names, desc):
            fd.create_group(name).create_dataset("global_descriptor", data=d)

    idx = {n: i for i, n in enumerate(names)}
    directions = sequence_directions(cfg, groups)
    folder_of = {name: folder for folder, rel in groups.items() for name in rel}
    kind_priority = {"temporal": 0, "same_direction_retrieval": 1, "cross_topk": 2, "cross_grid": 3, "retrieval": 4}
    pair_meta: dict[tuple[str, str], dict[str, Any]] = {}

    def add_pair(a: str, b: str, kind: str, score: float = 0.0) -> None:
        if a != b:
            pair = tuple(sorted((a, b)))
            meta = pair_meta.get(pair)
            if meta is None or kind_priority.get(kind, 9) < kind_priority.get(str(meta.get("kind")), 9) or score > float(meta.get("score", -1.0)):
                pair_meta[pair] = {"kind": kind, "score": float(score)}

    def sorted_pairs() -> list[tuple[str, str]]:
        return sorted(
            pair_meta,
            key=lambda x: (
                kind_priority.get(str(pair_meta[x].get("kind")), 9),
                -float(pair_meta[x].get("score", 0.0)),
                x[0],
                x[1],
            ),
        )

    sim = None
    if (
        cfg.pair_graph_mode == "legacy"
        or int(getattr(cfg, "same_direction_topk", 0)) > 0
    ):
        sim = desc @ desc.T
        np.fill_diagonal(sim, -1.0)

    if cfg.pair_graph_mode == "legacy":
        assert sim is not None
        for i, name in enumerate(names):
            for j in np.argsort(-sim[i])[:cfg.num_matched]:
                add_pair(name, names[int(j)], "retrieval", float(sim[i, int(j)]))
    n_retrieval = len(pair_meta)

    for rel in groups.values():
        for i, name in enumerate(rel):
            for off in range(1, cfg.seq_window + 1):
                if i + off < len(rel):
                    add_pair(name, rel[i + off], "temporal", 1.0 / off)
    n_temporal = len(pair_meta) - n_retrieval

    if cfg.pair_graph_mode == "directional" and int(getattr(cfg, "same_direction_topk", 0)) > 0:
        assert sim is not None
        for i, name in enumerate(names):
            f0 = folder_of.get(name, "")
            d0 = directions.get(f0, "unknown")
            if d0 == "unknown":
                continue
            candidates = []
            for j in np.argsort(-sim[i]):
                other = names[int(j)]
                if other == name:
                    continue
                f1 = folder_of.get(other, "")
                if f0 != f1 and directions.get(f1, "unknown") == d0:
                    candidates.append(int(j))
                if len(candidates) >= int(cfg.same_direction_topk):
                    break
            for j in candidates:
                add_pair(name, names[j], "same_direction_retrieval", float(sim[i, j]))

    before_cross = len(pair_meta)
    group_items = list(groups.items())
    has_known_direction = any(v in {"forward", "reverse"} for v in directions.values())
    for (_ga, a_names), (_gb, b_names) in itertools.combinations(group_items, 2):
        da, db = directions.get(_ga, "unknown"), directions.get(_gb, "unknown")
        if cfg.pair_graph_mode == "directional":
            is_bridge = da in {"forward", "reverse"} and db in {"forward", "reverse"} and da != db
            if not is_bridge and has_known_direction:
                continue
        ai = np.array([idx[n] for n in a_names], dtype=int)
        bi = np.array([idx[n] for n in b_names], dtype=int)
        s = desc[ai] @ desc[bi].T
        if cfg.cross_topk > 0:
            for i in range(len(a_names)):
                for j in np.argsort(-s[i])[:cfg.cross_topk]:
                    add_pair(a_names[i], b_names[int(j)], "cross_topk", float(s[i, int(j)]))
            for j in range(len(b_names)):
                for i in np.argsort(-s[:, j])[:cfg.cross_topk]:
                    add_pair(a_names[int(i)], b_names[j], "cross_topk", float(s[int(i), j]))
        if cfg.cross_grid > 0:
            for i in range(0, len(a_names), cfg.cross_grid):
                for j in range(0, len(b_names), cfg.cross_grid):
                    add_pair(a_names[i], b_names[j], "cross_grid", float(s[i, j]))
    n_cross = len(pair_meta) - before_cross

    uncapped_pairs = sorted_pairs()
    write_pairs(p.pairs_uncapped, uncapped_pairs)
    out_pairs = apply_pair_degree_cap(
        uncapped_pairs,
        groups,
        directions,
        total_cap=cfg.agg_pair_degree_cap,
        intra_cap=cfg.agg_intra_degree_cap,
        cross_direction_cap=cfg.agg_cross_direction_degree_cap,
    )
    write_pairs(p.pairs, out_pairs)
    kind_counts = Counter(str(pair_meta[p].get("kind")) for p in uncapped_pairs)
    write_pair_graph_reports(cfg, out_pairs, uncapped_pairs, groups, directions, pair_meta)
    write_json(p.megaloc / "pairs_summary.json", {
        "frames": len(names),
        "groups": {k: len(v) for k, v in groups.items()},
        "directions": directions,
        "pair_graph_mode": cfg.pair_graph_mode,
        "num_matched": cfg.num_matched,
        "seq_window": cfg.seq_window,
        "cross_topk": cfg.cross_topk,
        "cross_grid": cfg.cross_grid,
        "same_direction_topk": int(getattr(cfg, "same_direction_topk", 0)),
        "retrieval_pairs_after_stage": n_retrieval,
        "temporal_added": n_temporal,
        "cross_added": n_cross,
        "uncapped_pairs": len(uncapped_pairs),
        "pair_kinds": dict(kind_counts),
        "degree_caps": {
            "intra": cfg.agg_intra_degree_cap,
            "cross_direction": cfg.agg_cross_direction_degree_cap,
            "total": cfg.agg_pair_degree_cap,
        },
        "total_pairs": len(out_pairs),
        "pair_graph_diagnostics": str(p.pair_graph_diagnostics),
        "rotation_bridge_report": str(p.rotation_bridge_report),
    })
    log(
        f"pairs total={len(out_pairs)} uncapped={len(uncapped_pairs)} "
        f"mode={cfg.pair_graph_mode} temporal+={n_temporal} cross+={n_cross} -> {p.pairs}"
    )


def stage_doppelgangers(cfg: SimpleNamespace) -> None:
    if not cfg.doppelgangers_checkpoint:
        log("skip Doppelgangers++ filter: --doppelgangers-checkpoint not provided")
        return
    import numpy as np
    import torch
    from scipy.special import softmax
    from tqdm import tqdm

    p = cfg_paths(cfg)
    dg_root = as_path(cfg.doppelgangers_root)
    sys.path.insert(0, str(dg_root))
    import mast3r.utils.path_to_dust3r  # type: ignore  # noqa: F401
    try:
        from dust3r.image_pairs import make_pairs  # type: ignore
    except ModuleNotFoundError:
        from mast3r.image_pairs import make_pairs  # type: ignore
    from dust3r.utils.image import load_images  # type: ignore
    from mast3r.inference import inference  # type: ignore
    from mast3r.model import AsymmetricMASt3R  # type: ignore

    torch_load_orig = torch.load

    def torch_load_checkpoint_compat(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return torch_load_orig(*args, **kwargs)

    torch.load = torch_load_checkpoint_compat  # type: ignore[assignment]

    pairs = [tuple(line.split()) for line in p.pairs.read_text().splitlines() if len(line.split()) == 2]
    if not pairs:
        raise SystemExit("pair list is empty")
    shutil.copy2(p.pairs, p.pairs_before_dg)
    _, groups = list_images(p.images)
    folder_of = {name: folder for folder, rel in groups.items() for name in rel}
    directions = sequence_directions(cfg, groups)
    scope = getattr(cfg, "doppelgangers_filter_scope", "cross_direction")

    def should_filter_pair(a: str, b: str) -> bool:
        rel = pair_relation(a, b, folder_of, directions)
        if scope == "all":
            return True
        if scope == "cross_video":
            return bool(rel["cross_video"])
        return bool(rel["cross_direction"])

    device = "cuda" if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"
    model = AsymmetricMASt3R(
        pos_embed="RoPE100",
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(512, 512),
        head_type="catmlp+dpt",
        head_type_dg="transformer",
        output_mode="pts3d+desc24",
        output_mode_dg="dg_score",
        depth_mode=("exp", -np.inf, np.inf),
        conf_mode=("exp", 1, np.inf),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
        two_confs=True,
        desc_conf_mode=("exp", 0, np.inf),
        add_dg_pred_head=True,
        freeze=["mask", "encoder", "decoder", "head"],
    ).from_pretrained(cfg.doppelgangers_checkpoint).to(device)

    kept: list[tuple[str, str]] = []
    scores: list[float] = []
    skipped_by_scope = 0
    for a, b in tqdm(pairs, desc="Doppelgangers++"):
        if not should_filter_pair(a, b):
            kept.append((a, b))
            scores.append(1.0)
            skipped_by_scope += 1
            continue
        img_paths = [str(p.images / a), str(p.images / b)]
        images = load_images(img_paths, size=512, verbose=False)
        output = inference(make_pairs(images), model, device, verbose=False)
        pred1, pred2 = output["pred1"], output["pred2"]
        pred1 = torch.stack(pred1, dim=0) if isinstance(pred1, list) else pred1
        pred2 = torch.stack(pred2, dim=0) if isinstance(pred2, list) else pred2
        score_s1 = softmax(pred1.detach().cpu().numpy(), axis=1)
        score_s2 = softmax(pred2.detach().cpu().numpy(), axis=1)
        vote_0 = sum(score_s1[:, 0] > score_s1[:, 1]) + sum(score_s2[:, 0] > score_s2[:, 1])
        vote_1 = sum(score_s1[:, 1] > score_s1[:, 0]) + sum(score_s2[:, 1] > score_s2[:, 0])
        if vote_1 > vote_0:
            score = float(np.max((score_s1[:, 1], score_s2[:, 1])))
        elif vote_1 < vote_0:
            score = float(np.min((score_s1[:, 1], score_s2[:, 1])))
        else:
            score = float(np.mean((score_s1[:, 1], score_s2[:, 1])))
        scores.append(score)
        if score >= cfg.doppelgangers_threshold:
            kept.append((a, b))
    write_pairs(p.pairs, kept)
    np.save(str(p.megaloc / "pair_probability_list_dust3r.npy"), {"prob": np.asarray(scores).reshape(-1, 1)})
    log(
        f"Doppelgangers++ kept {len(kept)}/{len(pairs)} pairs at threshold {cfg.doppelgangers_threshold} "
        f"scope={scope} skipped_by_scope={skipped_by_scope}"
    )


def resolve_mvroma_runtime_paths(cfg: SimpleNamespace) -> SimpleNamespace:
    mvroma_root = as_path(cfg.mvroma_root).resolve(strict=True)
    ufm_root = as_path(getattr(cfg, "ufm_root", DEFAULT_UFM_ROOT)).resolve(strict=True)
    hf_cache = as_path(
        getattr(
            cfg,
            "ufm_hf_hub_cache",
            os.environ.get("HF_HUB_CACHE")
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
            or str(Path.home() / ".cache" / "huggingface" / "hub"),
        )
    ).resolve(strict=True)
    torch_home = as_path(
        os.environ.get("TORCH_HOME") or str(Path.home() / ".cache" / "torch")
    ).resolve(strict=True)
    dinov2_source = as_path(
        getattr(
            cfg,
            "dinov2_source_root",
            torch_home / "hub" / "facebookresearch_dinov2_main",
        )
    ).resolve(strict=True)
    dinov2_weights = as_path(
        getattr(
            cfg,
            "dinov2_weights",
            torch_home / "hub" / "checkpoints" / "dinov2_vitl14_pretrain.pth",
        )
    ).resolve(strict=True)
    snapshot = resolve_local_ufm_snapshot(
        hf_cache,
        expected_files=MVROMA_UFM_EXPECTED_FILES,
    )
    snapshot_path = Path(snapshot["snapshot_path"])
    asset_paths = {
        "mvroma_checkpoint": as_path(cfg.mvroma_weights).resolve(strict=True),
        "ufm_config": snapshot_path / "config.json",
        "ufm_weights": snapshot_path / "model.safetensors",
        "dinov2_weights": dinov2_weights,
        "dinov2_source": dinov2_source,
    }
    return SimpleNamespace(
        mvroma_root=mvroma_root,
        ufm_root=ufm_root,
        hf_cache=hf_cache,
        ufm_snapshot=snapshot,
        dinov2_source=dinov2_source,
        dinov2_weights=dinov2_weights,
        asset_paths=asset_paths,
    )


def _attest_dinov2_frozen_source(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    tree = mvroma_tree_content_identity(root, python_only=False)
    for key, expected in MVROMA_DINOV2_SOURCE_EXPECTED.items():
        if tree.get(key) != expected:
            raise RuntimeError(
                f"frozen DINOv2 source {key} mismatch: {tree.get(key)} != {expected}"
            )
    selected = build_mvroma_frozen_source_tree(
        root, [str(row[0]) for row in tree["files"]]
    )
    return tree, selected


def import_mvroma_runtime_modules() -> SimpleNamespace:
    import h5py
    import numpy as np
    import torch
    from argparse import Namespace
    from PIL import Image
    from src.build_model import build_our_model  # type: ignore
    from src.matchers.run_matcher_path import (  # type: ignore
        UniFlowMatchClassificationRefinement as runner_ufm_class,
    )
    from src.matchers.uniflowmatch.models.ufm import (  # type: ignore
        UniFlowMatchClassificationRefinement as vendored_ufm_class,
    )
    from src.mvroma import ModelConfig  # type: ignore
    from src.run_model import run_model_test  # type: ignore

    modules = {
        name: module
        for name, module in sys.modules.items()
        if _is_mvroma_target_module(name) and name != "src"
    }
    return SimpleNamespace(
        h5py=h5py,
        np=np,
        torch=torch,
        Namespace=Namespace,
        Image=Image,
        build_our_model=build_our_model,
        ModelConfig=ModelConfig,
        run_model_test=run_model_test,
        vendored_ufm_class=vendored_ufm_class,
        runner_ufm_class=runner_ufm_class,
        modules=modules,
    )


def prepare_mvroma_stage_runtime(
    cfg: SimpleNamespace, paths: SimpleNamespace
) -> Any:
    if not bool(getattr(cfg, "o101_mvroma_candidate", False)):
        return {"cfg": cfg, "paths": paths}

    resolved = resolve_mvroma_runtime_paths(cfg)
    jobs = build_mvroma_source_jobs(
        paths.pairs.read_text().splitlines(),
        limit_src=int(getattr(cfg, "limit_src", 0)),
        chunk_size=int(cfg.mvroma_chunk),
    )
    image_tree = build_mvroma_image_sha256_tree(paths.images, jobs)
    source_roots = attest_mvroma_python_source_roots(
        resolved.mvroma_root,
        resolved.ufm_root,
        expected=MVROMA_PYTHON_SOURCE_EXPECTED,
    )
    dino_tree, dino_source_identity = _attest_dinov2_frozen_source(
        resolved.dinov2_source
    )
    initial_assets = attest_mvroma_runtime_assets(
        resolved.asset_paths,
        phase="initial_pre_validation_pre_model",
        require_frozen_dinov2=True,
    )
    if initial_assets["files"]["mvroma_checkpoint"] != MVROMA_CHECKPOINT_EXPECTED:
        raise RuntimeError("frozen MV-RoMa checkpoint identity mismatch")
    for name, expected in MVROMA_UFM_EXPECTED_FILES.items():
        key = "ufm_config" if name == "config.json" else "ufm_weights"
        if initial_assets["files"][key] != expected:
            raise RuntimeError(f"frozen UFM {name} identity mismatch")

    stack = ExitStack()
    try:
        private_mvroma = stack.enter_context(
            private_attested_mvroma_source_tree(
                resolved.mvroma_root, source_roots["identity"]["mvroma"]
            )
        )
        private_ufm = stack.enter_context(
            private_attested_mvroma_source_tree(
                resolved.ufm_root, source_roots["identity"]["ufm"]
            )
        )
        private_dino = stack.enter_context(
            private_attested_mvroma_source_tree(
                resolved.dinov2_source, dino_source_identity
            )
        )
        stack.enter_context(
            private_mvroma_import_environment(private_mvroma, private_ufm)
        )
        runtime_objects = import_mvroma_runtime_modules()
        allowed_paths = {
            "mvroma_vendored": {
                str(row[0]) for row in source_roots["identity"]["mvroma"]["files"]
            },
            "external_ufm": {
                str(row[0]) for row in source_roots["identity"]["ufm"]["files"]
            },
        }
        module_identity = collect_mvroma_module_identity(
            runtime_objects.modules,
            mvroma_root=private_mvroma,
            ufm_root=private_ufm,
            allowed_relative_paths=allowed_paths,
        )
        runtime_probe = probe_mvroma_effective_runtime(
            runtime_objects.torch,
            device=str(cfg.device),
            modules=runtime_objects.modules,
        )
        sample_h, sample_w = parse_grid_spec(
            getattr(cfg, "mvroma_sample_grid", "8x12"), 8, 12
        )
        stage_contract = build_mvroma_stage_contract(
            implementation={
                "orchestrator": attest_mvroma_loaded_source(),
                "python_sources": source_roots["identity"],
                "dinov2_source": dino_source_identity,
                "modules": module_identity["identity"],
                "loader_policy": "a011-postmodel-v3-held-source-candidate/v2",
            },
            models={
                "runtime_assets": initial_assets,
                "ufm_snapshot": resolved.ufm_snapshot["identity"],
            },
            inference={
                "coarse_res_hw": [560, 560],
                "target_res_hw": [int(cfg.mvroma_grid_h), int(cfg.mvroma_grid_w)],
                "chunk_size": int(cfg.mvroma_chunk),
                "sample_mode": str(cfg.mvroma_sample_mode),
                "sample_grid": [sample_h, sample_w],
                "certainty_threshold": float(cfg.roma_cert_thresh),
                "max_correspondences": int(cfg.agg_maxkp),
                "upsample_preds": True,
                "num_cluster": 512,
                "prematcher": "ufm",
            },
            runtime=runtime_probe["identity"],
            post_model_expectation_ref=MVROMA_POST_MODEL_EXPECTED_REF,
        )
        provenance = {
            **source_roots["provenance"],
            "module_paths": module_identity["provenance"],
            "runtime": runtime_probe["provenance"],
            "mvroma_checkpoint": str(resolved.asset_paths["mvroma_checkpoint"]),
            "ufm_snapshot": str(resolved.ufm_snapshot["snapshot_path"]),
            "dinov2_source": str(resolved.dinov2_source),
            "dinov2_weights": str(resolved.dinov2_weights),
            "dinov2_source_tree": dino_tree,
        }
        return SimpleNamespace(
            cfg=cfg,
            paths=paths,
            close=stack.close,
            private_mvroma_root=str(private_mvroma),
            private_ufm_root=str(private_ufm),
            private_dinov2_root=str(private_dino),
            source_roots=source_roots,
            dino_source_identity=dino_source_identity,
            runtime_objects=runtime_objects,
            runtime_paths=resolved,
            module_identity=module_identity,
            initial_assets=initial_assets,
            jobs=jobs,
            image_tree=image_tree,
            stage_contract=stage_contract,
            provenance=provenance,
        )
    except BaseException as primary_error:
        try:
            stack.close()
        except BaseException as cleanup_error:
            _attach_mvroma_cleanup_error(primary_error, cleanup_error)
        raise


def snapshot_mvroma_candidate_cfg(
    cfg: SimpleNamespace, stage_payload: dict[str, Any]
) -> SimpleNamespace:
    required_inference = {
        "coarse_res_hw",
        "target_res_hw",
        "chunk_size",
        "sample_mode",
        "sample_grid",
        "certainty_threshold",
        "max_correspondences",
        "upsample_preds",
        "num_cluster",
        "prematcher",
    }
    inference = stage_payload.get("inference")
    runtime = stage_payload.get("runtime")
    if (
        not isinstance(inference, dict)
        or set(inference) != required_inference
        or not isinstance(runtime, dict)
        or not isinstance(runtime.get("gpu"), dict)
    ):
        raise ValueError("invalid MV-RoMa candidate stage semantics")
    target = inference["target_res_hw"]
    sample_grid = inference["sample_grid"]
    if (
        inference["coarse_res_hw"] != [560, 560]
        or not isinstance(target, list)
        or len(target) != 2
        or not all(isinstance(value, int) and value >= 2 for value in target)
        or not isinstance(sample_grid, list)
        or len(sample_grid) != 2
        or not all(isinstance(value, int) and value > 0 for value in sample_grid)
        or not isinstance(inference["chunk_size"], int)
        or inference["chunk_size"] < 1
        or inference["sample_mode"] not in {"score_grid", "random"}
        or not isinstance(inference["certainty_threshold"], (int, float))
        or isinstance(inference["certainty_threshold"], bool)
        or not math.isfinite(float(inference["certainty_threshold"]))
        or not 0.0 <= float(inference["certainty_threshold"]) <= 1.0
        or not isinstance(inference["max_correspondences"], int)
        or inference["max_correspondences"] < 16
        or inference["upsample_preds"] is not True
        or inference["num_cluster"] != 512
        or inference["prematcher"] != "ufm"
    ):
        raise ValueError("invalid MV-RoMa candidate inference semantics")
    selected_device = runtime["gpu"].get("selected_device")
    if not isinstance(selected_device, str) or not selected_device:
        raise ValueError("invalid MV-RoMa candidate runtime device")
    try:
        configured_grid = parse_grid_spec(cfg.mvroma_sample_grid, 8, 12)
        configured = {
            "device": str(cfg.device),
            "target_res_hw": [int(cfg.mvroma_grid_h), int(cfg.mvroma_grid_w)],
            "chunk_size": int(cfg.mvroma_chunk),
            "sample_mode": str(cfg.mvroma_sample_mode),
            "sample_grid": list(configured_grid),
            "certainty_threshold": float(cfg.roma_cert_thresh),
            "max_correspondences": int(cfg.agg_maxkp),
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("MV-RoMa candidate cfg drift") from exc
    expected = {
        "device": selected_device,
        "target_res_hw": target,
        "chunk_size": inference["chunk_size"],
        "sample_mode": inference["sample_mode"],
        "sample_grid": sample_grid,
        "certainty_threshold": float(inference["certainty_threshold"]),
        "max_correspondences": inference["max_correspondences"],
    }
    if configured != expected:
        raise RuntimeError(
            f"MV-RoMa candidate cfg drift: {configured} != {expected}"
        )
    return SimpleNamespace(
        device=selected_device,
        mvroma_grid_h=target[0],
        mvroma_grid_w=target[1],
        mvroma_chunk=inference["chunk_size"],
        mvroma_sample_mode=inference["sample_mode"],
        mvroma_sample_grid=f"{sample_grid[0]}x{sample_grid[1]}",
        roma_cert_thresh=float(inference["certainty_threshold"]),
        agg_maxkp=inference["max_correspondences"],
    )


def build_mvroma_candidate_source_runner(
    cfg: SimpleNamespace,
    runtime_objects: SimpleNamespace,
    *,
    model: Any,
    prematch: Any,
) -> Any:
    """Build the legacy-equivalent per-source runner over held image aliases."""
    np = runtime_objects.np
    torch = runtime_objects.torch
    Image = runtime_objects.Image
    run_model_test = runtime_objects.run_model_test
    grid_h = int(cfg.mvroma_grid_h)
    grid_w = int(cfg.mvroma_grid_w)
    if grid_h < 2 or grid_w < 2:
        raise ValueError("MV-RoMa candidate grid dimensions must be at least two")
    sample_grid_h, sample_grid_w = parse_grid_spec(
        getattr(cfg, "mvroma_sample_grid", "8x12"), 8, 12
    )
    size_cache: dict[str, tuple[int, int]] = {}

    def image_size(name: str, path: str) -> tuple[int, int]:
        if name not in size_cache:
            with Image.open(path) as image:
                size_cache[name] = image.size
        return size_cache[name]

    def source_grid(width: int, height: int) -> tuple[Any, Any]:
        grid_y, grid_x = np.meshgrid(
            np.arange(grid_h), np.arange(grid_w), indexing="ij"
        )
        original_x = (grid_x / (grid_w - 1) * (width - 1)).astype(np.float32)
        original_y = (grid_y / (grid_h - 1) * (height - 1)).astype(np.float32)
        return original_x, original_y

    def run_source(job: dict[str, Any]) -> dict[str, tuple[Any, Any, Any]]:
        source_index = int(job["source_index"])
        source_name = str(job["source"])
        targets = [str(value) for value in job["targets"]]
        chunks = [[str(value) for value in chunk] for chunk in job["chunks"]]
        if [value for chunk in chunks for value in chunk] != targets:
            raise ValueError(f"MV-RoMa source {source_index} chunks do not match targets")
        bound_targets = [str(value) for value in job["bound_target_paths"]]
        if len(bound_targets) != len(targets):
            raise ValueError(f"MV-RoMa source {source_index} bound target count mismatch")
        target_paths = dict(zip(targets, bound_targets, strict=True))
        bound_source = str(job["bound_source_path"])
        source_width, source_height = image_size(source_name, bound_source)
        source_x, source_y = source_grid(source_width, source_height)
        matches_by_target: dict[str, tuple[Any, Any, Any]] = {}

        for chunk in chunks:
            data = {
                "query_img_path": bound_source,
                "ref_img_paths": [target_paths[target] for target in chunk],
            }
            with torch.inference_mode():
                output = run_model_test(
                    model,
                    data,
                    coarse_res_hw=(560, 560),
                    target_res_hw=(grid_h, grid_w),
                    prematch_model=prematch,
                    prematch_model_name="ufm",
                    upsample_preds=True,
                    num_cluster=512,
                    device=cfg.device,
                )
            first_scale = min(output.keys())
            flow = output[first_scale]["flow"][0].cpu().numpy()
            certainty = (
                output[first_scale]["certainty"][0].sigmoid()[:, 0].cpu().numpy()
            )
            if flow.shape[:2] != (len(chunk), 2) or certainty.shape[0] != len(chunk):
                raise ValueError(
                    f"MV-RoMa source {source_index} output reference count mismatch"
                )
            certainty_max = certainty.max(0)
            rows, columns = np.where(certainty_max > cfg.roma_cert_thresh)
            if len(rows) == 0:
                continue
            if len(rows) > cfg.agg_maxkp:
                if cfg.mvroma_sample_mode == "random":
                    selected = np.random.RandomState(source_index).choice(
                        len(rows), cfg.agg_maxkp, replace=False
                    )
                else:
                    selected = score_grid_select_indices(
                        columns,
                        rows,
                        certainty_max[rows, columns],
                        cfg.agg_maxkp,
                        sample_grid_h,
                        sample_grid_w,
                        image_h=grid_h,
                        image_w=grid_w,
                    )
                rows, columns = rows[selected], columns[selected]

            for target_index, target in enumerate(chunk):
                target_width, target_height = image_size(target, target_paths[target])
                scores = certainty[target_index, rows, columns]
                keep = scores > cfg.roma_cert_thresh
                if int(keep.sum()) < 16:
                    continue
                kept_rows, kept_columns = rows[keep], columns[keep]
                target_x = (
                    (flow[target_index, 0, kept_rows, kept_columns] + 1)
                    / 2
                    * (grid_w - 1)
                )
                target_y = (
                    (flow[target_index, 1, kept_rows, kept_columns] + 1)
                    / 2
                    * (grid_h - 1)
                )
                inside = (
                    (target_x >= 0)
                    & (target_x < grid_w)
                    & (target_y >= 0)
                    & (target_y < grid_h)
                )
                if int(inside.sum()) < 16:
                    continue
                kept_rows = kept_rows[inside]
                kept_columns = kept_columns[inside]
                target_x = target_x[inside]
                target_y = target_y[inside]
                kept_scores = scores[keep][inside]
                keypoints0 = np.stack(
                    [source_x[kept_rows, kept_columns], source_y[kept_rows, kept_columns]],
                    axis=1,
                ).astype(np.float32)
                keypoints1 = np.stack(
                    [
                        target_x / (grid_w - 1) * (target_width - 1),
                        target_y / (grid_h - 1) * (target_height - 1),
                    ],
                    axis=1,
                ).astype(np.float32)
                matches_by_target[target] = (
                    keypoints0,
                    keypoints1,
                    kept_scores.astype(np.float32),
                )

        return matches_by_target

    return run_source


def execute_prepared_mvroma_stage(
    cfg: SimpleNamespace, paths: SimpleNamespace, prepared: Any
) -> dict[str, Any] | None:
    if isinstance(prepared, dict):
        if prepared.get("cfg") is not cfg or prepared.get("paths") is not paths:
            raise ValueError("prepared MV-RoMa stage does not match cfg/paths")
        _stage_mvroma_legacy_body(cfg)
        return
    if prepared.cfg is not cfg or prepared.paths is not paths:
        raise ValueError("prepared MV-RoMa candidate does not match cfg/paths")
    stage_contract = snapshot_mvroma_stage_contract(prepared.stage_contract)
    stage_payload = stage_contract["payload"]
    candidate_cfg = snapshot_mvroma_candidate_cfg(cfg, stage_payload)
    initial_assets = stage_payload["models"].get("runtime_assets")
    if not isinstance(initial_assets, dict) or initial_assets != prepared.initial_assets:
        raise RuntimeError("MV-RoMa prepared runtime assets differ from stage contract")
    if not callable(prepared.close):
        raise TypeError("MV-RoMa prepared runtime close must be callable")
    loaded_state: dict[str, Any] = {}
    verification_state = {"called": False}
    cleanup_state = {"complete": False}
    lifecycle_state = {"model_load_attempted": False}

    def cleanup_runtime() -> None:
        if cleanup_state["complete"]:
            return
        loaded_state.clear()
        try:
            import gc

            gc.collect()
            if lifecycle_state["model_load_attempted"] and str(
                candidate_cfg.device
            ).startswith("cuda"):
                prepared.runtime_objects.torch.cuda.synchronize(
                    candidate_cfg.device
                )
                prepared.runtime_objects.torch.cuda.empty_cache()
        finally:
            prepared.close()
        cleanup_state["complete"] = True

    def runner_loader() -> Any:
        if loaded_state:
            raise RuntimeError("MV-RoMa candidate loader was called twice")
        lifecycle_state["model_load_attempted"] = True
        runtime = prepared.runtime_objects
        mvroma_loaded = load_pinned_mvroma_model(
            runtime,
            checkpoint=prepared.runtime_paths.asset_paths["mvroma_checkpoint"],
            expected_checkpoint=initial_assets["files"]["mvroma_checkpoint"],
            device=str(candidate_cfg.device),
            expected_load_identity=MVROMA_STATE_LOAD_EXPECTED,
        )
        loaded_state["mvroma"] = mvroma_loaded
        expected_ufm_assets = {
            name: initial_assets["files"][name]
            for name in ("ufm_config", "ufm_weights", "dinov2_weights")
        }
        ufm_loaded = load_pinned_ufm_model_with_identity(
            runtime.runner_ufm_class,
            snapshot_path=prepared.runtime_paths.ufm_snapshot["snapshot_path"],
            device=str(candidate_cfg.device),
            torch_module=runtime.torch,
            dinov2_source=prepared.private_dinov2_root,
            dinov2_weights=prepared.runtime_paths.dinov2_weights,
            expected_assets=expected_ufm_assets,
            expected_load_identity=MVROMA_UFM_STATE_LOAD_EXPECTED,
        )
        loaded_state["ufm"] = ufm_loaded
        runner = build_mvroma_candidate_source_runner(
            candidate_cfg,
            runtime,
            model=mvroma_loaded.model,
            prematch=ufm_loaded.prematch,
        )
        return runner

    def post_model_verifier(post_model_assets: dict[str, Any]) -> None:
        if verification_state["called"]:
            raise RuntimeError("MV-RoMa post-model verifier was called twice")
        verification_state["called"] = True
        try:
            if set(loaded_state) != {"mvroma", "ufm"}:
                raise RuntimeError("MV-RoMa post-model verifier has no loaded models")
            collected = collect_prepared_mvroma_post_model_identity(
                candidate_cfg,
                prepared,
                mvroma_loaded=loaded_state["mvroma"],
                ufm_loaded=loaded_state["ufm"],
                post_model_assets=post_model_assets,
            )
            verify_mvroma_post_model_expectation(
                collected.identity,
                pre_model_runtime=stage_payload["runtime"],
                expected_ref=stage_payload["post_model_expectation_ref"],
            )
        finally:
            loaded_state.clear()

    try:
        result = execute_attested_mvroma_resume(
            prepared.jobs,
            paths.dense_matches.parent / "source-shards-v1",
            paths.dense_matches,
            stage_contract["sha256"],
            paths.images,
            prepared.image_tree,
            prepared.runtime_paths.asset_paths,
            initial_assets,
            max_correspondences=int(candidate_cfg.agg_maxkp),
            mvroma_resume=bool(getattr(cfg, "mvroma_resume", True)),
            overwrite=bool(getattr(cfg, "overwrite", False)),
            runner_loader=runner_loader,
            post_model_verifier=post_model_verifier,
            pre_merge_runtime_finalizer=cleanup_runtime,
        )
        log(
            "MV-RoMa candidate done: "
            f"groups={result['groups']} reused={result['reused_sources']} "
            f"recomputed={result['recomputed_sources']}"
        )
    except BaseException as execution_error:
        import traceback

        traceback.clear_frames(execution_error.__traceback__)
        try:
            cleanup_runtime()
        except BaseException as cleanup_error:
            _attach_mvroma_cleanup_error(execution_error, cleanup_error)
        raise
    else:
        cleanup_runtime()
        return result


def stage_mvroma(cfg: SimpleNamespace) -> None:
    paths = cfg_paths(cfg)
    with mvroma_stage_lock(mvroma_lock_path(paths.dense_matches)):
        prepared = prepare_mvroma_stage_runtime(cfg, paths)
        close = getattr(prepared, "close", None)
        try:
            execute_prepared_mvroma_stage(cfg, paths, prepared)
        except BaseException as execution_error:
            if callable(close):
                try:
                    close()
                except BaseException as cleanup_error:
                    _attach_mvroma_cleanup_error(execution_error, cleanup_error)
            raise
        else:
            if callable(close):
                close()


def _stage_mvroma_legacy_body(cfg: SimpleNamespace) -> None:
    import h5py
    import numpy as np
    import torch
    from PIL import Image

    p = cfg_paths(cfg)
    mvroma_root = as_path(cfg.mvroma_root)
    sys.path.insert(0, str(mvroma_root))
    os.chdir(mvroma_root)
    from argparse import Namespace
    from src.build_model import build_our_model  # type: ignore
    from src.matchers import build_prematch_model  # type: ignore
    from src.mvroma import ModelConfig  # type: ignore
    from src.run_model import run_model_test  # type: ignore

    pairs_by_src: dict[str, set[str]] = defaultdict(set)
    for line in p.pairs.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] != parts[1]:
            a, b = sorted(parts)
            pairs_by_src[a].add(b)
    srcs = sorted(pairs_by_src)
    if cfg.limit_src:
        srcs = srcs[:cfg.limit_src]
    pair_count = sum(len(pairs_by_src[s]) for s in srcs)
    sample_grid_h, sample_grid_w = parse_grid_spec(getattr(cfg, "mvroma_sample_grid", "8x12"), 8, 12)
    log(
        f"MV-RoMa sources={len(srcs)} pairs={pair_count} cert>{cfg.roma_cert_thresh} "
        f"maxkp={cfg.agg_maxkp} sample={cfg.mvroma_sample_mode}:{sample_grid_h}x{sample_grid_w}"
    )

    size_cache: dict[str, tuple[int, int]] = {}

    def size(name: str) -> tuple[int, int]:
        if name not in size_cache:
            with Image.open(p.images / name) as im:
                size_cache[name] = im.size
        return size_cache[name]

    def source_grid(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
        gy, gx = np.meshgrid(np.arange(cfg.mvroma_grid_h), np.arange(cfg.mvroma_grid_w), indexing="ij")
        ox = (gx / (cfg.mvroma_grid_w - 1) * (w - 1)).astype(np.float32)
        oy = (gy / (cfg.mvroma_grid_h - 1) * (h - 1)).astype(np.float32)
        return ox, oy

    args = Namespace(use_dinov2=True, train_until_16x=False, train_refiner=False,
                     train_all_model=False, num_cluster=512)
    model_cfg = ModelConfig()
    model_cfg.num_cluster = 512
    model, model_cfg = build_our_model(args, model_cfg, use_dinov2=True)
    model.load_state_dict(torch.load(cfg.mvroma_weights, map_location="cpu"), strict=False)
    model.eval().to(cfg.device)
    prematch = build_prematch_model(model_name="ufm", device=cfg.device)

    p.mvroma.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(p.dense_matches), "w") as fd:
        total_written = 0
        for src_ix, src in enumerate(srcs):
            sw, sh = size(src)
            src_ox, src_oy = source_grid(sw, sh)
            targets = sorted(pairs_by_src[src])
            for k in range(0, len(targets), cfg.mvroma_chunk):
                chunk = targets[k:k + cfg.mvroma_chunk]
                data = {"query_img_path": str(p.images / src),
                        "ref_img_paths": [str(p.images / t) for t in chunk]}
                with torch.inference_mode():
                    out = run_model_test(
                        model, data, coarse_res_hw=(560, 560),
                        target_res_hw=(cfg.mvroma_grid_h, cfg.mvroma_grid_w),
                        prematch_model=prematch, prematch_model_name="ufm",
                        upsample_preds=True, num_cluster=512, device=cfg.device,
                    )
                first_scale = min(out.keys())
                flow = out[first_scale]["flow"][0].cpu().numpy()
                cert = out[first_scale]["certainty"][0].sigmoid()[:, 0].cpu().numpy()
                cert_max = cert.max(0)
                ys, xs = np.where(cert_max > cfg.roma_cert_thresh)
                if len(ys) == 0:
                    continue
                if len(ys) > cfg.agg_maxkp:
                    if cfg.mvroma_sample_mode == "random":
                        sel = np.random.RandomState(src_ix).choice(len(ys), cfg.agg_maxkp, replace=False)
                    else:
                        sel = score_grid_select_indices(
                            xs,
                            ys,
                            cert_max[ys, xs],
                            cfg.agg_maxkp,
                            sample_grid_h,
                            sample_grid_w,
                            image_h=cfg.mvroma_grid_h,
                            image_w=cfg.mvroma_grid_w,
                        )
                    ys, xs = ys[sel], xs[sel]
                for ti, tgt in enumerate(chunk):
                    tw, th = size(tgt)
                    cc = cert[ti, ys, xs]
                    keep = cc > cfg.roma_cert_thresh
                    if int(keep.sum()) < 16:
                        continue
                    yy, xx = ys[keep], xs[keep]
                    tx = (flow[ti, 0, yy, xx] + 1) / 2 * (cfg.mvroma_grid_w - 1)
                    ty = (flow[ti, 1, yy, xx] + 1) / 2 * (cfg.mvroma_grid_h - 1)
                    inside = (
                        (tx >= 0) & (tx < cfg.mvroma_grid_w) &
                        (ty >= 0) & (ty < cfg.mvroma_grid_h)
                    )
                    if int(inside.sum()) < 16:
                        continue
                    yy, xx, tx, ty, sc = yy[inside], xx[inside], tx[inside], ty[inside], cc[keep][inside]
                    k0 = np.stack([src_ox[yy, xx], src_oy[yy, xx]], 1).astype(np.float32)
                    k1 = np.stack([
                        tx / (cfg.mvroma_grid_w - 1) * (tw - 1),
                        ty / (cfg.mvroma_grid_h - 1) * (th - 1),
                    ], 1).astype(np.float32)
                    group_name = pair_name(src, tgt)
                    if group_name in fd:
                        del fd[group_name]
                    grp = fd.create_group(group_name)
                    grp.create_dataset("keypoints0", data=k0)
                    grp.create_dataset("keypoints1", data=k1)
                    grp.create_dataset("scores", data=sc.astype(np.float32))
                    total_written += 1
            if (src_ix + 1) % 25 == 0 or src_ix + 1 == len(srcs):
                log(f"  MV-RoMa {src_ix + 1}/{len(srcs)} src, written={total_written}")
    log(f"MV-RoMa done: {total_written} dense pair groups -> {p.dense_matches}")


def dms_verify_geometry(k0: Any, k1: Any, scores: Any, cfg: SimpleNamespace) -> dict[str, Any]:
    import cv2
    import numpy as np

    if len(k0) < 8:
        return {
            "raw_matches": int(len(k0)),
            "sampled_matches": int(len(k0)),
            "f_inliers": 0,
            "h_inliers": 0,
            "f_ratio": 0.0,
            "h_ratio": 0.0,
            "h_over_f": 0.0,
        }
    grid_h, grid_w = parse_grid_spec(getattr(cfg, "dms_grid", "8x12"), 8, 12)
    sel = score_grid_select_indices(
        np.asarray(k0)[:, 0],
        np.asarray(k0)[:, 1],
        scores,
        int(cfg.dms_max_matches),
        grid_h,
        grid_w,
    )
    pts0 = np.asarray(k0, dtype=np.float32)[sel]
    pts1 = np.asarray(k1, dtype=np.float32)[sel]
    f_inl = h_inl = 0
    if len(pts0) >= 8:
        try:
            method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
            _F, fm = cv2.findFundamentalMat(
                pts0,
                pts1,
                method,
                float(cfg.dms_ransac_px),
                0.99,
                int(cfg.dms_max_trials),
            )
            if fm is not None:
                f_inl = int(fm.reshape(-1).astype(bool).sum())
        except Exception:
            f_inl = 0
    if len(pts0) >= 4:
        try:
            _H, hm = cv2.findHomography(pts0, pts1, cv2.RANSAC, float(cfg.dms_homography_px))
            if hm is not None:
                h_inl = int(hm.reshape(-1).astype(bool).sum())
        except Exception:
            h_inl = 0
    n = max(1, int(len(pts0)))
    return {
        "raw_matches": int(len(k0)),
        "sampled_matches": int(len(pts0)),
        "f_inliers": f_inl,
        "h_inliers": h_inl,
        "f_ratio": float(f_inl / n),
        "h_ratio": float(h_inl / n),
        "h_over_f": float(h_inl / max(1, f_inl)),
    }


def keep_verified_pair(metrics: dict[str, Any], rel: dict[str, Any], cfg: SimpleNamespace) -> tuple[bool, str]:
    sampled = int(metrics.get("sampled_matches", 0))
    if sampled < int(cfg.dms_min_sampled_matches):
        return False, "too_few_sampled_matches"
    f_inl = int(metrics.get("f_inliers", 0))
    h_inl = int(metrics.get("h_inliers", 0))
    f_ratio = float(metrics.get("f_ratio", 0.0))
    h_ratio = float(metrics.get("h_ratio", 0.0))
    f_pass = f_inl >= int(cfg.dms_min_inliers) and f_ratio >= float(cfg.dms_min_inlier_ratio)
    h_pass = h_inl >= int(cfg.dms_min_inliers) and h_ratio >= float(cfg.dms_min_inlier_ratio)
    rotation_like = h_inl >= int(cfg.dms_min_inliers) and h_inl >= float(cfg.dms_rotation_h_over_f) * max(1, f_inl)
    if bool(rel.get("same_video")):
        if f_pass:
            return True, "same_video_fundamental"
        if h_pass:
            return True, "same_video_rotation_bridge"
        return False, "same_video_geometry_failed"
    if bool(rel.get("cross_direction")):
        if f_pass:
            return True, "cross_direction_fundamental"
        if rotation_like:
            return False, "cross_direction_rotation_like"
        return False, "cross_direction_geometry_failed"
    if f_pass:
        return True, "cross_video_fundamental"
    if h_pass and not rotation_like:
        return True, "cross_video_homography"
    return False, "cross_video_geometry_failed"


def stage_verify_pairs(cfg: SimpleNamespace) -> None:
    mode = getattr(cfg, "pair_verification", "dms")
    action = getattr(cfg, "pair_verification_action", "filter")
    if mode == "off":
        log("skip pair verification: --pair-verification off")
        return

    import h5py
    import numpy as np

    p = cfg_paths(cfg)
    pairs = read_pairs(p.pairs)
    if not pairs:
        raise SystemExit("pair verification needs a non-empty pair list")
    shutil.copy2(p.pairs, p.pairs_before_verification)
    names, groups = list_images(p.images)
    folder_of = {name: folder for folder, rel in groups.items() for name in rel}
    directions = sequence_directions(cfg, groups)
    kept: list[tuple[str, str]] = []
    records: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    missing = 0
    with h5py.File(str(p.dense_matches), "r") as fd:
        for a, b in pairs:
            name = pair_name(a, b)
            rev = False
            if name not in fd:
                name = pair_name(b, a)
                rev = True
            rel = pair_relation(a, b, folder_of, directions)
            if name not in fd:
                missing += 1
                reason = "missing_dense_matches"
                keep = False
                metrics = {"raw_matches": 0, "sampled_matches": 0}
            else:
                grp = fd[name]
                k0 = grp["keypoints0"].__array__().astype(np.float32)
                k1 = grp["keypoints1"].__array__().astype(np.float32)
                if rev:
                    k0, k1 = k1, k0
                scores = grp["scores"].__array__().astype(np.float32) if "scores" in grp else np.ones((len(k0),), np.float32)
                metrics = dms_verify_geometry(k0, k1, scores, cfg)
                keep, reason = keep_verified_pair(metrics, rel, cfg)
            if action == "report":
                keep = True
            if keep:
                kept.append((a, b))
            reason_counts[reason] += 1
            records.append({
                "pair": [a, b],
                "keep": bool(keep),
                "reason": reason,
                "relation": rel,
                **metrics,
            })
    write_pairs(p.pairs_verified, kept)
    if action == "filter":
        write_pairs(p.pairs, kept)
    report = {
        "mode": mode,
        "action": action,
        "used_for": "summarized two-view edge verification only; full dense matches remain in H5 for aggregation/tracks",
        "pairs_before": len(pairs),
        "pairs_after": len(kept),
        "missing_dense_match_groups": missing,
        "directions": directions,
        "thresholds": {
            "dms_max_matches": cfg.dms_max_matches,
            "dms_grid": cfg.dms_grid,
            "dms_min_sampled_matches": cfg.dms_min_sampled_matches,
            "dms_min_inliers": cfg.dms_min_inliers,
            "dms_min_inlier_ratio": cfg.dms_min_inlier_ratio,
            "dms_rotation_h_over_f": cfg.dms_rotation_h_over_f,
        },
        "reason_counts": dict(reason_counts),
        "records": records,
    }
    write_json(p.pair_verification_report, report)
    log(f"pair verification {action}: kept {len(kept)}/{len(pairs)} missing_dense={missing}")


def apply_pair_degree_cap(pairs: list[tuple[str, str]], groups: dict[str, list[str]],
                          directions: dict[str, str] | None = None, total_cap: int = 0,
                          intra_cap: int = 0, cross_direction_cap: int = 0) -> list[tuple[str, str]]:
    if total_cap <= 0 and intra_cap <= 0 and cross_direction_cap <= 0:
        return pairs
    folder_of = {name: folder for folder, names in groups.items() for name in names}
    directions = directions or {}
    total_counts: Counter[str] = Counter()
    intra_counts: Counter[str] = Counter()
    cross_direction_counts: Counter[str] = Counter()
    kept: list[tuple[str, str]] = []
    for a, b in pairs:
        rel = pair_relation(a, b, folder_of, directions)
        if total_cap > 0 and (total_counts[a] >= total_cap or total_counts[b] >= total_cap):
            continue
        is_cross_direction = bool(rel["cross_direction"])
        if is_cross_direction and cross_direction_cap > 0 and (
            cross_direction_counts[a] >= cross_direction_cap or cross_direction_counts[b] >= cross_direction_cap
        ):
            continue
        if not is_cross_direction and intra_cap > 0 and (intra_counts[a] >= intra_cap or intra_counts[b] >= intra_cap):
            continue
        kept.append((a, b))
        total_counts[a] += 1
        total_counts[b] += 1
        if is_cross_direction:
            cross_direction_counts[a] += 1
            cross_direction_counts[b] += 1
        else:
            intra_counts[a] += 1
            intra_counts[b] += 1
    log(
        f"degree cap kept {len(kept)}/{len(pairs)} pairs "
        f"intra_cap={intra_cap} cross_direction_cap={cross_direction_cap} total_cap={total_cap}"
    )
    return kept


def unique_matches_by_score(matches: Any, scores: Any) -> tuple[Any, Any]:
    import numpy as np

    if len(matches) == 0:
        return matches.astype(np.int32), scores.astype(np.float32)
    order = np.argsort(-scores)
    used0: set[int] = set()
    used1: set[int] = set()
    keep: list[int] = []
    for idx in order.tolist():
        i0, i1 = int(matches[idx, 0]), int(matches[idx, 1])
        if i0 in used0 or i1 in used1:
            continue
        used0.add(i0)
        used1.add(i1)
        keep.append(idx)
    keep_arr = np.asarray(keep, dtype=np.int64)
    out_matches = matches[keep_arr].astype(np.int32)
    out_scores = scores[keep_arr].astype(np.float32)
    if len(out_matches):
        order0 = np.argsort(out_matches[:, 0])
        out_matches = out_matches[order0]
        out_scores = out_scores[order0]
    return out_matches, out_scores


def matches_to_matches0(matches: Any, scores: Any) -> tuple[Any, Any]:
    import numpy as np

    if len(matches) == 0:
        return np.full((0,), -1, dtype=np.int32), np.zeros((0,), dtype=np.float16)
    size = int(matches[:, 0].max()) + 1
    matches0 = np.full((size,), -1, dtype=np.int32)
    scores0 = np.zeros((size,), dtype=np.float16)
    matches0[matches[:, 0].astype(np.int64)] = matches[:, 1].astype(np.int32)
    scores0[matches[:, 0].astype(np.int64)] = scores.astype(np.float16)
    return matches0, scores0


def assign_matches_cached(pairs: list[tuple[str, str]], match_path: str | Path,
                          keypoints: dict[str, Any], max_error: float) -> dict[str, Any]:
    import h5py
    import numpy as np
    from scipy.spatial import cKDTree

    trees: dict[str, Any] = {}
    for name, pts in keypoints.items():
        arr = np.asarray(pts, dtype=np.float32)
        if len(arr) == 0:
            trees[name] = None
        else:
            trees[name] = cKDTree(arr)

    def query(name: str, pts: Any) -> Any:
        arr = np.asarray(pts, dtype=np.float32)
        tree = trees.get(name)
        base = np.asarray(keypoints.get(name, []), dtype=np.float32)
        if tree is None or len(base) == 0 or len(arr) == 0:
            return np.full((len(arr),), -1, dtype=np.int32)
        dist, ids = tree.query(arr, distance_upper_bound=float(max_error))
        ids = ids.astype(np.int32)
        bad = (~np.isfinite(dist)) | (dist > float(max_error)) | (ids >= len(base))
        ids[bad] = -1
        return ids

    stats = {"pairs": 0, "assigned_pairs": 0, "matches": 0, "skipped_missing_group": 0}
    with h5py.File(str(match_path), "a") as fd:
        for n0, n1 in pairs:
            name = pair_name(n0, n1)
            g0, g1 = n0, n1
            if name not in fd:
                name = pair_name(n1, n0)
                g0, g1 = n1, n0
            if name not in fd:
                stats["skipped_missing_group"] += 1
                continue
            grp = fd[name]
            k0 = grp["keypoints0"].__array__().astype(np.float32)
            k1 = grp["keypoints1"].__array__().astype(np.float32)
            scores = grp["scores"].__array__().astype(np.float32) if "scores" in grp else np.ones((len(k0),), np.float32)
            ids0 = query(g0, k0)
            ids1 = query(g1, k1)
            valid = (ids0 >= 0) & (ids1 >= 0)
            matches = np.stack([ids0[valid], ids1[valid]], axis=1).astype(np.int32) if int(valid.sum()) else np.empty((0, 2), np.int32)
            mscores = scores[valid].astype(np.float32) if int(valid.sum()) else np.empty((0,), np.float32)
            matches, mscores = unique_matches_by_score(matches, mscores)
            matches0, scores0 = matches_to_matches0(matches, mscores)
            for key in ("matches0", "matching_scores0"):
                if key in grp:
                    del grp[key]
            grp.create_dataset("matches0", data=matches0)
            grp.create_dataset("matching_scores0", data=scores0)
            stats["pairs"] += 1
            if len(matches):
                stats["assigned_pairs"] += 1
                stats["matches"] += int(len(matches))
    return stats


def _stage_aggregate_unlocked(cfg: SimpleNamespace) -> None:
    import h5py
    from hloc.match_dense import aggregate_matches, load_keypoints

    p = cfg_paths(cfg)
    names, groups = list_images(p.images)
    directions = sequence_directions(cfg, groups)
    all_pairs = [tuple(line.split()) for line in p.pairs.read_text().splitlines() if len(line.split()) == 2]
    with h5py.File(str(p.dense_matches), "a") as fd:
        present = [(a, b) for a, b in all_pairs if pair_name(a, b) in fd or pair_name(b, a) in fd]
        if getattr(cfg, "repair_dense_h5", False):
            to_delete: list[str] = []
            def visit(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Group) and ("matches0" in obj or "matching_scores0" in obj):
                    to_delete.append(name)
            fd.visititems(visit)
            for name in to_delete:
                for key in ("matches0", "matching_scores0"):
                    if key in fd[name]:
                        del fd[name][key]
            log(f"repair dense H5: cleared stale match assignments from {len(to_delete)} groups")
    skipped = len(all_pairs) - len(present)
    present = apply_pair_degree_cap(
        present,
        groups,
        directions,
        total_cap=cfg.agg_pair_degree_cap,
        intra_cap=cfg.agg_intra_degree_cap,
        cross_direction_cap=cfg.agg_cross_direction_degree_cap,
    )
    motion_roles = load_motion_roles(cfg)
    present, motion_filter_stats = filter_pairs_by_motion_roles(
        present,
        motion_roles,
        exclude_non_parallax=bool(getattr(cfg, "exclude_rotation_from_triangulation", False)),
    )
    write_pairs(p.aggregate_pairs, present)
    if p.features.exists():
        p.features.unlink()
    log(f"aggregate {len(present)}/{len(all_pairs)} matched pairs skipped_missing={skipped} maxkp={cfg.agg_maxkp}")
    conf = {"max_error": cfg.agg_max_error, "cell_size": cfg.agg_cell_size}
    cpdict, bindict = load_keypoints(conf, [], quantize=None)
    cpdict = aggregate_matches(
        conf,
        present,
        str(p.dense_matches),
        feature_path=str(p.features),
        required_queries=set(name for pair in present for name in pair),
        max_kps=cfg.agg_maxkp,
        cpdict=cpdict,
        bindict=bindict,
    )
    stats = assign_matches_cached(present, str(p.dense_matches), cpdict, max_error=conf["max_error"])
    _fsync_file(p.dense_matches)
    stats["motion_pair_filter"] = motion_filter_stats
    write_json(p.mvroma / "assign_matches_cached_summary.json", stats)
    log(f"aggregate done -> {p.features}; assigned={stats['assigned_pairs']} pairs matches={stats['matches']}")


def stage_aggregate(cfg: SimpleNamespace) -> None:
    p = cfg_paths(cfg)
    with mvroma_stage_lock(mvroma_lock_path(p.dense_matches)):
        _stage_aggregate_unlocked(cfg)


def stage_db(cfg: SimpleNamespace) -> None:
    import h5py
    import numpy as np
    import pycolmap

    p = cfg_paths(cfg)
    if p.tmp_database.exists():
        p.tmp_database.unlink()
    if p.database.exists():
        p.database.unlink()
    log(f"create COLMAP legacy DB on local/ext4-safe path -> {p.tmp_database}")
    open_pycolmap_database(pycolmap, p.tmp_database).close()

    def h5_names(path: Path) -> list[str]:
        names: list[str] = []
        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Group) and "keypoints" in obj:
                names.append(name)
        with h5py.File(str(path), "r") as fd:
            fd.visititems(visit)
        return sorted(names)

    def pair_name(n0: str, n1: str) -> str:
        return "/".join((n0.replace("/", "-"), n1.replace("/", "-")))

    def get_matches(fd: h5py.File, n0: str, n1: str) -> np.ndarray:
        name, rev = pair_name(n0, n1), False
        if name not in fd:
            name, rev = pair_name(n1, n0), True
        m = fd[name]["matches0"].__array__()
        idx = np.where(m != -1)[0]
        matches = np.stack([idx, m[idx]], -1)
        if rev:
            matches = matches[:, ::-1]
        return matches.astype(np.uint32)

    image_list = h5_names(p.features)
    log(f"import images PER_FOLDER n={len(image_list)}")
    pycolmap.import_images(
        database_path=str(p.tmp_database),
        image_path=str(p.images),
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
        image_list=image_list,
        options=pycolmap.ImageReaderOptions(camera_model=cfg.camera_model),
    )
    db = open_pycolmap_database(pycolmap, p.tmp_database)
    for cam in db.read_all_cameras():
        model, params = focal_for_resolution(cam.width, cam.height, cfg)
        cam.model = getattr(pycolmap.CameraModelId, model)
        cam.params = list(params)
        cam.has_prior_focal_length = True
        db.update_camera(cam)
        log(f"camera {cam.camera_id} {cam.width}x{cam.height} -> {model} {params}")
    image_ids = {img.name: img.image_id for img in db.read_all_images()}
    log(f"images in db: {len(image_ids)}")
    with h5py.File(str(p.features), "r") as fd:
        for name, iid in image_ids.items():
            db.write_keypoints(iid, fd[name]["keypoints"].__array__().astype(np.float32) + 0.5)
    pairs = [tuple(line.split()) for line in p.aggregate_pairs.read_text().splitlines() if len(line.split()) == 2]
    written = skipped = 0
    with h5py.File(str(p.dense_matches), "r") as fd:
        done: set[tuple[int, int]] = set()
        for n0, n1 in pairs:
            if n0 not in image_ids or n1 not in image_ids:
                skipped += 1
                continue
            i0, i1 = image_ids[n0], image_ids[n1]
            if (i0, i1) in done or (i1, i0) in done:
                continue
            try:
                matches = get_matches(fd, n0, n1)
            except KeyError:
                skipped += 1
                continue
            if len(matches):
                db.write_matches(i0, i1, matches)
                written += 1
            done.add((i0, i1))
    db.close()
    log(f"write matches done written={written} skipped={skipped}")
    pycolmap.verify_matches(
        str(p.tmp_database),
        str(p.aggregate_pairs),
        options=dict(ransac=dict(max_num_trials=cfg.verify_max_trials, min_inlier_ratio=cfg.verify_min_inlier_ratio)),
    )
    shutil.copy2(p.tmp_database, p.database)
    log(f"DB persisted -> {p.database}")


def resolve_lfoe_command(cfg: SimpleNamespace) -> str:
    lfoe_cmd = DEFAULT_LFOE_GLOMAP if Path(DEFAULT_LFOE_GLOMAP).exists() else cfg.glomap_command
    if Path(lfoe_cmd).name != "glomap_filter":
        return lfoe_cmd
    missing_libs = unresolved_shared_libs(lfoe_cmd)
    if missing_libs:
        log("WARNING: glomap_filter shared libraries are unresolved; LFOE fallback is unavailable.")
        log("WARNING: missing shared libraries: " + "; ".join(missing_libs))
        return ""
    return lfoe_cmd


def mpsfm_flat_name(rel_name: str) -> str:
    path = Path(rel_name)
    prefix = sanitize_stem(path.parent) if str(path.parent) not in {"", "."} else "image"
    return f"{prefix}__{path.name}"


def symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def prepare_mpsfm_images(cfg: SimpleNamespace) -> dict[str, str]:
    p = cfg_paths(cfg)
    names, _groups = list_images(p.images)
    if p.mpsfm_images.exists():
        shutil.rmtree(p.mpsfm_images)
    p.mpsfm_images.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for rel in names:
        flat = mpsfm_flat_name(rel)
        if flat in used:
            raise ValueError(f"duplicate MP-SfM flat image name {flat} from {rel}")
        used.add(flat)
        src = p.images / rel
        symlink_or_copy(src, p.mpsfm_images / flat)
        # Downstream color/snap stages read images relative to cfg_paths(cfg).images
        # using names stored by the reconstruction. Keep flat aliases there too.
        symlink_or_copy(src, p.images / flat)
        mapping[rel] = flat
    write_json(p.mpsfm_image_map, {"nested_to_flat": mapping, "flat_to_nested": {v: k for k, v in mapping.items()}})
    return mapping


def write_mpsfm_intrinsics(cfg: SimpleNamespace, name_map: dict[str, str]) -> None:
    p = cfg_paths(cfg)
    by_camera: dict[tuple[int, int], list[str]] = defaultdict(list)
    sizes: dict[tuple[int, int], tuple[str, list[float]]] = {}
    for name, flat_name in name_map.items():
        w, h = image_size(p.images / name)
        key = (w, h)
        by_camera[key].append(flat_name)
        if key not in sizes:
            sizes[key] = focal_for_resolution(w, h, cfg)
    lines: list[str] = []
    for cid, key in enumerate(sorted(by_camera), 1):
        model, params = sizes[key]
        if model == "PINHOLE" and len(params) >= 4:
            fx, fy, cx, cy = params[:4]
        elif len(params) >= 3:
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:
            raise ValueError(f"cannot convert camera params to MP-SfM intrinsics: {model} {params}")
        lines.append(f"{cid}:")
        lines.append(f"  params: [{float(fx)}, {float(fy)}, {float(cx)}, {float(cy)}]")
        lines.append("  images:")
        for name in by_camera[key]:
            lines.append(f"    - {json.dumps(name)}")
    p.mpsfm_intrinsics.parent.mkdir(parents=True, exist_ok=True)
    p.mpsfm_intrinsics.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_mpsfm_conf_arg(cfg: SimpleNamespace) -> str:
    p = cfg_paths(cfg)
    if getattr(cfg, "mpsfm_config_yaml", ""):
        path = as_path(cfg.mpsfm_config_yaml)
        return str(path.with_suffix("")) if path.suffix == ".yaml" else str(path)
    conf = str(getattr(cfg, "mpsfm_conf", "sp-mast3r-dense"))
    if conf in {"roma_m3dv2-large", "roma_m3dv2_large", "sp-roma-dense_m3dv2-large"}:
        repo = as_path(getattr(cfg, "mpsfm_repo", DEFAULT_MPSFM_REPO)) if hasattr(cfg, "mpsfm_repo") else Path(DEFAULT_MPSFM_REPO)
        default_conf = repo / "configs" / "defaults" / "m3dv2-large"
        p.mpsfm_custom_conf.parent.mkdir(parents=True, exist_ok=True)
        p.mpsfm_custom_conf.write_text(
            "\n".join([
                "# Generated by build_localizable_map_core.py",
                "defaults:",
                f"  - {default_conf}",
                "extractors:",
                "  matcher: roma_outdoor",
                "matches_mode: sparse+dense",
                "",
            ]),
            encoding="utf-8",
        )
        return str(p.mpsfm_custom_conf.with_suffix(""))
    return conf


def find_colmap_model_dir(root: Path) -> Path | None:
    if all((root / name).exists() for name in ("cameras.bin", "images.bin", "points3D.bin")):
        return root
    for cameras in sorted(root.rglob("cameras.bin")) if root.exists() else []:
        candidate = cameras.parent
        if all((candidate / name).exists() for name in ("cameras.bin", "images.bin", "points3D.bin")):
            return candidate
    return None


def copy_model_to_zero(src_model: Path, dst_root: Path) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst0 = dst_root / "0"
    dst0.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_model, dst0)


def run_mpsfm_backend(cfg: SimpleNamespace, fatal: bool = True) -> dict[str, Any]:
    p = cfg_paths(cfg)
    repo = as_path(getattr(cfg, "mpsfm_repo", DEFAULT_MPSFM_REPO))
    reconstruct = repo / "reconstruct.py"
    summary: dict[str, Any] = {
        "path": str(p.model_mpsfm),
        "repo": str(repo),
        "conf": getattr(cfg, "mpsfm_conf", "sp-mast3r-dense"),
    }
    if not reconstruct.exists():
        summary.update({"exists": False, "error": f"missing MP-SfM reconstruct.py: {reconstruct}"})
        if fatal:
            raise FileNotFoundError(summary["error"])
        return summary
    if p.model_mpsfm0.exists() and getattr(cfg, "resume", False) and not getattr(cfg, "overwrite", False):
        summary.update(glomap_summary(p.model_mpsfm0))
        summary["reused"] = True
        return summary
    if p.model_mpsfm.exists():
        shutil.rmtree(p.model_mpsfm)
    if p.mpsfm_data.exists() and getattr(cfg, "overwrite", False):
        shutil.rmtree(p.mpsfm_data)
    p.mpsfm_data.mkdir(parents=True, exist_ok=True)
    p.mpsfm_cache.mkdir(parents=True, exist_ok=True)
    name_map = prepare_mpsfm_images(cfg)
    write_mpsfm_intrinsics(cfg, name_map)
    conf_arg = resolve_mpsfm_conf_arg(cfg)
    cmd = [
        getattr(cfg, "python_mpsfm", DEFAULT_PY_MPSFM),
        str(reconstruct),
        "--conf", conf_arg,
        "--data_dir", str(p.mpsfm_data),
        "--intrinsics_pth", str(p.mpsfm_intrinsics),
        "--images_dir", str(p.mpsfm_images),
        "--cache_dir", str(p.mpsfm_cache),
        "--verbose", str(int(getattr(cfg, "mpsfm_verbose", 0))),
    ]
    if bool(getattr(cfg, "mpsfm_extract", True)):
        extract_items = [x.strip() for x in str(getattr(cfg, "mpsfm_extract_list", "")).split(",") if x.strip()]
        cmd += ["--extract", *extract_items]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        run_cmd(cmd, cwd=repo, env=env, dry_run=cfg.dry_run)
    except subprocess.CalledProcessError as exc:
        summary.update({"exists": False, "command_error": str(exc), "returncode": int(exc.returncode), "command": cmd})
        if fatal:
            raise
        return summary
    if cfg.dry_run:
        return summary
    src_model = find_colmap_model_dir(p.mpsfm_data / "sfm_outputs") or find_colmap_model_dir(p.mpsfm_data)
    if src_model is None:
        summary.update({"exists": False, "error": f"MP-SfM produced no COLMAP model under {p.mpsfm_data}"})
        if fatal:
            raise SystemExit(summary["error"])
        return summary
    copy_model_to_zero(src_model, p.model_mpsfm)
    summary.update(glomap_summary(p.model_mpsfm0))
    summary.update({
        "exists": bool(summary.get("exists", False)),
        "source_model": str(src_model),
        "command": cmd,
        "conf_arg": conf_arg,
        "intrinsics": str(p.mpsfm_intrinsics),
    })
    return summary


def glomap_model_dir(root: Path) -> Path:
    if (root / "0").exists():
        return root / "0"
    candidates = sorted([x for x in root.iterdir() if x.is_dir()]) if root.exists() else []
    return candidates[0] if candidates else root / "0"


def mapper_needs_lfoe(summary: dict[str, Any], cfg: SimpleNamespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not bool(summary.get("exists")):
        reasons.append("missing_model")
    if "registered_images" in summary:
        manifest_total = read_json(cfg_paths(cfg).manifest).get("total_frames", 0) if cfg_paths(cfg).manifest.exists() else 0
        reg = int(summary.get("registered_images", 0))
        ratio = reg / max(1, int(manifest_total))
        if reg < int(cfg.gate_min_registered_images):
            reasons.append(f"registered_images={reg}")
        if ratio < float(cfg.gate_min_registered_ratio):
            reasons.append(f"registered_ratio={ratio:.3f}")
    return bool(reasons), reasons


def mapper_quality_score(summary: dict[str, Any]) -> tuple[int, int, float]:
    return (
        int(summary.get("registered_images", 0)),
        int(summary.get("points3D", 0)),
        -float(summary.get("mean_reprojection_error") or 0.0),
    )


def build_glomap_mapper_cmd(cfg: SimpleNamespace, glomap_cmd: str, output_path: Path) -> list[str]:
    p = cfg_paths(cfg)
    cmd = [
        glomap_cmd,
        "mapper",
        "--database_path", str(p.database),
        "--image_path", str(p.images),
        "--output_path", str(output_path),
    ]
    if cfg.skip_bundle_adjustment:
        cmd += ["--skip_bundle_adjustment", "1"]
    if cfg.skip_retriangulation:
        cmd += ["--skip_retriangulation", "1"]
    if getattr(cfg, "optimize_intrinsics", None) is not None:
        cmd += ["--BundleAdjustment.optimize_intrinsics", str(int(cfg.optimize_intrinsics))]
    if cfg.optimize_principal_point is not None:
        cmd += ["--BundleAdjustment.optimize_principal_point", str(int(cfg.optimize_principal_point))]
    if cfg.max_num_tracks > 0:
        cmd += ["--TrackEstablishment.max_num_tracks", str(cfg.max_num_tracks)]
    min_views = int(getattr(cfg, "min_num_view_per_track", 0) or 0)
    if min_views > 0:
        cmd += ["--TrackEstablishment.min_num_view_per_track", str(min_views)]
    min_angle = float(getattr(cfg, "min_triangulation_angle", 0.0) or 0.0)
    if min_angle > 0:
        cmd += ["--Thresholds.min_triangulation_angle", str(min_angle)]
    return cmd


def glomap_run_cwd(glomap_cmd: str) -> Path | None:
    if Path(glomap_cmd).name == "glomap_filter":
        return Path(glomap_cmd).resolve().parent
    return None


def run_dense_mvroma_triangulation(cfg: SimpleNamespace, reference_model: Path, output_path: Path) -> dict[str, Any]:
    import pycolmap
    from hloc import triangulation

    p = cfg_paths(cfg)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.extract_colors = False
    opts.ba_refine_focal_length = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    if hasattr(opts, "ba_refine_sensor_from_rig"):
        opts.ba_refine_sensor_from_rig = False
    if hasattr(opts, "fix_existing_frames"):
        opts.fix_existing_frames = True
    if hasattr(opts, "mapper") and hasattr(opts.mapper, "fix_existing_frames"):
        opts.mapper.fix_existing_frames = True
    dense_max_error = float(getattr(cfg, "dense_tri_max_reproj_error", 2.0) or 2.0)
    dense_min_angle = float(getattr(cfg, "dense_tri_min_angle", 0.5) or 0.5)
    if hasattr(opts, "mapper"):
        opts.mapper.filter_max_reproj_error = dense_max_error
        opts.mapper.filter_min_tri_angle = dense_min_angle
        opts.mapper.ba_local_min_tri_angle = dense_min_angle
    opts.triangulation.ignore_two_view_tracks = not bool(getattr(cfg, "dense_tri_include_two_view_tracks", True))
    opts.triangulation.min_angle = dense_min_angle
    opts.triangulation.merge_max_reproj_error = dense_max_error
    opts.triangulation.complete_max_reproj_error = dense_max_error
    opts.triangulation.create_max_angle_error = dense_max_error
    opts.triangulation.continue_max_angle_error = dense_max_error
    log(
        "dense MV-RoMA triangulation: "
        f"ref={reference_model} out={output_path} min_angle={dense_min_angle} max_reproj={dense_max_error} "
        f"two_view={not opts.triangulation.ignore_two_view_tracks}"
    )
    rec = triangulation.main(
        output_path,
        reference_model,
        p.images,
        p.aggregate_pairs,
        p.features,
        p.dense_matches,
        skip_geometric_verification=bool(getattr(cfg, "dense_tri_skip_geometric_verification", False)),
        mapper_options=opts,
    )
    summary = glomap_summary(output_path)
    summary.update({
        "source": "fixed_pose_mvroma_triangulation",
        "reference_model": str(reference_model),
        "triangulation_summary": rec.summary(),
    })
    return summary


def stage_glomap(cfg: SimpleNamespace) -> None:
    p = cfg_paths(cfg)
    backend = str(getattr(cfg, "backend", "glomap"))
    if getattr(cfg, "use_lfoe", False) and backend == "glomap":
        backend = "lfoe"
    for path in (p.model, p.model_standard, p.model_lfoe, p.model_dense, p.model_mpsfm):
        if path.exists() and cfg.overwrite:
            shutil.rmtree(path)
    if backend == "mpsfm":
        diagnostics: dict[str, Any] = {"backend": backend, "mode": "mpsfm", "selected": "", "runs": {}}
        mpsfm_summary = run_mpsfm_backend(cfg, fatal=True)
        diagnostics["runs"]["mpsfm"] = {"path": str(p.model_mpsfm), "summary": mpsfm_summary}
        if cfg.dry_run:
            diagnostics["selected"] = "mpsfm"
            write_json(p.mapper_diagnostics, diagnostics)
            write_json(p.backend_comparison, diagnostics)
            return
        if p.model.exists():
            shutil.rmtree(p.model)
        shutil.copytree(p.model_mpsfm, p.model)
        diagnostics["selected"] = "mpsfm"
        diagnostics["final_summary"] = glomap_summary(glomap_model_dir(p.model))
        write_json(p.mapper_diagnostics, diagnostics)
        write_json(p.backend_comparison, diagnostics)
        return

    if backend == "lfoe":
        mode = "always"
    elif backend == "compare":
        mode = "diagnostic"
    else:
        mode = getattr(cfg, "lfoe_mode", "diagnostic")
    diagnostics = {"backend": backend, "mode": mode, "selected": "", "runs": {}}

    standard_summary: dict[str, Any] = {}
    if mode != "always":
        p.model_standard.mkdir(parents=True, exist_ok=True)
        try:
            run_cmd(build_glomap_mapper_cmd(cfg, cfg.glomap_command, p.model_standard), dry_run=cfg.dry_run)
            if cfg.dry_run:
                return
            standard_summary = glomap_summary(glomap_model_dir(p.model_standard))
            needs_lfoe, reasons = mapper_needs_lfoe(standard_summary, cfg)
            if backend == "compare":
                needs_lfoe = True
                reasons = [*reasons, "compare_backend"]
        except subprocess.CalledProcessError as exc:
            if mode == "off":
                raise
            standard_summary = {
                "exists": False,
                "command_error": str(exc),
                "returncode": int(exc.returncode),
            }
            needs_lfoe = True
            reasons = [f"standard_mapper_failed:{exc.returncode}"]
            if not getattr(cfg, "skip_retriangulation", False):
                log("standard GLOMAP failed; retrying with --skip_retriangulation 1 while keeping bundle adjustment enabled")
                retry_cfg = SimpleNamespace(**vars(cfg))
                retry_cfg.skip_retriangulation = True
                if p.model_standard.exists():
                    shutil.rmtree(p.model_standard)
                p.model_standard.mkdir(parents=True, exist_ok=True)
                try:
                    run_cmd(build_glomap_mapper_cmd(retry_cfg, retry_cfg.glomap_command, p.model_standard), dry_run=cfg.dry_run)
                    standard_summary = glomap_summary(glomap_model_dir(p.model_standard))
                    standard_summary["retried_without_retriangulation"] = True
                    needs_lfoe, reasons = mapper_needs_lfoe(standard_summary, cfg)
                    reasons.append("standard_retried_without_retriangulation")
                    if backend == "compare":
                        needs_lfoe = True
                        reasons.append("compare_backend")
                except subprocess.CalledProcessError as retry_exc:
                    standard_summary["retry_without_retriangulation_error"] = str(retry_exc)
                    standard_summary["retry_without_retriangulation_returncode"] = int(retry_exc.returncode)
        diagnostics["runs"]["standard"] = {"path": str(p.model_standard), "summary": standard_summary, "fallback_reasons": reasons}
    else:
        needs_lfoe = True
        reasons = ["forced_lfoe"]

    selected = p.model_standard
    selected_name = "standard"
    lfoe_summary: dict[str, Any] = {}
    if mode in {"diagnostic", "always"} and needs_lfoe:
        lfoe_cmd = resolve_lfoe_command(cfg)
        if not lfoe_cmd:
            if mode == "always":
                raise SystemExit(f"--use-lfoe requested but usable glomap_filter was not found: {DEFAULT_LFOE_GLOMAP}")
            log("LFOE fallback requested by diagnostics, but glomap_filter is unavailable; keeping standard GLOMAP output.")
            if not bool(standard_summary.get("exists")):
                write_json(p.mapper_diagnostics, diagnostics)
                raise SystemExit("standard GLOMAP failed and LFOE fallback is unavailable")
        else:
            p.model_lfoe.mkdir(parents=True, exist_ok=True)
            try:
                run_cmd(build_glomap_mapper_cmd(cfg, lfoe_cmd, p.model_lfoe), cwd=glomap_run_cwd(lfoe_cmd), dry_run=cfg.dry_run)
                lfoe_summary = glomap_summary(glomap_model_dir(p.model_lfoe))
                diagnostics["runs"]["lfoe"] = {"path": str(p.model_lfoe), "summary": lfoe_summary, "trigger_reasons": reasons}
                if mode == "always" or not standard_summary or mapper_quality_score(lfoe_summary) >= mapper_quality_score(standard_summary):
                    selected = p.model_lfoe
                    selected_name = "lfoe"
            except subprocess.CalledProcessError as exc:
                diagnostics["runs"]["lfoe"] = {
                    "path": str(p.model_lfoe),
                    "summary": {"exists": False, "command_error": str(exc), "returncode": int(exc.returncode)},
                    "trigger_reasons": reasons,
                }
                if mode == "always" or not bool(standard_summary.get("exists")):
                    write_json(p.mapper_diagnostics, diagnostics)
                    raise

    if backend == "compare":
        mpsfm_summary = run_mpsfm_backend(cfg, fatal=False)
        diagnostics["runs"]["mpsfm"] = {"path": str(p.model_mpsfm), "summary": mpsfm_summary, "trigger_reasons": ["compare_backend"]}
        if bool(mpsfm_summary.get("exists")) and mapper_quality_score(mpsfm_summary) >= mapper_quality_score(glomap_summary(glomap_model_dir(selected))):
            selected = p.model_mpsfm
            selected_name = "mpsfm"

    if selected.exists():
        if p.model.exists():
            shutil.rmtree(p.model)
        shutil.copytree(selected, p.model)
    p.model.mkdir(parents=True, exist_ok=True)
    if bool(getattr(cfg, "dense_map_triangulation", False)) and not cfg.dry_run:
        dense_summary = run_dense_mvroma_triangulation(cfg, glomap_model_dir(p.model), p.model_dense0)
        diagnostics["runs"]["dense_mvroma_triangulation"] = {
            "path": str(p.model_dense0),
            "summary": dense_summary,
            "reference": selected_name,
        }
        if mapper_quality_score(dense_summary) >= mapper_quality_score(glomap_summary(glomap_model_dir(p.model))):
            if p.model.exists():
                shutil.rmtree(p.model)
            shutil.copytree(p.model_dense, p.model)
            selected_name = "dense_mvroma_triangulation"
    diagnostics["selected"] = selected_name
    diagnostics["final_summary"] = glomap_summary(glomap_model_dir(p.model))
    write_json(p.mapper_diagnostics, diagnostics)
    write_json(p.backend_comparison, diagnostics)
    if not p.model0.exists():
        candidates = sorted([x for x in p.model.iterdir() if x.is_dir()])
        if candidates:
            log(f"GLOMAP output has no 0/ directory; first candidate: {candidates[0]}")
        else:
            raise SystemExit(f"GLOMAP produced no model directory under {p.model}")


def stage_color(cfg: SimpleNamespace) -> None:
    import numpy as np
    import pycolmap

    p = cfg_paths(cfg)
    rec = pycolmap.Reconstruction()
    rec.read(str(p.model0))
    log(f"color export: images={len(rec.images)} points={len(rec.points3D)}")
    rec.extract_colors_for_all_images(str(p.images))
    xyz, rgb = [], []
    for point in rec.points3D.values():
        color = np.asarray(point.color)
        if int(color.sum()) == 0:
            continue
        xyz.append(point.xyz)
        rgb.append(color)
    if not xyz:
        raise SystemExit("no colored points")
    xyz = np.asarray(xyz, np.float32)
    rgb = np.asarray(rgb, np.uint8)
    med = np.median(xyz, 0)
    dist = np.linalg.norm(xyz - med, axis=1)
    scale = np.linalg.norm(np.percentile(xyz, 99, 0) - np.percentile(xyz, 1, 0))
    keep = dist < cfg.color_outlier_scale * max(scale, 1e-9)
    xyz, rgb = xyz[keep], rgb[keep]
    p.rgb_ply.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    arr = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["r"], arr["g"], arr["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(p.rgb_ply, "wb") as f:
        f.write(header.encode())
        f.write(arr.tobytes())
    log(f"RGB PLY -> {p.rgb_ply} points={len(xyz)}")


def load_rgb(path: Path):
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def prepare_deploy_imports(cfg: SimpleNamespace) -> None:
    repo = as_path(cfg.template_repo)
    sys.path.insert(0, str(repo / "deploy"))
    sys.path.insert(0, str(repo / "scripts"))


def stage_snap(cfg: SimpleNamespace) -> None:
    import cv2
    import numpy as np
    import pycolmap
    import torch
    from scipy.spatial import cKDTree

    prepare_deploy_imports(cfg)
    try:
        import boq_lib as vpr_lib  # type: ignore
        vpr_kind = "boq"
        vpr_input = getattr(vpr_lib, "INPUT", None)
    except ModuleNotFoundError:
        import common as vpr_common  # type: ignore
        import megaloc_lib as vpr_lib  # type: ignore
        vpr_kind = "megaloc"
        vpr_input = getattr(vpr_common, "VPR_INPUT", None)
    from reloc_localizer_xfeat import extract_xfeat, load_xfeat  # type: ignore

    p = cfg_paths(cfg)
    rec = pycolmap.Reconstruction(str(p.model0))
    images_by_name = {im.name: im for im in rec.images.values()}
    ref_names = sorted(n for n in images_by_name if (p.images / n).exists())
    if cfg.limit_refs:
        ref_names = ref_names[:cfg.limit_refs]
    xfeat = load_xfeat(cfg.xfeat_topk)
    refs: dict[str, Any] = {}
    snap_counts, snap_meds, kp_counts = [], [], []
    desc_dtype = torch.float32 if cfg.xfeat_fp32 else torch.float16

    def image_obs_xyz(image: Any) -> tuple[np.ndarray, np.ndarray]:
        pts, xyz = [], []
        for p2d in image.points2D:
            if p2d.has_point3D():
                pts.append(np.asarray(p2d.xy, np.float32))
                xyz.append(np.asarray(rec.points3D[p2d.point3D_id].xyz, np.float32))
        if not pts:
            return np.empty((0, 2), np.float32), np.empty((0, 3), np.float32)
        return np.vstack(pts).astype(np.float32), np.vstack(xyz).astype(np.float32)

    for i, name in enumerate(ref_names, 1):
        rgb = load_rgb(p.images / name)
        feats = extract_xfeat(xfeat, rgb, cfg.xfeat_topk)
        feats_cpu = {}
        for key, value in feats.items():
            if torch.is_tensor(value):
                vv = value.detach().cpu()
                if key == "descriptors":
                    vv = vv.to(desc_dtype)
                feats_cpu[key] = vv
            else:
                feats_cpu[key] = value
        kpts = feats_cpu["keypoints"].float().numpy()
        obs_xy, obs_xyz = image_obs_xyz(images_by_name[name])
        xyz = np.full((len(kpts), 3), np.nan, np.float32)
        if len(kpts) and len(obs_xy):
            dist, jj = cKDTree(obs_xy).query(kpts, k=1, workers=-1)
            ok = dist <= cfg.snap_px
            xyz[ok] = obs_xyz[jj[ok]]
            snap_counts.append(int(ok.sum()))
            if np.any(ok):
                snap_meds.append(float(np.median(dist[ok])))
        else:
            snap_counts.append(0)
        kp_counts.append(len(kpts))
        refs[name] = {"feats": feats_cpu, "xyz": xyz}
        if i % 100 == 0 or i == len(ref_names):
            log(f"  snap {i}/{len(ref_names)} last={name} kp={len(kpts)} snap3d={snap_counts[-1]}")

    device = "cuda" if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"
    ref_global = vpr_lib.extract(ref_names, p.images, device)
    if vpr_kind == "boq":
        vpr_name = f"boq-{vpr_lib.BACKBONE}-{vpr_lib.DESC_DIM}"
    else:
        vpr_name = f"megaloc-{ref_global.shape[1]}"
    bundle = {
        "meta": {
            "site_name": cfg.site_name,
            "vpr": vpr_name,
            "vpr_input": vpr_input,
            "feature": "xfeat",
            "matcher": "xfeat+lighterglue",
            "top_k": cfg.xfeat_topk,
            "desc_dtype": "fp32" if cfg.xfeat_fp32 else "fp16",
            "xyz_source": "nearest fixed-GLOMAP image observation snap",
            "snap_px": cfg.snap_px,
            "source_model": str(p.model0),
            "image_root": str(p.images),
            "refs": len(ref_names),
            "total_3d_anchored_kp": int(sum(snap_counts)),
            "mean_3d_anchored_per_ref": float(np.mean(snap_counts)) if snap_counts else 0.0,
            "median_snap_px": float(np.median(snap_meds)) if snap_meds else None,
        },
        "ref_names": ref_names,
        "ref_global": ref_global.astype(np.float32),
        "refs": refs,
    }
    torch.save(bundle, str(p.snap_bundle))
    log(f"XFeat snap bundle -> {p.snap_bundle} refs={len(ref_names)}")


def stage_tracking(cfg: SimpleNamespace) -> None:
    import numpy as np
    import pycolmap
    import torch

    p = cfg_paths(cfg)
    bundle = torch.load(str(p.snap_bundle), map_location="cpu", weights_only=False)
    ref_names = list(bundle["ref_names"])
    name_to_idx = {n: i for i, n in enumerate(ref_names)}
    rec = pycolmap.Reconstruction(str(p.model0))
    images_by_name = {im.name: im for im in rec.images.values()}

    centers = np.full((len(ref_names), 3), np.nan, np.float32)
    yaws = np.full((len(ref_names),), np.nan, np.float32)
    for name, idx in name_to_idx.items():
        im = images_by_name.get(name)
        if im is None:
            continue
        T = im.cam_from_world
        T = T() if callable(T) else T
        R = T.rotation.matrix()
        t = np.asarray(T.translation)
        center = -R.T @ t
        fwd = R.T @ np.array([0.0, 0.0, 1.0])
        centers[idx] = center.astype(np.float32)
        yaws[idx] = math.atan2(fwd[1], fwd[0])

    p3d_to_imgs: dict[int, list[int]] = defaultdict(list)
    for name, idx in name_to_idx.items():
        im = images_by_name.get(name)
        if im is None:
            continue
        for p2d in im.points2D:
            if p2d.has_point3D():
                p3d_to_imgs[int(p2d.point3D_id)].append(idx)
    counters = [Counter() for _ in ref_names]
    for ids in p3d_to_imgs.values():
        ids = ids[:80]
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if a != b:
                    counters[a][b] += 1
                    counters[b][a] += 1
    covis = {name: [int(j) for j, _ in counters[i].most_common(cfg.tracking_top_covis)]
             for i, name in enumerate(ref_names)}
    bundle["ref_centers"] = centers
    bundle["ref_yaws"] = yaws
    bundle["covis"] = covis
    meta = dict(bundle.get("meta", {}))
    meta.update({
        "tracking_metadata": True,
        "tracking_model": str(p.model0),
        "tracking_top_covis": cfg.tracking_top_covis,
        "tracking_median_covis_degree": float(np.median([len(v) for v in covis.values()])) if covis else 0.0,
    })
    bundle["meta"] = meta
    torch.save(bundle, str(p.tracking_bundle))
    log(f"tracking bundle -> {p.tracking_bundle}")


def stage_triangulate(cfg: SimpleNamespace) -> None:
    import h5py
    import numpy as np
    import pycolmap
    import torch
    from hloc import triangulation
    from hloc.utils.io import names_to_pair

    prepare_deploy_imports(cfg)
    from reloc_localizer_xfeat import XFeatRelocMap, _to_device_feats, load_xfeat  # type: ignore

    p = cfg_paths(cfg)
    xmap = XFeatRelocMap.load(str(p.tracking_bundle))
    if not xmap.covis:
        raise SystemExit("tracking bundle lacks covis metadata")
    work = p.reloc_tri
    work.mkdir(parents=True, exist_ok=True)
    active = xmap.ref_names[:cfg.limit_refs] if cfg.limit_refs else list(xmap.ref_names)
    active_set = set(active)
    name_to_idx = {n: i for i, n in enumerate(xmap.ref_names)}
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[str, str]] = []
    for name0 in active:
        for j in xmap.covis.get(name0, [])[:cfg.tri_pair_topk]:
            name1 = xmap.ref_names[int(j)]
            if name1 not in active_set:
                continue
            i0, i1 = name_to_idx[name0], name_to_idx[name1]
            key = (min(i0, i1), max(i0, i1))
            if i0 != i1 and key not in seen:
                seen.add(key)
                pairs.append((name0, name1))
                if cfg.tri_max_pairs and len(pairs) >= cfg.tri_max_pairs:
                    break
        if cfg.tri_max_pairs and len(pairs) >= cfg.tri_max_pairs:
            break
    pairs_path = work / f"pairs-xfeat-covis-top{cfg.tri_pair_topk}.txt"
    pairs_path.write_text("".join(f"{a} {b}\n" for a, b in pairs))

    feats_path = work / "feats-xfeat.h5"
    matches_path = work / f"matches-xfeat-lg-top{cfg.tri_pair_topk}.h5"
    if cfg.overwrite or not feats_path.exists():
        with h5py.File(str(feats_path), "w") as h5:
            for i, name in enumerate(xmap.ref_names, 1):
                ref = xmap.refs[name]
                group = h5.create_group(name)
                kpts = ref.feats["keypoints"].detach().cpu().numpy().astype(np.float32)
                desc = ref.feats["descriptors"].detach().cpu().float().numpy().astype(np.float32)
                group.create_dataset("keypoints", data=kpts)
                group.create_dataset("descriptors", data=desc.T)
                if "scores" in ref.feats:
                    group.create_dataset("scores", data=ref.feats["scores"].detach().cpu().numpy().astype(np.float32))
                if "image_size" in ref.feats:
                    group.create_dataset("image_size", data=np.asarray(ref.feats["image_size"], np.float32))
                if i % 200 == 0 or i == len(xmap.ref_names):
                    log(f"  tri features {i}/{len(xmap.ref_names)}")
    if cfg.overwrite or not matches_path.exists():
        xfeat = load_xfeat(cfg.xfeat_topk)
        with h5py.File(str(matches_path), "w") as h5:
            for pi, (name0, name1) in enumerate(pairs, 1):
                ref0, ref1 = xmap.refs[name0], xmap.refs[name1]
                _mk0, _mk1, pair_idx = xfeat.match_lighterglue(
                    _to_device_feats(ref0.feats),
                    _to_device_feats(ref1.feats),
                    min_conf=cfg.xfeat_min_conf,
                )
                n0 = int(ref0.feats["keypoints"].shape[0])
                matches0 = np.full((n0,), -1, dtype=np.int32)
                scores0 = np.zeros((n0,), dtype=np.float32)
                if len(pair_idx):
                    arr = np.asarray(pair_idx, dtype=np.int64)
                    matches0[arr[:, 0]] = arr[:, 1].astype(np.int32)
                    scores0[arr[:, 0]] = 1.0
                group = h5.create_group(names_to_pair(name0, name1))
                group.create_dataset("matches0", data=matches0)
                group.create_dataset("matching_scores0", data=scores0)
                if pi % 500 == 0 or pi == len(pairs):
                    log(f"  tri matches {pi}/{len(pairs)}")

    tri_dir = work / "xfeat_model"
    opts = pycolmap.IncrementalPipelineOptions()
    opts.extract_colors = False
    tri = triangulation.main(
        tri_dir,
        p.model0,
        p.images,
        pairs_path,
        feats_path,
        matches_path,
        skip_geometric_verification=cfg.tri_skip_geometric_verification,
        mapper_options=opts,
    )

    name_to_img = {im.name: im for im in tri.images.values()}
    refs: dict[str, Any] = {}
    total = 0
    per_ref = []
    for i, name in enumerate(xmap.ref_names, 1):
        ref = xmap.refs[name]
        n_kp = int(ref.feats["keypoints"].shape[0])
        xyz = np.full((n_kp, 3), np.nan, np.float32)
        img = name_to_img.get(name)
        if img is not None:
            for k, p2d in enumerate(img.points2D):
                if k < n_kp and p2d.has_point3D():
                    xyz[k] = np.asarray(tri.points3D[p2d.point3D_id].xyz, np.float32)
        n3d = int(np.isfinite(xyz[:, 0]).sum())
        total += n3d
        per_ref.append(n3d)
        feats = {kk: vv.detach().cpu() if torch.is_tensor(vv) else vv for kk, vv in ref.feats.items()}
        refs[name] = {"feats": feats, "xyz": xyz}
        if i % 200 == 0 or i == len(xmap.ref_names):
            log(f"  tri pack {i}/{len(xmap.ref_names)} last_3d={n3d}")
    meta = dict(xmap.meta)
    meta.update({
        "xyz_source": "fixed-pose XFeat+LighterGlue triangulation on GLOMAP poses",
        "source_model": str(p.model0),
        "triangulation_pair_topk": cfg.tri_pair_topk,
        "triangulation_pairs": len(pairs),
        "triangulation_points3D": int(tri.num_points3D()),
        "total_3d_anchored_kp": int(total),
        "mean_3d_anchored_per_ref": float(np.mean(per_ref)) if per_ref else 0.0,
        "median_3d_anchored_per_ref": float(np.median(per_ref)) if per_ref else 0.0,
    })
    bundle = {
        "meta": meta,
        "ref_names": xmap.ref_names,
        "ref_global": xmap.ref_global.astype(np.float32),
        "refs": refs,
    }
    if xmap.ref_centers is not None:
        bundle["ref_centers"] = xmap.ref_centers
    if xmap.ref_yaws is not None:
        bundle["ref_yaws"] = xmap.ref_yaws
    if xmap.covis is not None:
        bundle["covis"] = xmap.covis
    torch.save(bundle, str(p.tri_bundle))
    log(f"triangulated localizable bundle -> {p.tri_bundle} total_3d_anchored={total}")


def stage_report(cfg: SimpleNamespace) -> None:
    p = cfg_paths(cfg)
    outputs = {
        "glomap_model": str(p.model0),
        "rgb_point_cloud": str(p.rgb_ply),
        "snap_bundle": str(p.snap_bundle),
        "tracking_bundle": str(p.tracking_bundle),
        "triangulated_bundle": str(p.tri_bundle),
        "frame_manifest": str(p.manifest),
        "intrinsics": str(p.intrinsics),
        "config": str(p.config),
        "preflight_report": str(p.preflight_report),
        "stage_times": str(p.stage_times),
        "pair_verification_report": str(p.pair_verification_report),
        "mapper_diagnostics": str(p.mapper_diagnostics),
        "backend_comparison": str(p.backend_comparison),
        "pair_graph_diagnostics": str(p.pair_graph_diagnostics),
        "rotation_bridge_report": str(p.rotation_bridge_report),
    }
    sizes = {}
    for key, value in outputs.items():
        path = Path(value)
        if path.is_file():
            sizes[key] = path.stat().st_size
        elif path.is_dir():
            try:
                sizes[key] = int(subprocess.check_output(["du", "-sb", str(path)]).split()[0])
            except Exception:
                sizes[key] = None
    report = {
        "site_name": cfg.site_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "outputs": outputs,
        "sizes_bytes": sizes,
        "parameters": vars(cfg),
        "stage_times": read_stage_times(cfg),
        "optional_modules": {
            "LFOE-GlobalSfM": {
                "use_when": "diagnostic fallback when standard GLOMAP fails registration gates or the view graph has translation-edge outliers",
                "how_to_enable": "--lfoe-mode diagnostic|always or --use-lfoe",
                "default": getattr(cfg, "lfoe_mode", "off"),
            },
            "MP-SfM": {
                "use_when": "primary candidate backend for low-overlap, low-texture, or low-parallax scenes",
                "how_to_enable": "--backend mpsfm|compare --mpsfm-conf sp-mast3r-dense",
                "default_conf": getattr(cfg, "mpsfm_conf", "sp-mast3r-dense"),
            },
            "Dense Match Summarization": {
                "use_when": "fast two-view verification before dense aggregation; does not replace full dense matches",
                "how_to_enable": "--pair-verification dms",
                "default": getattr(cfg, "pair_verification", "off"),
            },
            "Doppelgangers++": {
                "use_when": "visually repeated structures cause false image pairs",
                "how_to_enable": "--doppelgangers-checkpoint /path/checkpoint.pth",
                "default": bool(cfg.doppelgangers_checkpoint),
            },
            "UFM": {
                "use_when": "prematch module inside MV-RoMa dense matching",
                "note": "Normally used indirectly by MV-RoMa; no separate stage needed.",
            },
        },
    }
    write_json(p.report_json, report)
    lines = [
        f"# Localizable Map Build Report: {cfg.site_name}",
        "",
        "## Outputs",
        "",
    ]
    for key, value in outputs.items():
        lines.append(f"- `{key}`: `{value}`")
    lines += [
        "",
        "## Core Parameters",
        "",
        f"- frames fps: `{cfg.fps}`, max_side: `{cfg.max_side}`",
        f"- MegaLoc num_matched: `{cfg.num_matched}`, seq_window: `{cfg.seq_window}`, cross_topk: `{cfg.cross_topk}`, cross_grid: `{cfg.cross_grid}`",
        f"- Pair graph mode: `{getattr(cfg, 'pair_graph_mode', 'directional')}`, intra cap: `{getattr(cfg, 'agg_intra_degree_cap', 10)}`, cross-direction cap: `{getattr(cfg, 'agg_cross_direction_degree_cap', 8)}`, total cap: `{cfg.agg_pair_degree_cap}`",
        f"- MV-RoMA cert: `{cfg.roma_cert_thresh}`, grid: `{cfg.mvroma_grid_h}x{cfg.mvroma_grid_w}`, chunk: `{cfg.mvroma_chunk}`, sample: `{getattr(cfg, 'mvroma_sample_mode', 'random')}`",
        f"- Doppelgangers++ scope: `{getattr(cfg, 'doppelgangers_filter_scope', 'all')}`, threshold: `{getattr(cfg, 'doppelgangers_threshold', 0.8)}`",
        f"- Pair verification: `{getattr(cfg, 'pair_verification', 'off')}` action `{getattr(cfg, 'pair_verification_action', 'filter')}`",
        f"- Motion bridges: use_rotation_bridges=`{getattr(cfg, 'use_rotation_bridges', False)}`, exclude_rotation_from_triangulation=`{getattr(cfg, 'exclude_rotation_from_triangulation', False)}`",
        f"- Backend: `{getattr(cfg, 'backend', 'glomap')}`, MP-SfM conf: `{getattr(cfg, 'mpsfm_conf', 'sp-mast3r-dense')}`",
        f"- GLOMAP skip_BA: `{cfg.skip_bundle_adjustment}`, skip_retriangulation: `{cfg.skip_retriangulation}`, optimize_intrinsics: `{getattr(cfg, 'optimize_intrinsics', 0)}`, max_tracks: `{cfg.max_num_tracks}`, min_views: `{getattr(cfg, 'min_num_view_per_track', 0)}`, min_angle: `{getattr(cfg, 'min_triangulation_angle', 0.0)}`",
        f"- Dense MV-RoMA triangulation: `{getattr(cfg, 'dense_map_triangulation', False)}`, two_view: `{getattr(cfg, 'dense_tri_include_two_view_tracks', True)}`, min_angle: `{getattr(cfg, 'dense_tri_min_angle', 0.5)}`, max_reproj: `{getattr(cfg, 'dense_tri_max_reproj_error', 2.0)}`",
        f"- XFeat topk: `{cfg.xfeat_topk}`, snap_px: `{cfg.snap_px}`, tri_pair_topk: `{cfg.tri_pair_topk}`",
        "",
        "## Stage Timing",
        "",
    ]
    timing = read_stage_times(cfg)
    for item in timing.get("stages", []):
        lines.append(f"- `{item.get('stage')}`: `{item.get('duration_seconds')}` sec ({item.get('status')})")
    lines += [
        "",
        "## Optional Modules",
        "",
        "- LFOE-GlobalSfM: diagnostic fallback for bad translation edges, disconnected graphs, or weak standard GLOMAP output.",
        "- MP-SfM: primary candidate backend for low-overlap or low-texture scenes; configure with --mpsfm-conf.",
        "- Dense Match Summarization: uses compact score/grid samples only for RANSAC edge checks; full dense matches remain for aggregation.",
        "- Doppelgangers++: use before MV-RoMa when repeated structures create visually plausible but wrong pairs.",
        "- UFM: used indirectly as MV-RoMa's prematch model; keep it installed for the MV-RoMa stage.",
    ]
    p.report_md.write_text("\n".join(lines) + "\n")
    log(f"report -> {p.report_md}")


STAGE_FUNCS = {
    "preflight": stage_preflight,
    "extract": stage_extract,
    "manifest": stage_manifest,
    "pairs": stage_pairs,
    "doppelgangers": stage_doppelgangers,
    "mvroma": stage_mvroma,
    "verify_pairs": stage_verify_pairs,
    "aggregate": stage_aggregate,
    "db": stage_db,
    "glomap": stage_glomap,
    "color": stage_color,
    "snap": stage_snap,
    "tracking": stage_tracking,
    "triangulate": stage_triangulate,
    "report": stage_report,
}


RUNTIME_DEFAULTS: dict[str, Any] = {
    "strict_gates": False,
    "strict_profile": "",
    "pair_graph_mode": "directional",
    "same_direction_topk": 0,
    "direction_overrides_json": "",
    "agg_intra_degree_cap": 0,
    "agg_cross_direction_degree_cap": 8,
    "mvroma_sample_mode": "random",
    "mvroma_sample_grid": "8x12",
    "mvroma_resume": True,
    "repair_dense_h5": False,
    "doppelgangers_filter_scope": "all",
    "pair_verification": "off",
    "pair_verification_action": "filter",
    "dms_max_matches": 512,
    "dms_grid": "8x12",
    "dms_min_sampled_matches": 24,
    "dms_min_inliers": 24,
    "dms_min_inlier_ratio": 0.08,
    "dms_ransac_px": 2.0,
    "dms_homography_px": 3.0,
    "dms_max_trials": 2000,
    "dms_rotation_h_over_f": 1.5,
    "backend": "glomap",
    "use_rotation_bridges": False,
    "exclude_rotation_from_triangulation": False,
    "motion_classes": "parallax,pure_rotation,hover",
    "lfoe_mode": "off",
    "mpsfm_repo": DEFAULT_MPSFM_REPO,
    "python_mpsfm": DEFAULT_PY_MPSFM,
    "mpsfm_conf": "sp-mast3r-dense",
    "mpsfm_config_yaml": "",
    "mpsfm_extract": True,
    "mpsfm_extract_list": "sky,features,matches,depth,normal",
    "mpsfm_verbose": 0,
    "mpsfm_max_frames_per_chunk": 220,
    "min_num_view_per_track": 0,
    "min_triangulation_angle": 0.0,
    "optimize_intrinsics": 0,
    "python_mvroma": DEFAULT_PY_MVROMA,
    "dense_map_triangulation": False,
    "dense_tri_include_two_view_tracks": True,
    "dense_tri_min_angle": 0.5,
    "dense_tri_max_reproj_error": 2.0,
    "dense_tri_skip_geometric_verification": False,
}


def apply_config_defaults(data: dict[str, Any]) -> dict[str, Any]:
    merged = dict(RUNTIME_DEFAULTS)
    merged.update(data)
    if "agg_cross_direction_degree_cap" not in data and "agg_cross_degree_cap" in data:
        merged["agg_cross_direction_degree_cap"] = data["agg_cross_degree_cap"]
    if "min_num_view_per_track" not in data and "glomap_min_num_view_per_track" in data:
        merged["min_num_view_per_track"] = data["glomap_min_num_view_per_track"]
    if "min_triangulation_angle" not in data and "glomap_min_triangulation_angle" in data:
        merged["min_triangulation_angle"] = data["glomap_min_triangulation_angle"]
    return merged


def namespace_from_dict(data: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**data)


def save_config(args: argparse.Namespace) -> SimpleNamespace:
    data = apply_config_defaults(vars(args).copy())
    data.pop("run_stage", None)
    data.pop("config", None)
    stages = data.pop("stages")
    data["stages"] = ",".join(parse_stages(stages))
    if data.get("camera_init_json"):
        data["camera_init"] = read_json(data["camera_init_json"])
    else:
        data["camera_init"] = {
            "1280x720": {"model": "SIMPLE_RADIAL", "params": [936.5, 640.0, 360.0, 0.0035]},
            "1920x1080": {"model": "SIMPLE_RADIAL", "params": [1400.0, 960.0, 540.0, 0.0015]},
        }
    cfg = namespace_from_dict(data)
    ensure_dirs(cfg)
    write_json(cfg_paths(cfg).config, data)
    return cfg


def run_orchestrator(cfg: SimpleNamespace) -> None:
    ensure_dirs(cfg)
    stages = parse_stages(cfg.stages)
    if not cfg.doppelgangers_checkpoint:
        stages = [s for s in stages if s != "doppelgangers"]
    if getattr(cfg, "pair_verification", "off") == "off":
        stages = [s for s in stages if s != "verify_pairs"]
    dry_run_skip = {"color", "snap", "tracking", "triangulate", "report"}
    script = Path(__file__).resolve()
    for stage in stages:
        started = time.time()
        status = "success"
        error = ""
        try:
            if cfg.dry_run and stage in dry_run_skip:
                status = "skipped"
                log(f"dry-run: skip artifact-dependent stage {stage}")
            elif stage in PY_STAGE_ENV and not cfg.in_process:
                py = getattr(cfg, PY_STAGE_ENV[stage])
                cmd = [py, str(script), "--config", str(cfg_paths(cfg).config), "--run-stage", stage]
                env = os.environ.copy()
                ufm_root = as_path(getattr(cfg, "ufm_root", DEFAULT_UFM_ROOT))
                if ufm_root.exists():
                    env["PYTHONPATH"] = (
                        str(ufm_root) + os.pathsep +
                        str(ufm_root / "UniCeption") + os.pathsep +
                        env.get("PYTHONPATH", "")
                    )
                run_cmd(cmd, env=env, dry_run=cfg.dry_run)
            else:
                log(f"=== stage: {stage} ===")
                STAGE_FUNCS[stage](cfg)
            if status != "skipped":
                validate_stage_gate(stage, cfg)
        except BaseException as exc:
            status = "failed"
            error = str(exc)
            append_stage_time(cfg, stage_time_record(stage, started, time.time(), status, error))
            raise
        append_stage_time(cfg, stage_time_record(stage, started, time.time(), status, error))


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--site-name", default="field_map")
    ap.add_argument("--videos", nargs="*", default=[])
    ap.add_argument("--image-root", default="")
    ap.add_argument("--work-dir", required=False)
    ap.add_argument("--stages", default="all", help="comma list or all")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--in-process", action="store_true", help="do not re-enter this file with stage-specific Python")
    ap.add_argument("--disable-stage-gates", action="store_true", help="skip per-stage PASS/FAIL validation gates")
    ap.add_argument("--strict-gates", action="store_true", default=False, help="enforce profile-specific hard quality gates")
    ap.add_argument("--strict-profile", choices=["football_field_1920"], default="", help="strict gate threshold profile")
    ap.add_argument("--gate-min-frames", type=int, default=2)
    ap.add_argument("--gate-min-pairs", type=int, default=10)
    ap.add_argument("--gate-min-registered-images", type=int, default=2)
    ap.add_argument("--gate-min-registered-ratio", type=float, default=0.40)

    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-side", type=int, default=1920)
    ap.add_argument("--jpeg-quality", type=int, default=2)
    ap.add_argument("--motion-gate", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--motion-action", choices=["report", "filter"], default="report")
    ap.add_argument("--motion-min-tracks", type=int, default=20)
    ap.add_argument("--motion-min-flow-px", type=float, default=1.5)
    ap.add_argument("--motion-rotation-h-over-f", type=float, default=0.85)
    ap.add_argument("--motion-rotation-min-inliers", type=int, default=20)
    ap.add_argument("--motion-keep-rotation-every", type=int, default=0)
    ap.add_argument("--motion-classes", default="parallax,pure_rotation,hover")
    ap.add_argument("--use-rotation-bridges", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--exclude-rotation-from-triangulation", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--template-repo", default=DEFAULT_TEMPLATE_REPO)
    ap.add_argument("--mvroma-root", default=DEFAULT_MVROMA_ROOT)
    ap.add_argument("--mvroma-weights", default=f"{DEFAULT_MVROMA_ROOT}/outdoor_final.pth")
    ap.add_argument("--doppelgangers-root", default=DEFAULT_DG_ROOT)
    ap.add_argument("--doppelgangers-checkpoint", default="")
    ap.add_argument("--doppelgangers-threshold", type=float, default=0.8)
    ap.add_argument("--doppelgangers-filter-scope", choices=["all", "cross_video", "cross_direction"], default="all")
    ap.add_argument("--use-lfoe", action="store_true")
    ap.add_argument("--lfoe-mode", choices=["diagnostic", "always", "off"], default="off")
    ap.add_argument("--backend", choices=["glomap", "lfoe", "mpsfm", "compare"], default="glomap")
    ap.add_argument("--mpsfm-repo", default=DEFAULT_MPSFM_REPO)
    ap.add_argument("--python-mpsfm", default=DEFAULT_PY_MPSFM)
    ap.add_argument("--mpsfm-conf", default="sp-mast3r-dense")
    ap.add_argument("--mpsfm-config-yaml", default="", help="absolute or relative MP-SfM YAML config; passed through --conf")
    ap.add_argument("--mpsfm-extract", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mpsfm-extract-list", default="sky,features,matches,depth,normal")
    ap.add_argument("--mpsfm-verbose", type=int, default=0)
    ap.add_argument("--mpsfm-max-frames-per-chunk", type=int, default=220)
    ap.add_argument("--glomap-command", default=DEFAULT_GLOMAP)
    ap.add_argument("--python-sfm", default=DEFAULT_PY_SFM)
    ap.add_argument("--python-sfmdb", default=DEFAULT_PY_SFMDB)
    ap.add_argument("--python-mvroma", default=DEFAULT_PY_MVROMA)
    ap.add_argument("--device", default="cuda:0")

    ap.add_argument("--pair-graph-mode", choices=["directional", "legacy"], default="directional")
    ap.add_argument("--direction-overrides-json", default="")
    ap.add_argument("--num-matched", type=int, default=20)
    ap.add_argument("--seq-window", type=int, default=5)
    ap.add_argument("--same-direction-topk", type=int, default=0)
    ap.add_argument("--cross-topk", type=int, default=8)
    ap.add_argument("--cross-grid", type=int, default=10)
    ap.add_argument("--limit-src", type=int, default=0)

    ap.add_argument("--roma-cert-thresh", type=float, default=0.35)
    ap.add_argument("--mvroma-chunk", type=int, default=6)
    ap.add_argument("--mvroma-grid-h", type=int, default=560)
    ap.add_argument("--mvroma-grid-w", type=int, default=840)
    ap.add_argument("--agg-maxkp", type=int, default=4000)
    ap.add_argument("--agg-max-error", type=float, default=2.0)
    ap.add_argument("--agg-cell-size", type=float, default=1.0)
    ap.add_argument("--agg-pair-degree-cap", type=int, default=18)
    ap.add_argument("--agg-intra-degree-cap", type=int, default=0)
    ap.add_argument("--agg-cross-degree-cap", dest="agg_cross_direction_degree_cap", type=int, default=8)
    ap.add_argument("--agg-cross-direction-degree-cap", dest="agg_cross_direction_degree_cap", type=int, default=argparse.SUPPRESS)
    ap.add_argument("--mvroma-sample-mode", choices=["score_grid", "random"], default="random")
    ap.add_argument("--mvroma-sample-grid", default="8x12")
    ap.add_argument(
        "--mvroma-resume",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="reuse strict-valid per-source MV-RoMa shards",
    )
    ap.add_argument("--repair-dense-h5", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--pair-verification", choices=["dms", "off"], default="off")
    ap.add_argument("--pair-verification-action", choices=["filter", "report"], default="filter")
    ap.add_argument("--dms-max-matches", type=int, default=512)
    ap.add_argument("--dms-grid", default="8x12")
    ap.add_argument("--dms-min-sampled-matches", type=int, default=24)
    ap.add_argument("--dms-min-inliers", type=int, default=24)
    ap.add_argument("--dms-min-inlier-ratio", type=float, default=0.08)
    ap.add_argument("--dms-ransac-px", type=float, default=2.0)
    ap.add_argument("--dms-homography-px", type=float, default=3.0)
    ap.add_argument("--dms-max-trials", type=int, default=2000)
    ap.add_argument("--dms-rotation-h-over-f", type=float, default=1.5)

    ap.add_argument("--camera-model", default="SIMPLE_RADIAL")
    ap.add_argument("--camera-init-json", default="")
    ap.add_argument("--focal-px", type=float, default=0.0)
    ap.add_argument("--focal-ratio", type=float, default=1400.0 / 1920.0)
    ap.add_argument("--k1", type=float, default=0.0015)
    ap.add_argument("--verify-max-trials", type=int, default=20000)
    ap.add_argument("--verify-min-inlier-ratio", type=float, default=0.1)

    ap.add_argument("--skip-bundle-adjustment", action="store_true", default=False)
    ap.add_argument("--run-bundle-adjustment", dest="skip_bundle_adjustment", action="store_false")
    ap.add_argument("--skip-retriangulation", action="store_true", default=True)
    ap.add_argument("--run-retriangulation", dest="skip_retriangulation", action="store_false")
    ap.add_argument("--optimize-intrinsics", type=int, default=0)
    ap.add_argument("--optimize-principal-point", type=int, default=0)
    ap.add_argument("--max-num-tracks", type=int, default=600000)
    ap.add_argument("--min-num-view-per-track", type=int, default=0)
    ap.add_argument("--glomap-min-num-view-per-track", dest="min_num_view_per_track", type=int, default=argparse.SUPPRESS)
    ap.add_argument("--min-triangulation-angle", type=float, default=0.0)
    ap.add_argument("--glomap-min-triangulation-angle", dest="min_triangulation_angle", type=float, default=argparse.SUPPRESS)
    ap.add_argument("--dense-map-triangulation", action="store_true", default=False)
    ap.add_argument("--dense-tri-include-two-view-tracks", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dense-tri-min-angle", type=float, default=0.5)
    ap.add_argument("--dense-tri-max-reproj-error", type=float, default=2.0)
    ap.add_argument("--dense-tri-skip-geometric-verification", action="store_true", default=False)

    ap.add_argument("--color-outlier-scale", type=float, default=8.0)
    ap.add_argument("--xfeat-topk", type=int, default=2048)
    ap.add_argument("--xfeat-min-conf", type=float, default=0.1)
    ap.add_argument("--xfeat-fp32", action="store_true")
    ap.add_argument("--snap-px", type=float, default=5.0)
    ap.add_argument("--tracking-top-covis", type=int, default=40)
    ap.add_argument("--tri-pair-topk", type=int, default=20)
    ap.add_argument("--tri-max-pairs", type=int, default=0)
    ap.add_argument("--tri-skip-geometric-verification", action="store_true")
    ap.add_argument("--limit-refs", type=int, default=0)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="", help="reuse an existing build_config.json")
    ap.add_argument("--run-stage", choices=ALL_STAGES, default="", help=argparse.SUPPRESS)
    add_common_args(ap)
    args = ap.parse_args()
    if args.config:
        data = apply_config_defaults(read_json(args.config))
        if hasattr(args, "mvroma_resume"):
            data["mvroma_resume"] = args.mvroma_resume
        if args.stages != "all":
            data["stages"] = args.stages
        if args.run_stage:
            data["run_stage"] = args.run_stage
        else:
            data["run_stage"] = ""
        return namespace_from_dict(data)
    if not args.work_dir:
        raise SystemExit("--work-dir is required unless --config is used")
    return args


def main() -> None:
    args = parse_args()
    if isinstance(args, SimpleNamespace):
        cfg = args
        if cfg.run_stage:
            log(f"=== stage: {cfg.run_stage} ===")
            STAGE_FUNCS[cfg.run_stage](cfg)
        else:
            run_orchestrator(cfg)
        return
    cfg = save_config(args)
    run_orchestrator(cfg)


if __name__ == "__main__":
    main()
