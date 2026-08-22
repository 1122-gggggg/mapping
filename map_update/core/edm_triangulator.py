#!/usr/bin/env python3
"""Thin wrapper around machine-local build_reloc_map_edm.py stages A-E.

The upstream script does fixed-pose re-triangulation (no joint BA). This module
does not reimplement matching or triangulation; it only locates the script and
builds the argv. Fail-closed if the script is missing.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


DEFAULT_EDM_TRIANGULATOR = Path(
    "/home/allen/localization/定位演算法/EDM工具包/build/build_reloc_map_edm.py"
)


def require_edm_triangulator(path: str | Path | None = None) -> Path:
    candidate = Path(path) if path is not None else DEFAULT_EDM_TRIANGULATOR
    if not candidate.is_file():
        raise SystemExit(f"EDM triangulator missing: {candidate}")
    return candidate.resolve()


def frozen_pose_triangulate_argv(
    model: Path,
    image_root: Path,
    in_bundle: Path,
    work_dir: Path,
    out: Path,
    *,
    triangulator: str | Path | None = None,
    python: str = sys.executable,
) -> list[str]:
    script = require_edm_triangulator(triangulator)
    return [
        python,
        str(script),
        "--model",
        str(model),
        "--image-root",
        str(image_root),
        "--in-bundle",
        str(in_bundle),
        "--work-dir",
        str(work_dir),
        "--out",
        str(out),
    ]


def run_frozen_pose_triangulation(
    model: Path,
    image_root: Path,
    in_bundle: Path,
    work_dir: Path,
    out: Path,
    *,
    triangulator: str | Path | None = None,
    python: str = sys.executable,
) -> None:
    cmd = frozen_pose_triangulate_argv(
        model,
        image_root,
        in_bundle,
        work_dir,
        out,
        triangulator=triangulator,
        python=python,
    )
    subprocess.run(cmd, check=True)
