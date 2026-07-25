#!/usr/bin/env python3
"""Map-update adapter for the deployment MegaLoc cache contract."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


def _system_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if parent.name == "sfm_system":
            return parent
    raise ImportError(f"cannot find sfm_system root from {here}")


_SHARED_PATH = (
    _system_root()
    / "定位"
    / "deploy_code"
    / "sfm_glomap_deploy"
    / "megaloc_cache_io.py"
).resolve()
_SHARED_MODULE_NAME = "_sfm_system_deploy_megaloc_cache_io"


def _load_shared_module():
    existing = sys.modules.get(_SHARED_MODULE_NAME)
    if existing is not None:
        if Path(existing.__file__).resolve() != _SHARED_PATH:
            raise ImportError(f"unexpected cached shared module: {existing.__file__}")
        return existing
    spec = importlib.util.spec_from_file_location(_SHARED_MODULE_NAME, _SHARED_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load shared MegaLoc cache module: {_SHARED_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SHARED_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_SHARED_MODULE_NAME, None)
        raise
    return module


_shared = _load_shared_module()

MEGALOC_DESCRIPTOR_DIM = _shared.MEGALOC_DESCRIPTOR_DIM
write_megaloc_cache = _shared.write_megaloc_cache


def load_aligned_megaloc_cache(
    cache_path: str | Path,
    expected_names: Iterable[str],
    *,
    expected_dim: int | None = None,
    expected_input_size: int | None = None,
) -> np.ndarray:
    """Load and name-align a cache through the shared deployment contract."""
    return _shared.load_megaloc_cache(
        cache_path,
        expected_names,
        expected_dim=expected_dim,
        expected_input_size=expected_input_size,
    ).descriptors
