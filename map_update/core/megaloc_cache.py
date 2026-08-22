#!/usr/bin/env python3
"""Map-update adapter for the deployment MegaLoc cache contract."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


_SHARED_MODULE_NAME = "_sfm_system_deploy_megaloc_cache_io"
_MEGALOC_CACHE_IO_ENV = "MEGALOC_CACHE_IO"
_SHARED_RELATIVE = Path("定位") / "deploy_code" / "sfm_glomap_deploy" / "megaloc_cache_io.py"


def _system_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if parent.name == "sfm_system":
            return parent
    raise ImportError(
        f"cannot find sfm_system root from {here}; "
        f"missing {_SHARED_RELATIVE.as_posix()}; "
        f"pass {_MEGALOC_CACHE_IO_ENV} or override is required"
    )


def _shared_path(shared_path: str | Path | None = None) -> Path:
    if shared_path is not None:
        return Path(shared_path).expanduser().resolve()
    override = os.environ.get(_MEGALOC_CACHE_IO_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return (_system_root() / _SHARED_RELATIVE).resolve()


def _load_shared_module(shared_path: str | Path | None = None):
    path = _shared_path(shared_path)
    if not path.is_file():
        raise ImportError(
            f"shared MegaLoc cache module not found: {path}; "
            f"pass {_MEGALOC_CACHE_IO_ENV} or override is required"
        )
    existing = sys.modules.get(_SHARED_MODULE_NAME)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file is None or Path(existing_file).resolve() != path:
            raise ImportError(f"unexpected cached shared module: {existing_file}")
        return existing
    spec = importlib.util.spec_from_file_location(_SHARED_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load shared MegaLoc cache module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SHARED_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_SHARED_MODULE_NAME, None)
        raise
    return module


def write_megaloc_cache(*args, **kwargs):
    return _load_shared_module().write_megaloc_cache(*args, **kwargs)


def load_aligned_megaloc_cache(
    cache_path: str | Path,
    expected_names: Iterable[str],
    *,
    expected_dim: int | None = None,
    expected_input_size: int | None = None,
    shared_path: str | Path | None = None,
) -> np.ndarray:
    """Load and name-align a cache through the shared deployment contract."""
    return _load_shared_module(shared_path).load_megaloc_cache(
        cache_path,
        expected_names,
        expected_dim=expected_dim,
        expected_input_size=expected_input_size,
    ).descriptors


def __getattr__(name: str):
    if name == "_shared":
        return _load_shared_module()
    if name == "_SHARED_PATH":
        return _shared_path()
    if name == "MEGALOC_DESCRIPTOR_DIM":
        return _load_shared_module().MEGALOC_DESCRIPTOR_DIM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
