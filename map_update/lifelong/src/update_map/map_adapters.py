from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Callable

from .io.colmap import load_colmap_reconstruction
from .models import BaseMap


MapLoader = Callable[[str | Path], BaseMap]

_MAP_LOADERS: dict[str, MapLoader] = {
    "colmap": load_colmap_reconstruction,
    "glomap": load_colmap_reconstruction,
    "gluemap": load_colmap_reconstruction,
}


def register_map_adapter(
    name: str,
    loader: MapLoader,
    *,
    replace: bool = False,
) -> None:
    """Register a loader that normalizes any map format into ``BaseMap``."""
    key = name.strip().lower()
    if not key:
        raise ValueError("map adapter name must not be empty")
    if not callable(loader):
        raise TypeError("map adapter loader must be callable")
    if key in _MAP_LOADERS and not replace:
        raise ValueError(f"Map adapter '{key}' is already registered")
    _MAP_LOADERS[key] = loader


def _external_loader(spec: str) -> MapLoader:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(
            "External map adapters must use 'package.module:loader' syntax"
        )
    candidate = getattr(import_module(module_name), attribute)
    if isinstance(candidate, type):
        candidate = candidate()
    loader = getattr(candidate, "load", candidate)
    if not callable(loader):
        raise TypeError(f"External map adapter '{spec}' must be callable or define load()")
    return loader


def load_map(path: str | Path, adapter: str = "colmap") -> BaseMap:
    """Load a map through a built-in or importable adapter."""
    token = str(adapter).strip()
    if not token:
        raise ValueError("map adapter must not be empty")
    if ":" in token:
        loader = _external_loader(token)
    else:
        try:
            loader = _MAP_LOADERS[token.lower()]
        except KeyError as exc:
            builtins = ", ".join(sorted(_MAP_LOADERS))
            raise ValueError(
                f"Unknown map adapter '{adapter}'. Built-ins: {builtins}. "
                "External adapters use package.module:loader."
            ) from exc
    model = loader(path)
    if not isinstance(model, BaseMap):
        raise TypeError("map adapter must return update_map.models.BaseMap")
    return model


def list_map_adapters() -> tuple[str, ...]:
    return tuple(_MAP_LOADERS)
