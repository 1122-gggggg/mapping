from __future__ import annotations

from mapdoctor.adapters.base import MapAdapter
from mapdoctor.adapters.colmap import ColmapAdapter
from mapdoctor.adapters.glomap import GlomapAdapter
from mapdoctor.adapters.gluemap import GluemapAdapter

_ADAPTER_TYPES: dict[str, type[MapAdapter]] = {
    "colmap": ColmapAdapter,
    "glomap": GlomapAdapter,
    "gluemap": GluemapAdapter,
}


def get_adapter(backend: str) -> MapAdapter:
    try:
        return _ADAPTER_TYPES[backend.lower()]()
    except KeyError as exc:
        supported = ", ".join(sorted(_ADAPTER_TYPES))
        raise ValueError(f"Unknown map backend '{backend}'. Supported backends: {supported}") from exc


def list_adapters() -> tuple[str, ...]:
    return tuple(_ADAPTER_TYPES)
