"""MapDoctor: diagnostics and regression testing for visual-localization maps."""

from mapdoctor.adapters import (
    ColmapAdapter,
    GlomapAdapter,
    GluemapAdapter,
    get_adapter,
    register_adapter,
)
from mapdoctor.api import load_colmap, load_glomap, load_gluemap

__version__ = "1.2.1"

__all__ = [
    "ColmapAdapter",
    "GlomapAdapter",
    "GluemapAdapter",
    "get_adapter",
    "register_adapter",
    "load_colmap",
    "load_glomap",
    "load_gluemap",
    "__version__",
]
