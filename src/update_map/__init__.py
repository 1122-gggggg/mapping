"""Historical-view augmentation for a privileged GLUEMAP geometry."""

from .bundle import CandidateBundleManager
from .config import UpdateMapConfig, load_config
from .models import BaseMap, HistoricalReference, Pose, PoseQuality
from .online import CurrentFirstLocalizer, HistoricalReferenceIndex

__all__ = [
    "BaseMap",
    "CandidateBundleManager",
    "CurrentFirstLocalizer",
    "HistoricalReference",
    "HistoricalReferenceIndex",
    "Pose",
    "PoseQuality",
    "UpdateMapConfig",
    "load_config",
]

__version__ = "0.1.0"
