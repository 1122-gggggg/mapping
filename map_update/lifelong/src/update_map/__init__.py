"""Historical-view augmentation for a privileged GLUEMAP geometry."""

from .bundle import CandidateBundleManager
from .config import LifelongConfig, UpdateMapConfig, load_config
from .lifelong import (
    FeatureCandidate,
    FeatureEvent,
    FeatureMemoryRecord,
    FeatureState,
    MapManagementStrategy,
    PredictiveAdaptiveMapManager,
    classify_feature_events,
    fit_fremen_model,
)
from .models import BaseMap, HistoricalReference, Pose, PoseQuality
from .online import CurrentFirstLocalizer, HistoricalReferenceIndex

__all__ = [
    "BaseMap",
    "CandidateBundleManager",
    "CurrentFirstLocalizer",
    "FeatureCandidate",
    "FeatureEvent",
    "FeatureMemoryRecord",
    "FeatureState",
    "HistoricalReference",
    "HistoricalReferenceIndex",
    "LifelongConfig",
    "MapManagementStrategy",
    "Pose",
    "PoseQuality",
    "PredictiveAdaptiveMapManager",
    "UpdateMapConfig",
    "classify_feature_events",
    "fit_fremen_model",
    "load_config",
]

__version__ = "0.1.0"
