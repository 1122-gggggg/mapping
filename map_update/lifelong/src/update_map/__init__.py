"""Historical-view augmentation for an immutable current map."""

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
from .map_adapters import load_map, register_map_adapter
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
    "load_map",
    "register_map_adapter",
]

__version__ = "0.1.0"
