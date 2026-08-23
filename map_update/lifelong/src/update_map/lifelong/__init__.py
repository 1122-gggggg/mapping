"""Predictive and adaptive feature-map memory."""

from .fremen import (
    descriptor_distance,
    descriptor_uniqueness,
    fit_fremen_model,
    rank_candidates_by_uniqueness,
)
from .manager import PredictiveAdaptiveMapManager
from .models import (
    FeatureCandidate,
    FeatureEvent,
    FeatureMemoryRecord,
    FeatureObservation,
    FeatureSelection,
    FeatureState,
    HarmonicComponent,
    MapManagementStrategy,
    MapUpdatePlan,
    TemporalScoreModel,
    classify_feature_events,
    event_value,
)

__all__ = [
    "FeatureCandidate",
    "FeatureEvent",
    "FeatureMemoryRecord",
    "FeatureObservation",
    "FeatureSelection",
    "FeatureState",
    "HarmonicComponent",
    "MapManagementStrategy",
    "MapUpdatePlan",
    "PredictiveAdaptiveMapManager",
    "TemporalScoreModel",
    "classify_feature_events",
    "descriptor_distance",
    "descriptor_uniqueness",
    "event_value",
    "fit_fremen_model",
    "rank_candidates_by_uniqueness",
]
