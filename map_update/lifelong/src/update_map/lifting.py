from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .config import LiftingConfig
from .models import BaseMap, LiftedCorrespondence, MaskBundle, MatchSet, Observation
from .states import GeometryProvenance, MaskLabel


@dataclass
class LiftingDiagnostics:
    raw_matches: int = 0
    confidence_filtered: int = 0
    outside_stable_mask: int = 0
    no_nearby_observation: int = 0
    ambiguous_snap: int = 0
    forbidden_provenance: int = 0
    lifted: int = 0
    unique_point3d: int = 0
    duplicate_point3d: int = 0
    multi_reference_supported: int = 0
    snap_distances: list[float] | None = None

    def __post_init__(self) -> None:
        if self.snap_distances is None:
            self.snap_distances = []


class ReferenceObservationIndex:
    def __init__(self, observations: Sequence[Observation], base_map: BaseMap):
        valid = [item for item in observations if item.point3d_id in base_map.points3d]
        self.observations = valid
        self.base_map = base_map
        self.xy = (
            np.stack([item.xy for item in valid], axis=0)
            if valid
            else np.empty((0, 2), dtype=np.float64)
        )
        self.tree = cKDTree(self.xy) if len(self.xy) else None

    def query(self, xy: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
        if self.tree is None:
            return np.full(k, np.inf), np.full(k, -1, dtype=int)
        count = min(k, len(self.observations))
        distances, indices = self.tree.query(np.asarray(xy, dtype=np.float64), k=count)
        distances = np.atleast_1d(distances).astype(np.float64)
        indices = np.atleast_1d(indices).astype(int)
        if count < k:
            distances = np.pad(distances, (0, k - count), constant_values=np.inf)
            indices = np.pad(indices, (0, k - count), constant_values=-1)
        return distances, indices


def _is_allowed(provenance: GeometryProvenance, config: LiftingConfig) -> bool:
    if provenance == GeometryProvenance.CURRENT_REAL:
        return config.allow_current_real
    if provenance == GeometryProvenance.CURRENT_FEEDFORWARD_VERIFIED:
        return config.allow_feedforward_verified
    if provenance == GeometryProvenance.VIRTUAL_BA_ONLY:
        return config.allow_virtual_ba_only
    return False


def _mask_label(mask: MaskBundle | None, xy: np.ndarray) -> int:
    if mask is None:
        return int(MaskLabel.STABLE)
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    if x < 0 or y < 0 or y >= mask.labels.shape[0] or x >= mask.labels.shape[1]:
        return int(MaskLabel.INVALID)
    return int(mask.labels[y, x])


def lift_match_set(
    matches: MatchSet,
    observation_index: ReferenceObservationIndex,
    config: LiftingConfig,
    stable_mask: MaskBundle | None = None,
) -> tuple[list[LiftedCorrespondence], LiftingDiagnostics]:
    diagnostics = LiftingDiagnostics(raw_matches=len(matches.confidence))
    output: list[LiftedCorrespondence] = []
    for query_xy, reference_xy, confidence in zip(
        matches.query_xy, matches.reference_xy, matches.confidence, strict=True
    ):
        if confidence < config.min_confidence:
            diagnostics.confidence_filtered += 1
            continue
        if config.require_stable_mask and _mask_label(stable_mask, reference_xy) != int(MaskLabel.STABLE):
            diagnostics.outside_stable_mask += 1
            continue
        distances, indices = observation_index.query(reference_xy, k=2)
        if indices[0] < 0 or distances[0] > config.snap_radius_px:
            diagnostics.no_nearby_observation += 1
            continue
        if np.isfinite(distances[1]) and distances[1] - distances[0] < config.uniqueness_margin_px:
            first = observation_index.observations[int(indices[0])]
            second = observation_index.observations[int(indices[1])]
            if first.point3d_id != second.point3d_id:
                diagnostics.ambiguous_snap += 1
                continue
        observation = observation_index.observations[int(indices[0])]
        if not _is_allowed(observation.provenance, config):
            diagnostics.forbidden_provenance += 1
            continue
        landmark = observation_index.base_map.points3d[observation.point3d_id]
        output.append(
            LiftedCorrespondence(
                query_xy=query_xy,
                reference_xy=reference_xy,
                point3d_id=observation.point3d_id,
                xyz_w=landmark.xyz,
                confidence=float(confidence * observation.confidence),
                reference_id=matches.reference_id,
                snap_distance=float(distances[0]),
                provenance=observation.provenance,
            )
        )
        diagnostics.snap_distances.append(float(distances[0]))
    diagnostics.lifted = len(output)
    diagnostics.unique_point3d = len({item.point3d_id for item in output})
    diagnostics.duplicate_point3d = diagnostics.lifted - diagnostics.unique_point3d
    return output, diagnostics


def aggregate_lifted_correspondences(
    groups: Iterable[Sequence[LiftedCorrespondence]],
    query_merge_radius_px: float = 2.0,
) -> tuple[list[LiftedCorrespondence], LiftingDiagnostics]:
    flattened = [item for group in groups for item in group]
    by_point: dict[int, list[LiftedCorrespondence]] = {}
    for item in flattened:
        by_point.setdefault(item.point3d_id, []).append(item)
    output: list[LiftedCorrespondence] = []
    multi_reference = 0
    for point_id, candidates in by_point.items():
        candidates = sorted(candidates, key=lambda item: item.confidence, reverse=True)
        anchor = candidates[0]
        consistent = [
            item
            for item in candidates
            if np.linalg.norm(item.query_xy - anchor.query_xy) <= query_merge_radius_px
        ]
        reference_ids = {item.reference_id for item in consistent}
        if len(reference_ids) > 1:
            multi_reference += 1
        weights = np.asarray([max(item.confidence, 1e-9) for item in consistent], dtype=np.float64)
        query_xy = np.average(np.stack([item.query_xy for item in consistent]), axis=0, weights=weights)
        reference_xy = anchor.reference_xy
        confidence = float(1.0 - np.prod(1.0 - np.clip(weights, 0.0, 1.0)))
        output.append(
            LiftedCorrespondence(
                query_xy=query_xy,
                reference_xy=reference_xy,
                point3d_id=point_id,
                xyz_w=anchor.xyz_w,
                confidence=confidence,
                reference_id=anchor.reference_id,
                reference_support=len(reference_ids),
                snap_distance=min(item.snap_distance for item in consistent),
                provenance=anchor.provenance,
            )
        )
    output.sort(key=lambda item: item.confidence, reverse=True)
    diagnostics = LiftingDiagnostics(
        raw_matches=len(flattened),
        lifted=len(output),
        unique_point3d=len(output),
        duplicate_point3d=len(flattened) - len(output),
        multi_reference_supported=multi_reference,
        snap_distances=[item.snap_distance for item in output],
    )
    return output, diagnostics
