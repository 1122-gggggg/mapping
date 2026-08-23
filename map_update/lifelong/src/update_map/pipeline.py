from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .adapters.base import Matcher, Retriever
from .change import classify_mask_state, stable_ratio
from .config import UpdateMapConfig
from .geometry import project_points
from .io.hashing import create_map_snapshot, verify_map_snapshot
from .lifting import (
    LiftingDiagnostics,
    ReferenceObservationIndex,
    aggregate_lifted_correspondences,
    lift_match_set,
)
from .models import (
    BaseMap,
    Camera,
    HistoricalReference,
    ImageRecord,
    LiftedCorrespondence,
    MaskBundle,
    PoseEstimate,
    RetrievalResult,
    to_jsonable,
)
from .pose import leave_one_reference_out_stability, localize_with_reference_groups
from .reporting import write_json
from .states import (
    FailureClass,
    GeometryProvenance,
    MaskLabel,
    QualityStatus,
    ReferenceProvenance,
    ReferenceState,
    RegistrationStatus,
)


@dataclass(frozen=True)
class HistoricalAssociation:
    historical_image_id: str
    historical_xy: np.ndarray
    current_point3d_id: int
    confidence: float
    supporting_references: int
    provenance: ReferenceProvenance

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "historical_xy", np.asarray(self.historical_xy, dtype=np.float64).reshape(2)
        )


@dataclass
class DirectRegistrationResult:
    record: ImageRecord
    retrieval: list[RetrievalResult]
    pose_estimate: PoseEstimate
    correspondences_by_reference: dict[str, list[LiftedCorrespondence]]
    aggregated_correspondences: list[LiftedCorrespondence]
    lifting_diagnostics: dict[str, LiftingDiagnostics]
    accepted_associations: list[HistoricalAssociation] = field(default_factory=list)
    failure_class: FailureClass | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class CoreMutationError(RuntimeError):
    pass


class HistoricalAugmentationPipeline:
    def __init__(
        self,
        base_map: BaseMap,
        config: UpdateMapConfig,
        retriever: Retriever,
        matcher: Matcher,
        reference_paths: Mapping[str, str | Path],
    ):
        self.base_map = base_map
        self.config = config
        self.retriever = retriever
        self.matcher = matcher
        self.reference_paths = {key: Path(value) for key, value in reference_paths.items()}
        self.reference_images = {image.name: image for image in base_map.images.values()}
        self._indexes: dict[str, ReferenceObservationIndex] = {}
        self._snapshot = create_map_snapshot(base_map.root) if base_map.root is not None else None

    def verify_core_immutable(self) -> dict[str, object]:
        if self._snapshot is None or self.base_map.root is None:
            return {"ok": True, "reason": "in_memory_map"}
        report = verify_map_snapshot(self.base_map.root, self._snapshot)
        if not report["ok"]:
            raise CoreMutationError(f"Current base map changed: {report}")
        return report

    def _resolve_reference(self, reference_id: str):
        image = self.reference_images.get(reference_id)
        if image is None:
            try:
                image = self.base_map.images.get(int(reference_id))
            except ValueError:
                image = None
        return image

    def _reference_path(self, reference_id: str) -> Path:
        path = self.reference_paths.get(reference_id)
        if path is not None:
            return path
        image = self._resolve_reference(reference_id)
        if image is not None:
            fallback = self.reference_paths.get(image.name)
            if fallback is not None:
                return fallback
        raise FileNotFoundError(f"No image path registered for current reference: {reference_id}")

    def _observation_index(self, reference_id: str) -> ReferenceObservationIndex:
        if reference_id in self._indexes:
            return self._indexes[reference_id]
        image = self._resolve_reference(reference_id)
        if image is None:
            raise KeyError(f"Reference is not part of the current map: {reference_id}")
        observations = self.base_map.observations_for_image(image.image_id)
        index = ReferenceObservationIndex(observations, self.base_map)
        self._indexes[reference_id] = index
        return index

    def direct_register(
        self,
        record: ImageRecord,
        query_camera: Camera,
        top_k: int | None = None,
    ) -> DirectRegistrationResult:
        self.verify_core_immutable()
        if record.quality_status == QualityStatus.REJECT:
            from .models import PoseQuality

            estimate = PoseEstimate(
                query_id=record.image_id,
                pose=None,
                quality=PoseQuality(),
                status=RegistrationStatus.QUALITY_REJECT,
            )
            return DirectRegistrationResult(
                record,
                [],
                estimate,
                {},
                [],
                {},
                failure_class=FailureClass.BAD_IMAGE,
            )
        retrieval = list(
            self.retriever.retrieve(
                record.image_id,
                record.path,
                top_k or self.config.adapters.top_k,
            )
        )
        groups: dict[str, list[LiftedCorrespondence]] = {}
        diagnostics: dict[str, LiftingDiagnostics] = {}
        raw_matches = 0
        retrieval_errors: dict[str, str] = {}
        for result in retrieval:
            reference_id = result.reference_id
            if self._resolve_reference(reference_id) is None:
                retrieval_errors[reference_id] = "not_in_current_map"
                continue
            try:
                matches = self.matcher.match(
                    record.image_id,
                    record.path,
                    reference_id,
                    self._reference_path(reference_id),
                )
                raw_matches += len(matches.confidence)
                lifted, diag = lift_match_set(
                    matches,
                    self._observation_index(reference_id),
                    self.config.lifting,
                    stable_mask=None,
                )
                groups[reference_id] = lifted
                diagnostics[reference_id] = diag
            except Exception as exc:
                retrieval_errors[reference_id] = str(exc)
        aggregated, aggregate_diag = aggregate_lifted_correspondences(
            groups.values(), self.config.lifting.query_merge_radius_px
        )
        estimate = localize_with_reference_groups(
            record.image_id,
            groups,
            query_camera,
            self.config.pose,
            raw_match_count=raw_matches,
        )
        if estimate.pose is not None:
            rotation_loo, translation_loo = leave_one_reference_out_stability(
                estimate.pose, groups, query_camera, self.config.pose
            )
            estimate.quality.loo_rotation_p95_deg = rotation_loo
            estimate.quality.loo_translation_p95 = translation_loo
        accepted: list[HistoricalAssociation] = []
        if estimate.status == RegistrationStatus.DIRECT_STRONG and estimate.pose is not None:
            projected, depth = project_points(
                np.stack([item.xyz_w for item in aggregated], axis=0), estimate.pose, query_camera
            )
            observed = np.stack([item.query_xy for item in aggregated], axis=0)
            errors = np.linalg.norm(projected - observed, axis=1)
            for item, error, point_depth in zip(aggregated, errors, depth, strict=True):
                if error > self.config.pose.ransac_reprojection_px or point_depth <= 0:
                    continue
                if item.provenance not in {
                    GeometryProvenance.CURRENT_REAL,
                    GeometryProvenance.CURRENT_FEEDFORWARD_VERIFIED,
                }:
                    continue
                accepted.append(
                    HistoricalAssociation(
                        historical_image_id=record.image_id,
                        historical_xy=item.query_xy,
                        current_point3d_id=item.point3d_id,
                        confidence=item.confidence,
                        supporting_references=item.reference_support,
                        provenance=ReferenceProvenance.DIRECT,
                    )
                )
        failure_class = self.classify_direct_failure(estimate, record)
        self.verify_core_immutable()
        return DirectRegistrationResult(
            record=record,
            retrieval=retrieval,
            pose_estimate=estimate,
            correspondences_by_reference=groups,
            aggregated_correspondences=aggregated,
            lifting_diagnostics={**diagnostics, "__aggregate__": aggregate_diag},
            accepted_associations=accepted,
            failure_class=failure_class,
            metadata={"retrieval_errors": retrieval_errors},
        )

    @staticmethod
    def classify_direct_failure(
        estimate: PoseEstimate, record: ImageRecord
    ) -> FailureClass | None:
        if record.quality_status == QualityStatus.REJECT:
            return FailureClass.BAD_IMAGE
        if estimate.status == RegistrationStatus.AMBIGUOUS_MULTIMODAL:
            return FailureClass.AMBIGUOUS_ALIAS
        if estimate.status in {RegistrationStatus.DIRECT_STRONG, RegistrationStatus.BRIDGE_STRONG}:
            return None
        # Viewpoint gap versus scene change requires old-old connectivity and/or aligned change
        # evidence. A direct failure alone is deliberately not treated as a scene change.
        if estimate.quality.num_raw_matches > 0 and estimate.quality.num_lifted_matches < 4:
            return FailureClass.VIEWPOINT_GAP
        return FailureClass.UNRESOLVED

    def apply_historical_stable_mask(
        self,
        result: DirectRegistrationResult,
        mask: MaskBundle,
        mask_path: str | Path | None = None,
    ) -> HistoricalReference | None:
        if result.pose_estimate.pose is None or result.pose_estimate.status != RegistrationStatus.DIRECT_STRONG:
            return None
        filtered: list[HistoricalAssociation] = []
        for association in result.accepted_associations:
            x = int(round(float(association.historical_xy[0])))
            y = int(round(float(association.historical_xy[1])))
            if x < 0 or y < 0 or y >= mask.labels.shape[0] or x >= mask.labels.shape[1]:
                continue
            if int(mask.labels[y, x]) == int(MaskLabel.STABLE):
                filtered.append(association)
        result.accepted_associations = filtered
        state = classify_mask_state(mask, self.config.change)
        reference = HistoricalReference(
            reference_id=result.record.image_id,
            image_path=result.record.path,
            pose=result.pose_estimate.pose,
            provenance=ReferenceProvenance.DIRECT,
            state=state,
            registration_quality=result.pose_estimate.quality,
            stable_mask_path=Path(mask_path) if mask_path else None,
            stable_ratio=stable_ratio(mask),
            current_point3d_ids={item.current_point3d_id for item in filtered},
            metadata={
                "association_count": len(filtered),
                "source_session": result.record.session_id,
            },
        )
        if state == ReferenceState.HISTORICAL_ONLY:
            result.failure_class = FailureClass.HISTORICAL_SCENE_CHANGE
        return reference

    def export_direct_result(self, result: DirectRegistrationResult, output: str | Path) -> None:
        payload = {
            "record": result.record,
            "retrieval": result.retrieval,
            "pose_estimate": result.pose_estimate,
            "lifting_diagnostics": result.lifting_diagnostics,
            "accepted_associations": result.accepted_associations,
            "failure_class": result.failure_class,
            "metadata": result.metadata,
        }
        write_json(payload, output)

    def export_sidecar(
        self,
        references: Sequence[HistoricalReference],
        associations: Sequence[HistoricalAssociation],
        output_dir: str | Path,
    ) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        allowed_ids = self.base_map.real_point_ids()
        invalid = [item for item in associations if item.current_point3d_id not in allowed_ids]
        if invalid:
            raise ValueError("Sidecar contains non-current or forbidden point3D IDs")
        write_json([to_jsonable(item) for item in references], destination / "historical_references.json")
        write_json([to_jsonable(item) for item in associations], destination / "observations.json")
        write_json(
            {
                "schema_version": 1,
                "base_map_root": str(self.base_map.root) if self.base_map.root else None,
                "base_map_snapshot": self._snapshot,
                "reference_count": len(references),
                "association_count": len(associations),
                "invariants": {
                    "current_geometry_mutated": False,
                    "historical_only_geometry_promoted": False,
                },
            },
            destination / "manifest.json",
        )
