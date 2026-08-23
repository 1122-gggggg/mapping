from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, Sequence

import cv2
import numpy as np

from .config import ChangeConfig
from .models import MaskBundle
from .states import MaskLabel, ReferenceState


class ChangeDetector(Protocol):
    def detect(
        self,
        current_aligned: np.ndarray,
        historical: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> MaskBundle: ...


def _read_image(value: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(value, np.ndarray):
        image = value
    else:
        image = cv2.imread(str(value), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image: {value}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _ssim_map(gray_a: np.ndarray, gray_b: np.ndarray) -> np.ndarray:
    a = gray_a.astype(np.float32) / 255.0
    b = gray_b.astype(np.float32) / 255.0
    c1 = 0.01**2
    c2 = 0.03**2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    return np.clip(numerator / np.maximum(denominator, 1e-12), -1.0, 1.0)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    output = np.zeros(mask.shape, dtype=bool)
    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) >= min_area:
            output[labels == component] = True
    return output


class AlignedImageChangeDetector:
    """Executable baseline for already pose-aligned current/historical views.

    It is intentionally conservative: a pixel is marked changed only when both photometric
    and structural evidence agree. Pixels near either threshold are uncertain, not stable.
    """

    def __init__(self, config: ChangeConfig):
        self.config = config

    def detect(
        self,
        current_aligned: np.ndarray,
        historical: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> MaskBundle:
        current = _read_image(current_aligned)
        old = _read_image(historical)
        if current.shape[:2] != old.shape[:2]:
            raise ValueError("Aligned current and historical images must have the same size")
        current_lab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        old_lab = cv2.cvtColor(old, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        photometric = np.mean(np.abs(current_lab - old_lab), axis=2)
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        old_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
        structural = 1.0 - _ssim_map(current_gray, old_gray)
        valid = (
            np.ones(current.shape[:2], dtype=bool)
            if valid_mask is None
            else np.asarray(valid_mask, dtype=bool)
        )
        if valid.shape != current.shape[:2]:
            raise ValueError("valid_mask shape must match images")
        photo_threshold = self.config.absdiff_threshold
        structure_threshold = self.config.structural_threshold
        changed = valid & (photometric >= photo_threshold) & (structural >= structure_threshold)
        changed = _remove_small_components(changed, self.config.min_component_area)
        if self.config.morphology_kernel > 1:
            kernel = np.ones(
                (self.config.morphology_kernel, self.config.morphology_kernel), dtype=np.uint8
            )
            changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        band = self.config.uncertainty_band
        uncertain = valid & ~changed & (
            (photometric >= max(photo_threshold - band, 0.0))
            | (structural >= max(structure_threshold - band, 0.0))
        )
        stable = valid & ~changed & ~uncertain
        labels = np.full(valid.shape, int(MaskLabel.INVALID), dtype=np.uint8)
        labels[stable] = int(MaskLabel.STABLE)
        labels[changed] = int(MaskLabel.CHANGED)
        labels[uncertain] = int(MaskLabel.UNCERTAIN)
        confidence = np.zeros(valid.shape, dtype=np.float64)
        confidence[valid] = np.clip(
            np.maximum(
                np.abs(photometric[valid] - photo_threshold),
                np.abs(structural[valid] - structure_threshold),
            )
            / max(band, 1e-6),
            0.0,
            1.0,
        )
        return MaskBundle(
            labels=labels,
            confidence=confidence,
            metadata={
                "detector": "aligned_image_baseline",
                "photometric_mean": float(photometric[valid].mean()) if np.any(valid) else None,
                "structural_mean": float(structural[valid].mean()) if np.any(valid) else None,
            },
        )


class DenseFeatureChangeDetector:
    """Feature-aware detector with an injected dense feature extractor.

    The callable must return an HxWxD array. This allows DINOv2, user-provided feature
    renderings, or another dense encoder without hard-coding a model fork in this package.
    """

    def __init__(
        self,
        config: ChangeConfig,
        feature_extractor: Callable[[np.ndarray], np.ndarray],
        feature_threshold: float = 0.25,
    ):
        self.config = config
        self.feature_extractor = feature_extractor
        self.feature_threshold = feature_threshold
        self.structural_detector = AlignedImageChangeDetector(config)

    def detect(
        self,
        current_aligned: np.ndarray,
        historical: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> MaskBundle:
        current = _read_image(current_aligned)
        old = _read_image(historical)
        features_current = np.asarray(self.feature_extractor(current), dtype=np.float32)
        features_old = np.asarray(self.feature_extractor(old), dtype=np.float32)
        if features_current.shape != features_old.shape or features_current.ndim != 3:
            raise ValueError("Dense feature extractor must return matching HxWxD tensors")
        norm_current = features_current / np.maximum(
            np.linalg.norm(features_current, axis=2, keepdims=True), 1e-8
        )
        norm_old = features_old / np.maximum(np.linalg.norm(features_old, axis=2, keepdims=True), 1e-8)
        feature_distance = 1.0 - np.sum(norm_current * norm_old, axis=2)
        if feature_distance.shape != current.shape[:2]:
            feature_distance = cv2.resize(
                feature_distance,
                (current.shape[1], current.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        structural_bundle = self.structural_detector.detect(current, old, valid_mask)
        valid = structural_bundle.labels != int(MaskLabel.INVALID)
        structural_changed = structural_bundle.labels == int(MaskLabel.CHANGED)
        feature_changed = feature_distance >= self.feature_threshold
        changed = valid & structural_changed & feature_changed
        uncertain = valid & ~changed & (
            (structural_bundle.labels == int(MaskLabel.UNCERTAIN))
            | (feature_distance >= self.feature_threshold - self.config.uncertainty_band)
        )
        stable = valid & ~changed & ~uncertain
        labels = np.full(valid.shape, int(MaskLabel.INVALID), dtype=np.uint8)
        labels[stable] = int(MaskLabel.STABLE)
        labels[changed] = int(MaskLabel.CHANGED)
        labels[uncertain] = int(MaskLabel.UNCERTAIN)
        return MaskBundle(
            labels=labels,
            confidence=np.clip(
                np.abs(feature_distance - self.feature_threshold)
                / max(self.config.uncertainty_band, 1e-6),
                0.0,
                1.0,
            ),
            metadata={"detector": "dense_feature_plus_structure"},
        )


class PrecomputedMaskDetector:
    def load(self, path: str | Path) -> MaskBundle:
        source = Path(path)
        if source.suffix.lower() == ".npz":
            payload = np.load(source)
            labels = payload["labels"]
            confidence = payload["confidence"] if "confidence" in payload else None
        elif source.suffix.lower() == ".npy":
            labels = np.load(source)
            confidence = None
        else:
            raw = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError(f"Unable to read mask: {source}")
            if raw.ndim == 3:
                raw = raw[:, :, 0]
            unique = set(np.unique(raw).tolist())
            if unique.issubset({0, 1, 2, 3}):
                labels = raw.astype(np.uint8)
            else:
                labels = np.full(raw.shape, int(MaskLabel.INVALID), dtype=np.uint8)
                labels[raw >= 128] = int(MaskLabel.STABLE)
            confidence = None
        return MaskBundle(labels=labels, confidence=confidence, metadata={"source": str(source)})


def fuse_multiview_masks(
    masks: Sequence[MaskBundle], min_votes: int = 2
) -> MaskBundle:
    if not masks:
        raise ValueError("At least one mask is required")
    shape = masks[0].labels.shape
    if any(mask.labels.shape != shape for mask in masks):
        raise ValueError("All masks must have the same shape")
    stacked = np.stack([mask.labels for mask in masks], axis=0)
    valid_votes = np.sum(stacked != int(MaskLabel.INVALID), axis=0)
    stable_votes = np.sum(stacked == int(MaskLabel.STABLE), axis=0)
    changed_votes = np.sum(stacked == int(MaskLabel.CHANGED), axis=0)
    labels = np.full(shape, int(MaskLabel.INVALID), dtype=np.uint8)
    changed = changed_votes >= min_votes
    stable = (stable_votes >= min_votes) & ~changed
    uncertain = (valid_votes > 0) & ~stable & ~changed
    labels[stable] = int(MaskLabel.STABLE)
    labels[changed] = int(MaskLabel.CHANGED)
    labels[uncertain] = int(MaskLabel.UNCERTAIN)
    confidence = np.maximum(stable_votes, changed_votes) / np.maximum(valid_votes, 1)
    return MaskBundle(
        labels=labels,
        confidence=confidence.astype(np.float64),
        metadata={"views": len(masks), "min_votes": min_votes},
    )


def stable_ratio(mask: MaskBundle) -> float:
    return mask.ratio(int(MaskLabel.STABLE))


def classify_mask_state(mask: MaskBundle, config: ChangeConfig) -> ReferenceState:
    ratio = stable_ratio(mask)
    if ratio >= config.stable_ratio_active:
        return ReferenceState.HIST_STABLE
    if ratio >= config.stable_ratio_candidate:
        return ReferenceState.HIST_CANDIDATE
    return ReferenceState.HISTORICAL_ONLY


def save_mask_bundle(mask: MaskBundle, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"labels": mask.labels}
    if mask.confidence is not None:
        payload["confidence"] = mask.confidence
    np.savez_compressed(destination, **payload)
