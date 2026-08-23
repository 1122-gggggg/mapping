from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .config import QualityConfig
from .models import ImageRecord
from .states import QualityStatus


@dataclass(frozen=True)
class ImageQuality:
    blur_laplacian: float
    mean_luminance: float
    dark_clipped_ratio: float
    bright_clipped_ratio: float
    mean_saturation: float
    entropy: float
    dhash: int
    width: int
    height: int
    status: QualityStatus
    reasons: tuple[str, ...]


def _entropy(gray: np.ndarray) -> float:
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
    probabilities = histogram / max(float(histogram.sum()), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def difference_hash(gray: np.ndarray, hash_size: int = 8) -> int:
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(hash_a: int, hash_b: int) -> int:
    return int((hash_a ^ hash_b).bit_count())


def assess_image_quality(path: str | Path, config: QualityConfig) -> ImageQuality:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_luminance = float(gray.mean())
    dark_ratio = float(np.mean(gray <= 5))
    bright_ratio = float(np.mean(gray >= 250))
    saturation = float(hsv[:, :, 1].mean())
    entropy = _entropy(gray)
    reasons: list[str] = []
    hard_reject = False
    if blur < config.blur_reject:
        reasons.append("severe_blur")
        hard_reject = True
    elif blur < config.blur_good:
        reasons.append("borderline_blur")
    if mean_luminance < config.dark_mean_reject:
        reasons.append("underexposed")
        hard_reject = True
    if mean_luminance > config.bright_mean_reject:
        reasons.append("overexposed")
        hard_reject = True
    if max(dark_ratio, bright_ratio) > config.clipped_ratio_reject:
        reasons.append("excessive_clipping")
        hard_reject = True
    if entropy < config.entropy_reject:
        reasons.append("low_entropy")
        hard_reject = True
    if hard_reject:
        status = QualityStatus.REJECT
    elif reasons:
        status = QualityStatus.BORDERLINE
    else:
        status = QualityStatus.GOOD
    return ImageQuality(
        blur_laplacian=blur,
        mean_luminance=mean_luminance,
        dark_clipped_ratio=dark_ratio,
        bright_clipped_ratio=bright_ratio,
        mean_saturation=saturation,
        entropy=entropy,
        dhash=difference_hash(gray),
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        status=status,
        reasons=tuple(reasons),
    )


def enrich_records_with_quality(
    records: Iterable[ImageRecord], config: QualityConfig
) -> list[ImageRecord]:
    output: list[ImageRecord] = []
    for record in records:
        try:
            result = assess_image_quality(record.path, config)
            record.width = result.width
            record.height = result.height
            record.quality_status = result.status
            record.quality_metrics = {
                "blur_laplacian": result.blur_laplacian,
                "mean_luminance": result.mean_luminance,
                "dark_clipped_ratio": result.dark_clipped_ratio,
                "bright_clipped_ratio": result.bright_clipped_ratio,
                "mean_saturation": result.mean_saturation,
                "entropy": result.entropy,
                "dhash": float(result.dhash),
            }
            record.metadata["quality_reasons"] = list(result.reasons)
        except Exception as exc:  # Data audits should keep unreadable files visible.
            record.quality_status = QualityStatus.REJECT
            record.metadata["quality_reasons"] = ["decode_error"]
            record.metadata["quality_error"] = str(exc)
        output.append(record)
    return output


def select_historical_keyframes(
    records: Iterable[ImageRecord], config: QualityConfig
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    selected: list[ImageRecord] = []
    rejected: list[ImageRecord] = []
    by_session: dict[str, list[ImageRecord]] = {}
    for record in records:
        by_session.setdefault(record.session_id, []).append(record)
    for session_records in by_session.values():
        ordered = sorted(
            session_records,
            key=lambda item: (
                item.frame_index if item.frame_index is not None else 10**18,
                item.timestamp if item.timestamp is not None else 10**18,
                str(item.path),
            ),
        )
        last_hash: int | None = None
        last_frame: int | None = None
        for record in ordered:
            if record.quality_status == QualityStatus.REJECT:
                record.metadata["keyframe_rejection"] = "quality_reject"
                rejected.append(record)
                continue
            current_hash = int(record.quality_metrics.get("dhash", -1))
            frame = record.frame_index
            too_close = (
                last_frame is not None
                and frame is not None
                and frame - last_frame < config.min_frame_gap
            )
            near_duplicate = (
                last_hash is not None
                and current_hash >= 0
                and hamming_distance(last_hash, current_hash)
                <= config.duplicate_hamming_threshold
            )
            if too_close or near_duplicate:
                record.metadata["keyframe_rejection"] = (
                    "frame_gap" if too_close else "near_duplicate"
                )
                rejected.append(record)
                continue
            selected.append(record)
            last_hash = current_hash if current_hash >= 0 else None
            last_frame = frame
    return selected, rejected
