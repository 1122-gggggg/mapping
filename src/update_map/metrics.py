from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import PoseGateConfig
from .geometry import intrinsic_matrix, project_points
from .models import Camera, FIMMetrics, LiftedCorrespondence, Pose, PoseQuality


def reprojection_statistics(errors: NDArray[np.float64]) -> dict[str, float]:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"rmse": float("inf"), "p50": float("inf"), "p90": float("inf"), "p95": float("inf")}
    return {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


def convex_hull_ratio(
    points_xy: NDArray[np.float64], image_width: int, image_height: int
) -> float:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3 or image_width <= 0 or image_height <= 0:
        return 0.0
    finite = points[np.all(np.isfinite(points), axis=1)]
    if len(finite) < 3:
        return 0.0
    hull = cv2.convexHull(finite.astype(np.float32))
    area = float(cv2.contourArea(hull))
    return max(0.0, min(1.0, area / float(image_width * image_height)))


def grid_occupancy(
    points_xy: NDArray[np.float64],
    image_width: int,
    image_height: int,
    rows: int = 4,
    cols: int = 4,
) -> int:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0 or image_width <= 0 or image_height <= 0:
        return 0
    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0)
        & (points[:, 0] < image_width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < image_height)
    )
    points = points[valid]
    if len(points) == 0:
        return 0
    col_ids = np.minimum((points[:, 0] / image_width * cols).astype(int), cols - 1)
    row_ids = np.minimum((points[:, 1] / image_height * rows).astype(int), rows - 1)
    return len(set(zip(row_ids.tolist(), col_ids.tolist(), strict=True)))


def positive_depth_ratio(points_w: NDArray[np.float64], pose: Pose) -> float:
    points = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return 0.0
    return float(np.mean(pose.world_to_camera(points)[:, 2] > 0))


def compute_fim(
    points_w: NDArray[np.float64],
    pose: Pose,
    camera: Camera,
    weights: NDArray[np.float64] | None = None,
    pixel_sigma: float = 1.0,
    characteristic_length: float = 1.0,
    regularization: float = 1e-9,
) -> FIMMetrics:
    points = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        matrix = np.zeros((6, 6), dtype=np.float64)
        covariance = np.full((6, 6), np.inf, dtype=np.float64)
        return FIMMetrics(
            matrix=matrix,
            eigenvalues=np.zeros(6, dtype=np.float64),
            condition_number=float("inf"),
            logdet=float("-inf"),
            trace_covariance=float("inf"),
            covariance=covariance,
            marginal_std=np.full(6, np.inf, dtype=np.float64),
        )
    point_weights = (
        np.ones(len(points), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(-1)
    )
    if len(point_weights) != len(points):
        raise ValueError("weights must have one value per point")
    k = intrinsic_matrix(camera)
    fx, fy = k[0, 0], k[1, 1]
    information = np.zeros((6, 6), dtype=np.float64)
    length = max(float(characteristic_length), 1e-9)
    sigma2 = max(float(pixel_sigma) ** 2, 1e-12)
    points_c = pose.world_to_camera(points)
    x = points_c[:, 0]
    y = points_c[:, 1]
    z = points_c[:, 2]
    valid = (z > 1e-9) & (point_weights > 0) & np.isfinite(point_weights)
    if np.any(valid):
        x = x[valid]
        y = y[valid]
        z = z[valid]
        scaled = point_weights[valid] / sigma2
        inv_z = 1.0 / z
        inv_z2 = inv_z * inv_z
        jacobian = np.empty((len(x), 2, 6), dtype=np.float64)
        jacobian[:, 0, 0] = (fx * inv_z) * length
        jacobian[:, 0, 1] = 0.0
        jacobian[:, 0, 2] = (-fx * x * inv_z2) * length
        jacobian[:, 0, 3] = -fx * x * y * inv_z2
        jacobian[:, 0, 4] = fx * (1.0 + x * x * inv_z2)
        jacobian[:, 0, 5] = -fx * y * inv_z
        jacobian[:, 1, 0] = 0.0
        jacobian[:, 1, 1] = (fy * inv_z) * length
        jacobian[:, 1, 2] = (-fy * y * inv_z2) * length
        jacobian[:, 1, 3] = -fy * (1.0 + y * y * inv_z2)
        jacobian[:, 1, 4] = fy * x * y * inv_z2
        jacobian[:, 1, 5] = fy * x * inv_z
        information = np.einsum("n,nij,nik->jk", scaled, jacobian, jacobian)
    symmetric = 0.5 * (information + information.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    positive = eigenvalues[eigenvalues > regularization]
    condition = float(positive.max() / positive.min()) if len(positive) == 6 else float("inf")
    sign, logdet = np.linalg.slogdet(symmetric + regularization * np.eye(6))
    logdet_value = float(logdet) if sign > 0 else float("-inf")
    covariance_normalized = np.linalg.pinv(symmetric + regularization * np.eye(6), rcond=1e-12)
    scale_matrix = np.diag([length, length, length, 1.0, 1.0, 1.0])
    covariance_physical = scale_matrix @ covariance_normalized @ scale_matrix
    diagonal = np.clip(np.diag(covariance_physical), 0.0, np.inf)
    marginal_std = np.sqrt(diagonal)
    return FIMMetrics(
        matrix=symmetric,
        eigenvalues=eigenvalues,
        condition_number=condition,
        logdet=logdet_value,
        trace_covariance=float(np.trace(covariance_physical)),
        covariance=covariance_physical,
        marginal_std=marginal_std,
    )


def evaluate_pose_quality(
    pose: Pose,
    camera: Camera,
    correspondences: Sequence[LiftedCorrespondence],
    inlier_indices: NDArray[np.int64],
    raw_match_count: int,
    lifted_match_count: int,
    pose_mode_count: int,
    characteristic_length: float,
    pixel_sigma: float = 1.0,
) -> PoseQuality:
    inlier_indices = np.asarray(inlier_indices, dtype=np.int64).reshape(-1)
    selected = [correspondences[int(index)] for index in inlier_indices]
    point_ids = {item.point3d_id for item in selected}
    # Input correspondences should already be unique by point3D ID, but this protects the metric.
    unique_selected: list[LiftedCorrespondence] = []
    seen: set[int] = set()
    for item in sorted(selected, key=lambda value: value.confidence, reverse=True):
        if item.point3d_id not in seen:
            unique_selected.append(item)
            seen.add(item.point3d_id)
    points_w = (
        np.stack([item.xyz_w for item in unique_selected], axis=0)
        if unique_selected
        else np.empty((0, 3), dtype=np.float64)
    )
    observed_xy = (
        np.stack([item.query_xy for item in unique_selected], axis=0)
        if unique_selected
        else np.empty((0, 2), dtype=np.float64)
    )
    projected_xy, depth = project_points(points_w, pose, camera)
    errors = np.linalg.norm(projected_xy - observed_xy, axis=1) if len(points_w) else np.empty(0)
    stats = reprojection_statistics(errors)
    weights = np.asarray([item.confidence for item in unique_selected], dtype=np.float64)
    fim = compute_fim(
        points_w,
        pose,
        camera,
        weights=weights,
        pixel_sigma=pixel_sigma,
        characteristic_length=characteristic_length,
    )
    return PoseQuality(
        num_raw_matches=int(raw_match_count),
        num_lifted_matches=int(lifted_match_count),
        num_unique_point3d=len({item.point3d_id for item in correspondences}),
        num_inliers=len(point_ids),
        inlier_ratio=(len(point_ids) / max(len({item.point3d_id for item in correspondences}), 1)),
        reprojection_rmse=stats["rmse"],
        reprojection_p50=stats["p50"],
        reprojection_p90=stats["p90"],
        reprojection_p95=stats["p95"],
        convex_hull_ratio=convex_hull_ratio(observed_xy, camera.width, camera.height),
        grid_occupancy=grid_occupancy(observed_xy, camera.width, camera.height),
        positive_depth_ratio=float(np.mean(depth > 0)) if len(depth) else 0.0,
        independent_reference_support=max(
            len({item.reference_id for item in unique_selected}),
            max((item.reference_support for item in unique_selected), default=0),
        ),
        pose_mode_count=pose_mode_count,
        fim=fim,
    )


def apply_pose_gates(quality: PoseQuality, gates: PoseGateConfig) -> PoseQuality:
    failed: list[str] = []
    if quality.num_unique_point3d < gates.min_unique_point3d:
        failed.append("min_unique_point3d")
    if quality.num_inliers < gates.min_unique_point3d:
        failed.append("min_unique_inliers")
    if quality.inlier_ratio < gates.min_inlier_ratio:
        failed.append("min_inlier_ratio")
    if quality.reprojection_p90 > gates.max_reprojection_p90_px:
        failed.append("max_reprojection_p90_px")
    if quality.convex_hull_ratio < gates.min_convex_hull_ratio:
        failed.append("min_convex_hull_ratio")
    if quality.grid_occupancy < gates.min_grid_occupancy:
        failed.append("min_grid_occupancy")
    if quality.positive_depth_ratio + 1e-12 < gates.required_positive_depth_ratio:
        failed.append("required_positive_depth_ratio")
    if quality.independent_reference_support < gates.min_independent_reference_support:
        failed.append("min_independent_reference_support")
    if quality.pose_mode_count > gates.max_pose_modes:
        failed.append("max_pose_modes")
    if quality.fim is None:
        failed.append("missing_fim")
    else:
        if quality.fim.condition_number > gates.max_fim_condition_number:
            failed.append("max_fim_condition_number")
        if np.max(quality.fim.marginal_std[:3]) > gates.max_translation_std:
            failed.append("max_translation_std")
        rotation_std_deg = np.degrees(np.max(quality.fim.marginal_std[3:]))
        if rotation_std_deg > gates.max_rotation_std_deg:
            failed.append("max_rotation_std_deg")
    quality.failed_gates = failed
    quality.passed = not failed
    return quality


def normalized_reprojection_error(pixels: float, width: int, height: int) -> float:
    diagonal = math.hypot(width, height)
    return float(pixels / diagonal) if diagonal > 0 else float("inf")
