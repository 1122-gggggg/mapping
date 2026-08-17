from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .config import PoseConfig
from .geometry import intrinsic_matrix, opencv_distortion, pose_distance, pose_from_rvec_tvec, rvec_tvec_from_pose
from .metrics import apply_pose_gates, evaluate_pose_quality
from .models import Camera, LiftedCorrespondence, Pose, PoseEstimate
from .states import RegistrationStatus


@dataclass
class PoseHypothesis:
    reference_id: str
    pose: Pose
    inlier_indices: NDArray[np.int64]
    score: float
    median_reprojection: float


@dataclass
class PoseCluster:
    cluster_id: int
    hypotheses: list[PoseHypothesis]
    score: float

    @property
    def reference_ids(self) -> set[str]:
        return {item.reference_id for item in self.hypotheses}


def _arrays(correspondences: Sequence[LiftedCorrespondence]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points3d = np.stack([item.xyz_w for item in correspondences], axis=0).astype(np.float64)
    points2d = np.stack([item.query_xy for item in correspondences], axis=0).astype(np.float64)
    weights = np.asarray([max(item.confidence, 1e-6) for item in correspondences], dtype=np.float64)
    return points3d, points2d, weights


def _project_cv(points3d: np.ndarray, pose: Pose, camera: Camera) -> np.ndarray:
    rvec, tvec = rvec_tvec_from_pose(pose)
    projected, _ = cv2.projectPoints(
        points3d,
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        intrinsic_matrix(camera),
        opencv_distortion(camera),
    )
    return projected.reshape(-1, 2)


def solve_pnp_ransac(
    correspondences: Sequence[LiftedCorrespondence],
    camera: Camera,
    config: PoseConfig,
    initial_pose: Pose | None = None,
) -> tuple[Pose, NDArray[np.int64]] | None:
    if len(correspondences) < 4:
        return None
    points3d, points2d, _ = _arrays(correspondences)
    use_guess = initial_pose is not None
    if initial_pose is not None:
        rvec, tvec = rvec_tvec_from_pose(initial_pose)
    else:
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.zeros(3, dtype=np.float64)
    success, rvec_out, tvec_out, inliers = cv2.solvePnPRansac(
        objectPoints=points3d,
        imagePoints=points2d,
        cameraMatrix=intrinsic_matrix(camera),
        distCoeffs=opencv_distortion(camera),
        rvec=rvec.reshape(3, 1),
        tvec=tvec.reshape(3, 1),
        useExtrinsicGuess=use_guess,
        iterationsCount=config.ransac_iterations,
        reprojectionError=config.ransac_reprojection_px,
        confidence=config.ransac_confidence,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < 4:
        return None
    pose = pose_from_rvec_tvec(rvec_out, tvec_out)
    return pose, inliers.reshape(-1).astype(np.int64)


def refine_pose_weighted(
    pose: Pose,
    correspondences: Sequence[LiftedCorrespondence],
    camera: Camera,
    config: PoseConfig,
) -> Pose:
    if len(correspondences) < 4:
        return pose
    points3d, points2d, weights = _arrays(correspondences)
    rvec, tvec = rvec_tvec_from_pose(pose)
    initial = np.concatenate([rvec, tvec])
    sqrt_weights = np.sqrt(np.clip(weights, 1e-6, 1.0))

    def residual(parameters: np.ndarray) -> np.ndarray:
        candidate = pose_from_rvec_tvec(parameters[:3], parameters[3:])
        projected = _project_cv(points3d, candidate, camera)
        return ((projected - points2d) * sqrt_weights[:, None]).reshape(-1)

    result = least_squares(
        residual,
        initial,
        loss=config.refine_loss,
        f_scale=config.refine_f_scale,
        max_nfev=200,
    )
    return pose_from_rvec_tvec(result.x[:3], result.x[3:])


def recompute_inliers(
    pose: Pose,
    correspondences: Sequence[LiftedCorrespondence],
    camera: Camera,
    threshold_px: float,
) -> NDArray[np.int64]:
    if not correspondences:
        return np.empty(0, dtype=np.int64)
    points3d, points2d, _ = _arrays(correspondences)
    errors = np.linalg.norm(_project_cv(points3d, pose, camera) - points2d, axis=1)
    return np.flatnonzero(errors <= threshold_px).astype(np.int64)


def solve_reference_hypotheses(
    groups: Mapping[str, Sequence[LiftedCorrespondence]],
    camera: Camera,
    config: PoseConfig,
    min_correspondences: int = 6,
) -> list[PoseHypothesis]:
    hypotheses: list[PoseHypothesis] = []
    for reference_id, correspondences in groups.items():
        if len(correspondences) < min_correspondences:
            continue
        solution = solve_pnp_ransac(correspondences, camera, config)
        if solution is None:
            continue
        pose, inliers = solution
        refined = refine_pose_weighted(pose, [correspondences[int(i)] for i in inliers], camera, config)
        inliers = recompute_inliers(refined, correspondences, camera, config.ransac_reprojection_px)
        if len(inliers) < 4:
            continue
        points3d, points2d, _ = _arrays(correspondences)
        errors = np.linalg.norm(_project_cv(points3d, refined, camera) - points2d, axis=1)
        median = float(np.median(errors[inliers]))
        score = float(len(inliers) / max(median, 0.25))
        hypotheses.append(PoseHypothesis(reference_id, refined, inliers, score, median))
    return hypotheses


def cluster_pose_hypotheses(
    hypotheses: Sequence[PoseHypothesis],
    rotation_threshold_deg: float,
    translation_threshold: float,
) -> list[PoseCluster]:
    parent = list(range(len(hypotheses)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(len(hypotheses)):
        for j in range(i + 1, len(hypotheses)):
            rotation, translation = pose_distance(hypotheses[i].pose, hypotheses[j].pose)
            if rotation <= rotation_threshold_deg and translation <= translation_threshold:
                union(i, j)
    grouped: dict[int, list[PoseHypothesis]] = {}
    for index, hypothesis in enumerate(hypotheses):
        grouped.setdefault(find(index), []).append(hypothesis)
    clusters = [
        PoseCluster(cluster_id=idx, hypotheses=items, score=float(sum(item.score for item in items)))
        for idx, items in enumerate(grouped.values())
    ]
    return sorted(clusters, key=lambda item: item.score, reverse=True)


def count_competitive_modes(clusters: Sequence[PoseCluster], dominant_ratio: float) -> int:
    if not clusters:
        return 0
    best = clusters[0].score
    threshold = best / max(dominant_ratio, 1.0)
    return sum(cluster.score >= threshold for cluster in clusters)


def localize_with_reference_groups(
    query_id: str,
    groups: Mapping[str, Sequence[LiftedCorrespondence]],
    camera: Camera,
    config: PoseConfig,
    raw_match_count: int | None = None,
) -> PoseEstimate:
    hypotheses = solve_reference_hypotheses(groups, camera, config)
    if not hypotheses:
        from .models import PoseQuality

        return PoseEstimate(
            query_id=query_id,
            pose=None,
            quality=PoseQuality(
                num_raw_matches=raw_match_count or 0,
                num_lifted_matches=sum(len(items) for items in groups.values()),
            ),
            status=RegistrationStatus.DIRECT_FAILED,
        )
    clusters = cluster_pose_hypotheses(
        hypotheses,
        config.cluster_rotation_deg,
        config.cluster_translation,
    )
    mode_count = count_competitive_modes(clusters, config.dominant_cluster_ratio)
    dominant = clusters[0]
    dominant_references = dominant.reference_ids
    from .lifting import aggregate_lifted_correspondences

    pooled, _ = aggregate_lifted_correspondences(
        [groups[reference_id] for reference_id in dominant_references],
        query_merge_radius_px=2.0,
    )
    seed = dominant.hypotheses[0].pose
    solution = solve_pnp_ransac(pooled, camera, config, initial_pose=seed)
    if solution is None:
        from .models import PoseQuality

        return PoseEstimate(
            query_id=query_id,
            pose=None,
            quality=PoseQuality(
                num_raw_matches=raw_match_count or 0,
                num_lifted_matches=len(pooled),
                pose_mode_count=mode_count,
            ),
            status=RegistrationStatus.DIRECT_FAILED,
            supporting_references=sorted(dominant_references),
        )
    pose, inliers = solution
    pose = refine_pose_weighted(pose, [pooled[int(i)] for i in inliers], camera, config)
    inliers = recompute_inliers(pose, pooled, camera, config.ransac_reprojection_px)
    quality = evaluate_pose_quality(
        pose=pose,
        camera=camera,
        correspondences=pooled,
        inlier_indices=inliers,
        raw_match_count=raw_match_count or sum(len(items) for items in groups.values()),
        lifted_match_count=len(pooled),
        pose_mode_count=mode_count,
        characteristic_length=config.characteristic_length,
        pixel_sigma=config.pixel_sigma,
    )
    quality = apply_pose_gates(quality, config.gate)
    if mode_count > 1:
        status = RegistrationStatus.AMBIGUOUS_MULTIMODAL
    elif quality.passed:
        status = RegistrationStatus.DIRECT_STRONG
    else:
        status = RegistrationStatus.DIRECT_WEAK
    return PoseEstimate(
        query_id=query_id,
        pose=pose,
        quality=quality,
        status=status,
        inlier_indices=inliers,
        supporting_references=sorted(dominant_references),
        cluster_id=dominant.cluster_id,
        metadata={
            "hypothesis_count": len(hypotheses),
            "cluster_scores": [cluster.score for cluster in clusters],
        },
    )


def leave_one_reference_out_stability(
    full_pose: Pose,
    groups: Mapping[str, Sequence[LiftedCorrespondence]],
    camera: Camera,
    config: PoseConfig,
) -> tuple[float | None, float | None]:
    if len(groups) < 3:
        return None, None
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    for omitted in groups:
        subset = {key: value for key, value in groups.items() if key != omitted}
        result = localize_with_reference_groups("loo", subset, camera, config)
        if result.pose is None:
            continue
        rotation, translation = pose_distance(full_pose, result.pose)
        rotation_errors.append(rotation)
        translation_errors.append(translation)
    if not rotation_errors:
        return None, None
    return float(np.percentile(rotation_errors, 95)), float(np.percentile(translation_errors, 95))


def average_poses(poses: Sequence[Pose], weights: Sequence[float] | None = None) -> Pose:
    if not poses:
        raise ValueError("At least one pose is required")
    values = np.ones(len(poses), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    values /= values.sum()
    rotations_wc = Rotation.from_matrix(np.stack([pose.R_cw.T for pose in poses]))
    quaternions = rotations_wc.as_quat()  # xyzw
    reference = quaternions[0]
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index], reference) < 0:
            quaternions[index] *= -1
    quaternion = np.average(quaternions, axis=0, weights=values)
    quaternion /= np.linalg.norm(quaternion)
    rotation_wc = Rotation.from_quat(quaternion).as_matrix()
    centers = np.stack([pose.camera_center for pose in poses])
    center = np.average(centers, axis=0, weights=values)
    rotation_cw = rotation_wc.T
    translation_cw = -(rotation_cw @ center)
    return Pose(rotation_cw, translation_cw)
