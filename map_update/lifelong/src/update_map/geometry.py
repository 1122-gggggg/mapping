from __future__ import annotations


import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from .models import Camera, Pose, Sim3


_COLMAP_CAMERA_PARAM_COUNTS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
    "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
    "THIN_PRISM_FISHEYE": 12,
    "RAD_TAN_THIN_PRISM_FISHEYE": 16,
}


def _validate_camera_params(camera: Camera, model: str | None = None) -> str:
    model = camera.model.upper() if model is None else model
    required = _COLMAP_CAMERA_PARAM_COUNTS.get(model)
    if required is None:
        raise ValueError(f"Unsupported camera model: {camera.model}")
    if len(camera.params) < required:
        raise ValueError(
            f"Camera {camera.camera_id} model {camera.model} needs "
            f"{required} params, got {len(camera.params)}"
        )
    return model


def intrinsic_matrix(camera: Camera) -> NDArray[np.float64]:
    model = _validate_camera_params(camera)
    p = camera.params
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        f, cx, cy = p[:3]
        fx = fy = f
    elif model in {
        "PINHOLE",
        "OPENCV",
        "OPENCV_FISHEYE",
        "FULL_OPENCV",
        "FOV",
        "THIN_PRISM_FISHEYE",
        "RAD_TAN_THIN_PRISM_FISHEYE",
    }:
        fx, fy, cx, cy = p[:4]
    else:
        raise ValueError(f"Unsupported camera model: {camera.model}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def opencv_distortion(camera: Camera) -> NDArray[np.float64]:
    model = _validate_camera_params(camera)
    p = camera.params
    if model in {"SIMPLE_PINHOLE", "PINHOLE"}:
        return np.zeros(5, dtype=np.float64)
    if model == "SIMPLE_RADIAL":
        return np.array([p[3], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if model == "RADIAL":
        return np.array([p[3], p[4], 0.0, 0.0, 0.0], dtype=np.float64)
    if model == "OPENCV":
        return np.asarray([p[4], p[5], p[6], p[7], 0.0], dtype=np.float64)
    if model == "FULL_OPENCV":
        return np.asarray(p[4:12], dtype=np.float64)
    # Fisheye and FOV models need dedicated projection functions. For RANSAC initialization,
    # callers should undistort points before invoking solvePnP. Returning zeros is explicit and
    # avoids pretending OpenCV's pinhole distortion vector represents these models.
    return np.zeros(5, dtype=np.float64)


def skew(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def project_points(
    points_w: NDArray[np.float64], pose: Pose, camera: Camera
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    points = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    points_c = pose.world_to_camera(points)
    depth = points_c[:, 2]
    model = camera.model.upper()
    if model not in {
        "SIMPLE_PINHOLE",
        "PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "OPENCV",
        "FULL_OPENCV",
    }:
        raise ValueError(
            f"Projection for camera model {camera.model} requires a dedicated model"
        )
    _validate_camera_params(camera, model)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64), depth
    rvec, tvec = rvec_tvec_from_pose(pose)
    projected, _ = cv2.projectPoints(
        points,
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        intrinsic_matrix(camera),
        opencv_distortion(camera),
    )
    xy = projected.reshape(-1, 2)
    xy[np.abs(depth) <= 1e-12] = np.nan
    return xy, depth


def rotation_angle_deg(rotation_matrix: NDArray[np.float64]) -> float:
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    return float(np.degrees(Rotation.from_matrix(rotation_matrix).magnitude()))


def pose_distance(pose_a: Pose, pose_b: Pose) -> tuple[float, float]:
    relative_rotation = pose_a.R_cw @ pose_b.R_cw.T
    rotation_deg = rotation_angle_deg(relative_rotation)
    translation = float(np.linalg.norm(pose_a.camera_center - pose_b.camera_center))
    return rotation_deg, translation


def pose_error(estimate: Pose, ground_truth: Pose) -> tuple[float, float]:
    return pose_distance(estimate, ground_truth)


def pose_from_rvec_tvec(
    rvec: NDArray[np.float64], tvec: NDArray[np.float64]
) -> Pose:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return Pose(rotation, np.asarray(tvec, dtype=np.float64).reshape(3))


def rvec_tvec_from_pose(pose: Pose) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rvec, _ = cv2.Rodrigues(pose.R_cw)
    return rvec.reshape(3), pose.t_cw.copy()


def umeyama_alignment(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    with_scale: bool = True,
) -> Sim3:
    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape or len(source) < 3:
        raise ValueError("Umeyama alignment requires matching Nx3 arrays with N >= 3")
    mean_source = source.mean(axis=0)
    mean_target = target.mean(axis=0)
    centered_source = source - mean_source
    centered_target = target - mean_target
    covariance = centered_target.T @ centered_source / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1.0
    correction = np.diag(sign)
    rotation = u @ correction @ vt
    if with_scale:
        variance = float(np.mean(np.sum(centered_source**2, axis=1)))
        if variance <= 1e-15:
            raise ValueError("Source points have degenerate variance")
        scale = float(np.sum(singular_values * sign) / variance)
    else:
        scale = 1.0
    translation = mean_target - scale * (rotation @ mean_source)
    return Sim3(scale, rotation, translation)


def ransac_sim3(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    threshold: float,
    iterations: int = 2000,
    with_scale: bool = True,
    random_seed: int = 0,
) -> tuple[Sim3, NDArray[np.bool_], NDArray[np.float64]]:
    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape or len(source) < 3:
        raise ValueError("RANSAC Sim3 requires matching Nx3 arrays with N >= 3")
    rng = np.random.default_rng(random_seed)
    best_inliers = np.zeros(len(source), dtype=bool)
    best_transform: Sim3 | None = None
    for _ in range(iterations):
        sample = rng.choice(len(source), size=3, replace=False)
        try:
            candidate = umeyama_alignment(source[sample], target[sample], with_scale=with_scale)
        except (ValueError, np.linalg.LinAlgError):
            continue
        errors = np.linalg.norm(candidate.transform(source) - target, axis=1)
        inliers = errors <= threshold
        if int(inliers.sum()) > int(best_inliers.sum()):
            best_inliers = inliers
            best_transform = candidate
    if best_transform is None or int(best_inliers.sum()) < 3:
        raise RuntimeError("Unable to estimate a valid Sim3")
    refined = umeyama_alignment(source[best_inliers], target[best_inliers], with_scale=with_scale)
    errors = np.linalg.norm(refined.transform(source) - target, axis=1)
    return refined, errors <= threshold, errors


def yaw_pitch_from_rotation(pose: Pose) -> tuple[float, float]:
    # Camera-to-world orientation is more intuitive for route-view binning.
    rotation_wc = pose.R_cw.T
    yaw, pitch, _ = Rotation.from_matrix(rotation_wc).as_euler("zyx", degrees=True)
    return float(yaw), float(pitch)


def wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def angular_difference_deg(a: float, b: float) -> float:
    return abs(wrap_degrees(a - b))


def robust_characteristic_length(points: NDArray[np.float64]) -> float:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(values) < 2:
        return 1.0
    center = np.median(values, axis=0)
    radii = np.linalg.norm(values - center, axis=1)
    length = float(np.percentile(radii, 75))
    return max(length, 1e-6)
