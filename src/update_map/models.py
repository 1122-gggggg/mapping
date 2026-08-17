from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .states import (
    GeometryProvenance,
    ImageSource,
    QualityStatus,
    ReferenceProvenance,
    ReferenceState,
    RegistrationStatus,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def to_jsonable(value: Any) -> Any:
    """Convert nested dataclasses, enums, paths and NumPy values to JSON-safe objects."""

    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class Pose:
    """World-to-camera rigid transform: ``x_c = R_cw @ x_w + t_cw``."""

    R_cw: FloatArray
    t_cw: FloatArray

    def __post_init__(self) -> None:
        r = np.asarray(self.R_cw, dtype=np.float64)
        t = np.asarray(self.t_cw, dtype=np.float64).reshape(3)
        if r.shape != (3, 3):
            raise ValueError(f"R_cw must be 3x3, got {r.shape}")
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(t)):
            raise ValueError("Pose contains non-finite values")
        object.__setattr__(self, "R_cw", r)
        object.__setattr__(self, "t_cw", t)

    @classmethod
    def identity(cls) -> "Pose":
        return cls(np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))

    @classmethod
    def from_matrix(cls, matrix: FloatArray) -> "Pose":
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("Pose matrix must be 4x4")
        return cls(matrix[:3, :3], matrix[:3, 3])

    def as_matrix(self) -> FloatArray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.R_cw
        matrix[:3, 3] = self.t_cw
        return matrix

    @property
    def camera_center(self) -> FloatArray:
        return -(self.R_cw.T @ self.t_cw)

    def world_to_camera(self, points_w: FloatArray) -> FloatArray:
        points = np.asarray(points_w, dtype=np.float64)
        return (self.R_cw @ points.reshape(-1, 3).T).T + self.t_cw

    def camera_to_world(self, points_c: FloatArray) -> FloatArray:
        points = np.asarray(points_c, dtype=np.float64)
        return (self.R_cw.T @ (points.reshape(-1, 3) - self.t_cw).T).T

    def inverse(self) -> "Pose":
        r_wc = self.R_cw.T
        t_wc = -(r_wc @ self.t_cw)
        return Pose(r_wc, t_wc)


@dataclass(frozen=True)
class Sim3:
    """Similarity transform ``x_dst = scale * R @ x_src + t``."""

    scale: float
    R: FloatArray
    t: FloatArray

    def __post_init__(self) -> None:
        r = np.asarray(self.R, dtype=np.float64)
        t = np.asarray(self.t, dtype=np.float64).reshape(3)
        if self.scale <= 0 or not np.isfinite(self.scale):
            raise ValueError("Sim3 scale must be finite and positive")
        if r.shape != (3, 3):
            raise ValueError("Sim3 rotation must be 3x3")
        object.__setattr__(self, "R", r)
        object.__setattr__(self, "t", t)

    @classmethod
    def identity(cls) -> "Sim3":
        return cls(1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))

    def transform(self, points: FloatArray) -> FloatArray:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return self.scale * (self.R @ values.T).T + self.t

    def inverse(self) -> "Sim3":
        r_inv = self.R.T
        scale_inv = 1.0 / self.scale
        t_inv = -scale_inv * (r_inv @ self.t)
        return Sim3(scale_inv, r_inv, t_inv)

    def compose(self, other: "Sim3") -> "Sim3":
        """Return ``self(other(x))``."""

        return Sim3(
            self.scale * other.scale,
            self.R @ other.R,
            self.scale * (self.R @ other.t) + self.t,
        )


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: FloatArray

    def __post_init__(self) -> None:
        params = np.asarray(self.params, dtype=np.float64).reshape(-1)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera dimensions must be positive")
        object.__setattr__(self, "params", params)


@dataclass(frozen=True)
class Observation:
    image_id: int | str
    point2d_idx: int
    xy: FloatArray
    point3d_id: int
    provenance: GeometryProvenance = GeometryProvenance.CURRENT_REAL
    confidence: float = 1.0

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy, dtype=np.float64).reshape(2)
        object.__setattr__(self, "xy", xy)
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Observation confidence must be in [0, 1]")


@dataclass
class MapImage:
    image_id: int
    name: str
    camera_id: int
    pose: Pose
    xys: FloatArray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))
    point3d_ids: IntArray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    def __post_init__(self) -> None:
        self.xys = np.asarray(self.xys, dtype=np.float64).reshape(-1, 2)
        self.point3d_ids = np.asarray(self.point3d_ids, dtype=np.int64).reshape(-1)
        if len(self.xys) != len(self.point3d_ids):
            raise ValueError("xys and point3d_ids must have the same length")


@dataclass
class Landmark:
    point3d_id: int
    xyz: FloatArray
    rgb: NDArray[np.uint8] = field(default_factory=lambda: np.zeros(3, dtype=np.uint8))
    error: float = 0.0
    track: list[tuple[int, int]] = field(default_factory=list)
    provenance: GeometryProvenance = GeometryProvenance.CURRENT_REAL

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=np.float64).reshape(3)
        self.rgb = np.asarray(self.rgb, dtype=np.uint8).reshape(3)


@dataclass
class BaseMap:
    cameras: dict[int, Camera]
    images: dict[int, MapImage]
    points3d: dict[int, Landmark]
    root: Path | None = None
    source_format: str = "unknown"

    def real_point_ids(self) -> set[int]:
        return {
            point_id
            for point_id, point in self.points3d.items()
            if point.provenance
            in {
                GeometryProvenance.CURRENT_REAL,
                GeometryProvenance.CURRENT_FEEDFORWARD_VERIFIED,
            }
        }

    def virtual_point_ids(self) -> set[int]:
        return {
            point_id
            for point_id, point in self.points3d.items()
            if point.provenance == GeometryProvenance.VIRTUAL_BA_ONLY
        }

    def observations_for_image(self, image_id: int) -> list[Observation]:
        image = self.images[image_id]
        observations: list[Observation] = []
        for idx, (xy, point_id) in enumerate(zip(image.xys, image.point3d_ids, strict=True)):
            if point_id < 0 or point_id not in self.points3d:
                continue
            observations.append(
                Observation(
                    image_id=image_id,
                    point2d_idx=idx,
                    xy=xy,
                    point3d_id=int(point_id),
                    provenance=self.points3d[int(point_id)].provenance,
                )
            )
        return observations


@dataclass
class ImageRecord:
    image_id: str
    path: Path
    source: ImageSource
    session_id: str
    sequence_id: str | None = None
    frame_index: int | None = None
    timestamp: float | None = None
    camera_id: str | int | None = None
    width: int | None = None
    height: int | None = None
    quality_status: QualityStatus | None = None
    quality_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    reference_id: str
    score: float


@dataclass
class MatchSet:
    query_id: str
    reference_id: str
    query_xy: FloatArray
    reference_xy: FloatArray
    confidence: FloatArray
    sigma: FloatArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.query_xy = np.asarray(self.query_xy, dtype=np.float64).reshape(-1, 2)
        self.reference_xy = np.asarray(self.reference_xy, dtype=np.float64).reshape(-1, 2)
        self.confidence = np.asarray(self.confidence, dtype=np.float64).reshape(-1)
        if not (
            len(self.query_xy) == len(self.reference_xy) == len(self.confidence)
        ):
            raise ValueError("Match arrays must have equal length")
        if self.sigma is not None:
            sigma = np.asarray(self.sigma, dtype=np.float64)
            if sigma.shape[0] != len(self.confidence):
                raise ValueError("sigma must have one row/value per match")
            self.sigma = sigma


@dataclass(frozen=True)
class LiftedCorrespondence:
    query_xy: FloatArray
    reference_xy: FloatArray
    point3d_id: int
    xyz_w: FloatArray
    confidence: float
    reference_id: str
    reference_support: int = 1
    snap_distance: float = 0.0
    provenance: GeometryProvenance = GeometryProvenance.CURRENT_REAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_xy", np.asarray(self.query_xy, dtype=np.float64).reshape(2))
        object.__setattr__(self, "reference_xy", np.asarray(self.reference_xy, dtype=np.float64).reshape(2))
        object.__setattr__(self, "xyz_w", np.asarray(self.xyz_w, dtype=np.float64).reshape(3))


@dataclass
class FIMMetrics:
    matrix: FloatArray
    eigenvalues: FloatArray
    condition_number: float
    logdet: float
    trace_covariance: float
    covariance: FloatArray
    marginal_std: FloatArray


@dataclass
class PoseQuality:
    num_raw_matches: int = 0
    num_lifted_matches: int = 0
    num_unique_point3d: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_rmse: float = float("inf")
    reprojection_p50: float = float("inf")
    reprojection_p90: float = float("inf")
    reprojection_p95: float = float("inf")
    convex_hull_ratio: float = 0.0
    grid_occupancy: int = 0
    positive_depth_ratio: float = 0.0
    independent_reference_support: int = 0
    pose_mode_count: int = 0
    fim: FIMMetrics | None = None
    loo_rotation_p95_deg: float | None = None
    loo_translation_p95: float | None = None
    passed: bool = False
    failed_gates: list[str] = field(default_factory=list)


@dataclass
class PoseEstimate:
    query_id: str
    pose: Pose | None
    quality: PoseQuality
    status: RegistrationStatus
    inlier_indices: IntArray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    supporting_references: list[str] = field(default_factory=list)
    cluster_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MaskBundle:
    labels: NDArray[np.uint8]
    confidence: FloatArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels, dtype=np.uint8)
        if labels.ndim != 2:
            raise ValueError("Mask labels must be HxW")
        self.labels = labels
        if self.confidence is not None:
            confidence = np.asarray(self.confidence, dtype=np.float64)
            if confidence.shape != labels.shape:
                raise ValueError("Mask confidence shape must match labels")
            self.confidence = confidence

    def ratio(self, label: int) -> float:
        valid = self.labels != 0
        if not np.any(valid):
            return 0.0
        return float(np.mean(self.labels[valid] == label))


@dataclass
class HistoricalReference:
    reference_id: str
    image_path: Path
    pose: Pose
    provenance: ReferenceProvenance
    state: ReferenceState = ReferenceState.HIST_CANDIDATE
    registration_quality: PoseQuality | None = None
    stable_mask_path: Path | None = None
    stable_ratio: float = 0.0
    current_point3d_ids: set[int] = field(default_factory=set)
    bridge_depth: int = 0
    anchor_ids: set[str] = field(default_factory=set)
    bridge_path_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BridgeEdge:
    source: str
    target: str
    confidence: float
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    spatial_coverage: float
    relative_pose: Pose | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def cost(self) -> float:
        return float(-np.log(max(self.confidence, 1e-12)))


@dataclass(frozen=True)
class RouteCell:
    route_segment: str
    position_bin: int
    height_bin: int
    yaw_bin: int
    pitch_bin: int
    direction: str = "unknown"
    condition: str = "default"

    @property
    def key(self) -> str:
        return (
            f"{self.route_segment}:p{self.position_bin}:h{self.height_bin}:"
            f"y{self.yaw_bin}:t{self.pitch_bin}:{self.direction}:{self.condition}"
        )


@dataclass
class UtilityBreakdown:
    viewpoint_gain: float = 0.0
    edm_success_gain: float = 0.0
    pose_information_gain: float = 0.0
    stable_ratio: float = 0.0
    redundancy_penalty: float = 0.0
    runtime_cost: float = 0.0
    risk_penalty: float = 0.0
    total: float = 0.0


@dataclass
class ReferenceCandidate:
    reference: HistoricalReference
    supports_cells: set[str]
    utility: UtilityBreakdown
    cost: float = 1.0
    descriptor: FloatArray | None = None
    visible_point3d_ids: set[int] = field(default_factory=set)


@dataclass
class QueryResult:
    query_id: str
    success: bool
    pose: Pose | None
    quality: PoseQuality
    translation_error: float | None = None
    rotation_error_deg: float | None = None
    confident_wrong_pose: bool = False
    latency_ms: dict[str, float] = field(default_factory=dict)
    route_cell: str | None = None
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    experiment_id: str
    query_results: list[QueryResult]
    aggregate: dict[str, Any]
    references_used: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def stack_xyz(correspondences: Sequence[LiftedCorrespondence]) -> FloatArray:
    if not correspondences:
        return np.empty((0, 3), dtype=np.float64)
    return np.stack([item.xyz_w for item in correspondences], axis=0)


def stack_query_xy(correspondences: Sequence[LiftedCorrespondence]) -> FloatArray:
    if not correspondences:
        return np.empty((0, 2), dtype=np.float64)
    return np.stack([item.query_xy for item in correspondences], axis=0)


def unique_reference_ids(correspondences: Iterable[LiftedCorrespondence]) -> set[str]:
    return {item.reference_id for item in correspondences}
