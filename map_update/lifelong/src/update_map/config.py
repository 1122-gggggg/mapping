from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class PathsConfig:
    base_map: str = ""
    historical_data: str = ""
    current_validation: str = ""
    current_images: str = ""
    precomputed_edm: str = ""
    precomputed_change_masks: str = ""
    output_root: str = "runs"


@dataclass
class QualityConfig:
    blur_good: float = 1400.0
    blur_reject: float = 150.0
    dark_mean_reject: float = 20.0
    bright_mean_reject: float = 235.0
    clipped_ratio_reject: float = 0.35
    entropy_reject: float = 2.5
    duplicate_hamming_threshold: int = 4
    min_frame_gap: int = 1


@dataclass
class LiftingConfig:
    snap_radius_px: float = 2.0
    uniqueness_margin_px: float = 0.5
    require_stable_mask: bool = True
    allow_current_real: bool = True
    allow_feedforward_verified: bool = True
    allow_virtual_ba_only: bool = False
    min_confidence: float = 0.05
    query_merge_radius_px: float = 2.0


@dataclass
class PoseGateConfig:
    min_unique_point3d: int = 30
    target_unique_point3d: int = 50
    min_inlier_ratio: float = 0.25
    max_reprojection_p90_px: float = 3.0
    min_convex_hull_ratio: float = 0.15
    min_grid_occupancy: int = 6
    required_positive_depth_ratio: float = 1.0
    min_independent_reference_support: int = 2
    max_pose_modes: int = 1
    max_fim_condition_number: float = 1000.0
    max_translation_std: float = 0.2
    max_rotation_std_deg: float = 3.0


@dataclass
class PoseConfig:
    ransac_reprojection_px: float = 4.0
    ransac_iterations: int = 5000
    ransac_confidence: float = 0.999
    refine_loss: str = "huber"
    refine_f_scale: float = 2.0
    cluster_rotation_deg: float = 5.0
    cluster_translation: float = 0.1
    dominant_cluster_ratio: float = 1.5
    characteristic_length: float = 1.0
    pixel_sigma: float = 1.0
    gate: PoseGateConfig = field(default_factory=PoseGateConfig)


@dataclass
class ChangeConfig:
    stable_ratio_active: float = 0.70
    stable_ratio_candidate: float = 0.30
    absdiff_threshold: float = 0.15
    structural_threshold: float = 0.20
    min_multiview_votes: int = 2
    uncertainty_band: float = 0.05
    morphology_kernel: int = 3
    min_component_area: int = 64


@dataclass
class BridgeConfig:
    min_edge_matches: int = 30
    min_edge_inliers: int = 20
    min_edge_inlier_ratio: float = 0.20
    min_edge_spatial_coverage: float = 0.10
    min_edge_confidence: float = 0.50
    max_bridge_depth: int = 5
    min_current_point3d_for_pnp: int = 30
    min_anchor_count: int = 2
    min_disjoint_paths: int = 2
    max_rotation_cycle_deg: float = 3.0
    max_translation_cycle: float = 0.5
    max_scale_cycle_fraction: float = 0.05
    sim3_ransac_threshold: float = 0.2
    sim3_ransac_iterations: int = 2000


@dataclass
class SelectionWeights:
    viewpoint_gain: float = 1.0
    edm_success_gain: float = 3.0
    pose_information_gain: float = 1.0
    stable_ratio: float = 0.5
    redundancy_penalty: float = 1.0
    runtime_cost: float = 0.5
    risk_penalty: float = 2.0


@dataclass
class SelectionConfig:
    budget: int = 500
    min_k_cover: int = 2
    min_total_utility: float = 0.0
    pose_translation_redundancy: float = 0.25
    pose_rotation_redundancy_deg: float = 10.0
    landmark_jaccard_redundancy: float = 0.85
    descriptor_cosine_redundancy: float = 0.95
    weights: SelectionWeights = field(default_factory=SelectionWeights)


@dataclass
class StabilityConfig:
    decay_per_day: float = 0.995
    confirmation_gain: float = 0.10
    conflict_penalty: float = 0.25
    change_penalty: float = 0.30
    false_pose_penalty: float = 0.60
    unmatched_penalty: float = 0.0
    suspect_threshold: float = 0.30
    retire_threshold: float = 0.10
    active_threshold: float = 0.60


@dataclass
class LifelongConfig:
    """Predictive/adaptive feature-map management configuration."""

    strategy: str = "fremen"
    map_budget: int = 500
    query_budget: int = 500
    exchange_fraction: float = 0.05
    min_exchange_count: int = 1
    correct_reward: float = 1.0
    incorrect_penalty: float = 1.0
    unmatched_penalty: float = 0.0
    initial_score: float = 0.0
    score_min: float = -100.0
    score_max: float = 100.0
    min_temporal_samples: int = 8
    max_harmonics: int = 3
    candidate_periods_days: list[float] = field(
        default_factory=lambda: [0.5, 1.0, 7.0, 30.4375, 365.25]
    )
    frequency_grid_size: int = 0
    min_period_days: float = 0.25
    max_period_days: float = 365.25
    min_observed_cycles: float = 0.5
    min_log_period_separation: float = 0.05
    ridge: float = 1e-6
    prediction_min: float = -1.0
    prediction_max: float = 1.0
    descriptor_metric: str = "cosine"
    history_limit: int = 4096

    def validate(self) -> list[str]:
        errors: list[str] = []
        supported = {
            "static",
            "latest",
            "aggressive",
            "strict",
            "summary",
            "score",
            "fremen",
        }
        if self.strategy not in supported:
            errors.append(f"strategy must be one of {sorted(supported)}")
        if self.map_budget < 1:
            errors.append("map_budget must be >= 1")
        if self.query_budget < 0:
            errors.append("query_budget must be non-negative")
        if not (0.0 <= self.exchange_fraction <= 1.0):
            errors.append("exchange_fraction must be in [0, 1]")
        if self.min_exchange_count < 0:
            errors.append("min_exchange_count must be non-negative")
        if self.correct_reward < 0.0:
            errors.append("correct_reward must be non-negative")
        if self.incorrect_penalty < 0.0 or self.unmatched_penalty < 0.0:
            errors.append("feature penalties must be non-negative")
        if self.score_min >= self.score_max:
            errors.append("score_min must be smaller than score_max")
        if self.min_temporal_samples < 1:
            errors.append("min_temporal_samples must be >= 1")
        if self.max_harmonics < 0:
            errors.append("max_harmonics must be non-negative")
        if any(period <= 0.0 for period in self.candidate_periods_days):
            errors.append("candidate_periods_days must contain only positive periods")
        if self.frequency_grid_size < 0:
            errors.append("frequency_grid_size must be non-negative")
        if self.min_period_days <= 0.0 or self.max_period_days < self.min_period_days:
            errors.append("period search bounds are invalid")
        if self.min_observed_cycles < 0.0:
            errors.append("min_observed_cycles must be non-negative")
        if self.min_log_period_separation < 0.0:
            errors.append("min_log_period_separation must be non-negative")
        if self.ridge < 0.0:
            errors.append("ridge must be non-negative")
        if self.prediction_min >= self.prediction_max:
            errors.append("prediction_min must be smaller than prediction_max")
        if self.descriptor_metric not in {"cosine", "l2", "hamming"}:
            errors.append("descriptor_metric must be cosine, l2 or hamming")
        if self.history_limit < 0:
            errors.append("history_limit must be non-negative")
        return errors


@dataclass
class ValidationConfig:
    success_translation_thresholds_m: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0])
    success_rotation_thresholds_deg: list[float] = field(default_factory=lambda: [2.0, 5.0, 10.0])
    common_success_inlier_max_drop_fraction: float = 0.05
    require_zero_new_false_rejections: bool = True
    require_zero_new_confident_wrong_poses: bool = True
    max_p95_latency_increase_fraction: float = 0.25
    min_weak_cell_success_gain: float = 0.0
    min_failure_run_reduction: float = 0.0


@dataclass
class AdapterConfig:
    retrieval_type: str = "precomputed"
    matcher_type: str = "precomputed"
    retrieval_file: str = ""
    matches_root: str = ""
    python_retriever: str = ""
    python_matcher: str = ""
    command_retriever: str = ""
    command_matcher: str = ""
    top_k: int = 30


@dataclass
class RouteCellConfig:
    position_bin_size: float = 2.0
    height_bin_size: float = 2.0
    yaw_bin_deg: float = 15.0
    pitch_bin_deg: float = 15.0


@dataclass
class UpdateMapConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    lifting: LiftingConfig = field(default_factory=LiftingConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    change: ChangeConfig = field(default_factory=ChangeConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    lifelong: LifelongConfig = field(default_factory=LifelongConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    adapters: AdapterConfig = field(default_factory=AdapterConfig)
    route_cells: RouteCellConfig = field(default_factory=RouteCellConfig)
    random_seed: int = 20260818

    def validate(self, require_paths: bool = False) -> list[str]:
        errors: list[str] = []
        if self.pose.gate.min_unique_point3d < 4:
            errors.append("pose.gate.min_unique_point3d must be >= 4")
        if not (0.0 <= self.pose.gate.min_inlier_ratio <= 1.0):
            errors.append("pose.gate.min_inlier_ratio must be in [0, 1]")
        if self.lifting.snap_radius_px <= 0:
            errors.append("lifting.snap_radius_px must be > 0")
        if self.bridge.min_anchor_count < 1:
            errors.append("bridge.min_anchor_count must be >= 1")
        if self.change.stable_ratio_candidate > self.change.stable_ratio_active:
            errors.append("change.stable_ratio_candidate cannot exceed stable_ratio_active")
        if self.selection.budget < 0:
            errors.append("selection.budget must be non-negative")
        errors.extend(f"lifelong.{error}" for error in self.lifelong.validate())
        if require_paths:
            for field_name in ("base_map", "historical_data"):
                value = getattr(self.paths, field_name)
                if not value:
                    errors.append(f"paths.{field_name} is required")
                elif not Path(value).exists():
                    errors.append(f"paths.{field_name} does not exist: {value}")
        return errors

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dict(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge_dict(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _build_config(data: Mapping[str, Any]) -> UpdateMapConfig:
    pose_data = dict(data.get("pose", {}))
    pose_gate = PoseGateConfig(**pose_data.pop("gate", {}))
    selection_data = dict(data.get("selection", {}))
    selection_weights = SelectionWeights(**selection_data.pop("weights", {}))
    return UpdateMapConfig(
        paths=PathsConfig(**data.get("paths", {})),
        quality=QualityConfig(**data.get("quality", {})),
        lifting=LiftingConfig(**data.get("lifting", {})),
        pose=PoseConfig(gate=pose_gate, **pose_data),
        change=ChangeConfig(**data.get("change", {})),
        bridge=BridgeConfig(**data.get("bridge", {})),
        selection=SelectionConfig(weights=selection_weights, **selection_data),
        stability=StabilityConfig(**data.get("stability", {})),
        lifelong=LifelongConfig(**data.get("lifelong", {})),
        validation=ValidationConfig(**data.get("validation", {})),
        adapters=AdapterConfig(**data.get("adapters", {})),
        route_cells=RouteCellConfig(**data.get("route_cells", {})),
        random_seed=int(data.get("random_seed", 20260818)),
    )


def load_config(path: str | Path | None = None) -> UpdateMapConfig:
    default = UpdateMapConfig().as_dict()
    if path is None:
        return _build_config(default)
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError("Config root must be a mapping")
    return _build_config(_merge_dict(default, loaded))


def save_config(config: UpdateMapConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.as_dict(), handle, sort_keys=False, allow_unicode=True)
