from __future__ import annotations

from pathlib import Path

import numpy as np

from update_map.config import LifelongConfig, UpdateMapConfig, load_config, save_config
from update_map.lifelong import (
    FeatureCandidate,
    FeatureEvent,
    FeatureMemoryRecord,
    FeatureObservation,
    FeatureState,
    MapManagementStrategy,
    PredictiveAdaptiveMapManager,
    classify_feature_events,
    descriptor_uniqueness,
    fit_fremen_model,
    rank_candidates_by_uniqueness,
)


def _manager(count: int, config: LifelongConfig) -> PredictiveAdaptiveMapManager:
    manager = PredictiveAdaptiveMapManager(config)
    for index in range(count):
        manager.register_feature(
            f"f{index:03d}",
            index,
            np.asarray([float(index), 0.0]),
        )
    return manager


def test_classify_feature_events_handles_duplicates_and_unmatched() -> None:
    events = classify_feature_events(
        eligible_feature_ids=["a", "b", "c"],
        matched_feature_ids=["a", "a", "b"],
        inlier_mask=[False, True, False],
    )
    assert events == {
        "a": FeatureEvent.CORRECT,
        "b": FeatureEvent.INCORRECT,
        "c": FeatureEvent.UNMATCHED,
    }


def test_classify_feature_events_ignores_ineligible_matches() -> None:
    events = classify_feature_events(
        eligible_feature_ids=["a"],
        matched_feature_ids=["a", "changed"],
        inlier_mask=[True, True],
    )
    assert events == {"a": FeatureEvent.CORRECT}
    assert "changed" not in events


def test_unmatched_is_temporal_evidence_but_has_zero_scalar_penalty() -> None:
    config = LifelongConfig(strategy="score", unmatched_penalty=0.0, exchange_fraction=0.0)
    manager = _manager(1, config)
    before = manager.records["f000"].score
    plan = manager.update_session(
        events={"f000": FeatureEvent.UNMATCHED},
        timestamp_days=1.0,
        gate_passed=True,
    )
    record = manager.records["f000"]
    assert plan.applied
    assert record.score == before
    assert record.unmatched_count == 1
    assert len(record.observations) == 1
    assert record.observations[0].value == 0.0


def test_correct_and_incorrect_update_scalar_scores() -> None:
    config = LifelongConfig(strategy="score", exchange_fraction=0.0)
    manager = _manager(2, config)
    manager.update_session(
        events={"f000": FeatureEvent.CORRECT, "f001": FeatureEvent.INCORRECT},
        timestamp_days=1.0,
        gate_passed=True,
    )
    assert manager.records["f000"].score == 1.0
    assert manager.records["f001"].score == -1.0


def test_failed_registration_gate_is_exact_noop() -> None:
    config = LifelongConfig(strategy="fremen")
    manager = _manager(5, config)
    before = manager.to_dict()
    plan = manager.update_session(
        events={"f000": FeatureEvent.INCORRECT},
        timestamp_days=2.0,
        candidates=[FeatureCandidate("new", 100, np.asarray([100.0, 0.0]), True)],
        gate_passed=False,
        gate_reason="ambiguous_multimodal",
    )
    assert not plan.applied
    assert plan.metadata["reason"] == "registration_gate_failed"
    assert manager.to_dict() == before


def test_score_strategy_exchanges_exactly_five_percent() -> None:
    config = LifelongConfig(
        strategy="score",
        map_budget=20,
        query_budget=20,
        exchange_fraction=0.05,
        descriptor_metric="l2",
    )
    manager = _manager(20, config)
    manager.records["f000"].score = -10.0
    plan = manager.update_session(
        events={},
        timestamp_days=1.0,
        candidates=[
            FeatureCandidate("new-near", 100, np.asarray([0.1, 0.0]), True),
            FeatureCandidate("new-far", 101, np.asarray([100.0, 0.0]), True),
        ],
        gate_passed=True,
    )
    assert plan.exchange_target == 1
    assert plan.exchange_applied == 1
    assert plan.retired == ["f000"]
    assert plan.activated == ["new-far"]
    assert len(manager.active_ids) == 20


def test_unverified_geometry_is_quarantined_and_never_replaces_active_map() -> None:
    config = LifelongConfig(strategy="score", map_budget=20, exchange_fraction=0.05)
    manager = _manager(20, config)
    plan = manager.update_session(
        events={},
        timestamp_days=1.0,
        candidates=[FeatureCandidate("historical-only", None, None, False)],
        gate_passed=True,
    )
    assert plan.quarantined == ["historical-only"]
    assert plan.retired == []
    assert plan.activated == []
    assert manager.records["historical-only"].state == FeatureState.QUARANTINED
    assert len(manager.active_ids) == 20


def test_descriptor_uniqueness_and_greedy_ranking() -> None:
    baseline = [np.asarray([0.0, 0.0])]
    near = FeatureCandidate("near", 1, np.asarray([0.1, 0.0]), True)
    far = FeatureCandidate("far", 2, np.asarray([10.0, 0.0]), True)
    assert descriptor_uniqueness(far.descriptor, baseline, "l2") > descriptor_uniqueness(
        near.descriptor, baseline, "l2"
    )
    ranked = rank_candidates_by_uniqueness([near, far], baseline, "l2")
    assert [item.feature_id for item, _score in ranked] == ["far", "near"]


def test_fremen_learns_daily_visibility_cycle() -> None:
    config = LifelongConfig(
        candidate_periods_days=[1.0],
        min_temporal_samples=8,
        max_harmonics=1,
        min_observed_cycles=1.0,
    )
    observations: list[FeatureObservation] = []
    for step in range(56):
        timestamp = 0.25 * step
        phase = timestamp % 1.0
        event = FeatureEvent.CORRECT if phase < 0.5 else FeatureEvent.UNMATCHED
        observations.append(
            FeatureObservation(timestamp, event, 1.0 if event == FeatureEvent.CORRECT else 0.0)
        )
    model = fit_fremen_model(observations, config)
    assert model.sample_count == 56
    assert len(model.components) == 1
    assert abs(model.components[0].period_days - 1.0) < 1e-9
    assert model.predict(14.25) > model.predict(14.75) + 0.4


def test_fremen_query_selection_changes_with_time() -> None:
    config = LifelongConfig(
        strategy="fremen",
        query_budget=1,
        exchange_fraction=0.0,
        candidate_periods_days=[1.0],
        min_temporal_samples=8,
        max_harmonics=1,
        min_observed_cycles=1.0,
    )
    manager = _manager(2, config)
    for step in range(56):
        timestamp = 0.25 * step
        phase = timestamp % 1.0
        manager.records["f000"].observe(
            FeatureEvent.CORRECT if phase < 0.5 else FeatureEvent.UNMATCHED,
            timestamp,
            config,
        )
        manager.records["f001"].observe(
            FeatureEvent.UNMATCHED if phase < 0.5 else FeatureEvent.CORRECT,
            timestamp,
            config,
        )
    assert manager.select_features(14.25)[0].feature_id == "f000"
    assert manager.select_features(14.75)[0].feature_id == "f001"


def test_strict_and_aggressive_follow_paper_removal_rules() -> None:
    strict_config = LifelongConfig(strategy="strict", map_budget=3, descriptor_metric="l2")
    strict = _manager(3, strict_config)
    strict_plan = strict.update_session(
        events={"f000": FeatureEvent.UNMATCHED, "f001": FeatureEvent.INCORRECT},
        timestamp_days=1.0,
        candidates=[FeatureCandidate("new", 10, np.asarray([100.0, 0.0]), True)],
        gate_passed=True,
    )
    assert strict_plan.retired == ["f001"]
    assert "f000" in strict.active_ids

    aggressive_config = LifelongConfig(
        strategy="aggressive", map_budget=3, descriptor_metric="l2"
    )
    aggressive = _manager(3, aggressive_config)
    aggressive_plan = aggressive.update_session(
        events={"f000": FeatureEvent.UNMATCHED, "f001": FeatureEvent.INCORRECT},
        timestamp_days=1.0,
        candidates=[
            FeatureCandidate("new-a", 10, np.asarray([100.0, 0.0]), True),
            FeatureCandidate("new-b", 11, np.asarray([200.0, 0.0]), True),
        ],
        gate_passed=True,
    )
    assert set(aggressive_plan.retired) == {"f000", "f001"}


def test_memory_json_round_trip(tmp_path: Path) -> None:
    config = LifelongConfig(strategy="score", exchange_fraction=0.0)
    manager = _manager(1, config)
    manager.update_session(
        events={"f000": FeatureEvent.CORRECT},
        timestamp_days=3.0,
        gate_passed=True,
    )
    path = tmp_path / "memory.json"
    manager.save(path)
    loaded = PredictiveAdaptiveMapManager.load(path, config)
    assert loaded.to_dict() == manager.to_dict()
    assert np.array_equal(loaded.records["f000"].descriptor, np.asarray([0.0, 0.0]))


def test_config_yaml_round_trip_includes_lifelong_section(tmp_path: Path) -> None:
    config = UpdateMapConfig()
    config.lifelong.strategy = MapManagementStrategy.SCORE.value
    config.lifelong.exchange_fraction = 0.1
    path = tmp_path / "config.yaml"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.lifelong.strategy == "score"
    assert loaded.lifelong.exchange_fraction == 0.1
    assert loaded.validate() == []


def test_history_limit_is_enforced() -> None:
    config = LifelongConfig(history_limit=2)
    record = FeatureMemoryRecord("f", point3d_id=1)
    for timestamp in range(3):
        record.observe(FeatureEvent.CORRECT, float(timestamp), config)
    assert [item.timestamp_days for item in record.observations] == [1.0, 2.0]


def test_score_strategy_bootstraps_empty_sidecar_without_retirement() -> None:
    config = LifelongConfig(strategy="score", map_budget=2, descriptor_metric="l2")
    manager = PredictiveAdaptiveMapManager(config)
    plan = manager.update_session(
        events={},
        timestamp_days=1.0,
        candidates=[
            FeatureCandidate("a", 1, np.asarray([0.0]), True),
            FeatureCandidate("b", 2, np.asarray([10.0]), True),
            FeatureCandidate("c", 3, np.asarray([20.0]), True),
        ],
        gate_passed=True,
    )
    assert plan.retired == []
    assert len(plan.activated) == 2
    assert len(manager.active_ids) == 2
    assert plan.exchange_target == 0
