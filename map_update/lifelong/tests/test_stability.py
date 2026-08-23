from update_map.config import StabilityConfig
from update_map.stability import StabilityEvent, StabilityRecord
from update_map.states import ReferenceState


def test_unmatched_does_not_decay_view_utility_by_default() -> None:
    config = StabilityConfig(unmatched_penalty=0.0)
    record = StabilityRecord("ref", geometry_currentness=0.8, historical_view_utility=0.8)
    record.update(StabilityEvent.UNMATCHED, config)
    assert record.historical_view_utility == 0.8


def test_repeated_conflicts_retire_reference() -> None:
    config = StabilityConfig(conflict_penalty=0.4)
    record = StabilityRecord("ref", geometry_currentness=0.7, historical_view_utility=0.7)
    record.update(StabilityEvent.GEOMETRIC_CONFLICT, config)
    assert record.state == ReferenceState.HIST_SUSPECT
    record.update(StabilityEvent.GEOMETRIC_CONFLICT, config)
    assert record.state == ReferenceState.HIST_RETIRED


def test_out_of_order_timestamp_does_not_double_decay() -> None:
    config = StabilityConfig(decay_per_day=0.5)
    record = StabilityRecord("ref", geometry_currentness=0.8, historical_view_utility=0.8)

    record.advance_time(10.0, config)
    record.advance_time(5.0, config)
    record.advance_time(10.0, config)

    assert record.last_timestamp_days == 10.0
    assert record.geometry_currentness == 0.8
