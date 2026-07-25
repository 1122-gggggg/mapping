from __future__ import annotations

from tools.audit_hotspot_repair_reachability import minimum_track_merges_to_p90


def test_minimum_track_merges_matches_p110_formal_requirement() -> None:
    assert minimum_track_merges_to_p90(total_records=755, high_records=128) == 59


def test_minimum_track_merges_matches_p112_formal_requirement() -> None:
    assert minimum_track_merges_to_p90(total_records=924, high_records=204) == 124


def test_minimum_track_merges_is_zero_when_p90_is_already_reachable() -> None:
    assert minimum_track_merges_to_p90(total_records=100, high_records=10) == 0
