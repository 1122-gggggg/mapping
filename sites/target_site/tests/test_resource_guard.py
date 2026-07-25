from __future__ import annotations

import sys
from pathlib import Path


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from resource_guard import abort_reason  # noqa: E402


def test_low_memory_aborts_immediately() -> None:
    assert abort_reason(3.9, 0.0, 0) == "low-memory"


def test_swapout_requires_two_consecutive_samples() -> None:
    assert abort_reason(12.0, 40.0, 1) is None
    assert abort_reason(12.0, 40.0, 2) == "sustained-swapout"


def test_healthy_sample_does_not_abort() -> None:
    assert abort_reason(20.0, 0.0, 0) is None
