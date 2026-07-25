from __future__ import annotations

import sys
import fcntl
from pathlib import Path

import pytest


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from resource_guard import (  # noqa: E402
    GLOBAL_HEAVY_LOCK,
    STARTUP_MIN_DISK_GIB,
    STARTUP_MIN_MEMORY_GIB,
    STARTUP_MIN_SWAP_GIB,
    STARTUP_MIN_VRAM_GIB,
    abort_reason,
    contract_from_config,
    exclusive_resource_lock,
    run_global_heavy_job,
    startup_preflight,
)


def test_global_heavy_job_locks_then_preflights_then_runs(tmp_path: Path) -> None:
    events: list[str] = []

    class Lock:
        def __enter__(self):
            events.append("lock")

        def __exit__(self, *_args):
            events.append("unlock")

    result = run_global_heavy_job(
        tmp_path,
        lambda evidence: events.append(f"run:{evidence['ok']}") or "done",
        lock_factory=lambda path: events.append(f"lock-path:{path.name}") or Lock(),
        preflight=lambda path: events.append(f"preflight:{path.name}") or {"ok": True},
    )

    assert result == "done"
    assert events == [
        f"lock-path:{GLOBAL_HEAVY_LOCK.name}",
        "lock",
        f"preflight:{tmp_path.name}",
        "run:True",
        "unlock",
    ]


def test_low_memory_aborts_immediately() -> None:
    assert abort_reason(3.9, 0.0, 0) == "low-memory"


def test_swapout_requires_two_consecutive_samples() -> None:
    assert abort_reason(12.0, 40.0, 1) is None
    assert abort_reason(12.0, 40.0, 2) == "sustained-swapout"


def test_healthy_sample_does_not_abort() -> None:
    assert abort_reason(20.0, 0.0, 0) is None


def test_exclusive_resource_lock_fails_closed_on_contention(tmp_path: Path) -> None:
    lock_path = tmp_path / "run" / "locks" / "gluemap.lock"

    with exclusive_resource_lock(lock_path):
        with lock_path.open("a+", encoding="utf-8") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_contract_requires_fixed_global_lock_and_run_local_guard_log(
    tmp_path: Path,
) -> None:
    run_dir = (tmp_path / "run").resolve()
    run_dir.mkdir()
    guard = TARGET_SITE / "tools" / "resource_guard.py"
    config = {
        "resource_lock_path": str(GLOBAL_HEAVY_LOCK),
        "resource_guard_log_path": str(run_dir / "logs" / "guard.log"),
        "resource_guard_path": str(guard),
    }

    contract = contract_from_config(config, run_dir)

    assert contract["lock"]["path"] == str(GLOBAL_HEAVY_LOCK)
    assert contract["lock"]["scope"] == "global_sfm_heavy"
    assert not GLOBAL_HEAVY_LOCK.is_relative_to(run_dir)
    assert Path(contract["guard"]["log_path"]).is_relative_to(run_dir)

    config["resource_lock_path"] = str(run_dir / "locks" / "local.lock")
    with pytest.raises(ValueError, match="global"):
        contract_from_config(config, run_dir)


def test_startup_preflight_accepts_all_thresholds_via_injected_readers(
    tmp_path: Path,
) -> None:
    evidence = startup_preflight(
        tmp_path,
        read_mem_available=lambda: STARTUP_MIN_MEMORY_GIB,
        read_vram_free=lambda: STARTUP_MIN_VRAM_GIB,
        read_swap_free=lambda: STARTUP_MIN_SWAP_GIB,
        read_disk_free=lambda _path: STARTUP_MIN_DISK_GIB,
    )

    assert evidence["ok"] is True
    assert evidence["mem_available_gib"] == 24.0
    assert evidence["vram_free_gib"] == 24.0
    assert evidence["swap_free_gib"] == 6.0
    assert evidence["disk_free_gib"] == 100.0


@pytest.mark.parametrize(
    "low_resource",
    ["mem_available_gib", "vram_free_gib", "swap_free_gib", "disk_free_gib"],
)
def test_startup_preflight_fails_closed_when_any_resource_is_low(
    tmp_path: Path, low_resource: str
) -> None:
    values = {
        "mem_available_gib": STARTUP_MIN_MEMORY_GIB,
        "vram_free_gib": STARTUP_MIN_VRAM_GIB,
        "swap_free_gib": STARTUP_MIN_SWAP_GIB,
        "disk_free_gib": STARTUP_MIN_DISK_GIB,
    }
    values[low_resource] -= 0.01

    with pytest.raises(RuntimeError, match=low_resource):
        startup_preflight(
            tmp_path,
            read_mem_available=lambda: values["mem_available_gib"],
            read_vram_free=lambda: values["vram_free_gib"],
            read_swap_free=lambda: values["swap_free_gib"],
            read_disk_free=lambda _path: values["disk_free_gib"],
        )
