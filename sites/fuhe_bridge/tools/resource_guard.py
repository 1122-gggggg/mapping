#!/usr/bin/env python3
"""Guard a long-running SfM process against low RAM and sustained swap-out."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import os
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, TextIO


LOW_MEMORY_GIB = 4.0
WARN_MEMORY_GIB = 6.0
SWAPOUT_MIB_S = 32.0
SUSTAINED_SWAP_SAMPLES = 2
STARTUP_MIN_MEMORY_GIB = 24.0
STARTUP_MIN_VRAM_GIB = 24.0
STARTUP_MIN_SWAP_GIB = 6.0
STARTUP_MIN_DISK_GIB = 100.0
GLOBAL_HEAVY_LOCK = Path(
    "/media/cihcilab/新增磁碟區/sfm_system/建圖/.locks/global_sfm_heavy.lock"
).resolve()


def abort_reason(
    available_gib: float, swapout_mib_s: float, consecutive_swap_samples: int
) -> str | None:
    if available_gib < LOW_MEMORY_GIB:
        return "low-memory"
    if (
        swapout_mib_s >= SWAPOUT_MIB_S
        and consecutive_swap_samples >= SUSTAINED_SWAP_SAMPLES
    ):
        return "sustained-swapout"
    return None


def contract_from_config(
    config: Mapping[str, Any], run_dir: Path
) -> dict[str, Any]:
    """Validate the global heavy-job lock and run-local guard-log contract."""
    root = Path(run_dir).resolve()
    lock_path = Path(str(config.get("resource_lock_path", ""))).resolve()
    log_path = Path(str(config.get("resource_guard_log_path", ""))).resolve()
    guard_path = Path(str(config.get("resource_guard_path", ""))).resolve()
    if lock_path.is_relative_to(root):
        raise ValueError("resource_lock_path must be a global lock outside the run directory")
    if lock_path != GLOBAL_HEAVY_LOCK:
        raise ValueError(
            f"resource_lock_path must be the fixed global heavy-job lock: {GLOBAL_HEAVY_LOCK}"
        )
    if not log_path.is_relative_to(root):
        raise ValueError("resource_guard_log_path must be inside the run directory")
    if not guard_path.is_file():
        raise ValueError(f"resource guard script is absent: {guard_path}")
    return {
        "schema_version": "sfm-resource-contract-v2",
        "run_dir": str(root),
        "lock": {
            "path": str(lock_path),
            "scope": "global_sfm_heavy",
            "exclusive": True,
            "nonblocking": True,
            "outside_run_dir": True,
        },
        "startup_preflight": {
            "mem_available_gib": STARTUP_MIN_MEMORY_GIB,
            "vram_free_gib": STARTUP_MIN_VRAM_GIB,
            "swap_free_gib": STARTUP_MIN_SWAP_GIB,
            "disk_free_gib": STARTUP_MIN_DISK_GIB,
            "fail_closed": True,
        },
        "guard": {
            "script": str(guard_path),
            "log_path": str(log_path),
            "low_memory_gib": LOW_MEMORY_GIB,
            "warn_memory_gib": WARN_MEMORY_GIB,
            "swapout_mib_s": SWAPOUT_MIB_S,
            "sustained_swap_samples": SUSTAINED_SWAP_SAMPLES,
        },
    }


@contextmanager
def exclusive_resource_lock(lock_path: Path) -> Iterator[TextIO]:
    """Hold a non-blocking advisory lock for exactly one heavy run."""
    path = Path(lock_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another heavy SfM process holds {path}") from exc
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()}\n")
        lock_file.flush()
        try:
            yield lock_file
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def _swap_free_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("SwapFree:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("SwapFree is absent from /proc/meminfo")


def _vram_free_gib() -> float:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "free VRAM probe failed: " + (process.stderr.strip() or "nvidia-smi failed")
        )
    try:
        free_mib = [
            float(line.strip())
            for line in process.stdout.splitlines()
            if line.strip()
        ]
    except ValueError as exc:
        raise RuntimeError("free VRAM probe returned a non-numeric value") from exc
    if not free_mib:
        raise RuntimeError("free VRAM probe returned no GPUs")
    return max(free_mib) / 1024


def _disk_free_gib(path: Path) -> float:
    return shutil.disk_usage(Path(path).resolve()).free / 2**30


def startup_preflight(
    disk_path: Path,
    *,
    read_mem_available: Callable[[], float] = _mem_available_gib,
    read_vram_free: Callable[[], float] = _vram_free_gib,
    read_swap_free: Callable[[], float] = _swap_free_gib,
    read_disk_free: Callable[[Path], float] = _disk_free_gib,
) -> dict[str, Any]:
    """Read and enforce startup headroom before importing any heavy runtime."""
    values = {
        "mem_available_gib": float(read_mem_available()),
        "vram_free_gib": float(read_vram_free()),
        "swap_free_gib": float(read_swap_free()),
        "disk_free_gib": float(read_disk_free(Path(disk_path).resolve())),
    }
    minimums = {
        "mem_available_gib": STARTUP_MIN_MEMORY_GIB,
        "vram_free_gib": STARTUP_MIN_VRAM_GIB,
        "swap_free_gib": STARTUP_MIN_SWAP_GIB,
        "disk_free_gib": STARTUP_MIN_DISK_GIB,
    }
    failures = [
        f"{name}={values[name]:.3f}<{minimum:.3f}"
        for name, minimum in minimums.items()
        if values[name] < minimum
    ]
    if failures:
        raise RuntimeError("startup resource preflight failed: " + "; ".join(failures))
    return {
        "ok": True,
        **values,
        "minimums": minimums,
        "disk_path": str(Path(disk_path).resolve()),
    }


def run_global_heavy_job(
    disk_path: Path,
    operation: Callable[[dict[str, Any]], Any],
    *,
    lock_factory: Callable[[Path], Any] = exclusive_resource_lock,
    preflight: Callable[[Path], dict[str, Any]] = startup_preflight,
) -> Any:
    """Hold the one global SfM lock and preflight before a heavy operation."""
    requested_path = Path(disk_path).expanduser().resolve()
    probe_path = requested_path
    while not probe_path.exists() and probe_path != probe_path.parent:
        probe_path = probe_path.parent
    with lock_factory(GLOBAL_HEAVY_LOCK):
        evidence = preflight(probe_path)
        return operation(evidence)


def required_cli_path(argv: Sequence[str], option: str) -> Path:
    """Read one required path option without importing a heavy CLI runtime."""
    for index, argument in enumerate(argv):
        if argument == option and index + 1 < len(argv):
            return Path(argv[index + 1]).expanduser().resolve()
        if argument.startswith(f"{option}="):
            return Path(argument.split("=", 1)[1]).expanduser().resolve()
    raise ValueError(f"required CLI option is absent: {option}")


def _swapout_pages() -> int:
    for line in Path("/proc/vmstat").read_text(encoding="utf-8").splitlines():
        if line.startswith("pswpout "):
            return int(line.split()[1])
    raise RuntimeError("pswpout is absent from /proc/vmstat")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pid", type=int)
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()

    previous_pages = _swapout_pages()
    previous_time = time.monotonic()
    consecutive_swap = 0
    page_mib = os.sysconf("SC_PAGE_SIZE") / 1024**2
    while _alive(args.pid):
        now = time.monotonic()
        pages = _swapout_pages()
        elapsed = max(now - previous_time, 1e-6)
        swapout_rate = max(0, pages - previous_pages) * page_mib / elapsed
        available = _mem_available_gib()
        consecutive_swap = consecutive_swap + 1 if swapout_rate >= SWAPOUT_MIB_S else 0
        stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        print(
            f"{stamp} mem_available_gib={available:.3f} "
            f"swapout_mib_s={swapout_rate:.3f}",
            flush=True,
        )
        reason = abort_reason(available, swapout_rate, consecutive_swap)
        if reason is not None:
            print(f"{stamp} ABORT {reason}", flush=True)
            os.kill(args.pid, signal.SIGINT)
            raise SystemExit(2)
        previous_pages = pages
        previous_time = now
        interval = 5.0 if available < WARN_MEMORY_GIB else args.interval
        time.sleep(interval)
    print("target-completed", flush=True)


if __name__ == "__main__":
    main()
