#!/usr/bin/env python3
"""Guard a long-running SfM process against low RAM and sustained swap-out."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import signal
import time
from pathlib import Path


LOW_MEMORY_GIB = 4.0
WARN_MEMORY_GIB = 6.0
SWAPOUT_MIB_S = 32.0


def abort_reason(
    available_gib: float, swapout_mib_s: float, consecutive_swap_samples: int
) -> str | None:
    if available_gib < LOW_MEMORY_GIB:
        return "low-memory"
    if swapout_mib_s >= SWAPOUT_MIB_S and consecutive_swap_samples >= 2:
        return "sustained-swapout"
    return None


def _mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


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
