from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        destination = Path(log_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(destination, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
