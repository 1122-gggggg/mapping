#!/usr/bin/env python3
"""Run MapDoctor as an optional post-build QA layer for a target-site sparse map.

This does not replace S0-S9 or change the release contract. It is a downstream,
read-only diagnostic hook that consumes an already-built COLMAP/GLOMAP/GLUEMAP
sparse reconstruction.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Final sparse model directory or parent containing 0/")
    parser.add_argument(
        "--backend",
        choices=("colmap", "glomap", "gluemap"),
        default="gluemap",
        help="Producer interface used to load the sparse reconstruction",
    )
    parser.add_argument("--output", type=Path, required=True, help="Directory for MapDoctor JSON/CSV/HTML reports")
    parser.add_argument("--config", type=Path, default=None, help="Optional MapDoctor threshold configuration")
    return parser


def resolve_model(model: Path) -> Path:
    if (model / "cameras.bin").exists() or (model / "cameras.txt").exists():
        return model
    nested = model / "0"
    if (nested / "cameras.bin").exists() or (nested / "cameras.txt").exists():
        return nested
    raise SystemExit(
        f"No COLMAP cameras.bin or cameras.txt found in {model} or {nested}."
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from mapdoctor.cli import main as mapdoctor_main
    except ImportError as exc:
        raise SystemExit(
            "MapDoctor is not installed. From the repo root run: pip install -e ."
        ) from exc

    model = resolve_model(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    mapdoctor_args = [args.backend, str(model), "--output", str(args.output)]
    if args.config is not None:
        mapdoctor_args.extend(["--config", str(args.config)])
    return int(mapdoctor_main(mapdoctor_args))


if __name__ == "__main__":
    raise SystemExit(main())
