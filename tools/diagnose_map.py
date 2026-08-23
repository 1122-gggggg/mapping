#!/usr/bin/env python3
"""Run in-repo Stage 1/2 diagnosis on a finished sparse map.

Does not replace S0-S9. S9 held-out localization stays the release gate.
This is the complementary read-only screen: MapDoctor health + sfm-diagnosis
weak regions, and optional localization-log attribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Final sparse model directory or parent containing 0/",
    )
    parser.add_argument(
        "--backend",
        choices=("colmap", "glomap", "gluemap"),
        default="gluemap",
        help="Producer interface used to load the sparse reconstruction",
    )
    parser.add_argument("--output", type=Path, required=True, help="Directory for sfm-qa reports")
    parser.add_argument("--config", type=Path, default=None, help="Optional MapDoctor JSON settings")
    parser.add_argument(
        "--logs",
        type=Path,
        default=None,
        help="Optional MapDoctor-schema localization CSV for Stage 2",
    )
    parser.add_argument("--database", type=Path, default=None, help="Optional COLMAP database")
    parser.add_argument("--pairs", type=Path, default=None, help="Optional pair table")
    parser.add_argument("--images-manifest", type=Path, default=None, help="Optional image manifest")
    parser.add_argument("--images-dir", type=Path, default=None, help="Optional image directory")
    return parser


def resolve_model(model: Path) -> Path:
    if (model / "cameras.bin").exists() or (model / "cameras.txt").exists():
        return model
    nested = model / "0"
    if (nested / "cameras.bin").exists() or (nested / "cameras.txt").exists():
        return nested
    raise SystemExit(f"No COLMAP cameras.bin or cameras.txt found in {model} or {nested}.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from sfm_qa.pipeline import analyze, is_success_status
    except ImportError as exc:
        raise SystemExit(
            "Diagnosis packages are not installed. From the repo root run: pip install -e '.[dev]'"
        ) from exc

    model = resolve_model(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    report = analyze(
        model,
        backend=args.backend,
        logs_path=args.logs,
        config_path=args.config,
        output_dir=args.output,
        database=args.database,
        pairs=args.pairs,
        images_manifest=args.images_manifest,
        images_dir=args.images_dir,
    )
    print(f"overall_status: {report['overall_status']}")
    print(f"report: {args.output / 'report.json'}")
    return 0 if is_success_status(report["overall_status"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
