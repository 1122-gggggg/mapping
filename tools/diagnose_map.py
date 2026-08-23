#!/usr/bin/env python3
"""Run in-repo Stage 1/2 diagnosis on a map supported by a MapAdapter.

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
        "--map-adapter",
        "--backend",
        dest="map_adapter",
        required=True,
        help="Built-in map adapter or package.module:AdapterClass",
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
    resolved = model.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"Map input does not exist: {resolved}")
    return resolved


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
        backend=args.map_adapter,
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
