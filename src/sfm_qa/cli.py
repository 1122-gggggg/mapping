"""CLI for ``sfm-qa analyze`` / ``check`` / ``check-map`` / ``check-localize``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import analyze, is_success_status


def _add_common(parser: argparse.ArgumentParser, *, logs_required: bool) -> None:
    parser.add_argument("model", type=Path, help="Sparse model directory, e.g. sparse/0")
    parser.add_argument(
        "--backend",
        choices=("colmap", "glomap", "gluemap"),
        required=True,
        help="Map producer interface used to load the reconstruction",
    )
    parser.add_argument(
        "--logs",
        type=Path,
        required=logs_required,
        help="MapDoctor-schema localization CSV (query,success,inliers,...)",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional MapDoctor JSON settings")
    parser.add_argument("--output", type=Path, default=None, help="Write reports into this directory")
    parser.add_argument("--database", type=Path, default=None, help="Optional COLMAP database for build evidence")
    parser.add_argument("--pairs", type=Path, default=None, help="Optional pair table CSV/JSON/JSONL")
    parser.add_argument(
        "--images-manifest",
        type=Path,
        default=None,
        help="Optional registered-image metadata CSV/JSON",
    )
    parser.add_argument("--images-dir", type=Path, default=None, help="Optional image directory for quality evidence")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sfm-qa",
        description="Diagnose an SfM map first, then optionally attribute localization failures.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    analyze_p = sub.add_parser("analyze", help="Stage 1 map diagnosis, then Stage 2 if --logs is given")
    _add_common(analyze_p, logs_required=False)
    check_p = sub.add_parser("check", help="Alias for analyze")
    _add_common(check_p, logs_required=False)
    map_p = sub.add_parser("check-map", help="Screen the reconstruction only")
    _add_common(map_p, logs_required=False)
    loc_p = sub.add_parser("check-localize", help="Screen the map and require localization logs")
    _add_common(loc_p, logs_required=True)
    return parser


def _print_summary(report: dict, output_dir: Path | None, logs_given: bool) -> None:
    mapping = report["map"]
    readiness = mapping["readiness"]
    reconstruction = mapping.get("reconstruction") or {}
    loc = report["localization"]
    print("=== Stage 1: map diagnosis ===")
    print(
        f"map: grade={readiness['grade']} ok={readiness['map_ok']} status={readiness['map_status']}"
    )
    mode = reconstruction.get("diagnostic_mode")
    if mode is not None:
        print(f"diagnostic_mode: {mode}")
    print(
        f"weak_regions: {reconstruction.get('num_weak_regions')} "
        f"weak_images: {reconstruction.get('num_weak_images')}"
    )
    print("=== Stage 2: SfM localization ===")
    if not logs_given or loc is None:
        print("(skipped: no --logs)")
    else:
        print(
            f"localization: ok={loc['localization_ok']} "
            f"rate={loc['strict_success_rate']:.3f} "
            f"status={loc['localization_status']}"
        )
        counts = loc.get("attribution_counts") or {}
        pretty = ", ".join(f"{name}={count}" for name, count in counts.items()) or "(none)"
        print(f"attribution: {pretty}")
    print(f"=== overall_status: {report['overall_status']} ===")
    if output_dir is not None:
        print(f"report: {Path(output_dir) / 'report.json'}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logs_path = None if args.command == "check-map" else args.logs
    if args.command == "check-localize" and logs_path is None:
        raise SystemExit("check-localize requires --logs")
    report = analyze(
        args.model,
        backend=args.backend,
        logs_path=logs_path,
        config_path=args.config,
        output_dir=args.output,
        database=args.database,
        pairs=args.pairs,
        images_manifest=args.images_manifest,
        images_dir=args.images_dir,
    )
    _print_summary(report, args.output, logs_given=logs_path is not None)
    return 0 if is_success_status(report["overall_status"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
