#!/usr/bin/env python3
"""Plan cheap GLOMAP/triangulation sweeps from an existing COLMAP database.

This script is intentionally non-destructive: it does not run GLOMAP by default
and never touches the existing run outputs.  It writes a reproducible command
matrix so expensive MV-RoMa/GlueMap matching artifacts can be reused.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_GLOMAP = "/home/cihcilab/micromamba/envs/sfm/bin/glomap"
DEFAULT_VARIANTS = [
    "mv2_a0.5:2:0.5:600000",
    "mv3_a0.5:3:0.5:600000",
    "mv2_a1.0:2:1.0:600000",
    "mv3_a1.0:3:1.0:600000",
    "mv2_a0.5_t1m:2:0.5:1000000",
]


@dataclass(frozen=True)
class SweepVariant:
    name: str
    min_views: int
    min_angle: float
    max_tracks: int


def parse_variant(value: str) -> SweepVariant:
    parts = [part.strip() for part in value.split(":")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "variant must be NAME:MIN_VIEWS:MIN_TRIANGULATION_ANGLE:MAX_TRACKS"
        )
    name, min_views, min_angle, max_tracks = parts
    if not name:
        raise argparse.ArgumentTypeError("variant name must not be empty")
    return SweepVariant(name, int(min_views), float(min_angle), int(max_tracks))


def find_default_database(run_dir: Path) -> Path:
    candidates = [
        run_dir / "gluemap" / "database_merged.db",
        run_dir / "gluemap" / "database_sift.db",
        run_dir / "work" / "mvroma" / "database_mvroma_forced.db",
        run_dir / "work" / "tmp" / "database_mvroma_forced_tmp.db",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(
        "no reusable database found; pass --database explicitly"
    )


def file_fingerprint(path: Path) -> dict:
    st = path.stat()
    return {
        "path": str(path),
        "bytes": int(st.st_size),
        "mtime": int(st.st_mtime),
    }


def glomap_command(
    glomap: str,
    database: Path,
    image_root: Path,
    output_dir: Path,
    variant: SweepVariant,
    optimize_intrinsics: int,
    optimize_principal_point: int,
    skip_retriangulation: bool,
) -> list[str]:
    cmd = [
        glomap,
        "mapper",
        "--database_path",
        str(database),
        "--image_path",
        str(image_root),
        "--output_path",
        str(output_dir),
        "--BundleAdjustment.optimize_intrinsics",
        str(int(optimize_intrinsics)),
        "--BundleAdjustment.optimize_principal_point",
        str(int(optimize_principal_point)),
        "--TrackEstablishment.max_num_tracks",
        str(int(variant.max_tracks)),
        "--TrackEstablishment.min_num_view_per_track",
        str(int(variant.min_views)),
        "--Thresholds.min_triangulation_angle",
        str(float(variant.min_angle)),
    ]
    if skip_retriangulation:
        cmd += ["--GlobalMapper.skip_retriangulation", "1"]
    return cmd


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def build_plan(args: argparse.Namespace) -> dict:
    run_dir = Path(args.run_dir).resolve()
    database = Path(args.database).resolve() if args.database else find_default_database(run_dir).resolve()
    image_root = Path(args.image_root).resolve() if args.image_root else (run_dir / "images").resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else (run_dir / "db_reuse_sweeps").resolve()
    variants = [parse_variant(v) for v in (args.variant or DEFAULT_VARIANTS)]

    commands = []
    for variant in variants:
        out = output_root / variant.name
        cmd = glomap_command(
            args.glomap_command,
            database,
            image_root,
            out,
            variant,
            args.optimize_intrinsics,
            args.optimize_principal_point,
            args.skip_retriangulation,
        )
        commands.append({
            "name": variant.name,
            "output_dir": str(out),
            "variant": {
                "min_views": variant.min_views,
                "min_triangulation_angle": variant.min_angle,
                "max_tracks": variant.max_tracks,
            },
            "cmd": cmd,
            "shell": shell_join(cmd),
        })

    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "database": file_fingerprint(database),
        "image_root": str(image_root),
        "image_root_exists": image_root.exists(),
        "output_root": str(output_root),
        "glomap_command": args.glomap_command,
        "optimize_intrinsics": int(args.optimize_intrinsics),
        "optimize_principal_point": int(args.optimize_principal_point),
        "skip_retriangulation": bool(args.skip_retriangulation),
        "commands": commands,
        "notes": [
            "Reuses the existing COLMAP database; does not rerun dense matching.",
            "Compare registered images, points3D, mean/p95 reprojection error, and runtime.",
            "Keep the source database until the selected sweep result is validated.",
        ],
    }


def write_markdown(plan: dict, out_path: Path) -> None:
    lines = [
        "# DB Reuse Sweep Plan",
        "",
        f"- created_at: `{plan['created_at']}`",
        f"- run_dir: `{plan['run_dir']}`",
        f"- database: `{plan['database']['path']}` ({plan['database']['bytes']} bytes)",
        f"- image_root: `{plan['image_root']}`",
        f"- output_root: `{plan['output_root']}`",
        "",
        "## Commands",
        "",
    ]
    for item in plan["commands"]:
        v = item["variant"]
        lines.extend([
            f"### {item['name']}",
            "",
            f"- min_views: `{v['min_views']}`",
            f"- min_triangulation_angle: `{v['min_triangulation_angle']}`",
            f"- max_tracks: `{v['max_tracks']}`",
            "",
            "```bash",
            item["shell"],
            "```",
            "",
        ])
    lines.extend(["## Measurement Checklist", ""])
    for note in plan["notes"]:
        lines.append(f"- {note}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--database", default="")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--glomap-command", default=DEFAULT_GLOMAP)
    parser.add_argument("--optimize-intrinsics", type=int, default=0)
    parser.add_argument("--optimize-principal-point", type=int, default=0)
    parser.add_argument("--skip-retriangulation", action="store_true")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--md-out", default="")
    args = parser.parse_args()

    plan = build_plan(args)
    out_root = Path(plan["output_root"])
    json_out = Path(args.json_out) if args.json_out else out_root / "db_reuse_sweep_plan.json"
    md_out = Path(args.md_out) if args.md_out else out_root / "db_reuse_sweep_plan.md"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(plan, md_out)
    print(f"[plan_db_reuse_sweep] wrote {json_out}")
    print(f"[plan_db_reuse_sweep] wrote {md_out}")


if __name__ == "__main__":
    main()
