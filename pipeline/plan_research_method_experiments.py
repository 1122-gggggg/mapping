#!/usr/bin/env python3
"""Plan ordered, non-destructive experiments for new SfM research methods."""
from __future__ import annotations

import argparse
import json
import shlex
import time
from pathlib import Path

from plan_db_reuse_sweep import find_default_database, file_fingerprint


BUILD_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = BUILD_ROOT.parent
TOOLS = BUILD_ROOT / "external_tools"
DEFAULT_RUN = BUILD_ROOT / "runs" / "fuhe_full_no_undistort_official69_20260708"
DEFAULT_EXPERIMENT_ROOT = BUILD_ROOT / "experiments" / "research_methods_20260710"
DEFAULT_GLOMAP = "/home/cihcilab/micromamba/envs/sfm/bin/glomap"
DEFAULT_LFOE = TOOLS / "LFOE-GlobalSfM" / "build" / "glomap_filter"
DEFAULT_DG_ROOT = TOOLS / "doppelgangers-plusplus"
DEFAULT_DG_CKPT = DEFAULT_DG_ROOT / "checkpoints" / "checkpoint-dg+visym.pth"
DGPP_PAPER_URL = "https://doppelgangers25.github.io/doppelgangers_plusplus/static/pdf/doppelgangers_plusplus.pdf"


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def path_status(path: Path, executable: bool = False) -> dict:
    exists = path.exists()
    ok = exists and (not executable or path.is_file())
    return {
        "path": str(path),
        "exists": exists,
        "ok": ok,
    }


def command_step(name: str, priority: int, status: str, rationale: str,
                 command: list[str] | None, requirements: list[str],
                 notes: list[str]) -> dict:
    return {
        "name": name,
        "priority": priority,
        "status": status,
        "rationale": rationale,
        "command": command or [],
        "shell": shell_join(command) if command else "",
        "requirements": requirements,
        "notes": notes,
    }


def build_plan(args: argparse.Namespace) -> dict:
    run_dir = Path(args.run_dir).resolve()
    image_root = Path(args.image_root).resolve() if args.image_root else (run_dir / "images").resolve()
    experiment_root = Path(args.experiment_root).resolve()
    database = Path(args.database).resolve() if args.database else find_default_database(run_dir).resolve()
    lfoe = Path(args.lfoe_command).resolve()
    dg_root = Path(args.doppelgangers_root).resolve()
    dg_ckpt = Path(args.doppelgangers_checkpoint).resolve()

    baseline_out = experiment_root / "00_baseline_glomap_db_reuse"
    lfoe_out = experiment_root / "01_lfoe_glomap_filter"
    dg_out = experiment_root / "02_doppelgangers_pp_frontend"
    gap_out = experiment_root / "03_global_edge_prior_frontend"
    trip_out = experiment_root / "04_trip_translation_averaging"
    ggpt_out = experiment_root / "05_ggpt_dense_geometry"

    steps = [
        command_step(
            "baseline_glomap_db_reuse",
            0,
            "ready",
            "Control run from the same DB/H5 artifacts; no expensive front-end rerun.",
            [
                args.glomap_command, "mapper",
                "--database_path", str(database),
                "--image_path", str(image_root),
                "--output_path", str(baseline_out),
            ],
            ["existing COLMAP database", "image_root"],
            ["Run this before any research variant so every method has the same baseline."],
        ),
        command_step(
            "lfoe_outlier_edge_filter",
            1,
            "ready" if lfoe.exists() else "blocked",
            "LFOE directly targets outlier relative-translation edges in global SfM and reuses the existing database.",
            [
                str(lfoe), "mapper",
                "--database_path", str(database),
                "--image_path", str(image_root),
                "--output_path", str(lfoe_out),
            ] if lfoe.exists() else None,
            ["LFOE glomap_filter executable", "existing COLMAP database", "image_root"],
            ["Compare registered images, points3D, reprojection stats, and holdout localization."],
        ),
        command_step(
            "doppelgangers_pp_pair_filter",
            2,
            "ready" if dg_root.exists() and dg_ckpt.exists() else "blocked",
            "Doppelgangers++ filters visually aliased pairs before dense matching; useful for repeated structures.",
            [
                args.python,
                str(BUILD_ROOT / "pipeline" / "build_localizable_map_core.py"),
                "--site-name", "fuhe_dgpp_experiment_20260710",
                "--work-dir", str(dg_out),
                "--image-root", str(image_root),
                "--resume",
                "--doppelgangers-root", str(dg_root),
                "--doppelgangers-checkpoint", str(dg_ckpt),
                "--doppelgangers-threshold", str(args.doppelgangers_threshold),
                "--doppelgangers-filter-scope", args.doppelgangers_filter_scope,
            ] if dg_root.exists() and dg_ckpt.exists() else None,
            ["Doppelgangers++ repo", "checkpoint-dg+visym.pth", "front-end rebuild budget"],
            [
                f"Source paper: {DGPP_PAPER_URL}",
                "This is a front-end experiment and may rerun expensive matching.",
                "The preserved core is called directly; the removed one-click wrapper is not used.",
                "Use only after DB-reuse/LFOE variants are measured.",
            ],
        ),
        command_step(
            "global_aware_edge_prioritization",
            3,
            "blocked",
            "Promising for initial pose-graph construction, but it is a front-end candidate-edge selection method.",
            None,
            ["global_edge_prior repo/model", "adapter from retrieval pairs to local pair graph"],
            [
                "Do not insert into the current DB-reuse sweep.",
                "Evaluate in the next full DB/H5 generation cycle.",
            ],
        ),
        command_step(
            "trip_translation_averaging",
            4,
            "blocked",
            "TriP is a robust translation averaging solver; useful as a diagnostic/alternative backend, not a direct GLOMAP drop-in yet.",
            None,
            ["TriP implementation", "export relative translation graph", "adapter back to COLMAP/GLOMAP model"],
            [
                "Benchmark offline first; do not replace GLOMAP global positioning without an adapter and gate.",
            ],
        ),
        command_step(
            "ggpt_dense_geometry",
            5,
            "blocked",
            "GGPT improves dense feed-forward reconstruction with sparse geometry guidance; not a primary localization-map fix.",
            None,
            ["GGPT code/checkpoints", "dense-output evaluation target"],
            [
                "Use only for dense visual QA or changed-region visualization after localization gates pass.",
            ],
        ),
    ]

    return {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "database": file_fingerprint(database),
        "image_root": str(image_root),
        "experiment_root": str(experiment_root),
        "preflight": {
            "glomap": {"path": args.glomap_command},
            "lfoe": path_status(lfoe, executable=True),
            "doppelgangers_root": path_status(dg_root),
            "doppelgangers_checkpoint": path_status(dg_ckpt),
        },
        "steps": steps,
        "process_policy": [
            "Original production outputs and symlinks stay unchanged.",
            "Each method writes to its own experiment subdirectory.",
            "A method is promoted only after the active site's S0-S9 release gates pass.",
            "DB/H5 artifacts are preserved throughout the experiment.",
        ],
    }


def write_markdown(plan: dict, path: Path) -> None:
    lines = ["# Research Method Experiment Plan", ""]
    lines.append(f"- created_at: `{plan['created_at']}`")
    lines.append(f"- run_dir: `{plan['run_dir']}`")
    lines.append(f"- database: `{plan['database']['path']}` ({plan['database']['bytes']} bytes)")
    lines.append(f"- image_root: `{plan['image_root']}`")
    lines.append(f"- experiment_root: `{plan['experiment_root']}`")
    lines.append("")
    lines.append("## Process Policy")
    lines.append("")
    for item in plan["process_policy"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Ordered Steps")
    lines.append("")
    lines.append("| Priority | Method | Status | Rationale |")
    lines.append("|---:|---|---|---|")
    for step in plan["steps"]:
        lines.append(f"| {step['priority']} | {step['name']} | {step['status']} | {step['rationale']} |")
    lines.append("")
    lines.append("## Commands")
    lines.append("")
    for step in plan["steps"]:
        lines.append(f"### {step['priority']}. {step['name']}")
        lines.append("")
        lines.append(f"- status: `{step['status']}`")
        lines.append("- requirements:")
        for req in step["requirements"]:
            lines.append(f"  - {req}")
        lines.append("- notes:")
        for note in step["notes"]:
            lines.append(f"  - {note}")
        if step["shell"]:
            lines.append("")
            lines.append("```bash")
            lines.append(step["shell"])
            lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="/usr/bin/python3.12")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--database", default="")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--glomap-command", default=DEFAULT_GLOMAP)
    parser.add_argument("--lfoe-command", default=str(DEFAULT_LFOE))
    parser.add_argument("--doppelgangers-root", default=str(DEFAULT_DG_ROOT))
    parser.add_argument("--doppelgangers-checkpoint", default=str(DEFAULT_DG_CKPT))
    parser.add_argument("--doppelgangers-threshold", type=float, default=0.7)
    parser.add_argument("--doppelgangers-filter-scope", choices=["all", "cross_video", "cross_direction"], default="cross_video")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--md-out", default="")
    args = parser.parse_args()

    plan = build_plan(args)
    out_root = Path(plan["experiment_root"])
    json_out = Path(args.json_out) if args.json_out else out_root / "research_method_experiment_plan.json"
    md_out = Path(args.md_out) if args.md_out else out_root / "research_method_experiment_plan.md"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(plan, md_out)
    print(f"[plan_research_method_experiments] wrote {json_out}")
    print(f"[plan_research_method_experiments] wrote {md_out}")


if __name__ == "__main__":
    main()
