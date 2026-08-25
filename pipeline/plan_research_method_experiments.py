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
DEFAULT_COLMAP = "/home/cihcilab/micromamba/envs/sfm/bin/colmap"
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


def find_license(path: Path) -> Path | None:
    for root in [path.parent, *list(path.parents)[1:4]]:
        for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "COPYING.txt"):
            candidate = root / name
            if candidate.is_file():
                return candidate
    return None


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
    lfoe_license = find_license(lfoe)
    lfoe_ready = lfoe.is_file() and lfoe_license is not None
    dg_root = Path(args.doppelgangers_root).resolve()
    dg_ckpt = Path(args.doppelgangers_checkpoint).resolve()

    baseline_out = experiment_root / "00_baseline_colmap_global_db_reuse"
    lfoe_out = experiment_root / "01_lfoe_glomap_filter"
    dg_out = experiment_root / "02_doppelgangers_pp_frontend"
    gap_out = experiment_root / "03_global_edge_prior_frontend"
    ggpt_out = experiment_root / "05_ggpt_dense_geometry"

    steps = [
        command_step(
            "baseline_colmap_global_db_reuse",
            0,
            "ready",
            "Maintained COLMAP GlobalMapper control run from the same database.",
            [
                args.colmap_command, "global_mapper",
                "--database_path", str(database),
                "--image_path", str(image_root),
                "--output_path", str(baseline_out),
                "--GlobalMapper.ba_refine_focal_length", "0",
                "--GlobalMapper.ba_refine_principal_point", "0",
                "--GlobalMapper.ba_refine_extra_params", "0",
            ],
            ["existing COLMAP database", "image_root"],
            ["Run this before research variants so every method has the same maintained baseline."],
        ),
        command_step(
            "lfoe_outlier_edge_filter",
            1,
            "ready" if lfoe_ready else "blocked",
            "LFOE is a translation-edge-filtering experiment; its upstream release currently has no license grant.",
            [
                str(lfoe), "mapper",
                "--database_path", str(database),
                "--image_path", str(image_root),
                "--output_path", str(lfoe_out),
            ] if lfoe_ready else None,
            [
                "LFOE glomap_filter executable",
                "explicit local/upstream license grant",
                "existing COLMAP database",
                "image_root",
            ],
            [
                f"Detected license: {lfoe_license}" if lfoe_license else "BLOCKED: no license file found.",
                "Compare against COLMAP GlobalMapper with the same S6/S9 gates.",
            ],
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
            "adapter_ready",
            "Wei et al. CVPR 2026: replace per-image retrieval kNN with multi-MST + hop-distance modulation. In-repo selector; GNN weights remain optional.",
            [
                args.python,
                str(BUILD_ROOT / "pipeline" / "pose_graph_init.py"),
                "--scores", str(run_dir / "retrieval_pair_scores.csv"),
                "--required", str(run_dir / "forced_bridges.txt"),
                "--k-msts", "2",
                "--modulation-lambda", "0.5",
                "--output", str(gap_out / "pairs.txt"),
                "--report", str(gap_out / "pose_graph_init.json"),
            ],
            [
                "retrieval_pair_scores.csv (image_a,image_b,score)",
                "forced_bridges.txt as required reverse-direction edges",
                "optional global_edge_prior GNN ranks in the same CSV",
            ],
            [
                "Do not drop S3 VPR-blind forced bridges; pass them as --required.",
                "Do not skip S4 Doppelgangers++. The paper's VisymScenes win is not a substitute for UAV reverse-direction aliasing.",
                "Do not insert into the current DB-reuse sweep; evaluate at the next pair-graph generation.",
                "Paper: https://openaccess.thecvf.com/content/CVPR2026/html/Wei_Global-Aware_Edge_Prioritization_for_Pose_Graph_Initialization_CVPR_2026_paper.html",
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
            "admission_ready",
            "GGPT refines dense feed-forward points under locked SfM guidance. Sidecar admission is in-repo; checkpoints remain external. Not a localization-map fix.",
            [
                args.python,
                str(BUILD_ROOT / "pipeline" / "ggpt_sidecar.py"),
                "--covisibility", str(run_dir / "covisibility_pairs.json"),
                "--poses-locked",
                "--output", str(ggpt_out / "ggpt_admission.json"),
            ],
            [
                "locked S5 poses and seed-identical intrinsics",
                "covisibility pair counts from the sparse model",
                "GGPT code/checkpoints for the actual dense pass after admission",
            ],
            [
                "Reject when the overlap gate fails; GGPT cannot invent missing co-visibility.",
                "Accepted tiles are visualization_only. Do not feed GGPT dense points into EDM/S9.",
                "Paper: https://openaccess.thecvf.com/content/CVPR2026/html/Chen_GGPT_Geometry-Grounded_Point_Transformer_CVPR_2026_paper.html",
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
            "colmap_global": {"path": args.colmap_command},
            "lfoe": {
                **path_status(lfoe, executable=True),
                "license": str(lfoe_license) if lfoe_license else None,
                "ok": lfoe_ready,
            },
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
    parser.add_argument("--colmap-command", default=DEFAULT_COLMAP)
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
