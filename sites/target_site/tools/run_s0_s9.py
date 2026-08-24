#!/usr/bin/env python3
"""Run the existing target_site S0-S9 CLIs in order. Stop on first non-PASS gate.

Heavy stages keep their documented required args. This script does not invent
video paths or new gate logic. --run-dir is required; missing inputs fail closed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parent


def gate_passed(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing gate {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return False, f"unreadable gate {path}: {error}"
    status = payload.get("status")
    if status != "PASS":
        return False, f"{path.name} status={status!r}"
    return True, "PASS"


def require_path(path: Path | None, label: str) -> Path:
    if path is None:
        raise SystemExit(f"missing required argument for this stage: {label}")
    if not path.exists():
        raise SystemExit(f"missing {label}: {path}")
    return path


def plan_stages(args: argparse.Namespace) -> list[dict]:
    run_dir = args.run_dir.resolve()
    run_name = run_dir.name
    python = args.python
    images = run_dir / "images"
    forced_txt = run_dir / "forced_bridges.txt"
    forced_json = run_dir / "forced_bridges.json"
    frame_manifest = run_dir / "frame_manifest.json"
    corpus_manifest = run_dir / "corpus_manifest.json"
    gates = run_dir / "gates"
    model = args.model or (run_dir / "final_model")
    lfoe_model = run_dir / "lfoe_model"
    tracking = args.tracking_bundle or (
        run_dir / "edm" / "target_site_v1_seed_tracking.pt"
    )
    edm = args.edm_bundle or (run_dir / "edm" / "target_site_v1_reloc_map_edm.pt")

    def tool(name: str) -> str:
        return str(TOOLS / name)

    return [
        {
            "stage": "S0",
            "gate": gates / "S0_corpus.json",
            "cmd": [python, tool("s0_corpus_lock.py"), "--run-name", run_name],
        },
        {
            "stage": "S1",
            "gate": gates / "S1_motion.json",
            "cmd": [python, tool("s1_motion_scan.py"), "--run-name", run_name],
        },
        {
            "stage": "S1b",
            "gate": gates / "S1b_bridge_feasibility.json",
            "cmd": [python, tool("s1b_bridge_feasibility.py"), "--run-name", run_name],
        },
        {
            "stage": "S2",
            "gate": gates / "S2_extract.json",
            "cmd": [python, tool("s2_extract.py"), "--run-name", run_name],
        },
        {
            "stage": "S2b",
            "gate": gates / "S2b_intrinsics.json",
            "cmd": [python, tool("s2b_intrinsics_bakeoff.py"), "--run-name", run_name],
        },
        {
            "stage": "S3",
            "gate": gates / "S3_pairs.json",
            "cmd": [python, tool("s3_pairs.py"), "--run-name", run_name],
        },
        {
            "stage": "S4",
            "gate": gates / "S4_doppelgangers.json",
            "needs": {
                "--twoview": args.twoview,
                "--image-root": images,
                "--forced-pairs": forced_txt,
                "--forced-manifest": forced_json,
            },
            "cmd": [
                python,
                tool("audit_dg_graph.py"),
                "--twoview",
                args.twoview,
                "--image-root",
                images,
                "--forced-pairs",
                forced_txt,
                "--forced-manifest",
                forced_json,
                "--out",
                gates / "S4_doppelgangers.json",
            ],
        },
        {
            "stage": "S5",
            "gate": gates / "S5_fixed_intrinsics.json",
            "needs": {
                "--database": args.database,
                "--lfoe-bin": args.lfoe_bin,
                "--intrinsics-seed": args.intrinsics_seed,
                "--frame-manifest": frame_manifest,
                "--image-root": images,
            },
            "commands": [
                [
                    args.lfoe_bin,
                    "mapper",
                    "--database_path",
                    args.database,
                    "--image_path",
                    images,
                    "--output_path",
                    lfoe_model,
                    "--BundleAdjustment.optimize_intrinsics",
                    "0",
                    "--BundleAdjustment.optimize_principal_point",
                    "0",
                ],
                [
                    python,
                    tool("finalize_edm_model.py"),
                    "--input-model",
                    lfoe_model / "0",
                    "--output-model",
                    model,
                    "--frame-manifest",
                    frame_manifest,
                    "--intrinsics-seed",
                    args.intrinsics_seed,
                    "--metrics-out",
                    run_dir / "s5_metrics.json",
                ],
            ],
        },
        {
            "stage": "S5.7",
            "gate": gates / "S5_7_independent_sim3.json",
            "needs": {
                "--database": args.database,
                "--lfoe-bin": args.lfoe_bin,
                "--twoview": args.twoview,
                "--image-root": images,
                "--forced-pairs": forced_txt,
                "--forced-manifest": forced_json,
                "--s4-gate": gates / "S4_doppelgangers.json",
            },
            "cmd": [
                python,
                tool("audit_independent_sim3.py"),
                "--database",
                args.database,
                "--image-root",
                images,
                "--forced-pairs",
                forced_txt,
                "--forced-manifest",
                forced_json,
                "--twoview",
                args.twoview,
                "--s4-gate",
                gates / "S4_doppelgangers.json",
                "--work-dir",
                run_dir / "s5_7",
                "--lfoe-bin",
                args.lfoe_bin,
                "--out",
                gates / "S5_7_independent_sim3.json",
            ],
        },
        {
            "stage": "S6",
            "gate": gates / "S5_7_S6_geometry.json",
            "needs": {
                "--model": model,
                "--image-root": images,
                "--frame-manifest": frame_manifest,
                "--corpus-manifest": corpus_manifest,
                "--forced-pairs": forced_txt,
                "--forced-manifest": forced_json,
                "--twoview": args.twoview,
                "--s4-gate": gates / "S4_doppelgangers.json",
                "--s5-metrics": args.s5_metrics or (run_dir / "s5_metrics.json"),
                "--s5-7-gate": gates / "S5_7_independent_sim3.json",
            },
            "cmd": [
                python,
                tool("audit_map_geometry.py"),
                "--model",
                model,
                "--image-root",
                images,
                "--frame-manifest",
                frame_manifest,
                "--corpus-manifest",
                corpus_manifest,
                "--forced-pairs",
                forced_txt,
                "--forced-manifest",
                forced_json,
                "--twoview",
                args.twoview,
                "--s4-gate",
                gates / "S4_doppelgangers.json",
                "--s5-metrics",
                args.s5_metrics or (run_dir / "s5_metrics.json"),
                "--s5-7-gate",
                gates / "S5_7_independent_sim3.json",
                "--out",
                gates / "S5_7_S6_geometry.json",
            ],
        },
        {
            "stage": "S7",
            "gate": gates / "S7_tracking_bundle.json",
            "needs": {"--model": model, "--image-root": images, "--frame-manifest": frame_manifest},
            "cmd": [
                python,
                tool("validate_tracking_bundle.py"),
                "--bundle",
                tracking,
                "--model",
                model,
                "--image-root",
                images,
                "--frame-manifest",
                frame_manifest,
                "--out",
                gates / "S7_tracking_bundle.json",
            ],
        },
        {
            "stage": "S8",
            "gate": gates / "S8_edm_bundle.json",
            "needs": {
                "--bundle": edm,
                "--tracking-bundle": tracking,
                "--baseline-bundle": args.baseline_bundle,
                "--model": model,
            },
            "cmd": [
                python,
                tool("validate_edm_bundle.py"),
                "--bundle",
                edm,
                "--tracking-bundle",
                tracking,
                "--baseline-bundle",
                args.baseline_bundle,
                "--model",
                model,
                "--out",
                gates / "S8_edm_bundle.json",
            ],
        },
        {
            "stage": "S9",
            "gate": gates / "S9_heldout_localization.json",
            "needs": {
                "--result": args.result,
                "--forced-manifest": forced_json,
                "--corpus-manifest": corpus_manifest,
                "--edm-bundle": edm,
                "--tracking-bundle": tracking,
                **({} if args.package_bundle is None else {"--package-bundle": args.package_bundle}),
                **({} if args.package_config is None else {"--package-config": args.package_config}),
            },
            "cmd": [
                python,
                tool("validate_heldout_localization.py"),
                *[item for path in (args.result or []) for item in ("--result", path)],
                "--forced-manifest",
                forced_json,
                "--corpus-manifest",
                corpus_manifest,
                "--edm-bundle",
                edm,
                "--tracking-bundle",
                tracking,
                *([] if args.package_bundle is None else ["--package-bundle", args.package_bundle]),
                *([] if args.package_config is None else ["--package-config", args.package_config]),
                "--out",
                gates / "S9_heldout_localization.json",
            ],
        },
    ]


def resolve_stage_commands(spec: dict) -> list[list[str]]:
    for label, value in spec.get("needs", {}).items():
        if isinstance(value, list):
            if not value:
                raise SystemExit(f"{spec['stage']}: missing required argument {label}")
            for item in value:
                require_path(Path(item), f"{spec['stage']} {label}")
            continue
        require_path(None if value is None else Path(value), f"{spec['stage']} {label}")
    commands = spec.get("commands")
    if commands is None:
        commands = [spec["cmd"]]
    return [[str(part) for part in command] for command in commands]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--twoview", type=Path)
    parser.add_argument("--intrinsics-seed", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--lfoe-bin", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--s5-metrics", type=Path)
    parser.add_argument("--tracking-bundle", type=Path)
    parser.add_argument("--edm-bundle", type=Path)
    parser.add_argument("--baseline-bundle", type=Path)
    parser.add_argument("--result", type=Path, action="append")
    parser.add_argument("--package-bundle", type=Path)
    parser.add_argument("--package-config", type=Path)
    parser.add_argument("--start-from", default="S0")
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        raise SystemExit(f"missing --run-dir: {args.run_dir}")

    started = False
    for spec in plan_stages(args):
        if not started:
            if spec["stage"] != args.start_from:
                continue
            started = True
        for cmd in resolve_stage_commands(spec):
            print("[run_s0_s9] " + " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        ok, detail = gate_passed(spec["gate"])
        if not ok:
            raise SystemExit(f"stop at {spec['stage']}: {detail}")
    if not started:
        raise SystemExit(f"unknown --start-from {args.start_from}")


if __name__ == "__main__":
    main()
