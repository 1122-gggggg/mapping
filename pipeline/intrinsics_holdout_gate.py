#!/usr/bin/env python3
"""Gate intrinsics bake-off candidates with holdout localization results."""
from __future__ import annotations

import argparse
import json
import shlex
import time
from pathlib import Path


BUILD_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = BUILD_ROOT.parent
LOC_PIPELINE = SYSTEM_ROOT / "定位" / "pipeline" / "localize_pipeline.py"
DEFAULT_BASE = SYSTEM_ROOT / "定位" / "bundles" / "base_reloc_map_xfeat_tri.pt"
DEFAULT_CACHE = SYSTEM_ROOT / "定位" / "bundles" / "base_megaloc_cache_v3.npz"
DEFAULT_HOLDOUT = SYSTEM_ROOT / "更新地圖" / "inputs" / "補拍影片" / "test"


def parse_name_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name.strip(), Path(path)


def discover_eval_json(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "holdout_localization.json",
        run_dir / "validation_compare.json",
        run_dir / "gates" / "holdout_localization.json",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    matches = sorted(run_dir.glob("*holdout*localization*.json"))
    return matches[0] if matches else None


def evaluate_eval_stream_json(path: Path, min_success: float, max_ok_to_fail: int,
                              max_final_fail_run: int) -> tuple[bool, dict, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    reasons = []
    per_set = []
    for row in rows:
        name = row.get("set", "")
        final = float(row.get("final_success", 0.0))
        base = float(row.get("base_success", 0.0))
        ok_to_fail = int(row.get("ok_to_fail", 0))
        final_fail = int(row.get("final_max_fail_run", 0))
        base_fail = int(row.get("base_max_fail_run", max_final_fail_run))
        regression_ok = ok_to_fail <= max_ok_to_fail and final_fail <= max_final_fail_run
        absolute_ok = final >= min_success
        baseline_improved = (
            base < min_success
            and final >= base
            and final_fail <= base_fail
            and regression_ok
        )
        ok = regression_ok and (absolute_ok or baseline_improved)
        if not ok:
            reasons.append(f"{name}: final_success={final:.3f}, ok_to_fail={ok_to_fail}, final_fail_run={final_fail}")
        per_set.append({
            "set": name,
            "base_success": base,
            "final_success": final,
            "ok_to_fail": ok_to_fail,
            "final_max_fail_run": final_fail,
            "result": "PASS" if absolute_ok else "PASS_BASELINE_IMPROVED" if baseline_improved else "FAIL",
        })
    metrics = {
        "sets": per_set,
        "min_final_success": min((r["final_success"] for r in per_set), default=0.0),
        "max_ok_to_fail": max((r["ok_to_fail"] for r in per_set), default=0),
        "max_final_fail_run": max((r["final_max_fail_run"] for r in per_set), default=0),
    }
    if not rows:
        reasons.append("eval JSON has no rows")
    return not reasons, metrics, reasons


def eval_final_path(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("final", "final_bundle"):
        if data.get(key):
            return str(data[key])
    args = data.get("args")
    if isinstance(args, dict) and args.get("final"):
        return str(args["final"])
    return ""


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def holdout_command(args: argparse.Namespace, run_dir: Path, bundle: Path, out_json: Path) -> list[str]:
    return [
        args.python,
        str(LOC_PIPELINE),
        "--mode", "compare",
        "--base", str(args.base_bundle),
        "--final", str(bundle),
        "--base-megaloc-cache", str(args.base_megaloc_cache),
        "--test-dir", str(args.holdout_dir),
        "--stride", str(args.stride),
        "--resize", args.resize,
        "--min-success", str(args.min_success),
        "--max-ok-to-fail", str(args.max_ok_to_fail),
        "--max-final-fail-run", str(args.max_final_fail_run),
        "--out-json", str(out_json),
    ]


def build_gate(args: argparse.Namespace) -> dict:
    candidates = dict(parse_name_path(item) for item in args.candidate)
    eval_jsons = dict(parse_name_path(item) for item in args.eval_json)
    bundles = dict(parse_name_path(item) for item in args.candidate_bundle)
    rows = []
    for name, run_dir in candidates.items():
        eval_path = eval_jsons.get(name) or discover_eval_json(run_dir)
        bundle = bundles.get(name) or run_dir / "deploy" / "reloc_map_xfeat_tri.pt"
        default_out = run_dir / "holdout_localization.json"
        command = holdout_command(args, run_dir, bundle, default_out)
        row = {
            "name": name,
            "run_dir": str(run_dir),
            "bundle": str(bundle),
            "bundle_exists": bundle.exists(),
            "eval_json": str(eval_path) if eval_path else "",
            "holdout_command": command,
            "holdout_shell": shell_join(command),
            "ok": False,
            "status": "needs_holdout_eval",
            "metrics": {},
            "reasons": [],
        }
        if eval_path:
            ok, metrics, reasons = evaluate_eval_stream_json(
                eval_path, args.min_success, args.max_ok_to_fail, args.max_final_fail_run
            )
            if not bundle.exists():
                ok = False
                reasons.append("candidate deployment bundle missing")
            final_path = eval_final_path(eval_path)
            if not final_path:
                ok = False
                reasons.append("eval JSON missing final bundle provenance")
            elif Path(final_path).resolve(strict=False) != bundle.resolve(strict=False):
                ok = False
                reasons.append(f"eval final bundle mismatch: {final_path}")
            row.update({
                "ok": ok,
                "status": "pass" if ok else "fail",
                "metrics": metrics,
                "reasons": reasons,
            })
        else:
            row["reasons"].append("holdout localization eval JSON missing")
            if not bundle.exists():
                row["reasons"].append("candidate deployment bundle missing")
        rows.append(row)

    main = next((row for row in rows if row["name"] == args.main_candidate), None)
    overall_ok = bool(main and main["ok"])
    return {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "main_candidate": args.main_candidate,
        "overall_ok": overall_ok,
        "gate": {
            "min_success": args.min_success,
            "max_ok_to_fail": args.max_ok_to_fail,
            "max_final_fail_run": args.max_final_fail_run,
            "holdout_dir": str(args.holdout_dir),
        },
        "candidates": rows,
        "decision": (
            f"{args.main_candidate} can be promoted only after holdout localization passes"
            if not overall_ok else f"{args.main_candidate} passed holdout localization gate"
        ),
    }


def write_markdown(report: dict, out_path: Path) -> None:
    lines = ["# Intrinsics Holdout Gate", ""]
    gate = report["gate"]
    lines.append(f"- Main candidate: `{report['main_candidate']}`")
    lines.append(f"- Overall: `{'PASS' if report['overall_ok'] else 'FAIL'}`")
    lines.append(
        f"- Gate: final_success >= {gate['min_success']:.0%} or baseline-improved; "
        f"ok_to_fail <= {gate['max_ok_to_fail']}; final_fail_run <= {gate['max_final_fail_run']}"
    )
    lines.append(f"- Holdout dir: `{gate['holdout_dir']}`")
    lines.append("")
    lines.append("| Candidate | Status | Bundle | Eval JSON | Min success | Max fail run | Reasons |")
    lines.append("|---|---|---|---|---:|---:|---|")
    for row in report["candidates"]:
        metrics = row.get("metrics") or {}
        min_success = metrics.get("min_final_success")
        max_fail = metrics.get("max_final_fail_run")
        lines.append(
            f"| {row['name']} | {row['status']} | `{row['bundle']}` | `{row['eval_json']}` | "
            f"{'-' if min_success is None else f'{float(min_success):.1%}'} | "
            f"{'-' if max_fail is None else int(max_fail)} | {'; '.join(row.get('reasons') or [])} |"
        )
    lines.append("")
    lines.append("## Commands To Produce Missing Holdout Eval")
    lines.append("")
    for row in report["candidates"]:
        if row["eval_json"]:
            continue
        lines.append(f"### {row['name']}")
        lines.append("")
        lines.append("```bash")
        lines.append(row["holdout_shell"])
        lines.append("```")
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="/usr/bin/python3.12")
    parser.add_argument("--candidate", action="append", required=True, help="NAME=RUN_DIR")
    parser.add_argument("--main-candidate", default="no_undistort_official69")
    parser.add_argument("--candidate-bundle", action="append", default=[], help="NAME=BUNDLE_PATH")
    parser.add_argument("--eval-json", action="append", default=[], help="NAME=EVAL_STREAM_JSON")
    parser.add_argument("--base-bundle", default=str(DEFAULT_BASE))
    parser.add_argument("--base-megaloc-cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--holdout-dir", default=str(DEFAULT_HOLDOUT))
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--resize", default="1280x720")
    parser.add_argument("--min-success", type=float, default=0.90)
    parser.add_argument("--max-ok-to-fail", type=int, default=0)
    parser.add_argument("--max-final-fail-run", type=int, default=30)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()
    args.base_bundle = Path(args.base_bundle)
    args.base_megaloc_cache = Path(args.base_megaloc_cache)
    args.holdout_dir = Path(args.holdout_dir)

    report = build_gate(args)
    out_json = Path(args.out_json) if args.out_json else BUILD_ROOT / "outputs" / "intrinsics_holdout_gate.json"
    out_md = Path(args.out_md) if args.out_md else out_json.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, out_md)
    print(f"[intrinsics_holdout_gate] wrote {out_json}")
    print(f"[intrinsics_holdout_gate] wrote {out_md}")
    if not report["overall_ok"]:
        raise SystemExit("intrinsics holdout gate failed")


if __name__ == "__main__":
    main()
