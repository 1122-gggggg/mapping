#!/usr/bin/env python3
"""Map update pipeline entrypoint.

Input: base localization map plus new field videos or prepared frame folders.
Output: updated deployment bundle and colored PLY, without overwriting the base.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import os.path
from pathlib import Path


UPDATE_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = UPDATE_ROOT.parent
LOC_ROOT = SYSTEM_ROOT / "定位"
DEPLOY = LOC_ROOT / "deploy_code" / "sfm_glomap_deploy"
SCRIPTS = LOC_ROOT / "source" / "sfm_glomap" / "scripts"
CORE_DIR = UPDATE_ROOT / "pipeline" / "update_pipeline_core"
PREPARE = CORE_DIR / "prepare_update_frames.py"
UPDATE_TOOL = CORE_DIR / "map_update_tool.py"
SPARSIFY = CORE_DIR / "sparsify_reloc_bundle.py"
EVAL_CORE = LOC_ROOT / "validation" / "eval_stream_core.py"

DEFAULT_BASE = LOC_ROOT / "bundles" / "base_reloc_map_xfeat_tri.pt"
DEFAULT_MODEL = LOC_ROOT / "maps" / "base_glomap_fused_0"
DEFAULT_IMAGES = LOC_ROOT / "maps" / "base_images_fused"
DEFAULT_BASE_CACHE = LOC_ROOT / "bundles" / "base_megaloc_cache_v3.npz"
DEFAULT_FPS = 3.0
DEFAULT_MIN_FLOW_PX = 0.8
DEFAULT_MAX_FLOW_PX = 180.0
DEFAULT_KEEP_ROTATION_EVERY = 8


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def py_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(DEPLOY) + os.pathsep + str(SCRIPTS) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run(cmd: list[str]) -> None:
    print("[update_pipeline] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=py_env())


def rel_symlink(target: Path, link: Path) -> None:
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(os.path.relpath(target.resolve(), link.parent.resolve()))


def count_jpgs(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.jpg")) if root.exists() else 0


def write_gate(path: Path, stage: str, ok: bool, metrics: dict, reasons: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "stage": stage,
        "ok": ok,
        "metrics": metrics,
        "reasons": reasons,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[update_pipeline] gate {stage}: {'PASS' if ok else 'FAIL'} {metrics}", flush=True)
    if not ok:
        raise SystemExit(f"stage gate failed after {stage}: {'; '.join(reasons)}")


def inspect_flight_bundle(bundle: Path) -> tuple[dict, list[str]]:
    """Validate fields required by production tracking / real-flight deployment."""
    detail: dict = {"path": str(bundle), "exists": bundle.exists()}
    reasons: list[str] = []
    if not bundle.exists():
        return detail, ["bundle missing"]
    try:
        import numpy as np
        import torch
        b = torch.load(bundle, map_location="cpu", weights_only=False)
    except Exception as exc:
        detail["load_error"] = repr(exc)
        return detail, [f"cannot load bundle: {exc}"]

    names = list(b.get("ref_names", []))
    n = len(names)
    refs = b.get("refs", {})
    ref_global = np.asarray(b.get("ref_global", []), dtype=np.float32)
    centers = np.asarray(b.get("ref_centers", []), dtype=np.float32)
    yaws = np.asarray(b.get("ref_yaws", []), dtype=np.float32)
    ref_stability = np.asarray(b.get("ref_stability", []), dtype=np.float32)
    covis = b.get("covis", {})
    meta = dict(b.get("meta", {}))

    detail.update({
        "nrefs": n,
        "refs": len(refs) if hasattr(refs, "__len__") else 0,
        "ref_global_shape": list(ref_global.shape),
        "ref_centers_shape": list(centers.shape),
        "ref_yaws_shape": list(yaws.shape),
        "ref_stability_shape": list(ref_stability.shape),
        "covis_keys": len(covis) if isinstance(covis, dict) else 0,
        "bundle_vpr": meta.get("bundle_vpr") or meta.get("vpr"),
        "tracking_metadata": bool(meta.get("tracking_metadata")),
    })
    if n <= 0:
        reasons.append("ref_names empty")
    if not isinstance(refs, dict) or len(refs) != n:
        reasons.append(f"refs count {len(refs) if hasattr(refs, '__len__') else 0} != ref_names {n}")
    if ref_global.ndim != 2 or ref_global.shape[0] != n:
        reasons.append(f"ref_global shape {tuple(ref_global.shape)} does not match nrefs {n}")
    elif ref_global.shape[1] != 8448 or not np.isfinite(ref_global).all():
        reasons.append(
            f"ref_global must be finite MegaLoc descriptors with shape ({n},8448), got {tuple(ref_global.shape)}"
        )
    if centers.shape != (n, 3) or not np.isfinite(centers).all():
        reasons.append(f"ref_centers must be finite shape ({n},3), got {tuple(centers.shape)}")
    if yaws.shape != (n,) or not np.isfinite(yaws).all():
        reasons.append(f"ref_yaws must be finite shape ({n},), got {tuple(yaws.shape)}")
    if ref_stability.size and (ref_stability.shape != (n,) or not np.isfinite(ref_stability).all()):
        reasons.append(f"ref_stability must be finite shape ({n},), got {tuple(ref_stability.shape)}")
    if not isinstance(covis, dict):
        reasons.append("covis must be a dict")
    else:
        missing = [name for name in names if name not in covis]
        if missing:
            reasons.append(f"covis missing {len(missing)} keyframes")
        bad_refs = 0
        for vals in covis.values():
            if not isinstance(vals, (list, tuple)):
                bad_refs += 1
                continue
            for item in vals:
                try:
                    jj = int(item)
                except Exception:
                    bad_refs += 1
                    continue
                if jj < 0 or jj >= n:
                    bad_refs += 1
        if bad_refs:
            reasons.append(f"covis contains {bad_refs} invalid entries")
    if not str(meta.get("bundle_vpr") or meta.get("vpr") or "").lower().startswith("megaloc"):
        reasons.append("updated deployment bundle must use MegaLoc global descriptors")
    if not meta.get("tracking_metadata"):
        reasons.append("tracking_metadata meta flag is not set")
    return detail, reasons


def validate_prepared_frames(work_dir: Path, frames_root: Path, connector_root: Path | None,
                             min_geometry_frames: int, min_connector_frames: int) -> None:
    geom_count = count_jpgs(frames_root)
    conn_count = count_jpgs(connector_root) if connector_root else 0
    reasons = []
    if geom_count < min_geometry_frames:
        reasons.append(f"geometry frames {geom_count} < {min_geometry_frames}")
    if connector_root is not None and min_connector_frames and conn_count < min_connector_frames:
        reasons.append(f"connector frames {conn_count} < {min_connector_frames}")
    write_gate(
        work_dir / "gates" / "prepare_frames.json",
        "prepare_frames",
        not reasons,
        {
            "frames_root": str(frames_root),
            "geometry_frames": geom_count,
            "connector_root": str(connector_root) if connector_root else None,
            "connector_frames": conn_count,
        },
        reasons,
    )


def validate_update_outputs(out_dir: Path, require_ply: bool) -> None:
    bundle = out_dir / "reloc_map_updated.pt"
    report = out_dir / "update_report.md"
    ply = out_dir / "latest_map_realrgb.ply"
    flight_detail, flight_reasons = inspect_flight_bundle(bundle)
    summary_detail, summary_reasons = inspect_update_summary(out_dir)
    metrics = {
        "bundle": str(bundle),
        "bundle_bytes": bundle.stat().st_size if bundle.exists() else 0,
        "report": str(report),
        "report_exists": report.exists(),
        "ply": str(ply),
        "ply_bytes": ply.stat().st_size if ply.exists() else 0,
        "flight_bundle": flight_detail,
        "update_summary": summary_detail,
    }
    reasons = []
    if metrics["bundle_bytes"] <= 1000:
        reasons.append("updated bundle missing or too small")
    if not report.exists():
        reasons.append("update_report.md missing")
    if require_ply and metrics["ply_bytes"] <= 1000:
        reasons.append("latest_map_realrgb.ply missing or too small")
    reasons.extend(f"flight bundle metadata: {r}" for r in flight_reasons)
    reasons.extend(f"update summary metadata: {r}" for r in summary_reasons)
    write_gate(out_dir / "gates" / "update_outputs.json", "update_outputs", not reasons, metrics, reasons)


def inspect_update_summary(out_dir: Path) -> tuple[dict, list[str]]:
    summary = out_dir / "update_summary.json"
    detail: dict = {"path": str(summary), "summary_exists": summary.exists()}
    reasons: list[str] = []
    if not summary.exists():
        return detail, reasons
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
    except Exception as exc:
        detail["load_error"] = repr(exc)
        return detail, [f"cannot load update_summary.json: {exc}"]
    rows = data.get("rows", [])
    detail["rows"] = len(rows) if isinstance(rows, list) else 0
    if not isinstance(rows, list):
        return detail, ["update_summary rows must be a list"]
    unfinished = []
    for row in rows:
        route = str(row.get("route", ""))
        status = str(row.get("status", ""))
        if route in {"changed", "changed-region", "tile_replace"} or status == "needs_tile_replace":
            unfinished.append(str(row.get("seq", "<unknown>")))
    if unfinished:
        reasons.append(
            "changed/tile_replace route needs tile replacement but tile replacement is not implemented: "
            + ", ".join(unfinished)
        )
    return detail, reasons


def validate_slim_bundle(out_dir: Path, bundle: Path) -> None:
    report = bundle.with_suffix(bundle.suffix + ".sparsify_report.json")
    flight_detail, flight_reasons = inspect_flight_bundle(bundle)
    metrics = {
        "bundle": str(bundle),
        "bundle_bytes": bundle.stat().st_size if bundle.exists() else 0,
        "sparsify_report": str(report),
        "sparsify_report_exists": report.exists(),
        "flight_bundle": flight_detail,
    }
    reasons = []
    if metrics["bundle_bytes"] <= 1000:
        reasons.append("slim bundle missing or too small")
    if not report.exists():
        reasons.append("sparsify report missing")
    reasons.extend(f"slim flight bundle metadata: {r}" for r in flight_reasons)
    write_gate(out_dir / "gates" / "slim_bundle.json", "slim_bundle", not reasons, metrics, reasons)


def build_sparsify_command(args: argparse.Namespace, input_bundle: Path, slim_bundle: Path,
                           observation_stats: Path) -> list[str]:
    cmd = [
        args.python, str(SPARSIFY),
        "--bundle", str(input_bundle),
        "--out", str(slim_bundle),
        "--target-fraction", str(args.sparsify_target_fraction),
        "--min-per-prefix", str(args.sparsify_min_per_prefix),
    ]
    if observation_stats.exists():
        cmd += ["--observation-stats", str(observation_stats)]
    for prefix in args.sparsify_keep_prefix:
        cmd += ["--keep-prefix", prefix]
    return cmd


def build_validation_command(args: argparse.Namespace, final_bundle: Path, out_json: Path) -> list[str]:
    return [
        args.python, str(EVAL_CORE),
        "--base", args.base_bundle,
        "--final", str(final_bundle),
        "--base-megaloc-cache", str(args.base_megaloc_cache),
        "--test-dir", args.validate_dir,
        "--stride", str(args.validate_stride),
        "--resize", args.validate_resize,
        "--out-json", str(out_json),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified incremental map update pipeline.")
    parser.add_argument("--python", default="/usr/bin/python3.12")
    parser.add_argument("--video", action="append", default=[], help="SEQ=/path/video.mp4")
    parser.add_argument("--frames-root", default="", help="Prepared geometry frames root.")
    parser.add_argument("--connector-root", default="", help="Prepared connector frames root.")
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--base-bundle", default=str(DEFAULT_BASE))
    parser.add_argument("--base-model", default=str(DEFAULT_MODEL))
    parser.add_argument("--base-images", default=str(DEFAULT_IMAGES))
    parser.add_argument("--base-megaloc-cache", default=str(DEFAULT_BASE_CACHE))
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--motion-filter", choices=["none", "flow", "parallax"], default="parallax")
    parser.add_argument("--min-flow-px", type=float, default=DEFAULT_MIN_FLOW_PX)
    parser.add_argument("--max-flow-px", type=float, default=DEFAULT_MAX_FLOW_PX)
    parser.add_argument("--keep-rotation-every", type=int, default=DEFAULT_KEEP_ROTATION_EVERY)
    parser.add_argument("--no-split-classes", action="store_true")
    parser.add_argument("--overwrite-frames", action="store_true")
    parser.add_argument("--validate-dir", default="")
    parser.add_argument("--validate-stride", type=int, default=10)
    parser.add_argument("--validate-resize", default="1280x720")
    parser.add_argument("--min-success", type=float, default=0.90)
    parser.add_argument("--max-ok-to-fail", type=int, default=0)
    parser.add_argument("--max-final-fail-run", type=int, default=30)
    parser.add_argument("--disable-stage-gates", action="store_true")
    parser.add_argument("--gate-min-geometry-frames", type=int, default=4)
    parser.add_argument("--gate-min-connector-frames", type=int, default=0)
    parser.add_argument("--sparsify", action="store_true",
                        help="Create and gate a separate slim deployment bundle after update.")
    parser.add_argument("--sparsify-target-fraction", type=float, default=0.85)
    parser.add_argument("--sparsify-min-per-prefix", type=int, default=40)
    parser.add_argument("--sparsify-keep-prefix", action="append", default=[])
    parser.add_argument("--slim-out", default="")
    args, passthrough = parser.parse_known_args()

    if not args.video and not args.frames_root:
        raise SystemExit("provide --video or --frames-root")

    stamp = timestamp()
    work_dir = Path(args.work_dir) if args.work_dir else UPDATE_ROOT / "workspaces" / f"update_{stamp}"
    out_dir = Path(args.out_dir) if args.out_dir else UPDATE_ROOT / "outputs" / f"updated_map_{stamp}"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_root = Path(args.frames_root) if args.frames_root else None
    connector_root = Path(args.connector_root) if args.connector_root else None

    if args.video and frames_root is None:
        prepared_root = work_dir / "prepared_frames"
        cmd = [
            args.python, str(PREPARE),
            "--out-root", str(prepared_root),
            "--fps", str(args.fps),
            "--motion-filter", args.motion_filter,
            "--min-flow-px", str(args.min_flow_px),
            "--max-flow-px", str(args.max_flow_px),
            "--keep-rotation-every", str(args.keep_rotation_every),
            *sum((["--video", v] for v in args.video), []),
        ]
        if not args.no_split_classes:
            cmd.append("--split-classes")
        if args.overwrite_frames:
            cmd.append("--overwrite")
        run(cmd)
        frames_root = prepared_root / "geometry" if not args.no_split_classes else prepared_root
        maybe_connector = prepared_root / "connector"
        if maybe_connector.exists() and not args.no_split_classes:
            connector_root = maybe_connector
        if not args.disable_stage_gates:
            validate_prepared_frames(
                work_dir, frames_root, connector_root,
                args.gate_min_geometry_frames, args.gate_min_connector_frames,
            )

    if not args.disable_stage_gates:
        validate_prepared_frames(
            work_dir, frames_root, connector_root,
            args.gate_min_geometry_frames, args.gate_min_connector_frames,
        )

    cmd = [
        args.python, str(UPDATE_TOOL),
        "--base-bundle", args.base_bundle,
        "--base-model", args.base_model,
        "--base-images", args.base_images,
        "--base-megaloc-cache", args.base_megaloc_cache,
        "--new-data", str(frames_root),
        "--out-dir", str(out_dir),
    ]
    if connector_root and connector_root.exists():
        cmd += ["--connector-data", str(connector_root)]
    cmd += passthrough
    run(cmd)
    if not args.disable_stage_gates:
        validate_update_outputs(out_dir, require_ply="--no-ply" not in passthrough)

    slim_bundle = None
    if args.sparsify:
        input_bundle = out_dir / "reloc_map_updated.pt"
        slim_bundle = Path(args.slim_out) if args.slim_out else out_dir / "reloc_map_updated_slim.pt"
        run(build_sparsify_command(args, input_bundle, slim_bundle, out_dir / "observation_stats.json"))
        if not args.disable_stage_gates:
            validate_slim_bundle(out_dir, slim_bundle)

    if args.validate_dir:
        val_out = out_dir / "validation_compare.json"
        run(build_validation_command(args, out_dir / "reloc_map_updated.pt", val_out))
        validation_ok = write_quality_report(val_out, args.min_success, args.max_ok_to_fail, args.max_final_fail_run)
        if not args.disable_stage_gates and not validation_ok:
            raise SystemExit("validation gate failed; updated map is not deployable")
        print(f"[update_pipeline] validation saved {val_out}", flush=True)

        if slim_bundle is not None:
            slim_val_out = out_dir / "slim_validation_compare.json"
            run(build_validation_command(args, slim_bundle, slim_val_out))
            slim_ok = write_quality_report(
                slim_val_out, args.min_success, args.max_ok_to_fail, args.max_final_fail_run
            )
            if not args.disable_stage_gates and not slim_ok:
                raise SystemExit("slim validation gate failed; slim bundle is not deployable")
            print(f"[update_pipeline] slim validation saved {slim_val_out}", flush=True)
    elif slim_bundle is not None and not args.disable_stage_gates:
        write_gate(
            out_dir / "gates" / "slim_localization.json",
            "slim_localization",
            False,
            {"validate_dir": args.validate_dir, "slim_bundle": str(slim_bundle)},
            ["--sparsify requires --validate-dir so the slim bundle passes the localization gate"],
        )

    latest = UPDATE_ROOT / "outputs" / "latest_update"
    rel_symlink(out_dir, latest)

    print(f"[update_pipeline] updated map output: {out_dir}", flush=True)


def write_quality_report(result_json: Path, min_success: float, max_ok_to_fail: int,
                         max_final_fail_run: int) -> bool:
    data = json.loads(result_json.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    failures = []
    lines = ["# Update Localization Quality Report", ""]
    lines.append(f"- Source: `{result_json}`")
    lines.append(
        f"- Gate: final_success >= {min_success:.0%}, or baseline-improved when base is below target; "
        f"ok_to_fail <= {max_ok_to_fail}, final_max_fail_run <= {max_final_fail_run}"
    )
    lines.append("")
    lines.append("| Set | Frames | Base success | Updated success | Gain | ok->fail | maxfail updated | Result |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    if not rows:
        failures.append("<empty>")
        lines.append("| <empty> | 0 | 0.0% | 0.0% | +0.0pp | 0 | 0 | FAIL_EMPTY_ROWS |")
    for row in rows:
        name = row.get("set", "")
        n = int(row.get("n", 0))
        base = float(row.get("base_success", 0.0))
        final = float(row.get("final_success", 0.0))
        gain = float(row.get("gain_pp", 0.0))
        reg = int(row.get("ok_to_fail", 0))
        maxfail = int(row.get("final_max_fail_run", 0))
        base_maxfail = int(row.get("base_max_fail_run", max_final_fail_run))
        regression_ok = reg <= max_ok_to_fail and maxfail <= max_final_fail_run
        absolute_ok = final >= min_success
        baseline_improved = (
            base < min_success
            and final >= base
            and maxfail <= base_maxfail
            and regression_ok
        )
        ok = regression_ok and (absolute_ok or baseline_improved)
        if ok:
            result = "PASS" if absolute_ok else "PASS_BASELINE_IMPROVED"
        elif not regression_ok:
            result = "FAIL_REGRESSION"
        else:
            result = "FAIL"
        if not ok:
            failures.append(name)
        lines.append(
            f"| {name} | {n} | {base:.1%} | {final:.1%} | {gain:+.1f}pp | "
            f"{reg} | {maxfail} | {result} |"
        )
    lines.append("")
    lines.append(f"Overall: {'PASS' if not failures else 'FAIL'}")
    if failures:
        lines.append(f"Failed sets: {', '.join(failures)}")
    report = result_json.with_suffix(".quality_report.md")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[update_pipeline] quality report {report}", flush=True)
    return not failures


if __name__ == "__main__":
    main()
