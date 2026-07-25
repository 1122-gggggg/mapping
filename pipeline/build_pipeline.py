#!/usr/bin/env python3
"""Build pipeline entrypoint.

Input: field videos or an existing image root.
Output: a localizable GLOMAP map plus XFeat/MegaLoc deployment bundle.

This wrapper keeps the system layout stable and delegates the heavy work to the
preserved one-file builder in sfm_reshot25.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import os.path
from pathlib import Path


BUILD_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = BUILD_ROOT.parent
CORE = BUILD_ROOT / "pipeline" / "build_localizable_map_core.py"
FINAL_GATE = BUILD_ROOT / "pipeline" / "stage_gate_contract.py"
VERIFY_PACKAGE = SYSTEM_ROOT / "tools" / "verify_package.py"
SYSTEM_VERIFY = SYSTEM_ROOT / "tools" / "system_verify.py"
LOCALIZE_PIPELINE = SYSTEM_ROOT / "定位" / "pipeline" / "localize_pipeline.py"
TOOLS = BUILD_ROOT / "external_tools"
DEFAULT_MVROMA = TOOLS / "MV-RoMa"
DEFAULT_UFM = TOOLS / "UFM"
DEFAULT_DOPPELGANGERS = TOOLS / "doppelgangers-plusplus"
DEFAULT_DOPPELGANGERS_CHECKPOINT = DEFAULT_DOPPELGANGERS / "checkpoints" / "checkpoint-dg+visym.pth"
DEFAULT_LFOE = TOOLS / "LFOE-GlobalSfM" / "build" / "glomap_filter"
DEFAULT_MPSFM = TOOLS / "mpsfm"


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def run(cmd: list[str]) -> None:
    print("[build_pipeline] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def rel_symlink(target: Path, link: Path) -> None:
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(os.path.relpath(target.resolve(), link.parent.resolve()))


def final_gate_required(stages: str) -> bool:
    requested = {part.strip() for part in stages.split(",") if part.strip()}
    return not requested or "all" in requested or "report" in requested


def field_handoff_enabled(args: argparse.Namespace) -> bool:
    return getattr(args, "handoff_profile", "field") == "field"


def require_final_evidence(args: argparse.Namespace, attr: str) -> bool:
    return field_handoff_enabled(args) or bool(getattr(args, attr))


def latest_link_for_handoff_profile(args: argparse.Namespace) -> Path:
    if field_handoff_enabled(args):
        return BUILD_ROOT / "outputs" / "latest_build"
    return BUILD_ROOT / "outputs" / "latest_candidate_build"


def auto_package_verify_enabled(args: argparse.Namespace) -> bool:
    return (
        field_handoff_enabled(args)
        and not getattr(args, "final_gate_package_verify_json", "")
        and not getattr(args, "skip_auto_package_verify", False)
    )


def auto_system_verify_enabled(args: argparse.Namespace) -> bool:
    return (
        field_handoff_enabled(args)
        and not getattr(args, "final_gate_system_verify_json", "")
        and not getattr(args, "skip_auto_system_verify", False)
    )


def auto_validation_evidence_needed(args: argparse.Namespace) -> bool:
    return (
        field_handoff_enabled(args)
        and not getattr(args, "skip_auto_validation_evidence", False)
        and (
            not getattr(args, "final_gate_localization_json", [])
            or not getattr(args, "final_gate_production_json", "")
        )
    )


def split_build_validation_videos(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    videos = list(getattr(args, "videos", []) or [])
    explicit_validation = list(getattr(args, "validation_videos", []) or [])
    if explicit_validation:
        return videos, explicit_validation
    if not auto_validation_evidence_needed(args):
        return videos, []
    holdout_count = max(0, int(getattr(args, "auto_validation_holdout_count", 1)))
    if holdout_count <= 0 or len(videos) <= holdout_count:
        return videos, []
    return videos[:-holdout_count], videos[-holdout_count:]


def field_validation_preflight_reasons(
    args: argparse.Namespace,
    *,
    build_videos: list[str],
    validation_videos: list[str],
) -> list[str]:
    if not field_handoff_enabled(args) or getattr(args, "skip_auto_validation_evidence", False):
        return []
    needs_localization = not getattr(args, "final_gate_localization_json", [])
    needs_production = not getattr(args, "final_gate_production_json", "")
    if not needs_localization and not needs_production:
        return []

    reasons: list[str] = []
    if not validation_videos:
        if needs_localization:
            reasons.append("field handoff needs validation video for auto localization evidence")
        if needs_production:
            reasons.append("field handoff needs validation video for auto production replay evidence")
    for video in validation_videos:
        path = Path(video).expanduser()
        if not path.exists() or path.stat().st_size <= 0:
            reasons.append(f"validation video missing or empty: {path}")
    if getattr(args, "videos", None) and not build_videos and not getattr(args, "image_root", ""):
        reasons.append("no build videos remain after reserving validation holdout")
    return reasons


def validation_evidence_dir(work_dir: Path) -> Path:
    return work_dir / "validation_videos"


def prepare_validation_videos_dir(videos: list[str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, video in enumerate(videos):
        source = Path(video).expanduser()
        suffix = ".MP4"
        target = out_dir / f"validation_{idx:02d}{suffix}"
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source.resolve())
    return out_dir


def build_package_verify_command(args: argparse.Namespace, out_json: str | Path) -> list[str]:
    return [
        args.python,
        str(VERIFY_PACKAGE),
        "--allow-extra-top",
        "--json-out",
        str(out_json),
    ]


def build_holdout_localization_command(
    args: argparse.Namespace,
    *,
    bundle: str | Path,
    validation_dir: str | Path,
    out_json: str | Path,
) -> list[str]:
    return [
        args.python,
        str(LOCALIZE_PIPELINE),
        "--mode",
        "compare",
        "--base",
        str(bundle),
        "--final",
        str(bundle),
        "--base-megaloc-cache",
        "",
        "--test-dir",
        str(validation_dir),
        "--stride",
        str(args.auto_validation_stride),
        "--resize",
        str(args.auto_validation_resize),
        "--min-success",
        str(args.final_gate_min_success),
        "--max-ok-to-fail",
        str(args.final_gate_max_ok_to_fail),
        "--max-final-fail-run",
        str(args.final_gate_max_final_fail_run),
        "--out-json",
        str(out_json),
    ]


def build_production_replay_command(
    args: argparse.Namespace,
    *,
    bundle: str | Path,
    model: str | Path,
    query_video: str | Path,
    out_json: str | Path,
) -> list[str]:
    return [
        args.python,
        str(LOCALIZE_PIPELINE),
        "--mode",
        "production-stream",
        "--bundle",
        str(bundle),
        "--model",
        str(model),
        "--query-video",
        str(query_video),
        "--query-video-stride",
        str(args.auto_production_video_stride),
        "--query-video-limit",
        str(args.auto_production_video_limit),
        "--resize-width",
        str(args.auto_production_resize_width),
        "--min-production-success",
        str(args.final_gate_min_production_success),
        "--max-production-wall-p90",
        str(args.final_gate_max_production_wall_p90),
        "--min-production-frames",
        str(args.final_gate_min_production_frames),
        "--max-production-fail-run",
        str(args.final_gate_max_production_fail_run),
        "--min-production-inliers-p5",
        str(args.final_gate_min_production_inliers_p5),
        "--out-json",
        str(out_json),
    ]


def build_system_verify_command(
    args: argparse.Namespace,
    build_summary_json: str | Path,
    out_json: str | Path,
) -> list[str]:
    out_json = Path(out_json)
    return [
        args.python,
        str(SYSTEM_VERIFY),
        "--build-summary-json",
        str(build_summary_json),
        "--json-out",
        str(out_json),
        "--md-out",
        str(out_json.with_suffix(".md")),
        "--skip-runtime",
        "--allow-blocked",
    ]


def summary_overall_ok(path: Path) -> bool:
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("overall_ok"))
    except Exception:
        return False


def build_final_gate_command(
    args: argparse.Namespace,
    work_dir: Path,
    *,
    out_json: str | Path | None = None,
    out_md: str | Path | None = None,
    package_verify_json: str | Path | None = None,
    system_verify_json: str | Path | None = None,
    localization_jsons: list[str | Path] | None = None,
    production_json: str | Path | None = None,
    require_system_verify: bool | None = None,
) -> list[str]:
    cmd = [args.python, str(FINAL_GATE), "--run-dir", str(work_dir)]
    if out_json:
        cmd += ["--out-json", str(out_json)]
        cmd += ["--out-md", str(out_md or Path(out_json).with_suffix(".md"))]
    loc_jsons = args.final_gate_localization_json if localization_jsons is None else [str(path) for path in localization_jsons]
    production_arg = args.final_gate_production_json if production_json is None else str(production_json)
    for path in loc_jsons:
        cmd += ["--localization-json", path]
    if production_arg:
        cmd += ["--production-json", production_arg]
    package_verify_arg = args.final_gate_package_verify_json if package_verify_json is None else str(package_verify_json)
    system_verify_arg = args.final_gate_system_verify_json if system_verify_json is None else str(system_verify_json)
    if package_verify_arg:
        cmd += ["--package-verify-json", package_verify_arg]
    if system_verify_arg:
        cmd += ["--system-verify-json", system_verify_arg]
    if require_final_evidence(args, "require_final_localization"):
        cmd.append("--require-localization")
    if require_final_evidence(args, "require_final_production"):
        cmd.append("--require-production")
    if require_final_evidence(args, "require_final_package_verify"):
        cmd.append("--require-package-verify")
    should_require_system = (
        require_final_evidence(args, "require_final_system_verify")
        if require_system_verify is None
        else require_system_verify
    )
    if should_require_system:
        cmd.append("--require-system-verify")
    cmd += [
        "--min-success", str(args.final_gate_min_success),
        "--max-ok-to-fail", str(args.final_gate_max_ok_to_fail),
        "--max-final-fail-run", str(args.final_gate_max_final_fail_run),
        "--min-production-success", str(args.final_gate_min_production_success),
        "--max-production-wall-p90", str(args.final_gate_max_production_wall_p90),
        "--min-production-frames", str(args.final_gate_min_production_frames),
        "--max-production-fail-run", str(args.final_gate_max_production_fail_run),
        "--min-production-inliers-p5", str(args.final_gate_min_production_inliers_p5),
    ]
    if args.allow_final_gate_fail:
        cmd.append("--allow-fail")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified build pipeline: videos/images -> localizable map.",
    )
    parser.add_argument("--python", default="/usr/bin/python3.12")
    parser.add_argument("--site-name", default="")
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--validation-videos", nargs="*", default=[])
    parser.add_argument("--image-root", default="")
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--stages", default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-final-gate-summary", action="store_true")
    parser.add_argument("--allow-final-gate-fail", action="store_true")
    parser.add_argument(
        "--skip-auto-package-verify",
        action="store_true",
        help="do not auto-run tools/verify_package.py for field handoff when package evidence is not supplied",
    )
    parser.add_argument(
        "--skip-auto-system-verify",
        action="store_true",
        help="do not auto-run tools/system_verify.py for field handoff when system evidence is not supplied",
    )
    parser.add_argument(
        "--skip-auto-validation-evidence",
        action="store_true",
        help="do not reserve validation videos or auto-run holdout localization / production replay evidence",
    )
    parser.add_argument(
        "--handoff-profile",
        choices=["field", "candidate"],
        default="field",
        help=(
            "field requires holdout localization, production replay, package verify, "
            "and system verify before latest_build is updated; candidate is for "
            "research runs that still need build gates but are not field handoffs."
        ),
    )
    parser.add_argument("--final-gate-localization-json", action="append", default=[])
    parser.add_argument("--final-gate-production-json", default="")
    parser.add_argument("--final-gate-package-verify-json", default="")
    parser.add_argument("--final-gate-system-verify-json", default="")
    parser.add_argument("--require-final-localization", action="store_true")
    parser.add_argument("--require-final-production", action="store_true")
    parser.add_argument("--require-final-package-verify", action="store_true")
    parser.add_argument("--require-final-system-verify", action="store_true")
    parser.add_argument("--final-gate-min-success", type=float, default=0.90)
    parser.add_argument("--final-gate-max-ok-to-fail", type=int, default=0)
    parser.add_argument("--final-gate-max-final-fail-run", type=int, default=30)
    parser.add_argument("--final-gate-min-production-success", type=float, default=0.90)
    parser.add_argument("--final-gate-max-production-wall-p90", type=float, default=0.0)
    parser.add_argument("--final-gate-min-production-frames", type=int, default=30)
    parser.add_argument("--final-gate-max-production-fail-run", type=int, default=30)
    parser.add_argument("--final-gate-min-production-inliers-p5", type=float, default=30.0)
    parser.add_argument("--auto-validation-holdout-count", type=int, default=1)
    parser.add_argument("--auto-validation-stride", type=int, default=10)
    parser.add_argument("--auto-validation-resize", default="1280x720")
    parser.add_argument("--auto-production-video-stride", type=int, default=8)
    parser.add_argument("--auto-production-video-limit", type=int, default=300)
    parser.add_argument("--auto-production-resize-width", type=int, default=1280)
    parser.add_argument("--mvroma-root", default=str(DEFAULT_MVROMA))
    parser.add_argument("--mvroma-weights", default=str(DEFAULT_MVROMA / "outdoor_final.pth"))
    parser.add_argument("--doppelgangers-root", default=str(DEFAULT_DOPPELGANGERS))
    parser.add_argument("--doppelgangers-checkpoint", default="")
    parser.add_argument("--use-lfoe", action="store_true")
    parser.add_argument("--backend", choices=["glomap", "lfoe", "mpsfm", "compare"], default="")
    parser.add_argument("--mpsfm-repo", default=str(DEFAULT_MPSFM))
    parser.add_argument("--mpsfm-conf", default="")
    parser.add_argument("--mpsfm-config-yaml", default="")
    args, passthrough = parser.parse_known_args()

    site_name = args.site_name or f"site_{timestamp()}"
    work_dir = Path(args.work_dir) if args.work_dir else BUILD_ROOT / "runs" / site_name
    build_videos, validation_videos = split_build_validation_videos(args)
    validation_preflight_reasons = field_validation_preflight_reasons(
        args,
        build_videos=build_videos,
        validation_videos=validation_videos,
    )
    if validation_preflight_reasons:
        raise SystemExit(
            "field validation preflight failed: "
            + "; ".join(validation_preflight_reasons)
        )

    env = os.environ.copy()
    if DEFAULT_UFM.exists():
        env["PYTHONPATH"] = (
            str(DEFAULT_UFM) + os.pathsep +
            str(DEFAULT_UFM / "UniCeption") + os.pathsep +
            env.get("PYTHONPATH", "")
        )

    cmd = [args.python, str(CORE)]
    if args.config:
        cmd += ["--config", args.config]
    cmd += ["--site-name", site_name, "--work-dir", str(work_dir), "--stages", args.stages]
    if build_videos:
        cmd += ["--videos", *build_videos]
    if args.image_root:
        cmd += ["--image-root", args.image_root]
    if args.resume:
        cmd.append("--resume")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.dry_run:
        cmd.append("--dry-run")

    cmd += ["--mvroma-root", args.mvroma_root, "--mvroma-weights", args.mvroma_weights]
    if Path(args.doppelgangers_root).exists():
        cmd += ["--doppelgangers-root", args.doppelgangers_root]
    if args.doppelgangers_checkpoint:
        cmd += ["--doppelgangers-checkpoint", args.doppelgangers_checkpoint]
    if args.use_lfoe:
        cmd.append("--use-lfoe")
    if args.backend:
        cmd += ["--backend", args.backend]
    if args.mpsfm_repo:
        cmd += ["--mpsfm-repo", args.mpsfm_repo]
    if args.mpsfm_conf:
        cmd += ["--mpsfm-conf", args.mpsfm_conf]
    if args.mpsfm_config_yaml:
        cmd += ["--mpsfm-config-yaml", args.mpsfm_config_yaml]
    cmd += passthrough

    print(f"[build_pipeline] system_root={SYSTEM_ROOT}", flush=True)
    print(f"[build_pipeline] MV-RoMa={args.mvroma_root}", flush=True)
    print(f"[build_pipeline] UFM={DEFAULT_UFM if DEFAULT_UFM.exists() else 'not found'}", flush=True)
    print(f"[build_pipeline] MP-SfM={args.mpsfm_repo}", flush=True)
    subprocess.run(cmd, check=True, env=env)

    final_gate_ok = True
    if (
        work_dir.exists()
        and not args.dry_run
        and not args.skip_final_gate_summary
        and final_gate_required(args.stages)
    ):
        package_verify_json = args.final_gate_package_verify_json
        localization_jsons = list(args.final_gate_localization_json)
        production_json = args.final_gate_production_json
        validation_dir = validation_evidence_dir(work_dir)
        if validation_videos:
            prepare_validation_videos_dir(validation_videos, validation_dir)
            manifest = {
                "version": 1,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "kind": "auto_field_validation_videos",
                "build_videos": build_videos,
                "validation_videos": validation_videos,
                "validation_dir": str(validation_dir),
            }
            (work_dir / "auto_validation_videos_manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        bundle = work_dir / "deploy" / "reloc_map_xfeat_tri.pt"
        model = work_dir / "glomap" / "0"

        if final_gate_ok and validation_videos and not localization_jsons:
            holdout_json = work_dir / "holdout_localization.json"
            holdout_cmd = build_holdout_localization_command(
                args,
                bundle=bundle,
                validation_dir=validation_dir,
                out_json=holdout_json,
            )
            print("[build_pipeline] " + " ".join(holdout_cmd), flush=True)
            result = subprocess.run(holdout_cmd)
            final_gate_ok = result.returncode == 0
            if final_gate_ok:
                localization_jsons = [str(holdout_json)]
            if not final_gate_ok and not args.allow_final_gate_fail:
                raise SystemExit(result.returncode)

        if final_gate_ok and validation_videos and not production_json:
            production_json = str(work_dir / "production_stream_replay.json")
            production_cmd = build_production_replay_command(
                args,
                bundle=bundle,
                model=model,
                query_video=validation_videos[0],
                out_json=production_json,
            )
            print("[build_pipeline] " + " ".join(production_cmd), flush=True)
            result = subprocess.run(production_cmd)
            final_gate_ok = result.returncode == 0
            if not final_gate_ok and not args.allow_final_gate_fail:
                raise SystemExit(result.returncode)

        if auto_package_verify_enabled(args):
            package_verify_json = str(work_dir / "package_verify.json")
            package_cmd = build_package_verify_command(args, package_verify_json)
            print("[build_pipeline] " + " ".join(package_cmd), flush=True)
            result = subprocess.run(package_cmd)
            final_gate_ok = result.returncode == 0
            if not final_gate_ok and not args.allow_final_gate_fail:
                raise SystemExit(result.returncode)

        system_verify_json = args.final_gate_system_verify_json
        if final_gate_ok and auto_system_verify_enabled(args):
            pre_summary = work_dir / "BUILD_GATE_SUMMARY.pre_system_verify.json"
            pre_summary_cmd = build_final_gate_command(
                args,
                work_dir,
                out_json=pre_summary,
                package_verify_json=package_verify_json,
                system_verify_json="",
                localization_jsons=localization_jsons,
                production_json=production_json,
                require_system_verify=False,
            )
            print("[build_pipeline] " + " ".join(pre_summary_cmd), flush=True)
            result = subprocess.run(pre_summary_cmd)
            final_gate_ok = result.returncode == 0
            if not final_gate_ok and not args.allow_final_gate_fail:
                raise SystemExit(result.returncode)

            system_verify_json = str(work_dir / "system_verify.json")
            system_cmd = build_system_verify_command(args, pre_summary, system_verify_json)
            print("[build_pipeline] " + " ".join(system_cmd), flush=True)
            result = subprocess.run(system_cmd)
            final_gate_ok = result.returncode == 0
            if not final_gate_ok and not args.allow_final_gate_fail:
                raise SystemExit(result.returncode)

        summary_cmd = build_final_gate_command(
            args,
            work_dir,
            package_verify_json=package_verify_json,
            system_verify_json=system_verify_json,
            localization_jsons=localization_jsons,
            production_json=production_json,
        )
        print("[build_pipeline] " + " ".join(summary_cmd), flush=True)
        result = subprocess.run(summary_cmd)
        final_gate_ok = result.returncode == 0 and summary_overall_ok(work_dir / "BUILD_GATE_SUMMARY.json")
        if not final_gate_ok and not args.allow_final_gate_fail:
            raise SystemExit(result.returncode)

    latest = latest_link_for_handoff_profile(args)
    latest.parent.mkdir(parents=True, exist_ok=True)
    if work_dir.exists() and not args.dry_run and final_gate_ok:
        rel_symlink(work_dir, latest)
        print(f"[build_pipeline] latest -> {latest}", flush=True)
    elif work_dir.exists() and not args.dry_run:
        print("[build_pipeline] latest not updated because final gate failed", flush=True)


if __name__ == "__main__":
    main()
