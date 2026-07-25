#!/usr/bin/env python3
"""Fail-closed readiness audit for optional target_site research backends.

Downloading source code or a generic checkpoint is not equivalent to completing a
target-site localization experiment.  This audit records those states separately so
the experiment queue cannot silently promote an untrained scene model or an adapter
that has never produced target-site poses.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


SCHEMA_VERSION = 1
LICENSE_NAMES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "COPYING.md")


@dataclass(frozen=True)
class BackendSpec:
    backend_id: str
    relative_path: str
    role: str
    entrypoints: tuple[str, ...] = ()
    checkpoints: tuple[str, ...] = ()
    adapters: tuple[str, ...] = ()
    workspace_artifacts: tuple[str, ...] = ()
    scene_training_required: bool = False
    manual_blockers: tuple[str, ...] = ()
    map_contract: str = "fixed_gluemap_sidecar_or_pose_rescue"


def _git_output(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _existing_relative_paths(root: Path, paths: Iterable[str]) -> list[str]:
    return [relative for relative in paths if (root / relative).is_file()]


def inspect_backend(workspace_root: str | Path, spec: BackendSpec) -> dict[str, object]:
    workspace = Path(workspace_root).expanduser().resolve()
    repo = workspace / spec.relative_path
    spec_payload = asdict(spec)
    for key in (
        "entrypoints",
        "checkpoints",
        "adapters",
        "workspace_artifacts",
        "manual_blockers",
    ):
        spec_payload[key] = list(spec_payload[key])
    result: dict[str, object] = {
        **spec_payload,
        "path": str(repo),
        "source_state": "MISSING",
        "checkpoint_state": "NOT_REQUIRED",
        "execution_state": "BLOCKED",
        "git_commit": None,
        "git_clean": None,
        "license_present": False,
        "license_path": None,
        "present_entrypoints": [],
        "missing_entrypoints": list(spec.entrypoints),
        "present_checkpoints": [],
        "missing_checkpoints": list(spec.checkpoints),
        "present_adapters": [],
        "missing_adapters": list(spec.adapters),
        "present_workspace_artifacts": [],
        "missing_workspace_artifacts": list(spec.workspace_artifacts),
        "blockers": ["repository_missing", *spec.manual_blockers],
    }
    if not repo.is_dir():
        return result

    present_entrypoints = _existing_relative_paths(repo, spec.entrypoints)
    missing_entrypoints = [
        relative for relative in spec.entrypoints if relative not in present_entrypoints
    ]
    present_checkpoints = _existing_relative_paths(repo, spec.checkpoints)
    missing_checkpoints = [
        relative for relative in spec.checkpoints if relative not in present_checkpoints
    ]
    present_adapters = _existing_relative_paths(repo, spec.adapters)
    missing_adapters = [
        relative for relative in spec.adapters if relative not in present_adapters
    ]
    present_workspace_artifacts = _existing_relative_paths(
        workspace,
        spec.workspace_artifacts,
    )
    missing_workspace_artifacts = [
        relative
        for relative in spec.workspace_artifacts
        if relative not in present_workspace_artifacts
    ]
    license_path = next((repo / name for name in LICENSE_NAMES if (repo / name).is_file()), None)
    commit = _git_output(repo, "rev-parse", "HEAD")
    dirty = _git_output(repo, "status", "--porcelain")

    blockers: list[str] = []
    blockers.extend(f"entrypoint_missing:{path}" for path in missing_entrypoints)
    if not spec.scene_training_required:
        blockers.extend(f"checkpoint_missing:{path}" for path in missing_checkpoints)
    blockers.extend(f"adapter_missing:{path}" for path in missing_adapters)
    blockers.extend(
        f"workspace_artifact_missing:{path}"
        for path in missing_workspace_artifacts
    )
    blockers.extend(spec.manual_blockers)
    if spec.scene_training_required:
        blockers.insert(0, "scene_training_required")

    source_state = "CODE_READY" if not missing_entrypoints else "CODE_INCOMPLETE"
    if spec.scene_training_required:
        checkpoint_state = "SCENE_TRAINING_REQUIRED"
    elif spec.checkpoints:
        checkpoint_state = "READY" if not missing_checkpoints else "MISSING"
    else:
        checkpoint_state = "NOT_REQUIRED"

    return {
        **result,
        "source_state": source_state,
        "checkpoint_state": checkpoint_state,
        "execution_state": "READY_FOR_SMOKE" if not blockers else "BLOCKED",
        "git_commit": commit,
        "git_clean": None if dirty is None else dirty == "",
        "license_present": license_path is not None,
        "license_path": None if license_path is None else str(license_path),
        "present_entrypoints": present_entrypoints,
        "missing_entrypoints": missing_entrypoints,
        "present_checkpoints": present_checkpoints,
        "missing_checkpoints": missing_checkpoints,
        "present_adapters": present_adapters,
        "missing_adapters": missing_adapters,
        "present_workspace_artifacts": present_workspace_artifacts,
        "missing_workspace_artifacts": missing_workspace_artifacts,
        "blockers": blockers,
    }


def default_specs() -> tuple[BackendSpec, ...]:
    return (
        BackendSpec(
            backend_id="ggpt",
            relative_path="建圖/external_tools/GGPT",
            role="geometry_sidecar",
            entrypoints=("run_demo.py",),
            checkpoints=("ckpts/model.step228000.pth",),
            adapters=("scripts/run_gluemap_pi3_sidecar.py",),
            manual_blockers=(
                "target_site_tile_manifest_missing",
                "target_site_gpu_smoke_not_run",
                "root_license_missing_deployment_review",
            ),
            map_contract="fixed_gluemap_poses_and_intrinsics_only",
        ),
        BackendSpec(
            backend_id="slim",
            relative_path="建圖/external_tools/SLiM",
            role="pair_matcher_candidate",
            entrypoints=("src/slim.py", "src/lightning_slim.py", "test.py"),
            checkpoints=("ckpt/megadepth_19epochs.ckpt",),
            adapters=("runtime_single_pair.py",),
            manual_blockers=(
                "isolated_cuda118_environment_missing",
                "target_site_pair_benchmark_not_run",
                "matcher_specific_2d_to_3d_map_missing",
            ),
        ),
        BackendSpec(
            backend_id="deviloc",
            relative_path="建圖/external_tools/DeViLoc",
            role="depth_lift_pose_rescue",
            entrypoints=("evaluate.py", "deviloc/models/model.py"),
            checkpoints=("pretrained/deviloc_weights.ckpt",),
            adapters=("configs/target_site.yaml",),
            manual_blockers=(
                "isolated_legacy_cuda_environment_missing",
                "topicfm_dependency_unresolved",
                "target_site_gpu_smoke_not_run",
            ),
        ),
        BackendSpec(
            backend_id="rscore_scrstudio",
            relative_path="建圖/external_tools/scrstudio",
            role="standalone_scene_coordinate_map_or_pose_rescue",
            entrypoints=("scrstudio/scripts/train.py",),
            workspace_artifacts=(
                "建圖/target_site/runs/target_site_v1/experiments/"
                "map_localization_improvement_20260723/research_backends/"
                "rscore_target_site/manifest.json",
            ),
            scene_training_required=True,
            manual_blockers=(
                "three_seed_training_not_run",
            ),
            map_contract="separate_scene_model_supervised_by_frozen_gluemap_poses",
        ),
        BackendSpec(
            backend_id="splathloc",
            relative_path="建圖/external_tools/SplatHLoc",
            role="feature_gaussian_pose_rescue",
            entrypoints=("train_featgs.py",),
            adapters=("configs/target_site",),
            scene_training_required=True,
            manual_blockers=(
                "mixvpr_assets_unverified",
                "target_site_training_export_missing",
                "research_only_source_license_review_required",
            ),
            map_contract="separate_feature_gaussian_map_from_frozen_gluemap_poses",
        ),
        BackendSpec(
            backend_id="imloc",
            relative_path="建圖/external_tools/ImLoc",
            role="depth_lift_pose_rescue",
            manual_blockers=("official_implementation_not_present",),
        ),
    )


def build_report(
    workspace_root: str | Path,
    specs: Sequence[BackendSpec],
) -> dict[str, object]:
    backends = [inspect_backend(workspace_root, spec) for spec in specs]
    ready = sum(row["execution_state"] == "READY_FOR_SMOKE" for row in backends)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(Path(workspace_root).expanduser().resolve()),
        "policy": {
            "download_is_not_experiment_completion": True,
            "scene_models_require_target_site_training": True,
            "gluemap_core_is_immutable": True,
            "heavy_backends_must_run_sequentially": True,
        },
        "summary": {
            "backend_count": len(backends),
            "blocked": len(backends) - ready,
            "ready_for_smoke": ready,
        },
        "backends": backends,
    }


def write_report(
    output: str | Path,
    *,
    workspace_root: str | Path,
    specs: Sequence[BackendSpec],
) -> dict[str, object]:
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(workspace_root, specs)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    workspace_default = Path(__file__).resolve().parents[3]
    parser.add_argument("--workspace-root", type=Path, default=workspace_default)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = write_report(
        args.output,
        workspace_root=args.workspace_root,
        specs=default_specs(),
    )
    print(json.dumps(report["summary"], sort_keys=True))
    return 0 if report["summary"]["blocked"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
