from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from preflight_research_backends import (  # noqa: E402
    BackendSpec,
    inspect_backend,
    write_report,
)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def test_missing_repository_is_reported_without_guessing_readiness(tmp_path: Path) -> None:
    result = inspect_backend(
        tmp_path,
        BackendSpec(
            backend_id="missing",
            relative_path="external/missing",
            role="pose_rescue",
            entrypoints=("run.py",),
        ),
    )

    assert result["source_state"] == "MISSING"
    assert result["execution_state"] == "BLOCKED"
    assert result["git_commit"] is None
    assert result["blockers"] == ["repository_missing"]


def test_checkpoint_ready_is_distinct_from_target_site_ready(tmp_path: Path) -> None:
    repo = tmp_path / "external" / "matcher"
    repo.mkdir(parents=True)
    (repo / "LICENSE").write_text("test license\n")
    (repo / "run.py").write_text("print('ok')\n")
    (repo / "weights.ckpt").write_bytes(b"checkpoint")
    _git("init", cwd=repo)
    _git("config", "user.email", "test@example.invalid", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    _git("add", ".", cwd=repo)
    _git("commit", "-m", "fixture", cwd=repo)

    result = inspect_backend(
        tmp_path,
        BackendSpec(
            backend_id="matcher",
            relative_path="external/matcher",
            role="pair_matcher",
            entrypoints=("run.py",),
            checkpoints=("weights.ckpt",),
            adapters=("target_site_adapter.py",),
        ),
    )

    assert result["source_state"] == "CODE_READY"
    assert result["checkpoint_state"] == "READY"
    assert result["execution_state"] == "BLOCKED"
    assert result["git_clean"] is True
    assert result["license_present"] is True
    assert result["missing_adapters"] == ["target_site_adapter.py"]
    assert result["blockers"] == ["adapter_missing:target_site_adapter.py"]


def test_scene_training_backend_never_claims_checkpoint_ready(tmp_path: Path) -> None:
    repo = tmp_path / "external" / "scene_model"
    repo.mkdir(parents=True)
    (repo / "LICENSE").write_text("test license\n")
    (repo / "train.py").write_text("print('train')\n")

    result = inspect_backend(
        tmp_path,
        BackendSpec(
            backend_id="scene_model",
            relative_path="external/scene_model",
            role="standalone_map",
            entrypoints=("train.py",),
            scene_training_required=True,
            manual_blockers=("target_site_training_export_missing",),
        ),
    )

    assert result["checkpoint_state"] == "SCENE_TRAINING_REQUIRED"
    assert result["execution_state"] == "BLOCKED"
    assert "scene_training_required" in result["blockers"]
    assert "target_site_training_export_missing" in result["blockers"]


def test_workspace_artifact_is_checked_outside_the_backend_repository(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "external" / "scene_model"
    repo.mkdir(parents=True)
    (repo / "train.py").write_text("print('train')\n")
    artifact = tmp_path / "experiments" / "target_site" / "manifest.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")

    result = inspect_backend(
        tmp_path,
        BackendSpec(
            backend_id="scene_model",
            relative_path="external/scene_model",
            role="standalone_map",
            entrypoints=("train.py",),
            workspace_artifacts=("experiments/target_site/manifest.json",),
        ),
    )

    assert result["present_workspace_artifacts"] == [
        "experiments/target_site/manifest.json"
    ]
    assert result["missing_workspace_artifacts"] == []


def test_report_is_machine_readable_and_summarizes_states(tmp_path: Path) -> None:
    output = tmp_path / "preflight.json"
    report = write_report(
        output,
        workspace_root=tmp_path,
        specs=(
            BackendSpec(
                backend_id="missing",
                relative_path="external/missing",
                role="pose_rescue",
            ),
        ),
    )

    on_disk = json.loads(output.read_text())
    assert on_disk == report
    assert report["schema_version"] == 1
    assert report["summary"] == {
        "backend_count": 1,
        "blocked": 1,
        "ready_for_smoke": 0,
    }
