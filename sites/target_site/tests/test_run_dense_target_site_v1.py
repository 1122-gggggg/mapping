from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


TOOL = Path(__file__).parents[1] / "tools" / "run_dense_target_site_v1.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("run_dense_target_site_v1", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_workspace_plan_never_reuses_a_completed_fusion(tmp_path: Path) -> None:
    tool = load_tool()
    workspace = tmp_path / "dense_mvs"
    workspace.mkdir()
    (workspace / "fused.ply").write_text("already complete", encoding="utf-8")

    with pytest.raises(FileExistsError, match="fused.ply"):
        tool.build_plan(workspace)


def test_workspace_plan_marks_missing_phases_for_resume(tmp_path: Path) -> None:
    tool = load_tool()
    workspace = tmp_path / "dense_mvs"

    plan = tool.build_plan(workspace)

    assert plan.needs_undistortion is True
    assert plan.needs_patch_match is True
    assert plan.needs_fusion is True
    assert plan.fused_ply == workspace / "fused.ply"


def test_validate_inputs_reports_missing_registered_images(tmp_path: Path) -> None:
    tool = load_tool()
    model = tmp_path / "model"
    images = tmp_path / "images"
    model.mkdir()
    images.mkdir()

    with pytest.raises(FileNotFoundError, match="cameras.bin"):
        tool.validate_input_layout(model, images)


def test_fusion_writes_a_ply_instead_of_a_colmap_model_directory() -> None:
    tool = load_tool()

    assert tool.fusion_kwargs() == {"input_type": "geometric", "output_type": "ply"}


def test_count_depth_maps_recurses_into_sequence_directories(tmp_path: Path) -> None:
    tool = load_tool()
    depth_maps = tmp_path / "stereo" / "depth_maps"
    (depth_maps / "S01").mkdir(parents=True)
    (depth_maps / "S02").mkdir(parents=True)
    (depth_maps / "S01" / "000001.jpg.geometric.bin").touch()
    (depth_maps / "S01" / "000001.jpg.photometric.bin").touch()
    (depth_maps / "S02" / "000002.jpg.geometric.bin").touch()

    assert tool.count_depth_maps(tmp_path) == 2
