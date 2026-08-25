from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from plan_research_method_experiments import build_plan


def make_args(
    tmp_path: Path,
    lfoe_exists: bool = True,
    lfoe_license: bool = True,
    dg_exists: bool = True,
) -> Namespace:
    run = tmp_path / "run"
    db = run / "gluemap" / "database_merged.db"
    images = run / "images"
    db.parent.mkdir(parents=True)
    images.mkdir(parents=True)
    db.write_bytes(b"db")
    lfoe = tmp_path / "glomap_filter"
    dg_root = tmp_path / "doppelgangers-plusplus"
    dg_ckpt = dg_root / "checkpoints" / "checkpoint-dg+visym.pth"
    if lfoe_exists:
        lfoe.write_text("", encoding="utf-8")
        if lfoe_license:
            (tmp_path / "LICENSE").write_text("local grant", encoding="utf-8")
    if dg_exists:
        dg_ckpt.parent.mkdir(parents=True)
        dg_ckpt.write_bytes(b"ckpt")
    return Namespace(
        python="/usr/bin/python3.12",
        run_dir=str(run),
        database="",
        image_root="",
        experiment_root=str(tmp_path / "experiments"),
        colmap_command="/usr/bin/colmap",
        lfoe_command=str(lfoe),
        doppelgangers_root=str(dg_root),
        doppelgangers_checkpoint=str(dg_ckpt),
        doppelgangers_threshold=0.7,
        doppelgangers_filter_scope="cross_video",
    )


def test_research_method_plan_orders_licensed_lfoe_after_colmap(
    tmp_path: Path,
):
    plan = build_plan(make_args(tmp_path))

    names = [step["name"] for step in plan["steps"]]
    statuses = {step["name"]: step["status"] for step in plan["steps"]}
    assert names[:3] == [
        "baseline_colmap_global_db_reuse",
        "lfoe_outlier_edge_filter",
        "doppelgangers_pp_pair_filter",
    ]
    assert statuses["lfoe_outlier_edge_filter"] == "ready"
    assert statuses["doppelgangers_pp_pair_filter"] == "ready"
    baseline = plan["steps"][0]["command"]
    assert baseline[:2] == ["/usr/bin/colmap", "global_mapper"]
    assert "--GlobalMapper.ba_refine_extra_params" in baseline


def test_doppelgangers_step_has_image_root_and_checkpoint(tmp_path: Path):
    plan = build_plan(make_args(tmp_path))
    step = next(item for item in plan["steps"] if item["name"] == "doppelgangers_pp_pair_filter")

    command = step["command"]
    assert "--image-root" in command
    assert command[command.index("--image-root") + 1].endswith("/run/images")
    assert "--doppelgangers-checkpoint" in command
    assert command[command.index("--doppelgangers-checkpoint") + 1].endswith("checkpoint-dg+visym.pth")
    assert any("doppelgangers_plusplus.pdf" in note for note in step["notes"])


def test_doppelgangers_step_calls_preserved_core_without_removed_wrapper(tmp_path: Path):
    plan = build_plan(make_args(tmp_path))
    step = next(item for item in plan["steps"] if item["name"] == "doppelgangers_pp_pair_filter")

    command = step["command"]

    assert command[1].endswith("/pipeline/build_localizable_map_core.py")
    assert "--handoff-profile" not in command


def test_research_method_plan_blocks_missing_optional_tools(tmp_path: Path):
    plan = build_plan(make_args(tmp_path, lfoe_exists=False, dg_exists=False))

    statuses = {step["name"]: step["status"] for step in plan["steps"]}
    assert statuses["baseline_colmap_global_db_reuse"] == "ready"
    assert statuses["lfoe_outlier_edge_filter"] == "blocked"
    assert statuses["doppelgangers_pp_pair_filter"] == "blocked"


def test_research_method_plan_blocks_unlicensed_lfoe(tmp_path: Path):
    plan = build_plan(make_args(tmp_path, lfoe_license=False))
    step = next(item for item in plan["steps"] if item["name"] == "lfoe_outlier_edge_filter")

    assert step["status"] == "blocked"
    assert not step["command"]
    assert any("no license" in note.lower() for note in step["notes"])


def test_edge_prioritization_and_ggpt_use_inrepo_adapters(tmp_path: Path):
    plan = build_plan(make_args(tmp_path))
    statuses = {step["name"]: step["status"] for step in plan["steps"]}
    gap = next(item for item in plan["steps"] if item["name"] == "global_aware_edge_prioritization")
    ggpt = next(item for item in plan["steps"] if item["name"] == "ggpt_dense_geometry")

    assert statuses["global_aware_edge_prioritization"] == "adapter_ready"
    assert gap["command"][1].endswith("/pipeline/pose_graph_init.py")
    assert "--required" in gap["command"]
    assert any("forced" in note.lower() or "VPR-blind" in note for note in gap["notes"])

    assert statuses["ggpt_dense_geometry"] == "admission_ready"
    assert ggpt["command"][1].endswith("/pipeline/ggpt_sidecar.py")
    assert any("visualization_only" in note for note in ggpt["notes"])
