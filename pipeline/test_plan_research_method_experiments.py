from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from plan_research_method_experiments import build_plan


def make_args(tmp_path: Path, lfoe_exists: bool = True, dg_exists: bool = True) -> Namespace:
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
    if dg_exists:
        dg_ckpt.parent.mkdir(parents=True)
        dg_ckpt.write_bytes(b"ckpt")
    return Namespace(
        python="/usr/bin/python3.12",
        run_dir=str(run),
        database="",
        image_root="",
        experiment_root=str(tmp_path / "experiments"),
        glomap_command="/usr/bin/glomap",
        lfoe_command=str(lfoe),
        doppelgangers_root=str(dg_root),
        doppelgangers_checkpoint=str(dg_ckpt),
        doppelgangers_threshold=0.7,
        doppelgangers_filter_scope="cross_video",
    )


def test_research_method_plan_orders_ready_lfoe_before_doppelgangers(tmp_path: Path):
    plan = build_plan(make_args(tmp_path))

    names = [step["name"] for step in plan["steps"]]
    statuses = {step["name"]: step["status"] for step in plan["steps"]}
    assert names[:3] == [
        "baseline_glomap_db_reuse",
        "lfoe_outlier_edge_filter",
        "doppelgangers_pp_pair_filter",
    ]
    assert statuses["lfoe_outlier_edge_filter"] == "ready"
    assert statuses["doppelgangers_pp_pair_filter"] == "ready"


def test_doppelgangers_step_has_image_root_and_checkpoint(tmp_path: Path):
    plan = build_plan(make_args(tmp_path))
    step = next(item for item in plan["steps"] if item["name"] == "doppelgangers_pp_pair_filter")

    command = step["command"]
    assert "--image-root" in command
    assert command[command.index("--image-root") + 1].endswith("/run/images")
    assert "--doppelgangers-checkpoint" in command
    assert command[command.index("--doppelgangers-checkpoint") + 1].endswith("checkpoint-dg+visym.pth")
    assert any("doppelgangers_plusplus.pdf" in note for note in step["notes"])


def test_doppelgangers_step_is_candidate_not_field_handoff(tmp_path: Path):
    plan = build_plan(make_args(tmp_path))
    step = next(item for item in plan["steps"] if item["name"] == "doppelgangers_pp_pair_filter")

    command = step["command"]

    assert "--handoff-profile" in command
    assert command[command.index("--handoff-profile") + 1] == "candidate"


def test_research_method_plan_blocks_missing_optional_tools(tmp_path: Path):
    plan = build_plan(make_args(tmp_path, lfoe_exists=False, dg_exists=False))

    statuses = {step["name"]: step["status"] for step in plan["steps"]}
    assert statuses["baseline_glomap_db_reuse"] == "ready"
    assert statuses["lfoe_outlier_edge_filter"] == "blocked"
    assert statuses["doppelgangers_pp_pair_filter"] == "blocked"
