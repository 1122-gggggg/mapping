from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from plan_db_reuse_sweep import build_plan, parse_variant


def test_parse_variant_records_glomap_thresholds():
    variant = parse_variant("mv3_a1:3:1.0:900000")

    assert variant.name == "mv3_a1"
    assert variant.min_views == 3
    assert variant.min_angle == 1.0
    assert variant.max_tracks == 900000


def test_build_plan_reuses_existing_database(tmp_path: Path):
    run_dir = tmp_path / "run"
    database = run_dir / "gluemap" / "database_merged.db"
    image_root = run_dir / "images"
    database.parent.mkdir(parents=True)
    image_root.mkdir(parents=True)
    database.write_bytes(b"sqlite placeholder")

    plan = build_plan(Namespace(
        run_dir=str(run_dir),
        database="",
        image_root="",
        output_root=str(tmp_path / "out"),
        variant=["cheap:2:0.5:600000"],
        glomap_command="/bin/glomap",
        optimize_intrinsics=0,
        optimize_principal_point=0,
        skip_retriangulation=False,
    ))

    assert plan["database"]["path"] == str(database.resolve())
    assert plan["image_root"] == str(image_root.resolve())
    assert len(plan["commands"]) == 1
    command = plan["commands"][0]["cmd"]
    assert "--database_path" in command
    assert str(database.resolve()) in command
    assert "--TrackEstablishment.min_num_view_per_track" in command
