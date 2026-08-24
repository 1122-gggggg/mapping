from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from run_s0_s9 import (  # noqa: E402
    clone_mapper_database,
    gate_passed,
    plan_stages,
    resolve_stage_commands,
)


def _args(tmp_path: Path, **overrides) -> Namespace:
    values = dict(
        run_dir=tmp_path / "target_site_v1",
        python=sys.executable,
        twoview=None,
        intrinsics_seed=None,
        database=None,
        global_mapper_bin=None,
        model=None,
        s5_metrics=None,
        tracking_bundle=None,
        edm_bundle=None,
        baseline_bundle=None,
        result=None,
        package_bundle=None,
        package_config=None,
        start_from="S0",
    )
    values.update(overrides)
    return Namespace(**values)


def test_plan_includes_readme_s0_to_s9_order(tmp_path: Path) -> None:
    run_dir = tmp_path / "target_site_v1"
    run_dir.mkdir()
    stages = [spec["stage"] for spec in plan_stages(_args(tmp_path))]
    assert stages == [
        "S0",
        "S1",
        "S1b",
        "S2",
        "S2b",
        "S3",
        "S4",
        "S5",
        "S5.7",
        "S6",
        "S7",
        "S8",
        "S9",
    ]
    assert "--run-name" in plan_stages(_args(tmp_path))[0]["cmd"]
    assert plan_stages(_args(tmp_path))[0]["cmd"][-1] == "target_site_v1"


def test_s4_fails_closed_without_twoview(tmp_path: Path) -> None:
    run_dir = tmp_path / "target_site_v1"
    run_dir.mkdir()
    spec = next(item for item in plan_stages(_args(tmp_path)) if item["stage"] == "S4")
    with pytest.raises(SystemExit, match="S4"):
        resolve_stage_commands(spec)


def test_s5_7_wires_colmap_as_the_only_global_mapper(tmp_path: Path) -> None:
    run_dir = tmp_path / "target_site_v1"
    run_dir.mkdir()
    required = {
        "database": tmp_path / "database.db",
        "global_mapper_bin": tmp_path / "colmap",
        "twoview": tmp_path / "twoview.pt",
    }
    for path in required.values():
        path.write_bytes(b"x")
    (run_dir / "images").mkdir()
    (run_dir / "forced_bridges.txt").write_text("", encoding="utf-8")
    (run_dir / "forced_bridges.json").write_text("{}", encoding="utf-8")
    gates = run_dir / "gates"
    gates.mkdir()
    (gates / "S4_doppelgangers.json").write_text(
        '{"status": "PASS"}', encoding="utf-8"
    )
    spec = next(
        item for item in plan_stages(_args(tmp_path, **required))
        if item["stage"] == "S5.7"
    )
    cmd = resolve_stage_commands(spec)[0]
    assert "--global-mapper-bin" in cmd
    assert "--lfoe-bin" not in cmd
    assert str(required["global_mapper_bin"]) in cmd


def test_s5_runs_colmap_global_mapper_then_fixed_intrinsics_finalizer(tmp_path: Path) -> None:
    run_dir = tmp_path / "target_site_v1"
    images = run_dir / "images"
    images.mkdir(parents=True)
    database = tmp_path / "database.db"
    global_mapper_bin = tmp_path / "colmap"
    intrinsics = tmp_path / "intrinsics"
    for path in (database, global_mapper_bin):
        path.write_bytes(b"x")
    intrinsics.mkdir()
    (run_dir / "frame_manifest.json").write_text("{}", encoding="utf-8")
    spec = next(
        item
        for item in plan_stages(
            _args(
                tmp_path, database=database, global_mapper_bin=global_mapper_bin,
                intrinsics_seed=intrinsics,
            )
        )
        if item["stage"] == "S5"
    )
    mapper, finalizer = resolve_stage_commands(spec)
    assert mapper[:2] == [str(global_mapper_bin), "global_mapper"]
    assert "--database_path" in mapper
    assert "--GlobalMapper.ba_refine_focal_length" in mapper
    assert "--GlobalMapper.ba_refine_extra_params" in mapper
    assert "finalize_edm_model.py" in finalizer[1]
    assert str(run_dir / "colmap_global" / "0") in finalizer


def test_global_mapper_database_clone_is_immutable_and_collision_safe(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    clone = tmp_path / "work" / "database.db"
    source.write_bytes(b"immutable database")

    receipt = clone_mapper_database(source, clone)

    assert clone.read_bytes() == source.read_bytes()
    assert receipt["source"] == str(source.resolve())
    assert receipt["clone"] == str(clone.resolve())
    with pytest.raises(FileExistsError):
        clone_mapper_database(source, clone)


def test_global_mapper_database_clone_rejects_nonempty_wal(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    source.write_bytes(b"database")
    source.with_name("source.db-wal").write_bytes(b"pending")

    with pytest.raises(ValueError, match="WAL"):
        clone_mapper_database(source, tmp_path / "clone.db")


def test_gate_passed_requires_status_pass(tmp_path: Path) -> None:
    missing = tmp_path / "gates" / "S0_corpus.json"
    ok, detail = gate_passed(missing)
    assert ok is False
    assert "missing" in detail
    missing.parent.mkdir()
    missing.write_text('{"status": "FAIL"}', encoding="utf-8")
    assert gate_passed(missing)[0] is False
    missing.write_text('{"status": "PASS"}', encoding="utf-8")
    assert gate_passed(missing)[0] is True


def test_s9_canonical_command_wires_lineage_inputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "target_site_v1"
    run_dir.mkdir()
    (run_dir / "forced_bridges.json").write_text("{}", encoding="utf-8")
    (run_dir / "corpus_manifest.json").write_text("{}", encoding="utf-8")
    edm_dir = run_dir / "edm"
    edm_dir.mkdir()
    tracking = edm_dir / "target_site_v1_seed_tracking.pt"
    edm = edm_dir / "target_site_v1_reloc_map_edm.pt"
    tracking.write_bytes(b"tracking")
    edm.write_bytes(b"edm")
    result_a = tmp_path / "t01.json"
    result_b = tmp_path / "t02.json"
    result_a.write_text("{}", encoding="utf-8")
    result_b.write_text("{}", encoding="utf-8")

    spec = next(
        item
        for item in plan_stages(_args(tmp_path, result=[result_a, result_b]))
        if item["stage"] == "S9"
    )
    cmd = resolve_stage_commands(spec)[0]
    assert "--corpus-manifest" in cmd
    assert "--edm-bundle" in cmd
    assert "--tracking-bundle" in cmd
    assert "--package-bundle" not in cmd
    assert "--package-config" not in cmd
    assert str((run_dir / "corpus_manifest.json").resolve()) in cmd
    assert str(edm.resolve()) in cmd
    assert str(tracking.resolve()) in cmd
    assert str(result_a) in cmd
    assert str(result_b) in cmd

    missing = next(
        item for item in plan_stages(_args(tmp_path)) if item["stage"] == "S9"
    )
    with pytest.raises(SystemExit, match="S9"):
        resolve_stage_commands(missing)
