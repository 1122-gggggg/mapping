from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest


TOOL = Path(__file__).parents[1] / "tools" / "run_densesfm_refinement.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("run_densesfm_refinement", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_database(path: Path, *, name: str = "S01/frame.jpg", x_offset: float = 0.0) -> None:
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        db.execute(
            "CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)"
        )
        keypoints = np.asarray(
            [[10.0 + x_offset, 20.0, 1.0, 0.0], [30.0, 40.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        db.execute("INSERT INTO images VALUES(?, ?)", (7, name))
        db.execute(
            "INSERT INTO keypoints VALUES(?, ?, ?, ?)",
            (7, keypoints.shape[0], keypoints.shape[1], keypoints.tobytes()),
        )


def test_call_contract_is_refinement_only_and_preserves_target_site_calibration(
    tmp_path: Path,
) -> None:
    tool = load_tool()
    kwargs = tool.dense_call_contract(
        image_paths=[tmp_path / "images" / "S01" / "frame.jpg"],
        input_model=tmp_path / "model",
        output_model=tmp_path / "candidate",
        database_path=tmp_path / "database.db",
        dense_repo=tmp_path / "DenseSfM-Refine",
        mode="points_only",
        chunk_size=321,
        iterations=1,
    )

    assert kwargs["only_basename_in_colmap"] is False
    assert kwargs["database_path"] == str((tmp_path / "database.db").resolve())
    assert kwargs["refine_3D_pts_only"] is True
    assert kwargs["use_pycolmap"] is True
    assert kwargs["colmap_configs"]["no_refine_intrinsics"] is True
    assert kwargs["chunk_size"] == 321
    assert kwargs["refine_iter_n_times"] == 1
    assert kwargs["match_out_pth"] is None
    assert kwargs["covis_pairs_pth"] is None
    assert kwargs["model_refiner_no_filter_pts"] is True


def test_pose_and_points_mode_changes_only_pose_policy(tmp_path: Path) -> None:
    tool = load_tool()
    common = dict(
        image_paths=[tmp_path / "images" / "S01" / "frame.jpg"],
        input_model=tmp_path / "model",
        output_model=tmp_path / "candidate",
        database_path=tmp_path / "database.db",
        dense_repo=tmp_path / "DenseSfM-Refine",
        chunk_size=500,
        iterations=1,
    )

    fixed = tool.dense_call_contract(mode="points_only", **common)
    joint = tool.dense_call_contract(mode="poses_and_points", **common)

    assert fixed["refine_3D_pts_only"] is True
    assert joint["refine_3D_pts_only"] is False
    assert fixed["colmap_configs"] == joint["colmap_configs"]
    assert fixed["only_basename_in_colmap"] == joint["only_basename_in_colmap"]


def test_database_contract_checks_exact_id_name_and_keypoint_coordinates(
    tmp_path: Path,
) -> None:
    tool = load_tool()
    database = tmp_path / "database.db"
    make_database(database)
    records = [
        tool.ModelImageRecord(
            image_id=7,
            name="S01/frame.jpg",
            points2d=np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
        )
    ]

    report = tool.validate_database_contract(records, database)
    assert report["images_checked"] == 1
    assert report["keypoints_checked"] == 2
    assert report["max_xy_error_px"] == 0.0

    wrong_name = [tool.ModelImageRecord(7, "S02/frame.jpg", records[0].points2d)]
    with pytest.raises(RuntimeError, match="name mismatch"):
        tool.validate_database_contract(wrong_name, database)

    bad_database = tmp_path / "bad_database.db"
    make_database(bad_database, x_offset=0.25)
    with pytest.raises(RuntimeError, match="keypoint mismatch"):
        tool.validate_database_contract(records, bad_database)


def test_output_must_not_overlap_source_or_production_artifacts(tmp_path: Path) -> None:
    tool = load_tool()
    source = tmp_path / "run" / "final_model"
    source.mkdir(parents=True)
    protected = [source, tmp_path / "run" / "edm", tmp_path / "release"]

    with pytest.raises(ValueError, match="protected"):
        tool.validate_isolated_output(source, source / "candidate", protected)

    candidate = tmp_path / "experiments" / "densesfm" / "model"
    assert tool.validate_isolated_output(source, candidate, protected) == candidate.resolve()


def test_worker_command_never_invokes_matching_or_sfm_rebuild(tmp_path: Path) -> None:
    tool = load_tool()
    command = tool.build_worker_command(
        python=Path("/opt/dense/bin/python"),
        runner=TOOL,
        contract_path=tmp_path / "contract.json",
        dense_repo=tmp_path / "DenseSfM-Refine",
    )

    assert command == [
        "/opt/dense/bin/python",
        str(TOOL.resolve()),
        "--worker-contract",
        str((tmp_path / "contract.json").resolve()),
        "--dense-repo",
        str((tmp_path / "DenseSfM-Refine").resolve()),
    ]
    assert not any(token in command for token in ("run_full.py", "mapper", "matching"))


def test_runtime_preflight_preserves_virtualenv_python_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = load_tool()
    real_python = tmp_path / "python-real"
    real_python.write_text("", encoding="utf-8")
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(real_python)
    observed = {}

    class Result:
        stdout = '{"torch":"x"}\n'

    def fake_run(command, **kwargs):
        observed["command"] = command
        return Result()

    monkeypatch.setattr(tool.subprocess, "run", fake_run)

    assert tool.preflight_dense_runtime(venv_python, tmp_path) == {"torch": "x"}
    assert observed["command"][0] == str(venv_python)


def test_colmap311_compat_database_reuses_only_selected_images_and_pairs(
    tmp_path: Path,
) -> None:
    tool = load_tool()
    source = tmp_path / "modern.db"
    maximum = tool.MAX_IMAGE_ID
    with sqlite3.connect(source) as db:
        db.executescript(
            """
            CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY, model INTEGER, width INTEGER,
                                 height INTEGER, params BLOB, prior_focal_length INTEGER);
            CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT UNIQUE, camera_id INTEGER);
            CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
            CREATE TABLE descriptors(image_id INTEGER PRIMARY KEY, type INTEGER, rows INTEGER,
                                     cols INTEGER, data BLOB);
            CREATE TABLE matches(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
            CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER,
                                             data BLOB, config INTEGER, F BLOB, E BLOB, H BLOB,
                                             qvec BLOB, tvec BLOB);
            """
        )
        db.execute("INSERT INTO cameras VALUES(1, 1, 100, 80, ?, 1)", (b"camera",))
        for image_id, name in ((7, "S01/a.jpg"), (8, "S01/b.jpg"), (9, "query/c.jpg")):
            db.execute("INSERT INTO images VALUES(?, ?, 1)", (image_id, name))
            db.execute("INSERT INTO keypoints VALUES(?, 1, 2, ?)", (image_id, b"xy"))
            db.execute("INSERT INTO descriptors VALUES(?, 0, 1, 1, ?)", (image_id, b"d"))
        selected_pair = 7 * maximum + 8
        excluded_pair = 7 * maximum + 9
        for pair_id in (selected_pair, excluded_pair):
            db.execute("INSERT INTO matches VALUES(?, 1, 2, ?)", (pair_id, b"m"))
            db.execute(
                "INSERT INTO two_view_geometries VALUES(?, 1, 2, ?, 2, ?, ?, ?, ?, ?)",
                (pair_id, b"g", b"f", b"e", b"h", b"q", b"t"),
            )

    records = [
        tool.ModelImageRecord(7, "S01/a.jpg", np.empty((0, 2))),
        tool.ModelImageRecord(8, "S01/b.jpg", np.empty((0, 2))),
    ]
    output = tmp_path / "compat.db"
    report = tool.build_colmap311_compat_database(records, source, output)

    assert report["images"] == 2
    assert report["matches"] == 1
    assert report["two_view_geometries"] == 1
    with sqlite3.connect(output) as db:
        assert db.execute("SELECT image_id FROM images ORDER BY image_id").fetchall() == [
            (7,),
            (8,),
        ]
        assert db.execute("SELECT pair_id FROM matches").fetchall() == [(selected_pair,)]
        descriptor_columns = [row[1] for row in db.execute("PRAGMA table_info(descriptors)")]
        assert descriptor_columns == ["image_id", "rows", "cols", "data"]
