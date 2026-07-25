from __future__ import annotations

import json
import math
import sqlite3
import struct
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

import audit_dg_graph as audit  # noqa: E402
import run_gluemap_memory_safe as launcher  # noqa: E402
from audit_dg_graph import (  # noqa: E402
    LOFTR_SEQUENCE_ALLOWLIST,
    FIXED_CAMERA_CONTRACT,
    classify_accepted_cross_pairs,
    evaluate_loftr_promotion,
    load_forced_pairs,
    loftr_database_isolation,
    loftr_trigger_contract,
    targeted_loftr_branch_gate,
    independent_bridge_count,
    largest_component_fraction,
    robust_sequence_component_fraction,
)


def _write_sift_cap_fixture(
    root: Path,
    *,
    row_overrides: dict[int, int] | None = None,
    emit_runtime_marker: bool | None = None,
) -> dict[str, Path | list[str]]:
    names = [f"S{index % 6:02d}/{index:06d}.jpg" for index in range(240)]
    frame_manifest = root / "frame_manifest.json"
    frame_manifest.write_text(
        json.dumps(
            {
                "n_frames": len(names),
                "frames": [{"name": name} for name in names],
            }
        ),
        encoding="utf-8",
    )
    config = root / "gluemap_config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "sift_max_num_features": 2048,
                "sift_max_num_orientations": 1,
            }
        ),
        encoding="utf-8",
    )
    guard_log = root / "resource_guard.log"
    database = root / "database_sift.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE cameras(
            camera_id INTEGER PRIMARY KEY,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL
        );
        CREATE TABLE images(
            image_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            camera_id INTEGER NOT NULL
        );
        CREATE TABLE keypoints(
            image_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL,
            cols INTEGER NOT NULL, data BLOB NOT NULL
        );
        CREATE TABLE descriptors(
            image_id INTEGER PRIMARY KEY, type INTEGER NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL, data BLOB NOT NULL
        );
        """
    )
    connection.execute("INSERT INTO cameras VALUES (1, 1920, 1080)")
    overrides = row_overrides or {}
    for image_id, name in enumerate(names, 1):
        rows = overrides.get(image_id, 1 + (image_id - 1) % 5)
        connection.execute("INSERT INTO images VALUES (?, ?, 1)", (image_id, name))
        keypoint_blob = b"".join(
            struct.pack(
                "<6f",
                20.0 + row_index,
                30.0 + row_index,
                row_index + 1.0,
                0.0,
                0.0,
                row_index + 1.0,
            )
            for row_index in range(rows)
        )
        descriptor_blob = b"".join(
            bytes([(row_index % 251) + 1]) * 128 for row_index in range(rows)
        )
        connection.execute(
            "INSERT INTO keypoints VALUES (?, ?, 6, ?)",
            (image_id, rows, keypoint_blob),
        )
        connection.execute(
            "INSERT INTO descriptors VALUES (?, 0, ?, 128, ?)",
            (image_id, rows, descriptor_blob),
        )
    connection.commit()
    connection.close()
    should_emit_marker = (
        not row_overrides if emit_runtime_marker is None else emit_runtime_marker
    )
    if should_emit_marker:
        result = launcher.clamp_sift_database_rows(database, max_rows=2048)
        launcher.write_sift_runtime_marker(
            guard_log,
            result,
            max_num_features=2048,
            max_num_orientations=1,
        )
    else:
        guard_log.write_text("startup_preflight=PASS\n", encoding="utf-8")
    return {
        "names": names,
        "frame_manifest": frame_manifest,
        "config": config,
        "guard_log": guard_log,
        "database": database,
    }


def test_sift_db_cap_reads_immutable_sqlite_and_accepts_exact_240_image_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _write_sift_cap_fixture(tmp_path)
    opened: dict[str, object] = {}
    original_connect = sqlite3.connect

    def capturing_connect(database, *args, **kwargs):
        opened["database"] = database
        opened["uri"] = kwargs.get("uri")
        return original_connect(database, *args, **kwargs)

    monkeypatch.setattr(audit.sqlite3, "connect", capturing_connect)

    result = audit.audit_sift_database(
        fixture["database"],
        config_path=fixture["config"],
        guard_log_path=fixture["guard_log"],
        frame_manifest_path=fixture["frame_manifest"],
    )

    assert result["ok"] is True
    assert opened["uri"] is True
    assert "mode=ro&immutable=1" in str(opened["database"])
    assert result["manifest_image_count"] == 240
    assert result["database_image_count"] == 240
    assert result["keypoint_rows_min"] == 1
    assert result["keypoint_rows_avg"] == pytest.approx(3.0)
    assert result["keypoint_rows_max"] == 5
    assert result["descriptor_rows_avg"] == pytest.approx(3.0)
    assert result["descriptor_rows_max"] == 5
    assert result["checks"]["runtime_marker_exact"] is True
    assert result["checks"]["database_schema_exact"] is True
    assert result["checks"]["blob_layout_valid"] is True
    assert result["checks"]["keypoints_finite_valid"] is True
    assert result["violations"] == []
    assert len(result["database_sha256"]) == 64


def test_sift_db_cap_reproduces_multi_orientation_rows_above_2048(
    tmp_path: Path,
) -> None:
    fixture = _write_sift_cap_fixture(
        tmp_path, row_overrides={1: 2049, 2: 2500, 3: 4047}
    )

    result = audit.audit_sift_database(
        fixture["database"],
        config_path=fixture["config"],
        guard_log_path=fixture["guard_log"],
        frame_manifest_path=fixture["frame_manifest"],
    )

    assert result["ok"] is False
    assert result["keypoint_rows_max"] == 4047
    assert result["descriptor_rows_max"] == 4047
    assert result["row_cap_violation_count"] == 3
    assert any("2048" in violation for violation in result["violations"])


def test_sift_db_cap_rejects_nonempty_wal_without_opening_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _write_sift_cap_fixture(tmp_path)
    Path(str(fixture["database"]) + "-wal").write_bytes(b"uncheckpointed")
    monkeypatch.setattr(
        audit.sqlite3,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("database must not open with a live WAL")
        ),
    )

    result = audit.audit_sift_database(
        fixture["database"],
        config_path=fixture["config"],
        guard_log_path=fixture["guard_log"],
        frame_manifest_path=fixture["frame_manifest"],
    )

    assert result["ok"] is False
    assert result["database_opened"] is False
    assert result["wal_size_bytes"] > 0


def test_sift_db_cap_ignores_unselected_database_merged_artifacts(
    tmp_path: Path,
) -> None:
    fixture = _write_sift_cap_fixture(tmp_path)
    (tmp_path / "database_merged.db-wal").write_bytes(b"out of scope")

    result = audit.audit_sift_database(
        fixture["database"],
        config_path=fixture["config"],
        guard_log_path=fixture["guard_log"],
        frame_manifest_path=fixture["frame_manifest"],
    )

    assert result["ok"] is True
    assert result["database_path"].endswith("database_sift.db")


@pytest.mark.parametrize(
    "mutation",
    [
        "database_name",
        "descriptor_missing",
        "descriptor_rows",
        "orientation_two",
        "guard_line",
    ],
)
def test_sift_db_cap_fails_closed_on_cardinality_policy_or_log_drift(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _write_sift_cap_fixture(tmp_path)
    if mutation == "database_name":
        connection = sqlite3.connect(fixture["database"])
        connection.execute(
            "UPDATE images SET name='S99/not-in-manifest.jpg' WHERE image_id=1"
        )
        connection.commit()
        connection.close()
    elif mutation == "descriptor_missing":
        connection = sqlite3.connect(fixture["database"])
        connection.execute("DELETE FROM descriptors WHERE image_id=1")
        connection.commit()
        connection.close()
    elif mutation == "descriptor_rows":
        connection = sqlite3.connect(fixture["database"])
        connection.execute("UPDATE descriptors SET rows=999 WHERE image_id=1")
        connection.commit()
        connection.close()
    elif mutation == "orientation_two":
        Path(fixture["config"]).write_text(
            "sift_max_num_features: 2048\nsift_max_num_orientations: 2\n",
            encoding="utf-8",
        )
    else:
        Path(fixture["guard_log"]).write_text(
            "sift_row_cap=PASS sift_max_num_features=2048 "
            "sift_max_num_orientations=1 "
            "max_keypoint_descriptor_rows_per_image=2048\n",
            encoding="utf-8",
        )

    result = audit.audit_sift_database(
        fixture["database"],
        config_path=fixture["config"],
        guard_log_path=fixture["guard_log"],
        frame_manifest_path=fixture["frame_manifest"],
    )

    assert result["ok"] is False
    assert result["violations"]


@pytest.mark.parametrize(
    "mutation",
    [
        "images_extra_column",
        "keypoint_cols",
        "descriptor_cols",
        "descriptor_missing_type",
        "descriptor_type_nonzero",
        "descriptor_type_text",
        "descriptor_text",
        "descriptor_blob_length",
    ],
)
def test_sift_db_cap_rejects_schema_type_cols_or_blob_layout_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _write_sift_cap_fixture(tmp_path)
    with sqlite3.connect(fixture["database"]) as connection:
        if mutation == "images_extra_column":
            connection.execute("ALTER TABLE images ADD COLUMN stale INTEGER")
        elif mutation == "keypoint_cols":
            connection.execute("UPDATE keypoints SET cols=5 WHERE image_id=1")
        elif mutation == "descriptor_cols":
            connection.execute("UPDATE descriptors SET cols=127 WHERE image_id=1")
        elif mutation == "descriptor_missing_type":
            connection.executescript(
                """
                ALTER TABLE descriptors RENAME TO descriptors_original;
                CREATE TABLE descriptors(
                    image_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL,
                    cols INTEGER NOT NULL, data BLOB NOT NULL
                );
                INSERT INTO descriptors(image_id, rows, cols, data)
                SELECT image_id, rows, cols, data FROM descriptors_original;
                DROP TABLE descriptors_original;
                """
            )
        elif mutation == "descriptor_type_nonzero":
            connection.execute("UPDATE descriptors SET type=1 WHERE image_id=1")
        elif mutation == "descriptor_type_text":
            connection.execute("UPDATE descriptors SET type='SIFT' WHERE image_id=1")
        elif mutation == "descriptor_text":
            connection.execute("UPDATE descriptors SET data='not-a-blob' WHERE image_id=1")
        else:
            connection.execute(
                "UPDATE descriptors SET data=? WHERE image_id=1", (b"short",)
            )

    result = audit.audit_sift_database(
        fixture["database"],
        config_path=fixture["config"],
        guard_log_path=fixture["guard_log"],
        frame_manifest_path=fixture["frame_manifest"],
    )

    assert result["ok"] is False
    assert result["checks"]["database_schema_exact"] is (
        mutation not in {"images_extra_column", "descriptor_missing_type"}
    )
    assert result["violations"]


@pytest.mark.parametrize(
    "invalid_row",
    [
        (10.0, 10.0, math.nan, 0.0, 0.0, 2.0),
        (10.0, 10.0, 0.0, 0.0, 0.0, 0.0),
        (8000.0, 10.0, 2.0, 0.0, 0.0, 2.0),
    ],
)
def test_sift_db_cap_rejects_nonfinite_nonpositive_or_extreme_keypoints(
    tmp_path: Path,
    invalid_row: tuple[float, ...],
) -> None:
    fixture = _write_sift_cap_fixture(tmp_path)
    with sqlite3.connect(fixture["database"]) as connection:
        rows, cols, data = connection.execute(
            "SELECT rows, cols, data FROM keypoints WHERE image_id=1"
        ).fetchone()
        connection.execute(
            "UPDATE keypoints SET data=? WHERE image_id=1",
            (struct.pack("<6f", *invalid_row) + data[cols * 4 :],),
        )
        assert rows >= 1

    result = audit.audit_sift_database(
        fixture["database"],
        config_path=fixture["config"],
        guard_log_path=fixture["guard_log"],
        frame_manifest_path=fixture["frame_manifest"],
    )

    assert result["ok"] is False
    assert result["checks"]["keypoints_finite_valid"] is False
    assert result["invalid_keypoint_row_count"] == 1


def test_sift_db_cap_accepts_runtime_record_of_explicit_invalid_row_removal(
    tmp_path: Path,
) -> None:
    fixture = _write_sift_cap_fixture(tmp_path, emit_runtime_marker=False)
    with sqlite3.connect(fixture["database"]) as connection:
        rows, cols, data = connection.execute(
            "SELECT rows, cols, data FROM keypoints WHERE image_id=1"
        ).fetchone()
        connection.execute(
            "UPDATE keypoints SET data=? WHERE image_id=1",
            (
                struct.pack("<6f", 10.0, 10.0, math.nan, 0.0, 0.0, 2.0)
                + data[cols * 4 :],
            ),
        )
        assert rows >= 1
    clamp_result = launcher.clamp_sift_database_rows(
        fixture["database"], max_rows=2048
    )
    launcher.write_sift_runtime_marker(
        fixture["guard_log"],
        clamp_result,
        max_num_features=2048,
        max_num_orientations=1,
    )

    result = audit.audit_sift_database(
        fixture["database"],
        config_path=fixture["config"],
        guard_log_path=fixture["guard_log"],
        frame_manifest_path=fixture["frame_manifest"],
    )

    assert result["ok"] is True
    assert result["invalid_keypoint_row_count"] == 0
    assert result["runtime_marker"]["removed_invalid_rows"] == 1


def test_sift_db_cap_rejects_stale_runtime_marker_after_feature_blob_change(
    tmp_path: Path,
) -> None:
    fixture = _write_sift_cap_fixture(tmp_path)
    with sqlite3.connect(fixture["database"]) as connection:
        rows, cols, data = connection.execute(
            "SELECT rows, cols, data FROM keypoints WHERE image_id=1"
        ).fetchone()
        values = list(struct.unpack(f"<{rows * cols}f", data))
        values[2] = values[2] + 0.5
        connection.execute(
            "UPDATE keypoints SET data=? WHERE image_id=1",
            (struct.pack(f"<{len(values)}f", *values),),
        )

    result = audit.audit_sift_database(
        fixture["database"],
        config_path=fixture["config"],
        guard_log_path=fixture["guard_log"],
        frame_manifest_path=fixture["frame_manifest"],
    )

    assert result["ok"] is False
    assert result["checks"]["runtime_marker_exact"] is False
    assert any("runtime marker" in item for item in result["violations"])


def test_sift_db_cap_failure_writes_g40_before_importing_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fixture = _write_sift_cap_fixture(
        run_dir, row_overrides={1: 2049, 2: 2500, 3: 4047}
    )
    image_root = run_dir / "images"
    image_root.mkdir()
    twoview = run_dir / "twoview.pt"
    twoview.write_bytes(b"must not be loaded")
    forced_pairs = run_dir / "forced_bridges.txt"
    forced_pairs.write_text("# none\n", encoding="utf-8")
    forced_manifest = run_dir / "forced_bridges.json"
    forced_manifest.write_text('{"fwd": [], "rev": []}\n', encoding="utf-8")
    gate_dir = run_dir / "gates"
    gate_dir.mkdir()
    (gate_dir / "S3_pairs.json").write_text(
        '{"stage":"S3_pairs","status":"PASS","ok":true}\n',
        encoding="utf-8",
    )
    out = gate_dir / "S4_doppelgangers.json"
    torch_loaded = {"value": False}
    fake_torch = ModuleType("torch")

    def forbidden_load(*_args, **_kwargs):
        torch_loaded["value"] = True
        raise AssertionError("DG tensor loaded before SIFT_DB_CAP")

    fake_torch.load = forbidden_load
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(audit, "assert_gate_chain_fresh", lambda *_args: True)

    with pytest.raises(SystemExit) as exit_info:
        audit.main(
            [
                "--twoview",
                str(twoview),
                "--image-root",
                str(image_root),
                "--forced-pairs",
                str(forced_pairs),
                "--forced-manifest",
                str(forced_manifest),
                "--database",
                str(fixture["database"]),
                "--config",
                str(fixture["config"]),
                "--guard-log",
                str(fixture["guard_log"]),
                "--frame-manifest",
                str(fixture["frame_manifest"]),
                "--out",
                str(out),
            ]
        )

    assert exit_info.value.code
    assert torch_loaded["value"] is False
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["required_ids"] == ["G4.0", "G4.1", "G4.2", "G4.3", "G4.4"]
    assert payload["checks"][0]["id"] == "G4.0"
    assert payload["checks"][0]["state"] == "FAIL"
    assert payload["stage_metrics"]["sift_db_cap"]["database_sha256"]
    inputs = payload["provenance"]["input_artifacts"]
    for label in (
        "sift_database",
        "gluemap_config",
        "resource_guard_log",
        "frame_manifest",
    ):
        assert inputs[label]["sha256"]
    assert (
        inputs["sift_database"]["sha256"]
        == payload["stage_metrics"]["sift_db_cap"]["database_sha256"]
    )


@pytest.mark.parametrize(
    "missing_option",
    ["--database", "--config", "--guard-log", "--frame-manifest"],
)
def test_s4_cli_requires_every_sift_db_cap_artifact(
    tmp_path: Path, missing_option: str
) -> None:
    options = {
        "--twoview": tmp_path / "twoview.pt",
        "--image-root": tmp_path / "images",
        "--forced-pairs": tmp_path / "pairs.txt",
        "--forced-manifest": tmp_path / "pairs.json",
        "--database": tmp_path / "database_sift.db",
        "--config": tmp_path / "gluemap_config.yaml",
        "--guard-log": tmp_path / "resource_guard.log",
        "--frame-manifest": tmp_path / "frame_manifest.json",
        "--out": tmp_path / "S4.json",
    }
    argv = [
        item
        for option, value in options.items()
        if option != missing_option
        for item in (option, str(value))
    ]

    with pytest.raises(SystemExit):
        audit.parse_args(argv)


def test_all_dg_accepted_cross_pairs_are_evidence_with_source_labels() -> None:
    names = ["F/a.jpg", "F/b.jpg", "R/x.jpg", "R/y.jpg"]
    accepted = classify_accepted_cross_pairs(
        names,
        [(0, 2), (1, 3), (0, 1)],
        [0.91, 0.95, 0.99],
        threshold=0.8,
        directions={"F": "fwd", "R": "rev"},
        conditional_pairs={("F/b.jpg", "R/y.jpg")},
    )

    assert accepted == {
        ("F", "R"): [
            {
                "left": "F/a.jpg",
                "right": "R/x.jpg",
                "score": 0.91,
                "source": "natural",
            },
            {
                "left": "F/b.jpg",
                "right": "R/y.jpg",
                "score": 0.95,
                "source": "conditional",
            },
        ]
    }


def test_pure_natural_dg_pairs_can_form_two_independent_bridges() -> None:
    names = ["F/a.jpg", "F/b.jpg", "R/x.jpg", "R/y.jpg"]
    accepted = classify_accepted_cross_pairs(
        names,
        [(0, 2), (1, 3)],
        [0.91, 0.95],
        threshold=0.8,
        directions={"F": "fwd", "R": "rev"},
        conditional_pairs=set(),
    )
    positions = {"F/a.jpg": 0.0, "F/b.jpg": 1.0, "R/x.jpg": 0.0, "R/y.jpg": 1.0}
    normalized = [
        (positions[row["left"]], positions[row["right"]])
        for row in accepted[("F", "R")]
    ]

    assert independent_bridge_count(normalized, minimum_separation=0.25) == 2
    assert {row["source"] for row in accepted[("F", "R")]} == {"natural"}


def test_largest_component_fraction_counts_isolates() -> None:
    fraction, size = largest_component_fraction(5, [(0, 1), (1, 2), (3, 4)])

    assert fraction == 0.6
    assert size == 3


def test_s4_declares_the_same_fixed_fuhe_camera_contract() -> None:
    assert FIXED_CAMERA_CONTRACT == {
        "camera_count": 1,
        "model": "PINHOLE",
        "width": 1920,
        "height": 1080,
        "params": [1396.8086675255472, 1396.8086675255472, 960.0, 540.0],
        "maximum_drift": 1e-6,
    }


def test_independent_bridges_require_separation_on_both_sequences() -> None:
    pairs = [(0.05, 0.95), (0.10, 0.90), (0.85, 0.15), (0.90, 0.10)]

    assert independent_bridge_count(pairs, minimum_separation=0.25) == 2


def test_clustered_bridges_count_as_one() -> None:
    pairs = [(0.05, 0.95), (0.10, 0.90), (0.15, 0.85)]

    assert independent_bridge_count(pairs, minimum_separation=0.25) == 1


def test_forced_pair_parser_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "pairs.txt"
    path.write_text("# generated candidates\n\nS01/a.jpg S02/b.jpg\n", encoding="utf-8")

    assert load_forced_pairs(path) == {("S01/a.jpg", "S02/b.jpg")}


def test_robust_sequence_graph_ignores_nonessential_single_hinges() -> None:
    directions = {"F1": "fwd", "F2": "fwd", "R1": "rev", "R2": "rev"}
    edges = {("F1", "F2"), ("F1", "R2"), ("F2", "R1"), ("R1", "R2")}
    bridge_counts = {("F1", "R2"): 1, ("F2", "R1"): 2}

    fraction, retained = robust_sequence_component_fraction(
        directions, edges, bridge_counts
    )

    assert fraction == 1.0
    assert ("F1", "R2") not in retained
    assert ("F2", "R1") in retained


def test_loftr_trigger_is_allowlisted_and_requires_only_ghost_to_fail() -> None:
    edge = ("P1100110_005", "P1110111")
    pure_checks = {"G5.1": True, "G5.7": True, "G6.1": False, "G6.3": True}
    blocking_edges = {
        ("P1090109_002", "P1110111"),
        ("P1100110_005", "P1110111"),
    }

    contract = loftr_trigger_contract(
        edge,
        pure_checks,
        ghost_check_id="G6.1",
        blocking_edges=blocking_edges,
    )

    assert tuple(sorted(edge)) in LOFTR_SEQUENCE_ALLOWLIST
    assert contract["authorized"] is True
    assert contract["database_copy_required"] is True
    assert contract["full_glomap_rerun"] is False
    assert contract["pgo"] is False
    assert contract["all_blocking_edges_allowlisted"] is True

    assert loftr_trigger_contract(
        ("P1120112", "P1110111"),
        pure_checks,
        ghost_check_id="G6.1",
        blocking_edges=blocking_edges,
    )["authorized"] is False
    assert loftr_trigger_contract(
        edge,
        {**pure_checks, "G5.7": False},
        ghost_check_id="G6.1",
        blocking_edges=blocking_edges,
    )["authorized"] is False


def test_loftr_trigger_rejects_nonallowlisted_blocking_ghost_edges() -> None:
    contract = loftr_trigger_contract(
        ("P1100110_005", "P1110111"),
        {"G5.7": True, "G6.1": False, "G6.2": True, "G6.3": True},
        ghost_check_id="G6.1",
        blocking_edges={
            ("P1090109_002", "P1110111"),
            ("P1100110_005", "P1110111"),
            ("P1100110_005", "P1140114"),
        },
    )

    assert contract["authorized"] is False
    assert contract["all_blocking_edges_allowlisted"] is False
    assert contract["nonallowlisted_blocking_edges"] == [
        ["P1100110_005", "P1140114"]
    ]


def test_loftr_database_isolation_rejects_source_or_nonallowlisted_drift() -> None:
    before = {
        "P1090109_002|P1110111": "a",
        "P1100110_005|P1110111": "b",
        "P1120112|P1140114": "c",
    }
    after = {**before, "P1100110_005|P1110111": "loftr"}

    evidence = loftr_database_isolation(
        source_path=Path("/maps/source.db"),
        branch_path=Path("/maps/loftr_branch.db"),
        source_before_sha256="a" * 64,
        source_after_sha256="a" * 64,
        branch_base_sha256="a" * 64,
        pair_digests_before=before,
        pair_digests_after=after,
    )

    assert evidence["ok"] is True
    assert evidence["changed_sequence_pairs"] == ["P1100110_005|P1110111"]

    after["P1120112|P1140114"] = "forbidden"
    assert loftr_database_isolation(
        source_path=Path("/maps/source.db"),
        branch_path=Path("/maps/loftr_branch.db"),
        source_before_sha256="a" * 64,
        source_after_sha256="a" * 64,
        branch_base_sha256="a" * 64,
        pair_digests_before=before,
        pair_digests_after=after,
    )["ok"] is False


def test_loftr_promotion_requires_improvement_and_no_loftr_only_two_view() -> None:
    baseline = {
        "failed_edge_score": 0.050,
        "unaffected_score": 0.020,
        "registration_fraction": 0.97,
    }
    candidate = {
        "failed_edge_score": 0.045,
        "unaffected_score": 0.022,
        "registration_fraction": 0.96,
        "loftr_only_two_view_points": 0,
    }

    decision = evaluate_loftr_promotion(baseline, candidate)

    assert decision["status"] == "PASS"
    assert all(decision["checks"].values())

    candidate["loftr_only_two_view_points"] = 1
    assert evaluate_loftr_promotion(baseline, candidate)["status"] == "FAIL"


def test_loftr_branch_gate_combines_trigger_isolation_and_promotion() -> None:
    trigger = loftr_trigger_contract(
        ("P1090109_002", "P1110111"),
        {"G5.1": True, "G5.7": True, "G6.1": False},
        ghost_check_id="G6.1",
        blocking_edges={
            ("P1090109_002", "P1110111"),
            ("P1100110_005", "P1110111"),
        },
    )
    isolation = loftr_database_isolation(
        source_path=Path("/maps/source.db"),
        branch_path=Path("/maps/branch.db"),
        source_before_sha256="a" * 64,
        source_after_sha256="a" * 64,
        branch_base_sha256="a" * 64,
        pair_digests_before={"P1090109_002|P1110111": "old"},
        pair_digests_after={"P1090109_002|P1110111": "new"},
    )
    promotion = evaluate_loftr_promotion(
        {
            "failed_edge_score": 0.04,
            "unaffected_score": 0.02,
            "registration_fraction": 0.96,
        },
        {
            "failed_edge_score": 0.036,
            "unaffected_score": 0.022,
            "registration_fraction": 0.95,
            "loftr_only_two_view_count": 0,
        },
    )

    gate = targeted_loftr_branch_gate(trigger, isolation, promotion)

    assert gate["status"] == "PASS"
    assert all(gate["checks"].values())


def test_loftr_database_isolation_rejects_missing_hash_evidence() -> None:
    evidence = loftr_database_isolation(
        source_path=Path("/maps/source.db"),
        branch_path=Path("/maps/branch.db"),
        source_before_sha256="",
        source_after_sha256="",
        branch_base_sha256="",
        pair_digests_before={},
        pair_digests_after={},
    )

    assert evidence["ok"] is False
    assert evidence["checks"]["hash_evidence_valid"] is False


def test_loftr_promotion_rejects_out_of_range_metrics() -> None:
    baseline = {
        "failed_edge_score": 0.04,
        "unaffected_score": 0.02,
        "registration_fraction": 1.2,
    }
    candidate = {
        "failed_edge_score": -1.0,
        "unaffected_score": 0.01,
        "registration_fraction": 1.1,
        "loftr_only_two_view_count": 0,
    }

    with pytest.raises(ValueError, match="finite"):
        evaluate_loftr_promotion(baseline, candidate)
