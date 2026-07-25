from __future__ import annotations

import builtins
import json
import math
import sqlite3
import struct
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import pytest
import yaml


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

import run_gluemap_memory_safe as launcher  # noqa: E402


def _write_sift_tables(
    database: Path,
    records: dict[int, tuple[int, int]],
) -> dict[int, tuple[bytes, bytes]]:
    original_blobs: dict[int, tuple[bytes, bytes]] = {}
    with sqlite3.connect(database) as connection:
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
                image_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB NOT NULL
            );
            CREATE TABLE descriptors(
                image_id INTEGER PRIMARY KEY,
                type INTEGER NOT NULL,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB NOT NULL
            );
            """
        )
        connection.execute("INSERT INTO cameras VALUES (1, 1920, 1080)")
        for image_id, (keypoint_rows, descriptor_rows) in records.items():
            connection.execute(
                "INSERT INTO images VALUES (?, ?, 1)",
                (image_id, f"S00/{image_id:06d}.jpg"),
            )
            keypoint_values = []
            for index in range(keypoint_rows):
                keypoint_values.extend(
                    (20.0 + index % 100, 30.0 + index % 100, index + 1.0, 0.0)
                )
            keypoint_blob = struct.pack(f"<{len(keypoint_values)}f", *keypoint_values)
            descriptor_blob = b"".join(
                bytes([(index % 251) + 1]) * 128
                for index in range(descriptor_rows)
            )
            connection.execute(
                "INSERT INTO keypoints VALUES (?, ?, 4, ?)",
                (image_id, keypoint_rows, keypoint_blob),
            )
            connection.execute(
                "INSERT INTO descriptors VALUES (?, 0, ?, 128, ?)",
                (image_id, descriptor_rows, descriptor_blob),
            )
            original_blobs[image_id] = (keypoint_blob, descriptor_blob)
    return original_blobs


def _replace_keypoints(
    database: Path,
    image_id: int,
    rows: list[tuple[float, ...]],
) -> None:
    cols = len(rows[0])
    blob = struct.pack(
        f"<{len(rows) * cols}f", *(value for row in rows for value in row)
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE keypoints SET rows = ?, cols = ?, data = ? WHERE image_id = ?",
            (len(rows), cols, blob, image_id),
        )


def _keypoint_rows(database: Path, image_id: int) -> list[tuple[float, ...]]:
    with sqlite3.connect(database) as connection:
        rows, cols, data = connection.execute(
            "SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (image_id,)
        ).fetchone()
    assert len(data) == rows * cols * 4
    return list(struct.iter_unpack(f"<{cols}f", data))


def _descriptor_row_ids(database: Path, image_id: int) -> list[int]:
    with sqlite3.connect(database) as connection:
        rows, cols, data = connection.execute(
            "SELECT rows, cols, data FROM descriptors WHERE image_id = ?", (image_id,)
        ).fetchone()
    assert cols == 128
    return [data[index * cols] for index in range(rows)]


def test_launcher_fails_closed_on_nonempty_write_or_temp_paths(tmp_path: Path) -> None:
    for key in ("write_path", "temp_path"):
        occupied = tmp_path / key
        occupied.mkdir()
        (occupied / "stale.bin").write_bytes(b"stale")
        config = {
            "write_path": str(tmp_path / "write"),
            "temp_path": str(tmp_path / "temp"),
        }
        config[key] = str(occupied)

        try:
            launcher.assert_clean_gluemap_paths(config)
        except RuntimeError as exc:
            assert key in str(exc)
            assert "non-empty" in str(exc)
        else:
            raise AssertionError(f"occupied {key} was accepted")


def test_launcher_allows_absent_or_empty_write_and_temp_paths(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    evidence = launcher.assert_clean_gluemap_paths(
        {"write_path": str(empty), "temp_path": str(tmp_path / "absent")}
    )

    assert evidence["write_path"]["state"] == "EMPTY"
    assert evidence["temp_path"]["state"] == "ABSENT"


def test_sift_hard_row_cap_is_applied_after_extraction_returns(
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}
    database = tmp_path / "database.db"
    marker_log = tmp_path / "resource_guard.log"

    def extract_features(*args, **kwargs):
        seen["args"] = args
        seen["options"] = kwargs["extraction_options"]
        seen["original_returned"] = False
        _write_sift_tables(database, {1: (2049, 2049)})
        seen["original_returned"] = True
        return "result"

    fake_pycolmap = SimpleNamespace(extract_features=extract_features)
    original = launcher.install_sift_feature_cap(
        fake_pycolmap,
        2048,
        max_num_orientations=1,
        runtime_marker_path=marker_log,
    )
    options = SimpleNamespace(
        sift=SimpleNamespace(max_num_features=8192, max_num_orientations=2)
    )

    result = fake_pycolmap.extract_features(
        database, "images", extraction_options=options
    )

    assert result == "result"
    assert original is extract_features
    assert seen["original_returned"] is True
    assert seen["options"].sift.max_num_features == 2048
    assert seen["options"].sift.max_num_orientations == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT rows FROM keypoints").fetchone() == (2048,)
        assert connection.execute("SELECT rows FROM descriptors").fetchone() == (
            2048,
        )
    marker_lines = marker_log.read_text(encoding="utf-8").splitlines()
    assert len(marker_lines) == 1
    assert marker_lines[0].startswith(launcher.SIFT_RUNTIME_MARKER_PREFIX)
    marker = json.loads(marker_lines[0].removeprefix(launcher.SIFT_RUNTIME_MARKER_PREFIX))
    assert marker["status"] == "PASS"
    assert marker["database_path"] == str(database.resolve())
    assert marker["max_rows_after"] == 2048
    assert marker["removed_invalid_rows"] == 0
    assert len(marker["sift_tables_sha256"]) == 64


def test_sift_database_clamp_accepts_real_descriptor_schema_and_preserves_type_zero(
    tmp_path: Path,
) -> None:
    database = tmp_path / "database.db"
    _write_sift_tables(database, {1: (3, 3)})

    launcher.clamp_sift_database_rows(database, max_rows=2)

    with sqlite3.connect(database) as connection:
        schema = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(descriptors)")
        )
        descriptor_type, rows = connection.execute(
            "SELECT type, rows FROM descriptors WHERE image_id=1"
        ).fetchone()
    assert schema == ("image_id", "type", "rows", "cols", "data")
    assert descriptor_type == 0
    assert rows == 2


@pytest.mark.parametrize("invalid_type", [1, "SIFT"])
def test_sift_database_clamp_rejects_unsupported_descriptor_type(
    tmp_path: Path,
    invalid_type: object,
) -> None:
    database = tmp_path / "database.db"
    _write_sift_tables(database, {1: (1, 1)})
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE descriptors SET type=? WHERE image_id=1", (invalid_type,)
        )

    with pytest.raises(RuntimeError, match="descriptor type must be SQLite integer 0"):
        launcher.clamp_sift_database_rows(database, max_rows=2)


def test_sift_database_clamp_rejects_missing_descriptor_type_column(
    tmp_path: Path,
) -> None:
    database = tmp_path / "database.db"
    _write_sift_tables(database, {1: (1, 1)})
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            ALTER TABLE descriptors RENAME TO descriptors_original;
            CREATE TABLE descriptors(
                image_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB NOT NULL
            );
            INSERT INTO descriptors(image_id, rows, cols, data)
            SELECT image_id, rows, cols, data FROM descriptors_original;
            DROP TABLE descriptors_original;
            """
        )

    with pytest.raises(RuntimeError, match="descriptors schema"):
        launcher.clamp_sift_database_rows(database, max_rows=2)


def test_sift_tables_digest_covers_descriptor_type() -> None:
    images = {1: ("S00/000001.jpg", 1920, 1080)}
    keypoints = {1: (1, 4, struct.pack("<4f", 10.0, 20.0, 2.0, 0.0))}
    descriptor_blob = b"\x01" * 128

    type_zero = launcher.sift_tables_sha256(
        images, keypoints, {1: (0, 1, 128, descriptor_blob)}
    )
    type_one = launcher.sift_tables_sha256(
        images, keypoints, {1: (1, 1, 128, descriptor_blob)}
    )

    assert type_zero != type_one


def test_sift_database_rows_are_deterministically_clamped_as_paired_blobs(
    tmp_path: Path,
) -> None:
    database = tmp_path / "database.db"
    original_blobs = _write_sift_tables(database, {1: (3, 3), 2: (1, 1)})

    result = launcher.clamp_sift_database_rows(database, max_rows=2)

    with sqlite3.connect(database) as connection:
        keypoints = connection.execute(
            "SELECT image_id, rows, cols, data FROM keypoints ORDER BY image_id"
        ).fetchall()
        descriptors = connection.execute(
            "SELECT image_id, rows, cols, data FROM descriptors ORDER BY image_id"
        ).fetchall()

    assert result["image_count"] == 2
    assert result["clamped_image_count"] == 1
    assert result["removed_invalid_rows"] == 0
    assert result["max_rows_before"] == 3
    assert result["max_rows_after"] == 2
    assert result["immutable_verified"] is True
    assert result["wal_size_bytes"] == 0
    assert [(row[0], row[1], row[2], len(row[3])) for row in keypoints] == [
        (1, 2, 4, 2 * 4 * 4),
        (2, 1, 4, 1 * 4 * 4),
    ]
    assert [(row[0], row[1], row[2], len(row[3])) for row in descriptors] == [
        (1, 2, 128, 2 * 128),
        (2, 1, 128, 1 * 128),
    ]
    assert keypoints[0][3] == original_blobs[1][0][4 * 4 :]
    assert descriptors[0][3] == original_blobs[1][1][128:]
    assert keypoints[1][3] == original_blobs[2][0]
    assert descriptors[1][3] == original_blobs[2][1]
    assert {row[0] for row in keypoints} == {row[0] for row in descriptors}


def test_sift_database_clamp_ranks_affine_keypoints_by_derived_scale(
    tmp_path: Path,
) -> None:
    database = tmp_path / "database.db"
    _write_sift_tables(database, {1: (3, 3)})
    _replace_keypoints(
        database,
        1,
        [
            (10.0, 10.0, 3.0, 0.0, 0.0, 3.0),  # scale 3
            (11.0, 11.0, 1.0, 0.0, 0.0, 9.0),  # scale 5
            (12.0, 12.0, 4.0, 0.0, 0.0, 4.0),  # scale 4
        ],
    )

    launcher.clamp_sift_database_rows(database, max_rows=2)

    assert _keypoint_rows(database, 1) == [
        (11.0, 11.0, 1.0, 0.0, 0.0, 9.0),
        (12.0, 12.0, 4.0, 0.0, 0.0, 4.0),
    ]
    assert _descriptor_row_ids(database, 1) == [2, 3]


def test_sift_database_clamp_removes_invalid_rows_but_keeps_finite_edge_rows(
    tmp_path: Path,
) -> None:
    database = tmp_path / "database.db"
    _write_sift_tables(database, {1: (6, 6)})
    _replace_keypoints(
        database,
        1,
        [
            (1961.0, 1085.0, 2.0, 0.0, 0.0, 2.0),
            (20.0, 20.0, math.nan, 0.0, 0.0, 2.0),
            (30.0, 30.0, 0.0, 0.0, 0.0, 0.0),
            (7681.0, 40.0, 2.0, 0.0, 0.0, 2.0),
            (50.0, 50.0, 5.0, 0.0, 0.0, 5.0),
            (60.0, 60.0, 3.0, math.inf, 0.0, 3.0),
        ],
    )

    result = launcher.clamp_sift_database_rows(database, max_rows=10)

    assert result["removed_invalid_rows"] == 4
    assert result["clamped_image_count"] == 0
    assert _keypoint_rows(database, 1) == [
        (1961.0, 1085.0, 2.0, 0.0, 0.0, 2.0),
        (50.0, 50.0, 5.0, 0.0, 0.0, 5.0),
    ]
    assert _descriptor_row_ids(database, 1) == [1, 5]


def test_sift_database_clamp_replays_v3_six_image_invalid_row_distribution(
    tmp_path: Path,
) -> None:
    database = tmp_path / "database.db"
    invalid_by_name = {
        "P1090109_002/000044.jpg": 2,
        "P1100110_005/000015.jpg": 39,
        "P1100110_005/000041.jpg": 1,
        "P1100110_005/000045.jpg": 1,
        "P1140114/000002.jpg": 39,
        "P1140114/000020.jpg": 1,
    }
    _write_sift_tables(
        database,
        {
            image_id: (invalid_count + 1, invalid_count + 1)
            for image_id, invalid_count in enumerate(invalid_by_name.values(), 1)
        },
    )
    for image_id, (name, invalid_count) in enumerate(invalid_by_name.items(), 1):
        with sqlite3.connect(database) as connection:
            connection.execute(
                "UPDATE images SET name=? WHERE image_id=?", (name, image_id)
            )
        invalid_rows = [
            (20.0 + index, 30.0, math.nan, 0.0, 0.0, 2.0)
            for index in range(invalid_count)
        ]
        _replace_keypoints(
            database,
            image_id,
            [*invalid_rows, (1961.0, 1085.0, 3.0, 0.0, 0.0, 3.0)],
        )

    result = launcher.clamp_sift_database_rows(database, max_rows=2048)

    assert result["image_count"] == 6
    assert result["removed_invalid_rows"] == 83
    assert result["modified_image_count"] == 6
    assert result["invalid_rows_after"] == 0
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT SUM(rows) FROM keypoints").fetchone() == (6,)
        assert connection.execute("SELECT SUM(rows) FROM descriptors").fetchone() == (
            6,
        )


def test_sift_database_clamp_rolls_back_when_descriptor_update_trigger_fails(
    tmp_path: Path,
) -> None:
    database = tmp_path / "database.db"
    original_blobs = _write_sift_tables(database, {1: (3, 3)})[1]
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_descriptor_update
            BEFORE UPDATE ON descriptors
            BEGIN
                SELECT RAISE(ABORT, 'descriptor update rejected');
            END
            """
        )

    with pytest.raises(RuntimeError, match="descriptor update rejected"):
        launcher.clamp_sift_database_rows(database, max_rows=2)

    with sqlite3.connect(database) as connection:
        keypoint = connection.execute(
            "SELECT rows, data FROM keypoints WHERE image_id=1"
        ).fetchone()
        descriptor = connection.execute(
            "SELECT rows, data FROM descriptors WHERE image_id=1"
        ).fetchone()
    assert keypoint == (3, original_blobs[0])
    assert descriptor == (3, original_blobs[1])


def test_sift_database_clamp_requires_each_update_to_affect_one_row(
    tmp_path: Path,
) -> None:
    database = tmp_path / "database.db"
    original_blobs = _write_sift_tables(database, {1: (3, 3)})[1]
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TRIGGER ignore_keypoint_update
            BEFORE UPDATE ON keypoints
            BEGIN
                SELECT RAISE(IGNORE);
            END
            """
        )

    with pytest.raises(RuntimeError, match="updated 0 rows"):
        launcher.clamp_sift_database_rows(database, max_rows=2)

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT rows, data FROM keypoints WHERE image_id=1"
        ).fetchone() == (3, original_blobs[0])
        assert connection.execute(
            "SELECT rows, data FROM descriptors WHERE image_id=1"
        ).fetchone() == (3, original_blobs[1])


def test_sift_database_clamp_rejects_mismatched_table_image_sets(
    tmp_path: Path,
) -> None:
    database = tmp_path / "database.db"
    _write_sift_tables(database, {1: (3, 3), 2: (1, 1)})
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM descriptors WHERE image_id = 2")

    try:
        launcher.clamp_sift_database_rows(database, max_rows=2)
    except RuntimeError as exc:
        assert "image sets" in str(exc)
    else:
        raise AssertionError("mismatched SIFT table image sets were accepted")

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT rows FROM keypoints WHERE image_id = 1"
        ).fetchone() == (3,)


def test_sift_database_clamp_rejects_inconsistent_paired_rows(
    tmp_path: Path,
) -> None:
    database = tmp_path / "database.db"
    _write_sift_tables(database, {1: (3, 2)})

    try:
        launcher.clamp_sift_database_rows(database, max_rows=2)
    except RuntimeError as exc:
        assert "inconsistent keypoint/descriptor rows" in str(exc)
    else:
        raise AssertionError("inconsistent paired SIFT rows were accepted")


def test_sift_database_clamp_rejects_blob_length_mismatch(tmp_path: Path) -> None:
    database = tmp_path / "database.db"
    _write_sift_tables(database, {1: (3, 3)})
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE keypoints SET data = ?", (b"truncated",))

    try:
        launcher.clamp_sift_database_rows(database, max_rows=2)
    except RuntimeError as exc:
        assert "BLOB length" in str(exc)
        assert "rows=3, cols=4" in str(exc)
    else:
        raise AssertionError("malformed SIFT BLOB length was accepted")


def test_sift_database_clamp_rejects_empty_or_incomplete_image_schema(
    tmp_path: Path,
) -> None:
    database = tmp_path / "database.db"
    _write_sift_tables(database, {1: (1, 1)})
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE images RENAME TO images_original")
        connection.execute("CREATE TABLE images(image_id INTEGER, name TEXT)")

    with pytest.raises(RuntimeError, match="images schema"):
        launcher.clamp_sift_database_rows(database, max_rows=2)


def test_sift_database_clamp_checkpoints_wal_and_verifies_immutable_reopen(
    tmp_path: Path,
) -> None:
    database = tmp_path / "database.db"
    _write_sift_tables(database, {1: (3, 3)})
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("UPDATE images SET name=name WHERE image_id=1")

    result = launcher.clamp_sift_database_rows(database, max_rows=2)

    wal_path = Path(f"{database}-wal")
    assert result["immutable_verified"] is True
    assert result["wal_size_bytes"] == 0
    assert not wal_path.exists() or wal_path.stat().st_size == 0


def test_failed_verified_clamp_blocks_matching_and_writes_no_pass_marker(
    tmp_path: Path,
) -> None:
    database = tmp_path / "database.db"
    marker_log = tmp_path / "resource_guard.log"
    events: list[str] = []

    def extract_features(*_args, **_kwargs):
        events.append("extraction")
        _write_sift_tables(database, {1: (2, 2)})
        with sqlite3.connect(database) as connection:
            connection.execute("UPDATE descriptors SET cols=127 WHERE image_id=1")

    fake_pycolmap = SimpleNamespace(extract_features=extract_features)
    launcher.install_sift_feature_cap(
        fake_pycolmap,
        2048,
        max_num_orientations=1,
        runtime_marker_path=marker_log,
    )

    def prepare_then_match() -> None:
        options = SimpleNamespace(
            sift=SimpleNamespace(max_num_features=8192, max_num_orientations=2)
        )
        fake_pycolmap.extract_features(
            database, "images", extraction_options=options
        )
        events.append("matching")

    with pytest.raises(RuntimeError, match="descriptor cols"):
        prepare_then_match()

    assert events == ["extraction"]
    assert not marker_log.exists()


def test_sift_policy_fails_closed_when_orientation_limit_is_missing_or_not_one(
    tmp_path: Path,
) -> None:
    for suffix, value in (("missing", None), ("two", 2), ("bool", True)):
        config = tmp_path / f"{suffix}.yaml"
        payload = {"sift_max_num_features": 2048}
        if value is not None:
            payload["sift_max_num_orientations"] = value
        config.write_text(yaml.safe_dump(payload), encoding="utf-8")

        try:
            launcher.sift_policy_from_config(["--config", str(config)])
        except ValueError as exc:
            assert "sift_max_num_orientations" in str(exc)
        else:
            raise AssertionError(f"unsafe SIFT orientation policy was accepted: {value}")


def test_sift_cap_rejects_nonpositive_values() -> None:
    fake_pycolmap = SimpleNamespace(extract_features=lambda: None)

    try:
        launcher.install_sift_feature_cap(
            fake_pycolmap,
            0,
            max_num_orientations=1,
            runtime_marker_path=Path("/unused"),
        )
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("zero SIFT cap was accepted")


def test_ba_limits_override_gluemap_hardcoded_values() -> None:
    seen: dict[str, object] = {}

    def iterative_ba_options(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "options"

    module = SimpleNamespace(IterativeBAOptions=iterative_ba_options)
    original = launcher.install_ba_limits(
        module, max_num_iterations=50, max_filter_iterations=2
    )

    result = module.IterativeBAOptions(
        max_ba_iterations=200,
        max_filter_iterations=3,
        normalized_reproj_threshold=0.01,
    )

    assert result == "options"
    assert original is iterative_ba_options
    assert seen["kwargs"] == {
        "max_ba_iterations": 50,
        "max_filter_iterations": 2,
        "normalized_reproj_threshold": 0.01,
    }


def test_optional_ba_limits_are_read_from_config(tmp_path: Path) -> None:
    config = tmp_path / "recovery.yaml"
    config.write_text(
        "sift_max_num_features: 2048\n"
        "ba_max_num_iterations: 50\n"
        "ba_max_filter_iterations: 2\n",
        encoding="utf-8",
    )

    assert launcher.ba_limits_from_config(["--config", str(config)]) == (50, 2)


def test_main_locks_then_preflights_before_importing_heavy_runtime(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    events: list[str] = []
    config_path = tmp_path / "gluemap_config.yaml"
    guard_log = tmp_path / "logs" / "resource_guard.log"
    config_path.write_text(
        yaml.safe_dump(
            {
                "sift_max_num_features": 2048,
                "sift_max_num_orientations": 1,
                "resource_lock_path": "/global/heavy.lock",
                "resource_guard_path": "/guard.py",
                "resource_guard_log_path": str(guard_log),
                "write_path": str(tmp_path / "write"),
                "temp_path": str(tmp_path / "temp"),
            }
        ),
        encoding="utf-8",
    )
    contract = {
        "run_dir": str(tmp_path),
        "lock": {"path": "/global/heavy.lock"},
        "guard": {"script": "/guard.py", "log_path": str(guard_log)},
    }

    @contextmanager
    def fake_lock(_path):
        events.append("lock-acquired")
        yield

    class FakeMonitor:
        def terminate(self):
            events.append("monitor-terminate")

        def wait(self, timeout):
            del timeout

    fake_pycolmap = ModuleType("pycolmap")
    fake_gluemap = ModuleType("gluemap")
    fake_cli = ModuleType("gluemap.cli")
    fake_cli.demo_main = lambda: events.append("demo")
    fake_gluemap.cli = fake_cli
    monkeypatch.setitem(sys.modules, "pycolmap", fake_pycolmap)
    monkeypatch.setitem(sys.modules, "gluemap", fake_gluemap)
    monkeypatch.setitem(sys.modules, "gluemap.cli", fake_cli)
    monkeypatch.setattr(launcher, "contract_from_config", lambda *_args: contract)
    monkeypatch.setattr(launcher, "exclusive_resource_lock", fake_lock)
    monkeypatch.setattr(
        launcher,
        "startup_preflight",
        lambda _path: events.append("preflight")
        or {
            "ok": True,
            "mem_available_gib": 24.0,
            "vram_free_gib": 24.0,
            "swap_free_gib": 6.0,
            "disk_free_gib": 100.0,
        },
    )
    monkeypatch.setattr(
        launcher,
        "install_sift_feature_cap",
        lambda *_args, **_kwargs: events.append("sift-install"),
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FakeMonitor(),
    )
    original_import = builtins.__import__

    def tracked_import(name, *args, **kwargs):
        if name == "pycolmap":
            events.append("pycolmap-import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", tracked_import)

    launcher.main(["--config", str(config_path)])

    assert events.index("lock-acquired") < events.index("preflight")
    assert events.index("preflight") < events.index("pycolmap-import")
    assert events.index("pycolmap-import") < events.index("demo")
    stdout = capsys.readouterr().out
    assert "SIFT row-cap policy armed" in stdout
    assert "max_num_features=2048" in stdout
    assert "max_num_orientations=1" in stdout
    guard_text = guard_log.read_text(encoding="utf-8")
    assert "startup_preflight=PASS" in guard_text
    assert "sift_row_cap=PASS" not in guard_text
    assert launcher.SIFT_RUNTIME_MARKER_PREFIX not in guard_text
