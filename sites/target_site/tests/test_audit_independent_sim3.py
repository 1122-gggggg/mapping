from __future__ import annotations

import sys
import sqlite3
from pathlib import Path

import numpy as np


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from audit_independent_sim3 import (  # noqa: E402
    _filter_database,
    decode_image_pair_id,
    image_pair_id,
    resolve_quarantined_edges,
    robust_sim3,
)


def test_colmap_pair_id_round_trip_is_order_independent() -> None:
    pair_id = image_pair_id(123, 42)

    assert pair_id == image_pair_id(42, 123)
    assert decode_image_pair_id(pair_id) == (42, 123)


def test_robust_sim3_rejects_large_3d_outliers() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(80, 3))
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    target = 1.7 * (source @ rotation.T) + np.asarray([2.0, -3.0, 0.5])
    target[:20] = rng.normal(loc=100.0, scale=20.0, size=(20, 3))

    result = robust_sim3(source, target, max_error=1e-5, iterations=500)

    assert result["inliers"] == 60
    assert np.isclose(result["scale"], 1.7)
    assert np.allclose(result["rotation"], rotation)
    assert np.allclose(result["translation"], [2.0, -3.0, 0.5])


def test_filter_database_removes_every_cross_sequence_pair(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    output = tmp_path / "filtered.db"
    connection = sqlite3.connect(source)
    connection.executescript(
        """
        CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY);
        CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY);
        CREATE TABLE frame_data (frame_id INTEGER, data_id INTEGER);
        CREATE TABLE frames (frame_id INTEGER PRIMARY KEY);
        CREATE TABLE matches (pair_id INTEGER PRIMARY KEY);
        CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY);
        CREATE TABLE pose_priors (
            pose_prior_id INTEGER PRIMARY KEY,
            corr_data_id INTEGER,
            corr_sensor_id INTEGER,
            corr_sensor_type INTEGER,
            position BLOB,
            position_covariance BLOB,
            gravity BLOB,
            coordinate_system INTEGER
        );
        """
    )
    images = [(1, "S01/a.jpg"), (3, "S01/b.jpg"), (5, "S01/c.jpg"), (2, "S02/x.jpg")]
    connection.executemany("INSERT INTO images VALUES (?, ?)", images)
    connection.executemany("INSERT INTO keypoints VALUES (?)", ((row[0],) for row in images))
    connection.executemany("INSERT INTO descriptors VALUES (?)", ((row[0],) for row in images))
    connection.executemany("INSERT INTO frames VALUES (?)", ((row[0],) for row in images))
    connection.executemany(
        "INSERT INTO frame_data VALUES (?, ?)", ((row[0], row[0]) for row in images)
    )
    internal_pair = image_pair_id(1, 3)
    cross_pair = image_pair_id(1, 2)
    connection.executemany(
        "INSERT INTO matches VALUES (?)", ((internal_pair,), (cross_pair,))
    )
    connection.executemany(
        "INSERT INTO two_view_geometries VALUES (?)",
        ((internal_pair,), (cross_pair,)),
    )
    connection.commit()
    connection.close()

    stats = _filter_database(source, output, "S01")

    filtered = sqlite3.connect(output)
    assert filtered.execute("SELECT image_id FROM images ORDER BY image_id").fetchall() == [
        (1,),
        (3,),
        (5,),
    ]
    assert filtered.execute("SELECT pair_id FROM matches").fetchall() == [
        (internal_pair,)
    ]
    assert filtered.execute("SELECT pair_id FROM two_view_geometries").fetchall() == [
        (internal_pair,)
    ]
    assert [
        row[1] for row in filtered.execute("PRAGMA table_info(pose_priors)")
    ] == [
        "pose_prior_id",
        "corr_data_id",
        "corr_sensor_id",
        "corr_sensor_type",
        "position",
        "position_covariance",
        "gravity",
        "coordinate_system",
    ]
    filtered.close()
    assert stats == {
        "images": 3,
        "two_view_pairs": 1,
        "pose_priors_columns": [
            "pose_prior_id",
            "corr_data_id",
            "corr_sensor_id",
            "corr_sensor_type",
            "position",
            "position_covariance",
            "gravity",
            "coordinate_system",
        ],
    }


def test_failed_redundant_edge_is_allowed_only_after_complete_quarantine() -> None:
    evidence = {
        "A|X": {"status": "PASS"},
        "B|X": {"status": "PASS"},
        "C|X": {"status": "FAIL"},
    }

    passed, trusted = resolve_quarantined_edges(evidence, {"C|X": 0})

    assert passed
    assert trusted == ["A|X", "B|X"]
    assert evidence["C|X"]["status"] == "QUARANTINED"


def test_failed_edge_with_remaining_tracks_still_fails() -> None:
    evidence = {
        "A|X": {"status": "PASS"},
        "B|X": {"status": "PASS"},
        "C|X": {"status": "FAIL"},
    }

    passed, _ = resolve_quarantined_edges(evidence, {"C|X": 1})

    assert not passed
