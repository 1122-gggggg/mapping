from __future__ import annotations

import sys
import sqlite3
from pathlib import Path

import numpy as np


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from audit_independent_sim3 import (  # noqa: E402
    _filter_database,
    accepted_pair_records_for_edges,
    decode_image_pair_id,
    image_pair_id,
    resolve_quarantined_edges,
    robust_sim3,
)


def test_independent_sim3_uses_natural_and_conditional_dg_evidence() -> None:
    records = accepted_pair_records_for_edges(
        ["F/a.jpg", "F/b.jpg", "R/x.jpg", "R/y.jpg"],
        [(0, 2), (1, 3)],
        [0.92, 0.93],
        threshold=0.8,
        directions={"F": "fwd", "R": "rev"},
        conditional_pairs={("F/b.jpg", "R/y.jpg")},
        trusted_edges={("F", "R")},
    )

    assert [row["source"] for row in records[("F", "R")]] == [
        "natural",
        "conditional",
    ]
    assert [
        (row["left"], row["right"]) for row in records[("F", "R")]
    ] == [("F/a.jpg", "R/x.jpg"), ("F/b.jpg", "R/y.jpg")]


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


def test_filter_database_canonicalizes_camera_rig_and_preserves_local_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    output = tmp_path / "filtered.db"
    connection = sqlite3.connect(source)
    connection.executescript(
        """
        CREATE TABLE cameras (
            camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            model INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            params BLOB,
            prior_focal_length INTEGER NOT NULL
        );
        CREATE TABLE images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name TEXT NOT NULL UNIQUE,
            camera_id INTEGER NOT NULL,
            FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
        );
        CREATE TABLE rigs (
            rig_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            ref_sensor_id INTEGER NOT NULL,
            ref_sensor_type INTEGER NOT NULL
        );
        CREATE UNIQUE INDEX rig_ref_sensor_assignment
            ON rigs(ref_sensor_id, ref_sensor_type);
        CREATE TABLE frames (
            frame_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            rig_id INTEGER NOT NULL,
            FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE
        );
        CREATE TABLE frame_data (
            frame_id INTEGER NOT NULL,
            data_id INTEGER NOT NULL,
            sensor_id INTEGER NOT NULL,
            sensor_type INTEGER NOT NULL,
            FOREIGN KEY(frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE
        );
        CREATE UNIQUE INDEX frame_sensor_assignment
            ON frame_data(data_id, sensor_type);
        CREATE TABLE keypoints (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        );
        CREATE TABLE descriptors (
            image_id INTEGER PRIMARY KEY NOT NULL,
            type INTEGER NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        );
        CREATE TABLE matches (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB
        );
        CREATE TABLE two_view_geometries (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            config INTEGER NOT NULL,
            F BLOB,
            E BLOB,
            H BLOB,
            qvec BLOB,
            tvec BLOB
        );
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
    stale_params = np.asarray([1000.0, 1000.0, 900.0, 500.0]).tobytes()
    connection.executemany(
        "INSERT INTO cameras VALUES (?, 1, 1920, 1080, ?, 0)",
        ((camera_id, stale_params) for camera_id in range(1, 241)),
    )
    connection.executemany(
        "INSERT INTO rigs VALUES (?, ?, 0)",
        ((rig_id, rig_id) for rig_id in range(1, 241)),
    )
    images = [
        (1, "S01/a.jpg", 101),
        (3, "S01/b.jpg", 103),
        (5, "S01/c.jpg", 105),
        (2, "S02/x.jpg", 102),
    ]
    connection.executemany("INSERT INTO images VALUES (?, ?, ?)", images)
    connection.executemany(
        "INSERT INTO keypoints VALUES (?, 2, 2, ?)",
        ((row[0], f"keypoints-{row[0]}".encode()) for row in images),
    )
    connection.executemany(
        "INSERT INTO descriptors VALUES (?, 0, 2, 4, ?)",
        ((row[0], f"descriptors-{row[0]}".encode()) for row in images),
    )
    connection.executemany(
        "INSERT INTO frames VALUES (?, ?)", ((row[0], row[2]) for row in images)
    )
    connection.executemany(
        "INSERT INTO frame_data VALUES (?, ?, ?, 0)",
        ((row[0], row[0], row[2]) for row in images),
    )
    internal_pair = image_pair_id(1, 3)
    second_internal_pair = image_pair_id(3, 5)
    cross_pair = image_pair_id(1, 2)
    connection.executemany(
        "INSERT INTO matches VALUES (?, 2, 2, ?)",
        (
            (internal_pair, b"matches-internal-1"),
            (second_internal_pair, b"matches-internal-2"),
            (cross_pair, b"matches-cross"),
        ),
    )
    connection.executemany(
        "INSERT INTO two_view_geometries VALUES (?, 2, 2, ?, 3, ?, ?, ?, ?, ?)",
        (
            (
                internal_pair,
                b"geometry-internal-1",
                b"F1",
                b"E1",
                b"H1",
                b"q1",
                b"t1",
            ),
            (
                second_internal_pair,
                b"geometry-internal-2",
                b"F2",
                b"E2",
                b"H2",
                b"q2",
                b"t2",
            ),
            (
                cross_pair,
                b"geometry-cross",
                b"F3",
                b"E3",
                b"H3",
                b"q3",
                b"t3",
            ),
        ),
    )
    connection.commit()
    connection.close()

    stats = _filter_database(source, output, "S01")

    filtered = sqlite3.connect(output)
    camera_rows = filtered.execute(
        "SELECT camera_id, model, width, height, params, prior_focal_length FROM cameras"
    ).fetchall()
    assert len(camera_rows) == 1
    camera_id, model, width, height, params, prior_focal_length = camera_rows[0]
    assert (camera_id, model, width, height, prior_focal_length) == (1, 1, 1920, 1080, 1)
    assert np.array_equal(
        np.frombuffer(params, dtype=np.float64),
        [1396.8086675255472, 1396.8086675255472, 960.0, 540.0],
    )
    assert filtered.execute(
        "SELECT image_id, camera_id FROM images ORDER BY image_id"
    ).fetchall() == [(1, 1), (3, 1), (5, 1)]
    assert filtered.execute("SELECT * FROM rigs").fetchall() == [(1, 1, 0)]
    assert filtered.execute("SELECT * FROM frames ORDER BY frame_id").fetchall() == [
        (1, 1),
        (3, 1),
        (5, 1),
    ]
    assert filtered.execute(
        "SELECT * FROM frame_data ORDER BY frame_id"
    ).fetchall() == [(1, 1, 1, 0), (3, 3, 1, 0), (5, 5, 1, 0)]
    assert filtered.execute(
        "SELECT image_id, rows, cols, data FROM keypoints ORDER BY image_id"
    ).fetchall() == [
        (1, 2, 2, b"keypoints-1"),
        (3, 2, 2, b"keypoints-3"),
        (5, 2, 2, b"keypoints-5"),
    ]
    assert filtered.execute(
        "SELECT image_id, type, rows, cols, data FROM descriptors ORDER BY image_id"
    ).fetchall() == [
        (1, 0, 2, 4, b"descriptors-1"),
        (3, 0, 2, 4, b"descriptors-3"),
        (5, 0, 2, 4, b"descriptors-5"),
    ]
    assert filtered.execute("SELECT * FROM matches ORDER BY pair_id").fetchall() == [
        (internal_pair, 2, 2, b"matches-internal-1"),
        (second_internal_pair, 2, 2, b"matches-internal-2"),
    ]
    assert filtered.execute(
        "SELECT * FROM two_view_geometries ORDER BY pair_id"
    ).fetchall() == [
        (internal_pair, 2, 2, b"geometry-internal-1", 3, b"F1", b"E1", b"H1", b"q1", b"t1"),
        (second_internal_pair, 2, 2, b"geometry-internal-2", 3, b"F2", b"E2", b"H2", b"q2", b"t2"),
    ]
    filtered.execute("PRAGMA foreign_keys = ON")
    assert filtered.execute("PRAGMA foreign_key_check").fetchall() == []
    assert [
        row[1] for row in filtered.execute("PRAGMA table_info(pose_priors)")
    ] == ["image_id", "position", "coordinate_system", "position_covariance"]
    filtered.close()
    assert stats["images"] == 3
    assert stats["two_view_pairs"] == 2
    assert stats["cameras"] == 1
    assert stats["rigs"] == 1
    assert stats["frames"] == 3
    assert stats["frame_data"] == 3
    assert stats["legacy_pose_priors_schema"] is True


def test_failed_redundant_edge_is_allowed_only_after_complete_quarantine() -> None:
    evidence = {
        "P1090109_002|P1110111": {
            "status": "PASS",
            "estimates": [{"inliers": 30}, {"inliers": 31}],
        },
        "P1100110_005|P1110111": {
            "status": "PASS",
            "estimates": [{"inliers": 40}, {"inliers": 35}],
        },
        "P1120112|P1140114": {"status": "FAIL"},
    }

    passed, trusted = resolve_quarantined_edges(
        evidence, {"P1120112|P1140114": 0}
    )

    assert passed
    assert trusted == [
        "P1090109_002|P1110111",
        "P1100110_005|P1110111",
    ]
    assert evidence["P1120112|P1140114"]["status"] == "QUARANTINED"


def test_failed_edge_with_remaining_tracks_still_fails() -> None:
    evidence = {
        "P1090109_002|P1110111": {
            "status": "PASS",
            "estimates": [{"inliers": 30}, {"inliers": 30}],
        },
        "P1100110_005|P1110111": {
            "status": "PASS",
            "estimates": [{"inliers": 30}, {"inliers": 30}],
        },
        "P1120112|P1140114": {"status": "FAIL"},
    }

    passed, _ = resolve_quarantined_edges(
        evidence, {"P1120112|P1140114": 1}
    )

    assert not passed


def test_sim3_edge_with_one_cluster_below_30_inliers_is_not_trusted() -> None:
    evidence = {
        "P1090109_002|P1110111": {
            "status": "PASS",
            "estimates": [{"inliers": 30}, {"inliers": 29}],
        },
        "P1100110_005|P1110111": {
            "status": "PASS",
            "estimates": [{"inliers": 30}, {"inliers": 30}],
        },
    }

    passed, trusted = resolve_quarantined_edges(evidence, {})

    assert passed is False
    assert trusted == ["P1100110_005|P1110111"]


def test_sim3_edge_requires_exactly_two_independent_clusters() -> None:
    evidence = {
        "P1090109_002|P1110111": {
            "status": "PASS",
            "estimates": [
                {"inliers": 30},
                {"inliers": 30},
                {"inliers": 30},
            ],
        },
        "P1100110_005|P1110111": {
            "status": "PASS",
            "estimates": [{"inliers": 30}, {"inliers": 30}],
        },
    }

    passed, trusted = resolve_quarantined_edges(evidence, {})

    assert passed is False
    assert trusted == ["P1100110_005|P1110111"]
