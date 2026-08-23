import sqlite3

from sfm_diagnosis.evidence import COLMAP_MAX_IMAGE_ID, decode_pair_id, load_colmap_database


def test_decode_pair_id_roundtrip():
    i, j = 12, 987
    pair_id = i * COLMAP_MAX_IMAGE_ID + j
    assert decode_pair_id(pair_id) == (i, j)


def test_load_colmap_database_counts_without_match_blob(tmp_path):
    db = tmp_path / "database.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute(
            "CREATE TABLE matches("
            "pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)"
        )
        conn.execute(
            "CREATE TABLE two_view_geometries("
            "pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB, config INTEGER)"
        )
        conn.execute("INSERT INTO images VALUES(1, 'a.jpg')")
        conn.execute("INSERT INTO images VALUES(2, 'b.jpg')")
        pair_id = 1 * COLMAP_MAX_IMAGE_ID + 2
        conn.execute("INSERT INTO matches VALUES(?, 120, 2, NULL)", (pair_id,))
        conn.execute(
            "INSERT INTO two_view_geometries VALUES(?, 60, 2, NULL, 2)",
            (pair_id,),
        )
        conn.commit()
    finally:
        conn.close()

    images, pairs = load_colmap_database(db)
    assert {row["image_name"] for row in images} == {"a.jpg", "b.jpg"}
    assert len(pairs) == 1
    assert pairs[0]["num_matches"] == 120
    assert pairs[0]["num_inliers"] == 60
    assert pairs[0]["inlier_ratio"] == 0.5
