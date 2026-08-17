from pathlib import Path

from update_map.io.hashing import create_map_snapshot, verify_map_snapshot


def test_base_map_snapshot_detects_mutation(tmp_path: Path) -> None:
    (tmp_path / "cameras.bin").write_bytes(b"camera")
    (tmp_path / "images.bin").write_bytes(b"images")
    snapshot = create_map_snapshot(tmp_path)
    assert verify_map_snapshot(tmp_path, snapshot)["ok"]
    (tmp_path / "images.bin").write_bytes(b"changed")
    report = verify_map_snapshot(tmp_path, snapshot)
    assert not report["ok"]
    assert report["changed"] == ["images.bin"]
