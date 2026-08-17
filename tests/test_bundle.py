import json
from pathlib import Path

import pytest

from update_map.bundle import CandidateBundleManager
from update_map.io.hashing import create_map_snapshot


def _candidate(path: Path, snapshot: dict) -> None:
    path.mkdir()
    (path / "references.json").write_text("[]", encoding="utf-8")
    (path / "manifest.json").write_text(
        json.dumps({"base_map_snapshot": snapshot}), encoding="utf-8"
    )


def test_candidate_bundle_promotion_and_rollback(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "points3D.bin").write_bytes(b"immutable")
    snapshot = create_map_snapshot(base)

    candidate1 = tmp_path / "candidate1"
    candidate2 = tmp_path / "candidate2"
    _candidate(candidate1, snapshot)
    _candidate(candidate2, snapshot)

    manager = CandidateBundleManager(tmp_path / "registry", base)
    manager.stage(candidate1, "v1")
    manager.stage(candidate2, "v2")

    with pytest.raises(ValueError):
        manager.promote("v1", {"passed": False})

    first = manager.promote("v1", {"passed": True})
    second = manager.promote("v2", {"passed": True})
    rolled_back = manager.rollback()

    assert first.version == "v1"
    assert second.version == "v2"
    assert second.previous_version == "v1"
    assert rolled_back.version == "v1"
    assert manager.active().version == "v1"


def test_candidate_rejects_mutated_base_map(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    map_file = base / "images.bin"
    map_file.write_bytes(b"before")
    snapshot = create_map_snapshot(base)

    candidate = tmp_path / "candidate"
    _candidate(candidate, snapshot)
    manager = CandidateBundleManager(tmp_path / "registry", base)
    manager.stage(candidate, "v1")
    map_file.write_bytes(b"after")

    with pytest.raises(ValueError, match="Base map no longer matches"):
        manager.promote("v1", {"passed": True})


def test_candidate_bundle_rejects_embedded_reconstruction(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "points3D.bin").write_bytes(b"immutable")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "points3D.bin").write_bytes(b"old-only-geometry")
    manager = CandidateBundleManager(tmp_path / "registry", base)

    with pytest.raises(ValueError, match="may not contain a reconstruction"):
        manager.stage(candidate, "unsafe")
