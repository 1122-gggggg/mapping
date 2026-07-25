from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from audit_glomap_candidate import (  # noqa: E402
    AuditFailure,
    _audit_report,
    _snapshot_report,
    audit_models,
    atomic_json,
    database_snapshot,
    sha256_file,
    validate_mapper_post_contract,
    validate_database_contract,
    validate_migration_receipt,
)


class _Pose:
    def __mul__(self, xyz: np.ndarray) -> np.ndarray:
        return np.asarray(xyz, dtype=np.float64) + np.asarray([0.0, 0.0, 5.0])


class _IndexOnlyPoint3DMap:
    """Minimal pycolmap-style map: membership and indexing, no ``get`` method."""

    def __init__(self, items: dict[int, SimpleNamespace]) -> None:
        self._items = items

    def __contains__(self, point_id: int) -> bool:
        return point_id in self._items

    def __getitem__(self, point_id: int) -> SimpleNamespace:
        return self._items[point_id]

    def values(self):  # type: ignore[no-untyped-def]
        return self._items.values()


def _camera(*, params: list[float] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        camera_id=1,
        model=SimpleNamespace(name="PINHOLE"),
        width=1920,
        height=1080,
        params=np.asarray(params or [1000.0, 1000.0, 960.0, 540.0]),
    )


def _reconstruction(
    *,
    registered: bool = True,
    camera_params: list[float] | None = None,
    point_xyz: list[float] | None = None,
) -> SimpleNamespace:
    point = SimpleNamespace(
        xyz=np.asarray(point_xyz or [0.0, 0.0, 0.0]),
        track=SimpleNamespace(elements=[SimpleNamespace(image_id=1, point2D_idx=0)]),
    )
    point2d = SimpleNamespace(
        xy=np.asarray([960.0, 540.0]),
        point3D_id=1,
        has_point3D=lambda: True,
    )
    image = SimpleNamespace(
        image_id=1,
        name="S01/frame.jpg",
        has_pose=registered,
        points2D=[point2d],
        cam_from_world=lambda: _Pose(),
        project_point=lambda xyz: np.asarray([960.0, 540.0]),
        projection_center=lambda: np.asarray([0.0, 0.0, -5.0]),
    )
    return SimpleNamespace(
        cameras={1: _camera(params=camera_params)},
        images={1: image},
        points3D={1: point},
        num_cameras=lambda: 1,
        num_images=lambda: 1,
        num_reg_images=lambda: int(registered),
        num_points3D=lambda: 1,
    )


def _database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE images (image_id INTEGER PRIMARY KEY);
        CREATE TABLE matches (pair_id INTEGER PRIMARY KEY);
        CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY);
        CREATE TABLE pose_priors (
            image_id INTEGER PRIMARY KEY NOT NULL,
            position BLOB,
            coordinate_system INTEGER NOT NULL,
            position_covariance BLOB
        );
        INSERT INTO images VALUES (1);
        INSERT INTO matches VALUES (3);
        INSERT INTO two_view_geometries VALUES (3);
        """
    )
    connection.commit()
    connection.close()


def test_audit_models_records_raw_geometry_and_seed_intrinsics() -> None:
    seed = _reconstruction()
    raw = _reconstruction()

    result = audit_models(
        seed=seed,
        raw=raw,
        final=None,
        expected_camera_count=1,
        minimum_registered=1,
        raw_intrinsics_tolerance=1e-6,
        final_intrinsics_tolerance=1e-6,
    )

    assert result["checks"] == {
        "raw_camera_count": True,
        "raw_camera_signatures_match_seed": True,
        "raw_distinct_camera_models_match_seed": True,
        "raw_registered_floor": True,
        "raw_intrinsics_match_seed": True,
        "raw_finite": True,
        "raw_observation_track_consistent": True,
    }
    assert result["raw"]["registered_images"] == 1
    assert result["raw"]["points3D"] == 1
    assert result["raw"]["observations"] == 1
    assert result["raw"]["track_elements"] == 1
    assert result["raw"]["camera_signature_count"] == 1
    assert result["raw"]["distinct_camera_model_count"] == 1
    assert result["raw"]["reprojection_error_px"]["max"] == 0.0


def test_audit_models_supports_index_only_pycolmap_point3d_map() -> None:
    """Real Point3DMap exposes ``in``/``[]`` but not ``dict.get``."""
    seed = _reconstruction()
    raw = _reconstruction()
    raw.points3D = _IndexOnlyPoint3DMap(raw.points3D)

    result = audit_models(
        seed=seed,
        raw=raw,
        final=None,
        expected_camera_count=1,
        minimum_registered=1,
        raw_intrinsics_tolerance=1e-6,
        final_intrinsics_tolerance=1e-6,
    )

    assert result["raw"]["observations"] == 1
    assert result["raw"]["observation_track_consistent"] is True


def test_audit_models_rejects_under_registered_and_nonfinite_candidate() -> None:
    with pytest.raises(AuditFailure, match="registered"):
        audit_models(
            seed=_reconstruction(),
            raw=_reconstruction(registered=False),
            final=None,
            expected_camera_count=1,
            minimum_registered=1,
            raw_intrinsics_tolerance=1e-6,
            final_intrinsics_tolerance=1e-6,
        )

    with pytest.raises(AuditFailure, match="non-finite"):
        audit_models(
            seed=_reconstruction(),
            raw=_reconstruction(point_xyz=[float("nan"), 0.0, 0.0]),
            final=None,
            expected_camera_count=1,
            minimum_registered=1,
            raw_intrinsics_tolerance=1e-6,
            final_intrinsics_tolerance=1e-6,
        )


def test_audit_models_rejects_unexpected_camera_count_and_intrinsics() -> None:
    with pytest.raises(AuditFailure, match="camera count"):
        audit_models(
            seed=_reconstruction(),
            raw=_reconstruction(),
            final=None,
            expected_camera_count=2,
            minimum_registered=1,
            raw_intrinsics_tolerance=1e-6,
            final_intrinsics_tolerance=1e-6,
        )

    with pytest.raises(AuditFailure, match="intrinsics"):
        audit_models(
            seed=_reconstruction(),
            raw=_reconstruction(camera_params=[1001.0, 1000.0, 960.0, 540.0]),
            final=None,
            expected_camera_count=1,
            minimum_registered=1,
            raw_intrinsics_tolerance=1e-6,
            final_intrinsics_tolerance=1e-6,
        )


def test_database_contract_records_pre_post_and_rejects_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    candidate = tmp_path / "candidate.db"
    _database(source)
    _database(candidate)
    source_pre = database_snapshot(source)
    candidate_pre = database_snapshot(candidate)

    good = validate_database_contract(
        source=source,
        candidate=candidate,
        source_pre=source_pre,
        candidate_pre=candidate_pre,
        expected_source_sha256=source_pre["sha256"],
        expected_counts=source_pre["counts"],
    )

    assert good["checks"] == {
        "source_pre_expected": True,
        "candidate_pre_matches_source": True,
        "source_unchanged": True,
        "candidate_unchanged": True,
        "source_post_expected": True,
    }
    assert good["source"]["pre"] == good["source"]["post"]
    assert good["candidate"]["pre"] == good["candidate"]["post"]

    connection = sqlite3.connect(candidate)
    connection.execute("INSERT INTO matches VALUES (5)")
    connection.commit()
    connection.close()

    with pytest.raises(AuditFailure, match="candidate database changed"):
        validate_database_contract(
            source=source,
            candidate=candidate,
            source_pre=source_pre,
            candidate_pre=candidate_pre,
            expected_source_sha256=source_pre["sha256"],
            expected_counts=source_pre["counts"],
        )


def test_database_snapshot_uses_immutable_read_without_creating_sidecars(
    tmp_path: Path,
) -> None:
    database = tmp_path / "candidate.db"
    _database(database)

    snapshot = database_snapshot(database)

    assert snapshot["sidecars"] == {
        "wal": {"exists": False, "size_bytes": 0, "sha256": None},
        "shm": {"exists": False, "size_bytes": 0, "sha256": None},
    }
    semantic_tool = snapshot["semantic_sha3sum_tool"]
    assert Path(semantic_tool["path"]).is_absolute()
    assert len(semantic_tool["sha256"]) == 64
    assert semantic_tool["version"]
    assert semantic_tool["argv"][-1] == ".sha3sum --sha3-256 --schema"
    assert "mode=ro&immutable=1" in semantic_tool["argv"][1]
    assert not database.with_name("candidate.db-wal").exists()
    assert not database.with_name("candidate.db-shm").exists()


def test_database_snapshot_rejects_nonempty_wal_before_reading_sqlite(
    tmp_path: Path,
) -> None:
    database = tmp_path / "candidate.db"
    _database(database)
    database.with_name("candidate.db-wal").write_bytes(b"not-empty")

    with pytest.raises(AuditFailure, match="non-empty SQLite WAL"):
        database_snapshot(database)


def test_database_snapshot_records_empty_wal_and_derived_shm_without_mutation(
    tmp_path: Path,
) -> None:
    database = tmp_path / "candidate.db"
    _database(database)
    database.with_name("candidate.db-wal").write_bytes(b"")
    database.with_name("candidate.db-shm").write_bytes(b"derived-shm")

    snapshot = database_snapshot(database)

    assert snapshot["sidecars"]["wal"]["size_bytes"] == 0
    assert snapshot["sidecars"]["shm"]["size_bytes"] == len(b"derived-shm")
    assert snapshot["counts"] == {
        "images": 1,
        "matches": 1,
        "two_view_geometries": 1,
    }


def test_snapshot_report_requires_fresh_candidate_to_have_no_sidecars(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    candidate = tmp_path / "candidate.db"
    _database(source)
    shutil.copyfile(source, candidate)
    args = SimpleNamespace(
        source_db=source,
        candidate_db=candidate,
        expected_source_sha256=database_snapshot(source)["sha256"],
        expected_counts=database_snapshot(source)["counts"],
    )

    report, passed = _snapshot_report(args)

    assert passed
    assert report["checks"]["candidate_pre_sidecars_absent"]

    candidate.with_name("candidate.db-wal").write_bytes(b"")
    report, passed = _snapshot_report(args)

    assert not passed
    assert not report["checks"]["candidate_pre_sidecars_absent"]


def test_atomic_json_never_overwrites_existing_audit(tmp_path: Path) -> None:
    out = tmp_path / "audit.json"

    atomic_json(out, {"status": "PASS"})

    assert out.read_text(encoding="utf-8") == '{\n  "status": "PASS"\n}\n'
    with pytest.raises(FileExistsError):
        atomic_json(out, {"status": "FAIL"})


def test_mapper_post_audit_allows_only_sqlite_change_counter_header_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    candidate = tmp_path / "candidate.db"
    _database(source)
    shutil.copyfile(source, candidate)
    source_pre = database_snapshot(source)
    candidate_pre = database_snapshot(candidate)

    with candidate.open("r+b") as handle:
        handle.seek(24)
        original = handle.read(4)
        handle.seek(24)
        handle.write(bytes([original[0] ^ 1]) + original[1:])

    report = validate_mapper_post_contract(
        source=source,
        candidate=candidate,
        source_pre=source_pre,
        candidate_pre=candidate_pre,
        expected_source_sha256=source_pre["sha256"],
        expected_counts=source_pre["counts"],
    )

    assert report["checks"]["candidate_raw_header_only"]
    assert report["checks"]["candidate_semantic_sha3_exact"]
    assert report["checks"]["candidate_core_sha3_exact"]
    assert report["checks"]["candidate_pose_priors_empty"]


def test_mapper_post_audit_rejects_nonheader_raw_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    candidate = tmp_path / "candidate.db"
    _database(source)
    shutil.copyfile(source, candidate)
    source_pre = database_snapshot(source)
    candidate_pre = database_snapshot(candidate)

    with candidate.open("r+b") as handle:
        handle.seek(60)
        original = handle.read(4)
        handle.seek(60)
        handle.write(bytes([original[0] ^ 1]) + original[1:])

    with pytest.raises(
        AuditFailure, match="raw database changed outside SQLite header"
    ):
        validate_mapper_post_contract(
            source=source,
            candidate=candidate,
            source_pre=source_pre,
            candidate_pre=candidate_pre,
            expected_source_sha256=source_pre["sha256"],
            expected_counts=source_pre["counts"],
        )


def test_mapper_post_audit_rejects_semantic_mutation_and_records_integrity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    candidate = tmp_path / "candidate.db"
    _database(source)
    shutil.copyfile(source, candidate)
    source_pre = database_snapshot(source)
    candidate_pre = database_snapshot(candidate)
    connection = sqlite3.connect(candidate)
    connection.execute("INSERT INTO matches VALUES (8)")
    connection.commit()
    connection.close()

    with pytest.raises(AuditFailure, match="semantic SHA3") as captured:
        validate_mapper_post_contract(
            source=source,
            candidate=candidate,
            source_pre=source_pre,
            candidate_pre=candidate_pre,
            expected_source_sha256=source_pre["sha256"],
            expected_counts=source_pre["counts"],
        )

    assert captured.value.report is not None
    assert captured.value.report["candidate"]["post"]["quick_check"] == "ok"
    assert captured.value.report["candidate"]["post"]["integrity_check"] == "ok"


def test_mapper_post_audit_records_nonempty_wal_before_reading_sqlite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    candidate = tmp_path / "candidate.db"
    _database(source)
    shutil.copyfile(source, candidate)
    source_pre = database_snapshot(source)
    candidate_pre = database_snapshot(candidate)
    candidate.with_name("candidate.db-wal").write_bytes(b"uncheckpointed")

    with pytest.raises(AuditFailure, match="non-empty SQLite WAL") as captured:
        validate_mapper_post_contract(
            source=source,
            candidate=candidate,
            source_pre=source_pre,
            candidate_pre=candidate_pre,
            expected_source_sha256=source_pre["sha256"],
            expected_counts=source_pre["counts"],
        )

    assert captured.value.report is not None
    assert captured.value.report["candidate"]["post"]["sidecars"]["wal"] == {
        "exists": True,
        "size_bytes": len(b"uncheckpointed"),
        "sha256": sha256_file(candidate.with_name("candidate.db-wal")),
    }


def test_mapper_post_cli_publishes_full_semantic_evidence_once(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    candidate = tmp_path / "candidate.db"
    receipt = tmp_path / "migration.json"
    out = tmp_path / "post-audit.json"
    _database(source)
    shutil.copyfile(source, candidate)
    source_pre, _ = _migration_receipt(receipt, source=source, candidate=candidate)
    script = TARGET_SITE / "tools" / "audit_glomap_candidate.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "mapper-post-db",
            "--source-db",
            str(source),
            "--candidate-db",
            str(candidate),
            "--expected-source-sha256",
            source_pre["sha256"],
            "--expected-counts",
            json.dumps(source_pre["counts"]),
            "--migration-receipt",
            str(receipt),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    evidence = json.loads(out.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS"
    assert (
        len(evidence["database"]["candidate"]["pre"]["semantic_sha3_256_schema"]) == 64
    )


def _migration_receipt(
    path: Path,
    *,
    source: Path,
    candidate: Path,
    status: str = "PASS",
) -> tuple[dict[str, object], dict[str, object]]:
    source_evidence = database_snapshot(source)
    candidate_evidence = database_snapshot(candidate)
    path.write_text(
        json.dumps(
            {
                "status": status,
                "source_sha256": source_evidence["sha256"],
                "clone_sha256_after": candidate_evidence["sha256"],
                "source_evidence": source_evidence,
                "clone_evidence_after": candidate_evidence,
            }
        ),
        encoding="utf-8",
    )
    return source_evidence, candidate_evidence


@pytest.mark.parametrize(
    ("status", "source_offset", "candidate_offset", "expected"),
    [
        ("FAIL", 0, 0, "status is not PASS"),
        ("PASS", 1, 0, "source path differs"),
        ("PASS", 0, 1, "candidate path differs"),
    ],
)
def test_migration_receipt_must_be_complete_and_match_cli_paths(
    tmp_path: Path,
    status: str,
    source_offset: int,
    candidate_offset: int,
    expected: str,
) -> None:
    source = tmp_path / "source.db"
    candidate = tmp_path / "candidate.db"
    receipt = tmp_path / "migration.json"
    _database(source)
    shutil.copyfile(source, candidate)
    source_evidence, _ = _migration_receipt(
        receipt, source=source, candidate=candidate, status=status
    )
    alternate_source = tmp_path / "alternate-source.db"
    alternate_candidate = tmp_path / "alternate-candidate.db"
    shutil.copyfile(source, alternate_source)
    shutil.copyfile(candidate, alternate_candidate)
    args = SimpleNamespace(
        migration_receipt=receipt,
        source_db=alternate_source if source_offset else source,
        candidate_db=alternate_candidate if candidate_offset else candidate,
        expected_source_sha256=source_evidence["sha256"],
        expected_counts=source_evidence["counts"],
    )

    with pytest.raises(AuditFailure, match=expected):
        validate_migration_receipt(args)


@pytest.mark.parametrize(
    ("sha", "counts", "expected"),
    [
        ("0" * 64, None, "source SHA differs from CLI"),
        (
            None,
            {"images": 2, "matches": 1, "two_view_geometries": 1},
            "source counts differ from CLI",
        ),
    ],
)
def test_migration_receipt_must_match_cli_contract(
    tmp_path: Path, sha: str | None, counts: dict[str, int] | None, expected: str
) -> None:
    source = tmp_path / "source.db"
    candidate = tmp_path / "candidate.db"
    receipt = tmp_path / "migration.json"
    _database(source)
    shutil.copyfile(source, candidate)
    source_evidence, _ = _migration_receipt(receipt, source=source, candidate=candidate)
    args = SimpleNamespace(
        migration_receipt=receipt,
        source_db=source,
        candidate_db=candidate,
        expected_source_sha256=sha or source_evidence["sha256"],
        expected_counts=counts or source_evidence["counts"],
    )

    with pytest.raises(AuditFailure, match=expected):
        validate_migration_receipt(args)


def test_audit_parser_rejects_migration_receipt_and_db_snapshot_together(
    tmp_path: Path,
) -> None:
    script = TARGET_SITE / "tools" / "audit_glomap_candidate.py"
    counts = json.dumps(
        {"images": 1, "matches": 1, "two_view_geometries": 1}, separators=(",", ":")
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "audit",
            "--source-db",
            str(tmp_path / "source.db"),
            "--candidate-db",
            str(tmp_path / "candidate.db"),
            "--expected-source-sha256",
            "0" * 64,
            "--expected-counts",
            counts,
            "--db-snapshot",
            str(tmp_path / "snapshot.json"),
            "--migration-receipt",
            str(tmp_path / "migration.json"),
            "--raw-model",
            str(tmp_path / "raw"),
            "--intrinsics-seed",
            str(tmp_path / "seed"),
            "--out",
            str(tmp_path / "audit.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr


def test_full_audit_accepts_header_only_candidate_change_via_migration_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.db"
    candidate = tmp_path / "candidate.db"
    receipt = tmp_path / "migration.json"
    _database(source)
    shutil.copyfile(source, candidate)
    source_evidence, _ = _migration_receipt(receipt, source=source, candidate=candidate)
    with candidate.open("r+b") as handle:
        handle.seek(92)
        original = handle.read(8)
        handle.seek(92)
        handle.write(bytes([original[0] ^ 1]) + original[1:])
    monkeypatch.setitem(
        sys.modules,
        "pycolmap",
        SimpleNamespace(Reconstruction=lambda _: _reconstruction()),
    )
    args = SimpleNamespace(
        source_db=source,
        candidate_db=candidate,
        expected_source_sha256=source_evidence["sha256"],
        expected_counts=source_evidence["counts"],
        db_snapshot=None,
        migration_receipt=receipt,
        raw_model=tmp_path / "raw",
        final_model=None,
        intrinsics_seed=tmp_path / "seed",
        expected_camera_count=1,
        minimum_registered=1,
        raw_intrinsics_tolerance=1e-6,
        final_intrinsics_tolerance=1e-6,
    )

    report, passed = _audit_report(args)

    assert passed
    assert report["database_contract"] == "migration_receipt_mapper_post"
    assert report["database"]["checks"]["candidate_raw_header_only"]
