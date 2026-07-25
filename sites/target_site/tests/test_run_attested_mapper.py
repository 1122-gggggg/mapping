from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

import run_attested_mapper as attested_runner  # noqa: E402
from run_attested_mapper import (  # noqa: E402
    ArtifactCollisionError,
    ManifestContractError,
    ResourceSample,
    RunnerLimits,
    run_attested_mapper,
)


GIB = 1024**3
RUNNER = TARGET_SITE / "tools" / "run_attested_mapper.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_manifest_hash(payload: dict[str, object]) -> str:
    copied = dict(payload)
    copied.pop("manifest_sha256", None)
    canonical = json.dumps(
        copied, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _write_mapper(path: Path) -> None:
    path.write_text(
        """
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def option(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


mode = option("--mode")
output = Path(option("--output_path"))
output.mkdir(parents=True, exist_ok=True)
(output / "model.bin").write_text("model", encoding="utf-8")
print(f"mapper-mode={mode}", flush=True)
if mode == "nonzero":
    raise SystemExit(7)
if mode == "sleep":
    time.sleep(float(option("--sleep")))
if mode == "spawn":
    Path(option("--mapper-pid")).write_text(str(os.getpid()), encoding="utf-8")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    Path(option("--grandchild-pid")).write_text(str(child.pid), encoding="utf-8")
    time.sleep(60)
if mode == "stubborn-spawn":
    Path(option("--mapper-pid")).write_text(str(os.getpid()), encoding="utf-8")
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
        ]
    )
    Path(option("--grandchild-pid")).write_text(str(child.pid), encoding="utf-8")
    time.sleep(60)
if mode == "echo":
    print(option("--payload"), flush=True)
""".lstrip(),
        encoding="utf-8",
    )


def _write_manifest(
    tmp_path: Path,
    *,
    mode: str = "success",
    extra_args: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    candidate = tmp_path / "candidates" / "B_min_views2"
    candidate.mkdir(parents=True)
    database = candidate / "database.db"
    database.write_bytes(b"frozen-match-db")
    images = tmp_path / "images"
    images.mkdir()
    mapper = tmp_path / "fake_mapper.py"
    _write_mapper(mapper)
    snapshot = tmp_path / "pre_mapper_snapshot.json"
    snapshot.write_text('{"snapshot":"frozen"}\n', encoding="utf-8")
    migration = tmp_path / "legacy_migration.json"
    migration.write_text('{"migration":"frozen"}\n', encoding="utf-8")
    argv = [
        sys.executable,
        str(mapper),
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--output_path",
        str(candidate / "model"),
        "--mode",
        mode,
    ]
    if extra_args:
        argv.extend(extra_args)
    payload: dict[str, object] = {
        "schema": "target-site-glomap-db-reuse-retune/test-v1",
        "experiment_id": "attested-runner-test",
        "manifest_sha256_scope": "canonical_json_without_manifest_sha256",
        "manifest_sha256": "",
        "candidate": {"id": "B_min_views2", "path": "candidates/B_min_views2"},
        "mapper": {
            "path": sys.executable,
            "timeout_seconds": 7200,
            "argv": argv,
        },
        "tools": {
            "sha256": {
                "run_attested_mapper.py": _sha256(RUNNER),
                "glomap": _sha256(Path(sys.executable)),
                "guardian_python": _sha256(Path(sys.executable)),
                "mapper_script": _sha256(mapper),
            },
            "paths": {
                "run_attested_mapper.py": str(RUNNER),
                "glomap": sys.executable,
                "guardian_python": sys.executable,
                "mapper_script": str(mapper),
            },
        },
    }
    payload["candidate"] = {
        "id": "B_min_views2",
        "path": "candidates/B_min_views2",
        "pre_mapper_database": {
            "sha256": _sha256(database),
            "size_bytes": database.stat().st_size,
        },
        "frozen_inputs": {
            "pre_mapper_snapshot": {
                "path": snapshot.name,
                "sha256": _sha256(snapshot),
            },
            "legacy_migration": {
                "path": migration.name,
                "sha256": _sha256(migration),
            },
        },
    }
    payload["manifest_sha256"] = _canonical_manifest_hash(payload)
    manifest = tmp_path / "experiment_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest, candidate, mapper


def _normal_sample(_pid: int) -> ResourceSample:
    return ResourceSample(
        rss_bytes=1024,
        mem_available_bytes=8 * GIB,
        swap_used_bytes=0,
    )


def _wait_for(path: Path, *, timeout_seconds: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {path}")
        time.sleep(0.01)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]
    except FileNotFoundError:
        return False
    return state != "Z"


def _wait_for_dead(pid: int, *, timeout_seconds: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while _alive(pid):
        if time.monotonic() >= deadline:
            raise AssertionError(f"pid {pid} remained alive")
        time.sleep(0.02)


def test_success_publishes_durable_final_artifacts_and_completed_receipt(
    tmp_path: Path,
) -> None:
    manifest, candidate, _ = _write_manifest(tmp_path)

    result = run_attested_mapper(
        manifest,
        candidate_id="B_min_views2",
        sampler=_normal_sample,
        limits=RunnerLimits(poll_interval_seconds=0.01),
    )

    assert result == 0
    completed = json.loads((candidate / "mapper_completed.json").read_text())
    resource = json.loads((candidate / "resource.time").read_text())
    assert completed["status"] == "PASS"
    assert completed["wait_returncode"] == 0
    spawned = json.loads((candidate / "mapper_spawned.json").read_text())
    assert spawned["child_pid"] > 0
    assert spawned["process_group"] == spawned["child_pid"]
    assert completed["mapper_spawned_sha256"] == _sha256(
        candidate / "mapper_spawned.json"
    )
    assert completed["artifact_sha256"]["mapper.log"] == _sha256(
        candidate / "mapper.log"
    )
    assert resource["wall_seconds"] >= 0
    assert resource["ru_maxrss_kib"] >= 0
    assert resource["samples"] >= 1
    assert (candidate / "mapper_started.json").exists()
    assert (candidate / "mapper.exitcode").read_text() == "0\n"
    assert not list(candidate.glob("*.part"))


def test_nonzero_child_is_attested_and_returned_without_claiming_pass(
    tmp_path: Path,
) -> None:
    manifest, candidate, _ = _write_manifest(tmp_path, mode="nonzero")

    result = run_attested_mapper(
        manifest,
        candidate_id="B_min_views2",
        sampler=_normal_sample,
        limits=RunnerLimits(poll_interval_seconds=0.01),
    )

    completed = json.loads((candidate / "mapper_completed.json").read_text())
    assert result == 7
    assert completed["status"] == "CHILD_NONZERO"
    assert completed["wait_returncode"] == 7
    assert (candidate / "mapper.exitcode").read_text() == "7\n"


def test_timeout_terminates_the_new_process_group_and_attests_abort(
    tmp_path: Path,
) -> None:
    manifest, candidate, _ = _write_manifest(
        tmp_path, mode="sleep", extra_args=["--sleep", "60"]
    )

    result = run_attested_mapper(
        manifest,
        candidate_id="B_min_views2",
        sampler=_normal_sample,
        limits=RunnerLimits(
            timeout_seconds=0.15,
            poll_interval_seconds=0.01,
            term_grace_seconds=0.1,
        ),
    )

    completed = json.loads((candidate / "mapper_completed.json").read_text())
    assert result < 0
    assert completed["status"] == "ABORTED"
    assert completed["termination"]["reason"] == "timeout"
    assert completed["termination"]["term_sent"] is True


def test_resource_limit_terminates_process_group_without_orphaning_grandchild(
    tmp_path: Path,
) -> None:
    mapper_pid = tmp_path / "mapper.pid"
    grandchild_pid = tmp_path / "grandchild.pid"
    manifest, candidate, _ = _write_manifest(
        tmp_path,
        mode="stubborn-spawn",
        extra_args=[
            "--mapper-pid",
            str(mapper_pid),
            "--grandchild-pid",
            str(grandchild_pid),
        ],
    )

    def limit_after_grandchild(_pid: int) -> ResourceSample:
        if grandchild_pid.exists():
            return ResourceSample(26 * GIB, 8 * GIB, 0)
        return _normal_sample(_pid)

    result = run_attested_mapper(
        manifest,
        candidate_id="B_min_views2",
        sampler=limit_after_grandchild,
        limits=RunnerLimits(
            timeout_seconds=2,
            poll_interval_seconds=0.01,
            term_grace_seconds=0.1,
        ),
    )

    completed = json.loads((candidate / "mapper_completed.json").read_text())
    assert result < 0
    assert completed["termination"]["reason"] == "rss_limit"
    _wait_for(grandchild_pid)
    _wait_for_dead(int(mapper_pid.read_text()))
    _wait_for_dead(int(grandchild_pid.read_text()))


def test_sigterm_forwards_to_mapper_group_and_completes_abort_receipt(
    tmp_path: Path,
) -> None:
    manifest, candidate, _ = _write_manifest(
        tmp_path, mode="sleep", extra_args=["--sleep", "60"]
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(RUNNER),
            "--manifest",
            str(manifest),
            "--candidate-id",
            "B_min_views2",
            "--poll-interval-seconds",
            "0.01",
            "--term-grace-seconds",
            "0.1",
        ],
    )
    _wait_for(candidate / "mapper_started.json")
    _wait_for(candidate / "mapper_spawned.json")
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=5) != 0
    completed = json.loads((candidate / "mapper_completed.json").read_text())
    assert completed["status"] == "ABORTED"
    assert completed["termination"]["reason"] == "signal:SIGTERM"


def test_sigkill_leaves_started_and_part_artifacts_fail_closed(tmp_path: Path) -> None:
    mapper_pid = tmp_path / "mapper.pid"
    grandchild_pid = tmp_path / "grandchild.pid"
    manifest, candidate, _ = _write_manifest(
        tmp_path,
        mode="stubborn-spawn",
        extra_args=[
            "--mapper-pid",
            str(mapper_pid),
            "--grandchild-pid",
            str(grandchild_pid),
        ],
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(RUNNER),
            "--manifest",
            str(manifest),
            "--candidate-id",
            "B_min_views2",
            "--poll-interval-seconds",
            "0.01",
            "--term-grace-seconds",
            "0.1",
        ],
    )
    _wait_for(candidate / "mapper_started.json")
    _wait_for(candidate / "mapper_spawned.json")
    _wait_for(mapper_pid)
    _wait_for(grandchild_pid)
    process.kill()
    process.wait(timeout=5)

    assert (candidate / "mapper_started.json").exists()
    assert not (candidate / "mapper_completed.json").exists()
    assert list(candidate.glob("*.part"))
    _wait_for_dead(int(mapper_pid.read_text()))
    _wait_for_dead(int(grandchild_pid.read_text()))


def test_pre_mapper_database_drift_after_reservation_is_rejected_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, candidate, _ = _write_manifest(tmp_path)
    original_reserve = attested_runner._reserve_started_artifacts

    def reserve_then_drift(contract: object) -> tuple[str, str]:
        result = original_reserve(contract)  # type: ignore[arg-type]
        (candidate / "database.db").write_bytes(b"drifted-after-reservation")
        return result

    monkeypatch.setattr(
        attested_runner, "_reserve_started_artifacts", reserve_then_drift
    )
    with pytest.raises(ManifestContractError, match="pre-mapper database"):
        run_attested_mapper(manifest, candidate_id="B_min_views2")

    assert (candidate / "mapper_started.json").exists()
    assert not (candidate / "mapper_spawned.json").exists()


def test_frozen_migration_receipt_drift_after_reservation_is_rejected_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, candidate, _ = _write_manifest(tmp_path)
    original_reserve = attested_runner._reserve_started_artifacts

    def reserve_then_drift(contract: object) -> tuple[str, str]:
        result = original_reserve(contract)  # type: ignore[arg-type]
        (tmp_path / "legacy_migration.json").write_text("drift\n", encoding="utf-8")
        return result

    monkeypatch.setattr(
        attested_runner, "_reserve_started_artifacts", reserve_then_drift
    )
    with pytest.raises(ManifestContractError, match="legacy_migration"):
        run_attested_mapper(manifest, candidate_id="B_min_views2")

    assert (candidate / "mapper_started.json").exists()
    assert not (candidate / "mapper_spawned.json").exists()


@pytest.mark.parametrize("sidecar", ["database.db-wal", "database.db-shm"])
def test_pre_mapper_database_sidecars_are_rejected_before_spawn(
    tmp_path: Path, sidecar: str
) -> None:
    manifest, candidate, _ = _write_manifest(tmp_path)
    (candidate / sidecar).write_bytes(b"sidecar")

    with pytest.raises(ManifestContractError, match="sidecar"):
        run_attested_mapper(manifest, candidate_id="B_min_views2")

    assert not (candidate / "mapper_started.json").exists()


def test_hash_drift_is_rejected_before_started_receipt_or_spawn(tmp_path: Path) -> None:
    manifest, candidate, mapper = _write_manifest(tmp_path)
    mapper.write_text("raise SystemExit(0)\n", encoding="utf-8")

    with pytest.raises(ManifestContractError, match="hash"):
        run_attested_mapper(manifest, candidate_id="B_min_views2")

    assert not (candidate / "mapper_started.json").exists()


@pytest.mark.parametrize("artifact", ["mapper.log", "model"])
def test_existing_output_artifacts_are_rejected_before_started_receipt(
    tmp_path: Path, artifact: str
) -> None:
    manifest, candidate, _ = _write_manifest(tmp_path)
    collision = candidate / artifact
    if artifact == "model":
        collision.mkdir()
    else:
        collision.write_text("already exists\n", encoding="utf-8")

    with pytest.raises(ArtifactCollisionError):
        run_attested_mapper(manifest, candidate_id="B_min_views2")

    assert not (candidate / "mapper_started.json").exists()


def test_manifest_output_path_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest, candidate, _ = _write_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    argv = payload["mapper"]["argv"]
    argv[argv.index("--output_path") + 1] = str(tmp_path / "wrong-model")
    payload["manifest_sha256"] = _canonical_manifest_hash(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestContractError, match="output"):
        run_attested_mapper(manifest, candidate_id="B_min_views2")


def test_sampler_polls_more_than_three_times_for_a_live_child(tmp_path: Path) -> None:
    manifest, _candidate, _ = _write_manifest(
        tmp_path, mode="sleep", extra_args=["--sleep", "0.12"]
    )
    calls = 0

    def counting_sampler(pid: int) -> ResourceSample:
        nonlocal calls
        calls += 1
        return _normal_sample(pid)

    assert (
        run_attested_mapper(
            manifest,
            candidate_id="B_min_views2",
            sampler=counting_sampler,
            limits=RunnerLimits(poll_interval_seconds=0.01),
        )
        == 0
    )
    assert calls > 3


def test_shell_metacharacter_is_logged_literal_and_never_executed(
    tmp_path: Path,
) -> None:
    poison = tmp_path / "must-not-exist"
    payload = f"; touch {poison}; #"
    manifest, candidate, _ = _write_manifest(
        tmp_path, mode="echo", extra_args=["--payload", payload]
    )

    assert (
        run_attested_mapper(
            manifest,
            candidate_id="B_min_views2",
            sampler=_normal_sample,
            limits=RunnerLimits(poll_interval_seconds=0.01),
        )
        == 0
    )
    assert not poison.exists()
    assert payload in (candidate / "mapper.log").read_text(encoding="utf-8")
