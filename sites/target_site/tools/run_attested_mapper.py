#!/usr/bin/env python3
"""Run one manifest-pinned GLOMAP mapper with crash-durable evidence.

The runner deliberately has a small, strict contract.  A production manifest
must pin this runner and every executable in ``tools.paths`` by SHA-256, contain
its own canonical ``manifest_sha256``, and declare exactly one candidate.  The
candidate must be fresh: any mapper output, receipt, or partial artifact is a
collision.  Thus a killed launcher leaves an immutable ``mapper_started.json``
and reserved ``*.part`` files rather than a plausible but unattested map.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


GIB = 1024**3
RUNNER_KEY = "run_attested_mapper.py"
MAPPER_KEY = "glomap"
GUARDIAN_PYTHON_KEY = "guardian_python"
STARTED_NAME = "mapper_started.json"
SPAWNED_NAME = "mapper_spawned.json"
COMPLETED_NAME = "mapper_completed.json"
PART_FINAL_NAMES = {
    "mapper.log.part": "mapper.log",
    "resource.time.part": "resource.time",
    "mapper.exitcode.part": "mapper.exitcode",
}
COLLISION_NAMES = (
    tuple(PART_FINAL_NAMES)
    + tuple(PART_FINAL_NAMES.values())
    + (
        STARTED_NAME,
        SPAWNED_NAME,
        COMPLETED_NAME,
        "model",
    )
)


class ManifestContractError(ValueError):
    """The manifest cannot safely authorize a mapper process."""


class ArtifactCollisionError(FileExistsError):
    """The requested candidate is not fresh and must never be reused."""


@dataclass(frozen=True)
class ResourceSample:
    """One process/system resource observation made while the child is live."""

    rss_bytes: int
    mem_available_bytes: int
    swap_used_bytes: int


@dataclass(frozen=True)
class RunnerLimits:
    """Production limits; tests may inject shorter limits without changing manifests."""

    timeout_seconds: float = 7200.0
    max_rss_bytes: int = 26 * GIB
    min_mem_available_bytes: int = 2 * GIB
    poll_interval_seconds: float = 1.0
    term_grace_seconds: float = 15.0

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_rss_bytes <= 0 or self.min_mem_available_bytes < 0:
            raise ValueError("resource limits must be non-negative and non-zero")
        if self.poll_interval_seconds <= 0 or self.term_grace_seconds < 0:
            raise ValueError("poll interval must be positive and grace non-negative")


@dataclass(frozen=True)
class MapperContract:
    """Resolved, fully pinned inputs that the child will receive verbatim."""

    manifest_path: Path
    manifest_sha256: str
    canonical_manifest_sha256: str
    candidate_id: str
    candidate_path: Path
    argv: tuple[str, ...]
    argv_sha256: str
    mapper_path: Path
    runner_path: Path
    frozen_paths: dict[str, Path]
    frozen_sha256: dict[str, str]
    pre_mapper_database_sha256: str
    pre_mapper_database_size: int
    frozen_inputs: dict[str, dict[str, str]]


def _utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="microseconds")
    )


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 without changing the artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _manifest_canonical_hash(payload: Mapping[str, Any]) -> str:
    copied = dict(payload)
    copied.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(copied)).hexdigest()


def _argv_hash(argv: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_json_bytes({"argv": list(argv)})).hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestContractError(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestContractError(f"{label} must be a non-empty string")
    return value


def _resolved_relative(root: Path, value: Any, label: str) -> Path:
    raw = Path(_require_string(value, label))
    if raw.is_absolute():
        raise ManifestContractError(f"{label} must be relative to the manifest")
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ManifestContractError(
            f"{label} escapes the manifest directory"
        ) from error
    return resolved


def _argument_value(argv: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ManifestContractError(f"mapper argv must contain exactly one {flag}")
    return argv[positions[0] + 1]


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _require_sha256(value: Any, label: str) -> str:
    digest = _require_string(value, label)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ManifestContractError(f"{label} must be a lowercase SHA-256")
    return digest


def _frozen_pre_mapper_inputs(
    *, root: Path, candidate: Mapping[str, Any], database_path: Path
) -> tuple[str, int, dict[str, dict[str, str]]]:
    database = _require_mapping(
        candidate.get("pre_mapper_database"), "candidate.pre_mapper_database"
    )
    database_hash = _require_sha256(
        database.get("sha256"), "candidate.pre_mapper_database.sha256"
    )
    database_size = database.get("size_bytes")
    if not isinstance(database_size, int) or database_size < 0:
        raise ManifestContractError(
            "candidate.pre_mapper_database.size_bytes must be a non-negative integer"
        )
    inputs_raw = _require_mapping(
        candidate.get("frozen_inputs"), "candidate.frozen_inputs"
    )
    required_inputs = {"pre_mapper_snapshot", "legacy_migration"}
    if set(inputs_raw) != required_inputs:
        raise ManifestContractError(
            "candidate.frozen_inputs must pin pre_mapper_snapshot and legacy_migration"
        )
    frozen_inputs: dict[str, dict[str, str]] = {}
    for name in sorted(required_inputs):
        receipt = _require_mapping(inputs_raw[name], f"candidate.frozen_inputs.{name}")
        receipt_path = _resolved_relative(
            root, receipt.get("path"), f"candidate.frozen_inputs.{name}.path"
        )
        receipt_hash = _require_sha256(
            receipt.get("sha256"), f"candidate.frozen_inputs.{name}.sha256"
        )
        frozen_inputs[name] = {"path": str(receipt_path), "sha256": receipt_hash}
    _verify_frozen_inputs(frozen_inputs)
    _verify_pre_mapper_database(
        database_path=database_path,
        expected_sha256=database_hash,
        expected_size=database_size,
    )
    return database_hash, database_size, frozen_inputs


def _verify_frozen_inputs(frozen_inputs: Mapping[str, Mapping[str, str]]) -> None:
    for name, receipt in frozen_inputs.items():
        receipt_path = Path(receipt["path"])
        if not receipt_path.is_file() or sha256_file(receipt_path) != receipt["sha256"]:
            raise ManifestContractError(f"frozen input differs for {name}")


def _verify_pre_mapper_database(
    *, database_path: Path, expected_sha256: str, expected_size: int
) -> None:
    sidecars = [
        str(path.name)
        for path in (Path(f"{database_path}-wal"), Path(f"{database_path}-shm"))
        if path.exists()
    ]
    if sidecars:
        raise ManifestContractError(
            f"pre-mapper database sidecar is present: {', '.join(sidecars)}"
        )
    if not database_path.is_file():
        raise ManifestContractError("pre-mapper database is absent")
    if database_path.stat().st_size != expected_size:
        raise ManifestContractError("pre-mapper database size differs from manifest")
    if sha256_file(database_path) != expected_sha256:
        raise ManifestContractError("pre-mapper database hash differs from manifest")


def verify_manifest_contract(
    manifest_path: Path | str,
    *,
    candidate_id: str,
    runner_path: Path | None = None,
) -> MapperContract:
    """Validate every immutable input before reserving any mapper output path."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.is_file():
        raise ManifestContractError(f"manifest is absent: {manifest_file}")
    try:
        manifest = _require_mapping(
            json.loads(manifest_file.read_text(encoding="utf-8")), "manifest"
        )
    except json.JSONDecodeError as error:
        raise ManifestContractError(
            f"manifest is invalid JSON: {manifest_file}"
        ) from error
    if (
        manifest.get("manifest_sha256_scope")
        != "canonical_json_without_manifest_sha256"
    ):
        raise ManifestContractError(
            "manifest_sha256_scope is not the attested canonical scope"
        )
    recorded_manifest_hash = _require_string(
        manifest.get("manifest_sha256"), "manifest_sha256"
    )
    canonical_manifest_hash = _manifest_canonical_hash(manifest)
    if recorded_manifest_hash != canonical_manifest_hash:
        raise ManifestContractError(
            "manifest canonical self-hash differs from manifest_sha256"
        )

    root = manifest_file.parent
    candidate = _require_mapping(manifest.get("candidate"), "candidate")
    declared_candidate_id = _require_string(candidate.get("id"), "candidate.id")
    if declared_candidate_id != candidate_id:
        raise ManifestContractError(
            f"candidate id mismatch: manifest={declared_candidate_id!r}, requested={candidate_id!r}"
        )
    candidate_path = _resolved_relative(root, candidate.get("path"), "candidate.path")
    if candidate_path.parent != (root / "candidates").resolve():
        raise ManifestContractError(
            "candidate.path must be exactly below manifest/candidates"
        )
    if candidate_path.name != candidate_id or not candidate_path.is_dir():
        raise ManifestContractError(
            "candidate path does not resolve to the requested directory"
        )

    mapper = _require_mapping(manifest.get("mapper"), "mapper")
    mapper_path = Path(_require_string(mapper.get("path"), "mapper.path")).expanduser()
    if not mapper_path.is_absolute():
        raise ManifestContractError("mapper.path must be absolute")
    mapper_path = mapper_path.resolve()
    if not mapper_path.is_file():
        raise ManifestContractError(f"mapper binary is absent: {mapper_path}")
    if mapper.get("timeout_seconds") != 7200:
        raise ManifestContractError("mapper.timeout_seconds must be exactly 7200")
    argv_raw = mapper.get("argv")
    if (
        not isinstance(argv_raw, list)
        or not argv_raw
        or not all(isinstance(value, str) and value for value in argv_raw)
    ):
        raise ManifestContractError("mapper.argv must be a non-empty string array")
    argv = tuple(argv_raw)
    if not _same_path(Path(argv[0]), mapper_path):
        raise ManifestContractError("mapper.argv[0] differs from mapper.path")
    expected_database = (candidate_path / "database.db").resolve()
    expected_output = (candidate_path / "model").resolve()
    database_path = (
        Path(_argument_value(argv, "--database_path")).expanduser().resolve()
    )
    image_path = Path(_argument_value(argv, "--image_path")).expanduser().resolve()
    output_path = Path(_argument_value(argv, "--output_path")).expanduser().resolve()
    if database_path != expected_database or not database_path.is_file():
        raise ManifestContractError(
            "mapper database path is not the candidate database"
        )
    if output_path != expected_output:
        raise ManifestContractError("mapper output path is not candidate/model")
    if not image_path.is_dir():
        raise ManifestContractError("mapper image path is absent")
    (
        pre_mapper_database_sha256,
        pre_mapper_database_size,
        frozen_inputs,
    ) = _frozen_pre_mapper_inputs(
        root=root, candidate=candidate, database_path=expected_database
    )

    tools = _require_mapping(manifest.get("tools"), "tools")
    hashes_raw = _require_mapping(tools.get("sha256"), "tools.sha256")
    paths_raw = _require_mapping(tools.get("paths"), "tools.paths")
    if set(hashes_raw) != set(paths_raw):
        raise ManifestContractError(
            "tools.sha256 and tools.paths must contain identical keys"
        )
    if {RUNNER_KEY, MAPPER_KEY, GUARDIAN_PYTHON_KEY} - set(hashes_raw):
        raise ManifestContractError(
            "tools must pin the attested runner and glomap binary"
        )
    frozen_paths: dict[str, Path] = {}
    frozen_hashes: dict[str, str] = {}
    for name in sorted(hashes_raw):
        expected_hash = _require_string(hashes_raw[name], f"tools.sha256.{name}")
        path = Path(
            _require_string(paths_raw[name], f"tools.paths.{name}")
        ).expanduser()
        if not path.is_absolute():
            raise ManifestContractError(f"tools.paths.{name} must be absolute")
        path = path.resolve()
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ManifestContractError(f"frozen hash differs for {name}")
        frozen_paths[str(name)] = path
        frozen_hashes[str(name)] = expected_hash
    resolved_runner = (runner_path or Path(__file__)).resolve()
    if frozen_paths[RUNNER_KEY] != resolved_runner:
        raise ManifestContractError(
            "manifest runner path differs from executing runner"
        )
    if frozen_paths[MAPPER_KEY] != mapper_path:
        raise ManifestContractError("manifest glomap path differs from mapper.path")
    if frozen_paths[GUARDIAN_PYTHON_KEY] != Path(sys.executable).resolve():
        raise ManifestContractError(
            "manifest guardian_python differs from this Python interpreter"
        )

    return MapperContract(
        manifest_path=manifest_file,
        manifest_sha256=sha256_file(manifest_file),
        canonical_manifest_sha256=canonical_manifest_hash,
        candidate_id=candidate_id,
        candidate_path=candidate_path,
        argv=argv,
        argv_sha256=_argv_hash(argv),
        mapper_path=mapper_path,
        runner_path=resolved_runner,
        frozen_paths=frozen_paths,
        frozen_sha256=frozen_hashes,
        pre_mapper_database_sha256=pre_mapper_database_sha256,
        pre_mapper_database_size=pre_mapper_database_size,
        frozen_inputs=frozen_inputs,
    )


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    _fsync_directory(path.parent)


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish an O_EXCL JSON receipt after fully syncing a temp file."""
    temporary = path.parent / f".{path.name}.{uuid.uuid4()}.tmp"
    try:
        _exclusive_bytes(
            temporary,
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n",
        )
        os.link(temporary, path)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_part_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("r+b") as handle:
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _write_part_exitcode(path: Path, returncode: int) -> None:
    with path.open("r+b") as handle:
        handle.seek(0)
        handle.truncate()
        handle.write(f"{returncode}\n".encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())


def _publish_part_exclusively(part: Path, final: Path) -> None:
    """Publish an already-fsynced file without permitting overwrite."""
    os.link(part, final)
    part.unlink()
    _fsync_directory(final.parent)


def _default_sampler(pid: int) -> ResourceSample:
    rss_bytes = 0
    try:
        for line in (
            Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
        ):
            if line.startswith("VmRSS:"):
                rss_bytes = int(line.split()[1]) * 1024
                break
    except FileNotFoundError:
        pass
    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = int(value.split()[0]) * 1024
    return ResourceSample(
        rss_bytes=rss_bytes,
        mem_available_bytes=meminfo["MemAvailable"],
        swap_used_bytes=meminfo["SwapTotal"] - meminfo["SwapFree"],
    )


def _wait4_nowait(pid: int) -> tuple[int, resource.struct_rusage] | None:
    waited_pid, status, usage = os.wait4(pid, os.WNOHANG)
    if waited_pid == 0:
        return None
    return os.waitstatus_to_exitcode(status), usage


def _wait4_blocking(pid: int) -> tuple[int, resource.struct_rusage]:
    while True:
        try:
            _waited_pid, status, usage = os.wait4(pid, 0)
            return os.waitstatus_to_exitcode(status), usage
        except InterruptedError:
            continue


def _signal_process_group(process_group: int, signum: signal.Signals) -> bool:
    try:
        os.killpg(process_group, signum)
    except ProcessLookupError:
        return False
    return True


def _linux_parent_death_preexec(expected_parent_pid: int) -> Callable[[], None]:
    """Require a guardian to die if its attesting launcher is SIGKILLed.

    ``PR_SET_PDEATHSIG`` is deliberately established in the child immediately
    before exec.  The PPID check closes the small fork-to-prctl race: if the
    launcher has already disappeared, terminate before running any mapper.
    """

    def preexec() -> None:
        pr_set_pdeathsig = 1
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(pr_set_pdeathsig, int(signal.SIGTERM), 0, 0, 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, "prctl(PR_SET_PDEATHSIG) failed")
        if os.getppid() != expected_parent_pid:
            os.kill(os.getpid(), signal.SIGTERM)

    return preexec


def _process_group_members(process_group: int) -> set[int]:
    """Return live Linux PIDs in one process group without traversing ancestry."""
    members: set[int] = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            contents = (entry / "stat").read_text(encoding="utf-8")
            remainder = contents.rsplit(")", 1)[1].split()
            # After the final ')' the sequence begins at field 3 (state); pgrp
            # is field 5, hence index 2.  This handles process names with spaces.
            if int(remainder[2]) == process_group and remainder[0] != "Z":
                members.add(int(entry.name))
        except (FileNotFoundError, IndexError, ValueError):
            continue
    return members


def _signal_group_members(
    *, process_group: int, signum: signal.Signals, exclude: set[int] = frozenset()
) -> bool:
    signaled = False
    for member in _process_group_members(process_group) - exclude:
        try:
            os.kill(member, signum)
            signaled = True
        except ProcessLookupError:
            continue
    return signaled


def _guardian_main(
    *, control_fd: int, grace_seconds: float, mapper_argv: Sequence[str]
) -> int:
    """Supervise mapper descendants when the attested runner is forcibly killed.

    This process is the session/process-group leader.  Its parent-death SIGTERM
    is converted into TERM for every mapper-group member, followed by KILL for
    any TERM-ignoring descendant.  It also performs the same sweep when the
    mapper leader exits first, so a successful wait never leaves a background
    child in the mapper process group.
    """
    process_group = os.getpgrp()
    guardian_pid = os.getpid()
    requested_signal: signal.Signals | None = None
    term_sent = False
    deadline: float | None = None

    def request_termination(signum: int, _frame: Any) -> None:
        nonlocal requested_signal
        if requested_signal is None:
            requested_signal = signal.Signals(signum)

    signal.signal(signal.SIGTERM, request_termination)
    signal.signal(signal.SIGINT, request_termination)
    try:
        mapper = subprocess.Popen(
            list(mapper_argv),
            stdin=subprocess.DEVNULL,
            shell=False,
            close_fds=True,
        )
        control_payload = json.dumps(
            {"mapper_pid": mapper.pid, "process_group": process_group},
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            os.write(control_fd, control_payload)
        except BrokenPipeError:
            # The launcher has already gone away.  Continue into the normal
            # guardian cleanup loop rather than abandoning a live mapper.
            requested_signal = signal.SIGTERM
    finally:
        os.close(control_fd)

    mapper_returncode: int | None = None
    while True:
        if mapper_returncode is None:
            mapper_returncode = mapper.poll()
        group_members = _process_group_members(process_group) - {guardian_pid}
        needs_cleanup = requested_signal is not None or (
            mapper_returncode is not None and bool(group_members)
        )
        if needs_cleanup and not term_sent:
            _signal_group_members(
                process_group=process_group,
                signum=signal.SIGTERM,
                exclude={guardian_pid},
            )
            term_sent = True
            deadline = time.monotonic() + grace_seconds
        if term_sent and deadline is not None and time.monotonic() >= deadline:
            _signal_group_members(
                process_group=process_group,
                signum=signal.SIGKILL,
                exclude={guardian_pid},
            )
        if mapper_returncode is not None and not group_members:
            break
        time.sleep(0.01)

    if requested_signal is not None:
        signal.signal(requested_signal, signal.SIG_DFL)
        os.kill(guardian_pid, requested_signal)
        raise AssertionError("self-signal should not return")
    return mapper_returncode


class _ForwardedSignals:
    """Turn external TERM/INT into an audited child process-group termination."""

    def __init__(self) -> None:
        self.received: signal.Signals | None = None
        self._previous: dict[signal.Signals, Any] = {}

    def install(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handler)

    def restore(self) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()

    def _handler(self, signum: int, _frame: Any) -> None:
        self.received = signal.Signals(signum)


def _assert_fresh_output(candidate: Path) -> None:
    collisions = [name for name in COLLISION_NAMES if (candidate / name).exists()]
    if collisions:
        raise ArtifactCollisionError(
            f"candidate already contains mapper artifacts: {', '.join(sorted(collisions))}"
        )


def _reserve_started_artifacts(contract: MapperContract) -> tuple[str, str]:
    candidate = contract.candidate_path
    _assert_fresh_output(candidate)
    for part_name in PART_FINAL_NAMES:
        _exclusive_bytes(candidate / part_name, b"")
    run_uuid = str(uuid.uuid4())
    started_at = _utc_now()
    _exclusive_json(
        candidate / STARTED_NAME,
        {
            "schema": "target-site-attested-mapper-started/v1",
            "status": "PRE_SPAWN_RESERVED",
            "run_uuid": run_uuid,
            "started_at": started_at,
            "launcher_pid": os.getpid(),
            "child_pid": None,
            "candidate_id": contract.candidate_id,
            "manifest": {
                "path": str(contract.manifest_path),
                "file_sha256": contract.manifest_sha256,
                "canonical_sha256": contract.canonical_manifest_sha256,
            },
            "runner": {
                "path": str(contract.runner_path),
                "sha256": contract.frozen_sha256[RUNNER_KEY],
            },
            "mapper": {
                "path": str(contract.mapper_path),
                "sha256": contract.frozen_sha256[MAPPER_KEY],
                "argv": list(contract.argv),
                "argv_sha256": contract.argv_sha256,
            },
            "frozen_paths": {
                name: {"path": str(path), "sha256": contract.frozen_sha256[name]}
                for name, path in sorted(contract.frozen_paths.items())
            },
        },
    )
    return run_uuid, started_at


def _write_spawned_receipt(
    *, contract: MapperContract, run_uuid: str, child_pid: int
) -> None:
    _exclusive_json(
        contract.candidate_path / SPAWNED_NAME,
        {
            "schema": "target-site-attested-mapper-spawned/v1",
            "status": "SPAWNED",
            "run_uuid": run_uuid,
            "spawned_at": _utc_now(),
            "launcher_pid": os.getpid(),
            "child_pid": child_pid,
            "process_group": child_pid,
            "candidate_id": contract.candidate_id,
            "manifest_file_sha256": contract.manifest_sha256,
            "mapper_argv_sha256": contract.argv_sha256,
        },
    )


def _sample_stats(sample: ResourceSample, stats: dict[str, int]) -> None:
    stats["samples"] += 1
    stats["peak_rss_bytes"] = max(stats["peak_rss_bytes"], sample.rss_bytes)
    stats["min_mem_available_bytes"] = min(
        stats["min_mem_available_bytes"], sample.mem_available_bytes
    )
    stats["peak_swap_used_bytes"] = max(
        stats["peak_swap_used_bytes"], sample.swap_used_bytes
    )


def _termination_reason(
    *,
    now: float,
    deadline: float,
    sample: ResourceSample,
    limits: RunnerLimits,
    forwarded: _ForwardedSignals,
) -> str | None:
    if forwarded.received is not None:
        return f"signal:{forwarded.received.name}"
    if now >= deadline:
        return "timeout"
    if sample.rss_bytes >= limits.max_rss_bytes:
        return "rss_limit"
    if sample.mem_available_bytes <= limits.min_mem_available_bytes:
        return "mem_available_limit"
    return None


def _wait_after_termination(
    *,
    pid: int,
    process_group: int,
    deadline: float,
) -> tuple[int, resource.struct_rusage, bool]:
    """Wait through TERM grace and prove the mapper process group is empty.

    A mapper leader can exit before a TERM-ignoring descendant.  Do not return
    merely because ``waitpid(leader)`` succeeded: inspect the full process
    group and KILL every remaining member after the grace interval.
    """
    leader_result: tuple[int, resource.struct_rusage] | None = None
    kill_sent = False
    while True:
        if leader_result is None:
            leader_result = _wait4_nowait(pid)
        remaining = _process_group_members(process_group)
        if leader_result is not None and not remaining:
            return leader_result[0], leader_result[1], kill_sent
        if time.monotonic() >= deadline:
            kill_sent = (
                _signal_group_members(
                    process_group=process_group, signum=signal.SIGKILL
                )
                or kill_sent
            )
        time.sleep(0.02)


def _directory_hashes(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    if not path.is_dir() or path.is_symlink():
        return {"__invalid__": "not-a-real-directory"}
    return {
        str(file.relative_to(path)): sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file() and not file.is_symlink()
    }


def run_attested_mapper(
    manifest_path: Path | str,
    *,
    candidate_id: str,
    sampler: Callable[[int], ResourceSample] = _default_sampler,
    limits: RunnerLimits = RunnerLimits(),
) -> int:
    """Run the exact manifest argv and return its child return code.

    A negative return code is preserved for a child terminated by a signal.  The
    CLI maps that value to the conventional ``128 + signal`` process status.
    """
    contract = verify_manifest_contract(manifest_path, candidate_id=candidate_id)
    run_uuid, started_at = _reserve_started_artifacts(contract)
    candidate = contract.candidate_path
    log_part = candidate / "mapper.log.part"
    resource_part = candidate / "resource.time.part"
    exitcode_part = candidate / "mapper.exitcode.part"
    log_handle = log_part.open("ab", buffering=0)
    forwarded = _ForwardedSignals()
    process: subprocess.Popen[bytes] | None = None
    guardian_control_read: int | None = None
    guardian_control_write: int | None = None
    mapper_pid: int | None = None
    rusage: resource.struct_rusage | None = None
    termination: dict[str, Any] = {
        "reason": None,
        "term_sent": False,
        "kill_sent": False,
        "requested_at": None,
    }
    stats = {
        "samples": 0,
        "peak_rss_bytes": 0,
        "min_mem_available_bytes": sys.maxsize,
        "peak_swap_used_bytes": 0,
    }
    started_monotonic = time.monotonic()
    returncode: int | None = None
    try:
        forwarded.install()
        # Verify the database and migration evidence again after the started
        # reservation, immediately before the child can consume it.
        _verify_pre_mapper_database(
            database_path=contract.candidate_path / "database.db",
            expected_sha256=contract.pre_mapper_database_sha256,
            expected_size=contract.pre_mapper_database_size,
        )
        _verify_frozen_inputs(contract.frozen_inputs)
        guardian_control_read, guardian_control_write = os.pipe()
        os.set_blocking(guardian_control_read, False)
        process = subprocess.Popen(
            [
                str(contract.frozen_paths[GUARDIAN_PYTHON_KEY]),
                str(contract.runner_path),
                "--_guardian",
                "--control-fd",
                str(guardian_control_write),
                "--grace-seconds",
                str(limits.term_grace_seconds),
                "--",
                *contract.argv,
            ],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            shell=False,
            start_new_session=True,
            close_fds=True,
            pass_fds=(guardian_control_write,),
            preexec_fn=_linux_parent_death_preexec(os.getpid()),
        )
        os.close(guardian_control_write)
        guardian_control_write = None
        _write_spawned_receipt(
            contract=contract, run_uuid=run_uuid, child_pid=process.pid
        )
        deadline = started_monotonic + limits.timeout_seconds
        while True:
            if guardian_control_read is not None and mapper_pid is None:
                try:
                    control_data = os.read(guardian_control_read, 4096)
                except BlockingIOError:
                    control_data = b""
                if control_data:
                    try:
                        mapper_pid = int(json.loads(control_data)["mapper_pid"])
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        raise RuntimeError(
                            "guardian emitted invalid mapper pid evidence"
                        )
            sample = sampler(mapper_pid or process.pid)
            _sample_stats(sample, stats)
            result = _wait4_nowait(process.pid)
            if result is not None:
                returncode, rusage = result
                process.returncode = returncode
                break
            reason = _termination_reason(
                now=time.monotonic(),
                deadline=deadline,
                sample=sample,
                limits=limits,
                forwarded=forwarded,
            )
            if reason is not None:
                termination.update(
                    {
                        "reason": reason,
                        "term_sent": _signal_process_group(process.pid, signal.SIGTERM),
                        "requested_at": _utc_now(),
                    }
                )
                returncode, rusage, termination["kill_sent"] = _wait_after_termination(
                    pid=process.pid,
                    process_group=process.pid,
                    deadline=time.monotonic() + limits.term_grace_seconds,
                )
                process.returncode = returncode
                break
            time.sleep(limits.poll_interval_seconds)
    finally:
        forwarded.restore()
        if guardian_control_write is not None:
            os.close(guardian_control_write)
        if guardian_control_read is not None:
            os.close(guardian_control_read)
        log_handle.flush()
        os.fsync(log_handle.fileno())
        log_handle.close()

    if returncode is None or rusage is None:
        # The started receipt and parts remain intentionally incomplete if the
        # runner cannot prove a waitpid result.
        raise RuntimeError("mapper process ended without an attested waitpid result")
    ended_monotonic = time.monotonic()
    ended_at = _utc_now()
    if stats["min_mem_available_bytes"] == sys.maxsize:
        stats["min_mem_available_bytes"] = 0
    resource_payload = {
        "schema": "target-site-attested-mapper-resource/v1",
        "run_uuid": run_uuid,
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_seconds": ended_monotonic - started_monotonic,
        "user_seconds": rusage.ru_utime,
        "system_seconds": rusage.ru_stime,
        "ru_maxrss_kib": rusage.ru_maxrss,
        **stats,
        "termination": termination,
        "wait_returncode": returncode,
    }
    _write_part_json(resource_part, resource_payload)
    _write_part_exitcode(exitcode_part, returncode)
    for part_name, final_name in PART_FINAL_NAMES.items():
        _publish_part_exclusively(candidate / part_name, candidate / final_name)
    status = (
        "ABORTED"
        if termination["reason"] is not None
        else ("PASS" if returncode == 0 else "CHILD_NONZERO")
    )
    completed_payload = {
        "schema": "target-site-attested-mapper-completed/v1",
        "status": status,
        "run_uuid": run_uuid,
        "candidate_id": contract.candidate_id,
        "started_at": started_at,
        "completed_at": ended_at,
        "child_pid": process.pid,
        "mapper_pid": mapper_pid,
        "wait_returncode": returncode,
        "termination": termination,
        "manifest": {
            "path": str(contract.manifest_path),
            "file_sha256": contract.manifest_sha256,
            "canonical_sha256": contract.canonical_manifest_sha256,
        },
        "mapper": {
            "path": str(contract.mapper_path),
            "sha256": contract.frozen_sha256[MAPPER_KEY],
            "argv_sha256": contract.argv_sha256,
        },
        "mapper_spawned_sha256": sha256_file(candidate / SPAWNED_NAME),
        "pre_mapper_database": {
            "sha256": contract.pre_mapper_database_sha256,
            "size_bytes": contract.pre_mapper_database_size,
        },
        "frozen_inputs": contract.frozen_inputs,
        "artifact_sha256": {
            name: sha256_file(candidate / name)
            for name in sorted(PART_FINAL_NAMES.values())
        },
        "model_file_sha256": _directory_hashes(candidate / "model"),
    }
    _exclusive_json(candidate / COMPLETED_NAME, completed_payload)
    _fsync_directory(candidate)
    return returncode


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--_guardian"]:
        guardian_parser = argparse.ArgumentParser(add_help=False)
        guardian_parser.add_argument("--_guardian", action="store_true")
        guardian_parser.add_argument("--control-fd", type=int, required=True)
        guardian_parser.add_argument("--grace-seconds", type=float, required=True)
        guardian_args, mapper_argv = guardian_parser.parse_known_args(arguments)
        if not mapper_argv[:1] == ["--"] or len(mapper_argv) == 1:
            guardian_parser.error("guardian must receive mapper argv after --")
        return _guardian_main(
            control_fd=guardian_args.control_fd,
            grace_seconds=guardian_args.grace_seconds,
            mapper_argv=mapper_argv[1:],
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--term-grace-seconds", type=float, default=15.0)
    args = parser.parse_args(arguments)
    try:
        returncode = run_attested_mapper(
            args.manifest,
            candidate_id=args.candidate_id,
            limits=RunnerLimits(
                poll_interval_seconds=args.poll_interval_seconds,
                term_grace_seconds=args.term_grace_seconds,
            ),
        )
    except (
        ArtifactCollisionError,
        ManifestContractError,
        OSError,
        RuntimeError,
    ) as error:
        print(str(error), file=sys.stderr, flush=True)
        return 2
    return returncode if returncode >= 0 else 128 + abs(returncode)


if __name__ == "__main__":
    raise SystemExit(main())
