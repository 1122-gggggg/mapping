#!/usr/bin/env python3
"""Shared definitions for the football_field gluemap -> EDM build.

The corpus split is declared HERE, once, and every stage imports it. The
held-out test video (P1280128) is byte-for-byte the same resolution and camera
as the two build videos, so nothing geometric can tell it apart at any later
stage -- path-based exclusion is the ONLY thing keeping it out of the map.
Declaring the split in one place, and asserting it at every stage, is the
whole defence.

Provisional split (S0/S1 forked from target_site 2026-07-19): football is NOT
a corridor corpus -- straightness 0.10-0.20 (orbital/area survey), all
pairwise net-displacement angles 96-117deg ("crossing"), no opposite pair.
Every video is labelled direction="unknown"; do NOT invent fwd/rev labels.
BUILD = P1270127, P1290129 (44-53% mutual overlap). TEST = P1280128 (shortest,
only 51 refs, 9-13% overlap with the others).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

DATA = Path("/media/cihcilab/新增磁碟區/足球場")
BUILD_ROOT = Path("/media/cihcilab/新增磁碟區/sfm_system/建圖/football_field")
RUNS = BUILD_ROOT / "runs"
RUN_ID = "football_field_v1"

GLUEMAP_REPO = Path("/media/cihcilab/新增磁碟區/河濱場域/gluemap_build/tools/gluemap")
GLUEMAP_ENV = Path("/home/cihcilab/micromamba/envs/target-site-gluemap-run")
GLUEMAP_PY = GLUEMAP_ENV / "bin" / "python"
GLUEMAP_DEMO = GLUEMAP_ENV / "bin" / "gluemap-demo"

# EDM's canvas (deploy/edm_matcher.py). Exactly 16:9. EDM plain-resizes every
# reference into it and rescales keypoints with a SINGLE width-derived factor
# (scale = camera.width / EDM_W), applied to BOTH axes. That is only correct
# while every source image is 16:9. A 4:3 frame would be silently squashed and
# its y-coordinates would be wrong by 33% -- with no error anywhere.
EDM_W, EDM_H = 1024, 576
ASPECT = 16 / 9

# Sequence folder names must match this, or gluemap's multi-sequence loader
# discovers ZERO sequences and drops them without raising.
SUBFOLDER_REGEX = r"^P[0-9]+"


@dataclass(frozen=True)
class Video:
    seq: str  # sequence folder name; must match SUBFOLDER_REGEX
    rel: str  # path relative to DATA
    width: int
    height: int
    fps: float
    direction: str  # "fwd" | "rev" | "unknown" -- resolved by S1 where unknown
    epoch: str

    @property
    def path(self) -> Path:
        return DATA / self.rel

    @property
    def shape(self) -> tuple[int, int]:
        return (self.width, self.height)


# ---------------------------------------------------------------- the corpus
# 3 videos. All Parrot Anafi, same drone serial PI040416BA8L105488, all shot on
# 2026-07-06, all 2688x1512 @ 24000/1001. Orbital/area survey, NOT a corridor
# -- no fwd/rev pair exists (see module docstring), so every direction is
# "unknown" pending S1.
BUILD: list[Video] = [
    Video(
        "P1270127",
        "raw_videos/P1270127.MP4",
        2688,
        1512,
        24000 / 1001,
        "unknown",
        "2026-07-06",
    ),
    Video(
        "P1290129",
        "raw_videos/P1290129.MP4",
        2688,
        1512,
        24000 / 1001,
        "unknown",
        "2026-07-06",
    ),
]

# NEVER enters the map. Same camera and resolution as the build set -- shortest
# video (51 refs) with only 9-13% spatial overlap with the others.
TEST: list[Video] = [
    Video(
        "P1280128",
        "raw_videos/P1280128.MP4",
        2688,
        1512,
        24000 / 1001,
        "unknown",
        "2026-07-06",
    ),
]

# Nothing dropped for this corpus -- all three videos carry Parrot metadata and
# probe at their declared resolution (see metadata/videos_manifest.json).
DROPPED: dict[str, str] = {}

# Nothing to exclude for this corpus.
EXCLUDED_DIRS: list[Path] = []

BUILD_BY_SEQ = {v.seq: v for v in BUILD}
FORBIDDEN_STEMS = {Path(v.rel).stem for v in TEST}


def resolution_groups(
    videos: list[Video] | None = None,
) -> dict[tuple[int, int], list[Video]]:
    """Group videos by (width, height). One gluemap camera per group.

    gluemap's ``intrinsics_mode: SHARED`` already buckets one camera per unique
    image SHAPE, so a per-resolution seed lines up with it 1:1.
    """
    groups: dict[tuple[int, int], list[Video]] = {}
    for v in videos if videos is not None else BUILD:
        groups.setdefault(v.shape, []).append(v)
    return dict(sorted(groups.items()))


# ---------------------------------------------------------------- gates
GateState = Literal["PASS", "FAIL", "NOT_RUN", "INCOMPLETE"]
GATE_STATES: frozenset[str] = frozenset({"PASS", "FAIL", "NOT_RUN", "INCOMPLETE"})

_SEQUENCES = tuple(v.seq for v in BUILD)
REQUIRED_GATE_IDS: dict[str, frozenset[str]] = {
    "S0_corpus": frozenset(
        {
            "G0.1a",
            "G0.1b",
            "G0.2a",
            "G0.2b",
            "G0.3",
            "G0.4",
            "G0.5",
            "G0.6",
            "G0.7",
            "G0.8",
        }
    ),
    "S1_motion": frozenset(
        {"G0.2", "G1.4a", "G1.4b", "G1.4c", "G1.5", "G1.6"}
        | {f"G1.1/{seq}" for seq in _SEQUENCES}
        | {f"G1.2/{seq}" for seq in _SEQUENCES}
    ),
    "S2_extract": frozenset(
        {
            "G0.2",
            "G2.1",
            "G2.2",
            "G2.3a",
            "G2.4",
            "G2.5",
            "G2.6",
        }
        | {f"G1.3/{seq}" for seq in _SEQUENCES}
    ),
    "S2b_intrinsics": frozenset(
        {
            "G2.7/results_complete",
            "G2.7/2688x1512",
            "G2.8",
        }
    ),
    "S3_pairs": frozenset(
        {
            "G0.2",
            "G3.0",
            "G3.1",
            "G3.2",
            "G3.3",
            "G3.4",
            "G3.5a",
            "G3.5b",
            "G3.5c",
            "G3.5d",
            "G3.6",
        }
    ),
}


class GateDefinitionError(ValueError):
    """The producer violated the declared gate API contract."""


class GateFreshnessError(RuntimeError):
    """Recorded gate provenance no longer matches bytes on disk."""


class GateFailure(SystemExit):
    """A finalized gate was not PASS while fail_hard was enabled."""


def required_check_ids(stage: str) -> frozenset[str]:
    """Return the immutable, exact check-ID set for a release stage."""
    try:
        return REQUIRED_GATE_IDS[stage]
    except KeyError as exc:
        known = ", ".join(sorted(REQUIRED_GATE_IDS))
        raise GateDefinitionError(
            f"unknown release stage {stage!r}; expected one of: {known}"
        ) from exc


def stage_material_artifacts(stage: str, run_dir: Path | str) -> dict[str, Path]:
    """Return the exact persisted artifact closure for one release stage."""
    root = Path(run_dir)
    common = {"corpus_manifest": root / "corpus_manifest.json"}
    if stage == "S0_corpus":
        return {
            **common,
            **{f"declared_video/{video.seq}": video.path for video in [*BUILD, *TEST]},
        }
    if stage == "S1_motion":
        return {**common, "motion_manifest": root / "motion_manifest.json"}
    s2 = {
        **common,
        "motion_manifest": root / "motion_manifest.json",
        "frame_manifest": root / "frame_manifest.json",
        "images": root / "images",
        "intrinsics_seed_manifest": root / "intrinsics_seed.json",
        "intrinsics_seed": root / "intrinsics_seed",
    }
    if stage == "S2_extract":
        return s2
    if stage == "S2b_intrinsics":
        artifacts = {**s2, "intrinsics_bakeoff": root / "intrinsics_bakeoff.json"}
        for shape in (f"{w}x{h}" for w, h in resolution_groups()):
            for seed in ("official69", "charuco"):
                prefix = f"solve/{shape}/{seed}"
                work = root / "intrinsics_bakeoff" / f"{shape}__{seed}"
                artifacts.update(
                    {
                        f"{prefix}/status": work / "status.json",
                        f"{prefix}/result": work / "result.json",
                        f"{prefix}/config": work / "gluemap_config.yaml",
                        f"{prefix}/seed": work / "seed",
                        f"{prefix}/model": work / "gluemap" / "gluemap_aba",
                    }
                )
        return artifacts
    if stage == "S3_pairs":
        return {
            **s2,
            "intrinsics_bakeoff": root / "intrinsics_bakeoff.json",
            "forced_bridges_manifest": root / "forced_bridges.json",
            "forced_bridges_text": root / "forced_bridges.txt",
            "s3_loader_probe": root / "s3_loader_probe.json",
            "s3_loader_probe_log": root / "s3_loader_probe.log",
            "gluemap_config": root / "gluemap_config.yaml",
        }
    raise GateDefinitionError(f"unknown stage for material closure: {stage}")


class Gate:
    """Exact stage gate with explicit checks, states, and byte provenance.

    ``check()`` remains source-compatible with the old call shape. Producers must
    now declare a non-empty required-ID set and provenance; otherwise finalization
    writes a non-green status.
    """

    def __init__(
        self,
        stage: str,
        required_ids: Iterable[str],
        *,
        script_path: Path | str | None = None,
        input_artifacts: Mapping[str, Path | str] | None = None,
        input_manifests: Mapping[str, Path | str] | None = None,
        source_files: Iterable[Path | str] = (),
    ) -> None:
        ids = tuple(required_ids)
        if not ids:
            raise GateDefinitionError(f"{stage}: required_ids must be non-empty")
        if len(ids) != len(set(ids)):
            raise GateDefinitionError(f"{stage}: required_ids contains duplicates")
        if any(not isinstance(gid, str) or not gid.strip() for gid in ids):
            raise GateDefinitionError(
                f"{stage}: required_ids must be non-empty strings"
            )

        self.stage = stage
        self.required_ids = frozenset(ids)
        self.checks: list[dict[str, Any]] = []
        self.script_path: Path | None = None
        self._source_paths: dict[str, Path] = {}
        self._input_paths: dict[str, Path] = {}
        self._predecessor_paths: dict[str, tuple[Path, str | None]] = {}

        if script_path is not None:
            self.record_script_hash(script_path)
        for path in source_files:
            self.record_source_hash(path)
        for label, path in (input_artifacts or {}).items():
            self.record_input_hash(label, path)
        for label, path in (input_manifests or {}).items():
            self.record_input_hash(label, path)

    @staticmethod
    def _path(path: Path | str) -> Path:
        return Path(path).expanduser().resolve()

    @staticmethod
    def _label(label: str) -> str:
        if not isinstance(label, str) or not label.strip():
            raise GateDefinitionError("provenance labels must be non-empty strings")
        return label

    def record_script_hash(self, path: Path | str) -> Gate:
        """Declare the generating script; bytes are hashed at ``write()`` time."""
        candidate = self._path(path)
        if self.script_path is not None and candidate != self.script_path:
            raise GateDefinitionError(
                f"{self.stage}: generating script already declared"
            )
        self.script_path = candidate
        return self

    def record_source_hash(
        self,
        path: Path | str,
        *,
        label: str | None = None,
        generating_script: bool = False,
    ) -> Gate:
        """Declare source provenance, optionally as the generating script."""
        if generating_script:
            return self.record_script_hash(path)
        candidate = self._path(path)
        source_label = self._label(label or str(candidate))
        existing = self._source_paths.get(source_label)
        if existing is not None and existing != candidate:
            raise GateDefinitionError(
                f"{self.stage}: source label {source_label!r} already declared"
            )
        self._source_paths[source_label] = candidate
        return self

    def record_input_hash(self, label: str, path: Path | str) -> Gate:
        """Declare a required input artifact; hash bytes at ``write()`` time."""
        input_label = self._label(label)
        candidate = self._path(path)
        existing = self._input_paths.get(input_label)
        if existing is not None and existing != candidate:
            raise GateDefinitionError(
                f"{self.stage}: input label {input_label!r} already declared"
            )
        self._input_paths[input_label] = candidate
        return self

    def record_predecessor_gate(
        self,
        label: str,
        path: Path | str,
        *,
        expected_stage: str | None = None,
    ) -> Gate:
        """Declare the prior gate whose file hash forms this gate's chain link."""
        predecessor_label = self._label(label)
        candidate = self._path(path)
        existing = self._predecessor_paths.get(predecessor_label)
        declared = (candidate, expected_stage)
        if existing is not None and existing != declared:
            raise GateDefinitionError(
                f"{self.stage}: predecessor label {predecessor_label!r} already declared"
            )
        self._predecessor_paths[predecessor_label] = declared
        return self

    def check(
        self,
        gid: str,
        ok: bool | None,
        detail: str,
        *,
        state: GateState | None = None,
        **metrics: Any,
    ) -> bool:
        """Emit exactly one declared check ID with an explicit semantic state."""
        seen = {check["id"] for check in self.checks}
        if gid in seen:
            raise GateDefinitionError(f"{self.stage}: duplicate check id {gid}")
        if gid not in self.required_ids:
            raise GateDefinitionError(f"{self.stage}: unexpected check id {gid}")
        if not isinstance(detail, str) or not detail.strip():
            raise GateDefinitionError(f"{self.stage}/{gid}: detail must be non-empty")

        if state is None:
            if ok is None:
                raise GateDefinitionError(
                    f"{self.stage}/{gid}: ok=None requires an explicit state"
                )
            state = "PASS" if bool(ok) else "FAIL"
        if state not in GATE_STATES:
            raise GateDefinitionError(f"{self.stage}/{gid}: invalid state {state!r}")
        if state == "PASS" and not (ok is not None and bool(ok)):
            raise GateDefinitionError(f"{self.stage}/{gid}: PASS requires ok=True")
        if state != "PASS" and ok is not None and bool(ok):
            raise GateDefinitionError(
                f"{self.stage}/{gid}: {state} cannot carry ok=True"
            )

        passed = state == "PASS"
        self.checks.append(
            {
                "id": gid,
                "state": state,
                "ok": passed,
                "detail": detail,
                "metrics": metrics,
                "evidence": metrics,
            }
        )
        print(f"  [{state}] {gid}: {detail}", flush=True)
        return passed

    def incomplete(self, gid: str, detail: str, **metrics: Any) -> bool:
        return self.check(gid, None, detail, state="INCOMPLETE", **metrics)

    def not_run(self, gid: str, detail: str, **metrics: Any) -> bool:
        return self.check(gid, None, detail, state="NOT_RUN", **metrics)

    @property
    def missing_ids(self) -> set[str]:
        return set(self.required_ids) - {check["id"] for check in self.checks}

    def _provenance_blockers(self) -> list[str]:
        blockers: list[str] = []
        if self.script_path is None:
            blockers.append("generating script not declared")
        elif not self.script_path.is_file():
            blockers.append(f"generating script absent: {self.script_path}")
        if not self._input_paths:
            blockers.append("no input artifacts declared")
        blockers.extend(
            f"input artifact absent: {label}={path}"
            for label, path in self._input_paths.items()
            if not path.exists()
        )
        blockers.extend(
            f"source absent: {label}={path}"
            for label, path in self._source_paths.items()
            if not path.exists()
        )
        for label, (path, expected_stage) in self._predecessor_paths.items():
            if not path.is_file():
                blockers.append(f"predecessor gate absent: {label}={path}")
                continue
            try:
                payload = read_json(path)
            except (OSError, ValueError, TypeError) as exc:
                blockers.append(f"predecessor gate unreadable: {label}: {exc}")
                continue
            if payload.get("status") != "PASS" or payload.get("ok") is not True:
                blockers.append(f"predecessor gate is not PASS: {label}")
            if expected_stage is not None and payload.get("stage") != expected_stage:
                blockers.append(
                    f"predecessor stage mismatch: {label} expected {expected_stage!r}"
                )
        return blockers

    @property
    def status(self) -> GateState:
        states = {check["state"] for check in self.checks}
        if not self.checks or self._provenance_blockers() or "NOT_RUN" in states:
            return "NOT_RUN"
        if self.missing_ids or "INCOMPLETE" in states:
            return "INCOMPLETE"
        if "FAIL" in states:
            return "FAIL"
        return "PASS"

    @property
    def ok(self) -> bool:
        return self.status == "PASS"

    def _provenance(self) -> dict[str, Any]:
        script = (
            hash_artifact(self.script_path)
            if self.script_path is not None
            else {"path": None, "kind": "missing", "sha256": None, "size_bytes": None}
        )
        sources = {
            label: hash_artifact(path)
            for label, path in sorted(self._source_paths.items())
        }
        inputs = {
            label: hash_artifact(path)
            for label, path in sorted(self._input_paths.items())
        }
        predecessors: dict[str, dict[str, Any]] = {}
        for label, (path, expected_stage) in sorted(self._predecessor_paths.items()):
            record = hash_artifact(path)
            record["expected_stage"] = expected_stage
            if path.is_file():
                try:
                    payload = read_json(path)
                except (OSError, ValueError, TypeError):
                    payload = {}
                record["stage"] = payload.get("stage")
                record["status"] = payload.get("status")
            predecessors[label] = record
        return {
            "script": script,
            "sources": sources,
            "input_artifacts": inputs,
            # Compatibility alias for the fix-plan schema. Both names contain
            # the same immutable-at-write hash records.
            "input_manifests": inputs,
            "predecessor_gates": predecessors,
            "python_executable": sys.executable,
        }

    def write(
        self,
        run_dir: Path,
        *,
        fail_hard: bool = True,
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        """Hash provenance, atomically write the gate, and enforce non-PASS."""
        status = self.status
        non_pass = [check for check in self.checks if check["state"] != "PASS"]
        result = {
            "stage": self.stage,
            "status": status,
            "ok": status == "PASS",
            "required_ids": sorted(self.required_ids),
            "missing_ids": sorted(self.missing_ids),
            "n_checks": len(self.checks),
            "failed": [
                check["id"] for check in self.checks if check["state"] == "FAIL"
            ],
            "non_pass_ids": [check["id"] for check in non_pass],
            "provenance_blockers": self._provenance_blockers(),
            "provenance": self._provenance(),
            "checks": self.checks,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        }
        out = output_path or (Path(run_dir) / "gates" / f"{self.stage}.json")
        write_json(out, result)
        if status != "PASS" and fail_hard:
            reasons = (
                [f"missing IDs: {sorted(self.missing_ids)}"] if self.missing_ids else []
            )
            reasons.extend(result["provenance_blockers"])
            reasons.extend(f"{check['id']}: {check['detail']}" for check in non_pass)
            raise GateFailure(
                f"GATE {status} [{self.stage}] " + ("; ".join(reasons) or "not PASS")
            )
        return result


def initialize_not_run_gate(
    run_dir: Path,
    stage: str,
    reason: str,
    *,
    script_path: Path | str,
) -> dict[str, Any]:
    """Write a complete-ID red placeholder used to invalidate stale gates."""
    gate = Gate(
        stage,
        required_check_ids(stage),
        script_path=script_path,
        input_artifacts={"not_run_prerequisite": Path(run_dir) / ".not_run"},
    )
    for gid in sorted(gate.required_ids):
        gate.not_run(gid, reason)
    return gate.write(run_dir, fail_hard=False)


def assert_no_test_leakage(gate: Gate, images_dir: Path, gid: str = "G0.2") -> bool:
    """G0.2 -- re-run at EVERY stage.

    The test video shares a camera and a resolution with both build videos, so by
    the time frames are on disk nothing distinguishes them. This is the only
    check that can catch a leak, so it is cheap and it runs everywhere.
    """
    if not images_dir.is_dir():
        return gate.not_run(gid, f"required image root absent: {images_dir}")
    hits = [
        str(p.relative_to(images_dir))
        for p in images_dir.rglob("*")
        if p.is_file() and any(s in p.as_posix() for s in FORBIDDEN_STEMS)
    ]
    stray_seq = [
        d.name
        for d in images_dir.iterdir()
        if d.is_dir() and d.name not in BUILD_BY_SEQ
    ]
    return gate.check(
        gid,
        not hits and not stray_seq,
        (
            f"no test-video frames and no unknown sequences under {images_dir}"
            if not hits and not stray_seq
            else f"LEAK: test frames={hits[:3]} unknown_sequences={stray_seq}"
        ),
        test_frame_hits=len(hits),
        unknown_sequences=stray_seq,
    )


# ---------------------------------------------------------------- io helpers
def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    tmp.rename(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    """Return the SHA-256 of one regular file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash a directory deterministically from relative paths and file hashes."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")
    digest = hashlib.sha256()
    for child in sorted(
        (p for p in root.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(root).as_posix(),
    ):
        rel = child.relative_to(root).as_posix().encode("utf-8")
        child_hash = sha256(child).encode("ascii")
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        digest.update(child_hash)
    return digest.hexdigest()


def hash_artifact(path: Path | str) -> dict[str, Any]:
    """Return a JSON-safe hash record for a file, directory, or absent path."""
    artifact = Path(path).expanduser().resolve()
    if artifact.is_file():
        return {
            "path": str(artifact),
            "kind": "file",
            "sha256": sha256(artifact),
            "size_bytes": artifact.stat().st_size,
        }
    if artifact.is_dir():
        files = [child for child in artifact.rglob("*") if child.is_file()]
        return {
            "path": str(artifact),
            "kind": "directory",
            "sha256": sha256_tree(artifact),
            "size_bytes": sum(child.stat().st_size for child in files),
            "n_files": len(files),
        }
    return {
        "path": str(artifact),
        "kind": "missing",
        "sha256": None,
        "size_bytes": None,
    }


def source_sha256(path: Path | str) -> str:
    """Hash source bytes; named wrapper makes lineage call sites explicit."""
    return sha256(Path(path))


def input_artifact_sha256(path: Path | str) -> str:
    """Hash a required input file or deterministic directory tree."""
    record = hash_artifact(path)
    if record["sha256"] is None:
        raise FileNotFoundError(f"input artifact absent: {record['path']}")
    return str(record["sha256"])


def predecessor_gate_hash(path: Path | str, *, require_pass: bool = True) -> str:
    """Hash a predecessor gate, optionally requiring release-green semantics."""
    gate_path = Path(path).expanduser().resolve()
    payload = read_json(gate_path)
    if require_pass and (
        payload.get("status") != "PASS" or payload.get("ok") is not True
    ):
        raise GateFreshnessError(f"predecessor gate is not PASS: {gate_path}")
    return sha256(gate_path)


def make_lineage_record(
    source_path: Path | str,
    *,
    source_rel: str,
    artifact_path: Path | str | None = None,
    artifact_field: str = "artifact_sha256",
) -> dict[str, str]:
    """Build the standard source lineage fields, plus an optional output hash."""
    if not source_rel or Path(source_rel).is_absolute():
        raise ValueError("source_rel must be a non-empty relative path")
    record = {
        "source_rel": Path(source_rel).as_posix(),
        "source_sha256": source_sha256(source_path),
    }
    if artifact_path is not None:
        if not artifact_field or not artifact_field.endswith("_sha256"):
            raise ValueError("artifact_field must be a non-empty *_sha256 name")
        record[artifact_field] = input_artifact_sha256(artifact_path)
    return record


def _load_gate(
    gate: Path | str | Mapping[str, Any],
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(gate, Mapping):
        return dict(gate), None
    path = Path(gate).expanduser().resolve()
    return read_json(path), path


def assert_gate_fresh(gate: Path | str | Mapping[str, Any]) -> bool:
    """Re-hash every recorded provenance item and reject missing/drifted bytes."""
    payload, gate_path = _load_gate(gate)
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise GateFreshnessError(
            f"gate has no provenance object: {gate_path or '<memory>'}"
        )

    records: list[tuple[str, Any]] = [("script", provenance.get("script"))]
    for bucket_name in ("sources", "input_artifacts", "predecessor_gates"):
        bucket = provenance.get(bucket_name, {})
        if not isinstance(bucket, dict):
            raise GateFreshnessError(f"invalid provenance bucket: {bucket_name}")
        records.extend(
            (f"{bucket_name}.{label}", record) for label, record in bucket.items()
        )

    for label, record in records:
        if not isinstance(record, dict) or not record.get("path"):
            raise GateFreshnessError(f"missing provenance path: {label}")
        expected = record.get("sha256")
        if not expected:
            raise GateFreshnessError(f"missing recorded sha256: {label}")
        actual = hash_artifact(record["path"])
        if actual["sha256"] != expected:
            raise GateFreshnessError(
                f"sha256 drift for {label}: expected {expected}, got {actual['sha256']}"
            )
    return True


def verify_predecessor_chain(gate_paths: Sequence[Path | str]) -> bool:
    """Verify ordered gate files are linked by exact predecessor file hashes."""
    if len(gate_paths) < 2:
        return True
    resolved = [Path(path).expanduser().resolve() for path in gate_paths]
    payloads = [read_json(path) for path in resolved]
    for index in range(1, len(resolved)):
        previous_path = resolved[index - 1]
        previous_payload = payloads[index - 1]
        current_payload = payloads[index]
        predecessors = current_payload.get("provenance", {}).get(
            "predecessor_gates", {}
        )
        matching = [
            (label, record)
            for label, record in predecessors.items()
            if isinstance(record, dict)
            and Path(record.get("path", "")).expanduser().resolve() == previous_path
        ]
        if len(matching) != 1:
            raise GateFreshnessError(
                f"missing unique predecessor link: {previous_path} -> {resolved[index]}"
            )
        label, record = matching[0]
        actual_hash = sha256(previous_path)
        if record.get("sha256") != actual_hash:
            raise GateFreshnessError(
                f"predecessor hash mismatch for {label}: "
                f"expected {record.get('sha256')}, got {actual_hash}"
            )
        expected_stage = record.get("expected_stage")
        if (
            expected_stage is not None
            and previous_payload.get("stage") != expected_stage
        ):
            raise GateFreshnessError(
                f"predecessor stage mismatch for {label}: expected {expected_stage!r}, "
                f"got {previous_payload.get('stage')!r}"
            )
        if (
            previous_payload.get("status") != "PASS"
            or previous_payload.get("ok") is not True
        ):
            raise GateFreshnessError(f"predecessor gate is not PASS: {previous_path}")
    return True


def ffprobe(path: Path) -> dict[str, Any]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration,codec_name",
            "-show_entries",
            "format_tags=make,model,creation_time",
            "-show_entries",
            "stream_tags=handler_name",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    d = json.loads(out.stdout)
    st = d["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    return {
        "width": int(st["width"]),
        "height": int(st["height"]),
        "fps": float(num) / float(den),
        "nb_frames": int(st.get("nb_frames", 0) or 0),
        "duration": float(st.get("duration", 0) or 0),
        "codec": st.get("codec_name", ""),
        "tags": {**d.get("format", {}).get("tags", {}), **st.get("tags", {})},
    }


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
