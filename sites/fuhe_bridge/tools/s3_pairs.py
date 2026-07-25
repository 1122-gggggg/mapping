#!/usr/bin/env python3
"""S3 -- natural retrieval first, with conditional deterministic gap bridges.

The forced grid only proposes forward/reverse candidates; two-view geometry and
Doppelgangers still decide which survive. This stage proves the candidates cross
the real gluemap loader boundary before matching, using the production absolute
``images_path`` and relative ``extra_pairs_path`` names.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ts_common import (  # noqa: E402
    BUILD,
    GLUEMAP_PY,
    GLUEMAP_REPO,
    PROBE_FRAME_COUNT,
    RUNS,
    RUN_ID,
    SUBFOLDER_REGEX,
    TEST,
    Gate,
    log,
    read_json,
    required_check_ids,
    sha256,
    stage_material_artifacts,
    write_json,
)
from ts_env import verify_pycolmap_runtime  # noqa: E402
from resource_guard import GLOBAL_HEAVY_LOCK, contract_from_config  # noqa: E402


TS_COMMON = Path(__file__).with_name("ts_common.py")
TS_ENV = Path(__file__).with_name("ts_env.py")
MEMORY_SAFE_LAUNCHER = Path(__file__).with_name("run_gluemap_memory_safe.py")
GLUEMAP_LOADER = GLUEMAP_REPO / "gluemap" / "datasets" / "multi_sequence_twoview.py"
GLUEMAP_COLMAP = GLUEMAP_REPO / "gluemap" / "utils" / "colmap.py"

FWD_STRIDE = 16
REV_STRIDE = 12
ENDPOINT_FRAC = 0.08
ENDPOINT_STRIDE = 3
EXPECTED_SEQUENCE_PAIRS = sum(v.direction == "fwd" for v in BUILD) * sum(
    v.direction == "rev" for v in BUILD
)
EXPECTED_IMAGES = PROBE_FRAME_COUNT
EXPECTED_CAMERAS = 1
MIN_PAIR_DENSITY = 4.0
MIN_ROUTE_SEPARATION = 0.25
REQUIRED_ROUTE_CLUSTERS = 2
MAX_GAP_PAIRS = 12_000


def frames_by_seq(run_dir: Path) -> dict[str, list[dict]]:
    manifest = read_json(run_dir / "frame_manifest.json")
    output: dict[str, list[dict]] = {}
    for frame in manifest["frames"]:
        output.setdefault(frame["seq"], []).append(frame)
    for frames in output.values():
        frames.sort(key=lambda frame: frame["t"])
    return output


def sample(
    frames: list[dict], stride: int, endpoint_stride: int, endpoint_frac: float
) -> list[dict]:
    """Uniform stride, densified at both ends of a sequence."""
    n_frames = len(frames)
    endpoint_n = max(int(n_frames * endpoint_frac), 1)
    indexes = set(range(0, n_frames, stride))
    indexes |= set(range(0, endpoint_n, endpoint_stride))
    indexes |= set(
        range(max(n_frames - endpoint_n, 0), n_frames, endpoint_stride)
    )
    return [frames[index] for index in sorted(indexes)]


def parse_pair_file(path: Path) -> set[tuple[str, str]]:
    """Parse, normalize, and reject malformed/duplicate/self pair records."""
    pairs: set[tuple[str, str]] = set()
    seen_lines = 0
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{lineno}: expected exactly two image names")
        a, b = (Path(field).as_posix() for field in fields)
        if Path(a).is_absolute() or Path(b).is_absolute() or a == b:
            raise ValueError(f"{path}:{lineno}: invalid relative undirected pair")
        seen_lines += 1
        pairs.add(tuple(sorted((a, b))))
    if len(pairs) != seen_lines:
        raise ValueError(f"{path}: contains duplicate forced pairs")
    return pairs


def _has_independent_route_clusters(
    clusters: list[dict[str, float]], minimum_separation: float
) -> bool:
    for index, first in enumerate(clusters):
        for second in clusters[index + 1 :]:
            if (
                abs(float(first["fwd_normalized"]) - float(second["fwd_normalized"]))
                >= minimum_separation
                and abs(float(first["rev_normalized"]) - float(second["rev_normalized"]))
                >= minimum_separation
            ):
                return True
    return False


def decide_gap_bridge_policy(
    fwd: list[str],
    rev: list[str],
    connectivity_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    """Force only sequence routes lacking two independently separated clusters."""
    combinations = [f"{a}|{b}" for a in sorted(fwd) for b in sorted(rev)]
    evidence = connectivity_evidence or {}
    missing: list[str] = []
    pending: list[str] = []
    for combination in combinations:
        record = evidence.get(combination)
        if not isinstance(record, dict):
            pending.append(combination)
            continue
        clusters = record.get("route_clusters", [])
        if not isinstance(clusters, list) or not _has_independent_route_clusters(
            clusters, MIN_ROUTE_SEPARATION
        ):
            missing.append(combination)
    return {
        "mode": "natural_retrieval_first",
        "natural_sources": ["global_retrieval", "sequential_neighbors"],
        "expected_sequence_pairs": combinations,
        "required_route_clusters": REQUIRED_ROUTE_CLUSTERS,
        "minimum_normalized_separation": MIN_ROUTE_SEPARATION,
        "missing_sequence_pairs": missing,
        "pending_sequence_pairs": pending,
        "force_gap_bridges": bool(missing),
        "applicable": not pending,
        "reason": (
            "conditional gaps identified from post-retrieval route-cluster evidence"
            if missing
            else (
                "awaiting natural retrieval evidence; no gap pairs are injected"
                if pending
                else "natural retrieval proves two independent route clusters for every direction pair"
            )
        ),
    }


def deterministic_gap_pairs(
    frames: dict[str, list[dict]],
    missing_sequence_pairs: list[str],
    *,
    max_pairs: int = MAX_GAP_PAIRS,
) -> set[tuple[str, str]]:
    """Build a deterministic, route-spread pair grid with a global hard cap."""
    if type(max_pairs) is not int or max_pairs <= 0 or max_pairs > MAX_GAP_PAIRS:
        raise ValueError(f"max_pairs must be in [1, {MAX_GAP_PAIRS}]")
    combinations = sorted(set(missing_sequence_pairs))
    if not combinations:
        return set()
    base, extra = divmod(max_pairs, len(combinations))
    output: set[tuple[str, str]] = set()
    for combo_index, combination in enumerate(combinations):
        try:
            fwd_seq, rev_seq = combination.split("|", 1)
            left = sorted(frames[fwd_seq], key=lambda frame: (frame["t"], frame["name"]))
            right = sorted(frames[rev_seq], key=lambda frame: (frame["t"], frame["name"]))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid missing sequence pair {combination!r}") from exc
        candidates = [
            tuple(sorted((first["name"], second["name"])))
            for first in left
            for second in right
        ]
        allocation = min(len(candidates), base + (combo_index < extra))
        if allocation == len(candidates):
            selected = candidates
        else:
            indexes = np.linspace(0, len(candidates) - 1, allocation, dtype=int)
            selected = [candidates[int(index)] for index in indexes]
        output.update(selected)
    if len(output) > max_pairs:
        raise AssertionError("deterministic gap-pair allocator exceeded its cap")
    return output


def build_gluemap_config(run_dir: Path, forced_path: Path) -> dict[str, Any]:
    """Return a clean-run, one-camera GlueMap probe config."""
    root = Path(run_dir).resolve()
    return {
        "chosen_model": "pi3",
        "path_feedforward": str(GLUEMAP_REPO / "checkpoints" / "pi3.safetensors"),
        "path_retrieval": str(GLUEMAP_REPO / "checkpoints" / "dino_salad.ckpt"),
        "path_tracker": str(GLUEMAP_REPO / "checkpoints" / "vggsfm_v2_0_0_track_predictor.bin"),
        "path_dg": str(GLUEMAP_REPO / "checkpoints" / "checkpoint-dg+visym.pth"),
        "images_path": str(root / "images"),
        "write_path": str(root / "gluemap"),
        "temp_path": str(root / "tmp"),
        "chosen_output": "gluemap_aba",
        "camera_model": "PINHOLE",
        "intrinsics_mode": "SHARED",
        "use_gt_intrinsics": True,
        "gt_intrinsics_path": str(root / "intrinsics_seed"),
        "refine_intrinsics": False,
        "skip_doppelgangers": False,
        "valid_dg_threshold": 0.8,
        "is_multi_sequence": True,
        "subfolder_regex": SUBFOLDER_REGEX,
        "is_sequential": True,
        "sample_frequency": 1,
        "num_neighbors": 100,
        "num_neighbors_sequential": 30,
        "extra_pairs_path": str(Path(forced_path).resolve()),
        "num_track_per_img": 512,
        "sift_max_num_features": 2048,
        # max_num_features caps SIFT locations, not database rows.  One
        # orientation per location makes 2048 an actual keypoint/descriptor-row cap.
        "sift_max_num_orientations": 1,
        "max_num_tracks": 400000,
        "memory_safe_launcher": str(MEMORY_SAFE_LAUNCHER.resolve()),
        "resource_lock_path": str(GLOBAL_HEAVY_LOCK),
        "resource_guard_path": str(Path(__file__).with_name("resource_guard.py").resolve()),
        "resource_guard_log_path": str(root / "logs" / "resource_guard.log"),
        "min_track_length": 3,
        "valid_pose_threshold": 0.05,
        "batch_size": 8,
        "retrieval_batch_size": 30,
        "num_workers": 0,
        "coarse_only": False,
        "use_dummy_tracks": False,
    }


def _empty_retrieval_method(self, args, dataset, image_list):
    """Keep real discovery/pair construction but avoid SALAD descriptor work."""
    from gluemap.datasets.utils import establish_neighbors_sequential

    neighbors = establish_neighbors_sequential(
        image_list, num_neighbors=args.num_neighbors_sequential
    )
    return neighbors, None


def real_loader_probe(run_dir: Path, output_path: Path) -> dict[str, Any]:
    """Instantiate the live gluemap loader and persist pre-matching evidence."""
    # Import from the live checkout only. This subprocess runs under GLUEMAP_PY.
    sys.path.insert(0, str(GLUEMAP_REPO))
    import pycolmap
    import gluemap.datasets.multi_sequence_twoview as loader_module
    from gluemap.utils.colmap import extract_gt_intrinsics

    config = yaml.safe_load((run_dir / "gluemap_config.yaml").read_text(encoding="utf-8"))
    images_root = Path(config["images_path"]).resolve()
    pattern = re.compile(config["subfolder_regex"])
    datasets = [
        name
        for name in sorted(os.listdir(images_root))
        if (images_root / name).is_dir() and pattern.match(name)
    ]
    args = SimpleNamespace(**config)
    args.images_path = str(images_root)
    args.curr_processed = str(Path(config["write_path"]).resolve())
    args.curr_path = args.curr_processed

    original_retrieval = loader_module.retrieve_global_neighbors
    original_method = loader_module.MultiSequencePairs._retrieve_descriptors_and_neighbors
    try:
        loader_module.retrieve_global_neighbors = (
            lambda _args, _neighbors, _descriptors: np.empty((0, 2), dtype=np.int64)
        )
        loader_module.MultiSequencePairs._retrieve_descriptors_and_neighbors = (
            _empty_retrieval_method
        )
        dataset = loader_module.MultiSequencePairs(args, datasets)
    finally:
        loader_module.retrieve_global_neighbors = original_retrieval
        loader_module.MultiSequencePairs._retrieve_descriptors_and_neighbors = original_method

    gt_intrinsics = extract_gt_intrinsics(
        config["gt_intrinsics_path"],
        dataset.images_list,
        dataset.intrinsics_mapping,
    )
    non_null_cameras = sum(value is not None for value in gt_intrinsics)

    seed = pycolmap.Reconstruction(str(Path(config["gt_intrinsics_path"]).resolve()))
    seed_image_by_name = {image.name: image for image in seed.images.values()}
    dimension_mismatches: list[dict[str, Any]] = []
    missing_seed_names: list[str] = []
    for index, name in enumerate(dataset.images_list):
        seed_image = seed_image_by_name.get(name)
        if seed_image is None:
            missing_seed_names.append(name)
            continue
        camera = seed.cameras[seed_image.camera_id]
        disk_h, disk_w = dataset.images_shape_ori[index]
        if (disk_w, disk_h) != (camera.width, camera.height):
            dimension_mismatches.append(
                {
                    "name": name,
                    "disk": [disk_w, disk_h],
                    "seed": [camera.width, camera.height],
                }
            )

    forced_names = parse_pair_file(Path(config["extra_pairs_path"]))
    index_of = {Path(name).as_posix(): index for index, name in enumerate(dataset.images_list)}
    missing_forced_endpoints = sorted(
        {name for pair in forced_names for name in pair if name not in index_of}
    )
    expected_indexes = {
        tuple(sorted((index_of[a], index_of[b])))
        for a, b in forced_names
        if a in index_of and b in index_of
    }
    loaded_indexes = {
        tuple(sorted((int(pair[0]), int(pair[1])))) for pair in np.asarray(dataset.pairs)
    }
    injected = expected_indexes & loaded_indexes
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "loader_source": str(GLUEMAP_LOADER.resolve()),
        "loader_source_sha256": sha256(GLUEMAP_LOADER),
        "colmap_source": str(GLUEMAP_COLMAP.resolve()),
        "colmap_source_sha256": sha256(GLUEMAP_COLMAP),
        "images_path": str(images_root),
        "images_path_is_absolute": images_root.is_absolute(),
        "datasets": datasets,
        "n_images": len(dataset.images_list),
        "n_unique_images": len(set(dataset.images_list)),
        "n_gt_camera_slots": len(gt_intrinsics),
        "n_non_null_gt_cameras": non_null_cameras,
        "n_dimension_matches": len(dataset.images_list)
        - len(missing_seed_names)
        - len(dimension_mismatches),
        "missing_seed_names": missing_seed_names,
        "dimension_mismatches": dimension_mismatches,
        "n_loader_pairs": len(loaded_indexes),
        "pair_density": len(loaded_indexes) / max(len(dataset.images_list), 1),
        "n_forced_expected": len(expected_indexes),
        "n_forced_injected": len(injected),
        "missing_forced_endpoints": missing_forced_endpoints,
        "missing_forced_pairs": len(expected_indexes - loaded_indexes),
    }
    write_json(output_path, payload)
    return payload


def run_probe_subprocess(run_dir: Path, output_path: Path) -> tuple[dict | None, str]:
    """Run the real loader under the pinned interpreter, never a stale copy."""
    output_path.unlink(missing_ok=True)
    process = subprocess.run(
        [
            str(GLUEMAP_PY),
            str(Path(__file__).resolve()),
            "--run-name",
            run_dir.name,
            "--internal-loader-probe",
            "--probe-output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    log_path = run_dir / "s3_loader_probe.log"
    log_path.write_text(process.stdout + process.stderr, encoding="utf-8")
    if process.returncode != 0 or not output_path.is_file():
        return None, (
            f"real loader probe failed with exit {process.returncode}; see {log_path}"
        )
    return read_json(output_path), "real loader probe completed"


def source_lineage_evidence(
    corpus_manifest: dict, frame_manifest: dict
) -> tuple[bool, dict[str, Any]]:
    locked = {
        (
            record.get("source_rel", record.get("rel")),
            record.get("source_sha256", record.get("sha256")),
        )
        for record in corpus_manifest.get("build", [])
    }
    actual = {
        (frame.get("source_rel"), frame.get("source_sha256"))
        for frame in frame_manifest.get("frames", [])
    }
    heldout_paths = {video.rel for video in TEST}
    heldout_sequences = {video.seq for video in TEST}
    forbidden = sorted(
        {
            str(frame.get("source_rel"))
            for frame in frame_manifest.get("frames", [])
            if frame.get("source_rel") in heldout_paths
            or frame.get("seq") in heldout_sequences
        }
    )
    expected = {
        (
            video.rel,
            next(
                (
                    record.get("source_sha256", record.get("sha256"))
                    for record in corpus_manifest.get("build", [])
                    if record.get("seq") == video.seq
                ),
                None,
            ),
        )
        for video in BUILD
    }
    ok = locked == expected == actual and len(actual) == len(BUILD) and not forbidden
    return ok, {
        "expected_lineage": sorted(expected, key=repr),
        "actual_lineage": sorted(actual, key=repr),
        "forbidden_lineage": forbidden,
    }


def _emit_probe_checks(
    gate: Gate,
    payload: dict | None,
    error: str,
    policy: dict[str, Any],
) -> None:
    if payload is None:
        for gid in ("G3.1", "G3.2", "G3.3", "G3.4", "G3.5d"):
            gate.incomplete(gid, error)
        return

    gate.check(
        "G3.1",
        payload["n_gt_camera_slots"] == EXPECTED_CAMERAS
        and payload["n_non_null_gt_cameras"] == EXPECTED_CAMERAS,
        f"real extract_gt_intrinsics returned {payload['n_non_null_gt_cameras']}/"
        f"{payload['n_gt_camera_slots']} non-null cameras",
        n_camera_slots=payload["n_gt_camera_slots"],
        n_non_null=payload["n_non_null_gt_cameras"],
    )
    gate.check(
        "G3.2",
        payload["n_images"] == EXPECTED_IMAGES
        and payload["n_unique_images"] == EXPECTED_IMAGES
        and payload["n_dimension_matches"] == EXPECTED_IMAGES
        and not payload["missing_seed_names"]
        and not payload["dimension_mismatches"],
        f"seed/disk dimensions match {payload['n_dimension_matches']}/"
        f"{payload['n_images']} real-loader images",
        n_images=payload["n_images"],
        n_unique_images=payload["n_unique_images"],
        missing_seed_names=payload["missing_seed_names"],
        dimension_mismatches=payload["dimension_mismatches"],
    )
    forced_density = payload["n_forced_injected"] / max(payload["n_images"], 1)
    density_metrics = {
        "policy_mode": policy.get("mode"),
        "force_gap_bridges": policy.get("force_gap_bridges"),
        "retrieval_executed": bool(policy.get("applicable")),
        "n_forced_pairs": payload["n_forced_injected"],
        "n_images": payload["n_images"],
        "conditional_pair_density": forced_density,
        "minimum": MIN_PAIR_DENSITY,
        "missing_sequence_pairs": policy.get("missing_sequence_pairs", []),
        "pending_sequence_pairs": policy.get("pending_sequence_pairs", []),
        "required_route_clusters": policy.get("required_route_clusters"),
        "minimum_normalized_separation": policy.get(
            "minimum_normalized_separation"
        ),
    }
    if policy.get("force_gap_bridges") is False and not payload["n_forced_injected"]:
        gate.not_applicable(
            "G3.3",
            "conditional forced-pair density does not apply; natural retrieval "
            "has not been represented by this loader contract probe",
            **density_metrics,
        )
    else:
        cluster_policy_valid = (
            policy.get("mode") == "natural_retrieval_first"
            and policy.get("force_gap_bridges") is True
            and policy.get("applicable") is True
            and not policy.get("pending_sequence_pairs")
            and bool(policy.get("missing_sequence_pairs"))
            and policy.get("required_route_clusters") == REQUIRED_ROUTE_CLUSTERS
            and policy.get("minimum_normalized_separation")
            == MIN_ROUTE_SEPARATION
        )
        gate.check(
            "G3.3",
            cluster_policy_valid and forced_density >= MIN_PAIR_DENSITY,
            f"conditional forced-pair density {forced_density:.3f} pairs/frame; "
            f"route-cluster policy valid={cluster_policy_valid}",
            cluster_policy_valid=cluster_policy_valid,
            **density_metrics,
        )
    expected_sequences = {video.seq for video in BUILD}
    gate.check(
        "G3.4",
        set(payload["datasets"]) == expected_sequences,
        f"real CLI-equivalent discovery found {len(payload['datasets'])} sequences",
        expected=sorted(expected_sequences),
        discovered=payload["datasets"],
    )
    gate.check(
        "G3.5d",
        payload["images_path_is_absolute"]
        and payload["n_forced_injected"] == payload["n_forced_expected"]
        and payload["missing_forced_pairs"] == 0
        and not payload["missing_forced_endpoints"],
        f"real loader injected {payload['n_forced_injected']}/"
        f"{payload['n_forced_expected']} relative forced pairs before matching",
        images_path=payload["images_path"],
        expected=payload["n_forced_expected"],
        injected=payload["n_forced_injected"],
        missing_pairs=payload["missing_forced_pairs"],
        missing_endpoints=payload["missing_forced_endpoints"],
    )


def stage_gate(run_dir: Path) -> Gate:
    gate = Gate(
        "S3_pairs",
        required_check_ids("S3_pairs"),
        script_path=Path(__file__),
        input_artifacts=stage_material_artifacts("S3_pairs", run_dir),
        source_files=[
            TS_COMMON,
            TS_ENV,
            GLUEMAP_LOADER,
            GLUEMAP_COLMAP,
            MEMORY_SAFE_LAUNCHER,
        ],
    )
    gate.record_predecessor_gate(
        "S2b_intrinsics",
        run_dir / "gates" / "S2b_intrinsics.json",
        expected_stage="S2b_intrinsics",
    )
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=RUN_ID)
    parser.add_argument("--candidate", default="fuhe_v2_fixed", choices=["fuhe_v2_fixed"])
    parser.add_argument("--max-pairs", type=int, default=MAX_GAP_PAIRS)
    parser.add_argument(
        "--connectivity-evidence",
        type=Path,
        help="post-retrieval route-cluster JSON; absent means natural-only probe",
    )
    parser.add_argument("--internal-loader-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Mandatory at both public S3 startup and the real-loader child startup.
    runtime = verify_pycolmap_runtime()
    run_dir = RUNS / args.run_name
    if args.internal_loader_probe:
        if args.probe_output is None:
            raise SystemExit("--internal-loader-probe requires --probe-output")
        real_loader_probe(run_dir, args.probe_output)
        return

    motion_path = run_dir / "motion_manifest.json"
    frame_path = run_dir / "frame_manifest.json"
    corpus_path = run_dir / "corpus_manifest.json"
    motion = read_json(motion_path)
    frame_manifest = read_json(frame_path)
    corpus_manifest = read_json(corpus_path)

    direction = {video.seq: video.direction for video in BUILD}
    for seq, record in motion.get("directions", {}).items():
        direction[seq] = record["direction"]
    fwd = sorted(seq for seq, value in direction.items() if value == "fwd")
    rev = sorted(seq for seq, value in direction.items() if value == "rev")
    log(f"forward: {fwd}")
    log(f"reverse: {rev}")

    by_seq = frames_by_seq(run_dir)
    connectivity = None
    if args.connectivity_evidence is not None:
        payload = read_json(args.connectivity_evidence)
        connectivity = payload.get("sequence_pairs", payload)
        if not isinstance(connectivity, dict):
            raise SystemExit("connectivity evidence must contain a sequence-pair mapping")
    policy = decide_gap_bridge_policy(fwd, rev, connectivity)
    pairs = deterministic_gap_pairs(
        by_seq, policy["missing_sequence_pairs"], max_pairs=args.max_pairs
    )
    per_pair = {
        combination: {
            "n_pairs": sum(
                {first.split("/", 1)[0], second.split("/", 1)[0]}
                == set(combination.split("|", 1))
                for first, second in pairs
            )
        }
        for combination in policy["expected_sequence_pairs"]
    }
    forced_path = run_dir / "forced_bridges.txt"
    forced_path.write_text(
        "# Natural retrieval + sequential neighbors run first.\n"
        "# Conditional deterministic gap candidates; geometry and DG still filter.\n"
        + "".join(f"{a} {b}\n" for a, b in sorted(pairs)),
        encoding="utf-8",
    )
    parsed_pairs = parse_pair_file(forced_path)

    config = build_gluemap_config(run_dir, forced_path)
    config_path = run_dir / "gluemap_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    resource_contract = contract_from_config(config, run_dir)
    write_json(run_dir / "resource_contract.json", resource_contract)

    lineage_ok, lineage_metrics = source_lineage_evidence(
        corpus_manifest, frame_manifest
    )
    forced_manifest_path = run_dir / "forced_bridges.json"
    forced_manifest = {
        "schema_version": "fuhe-pair-policy-v2",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "fwd": fwd,
        "rev": rev,
        "policy": policy,
        "connectivity_evidence": (
            str(args.connectivity_evidence.resolve())
            if args.connectivity_evidence is not None
            else None
        ),
        "n_pairs": len(pairs),
        "per_video_pair": per_pair,
        "forced_pairs_path": str(forced_path.resolve()),
        "forced_pairs_sha256": sha256(forced_path),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "source_lineage": lineage_metrics["actual_lineage"],
    }
    write_json(forced_manifest_path, forced_manifest)

    probe_path = run_dir / "s3_loader_probe.json"
    probe, probe_detail = run_probe_subprocess(run_dir, probe_path)

    gate = stage_gate(run_dir)
    gate.check(
        "G0.2",
        lineage_ok,
        f"exact {len(BUILD)} locked source rel/hash pairs; no held-out lineage"
        if lineage_ok else "frame source lineage is not the exact S0 build lock",
        **lineage_metrics,
    )
    gate.check(
        "G3.0",
        runtime.get("version") == "4.0.4"
        and runtime.get("providers")
        == [{"name": "pycolmap-cuda12", "version": "4.0.4"}],
        "pinned pycolmap runtime fingerprint verified",
        runtime_fingerprint=runtime,
    )
    _emit_probe_checks(gate, probe, probe_detail, policy)

    expected_sequences = {video.seq for video in BUILD}
    gate.check(
        "G3.5a",
        set(fwd) | set(rev) == expected_sequences
        and set(fwd).isdisjoint(rev)
        and len(fwd) == sum(video.direction == "fwd" for video in BUILD)
        and len(rev) == sum(video.direction == "rev" for video in BUILD),
        f"all {len(BUILD)} directions resolved exactly once: {len(fwd)} fwd/{len(rev)} rev",
        fwd=fwd,
        rev=rev,
        expected=sorted(expected_sequences),
    )
    gate.check(
        "G3.5b",
        len(pairs) == len(parsed_pairs)
        and len(per_pair) == EXPECTED_SEQUENCE_PAIRS
        and len(pairs) <= args.max_pairs
        and policy["mode"] == "natural_retrieval_first"
        and set(policy["missing_sequence_pairs"]) <= set(policy["expected_sequence_pairs"]),
        f"natural-first policy emitted {len(pairs)} conditional pairs across {len(per_pair)} direction combinations",
        n_pairs=len(pairs),
        n_parsed=len(parsed_pairs),
        n_sequence_pairs=len(per_pair),
        max_pairs=args.max_pairs,
        per_video_pair=per_pair,
        policy=policy,
    )
    known = {frame["name"] for frames in by_seq.values() for frame in frames}
    missing_manifest = sorted({name for pair in parsed_pairs for name in pair if name not in known})
    missing_disk = sorted(
        {
            name
            for pair in parsed_pairs
            for name in pair
            if not (run_dir / "images" / name).is_file()
        }
    )
    gate.check(
        "G3.5c",
        not missing_manifest and not missing_disk,
        "every relative forced-pair endpoint exists in the frame manifest and on disk",
        missing_manifest=missing_manifest,
        missing_disk=missing_disk,
    )

    persisted_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    injected = probe is not None and probe.get("n_forced_injected") == len(parsed_pairs)
    gate.check(
        "G3.6",
        persisted_config.get("skip_doppelgangers") is False
        and persisted_config.get("refine_intrinsics") is False
        and type(persisted_config.get("num_workers")) is int
        and persisted_config["num_workers"] == 0
        and persisted_config.get("num_track_per_img") == 512
        and persisted_config.get("sift_max_num_features") == 2048
        and type(persisted_config.get("sift_max_num_orientations")) is int
        and persisted_config["sift_max_num_orientations"] == 1
        and persisted_config.get("max_num_tracks") == 400000
        and Path(persisted_config.get("memory_safe_launcher", "")).resolve()
        == MEMORY_SAFE_LAUNCHER.resolve()
        and MEMORY_SAFE_LAUNCHER.is_file()
        and Path(persisted_config.get("extra_pairs_path", "")).resolve()
        == forced_path.resolve()
        and "rerun_from" not in persisted_config
        and Path(persisted_config.get("resource_lock_path", "")).resolve()
        == GLOBAL_HEAVY_LOCK
        and not GLOBAL_HEAVY_LOCK.is_relative_to(run_dir)
        and Path(persisted_config.get("resource_guard_log_path", "")).is_relative_to(run_dir)
        and Path(persisted_config.get("resource_guard_path", "")).is_file()
        and resource_contract["lock"].get("scope") == "global_sfm_heavy"
        and resource_contract["lock"].get("outside_run_dir") is True
        and resource_contract["startup_preflight"]
        == {
            "mem_available_gib": 24.0,
            "vram_free_gib": 24.0,
            "swap_free_gib": 6.0,
            "disk_free_gib": 100.0,
            "fail_closed": True,
        }
        and forced_path.is_file()
        and sha256(forced_path) == forced_manifest["forced_pairs_sha256"]
        and len(parsed_pairs) == forced_manifest["n_pairs"]
        and injected,
        "persisted config parses the hashed forced file, binds a 2048-row SIFT hard cap (2048 locations x 1 orientation), keeps zero workers/DG on/intrinsics fixed, and injects it"
        if injected else "persisted extra_pairs_path has not been proven through real-loader injection",
        config=str(config_path.resolve()),
        extra_pairs_path=persisted_config.get("extra_pairs_path"),
        forced_pairs_sha256=forced_manifest["forced_pairs_sha256"],
        parsed_pairs=len(parsed_pairs),
        injected_pairs=probe.get("n_forced_injected") if probe else None,
        refine_intrinsics=persisted_config.get("refine_intrinsics"),
        skip_doppelgangers=persisted_config.get("skip_doppelgangers"),
        num_workers=persisted_config.get("num_workers"),
        num_track_per_img=persisted_config.get("num_track_per_img"),
        sift_max_num_features=persisted_config.get("sift_max_num_features"),
        sift_max_num_orientations=persisted_config.get(
            "sift_max_num_orientations"
        ),
        sift_max_rows_per_image=(
            persisted_config.get("sift_max_num_features")
            if persisted_config.get("sift_max_num_orientations") == 1
            else None
        ),
        max_num_tracks=persisted_config.get("max_num_tracks"),
        memory_safe_launcher=persisted_config.get("memory_safe_launcher"),
        rerun_from=persisted_config.get("rerun_from"),
        resource_lock_path=persisted_config.get("resource_lock_path"),
        resource_guard_path=persisted_config.get("resource_guard_path"),
        resource_guard_log_path=persisted_config.get("resource_guard_log_path"),
        resource_contract=resource_contract,
    )
    gate.write(run_dir)
    log(f"S3 PASS -- natural-first policy, {len(pairs)} conditional candidates, loader verified")


if __name__ == "__main__":
    main()
