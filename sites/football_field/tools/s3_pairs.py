#!/usr/bin/env python3
"""S3 -- forced VPR-blind bridge candidates and real-loader contract gate.

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
    RUNS,
    SUBFOLDER_REGEX,
    Gate,
    log,
    read_json,
    required_check_ids,
    sha256,
    stage_material_artifacts,
    write_json,
)
from ts_env import verify_pycolmap_runtime  # noqa: E402


TS_COMMON = Path(__file__).with_name("ts_common.py")
TS_ENV = Path(__file__).with_name("ts_env.py")
MEMORY_SAFE_LAUNCHER = Path(__file__).with_name("run_gluemap_memory_safe.py")
GLUEMAP_LOADER = GLUEMAP_REPO / "gluemap" / "datasets" / "multi_sequence_twoview.py"
GLUEMAP_COLMAP = GLUEMAP_REPO / "gluemap" / "utils" / "colmap.py"

FWD_STRIDE = 16
REV_STRIDE = 12
ENDPOINT_FRAC = 0.08
ENDPOINT_STRIDE = 3
EXPECTED_FORCED_PAIRS = 6000
EXPECTED_SEQUENCE_PAIRS = 12
EXPECTED_IMAGES = 1414
EXPECTED_CAMERAS = 3
MIN_PAIR_DENSITY = 4.0


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
    forbidden = sorted(
        {
            str(frame.get("source_rel"))
            for frame in frame_manifest.get("frames", [])
            if any(
                token in str(frame.get("source_rel", ""))
                for token in ("P0710071", "P1230123", "P1260126")
            )
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


def _emit_probe_checks(gate: Gate, payload: dict | None, error: str) -> None:
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
    gate.check(
        "G3.3",
        payload["pair_density"] >= MIN_PAIR_DENSITY,
        f"pre-matching pair density {payload['pair_density']:.3f} pairs/frame",
        n_pairs=payload["n_loader_pairs"],
        n_images=payload["n_images"],
        pair_density=payload["pair_density"],
        minimum=MIN_PAIR_DENSITY,
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
        and payload["n_forced_expected"] == EXPECTED_FORCED_PAIRS
        and payload["n_forced_injected"] == EXPECTED_FORCED_PAIRS
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
        source_files=[TS_COMMON, TS_ENV, GLUEMAP_LOADER, GLUEMAP_COLMAP],
    )
    gate.record_predecessor_gate(
        "S2b_intrinsics",
        run_dir / "gates" / "S2b_intrinsics.json",
        expected_stage="S2b_intrinsics",
    )
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="football_field_v1")
    parser.add_argument("--candidate", default="official69")
    parser.add_argument("--max-pairs", type=int, default=12000)
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
    pairs: set[tuple[str, str]] = set()
    per_pair: dict[str, dict[str, int]] = {}
    for seq_a in fwd:
        sampled_a = sample(by_seq[seq_a], FWD_STRIDE, ENDPOINT_STRIDE, ENDPOINT_FRAC)
        for seq_b in rev:
            sampled_b = sample(by_seq[seq_b], REV_STRIDE, ENDPOINT_STRIDE, ENDPOINT_FRAC)
            before = len(pairs)
            for frame_a in sampled_a:
                for frame_b in sampled_b:
                    pairs.add(tuple(sorted((frame_a["name"], frame_b["name"]))))
            per_pair[f"{seq_a}|{seq_b}"] = {
                "n_fwd_sampled": len(sampled_a),
                "n_rev_sampled": len(sampled_b),
                "n_pairs": len(pairs) - before,
            }
    if len(pairs) > args.max_pairs:
        raise SystemExit(
            f"{len(pairs)} forced pairs exceeds --max-pairs {args.max_pairs}"
        )
    forced_path = run_dir / "forced_bridges.txt"
    forced_path.write_text(
        "# VPR-blind forced forward<->reverse bridge CANDIDATES.\n"
        "# Geometry and Doppelgangers still decide which candidates survive.\n"
        + "".join(f"{a} {b}\n" for a, b in sorted(pairs)),
        encoding="utf-8",
    )
    parsed_pairs = parse_pair_file(forced_path)

    config = {
        "chosen_model": "pi3",
        "path_feedforward": str(GLUEMAP_REPO / "checkpoints" / "pi3.safetensors"),
        "path_retrieval": str(GLUEMAP_REPO / "checkpoints" / "dino_salad.ckpt"),
        "path_tracker": str(GLUEMAP_REPO / "checkpoints" / "vggsfm_v2_0_0_track_predictor.bin"),
        "path_dg": str(GLUEMAP_REPO / "checkpoints" / "checkpoint-dg+visym.pth"),
        "images_path": str((run_dir / "images").resolve()),
        "write_path": str((run_dir / "gluemap").resolve()),
        "temp_path": str((run_dir / "tmp").resolve()),
        "chosen_output": "gluemap_aba",
        "camera_model": "PINHOLE",
        "intrinsics_mode": "SHARED",
        "use_gt_intrinsics": True,
        "gt_intrinsics_path": str((run_dir / "intrinsics_seed").resolve()),
        "refine_intrinsics": False,
        "skip_doppelgangers": False,
        "valid_dg_threshold": 0.8,
        "is_multi_sequence": True,
        "subfolder_regex": SUBFOLDER_REGEX,
        "is_sequential": True,
        "sample_frequency": 1,
        "num_neighbors": 100,
        "num_neighbors_sequential": 30,
        "extra_pairs_path": str(forced_path.resolve()),
        "num_track_per_img": 512,
        "sift_max_num_features": 2048,
        "max_num_tracks": 400000,
        "memory_safe_launcher": str(MEMORY_SAFE_LAUNCHER.resolve()),
        "min_track_length": 3,
        "valid_pose_threshold": 0.05,
        "batch_size": 8,
        "retrieval_batch_size": 30,
        "num_workers": 0,
        "force_load": True,
        "rerun_from": "star",
        "coarse_only": False,
        "use_dummy_tracks": False,
    }
    config_path = run_dir / "gluemap_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    lineage_ok, lineage_metrics = source_lineage_evidence(
        corpus_manifest, frame_manifest
    )
    forced_manifest_path = run_dir / "forced_bridges.json"
    forced_manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "fwd": fwd,
        "rev": rev,
        "strides": {
            "fwd": FWD_STRIDE,
            "rev": REV_STRIDE,
            "endpoint": ENDPOINT_STRIDE,
            "endpoint_frac": ENDPOINT_FRAC,
        },
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
        "exact seven locked source rel/hash pairs; no P071/P123/P126 lineage"
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
    _emit_probe_checks(gate, probe, probe_detail)

    expected_sequences = {video.seq for video in BUILD}
    gate.check(
        "G3.5a",
        set(fwd) | set(rev) == expected_sequences
        and set(fwd).isdisjoint(rev)
        and len(fwd) == 4
        and len(rev) == 3,
        f"all seven directions resolved exactly once: {len(fwd)} fwd/{len(rev)} rev",
        fwd=fwd,
        rev=rev,
        expected=sorted(expected_sequences),
    )
    gate.check(
        "G3.5b",
        len(pairs) == EXPECTED_FORCED_PAIRS
        and len(parsed_pairs) == EXPECTED_FORCED_PAIRS
        and len(per_pair) == EXPECTED_SEQUENCE_PAIRS
        and len(pairs) <= args.max_pairs,
        f"{len(pairs)} unique forced pairs across {len(per_pair)} direction combinations",
        n_pairs=len(pairs),
        n_parsed=len(parsed_pairs),
        n_sequence_pairs=len(per_pair),
        max_pairs=args.max_pairs,
        per_video_pair=per_pair,
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
        and persisted_config.get("max_num_tracks") == 400000
        and Path(persisted_config.get("memory_safe_launcher", "")).resolve()
        == MEMORY_SAFE_LAUNCHER.resolve()
        and MEMORY_SAFE_LAUNCHER.is_file()
        and Path(persisted_config.get("extra_pairs_path", "")).resolve()
        == forced_path.resolve()
        and forced_path.is_file()
        and sha256(forced_path) == forced_manifest["forced_pairs_sha256"]
        and len(parsed_pairs) == forced_manifest["n_pairs"]
        and injected,
        "persisted config parses the hashed forced file, binds memory-safe feature budgets, keeps zero workers/DG on/intrinsics fixed, and injects it"
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
        max_num_tracks=persisted_config.get("max_num_tracks"),
        memory_safe_launcher=persisted_config.get("memory_safe_launcher"),
    )
    gate.write(run_dir)
    log(f"S3 PASS -- {len(pairs)} forced candidates and real-loader contract verified")


if __name__ == "__main__":
    main()
