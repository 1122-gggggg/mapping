#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap
import yaml


WORKSPACE = Path("/media/cihcilab/新增磁碟區/sfm_system/建圖")
BASE_RUN = WORKSPACE / "runs" / "fuhe_bridge_gluemap_pi3_1fps_1920_20260707"
DEFAULT_OUT_ROOT = WORKSPACE / "runs" / "fuhe_bridge_edge_debug_20260708"
DEFAULT_REPO = Path("/media/cihcilab/新增磁碟區/河濱場域/gluemap_build/tools/gluemap")
DEFAULT_GLUEMAP_ENV = Path("/home/cihcilab/micromamba/envs/gluemap")
DEFAULT_RIVER_HELPER = Path("/media/cihcilab/新增磁碟區/河濱場域/gluemap_build/tools/run_highres_optimized_build.py")
DEFAULT_BASE_PIPELINE = WORKSPACE / "pipeline" / "run_football_gluemap_from_motion_manifest.py"
DEFAULT_REPAIR = WORKSPACE / "pipeline" / "repair_fuhe_gluemap_fixed_ba.py"


VARIANT_PRESETS: dict[str, dict[str, Any]] = {
    "temporal_w10_top0": {
        "description": "sequential ±5-ish window only; FAISS retrieval disabled",
        "num_neighbors": 0,
        "num_neighbors_sequential": 10,
        "valid_dg_threshold": 0.80,
    },
    "top1_w10_dg080": {
        "description": "sequential window plus top-1 global retrieval",
        "num_neighbors": 1,
        "num_neighbors_sequential": 10,
        "valid_dg_threshold": 0.80,
    },
    "top3_w10_dg080": {
        "description": "sequential window plus top-3 global retrieval",
        "num_neighbors": 3,
        "num_neighbors_sequential": 10,
        "valid_dg_threshold": 0.80,
    },
    "top5_w10_dg080": {
        "description": "sequential window plus top-5 global retrieval",
        "num_neighbors": 5,
        "num_neighbors_sequential": 10,
        "valid_dg_threshold": 0.80,
    },
    "top10_w10_dg080": {
        "description": "sequential window plus top-10 global retrieval",
        "num_neighbors": 10,
        "num_neighbors_sequential": 10,
        "valid_dg_threshold": 0.80,
    },
    "top5_w10_dg090": {
        "description": "strict DG++ gate on top-5 retrieval",
        "num_neighbors": 5,
        "num_neighbors_sequential": 10,
        "valid_dg_threshold": 0.90,
    },
    "top10_w10_dg090": {
        "description": "strict DG++ gate on top-10 retrieval",
        "num_neighbors": 10,
        "num_neighbors_sequential": 10,
        "valid_dg_threshold": 0.90,
    },
    "core_top1_w10_dg080": {
        "description": "core sequences only: P109/P110/P112/P114 with top-1 retrieval",
        "num_neighbors": 1,
        "num_neighbors_sequential": 10,
        "valid_dg_threshold": 0.80,
        "subfolder_regex": "^(P1090109_002|P1100110_005|P1120112|P1140114)$",
    },
    "weak_tail_top1_w10_dg080": {
        "description": "weak bridge audit: P111/P113/P114 with top-1 retrieval",
        "num_neighbors": 1,
        "num_neighbors_sequential": 10,
        "valid_dg_threshold": 0.80,
        "subfolder_regex": "^(P1110111|P1130113|P1140114)$",
    },
}


def log(message: str) -> None:
    print(f"[fuhe_edge_debug] {message}", flush=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_logged(run_dir: Path, stage: str, cmd: list[str], cwd: Path, env_root: Path) -> None:
    log_path = run_dir / "logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = f"{env_root / 'bin'}:" + env.get("PATH", "")
    env["CONDA_PREFIX"] = str(env_root)
    env["LD_LIBRARY_PATH"] = f"{env_root / 'lib'}:" + env.get("LD_LIBRARY_PATH", "")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    started = time.time()
    log(f"run {stage}: {' '.join(str(x) for x in cmd)}")
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run([str(x) for x in cmd], cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT)
    record = {
        "stage": stage,
        "status": "success" if proc.returncode == 0 else "failed",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(time.time() - started, 3),
        "log": str(log_path),
        "cmd": [str(x) for x in cmd],
    }
    append_stage(run_dir, record)
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-120:])
        raise SystemExit(f"{stage} failed ({proc.returncode}); log tail:\n{tail}")


def append_stage(run_dir: Path, record: dict[str, Any]) -> None:
    path = run_dir / "stage_times.json"
    data = read_json(path) if path.exists() else {"stages": []}
    data.setdefault("stages", []).append(record)
    data["total_seconds"] = round(sum(float(x.get("duration_seconds", 0.0)) for x in data["stages"]), 3)
    data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(path, data)


def hardlink_or_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def prepare_variant_run(base_run: Path, run_dir: Path, force: bool = False) -> None:
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    for filename in (
        "map_intrinsics.json",
        "intrinsics_scaled_debug.json",
        "frame_manifest.json",
        "motion_quality.json",
        "motion_bridge_downsample.json",
        "video_records.json",
    ):
        src = base_run / filename
        if src.exists():
            shutil.copy2(src, run_dir / filename)
    for dirname in ("images", "intrinsics_seed"):
        src = base_run / dirname
        dst = run_dir / dirname
        if not dst.exists() and not dst.is_symlink():
            dst.symlink_to(src, target_is_directory=True)
    base_gluemap = base_run / "gluemap"
    for desc in sorted(base_gluemap.glob("P*/salad_descriptors.pt")):
        hardlink_or_symlink(desc, run_dir / "gluemap" / desc.parent.name / desc.name)


def write_gluemap_config(
    run_dir: Path,
    repo: Path,
    variant: str,
    settings: dict[str, Any],
    batch_size: int,
    retrieval_batch_size: int,
    num_workers: int,
    num_track_per_img: int,
) -> Path:
    config = {
        "chosen_model": "pi3",
        "path_feedforward": str(repo / "checkpoints" / "pi3.safetensors"),
        "path_retrieval": str(repo / "checkpoints" / "dino_salad.ckpt"),
        "path_tracker": str(repo / "checkpoints" / "vggsfm_v2_0_0_track_predictor.bin"),
        "path_dg": str(repo / "checkpoints" / "checkpoint-dg+visym.pth"),
        "images_path": str(run_dir / "images"),
        "write_path": str(run_dir / "gluemap"),
        "temp_path": str(run_dir / "tmp"),
        "chosen_output": "gluemap_aba",
        "num_track_per_img": int(num_track_per_img),
        "max_num_tracks": None,
        "camera_model": "PINHOLE",
        "intrinsics_mode": "SHARED",
        "num_neighbors": int(settings["num_neighbors"]),
        "num_neighbors_sequential": int(settings["num_neighbors_sequential"]),
        "batch_size": int(batch_size),
        "retrieval_batch_size": int(retrieval_batch_size),
        "num_workers": int(num_workers),
        "valid_pose_threshold": 0.05,
        "valid_dg_threshold": float(settings["valid_dg_threshold"]),
        "force_load": True,
        "rerun_from": None,
        "coarse_only": False,
        "use_dummy_tracks": False,
        "skip_doppelgangers": False,
        "use_gt_intrinsics": True,
        "gt_intrinsics_path": str(run_dir / "intrinsics_seed"),
        "is_sequential": True,
        "sample_frequency": 1,
        "is_multi_sequence": True,
        "subfolder_regex": settings.get("subfolder_regex", "^P[0-9]+"),
        "variant_name": variant,
        "variant_description": settings.get("description", ""),
    }
    path = run_dir / "gluemap_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(run_dir / "build_config.json", config)
    return path


def read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not path.exists():
        return pairs
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    return pairs


def image_index(name: str) -> tuple[str, int | None]:
    seq, _, file_name = name.partition("/")
    stem = Path(file_name).stem
    try:
        return seq, int(stem)
    except ValueError:
        return seq, None


def pair_graph_metrics(image_names: list[str], pairs: list[tuple[str, str]]) -> dict[str, Any]:
    nodes = sorted(set(image_names))
    known = set(nodes)
    adj = {n: set() for n in nodes}
    valid_pairs = 0
    cross_sequence = 0
    temporal_deltas: list[int] = []
    retrieval_like = 0
    for a, b in pairs:
        if a not in known or b not in known or a == b:
            continue
        valid_pairs += 1
        adj[a].add(b)
        adj[b].add(a)
        seq_a, idx_a = image_index(a)
        seq_b, idx_b = image_index(b)
        if seq_a != seq_b:
            cross_sequence += 1
            retrieval_like += 1
        elif idx_a is not None and idx_b is not None:
            delta = abs(idx_a - idx_b)
            temporal_deltas.append(delta)
            if delta > 10:
                retrieval_like += 1
        else:
            retrieval_like += 1
    seen = set()
    components = []
    for node in nodes:
        if node in seen:
            continue
        q = deque([node])
        seen.add(node)
        size = 0
        while q:
            cur = q.popleft()
            size += 1
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        components.append(size)
    degrees = [len(adj[n]) for n in nodes]
    delta_counter = Counter(temporal_deltas)
    return {
        "total_frames": len(nodes),
        "pairs": valid_pairs,
        "connected_components": len(components),
        "component_sizes": sorted(components, reverse=True),
        "largest_component_ratio": (max(components) / len(nodes)) if nodes and components else 0.0,
        "isolated_images": sum(1 for d in degrees if d == 0),
        "median_pair_degree": float(np.median(degrees)) if degrees else 0.0,
        "min_pair_degree": min(degrees) if degrees else 0,
        "max_pair_degree": max(degrees) if degrees else 0,
        "cross_sequence_pairs": cross_sequence,
        "retrieval_like_pairs": retrieval_like,
        "temporal_delta_histogram": {str(k): int(v) for k, v in sorted(delta_counter.items())[:30]},
    }


def decode_pair_id(pair_id: int) -> tuple[int, int]:
    # COLMAP pair id convention.
    max_image_id = 2147483647
    image_id2 = pair_id % max_image_id
    image_id1 = (pair_id - image_id2) // max_image_id
    return int(image_id1), int(image_id2)


def database_pair_quality(db_path: Path, out_csv: Path) -> dict[str, Any]:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    id_to_name = {int(i): n for i, n in cur.execute("select image_id, name from images")}
    raw_rows = {
        int(pid): int(rows)
        for pid, rows in cur.execute("select pair_id, rows from matches")
    }
    geom_rows = {
        int(pid): int(rows)
        for pid, rows in cur.execute("select pair_id, rows from two_view_geometries")
    }
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for pair_id, raw in raw_rows.items():
        i1, i2 = decode_pair_id(pair_id)
        name1 = id_to_name.get(i1, f"<missing:{i1}>")
        name2 = id_to_name.get(i2, f"<missing:{i2}>")
        verified = int(geom_rows.get(pair_id, 0))
        seq1, idx1 = image_index(name1)
        seq2, idx2 = image_index(name2)
        temporal_delta = abs(idx1 - idx2) if seq1 == seq2 and idx1 is not None and idx2 is not None else None
        records.append(
            {
                "pair_id": pair_id,
                "image1": name1,
                "image2": name2,
                "seq1": seq1,
                "seq2": seq2,
                "cross_sequence": seq1 != seq2,
                "temporal_delta": temporal_delta,
                "raw_matches": raw,
                "verified_matches": verified,
                "inlier_ratio": verified / raw if raw > 0 else 0.0,
            }
        )
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_id",
                "image1",
                "image2",
                "seq1",
                "seq2",
                "cross_sequence",
                "temporal_delta",
                "raw_matches",
                "verified_matches",
                "inlier_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    con.close()
    ratios = np.asarray([r["inlier_ratio"] for r in records if r["raw_matches"] > 0], dtype=np.float64)
    verified = np.asarray([r["verified_matches"] for r in records], dtype=np.float64)
    raw = np.asarray([r["raw_matches"] for r in records], dtype=np.float64)
    suspect = [
        r for r in records
        if (r["verified_matches"] < 30 or r["inlier_ratio"] < 0.15)
        and (r["cross_sequence"] or (r["temporal_delta"] is None or r["temporal_delta"] > 10))
    ]
    suspect_sorted = sorted(suspect, key=lambda r: (r["inlier_ratio"], -r["raw_matches"]))[:50]
    write_json(out_csv.with_suffix(".suspect.json"), suspect_sorted)
    return {
        "database": str(db_path),
        "pairs_with_raw_matches": len(records),
        "pairs_with_verified_geometry": int(sum(1 for r in records if r["verified_matches"] > 0)),
        "raw_matches_mean": float(raw.mean()) if raw.size else 0.0,
        "verified_matches_mean": float(verified.mean()) if verified.size else 0.0,
        "inlier_ratio_median": float(np.median(ratios)) if ratios.size else 0.0,
        "inlier_ratio_p10": float(np.percentile(ratios, 10)) if ratios.size else 0.0,
        "inlier_ratio_p90": float(np.percentile(ratios, 90)) if ratios.size else 0.0,
        "suspect_nonlocal_pairs": len(suspect),
        "pair_quality_csv": str(out_csv),
        "suspect_pairs_json": str(out_csv.with_suffix(".suspect.json")),
    }


def point_support_metrics(model_dir: Path) -> dict[str, Any]:
    rec = pycolmap.Reconstruction(str(model_dir))
    image_seq = {
        int(image_id): image.name.split("/", 1)[0]
        for image_id, image in rec.images.items()
    }
    support_counts = Counter()
    single_by_seq = Counter()
    track_lengths: list[int] = []
    point_errors: list[float] = []
    for point in rec.points3D.values():
        seqs = {image_seq.get(int(elem.image_id), "<unknown>") for elem in point.track.elements}
        support_counts[len(seqs)] += 1
        if len(seqs) == 1:
            single_by_seq[next(iter(seqs))] += 1
        track_lengths.append(len(point.track.elements))
        point_errors.append(float(point.error))
    total = max(int(rec.num_points3D()), 1)
    return {
        "model_dir": str(model_dir),
        "points3D": int(rec.num_points3D()),
        "observations": int(rec.compute_num_observations()),
        "mean_track_length": float(np.mean(track_lengths)) if track_lengths else 0.0,
        "track_length_p5_med_p95": [float(x) for x in np.percentile(track_lengths, [5, 50, 95])] if track_lengths else [0.0, 0.0, 0.0],
        "point_error_p50_p95_p99": [float(x) for x in np.percentile(point_errors, [50, 95, 99])] if point_errors else [0.0, 0.0, 0.0],
        "sequence_support_counts": {str(k): int(v) for k, v in sorted(support_counts.items())},
        "single_sequence_points": int(support_counts.get(1, 0)),
        "single_sequence_ratio": float(support_counts.get(1, 0) / total),
        "single_sequence_by_seq": dict(sorted((k, int(v)) for k, v in single_by_seq.items())),
    }


def database_stats(db_path: Path) -> dict[str, Any]:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    stats: dict[str, Any] = {"path": str(db_path), "bytes": db_path.stat().st_size}
    for table in ("cameras", "images", "keypoints", "matches", "two_view_geometries"):
        stats[table] = int(cur.execute(f"select count(*) from {table}").fetchone()[0])
    row = cur.execute("select min(rows), avg(rows), max(rows) from two_view_geometries").fetchone()
    stats["two_view_geometry_rows"] = {
        "min": int(row[0] or 0),
        "mean": float(row[1] or 0.0),
        "max": int(row[2] or 0),
    }
    con.close()
    return stats


def find_model_dir(run_dir: Path) -> Path:
    candidates = [
        run_dir / "gluemap" / "gluemap_aba",
        run_dir / "gluemap" / "coarse",
    ]
    for path in candidates:
        if all((path / f).exists() for f in ("cameras.bin", "images.bin", "points3D.bin")):
            return path
    raise FileNotFoundError(f"no GLUEMAP model found under {run_dir / 'gluemap'}")


def write_markdown_report(run_dir: Path, variant: str, report: dict[str, Any]) -> None:
    model = report.get("model_summary", {})
    pair_graph = report.get("pair_graph", {})
    support = report.get("point_support", {})
    pair_quality = report.get("pair_quality", {})
    lines = [
        f"# Fuhe Edge Debug: {variant}",
        "",
        f"- description: `{report['variant'].get('description', '')}`",
        f"- num_neighbors: `{report['variant']['num_neighbors']}`",
        f"- num_neighbors_sequential: `{report['variant']['num_neighbors_sequential']}`",
        f"- valid_dg_threshold: `{report['variant']['valid_dg_threshold']}`",
        "",
        "## Pair Graph",
        "",
        f"- pairs: `{pair_graph.get('pairs')}`",
        f"- connected components: `{pair_graph.get('connected_components')}`",
        f"- largest component ratio: `{pair_graph.get('largest_component_ratio')}`",
        f"- median degree: `{pair_graph.get('median_pair_degree')}`",
        f"- cross-sequence pairs: `{pair_graph.get('cross_sequence_pairs')}`",
        f"- retrieval-like pairs: `{pair_graph.get('retrieval_like_pairs')}`",
        "",
        "## Database Pair Quality",
        "",
        f"- matches rows: `{report.get('database', {}).get('matches')}`",
        f"- two-view geometries: `{report.get('database', {}).get('two_view_geometries')}`",
        f"- mean verified rows: `{report.get('database', {}).get('two_view_geometry_rows', {}).get('mean')}`",
        f"- median inlier ratio: `{pair_quality.get('inlier_ratio_median')}`",
        f"- suspect nonlocal pairs: `{pair_quality.get('suspect_nonlocal_pairs')}`",
        "",
        "## Model",
        "",
        f"- model dir: `{model.get('model_dir')}`",
        f"- registered images: `{model.get('registered_images')}`",
        f"- points3D: `{model.get('points3D')}`",
        f"- mean reprojection error: `{model.get('mean_reprojection_error')}`",
        f"- p95 reprojection error: `{model.get('reprojection_stats', {}).get('p95_px')}`",
        f"- invalid projections: `{model.get('reprojection_stats', {}).get('invalid_projection_count')}`",
        "",
        "## Point Support",
        "",
        f"- single-sequence ratio: `{support.get('single_sequence_ratio')}`",
        f"- mean track length: `{support.get('mean_track_length')}`",
        "",
        "## Outputs",
        "",
        f"- PLY: `{report.get('ply', {}).get('path')}`",
        f"- pair quality CSV: `{pair_quality.get('pair_quality_csv')}`",
        f"- JSON report: `{run_dir / 'edge_debug_report.json'}`",
    ]
    (run_dir / "EDGE_DEBUG_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_variant(args: argparse.Namespace, variant: str, settings: dict[str, Any]) -> dict[str, Any]:
    base_run = args.base_run
    run_dir = args.out_root / variant
    prepare_variant_run(base_run, run_dir, force=args.force_variant)
    config_path = write_gluemap_config(
        run_dir,
        args.repo,
        variant,
        settings,
        args.batch_size,
        args.retrieval_batch_size,
        args.num_workers,
        args.num_track_per_img,
    )
    write_json(run_dir / "variant.json", settings | {"name": variant})

    if not args.skip_gluemap:
        run_logged(
            run_dir,
            "gluemap",
            [args.gluemap_env / "bin" / "gluemap-demo", "--config", config_path],
            args.repo,
            args.gluemap_env,
        )

    manifest = read_json(base_run / "frame_manifest.json")
    subfolder_pattern = re.compile(settings.get("subfolder_regex", "^P[0-9]+"))
    image_names = [
        frame["name"]
        for frame in manifest["frames"]
        if subfolder_pattern.match(frame["name"].split("/", 1)[0])
    ]
    pair_graph = pair_graph_metrics(image_names, read_pairs(run_dir / "gluemap" / "pairs.txt"))
    write_json(run_dir / "qa" / "pair_graph.json", pair_graph)
    db_path = run_dir / "gluemap" / "database_merged.db"
    database = database_stats(db_path) if db_path.exists() else {}
    pair_quality = database_pair_quality(db_path, run_dir / "qa" / "pair_quality.csv") if db_path.exists() else {}

    seed_params = [float(x) for x in read_json(run_dir / "map_intrinsics.json")["params"]]
    model_input = find_model_dir(run_dir)
    model_dir = model_input
    model_summary: dict[str, Any] = {}
    repair_report: dict[str, Any] = {}
    if args.repair:
        repair_out = run_dir / "gluemap" / "gluemap_fixed_intrinsics_ba_repaired_keep_density"
        run_logged(
            run_dir,
            "fixed_intrinsics_ba_repair",
            [
                args.gluemap_env / "bin" / "python",
                args.repair_script,
                "--run-dir",
                run_dir,
                "--input-model",
                model_input,
                "--output-model",
                repair_out,
                "--intrinsics",
                run_dir / "map_intrinsics.json",
                "--selected-images",
                str(len(image_names)),
                "--filter-thresholds",
                args.filter_thresholds,
                "--min-tri-angle",
                str(args.min_tri_angle),
                "--min-track-length",
                str(args.min_track_length),
                "--max-mean-reprojection-px",
                str(args.max_mean_reprojection_px),
                "--max-p95-reprojection-px",
                str(args.max_p95_reprojection_px),
            ],
            WORKSPACE,
            args.gluemap_env,
        )
        model_dir = repair_out
        repair_report = read_json(run_dir / "build_report_fixed_ba_repair.json")
        model_summary = repair_report["summary"]
    else:
        base = import_module(args.base_pipeline, "football_gluemap_base")
        model_summary = base.strict_model_summary(model_dir, seed_params)

    river = import_module(args.river_helper, "river_helper")
    ply_path = run_dir / "deploy" / f"map_rgb_{variant}.ply"
    ply_stats = river.export_rgb_ply(model_dir, run_dir / "images", ply_path, 8.0)
    point_support = point_support_metrics(model_dir)
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_run": str(base_run),
        "run_dir": str(run_dir),
        "variant": settings | {"name": variant},
        "config": str(config_path),
        "pair_graph": pair_graph,
        "database": database,
        "pair_quality": pair_quality,
        "repair_report": repair_report,
        "model_summary": model_summary,
        "point_support": point_support,
        "ply": {"path": str(ply_path), **ply_stats},
        "stage_times": read_json(run_dir / "stage_times.json") if (run_dir / "stage_times.json").exists() else {},
    }
    write_json(run_dir / "edge_debug_report.json", report)
    write_markdown_report(run_dir, variant, report)
    log(
        f"{variant}: pairs={pair_graph['pairs']} cross={pair_graph['cross_sequence_pairs']} "
        f"registered={model_summary.get('registered_images')} points={model_summary.get('points3D')} "
        f"mean={model_summary.get('mean_reprojection_error')} single_seq={point_support.get('single_sequence_ratio'):.3f}"
    )
    return report


def parse_variants(raw: str) -> list[str]:
    names = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in names if x not in VARIANT_PRESETS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown variants: {unknown}; choices={sorted(VARIANT_PRESETS)}")
    return names


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Fuhe GLUEMAP temporal/top-k edge debug variants.")
    ap.add_argument("--base-run", type=Path, default=BASE_RUN)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--gluemap-env", type=Path, default=DEFAULT_GLUEMAP_ENV)
    ap.add_argument("--river-helper", type=Path, default=DEFAULT_RIVER_HELPER)
    ap.add_argument("--base-pipeline", type=Path, default=DEFAULT_BASE_PIPELINE)
    ap.add_argument("--repair-script", type=Path, default=DEFAULT_REPAIR)
    ap.add_argument("--variants", type=parse_variants, default=["temporal_w10_top0"])
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--retrieval-batch-size", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--num-track-per-img", type=int, default=2048)
    ap.add_argument("--filter-thresholds", default="8")
    ap.add_argument("--min-tri-angle", type=float, default=0.0)
    ap.add_argument("--min-track-length", type=int, default=2)
    ap.add_argument("--max-mean-reprojection-px", type=float, default=3.5)
    ap.add_argument("--max-p95-reprojection-px", type=float, default=8.0)
    ap.add_argument("--skip-gluemap", action="store_true")
    ap.add_argument("--no-repair", dest="repair", action="store_false", default=True)
    ap.add_argument("--force-variant", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.base_run.exists():
        raise FileNotFoundError(args.base_run)
    args.out_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for variant in args.variants:
        settings = dict(VARIANT_PRESETS[variant])
        reports.append(run_variant(args, variant, settings))
    comparison = []
    for report in reports:
        model = report["model_summary"]
        pair_graph = report["pair_graph"]
        point_support = report["point_support"]
        comparison.append(
            {
                "variant": report["variant"]["name"],
                "num_neighbors": report["variant"]["num_neighbors"],
                "valid_dg_threshold": report["variant"]["valid_dg_threshold"],
                "pairs": pair_graph["pairs"],
                "cross_sequence_pairs": pair_graph["cross_sequence_pairs"],
                "retrieval_like_pairs": pair_graph["retrieval_like_pairs"],
                "registered_images": model.get("registered_images"),
                "points3D": model.get("points3D"),
                "mean_reprojection_error": model.get("mean_reprojection_error"),
                "p95_reprojection_error": model.get("reprojection_stats", {}).get("p95_px"),
                "invalid_projection_count": model.get("reprojection_stats", {}).get("invalid_projection_count"),
                "single_sequence_ratio": point_support.get("single_sequence_ratio"),
                "ply": report["ply"]["path"],
                "report": str(Path(report["run_dir"]) / "edge_debug_report.json"),
            }
        )
    write_json(args.out_root / "edge_debug_comparison.json", comparison)
    log(f"comparison: {args.out_root / 'edge_debug_comparison.json'}")


if __name__ == "__main__":
    main()
